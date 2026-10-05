# Combined EMNIST (ByMerge) + BanglaLekha-Isolated CNN — training & ONNX export.
#
# Runs on Kaggle (GPU) with datasets attached:
#   crawford/emnist                       (CSV)
#   mdnaorezahabib/banglalekha-isolated   (Images/1..84/*.png)
#
# MODE="sweep": short runs over several widths, prints a results table.
# MODE="final": full run with WIDTH, exports everything the web app needs:
#   out/models/combined-cnn/model.onnx         multi-output (11 layers)
#   out/models/combined-cnn/conv1-weights.json
#   out/models/checkpoints/epoch-XX/model.onnx
#   out/training/history.json, weight-snapshots.json
#
# Output index space is unchanged (146): 0-61 EMNIST ByClass order, 62-145 Bengali.
# Training uses ByMerge, so the 15 merged lowercase indices stay untrained — the
# web app already masks exactly those (lib/model/classes.ts BYMERGE_MERGED_INDICES).

import copy
import glob
import json
import math
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter

MODE = "final"                      # "sweep" | "final"
WIDTH = (32, 64, 128, 256)          # conv1, conv2, conv3, dense1 (final mode)
SWEEP = [(16, 32, 64, 128), (32, 64, 128, 128), (32, 64, 128, 256), (48, 96, 192, 256)]
EPOCHS = 40 if MODE == "final" else 10
EPOCH_SAMPLES = 400_000             # samples drawn per epoch (60% EMNIST / 40% Bengali)
BATCH = 512
PEAK_LR = 2e-3
WEIGHT_DECAY = 5e-2                 # AdamW (decoupled)
LABEL_SMOOTHING = 0.05
SNAPSHOT_EPOCHS = {0, 1, 2, 3, 5, 10, 15, 20, 25, 30, 35, EPOCHS - 1}
NUM_CLASSES = 146
OUT = "/kaggle/working/out" if os.path.isdir("/kaggle") else "./out"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

EMNIST62 = [str(d) for d in range(10)] + [chr(c) for c in range(65, 91)] + [chr(c) for c in range(97, 123)]


def find(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    assert hits, f"not found: {pattern}"
    return hits[0]


# ---------------------------------------------------------------- data

def load_emnist(split):
    """ByMerge CSV → (uint8 N×28×28 in raw/transposed EMNIST orientation, labels in 62-index space)."""
    path = find(f"/kaggle/input/**/emnist-bymerge-{split}.csv")
    mapping = find("/kaggle/input/**/emnist-bymerge-mapping.txt")
    to62 = {}
    for line in open(mapping):
        k, ascii_code = line.split()
        to62[int(k)] = EMNIST62.index(chr(int(ascii_code)))
    arr = _read_csv(path)
    labels = np.vectorize(to62.get)(arr[:, 0]).astype(np.int64)
    return arr[:, 1:].reshape(-1, 28, 28), labels


def _read_csv(path):
    import pandas as pd
    return pd.read_csv(path, header=None, dtype=np.uint8, engine="c").values


def normalize_glyph(img, thicken=False):
    """EMNIST-style: crop to ink bbox, fit long side to 24px (aspect kept), center in 28×28.
    Must stay in sync with lib/model/preprocess.ts. thicken: dilate thin pen scans
    (BanglaLekha) to roughly EMNIST / canvas stroke width (~2px at 28x28)."""
    a = np.asarray(img, dtype=np.float32)
    if a[[0, -1], :].mean() + a[:, [0, -1]].mean() > 255:  # dark ink on light paper → invert
        a = 255 - a
    ys, xs = np.nonzero(a > 0.2 * a.max())
    if len(ys) == 0:
        return np.zeros((28, 28), np.uint8)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = a.shape
    s = 24 / max(h, w)
    if thicken:
        k = max(3, int(round(1 / s)) | 1)
        a = np.asarray(Image.fromarray(a.astype(np.uint8)).filter(ImageFilter.MaxFilter(k)), np.float32)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    small = np.asarray(Image.fromarray(a.astype(np.uint8)).resize((nw, nh), Image.BOX), np.float32)
    small = small * (255 / max(small.max(), 1))
    out = np.zeros((28, 28), np.float32)
    y0, x0 = (28 - nh) // 2, (28 - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = small
    return out.astype(np.uint8)


def _load_bangla_one(path):
    return normalize_glyph(Image.open(path).convert("L"), thicken=True).T  # .T → EMNIST raw orientation


def load_bangla():
    root = os.path.dirname(find("/kaggle/input/**/BanglaLekha-Isolated/Images/1"))
    paths, labels = [], []
    for folder in range(1, 85):
        files = sorted(glob.glob(os.path.join(root, str(folder), "*.png")))
        paths += files
        labels += [62 + folder - 1] * len(files)
    with Pool(os.cpu_count()) as p:
        imgs = p.map(_load_bangla_one, paths, chunksize=512)
    x, y = np.stack(imgs), np.array(labels, np.int64)
    rng = np.random.RandomState(42)                     # stratified 80/20 split
    tr, te = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        cut = int(len(idx) * 0.8)
        tr += list(idx[:cut]); te += list(idx[cut:])
    return x[tr], y[tr], x[te], y[te]


def to_dev(x, y):
    return torch.from_numpy(x).to(DEV), torch.from_numpy(y).to(DEV)


# ---------------------------------------------------------------- augmentation (GPU, per-sample)

def augment(x, k=1.0):
    """x: float N×1×28×28 in [0,1]. Random affine + stroke-width jitter, scaled by strength k."""
    n = x.shape[0]
    rot = torch.empty(n, device=DEV).uniform_(-12 * k, 12 * k) * math.pi / 180
    shear = torch.empty(n, device=DEV).uniform_(-0.15 * k, 0.15 * k)
    scale = torch.empty(n, device=DEV).uniform_(1 - 0.15 * k, 1 + 0.12 * k)
    sx = scale * torch.empty(n, device=DEV).uniform_(1 - 0.1 * k, 1 + 0.1 * k)   # mild aspect jitter
    tx, ty = (torch.empty(2, n, device=DEV).uniform_(-0.12 * k, 0.12 * k))
    cos, sin = torch.cos(rot), torch.sin(rot)
    theta = torch.stack([
        torch.stack([cos / sx, (-sin + shear) / sx, tx], 1),
        torch.stack([sin / scale, cos / scale, ty], 1),
    ], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
    r = torch.rand(n, 1, 1, 1, device=DEV)
    thick = F.max_pool2d(x, 3, 1, 1)
    thin = -F.max_pool2d(-x, 3, 1, 1)
    x = torch.where(r < 0.3 * k, thick, torch.where(r > 1 - 0.1 * k, thin, x))
    # style randomization, script-independent: EMNIST is soft/blurry, BanglaLekha scans and
    # canvas strokes are crisp — without this the model learns "crisp ⇒ Bengali".
    s = torch.rand(n, 1, 1, 1, device=DEV)
    blur = F.avg_pool2d(F.pad(x, (1, 1, 1, 1)), 3, 1) * 0.6 + x * 0.4
    blur = blur / blur.amax((2, 3), keepdim=True).clamp_min(1e-6)
    crisp = (x > torch.empty(n, 1, 1, 1, device=DEV).uniform_(0.2, 0.5)).float()
    return torch.where(s < 0.35, blur, torch.where(s > 0.65, crisp, x))


# ---------------------------------------------------------------- model

class CombinedNet(nn.Module):
    """Same layer names/types as the web visualization expects; widths configurable."""

    def __init__(self, c1, c2, c3, d):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(1, c1, 3, padding=1), nn.BatchNorm2d(c1)
        self.conv2, self.bn2 = nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2)
        self.conv3, self.bn3 = nn.Conv2d(c2, c3, 3, padding=1), nn.BatchNorm2d(c3)
        self.dense1 = nn.Linear(c3 * 7 * 7, d)
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(d, NUM_CLASSES)

    def forward(self, x, all_outputs=False):
        conv1 = self.conv1(x); relu1 = F.relu(self.bn1(conv1))
        conv2 = self.conv2(relu1); relu2 = F.relu(self.bn2(conv2)); pool1 = F.max_pool2d(relu2, 2)
        conv3 = self.conv3(pool1); relu3 = F.relu(self.bn3(conv3)); pool2 = F.max_pool2d(relu3, 2)
        dense1 = self.dense1(pool2.flatten(1)); relu4 = F.relu(dense1)
        output = self.output(self.dropout(relu4))
        if all_outputs:
            return conv1, relu1, conv2, relu2, pool1, conv3, relu3, pool2, dense1, relu4, output
        return output


class MultiOut(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x, all_outputs=True)


def fold_bn(model):
    """CPU copy with each BatchNorm folded into its conv, so exported conv outputs are the
    exact pre-activation values and relu = max(0, conv) holds in the visualization."""
    m = copy.deepcopy(model).cpu().eval()
    for c, b in [("conv1", "bn1"), ("conv2", "bn2"), ("conv3", "bn3")]:
        conv, bn = getattr(m, c), getattr(m, b)
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        with torch.no_grad():
            conv.weight.mul_(scale.view(-1, 1, 1, 1))
            conv.bias.copy_((conv.bias - bn.running_mean) * scale + bn.bias)
        setattr(m, b, nn.Identity())
    return m


ONNX_NAMES = ["conv1", "relu1", "conv2", "relu2", "pool1", "conv3", "relu3", "pool2", "dense1", "relu4", "output"]


def export_onnx(model, path):
    model = fold_bn(model)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    kw = dict(input_names=["input"], output_names=ONNX_NAMES, dynamic_axes={n: {0: "batch"} for n in ["input"] + ONNX_NAMES}, opset_version=17)
    try:
        torch.onnx.export(MultiOut(model).cpu(), torch.zeros(1, 1, 28, 28), path, dynamo=False, **kw)
    except TypeError:  # older torch without the dynamo kwarg
        torch.onnx.export(MultiOut(model).cpu(), torch.zeros(1, 1, 28, 28), path, **kw)


# ---------------------------------------------------------------- train / eval

@torch.no_grad()
def evaluate(model, x, y):
    """Clean (no aug, eval mode, no smoothing) CE loss + accuracy."""
    model.eval()
    loss = correct = 0.0
    for i in range(0, len(x), 4096):
        xb = x[i:i + 4096].unsqueeze(1).float() / 255
        with torch.autocast(DEV.type, enabled=DEV.type == "cuda"):
            out = model(xb).float()
        loss += F.cross_entropy(out, y[i:i + 4096], reduction="sum").item()
        correct += (out.argmax(1) == y[i:i + 4096]).sum().item()
    return loss / len(x), correct / len(x)


@torch.no_grad()
def crisp_eval(model, ex, ey, bx, by):
    """Accuracy on binarized (canvas-like) test glyphs + rate of predicting the wrong script."""
    model.eval()
    out = []
    wrong = total = 0
    for x, y, latin in [(ex, ey, True), (bx, by, False)]:
        correct = 0
        for i in range(0, len(x), 4096):
            xb = (x[i:i + 4096].unsqueeze(1).float() / 255 > 0.3).float()
            with torch.autocast(DEV.type, enabled=DEV.type == "cuda"):
                pred = model(xb).float().argmax(1)
            correct += (pred == y[i:i + 4096]).sum().item()
            wrong += ((pred >= 62) if latin else (pred < 62)).sum().item()
        out.append(correct / len(x))
        total += len(x)
    return out[0], out[1], wrong / total


def train(width, epochs, data, export=False):
    (ex, ey, bx, by, ex_te, ey_te, bx_te, by_te) = data
    model = CombinedNet(*width).to(DEV).to(memory_format=torch.channels_last)
    params = sum(p.numel() for p in model.parameters())
    steps_per_epoch = EPOCH_SAMPLES // BATCH
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, PEAK_LR, total_steps=epochs * steps_per_epoch,
                                                pct_start=0.15, div_factor=20, final_div_factor=200)
    scaler = torch.amp.GradScaler(enabled=DEV.type == "cuda")
    # EMA of weights + BN buffers, warmup decay so early checkpoints aren't dominated by init.
    # Evaluated/exported instead of the raw model → smooth, stable curves.
    ema = torch.optim.swa_utils.AveragedModel(
        model, use_buffers=True,
        avg_fn=lambda e, p, n: e + (p - e) * (1 - min(0.999, (1 + n) / (10 + n))))
    # fixed clean subset of train data for train curves comparable to val
    g = torch.Generator(device="cpu").manual_seed(1)
    tr_e = torch.randperm(len(ex), generator=g)[:30_000].to(DEV)
    tr_b = torch.randperm(len(bx), generator=g)[:20_000].to(DEV)
    tr_x, tr_y = torch.cat([ex[tr_e], bx[tr_b]]), torch.cat([ey[tr_e], by[tr_b]])
    val_x, val_y = torch.cat([ex_te, bx_te]), torch.cat([ey_te, by_te])
    hist = {k: [] for k in ["loss", "accuracy", "val_loss", "val_accuracy", "emnist_val_accuracy",
                            "bangla_val_accuracy", "emnist_crisp_accuracy", "bangla_crisp_accuracy",
                            "cross_script_rate", "train_batch_loss", "lr"]}
    snaps = {}
    n_e = int(EPOCH_SAMPLES * 0.6)
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        idx_e = torch.randint(len(ex), (n_e,), device=DEV)
        idx_b = torch.randint(len(bx), (EPOCH_SAMPLES - n_e,), device=DEV)
        xs, ys = torch.cat([ex[idx_e], bx[idx_b]]), torch.cat([ey[idx_e], by[idx_b]])
        perm = torch.randperm(len(xs), device=DEV)
        xs, ys = xs[perm], ys[perm]
        run = 0.0
        # anneal augmentation over the last 30% so the model settles on clean-looking data
        k = 1.0 - 0.75 * max(0.0, (epoch + 1 - 0.7 * epochs) / (0.3 * epochs))
        for s in range(steps_per_epoch):
            xb = augment(xs[s * BATCH:(s + 1) * BATCH].unsqueeze(1).float() / 255, k)
            yb = ys[s * BATCH:(s + 1) * BATCH]
            with torch.autocast(DEV.type, enabled=DEV.type == "cuda"):
                loss = F.cross_entropy(model(xb.contiguous(memory_format=torch.channels_last)), yb,
                                       label_smoothing=LABEL_SMOOTHING)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update_parameters(model)
            run += loss.item() if s % 20 == 0 else 0
        m = ema.module
        tl, ta = evaluate(m, tr_x, tr_y)
        vl, va = evaluate(m, val_x, val_y)
        _, ea = evaluate(m, ex_te, ey_te)
        _, ba = evaluate(m, bx_te, by_te)
        ec, bc, cross = crisp_eval(m, ex_te, ey_te, bx_te, by_te)
        for k, v in zip(hist, [tl, ta, vl, va, ea, ba, ec, bc, cross, run / math.ceil(steps_per_epoch / 20), sched.get_last_lr()[0]]):
            hist[k].append(v if k == "lr" else round(v, 5))
        print(f"[{width}] ep {epoch:2d} train {tl:.4f}/{ta:.4f} val {vl:.4f}/{va:.4f} "
              f"en {ea:.4f} bn {ba:.4f} crisp en {ec:.4f} bn {bc:.4f} xscript {cross:.4f} lr {hist['lr'][-1]:.1e} {time.time() - t0:.0f}s", flush=True)
        if export:
            export_onnx(m, f"{OUT}/models/checkpoints/epoch-{epoch:02d}/model.onnx")
            if epoch in SNAPSHOT_EPOCHS:
                snaps[str(epoch)] = snapshot(m)
    return ema.module, hist, snaps, params


def snapshot(model):
    model = fold_bn(model)
    s = {"conv1": model.conv1.weight.detach().cpu().numpy().tolist()}
    for name in ["conv2", "conv3", "dense1"]:
        w = getattr(model, name).weight.detach().cpu().numpy()
        s[name] = {"mean": float(w.mean()), "std": float(w.std()), "min": float(w.min()),
                   "max": float(w.max()), "shape": list(w.shape)}
    return s


def verify_onnx(model, path, x):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime"], check=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(path)
    m = fold_bn(model)  # fp32 CPU reference (no TF32), same folded graph as the export
    with torch.no_grad():
        raw = copy.deepcopy(model).cpu().eval()(x[:64].unsqueeze(1).float().cpu() / 255)
        print(f"  fold_bn logits max|diff| {(raw - m(x[:64].unsqueeze(1).float().cpu() / 255)).abs().max():.2e}")
    xb = x[:64].unsqueeze(1).float().cpu() / 255
    outs = sess.run(None, {"input": xb.numpy()})
    with torch.no_grad():
        ref = m(xb, all_outputs=True)
    for name, o, r in zip(ONNX_NAMES, outs, ref):
        ok = np.allclose(o, r.numpy(), rtol=1e-3, atol=1e-3)
        print(f"  onnx {name:7s} {tuple(o.shape)} max|diff| {np.abs(o - r.numpy()).max():.2e} {'ok' if ok else 'MISMATCH'}")
    for c, r_ in [(0, 1), (2, 3), (5, 6), (8, 9)]:  # relu == max(0, conv) after BN folding
        print(f"  relu==max(0,{ONNX_NAMES[c]}): {np.allclose(np.maximum(outs[c], 0), outs[r_], atol=1e-5)}")


# ---------------------------------------------------------------- main

if __name__ == "__main__":
    t = time.time()
    ex, ey = load_emnist("train")
    ex_te, ey_te = load_emnist("test")
    bx, by, bx_te, by_te = load_bangla()
    print(f"EMNIST {len(ex)}/{len(ex_te)}  Bengali {len(bx)}/{len(bx_te)}  load {time.time() - t:.0f}s", flush=True)
    os.makedirs(OUT, exist_ok=True)
    # preview grid for sanity (display orientation = transpose back)
    prev = np.concatenate([np.concatenate([x.T for x in arr[i * 16:(i + 1) * 16]], 1)
                           for arr in (ex[:64], bx[::max(1, len(bx) // 64)][:64]) for i in range(4)], 0)
    Image.fromarray(prev).save(f"{OUT}/samples.png")
    data = [to_dev(a, b) for a, b in [(ex, ey), (bx, by), (ex_te, ey_te), (bx_te, by_te)]]
    data = [t_ for pair in data for t_ in pair]  # ex, ey, bx, by, ex_te, ey_te, bx_te, by_te

    if MODE == "sweep":
        results = []
        for w in SWEEP:
            model, hist, _, params = train(w, EPOCHS, data)
            results.append({"width": w, "params": params, "onnx_mb": round(params * 4 / 1e6, 2),
                            **{k: hist[k][-1] for k in ["val_accuracy", "emnist_val_accuracy", "bangla_val_accuracy",
                                                        "loss", "val_loss"]}, "history": hist})
            print(json.dumps({k: v for k, v in results[-1].items() if k != "history"}), flush=True)
        json.dump(results, open(f"{OUT}/sweep.json", "w"), indent=1)
    else:
        model, hist, snaps, params = train(WIDTH, EPOCHS, data, export=True)
        final = f"{OUT}/models/combined-cnn/model.onnx"
        export_onnx(model, final)
        os.makedirs(f"{OUT}/training", exist_ok=True)
        json.dump(hist, open(f"{OUT}/training/history.json", "w"), indent=1)
        json.dump(snaps, open(f"{OUT}/training/weight-snapshots.json", "w"))
        fm = fold_bn(model)
        w = fm.conv1.weight.detach().cpu().numpy()
        json.dump({"weights": w.ravel().tolist(), "biases": fm.conv1.bias.detach().cpu().numpy().tolist(),
                   "shape": list(w.shape)}, open(f"{OUT}/models/combined-cnn/conv1-weights.json", "w"))
        torch.save(model.state_dict(), f"{OUT}/model.pt")
        try:
            verify_onnx(model, final, data[4])
        except Exception as e:  # never lose a finished run over verification
            print(f"ONNX verify failed: {e!r}")
        print(f"params {params:,}  onnx {os.path.getsize(final) / 1e6:.2f} MB  "
              f"val {hist['val_accuracy'][-1]:.4f} en {hist['emnist_val_accuracy'][-1]:.4f} "
              f"bn {hist['bangla_val_accuracy'][-1]:.4f}")
    print(f"total {time.time() - t:.0f}s")
