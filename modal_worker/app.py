"""
Modal GPU Training Worker for OCR Model Lab

Accepts CNN architecture config, trains on soyeb-jim285/ocr-handwriting-data,
streams epoch metrics via SSE. Trained ONNX models are cached on HuggingFace;
identical configs skip training and replay stored metrics.

Deploy: modal deploy modal_worker/app.py
"""

import modal

DATA_DIR = "/root/data"
HF_DATASET = "soyeb-jim285/ocr-handwriting-data"
HF_REPO = "soyeb-jim285/ocr-visualization-models"
NUM_CLASSES_MAP = {"digits": 10, "emnist": 62, "bengali": 84, "combined": 146}


# ---------------------------------------------------------------------------
# Build-time: pre-process all dataset types into .npy files
# ---------------------------------------------------------------------------

def preprocess_datasets():
    """Runs during image build. Loads HF dataset, filters each type, saves as .npy."""
    import os
    import numpy as np
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET)

    for dtype in ["digits", "emnist", "bengali", "combined"]:
        out_dir = f"{DATA_DIR}/{dtype}"
        os.makedirs(out_dir, exist_ok=True)

        for split_name in ["train", "test"]:
            split_ds = ds[split_name]

            if dtype == "digits":
                split_ds = split_ds.filter(lambda x: x["label"] < 10)
            elif dtype == "emnist":
                split_ds = split_ds.filter(lambda x: x["label"] < 62)
            elif dtype == "bengali":
                split_ds = split_ds.filter(lambda x: x["label"] >= 62)

            images, labels = [], []
            for row in split_ds:
                arr = np.array(row["image"].convert("L"), dtype=np.float32) / 255.0
                images.append(arr)
                label = row["label"]
                if dtype == "bengali":
                    label -= 62
                labels.append(label)

            imgs = np.array(images)[:, np.newaxis, :, :]
            lbls = np.array(labels, dtype=np.int64)

            np.save(f"{out_dir}/{split_name}_images.npy", imgs)
            np.save(f"{out_dir}/{split_name}_labels.npy", lbls)
            print(f"  {dtype}/{split_name}: {len(imgs)} samples saved")

    print("All datasets preprocessed.")


# ---------------------------------------------------------------------------
# Modal image — deps + pre-baked datasets
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "datasets", "onnx", "onnxscript", "numpy", "fastapi", "Pillow", "huggingface_hub")
    .run_function(preprocess_datasets)
)

app = modal.App("ocr-training", image=image)


# ---------------------------------------------------------------------------
# ASGI app — everything using torch/fastapi lives inside serve()
# ---------------------------------------------------------------------------

@app.function(gpu="T4", timeout=600, secrets=[modal.Secret.from_name("huggingface")])
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def serve():
    import hashlib
    import io
    import json
    import os
    import time

    import numpy as np
    import torch
    import torch.nn as nn
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
    from huggingface_hub import hf_hub_download, upload_file, file_exists
    from torch.utils.data import DataLoader, TensorDataset

    HF_TOKEN = os.environ.get("HF_TOKEN", "")

    # --- Config hashing ---

    def compute_config_hash(arch: dict, training: dict) -> str:
        """Canonical JSON of full config → SHA-256 truncated to 16 hex chars."""
        canonical = json.dumps({"architecture": arch, "training": training}, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    # --- HuggingFace cache helpers ---

    def check_hf_cache(config_hash: str) -> dict | None:
        """Check if a cached model exists on HF. Returns parsed metadata or None."""
        meta_path = f"model-lab/{config_hash}/metadata.json"
        try:
            if not file_exists(HF_REPO, meta_path, token=HF_TOKEN):
                return None
            local = hf_hub_download(HF_REPO, meta_path, token=HF_TOKEN)
            with open(local) as f:
                return json.load(f)
        except Exception as e:
            print(f"[cache] Check failed: {e}")
            return None

    def upload_to_hf(config_hash: str, onnx_bytes: bytes, metadata: dict):
        """Upload model.onnx and metadata.json to HF repo."""
        prefix = f"model-lab/{config_hash}"
        try:
            upload_file(
                path_or_fileobj=onnx_bytes,
                path_in_repo=f"{prefix}/model.onnx",
                repo_id=HF_REPO,
                token=HF_TOKEN,
            )
            meta_bytes = json.dumps(metadata, indent=2).encode()
            upload_file(
                path_or_fileobj=meta_bytes,
                path_in_repo=f"{prefix}/metadata.json",
                repo_id=HF_REPO,
                token=HF_TOKEN,
            )
            print(f"[cache] Uploaded to {prefix}/")
        except Exception as e:
            print(f"[cache] Upload failed: {e}")

    def get_model_url(config_hash: str) -> str:
        return f"https://huggingface.co/{HF_REPO}/resolve/main/model-lab/{config_hash}/model.onnx"

    # --- Activation factory ---

    def make_activation(name: str) -> nn.Module:
        mapping = {
            "relu": lambda: nn.ReLU(),
            "gelu": lambda: nn.GELU(),
            "silu": lambda: nn.SiLU(),
            "leakyRelu": lambda: nn.LeakyReLU(0.01),
            "tanh": lambda: nn.Tanh(),
        }
        return mapping.get(name, lambda: nn.ReLU())()

    # --- Dynamic CNN ---

    class DynamicCNN(nn.Module):
        def __init__(self, conv_layers, dense, num_classes):
            super().__init__()
            self.layers = nn.ModuleDict()
            self.layer_order: list[str] = []
            self.intermediate_names: list[str] = []

            in_channels = 1
            spatial = 28

            for i, cfg in enumerate(conv_layers):
                idx = i + 1
                filters = cfg["filters"]
                ks = cfg["kernelSize"]
                pad = ks // 2

                conv_name = f"conv{idx}"
                self.layers[conv_name] = nn.Conv2d(in_channels, filters, ks, padding=pad)
                self.layer_order.append(conv_name)
                self.intermediate_names.append(conv_name)

                if cfg.get("batchNorm", False):
                    bn_name = f"bn{idx}"
                    self.layers[bn_name] = nn.BatchNorm2d(filters)
                    self.layer_order.append(bn_name)

                act_name = f"act{idx}"
                self.layers[act_name] = make_activation(cfg.get("activation", "relu"))
                self.layer_order.append(act_name)
                self.intermediate_names.append(act_name)

                pooling = cfg.get("pooling", "none")
                if pooling in ("max", "avg"):
                    pool_name = f"pool{idx}"
                    self.layers[pool_name] = (
                        nn.MaxPool2d(2) if pooling == "max" else nn.AvgPool2d(2)
                    )
                    self.layer_order.append(pool_name)
                    self.intermediate_names.append(pool_name)
                    spatial = spatial // 2

                in_channels = filters

            flat_size = in_channels * spatial * spatial

            dense_width = dense.get("width", 128)
            self.layers["dense1"] = nn.Linear(flat_size, dense_width)
            self.layer_order.append("dense1")
            self.intermediate_names.append("dense1")

            self.layers["dense1_act"] = make_activation(dense.get("activation", "relu"))
            self.layer_order.append("dense1_act")
            self.intermediate_names.append("dense1_act")

            dropout = dense.get("dropout", 0.0)
            if dropout > 0:
                self.layers["dropout"] = nn.Dropout(dropout)
                self.layer_order.append("dropout")

            self.layers["output"] = nn.Linear(dense_width, num_classes)
            self.layer_order.append("output")

        def forward(self, x):
            for name in self.layer_order:
                if name == "dense1":
                    x = x.flatten(1)
                x = self.layers[name](x)
            return x

    class MultiOutputWrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, x):
            outputs = []
            for name in self.base.layer_order:
                if name == "dense1":
                    x = x.flatten(1)
                x = self.base.layers[name](x)
                if name in self.base.intermediate_names or name == "output":
                    outputs.append(x)
            return tuple(outputs)

    # --- Dataset: load pre-baked .npy files (instant) ---

    def _load_data(dataset_type: str):
        d = f"{DATA_DIR}/{dataset_type}"
        return {
            "train_images": np.load(f"{d}/train_images.npy"),
            "train_labels": np.load(f"{d}/train_labels.npy"),
            "test_images": np.load(f"{d}/test_images.npy"),
            "test_labels": np.load(f"{d}/test_labels.npy"),
        }

    def _stratified_subsample(images, labels, max_samples, rng):
        if len(images) <= max_samples:
            return images, labels

        classes = np.unique(labels)
        per_class = max(1, max_samples // len(classes))
        selected = []
        for c in classes:
            idx = np.where(labels == c)[0]
            rng.shuffle(idx)
            selected.extend(idx[:per_class].tolist())

        selected_set = set(selected)
        if len(selected) < max_samples:
            remaining = [i for i in range(len(labels)) if i not in selected_set]
            rng.shuffle(remaining)
            selected.extend(remaining[: max_samples - len(selected)])

        selected = np.array(selected)
        return images[selected], labels[selected]

    def make_loaders(dataset_type: str, max_samples: int, batch_size: int):
        data = _load_data(dataset_type)
        rng = np.random.default_rng(42)

        train_imgs, train_labels = _stratified_subsample(
            data["train_images"], data["train_labels"], max_samples, rng
        )
        test_imgs, test_labels = _stratified_subsample(
            data["test_images"], data["test_labels"], max(max_samples // 4, 1000), rng
        )

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(train_imgs), torch.from_numpy(train_labels)),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(test_imgs), torch.from_numpy(test_labels)),
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        return train_loader, test_loader

    # --- ONNX export ---

    def export_onnx_bytes(model):
        model.eval()
        wrapper = MultiOutputWrapper(model)
        wrapper.eval()

        output_names = list(model.intermediate_names) + ["output"]
        dummy = torch.randn(1, 1, 28, 28, device=next(model.parameters()).device)

        buf = io.BytesIO()
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                dummy,
                buf,
                input_names=["input"],
                output_names=output_names,
                dynamic_axes={"input": {0: "batch"}},
                opset_version=17,
            )
        return buf.getvalue(), output_names

    # --- Evaluate ---

    def evaluate(model, loader, criterion, device):
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad(), torch.amp.autocast(device.type):
            for X, y in loader:
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                out = model(X)
                total_loss += criterion(out, y).item() * X.size(0)
                correct += (out.argmax(1) == y).sum().item()
                total += X.size(0)
        return total_loss / max(total, 1), correct / max(total, 1)

    # --- FastAPI app ---

    web_app = FastAPI()

    @web_app.post("/train")
    async def train(request: Request):
        config = await request.json()

        def generate():
            arch = config.get("architecture", {})
            train_cfg = config.get("training", {})

            dataset_type = train_cfg.get("dataset", "digits")
            num_classes = NUM_CLASSES_MAP.get(dataset_type, 10)
            lr = train_cfg.get("learningRate", 0.001)
            epochs = min(train_cfg.get("epochs", 10), 50)
            batch_size = train_cfg.get("batchSize", 64)
            optimizer_name = train_cfg.get("optimizer", "adam")
            max_samples = train_cfg.get("maxSamples", 20000)

            # Compute config hash for caching
            config_hash = compute_config_hash(arch, train_cfg)
            print(f"[train] Config hash: {config_hash}")

            # Check HF cache
            cached_meta = check_hf_cache(config_hash)
            if cached_meta is not None:
                print(f"[train] Cache hit for {config_hash}")
                yield f"data: {json.dumps({'type': 'status', 'message': 'Found cached model'})}\n\n"

                # Replay stored training curve with short delays
                for metric in cached_meta.get("trainingCurve", []):
                    yield f"data: {json.dumps({'type': 'epoch', **metric})}\n\n"
                    time.sleep(0.05)

                yield f"data: {json.dumps({'type': 'complete', 'layerNames': cached_meta['layerNames'], 'numClasses': cached_meta['numClasses'], 'finalMetrics': cached_meta['finalMetrics'], 'modelUrl': get_model_url(config_hash), 'cached': True})}\n\n"
                return

            # Cache miss — train normally
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[train] device={device}, CUDA={torch.cuda.is_available()}")
            if device.type == "cuda":
                print(f"[train] GPU: {torch.cuda.get_device_name(0)}")

            try:
                model = DynamicCNN(
                    arch.get("convLayers", []),
                    arch.get("dense", {}),
                    num_classes,
                ).to(device)

                train_loader, val_loader = make_loaders(dataset_type, max_samples, batch_size)

                if optimizer_name == "sgd":
                    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
                elif optimizer_name == "rmsprop":
                    opt = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-4)
                else:
                    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

                criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    opt, max_lr=lr * 10, epochs=epochs,
                    steps_per_epoch=len(train_loader), pct_start=0.15, anneal_strategy="cos",
                )
                scaler = torch.amp.GradScaler(device.type)
                start_time = time.time()
                val_loss, val_acc = 0.0, 0.0
                training_curve = []

                for epoch in range(1, epochs + 1):
                    model.train()
                    train_loss, train_correct, train_total = 0.0, 0, 0

                    for X, y in train_loader:
                        X = X.to(device, non_blocking=True)
                        y = y.to(device, non_blocking=True)
                        opt.zero_grad(set_to_none=True)
                        with torch.amp.autocast(device.type):
                            out = model(X)
                            loss = criterion(out, y)
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                        scheduler.step()
                        train_loss += loss.item() * X.size(0)
                        train_correct += (out.argmax(1) == y).sum().item()
                        train_total += X.size(0)

                    train_loss /= max(train_total, 1)
                    train_acc = train_correct / max(train_total, 1)
                    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

                    epoch_metrics = {
                        'epoch': epoch,
                        'totalEpochs': epochs,
                        'loss': round(train_loss, 4),
                        'acc': round(train_acc, 4),
                        'valLoss': round(val_loss, 4),
                        'valAcc': round(val_acc, 4),
                        'elapsedSec': round(time.time() - start_time, 1),
                    }
                    training_curve.append(epoch_metrics)

                    yield f"data: {json.dumps({'type': 'epoch', **epoch_metrics})}\n\n"

                # Export ONNX
                print("[train] Exporting ONNX model...")
                model.cpu()
                onnx_bytes, layer_names = export_onnx_bytes(model)
                print(f"[train] ONNX: {len(onnx_bytes)} bytes")

                # Upload to HuggingFace
                yield f"data: {json.dumps({'type': 'status', 'message': 'Uploading model to cache...'})}\n\n"
                metadata = {
                    "configHash": config_hash,
                    "config": {"architecture": arch, "training": train_cfg},
                    "layerNames": layer_names,
                    "numClasses": num_classes,
                    "trainingCurve": training_curve,
                    "finalMetrics": {"valLoss": round(val_loss, 4), "valAcc": round(val_acc, 4)},
                    "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                upload_to_hf(config_hash, onnx_bytes, metadata)

                model_url = get_model_url(config_hash)
                yield f"data: {json.dumps({'type': 'complete', 'layerNames': layer_names, 'numClasses': num_classes, 'finalMetrics': {'valLoss': round(val_loss, 4), 'valAcc': round(val_acc, 4)}, 'modelUrl': model_url, 'cached': False})}\n\n"

            except Exception as e:
                import traceback
                traceback.print_exc()
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return web_app
