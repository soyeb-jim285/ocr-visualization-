"""
RunPod Serverless Handler — GPU Training for OCR Model Lab

Accepts CNN architecture config, trains on soyeb-jim285/ocr-handwriting-data,
yields epoch metrics via RunPod streaming, returns ONNX model as base64.
"""

import base64
import io
import json
import time

import numpy as np
import runpod
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_DATASET = "soyeb-jim285/ocr-handwriting-data"
NUM_CLASSES_MAP = {"digits": 10, "emnist": 62, "bengali": 84, "combined": 146}

_dataset_cache: dict[str, object] = {}

# ---------------------------------------------------------------------------
# Activation factory
# ---------------------------------------------------------------------------

ACTIVATION_MAP = {
    "relu": lambda: nn.ReLU(),
    "gelu": lambda: nn.GELU(),
    "silu": lambda: nn.SiLU(),
    "leakyRelu": lambda: nn.LeakyReLU(0.01),
    "tanh": lambda: nn.Tanh(),
}


def make_activation(name: str) -> nn.Module:
    return ACTIVATION_MAP.get(name, lambda: nn.ReLU())()


# ---------------------------------------------------------------------------
# Dynamic model builder
# ---------------------------------------------------------------------------


class DynamicCNN(nn.Module):
    def __init__(self, conv_layers: list[dict], dense: dict, num_classes: int):
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

        self._flat_size = in_channels * spatial * spatial

        dense_width = dense.get("width", 128)
        self.layers["dense1"] = nn.Linear(self._flat_size, dense_width)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for name in self.layer_order:
            if name == "dense1":
                x = x.flatten(1)
            x = self.layers[name](x)
        return x


class MultiOutputWrapper(nn.Module):
    def __init__(self, base: DynamicCNN):
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor):
        outputs = []
        for name in self.base.layer_order:
            if name == "dense1":
                x = x.flatten(1)
            x = self.base.layers[name](x)
            if name in self.base.intermediate_names or name == "output":
                outputs.append(x)
        return tuple(outputs)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _load_hf_data(dataset_type: str):
    if dataset_type in _dataset_cache:
        return _dataset_cache[dataset_type]

    print(f"Loading dataset '{dataset_type}' from {HF_DATASET}...")
    ds = load_dataset(HF_DATASET)

    def process_split(split_ds):
        if dataset_type == "digits":
            split_ds = split_ds.filter(lambda x: x["label"] < 10)
        elif dataset_type == "emnist":
            split_ds = split_ds.filter(lambda x: x["label"] < 62)
        elif dataset_type == "bengali":
            split_ds = split_ds.filter(lambda x: x["label"] >= 62)

        images, labels = [], []
        for row in split_ds:
            arr = np.array(row["image"].convert("L"), dtype=np.float32) / 255.0
            images.append(arr)
            label = row["label"]
            if dataset_type == "bengali":
                label -= 62
            labels.append(label)

        return (
            np.array(images)[:, np.newaxis, :, :],
            np.array(labels, dtype=np.int64),
        )

    train_imgs, train_labels = process_split(ds["train"])
    test_imgs, test_labels = process_split(ds["test"])

    result = {
        "train_images": train_imgs,
        "train_labels": train_labels,
        "test_images": test_imgs,
        "test_labels": test_labels,
    }
    _dataset_cache[dataset_type] = result
    print(f"  Loaded: {len(train_imgs)} train, {len(test_imgs)} test")
    return result


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
    data = _load_hf_data(dataset_type)
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
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(test_imgs), torch.from_numpy(test_labels)),
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


def export_onnx_bytes(model: DynamicCNN) -> tuple[bytes, list[str]]:
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


# ---------------------------------------------------------------------------
# Training evaluate
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# RunPod Handler (generator for streaming)
# ---------------------------------------------------------------------------


def handler(event):
    """RunPod serverless handler. Yields epoch metrics, then final ONNX model."""
    config = event["input"]

    # Health-check ping — just confirm the worker is alive
    if config.get("ping"):
        return {"type": "pong"}

    arch = config.get("architecture", {})
    train_cfg = config.get("training", {})

    dataset_type = train_cfg.get("dataset", "digits")
    num_classes = NUM_CLASSES_MAP.get(dataset_type, 10)
    lr = train_cfg.get("learningRate", 0.001)
    epochs = min(train_cfg.get("epochs", 10), 50)
    batch_size = train_cfg.get("batchSize", 64)
    optimizer_name = train_cfg.get("optimizer", "adam")
    max_samples = train_cfg.get("maxSamples", 20000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[handler] Using device: {device} | CUDA available: {torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"[handler] GPU: {torch.cuda.get_device_name(0)}")

    try:
        model = DynamicCNN(
            arch.get("convLayers", []),
            arch.get("dense", {}),
            num_classes,
        ).to(device)

        train_loader, val_loader = make_loaders(dataset_type, max_samples, batch_size)

        if optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
            )
        elif optimizer_name == "rmsprop":
            optimizer = torch.optim.RMSprop(
                model.parameters(), lr=lr, weight_decay=1e-4
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=1e-4
            )

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr * 10,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.15,
            anneal_strategy="cos",
        )
        scaler = torch.amp.GradScaler(device.type)
        start_time = time.time()
        val_loss, val_acc = 0.0, 0.0

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            for X, y in train_loader:
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device.type):
                    out = model(X)
                    loss = criterion(out, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                train_loss += loss.item() * X.size(0)
                train_correct += (out.argmax(1) == y).sum().item()
                train_total += X.size(0)

            train_loss /= max(train_total, 1)
            train_acc = train_correct / max(train_total, 1)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            # Yield epoch metrics (RunPod streaming)
            yield {
                "type": "epoch",
                "epoch": epoch,
                "totalEpochs": epochs,
                "loss": round(train_loss, 4),
                "acc": round(train_acc, 4),
                "valLoss": round(val_loss, 4),
                "valAcc": round(val_acc, 4),
                "elapsedSec": round(time.time() - start_time, 1),
            }

        # Export ONNX
        model.cpu()
        onnx_bytes, layer_names = export_onnx_bytes(model)
        onnx_b64 = base64.b64encode(onnx_bytes).decode("ascii")

        yield {
            "type": "complete",
            "onnxBase64": onnx_b64,
            "layerNames": layer_names,
            "numClasses": num_classes,
            "finalMetrics": {
                "valLoss": round(val_loss, 4),
                "valAcc": round(val_acc, 4),
            },
        }

    except Exception as e:
        yield {"type": "error", "message": str(e)}


runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
