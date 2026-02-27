"""
OCR Model Lab — GPU Training Backend

Gradio app for HuggingFace Spaces (ZeroGPU). Accepts a CNN architecture config,
trains on soyeb-jim285/ocr-handwriting-data, streams epoch metrics via SSE,
and returns a multi-output ONNX model as base64.
"""

import base64
import io
import json
import time

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset

# ZeroGPU decorator — no-op on CPU-only Spaces
try:
    import spaces
    gpu_decorator = spaces.GPU(duration=300)
except (ImportError, Exception):
    gpu_decorator = lambda fn: fn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_DATASET = "soyeb-jim285/ocr-handwriting-data"
NUM_CLASSES_MAP = {"digits": 10, "emnist": 62, "bengali": 84, "combined": 146}

# Module-level dataset cache
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
    factory = ACTIVATION_MAP.get(name)
    if factory is None:
        return nn.ReLU()
    return factory()


# ---------------------------------------------------------------------------
# Dynamic model builder
# ---------------------------------------------------------------------------


class DynamicCNN(nn.Module):
    """CNN built dynamically from a frontend ArchitectureConfig."""

    def __init__(self, conv_layers: list[dict], dense: dict, num_classes: int):
        super().__init__()
        self.layers = nn.ModuleDict()
        self.layer_order: list[str] = []  # execution order
        self.intermediate_names: list[str] = []  # names to export in ONNX

        in_channels = 1
        spatial = 28

        # Conv blocks
        for i, cfg in enumerate(conv_layers):
            idx = i + 1
            filters = cfg["filters"]
            ks = cfg["kernelSize"]
            pad = ks // 2

            # Conv
            conv_name = f"conv{idx}"
            self.layers[conv_name] = nn.Conv2d(in_channels, filters, ks, padding=pad)
            self.layer_order.append(conv_name)
            self.intermediate_names.append(conv_name)

            # BatchNorm (optional)
            if cfg.get("batchNorm", False):
                bn_name = f"bn{idx}"
                self.layers[bn_name] = nn.BatchNorm2d(filters)
                self.layer_order.append(bn_name)

            # Activation
            act_name = f"act{idx}"
            self.layers[act_name] = make_activation(cfg.get("activation", "relu"))
            self.layer_order.append(act_name)
            self.intermediate_names.append(act_name)

            # Pooling (optional)
            pooling = cfg.get("pooling", "none")
            if pooling in ("max", "avg"):
                pool_name = f"pool{idx}"
                if pooling == "max":
                    self.layers[pool_name] = nn.MaxPool2d(2)
                else:
                    self.layers[pool_name] = nn.AvgPool2d(2)
                self.layer_order.append(pool_name)
                self.intermediate_names.append(pool_name)
                spatial = spatial // 2

            in_channels = filters

        # Flatten dim
        self._flat_size = in_channels * spatial * spatial

        # Dense hidden
        dense_width = dense.get("width", 128)
        self.layers["dense1"] = nn.Linear(self._flat_size, dense_width)
        self.layer_order.append("dense1")
        self.intermediate_names.append("dense1")

        dense_act_name = "dense1_act"
        self.layers[dense_act_name] = make_activation(dense.get("activation", "relu"))
        self.layer_order.append(dense_act_name)
        self.intermediate_names.append(dense_act_name)

        # Dropout
        dropout = dense.get("dropout", 0.0)
        if dropout > 0:
            self.layers["dropout"] = nn.Dropout(dropout)
            self.layer_order.append("dropout")

        # Output
        self.layers["output"] = nn.Linear(dense_width, num_classes)
        self.layer_order.append("output")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for name in self.layer_order:
            if name == "dense1":
                x = x.flatten(1)
            x = self.layers[name](x)
        return x


class MultiOutputWrapper(nn.Module):
    """Wraps DynamicCNN to output all intermediate activations for ONNX export."""

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
    """Load and cache the HF dataset, filtered by type."""
    if dataset_type in _dataset_cache:
        return _dataset_cache[dataset_type]

    print(f"Loading dataset '{dataset_type}' from {HF_DATASET}...")
    ds = load_dataset(HF_DATASET)

    def process_split(split_ds):
        # Filter by dataset type
        if dataset_type == "digits":
            split_ds = split_ds.filter(lambda x: x["label"] < 10)
        elif dataset_type == "emnist":
            split_ds = split_ds.filter(lambda x: x["label"] < 62)
        elif dataset_type == "bengali":
            split_ds = split_ds.filter(lambda x: x["label"] >= 62)

        # Extract images and labels
        images = []
        labels = []
        for row in split_ds:
            img = row["image"].convert("L")
            arr = np.array(img, dtype=np.float32) / 255.0
            images.append(arr)
            label = row["label"]
            if dataset_type == "bengali":
                label = label - 62  # remap to 0-83
            labels.append(label)

        images = np.array(images)[:, np.newaxis, :, :]  # (N, 1, 28, 28)
        labels = np.array(labels, dtype=np.int64)
        return images, labels

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
    """Take a stratified subsample of at most max_samples."""
    if len(images) <= max_samples:
        return images, labels

    classes = np.unique(labels)
    per_class = max(1, max_samples // len(classes))
    selected = []
    for c in classes:
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        selected.extend(idx[:per_class].tolist())

    # Fill remaining slots randomly from unselected
    selected_set = set(selected)
    if len(selected) < max_samples:
        remaining = [i for i in range(len(labels)) if i not in selected_set]
        rng.shuffle(remaining)
        selected.extend(remaining[: max_samples - len(selected)])

    selected = np.array(selected)
    return images[selected], labels[selected]


def make_loaders(dataset_type: str, max_samples: int, batch_size: int):
    """Create train/val DataLoaders."""
    data = _load_hf_data(dataset_type)
    rng = np.random.default_rng(42)

    train_imgs, train_labels = _stratified_subsample(
        data["train_images"], data["train_labels"], max_samples, rng
    )
    test_imgs, test_labels = _stratified_subsample(
        data["test_images"], data["test_labels"], max(max_samples // 4, 1000), rng
    )

    train_ds = TensorDataset(
        torch.from_numpy(train_imgs), torch.from_numpy(train_labels)
    )
    test_ds = TensorDataset(
        torch.from_numpy(test_imgs), torch.from_numpy(test_labels)
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


def export_onnx_bytes(model: DynamicCNN) -> tuple[bytes, list[str]]:
    """Export multi-output ONNX to bytes, return (onnx_bytes, layer_names)."""
    model.eval()
    wrapper = MultiOutputWrapper(model)
    wrapper.eval()

    # Output names = intermediate_names + "output"
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
# Training
# ---------------------------------------------------------------------------


def evaluate(model, loader, criterion, device):
    """Evaluate model on a DataLoader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad(), torch.amp.autocast(device.type):
        for X, y in loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(X)
            total_loss += criterion(out, y).item() * X.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += X.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


@gpu_decorator
def train(config_json: str):
    """Train a CNN and stream epoch metrics + final ONNX model."""
    try:
        config = json.loads(config_json)
    except json.JSONDecodeError as e:
        yield json.dumps({"type": "error", "message": f"Invalid JSON: {e}"})
        return

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

    try:
        # Build model
        model = DynamicCNN(
            arch.get("convLayers", []),
            arch.get("dense", {}),
            num_classes,
        ).to(device)

        # Data
        train_loader, val_loader = make_loaders(dataset_type, max_samples, batch_size)

        # Optimizer
        if optimizer_name == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        elif optimizer_name == "rmsprop":
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-4)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr * 10,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.15,
            anneal_strategy="cos",
        )
        scaler = torch.amp.GradScaler()

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            # Train
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

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

            # Validate
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            yield json.dumps({
                "type": "epoch",
                "epoch": epoch,
                "totalEpochs": epochs,
                "loss": round(train_loss, 4),
                "acc": round(train_acc, 4),
                "valLoss": round(val_loss, 4),
                "valAcc": round(val_acc, 4),
                "elapsedSec": round(time.time() - start_time, 1),
            })

        # Export ONNX
        model.cpu()
        onnx_bytes, layer_names = export_onnx_bytes(model)
        onnx_b64 = base64.b64encode(onnx_bytes).decode("ascii")

        yield json.dumps({
            "type": "complete",
            "onnxBase64": onnx_b64,
            "layerNames": layer_names,
            "numClasses": num_classes,
            "finalMetrics": {
                "valLoss": round(val_loss, 4),
                "valAcc": round(val_acc, 4),
            },
        })

    except Exception as e:
        yield json.dumps({"type": "error", "message": str(e)})


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

demo = gr.Interface(
    fn=train,
    inputs=gr.Textbox(label="Config JSON"),
    outputs=gr.Textbox(label="Training Output"),
    api_name="train",
    title="OCR Model Lab — GPU Training",
    description="Submit a CNN architecture config to train on GPU. Used as an API backend.",
)

demo.queue()
demo.launch()
