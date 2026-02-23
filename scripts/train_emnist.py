# %% [markdown]
# # EMNIST CNN Training & ONNX Export
#
# Train a CNN on EMNIST ByMerge (62 classes: 0-9, A-Z, a-z) and export to ONNX format.
#
# **Exports:**
# - Final ONNX model with all intermediate layer outputs (for visualization)
# - 50 epoch checkpoint ONNX models (for epoch prediction slider)
# - Training history JSON
# - Weight snapshots JSON for weight evolution visualization
#
# **Kaggle setup:** Enable GPU accelerator in Settings → Accelerator → GPU

# %% [code]
# !pip install -q onnxscript

# %% [code]
import os
import json
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

# %% [code]
# Configuration
OUTPUT_DIR = "./emnist-export"
FINAL_MODEL_DIR = os.path.join(OUTPUT_DIR, "models", "emnist-cnn")
CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, "models", "checkpoints")
TRAINING_DIR = os.path.join(OUTPUT_DIR, "training")

for d in [FINAL_MODEL_DIR, CHECKPOINTS_DIR, TRAINING_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 62
EPOCHS = 50
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
BATCH_SIZE = 256 * max(NUM_GPUS, 1)  # scale batch size with GPU count

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    for i in range(NUM_GPUS):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Using {NUM_GPUS} GPU(s), batch size: {BATCH_SIZE}")

# %% [markdown]
# ## Model Architecture
#
# No BatchNorm — so early epochs show genuinely chaotic activations,
# creating a dramatic visual improvement arc across training.
#
# ```
# Input (1, 28, 28)
#   → Conv2D(32, 3x3) → ReLU                      → (32, 28, 28)
#   → Conv2D(64, 3x3) → ReLU → MaxPool            → (64, 14, 14)
#   → Conv2D(128, 3x3) → ReLU → MaxPool           → (128, 7, 7)
#   → Flatten → Dense(256) → ReLU → Dropout(0.5)
#   → Dense(62)
# ```

# %% [code]
class EMNISTNet(nn.Module):
    """CNN matching the visualization layer config (no BatchNorm)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(128 * 7 * 7, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.output(x)
        return x

# %% [code]
class EMNISTNetMultiOutput(nn.Module):
    """Wrapper that outputs all intermediates as a tuple (for ONNX export)."""

    def __init__(self, base_model: EMNISTNet):
        super().__init__()
        self.base = base_model

    def forward(self, x):
        x1 = self.base.conv1(x)
        conv1_out = x1
        x1 = self.base.relu1(x1)
        relu1_out = x1

        x1 = self.base.conv2(x1)
        conv2_out = x1
        x1 = self.base.relu2(x1)
        relu2_out = x1
        x1 = self.base.pool1(x1)
        pool1_out = x1

        x1 = self.base.conv3(x1)
        conv3_out = x1
        x1 = self.base.relu3(x1)
        relu3_out = x1
        x1 = self.base.pool2(x1)
        pool2_out = x1

        x1 = self.base.flatten(x1)
        x1 = self.base.dense1(x1)
        dense1_out = x1
        x1 = self.base.relu4(x1)
        relu4_out = x1
        x1 = self.base.dropout(x1)
        x1 = self.base.output(x1)
        output_out = x1

        return (
            conv1_out, relu1_out,
            conv2_out, relu2_out, pool1_out,
            conv3_out, relu3_out, pool2_out,
            dense1_out, relu4_out,
            output_out,
        )

# %% [markdown]
# ## Load EMNIST Dataset

# %% [code]
train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=10),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
])

data_dir = "./data"
train_ds = datasets.EMNIST(data_dir, split="byclass", train=True, download=True, transform=train_transform)
test_ds = datasets.EMNIST(data_dir, split="byclass", train=False, download=True, transform=test_transform)

print(f"Training: {len(train_ds)} samples")
print(f"Test: {len(test_ds)} samples")
print(f"Classes: {NUM_CLASSES}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=4, pin_memory=True, persistent_workers=True)

# %% [markdown]
# ## Helper Functions

# %% [code]
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad(), torch.amp.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
        for X, y in loader:
            X, y = X.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            out = model(X)
            total_loss += criterion(out, y).item() * X.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += X.size(0)
    return total_loss / total, correct / total


ONNX_OUTPUT_NAMES = [
    "conv1", "relu1",
    "conv2", "relu2", "pool1",
    "conv3", "relu3", "pool2",
    "dense1", "relu4",
    "output",
]

# Reuse across exports to avoid re-allocation
_onnx_dummy = None
_onnx_multi = None

def export_onnx(model, path, verbose=False):
    """Export multi-output ONNX model. Reuses wrapper + dummy tensor."""
    global _onnx_dummy, _onnx_multi
    base = unwrap_model(model)
    base.eval()

    if _onnx_multi is None:
        _onnx_multi = EMNISTNetMultiOutput(base).to(DEVICE)
        _onnx_dummy = torch.randn(1, 1, 28, 28, device=DEVICE)
    else:
        _onnx_multi.base = base

    _onnx_multi.eval()
    torch.onnx.export(
        _onnx_multi, _onnx_dummy, path,
        input_names=["input"],
        output_names=ONNX_OUTPUT_NAMES,
        dynamic_axes={"input": {0: "batch"}},
        opset_version=17,
    )
    if verbose:
        print(f"  Exported multi-output ONNX: {path}")


def unwrap_model(model):
    """Unwrap DataParallel model to get the base EMNISTNet."""
    if hasattr(model, "module"):  # DataParallel
        return model.module
    return model


def get_weight_snapshot(model, epoch):
    """Capture weight stats for visualization."""
    base = unwrap_model(model)
    snapshot = {}
    key_layers = {
        "conv1": base.conv1,
        "conv2": base.conv2,
        "conv3": base.conv3,
        "dense1": base.dense1,
    }
    for name, layer in key_layers.items():
        w = layer.weight.detach().cpu().numpy()
        if w.size > 10000:
            snapshot[name] = {
                "mean": float(np.mean(w)),
                "std": float(np.std(w)),
                "min": float(np.min(w)),
                "max": float(np.max(w)),
                "shape": list(w.shape),
            }
        else:
            snapshot[name] = w.tolist()
    return snapshot

# %% [markdown]
# ## Training Loop

# %% [code]
model = EMNISTNet().to(DEVICE)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Multi-GPU: wrap with DataParallel
if NUM_GPUS > 1:
    model = nn.DataParallel(model)
    print(f"Using DataParallel across {NUM_GPUS} GPUs")

# Use TF32 for faster matmul on Ampere+ GPUs
if DEVICE.type == "cuda":
    torch.set_float32_matmul_precision("high")

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4, fused=DEVICE.type == "cuda")
criterion = nn.CrossEntropyLoss()

# Mixed precision for ~2x speedup on GPU (same model quality)
scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, epochs=EPOCHS,
    steps_per_epoch=len(train_loader),
    pct_start=0.2, anneal_strategy='cos',
    div_factor=25, final_div_factor=1000,
)

history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
weight_snapshots = {}
snapshot_epochs = {0, 1, 2, 5, 10, 15, 20, 25, 30, 40, 49}

# %% [code]
epoch_bar = tqdm(range(EPOCHS), desc="Training", unit="epoch")
for epoch in epoch_bar:
    # Train
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_bar = tqdm(train_loader, desc=f"  Epoch {epoch:2d}", leave=False, unit="batch")
    for X, y in batch_bar:
        X, y = X.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
            out = model(X)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running_loss += loss.item() * X.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += X.size(0)
        batch_bar.set_postfix(loss=f"{running_loss/total:.4f}", acc=f"{correct/total:.3f}")

    train_loss = running_loss / total
    train_acc = correct / total

    # Validate
    val_loss, val_acc = evaluate(model, test_loader)

    history["loss"].append(train_loss)
    history["accuracy"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_acc)

    epoch_bar.set_postfix(
        loss=f"{train_loss:.4f}",
        acc=f"{train_acc:.3f}",
        val_acc=f"{val_acc:.3f}",
        lr=f"{optimizer.param_groups[0]['lr']:.1e}",
    )

    # Save checkpoint ONNX
    ckpt_dir = os.path.join(CHECKPOINTS_DIR, f"epoch-{epoch:02d}")
    os.makedirs(ckpt_dir, exist_ok=True)
    export_onnx(model, os.path.join(ckpt_dir, "model.onnx"))

    # Weight snapshot for select epochs
    if epoch in snapshot_epochs:
        weight_snapshots[str(epoch)] = get_weight_snapshot(model, epoch)

print(f"\nTraining complete!")

# %% [markdown]
# ## Export Final Model & Training Data

# %% [code]
# Export final multi-output model (all intermediate activations)
print("Exporting final multi-output ONNX model...")
export_onnx(model, os.path.join(FINAL_MODEL_DIR, "model.onnx"), verbose=True)

# Save training history
history_path = os.path.join(TRAINING_DIR, "history.json")
with open(history_path, "w") as f:
    json.dump(history, f, indent=2)
print(f"Training history saved: {history_path}")

# Save weight snapshots
snapshots_path = os.path.join(TRAINING_DIR, "weight-snapshots.json")
with open(snapshots_path, "w") as f:
    json.dump(weight_snapshots, f, indent=2)
print(f"Weight snapshots saved: {snapshots_path}")

# Final stats
final_val_loss, final_val_acc = evaluate(model, test_loader)
print(f"\nFinal validation accuracy: {final_val_acc:.4f}")

# %% [markdown]
# ## Package for Download
#
# Creates a zip file with the folder structure the web app expects:
# ```
# emnist-export/
#   models/
#     emnist-cnn/model.onnx        ← main model (multi-output)
#     checkpoints/epoch-00/model.onnx
#     checkpoints/epoch-01/model.onnx
#     ...
#     checkpoints/epoch-49/model.onnx
#   training/
#     history.json
#     weight-snapshots.json
# ```
#
# After downloading, extract into `public/` in the project:
# ```bash
# unzip emnist-export.zip -d public/
# ```

# %% [code]
# Create zip for easy download
zip_path = "./emnist-export"
shutil.make_archive(zip_path, "zip", ".", "emnist-export")
zip_size = os.path.getsize(f"{zip_path}.zip") / (1024 * 1024)
print(f"\nDownload ready: emnist-export.zip ({zip_size:.1f} MB)")
print("Extract into your project's public/ folder:")
print("  unzip emnist-export.zip -d public/")
