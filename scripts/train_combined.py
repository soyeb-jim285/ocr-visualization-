# %% [markdown]
# # Combined EMNIST + BanglaLekha CNN Training & ONNX Export
#
# Train a single CNN on both EMNIST ByClass (62 classes: 0-9, A-Z, a-z)
# and BanglaLekha-Isolated (84 classes: Bengali digits, vowels, consonants, compounds)
# for a total of **146 output classes**.
#
# **Exports:**
# - Final ONNX model with all intermediate layer outputs (for visualization)
# - 75 epoch checkpoint ONNX models (for epoch prediction slider)
# - Training history JSON
# - Weight snapshots JSON for weight evolution visualization
# - Class mapping JSON (index → label, script, unicode)
#
# **Kaggle setup:** Enable GPU accelerator in Settings → Accelerator → GPU
#
# **BanglaLekha dataset:**
# Upload `banglalekha-isolated` dataset from Kaggle Datasets, or set `BANGLA_DIR` below.

# %% [code]
# !pip install -q onnxscript

# %% [code]
# === Configuration ===
# Set BANGLA_DIR to the path containing the BanglaLekha-Isolated dataset
# (the folder with Images/ and Readme.txt inside).
# On Kaggle, this is typically auto-detected.

import os

# Auto-detect Kaggle environment
_KAGGLE = os.path.isdir("/kaggle/input")

BANGLA_DIR = (
    "/kaggle/input/banglalekha-isolated/BanglaLekha-Isolated"
    if _KAGGLE
    else os.environ.get("BANGLA_DIR", "./BanglaLekha-Isolated")
)

# %% [code]
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)
from torchvision import datasets, transforms
from tqdm.auto import tqdm

# CLI override: python scripts/train_combined.py --bn_dir /path/to/BanglaLekha-Isolated
if not hasattr(sys, "ps1") and not _KAGGLE:  # not interactive / not Kaggle
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--bn_dir", type=str, default=None, help="BanglaLekha-Isolated root dir")
    _args, _ = _parser.parse_known_args()
    if _args.bn_dir:
        BANGLA_DIR = _args.bn_dir

print(f"BanglaLekha dir: {BANGLA_DIR}")
assert os.path.isdir(os.path.join(BANGLA_DIR, "Images")), (
    f"BanglaLekha Images/ not found at {BANGLA_DIR}. "
    "Set BANGLA_DIR or pass --bn_dir /path/to/BanglaLekha-Isolated"
)

# %% [code]
# Configuration
OUTPUT_DIR = "./combined-export"
FINAL_MODEL_DIR = os.path.join(OUTPUT_DIR, "models", "combined-cnn")
CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, "models", "checkpoints")
TRAINING_DIR = os.path.join(OUTPUT_DIR, "training")

for d in [FINAL_MODEL_DIR, CHECKPOINTS_DIR, TRAINING_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EMNIST_CLASSES = 62
NUM_BANGLA_CLASSES = 84
NUM_CLASSES = NUM_EMNIST_CLASSES + NUM_BANGLA_CLASSES  # 146

EPOCHS = 75
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
BATCH_SIZE = 256 * max(NUM_GPUS, 1)

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    for i in range(NUM_GPUS):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Using {NUM_GPUS} GPU(s), batch size: {BATCH_SIZE}")
print(f"Total classes: {NUM_CLASSES} (EMNIST {NUM_EMNIST_CLASSES} + BanglaLekha {NUM_BANGLA_CLASSES})")

# %% [markdown]
# ## Class Mapping
#
# - Indices 0–61: EMNIST (0-9, A-Z, a-z) — same as original model
# - Indices 62–145: BanglaLekha (folder 1→62, folder 2→63, ..., folder 84→145)
#
# BanglaLekha folder order (verified against actual dataset images):
# - Folders 1-11:  Vowels (স্বরবর্ণ)
# - Folders 12-50: Consonants + signs (ব্যঞ্জনবর্ণ)
# - Folders 51-60: Bengali digits ০-৯
# - Folders 61-84: Compound characters (যুক্তবর্ণ)

# %% [code]
# Bengali character labels for folders 1-84
# Based on BanglaLekha-Isolated dataset (Biswas et al., 2017)
#
# Correct folder order (verified against actual dataset images):
#   Folders 1-11:  Vowels (স্বরবর্ণ)
#   Folders 12-50: Consonants + signs (ব্যঞ্জনবর্ণ)
#   Folders 51-60: Digits (সংখ্যা)
#   Folders 61-84: Compound characters (যুক্তবর্ণ)
_BANGLA_CHARS = [
    # Vowels (folders 1-11)
    ("অ", "U+0985", "vowel"),
    ("আ", "U+0986", "vowel"),
    ("ই", "U+0987", "vowel"),
    ("ঈ", "U+0988", "vowel"),
    ("উ", "U+0989", "vowel"),
    ("ঊ", "U+098A", "vowel"),
    ("ঋ", "U+098B", "vowel"),
    ("এ", "U+098F", "vowel"),
    ("ঐ", "U+0990", "vowel"),
    ("ও", "U+0993", "vowel"),
    ("ঔ", "U+0994", "vowel"),
    # Consonants (folders 12-50)
    ("ক", "U+0995", "consonant"),
    ("খ", "U+0996", "consonant"),
    ("গ", "U+0997", "consonant"),
    ("ঘ", "U+0998", "consonant"),
    ("ঙ", "U+0999", "consonant"),
    ("চ", "U+099A", "consonant"),
    ("ছ", "U+099B", "consonant"),
    ("জ", "U+099C", "consonant"),
    ("ঝ", "U+099D", "consonant"),
    ("ঞ", "U+099E", "consonant"),
    ("ট", "U+099F", "consonant"),
    ("ঠ", "U+09A0", "consonant"),
    ("ড", "U+09A1", "consonant"),
    ("ঢ", "U+09A2", "consonant"),
    ("ণ", "U+09A3", "consonant"),
    ("ত", "U+09A4", "consonant"),
    ("থ", "U+09A5", "consonant"),
    ("দ", "U+09A6", "consonant"),
    ("ধ", "U+09A7", "consonant"),
    ("ন", "U+09A8", "consonant"),
    ("প", "U+09AA", "consonant"),
    ("ফ", "U+09AB", "consonant"),
    ("ব", "U+09AC", "consonant"),
    ("ভ", "U+09AD", "consonant"),
    ("ম", "U+09AE", "consonant"),
    ("য", "U+09AF", "consonant"),
    ("র", "U+09B0", "consonant"),
    ("ল", "U+09B2", "consonant"),
    ("শ", "U+09B6", "consonant"),
    ("ষ", "U+09B7", "consonant"),
    ("স", "U+09B8", "consonant"),
    ("হ", "U+09B9", "consonant"),
    ("ড়", "U+09DC", "consonant"),
    ("ঢ়", "U+09DD", "consonant"),
    ("য়", "U+09DF", "consonant"),
    # Signs/modifiers (folders 47-50, part of consonant block)
    ("ৎ", "U+09CE", "sign"),
    ("ং", "U+0982", "sign"),
    ("ঃ", "U+0983", "sign"),
    ("ঁ", "U+0981", "sign"),
    # Digits (folders 51-60)
    ("০", "U+09E6", "digit"),
    ("১", "U+09E7", "digit"),
    ("২", "U+09E8", "digit"),
    ("৩", "U+09E9", "digit"),
    ("৪", "U+09EA", "digit"),
    ("৫", "U+09EB", "digit"),
    ("৬", "U+09EC", "digit"),
    ("৭", "U+09ED", "digit"),
    ("৮", "U+09EE", "digit"),
    ("৯", "U+09EF", "digit"),
    # Compound characters (folders 61-84)
    ("ক্ষ", "U+0995+U+09CD+U+09B7", "compound"),
    ("জ্ঞ", "U+099C+U+09CD+U+099E", "compound"),
    ("ঞ্চ", "U+099E+U+09CD+U+099A", "compound"),
    ("ঞ্ছ", "U+099E+U+09CD+U+099B", "compound"),
    ("ঞ্জ", "U+099E+U+09CD+U+099C", "compound"),
    ("ত্ত", "U+09A4+U+09CD+U+09A4", "compound"),
    ("ত্র", "U+09A4+U+09CD+U+09B0", "compound"),
    ("দ্ধ", "U+09A6+U+09CD+U+09A7", "compound"),
    ("দ্ব", "U+09A6+U+09CD+U+09AC", "compound"),
    ("ন্ত", "U+09A8+U+09CD+U+09A4", "compound"),
    ("ন্দ", "U+09A8+U+09CD+U+09A6", "compound"),
    ("ন্ধ", "U+09A8+U+09CD+U+09A7", "compound"),
    ("ম্প", "U+09AE+U+09CD+U+09AA", "compound"),
    ("ল্ক", "U+09B2+U+09CD+U+0995", "compound"),
    ("ষ্ট", "U+09B7+U+09CD+U+099F", "compound"),
    ("স্ত", "U+09B8+U+09CD+U+09A4", "compound"),
    ("ক্ত", "U+0995+U+09CD+U+09A4", "compound"),
    ("ক্র", "U+0995+U+09CD+U+09B0", "compound"),
    ("ক্ম", "U+0995+U+09CD+U+09AE", "compound"),
    ("গ্ন", "U+0997+U+09CD+U+09A8", "compound"),
    ("ঙ্ক", "U+0999+U+09CD+U+0995", "compound"),
    ("ঙ্গ", "U+0999+U+09CD+U+0997", "compound"),
    ("ণ্ড", "U+09A3+U+09CD+U+09A1", "compound"),
    ("হ্ম", "U+09B9+U+09CD+U+09AE", "compound"),
]

assert len(_BANGLA_CHARS) == 84, f"Expected 84 BanglaLekha chars, got {len(_BANGLA_CHARS)}"

# EMNIST labels (indices 0-61)
_EMNIST_LABELS = (
    [str(d) for d in range(10)]
    + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    + [chr(c) for c in range(ord("a"), ord("z") + 1)]
)

# Build full class mapping (146 classes)
CLASS_MAPPING = {}
for i, label in enumerate(_EMNIST_LABELS):
    CLASS_MAPPING[str(i)] = {
        "label": label,
        "script": "latin",
        "unicode": f"U+{ord(label):04X}",
    }
for i, (char, uni, char_type) in enumerate(_BANGLA_CHARS):
    idx = NUM_EMNIST_CLASSES + i  # 62..145
    CLASS_MAPPING[str(idx)] = {
        "label": char,
        "script": "bengali",
        "unicode": uni,
        "type": char_type,
        "bangla_folder": i + 1,
    }

print(f"Class mapping: {len(CLASS_MAPPING)} entries")
print(f"  EMNIST: 0-{NUM_EMNIST_CLASSES - 1} ({_EMNIST_LABELS[0]}..{_EMNIST_LABELS[-1]})")
print(f"  BanglaLekha: {NUM_EMNIST_CLASSES}-{NUM_CLASSES - 1} ({_BANGLA_CHARS[0][0]}..{_BANGLA_CHARS[-1][0]})")

# %% [markdown]
# ## Model Architecture
#
# Bigger CNN with BatchNorm for the harder 146-class task.
# Same layer *types* and *names* as the original for visualization compatibility.
#
# ```
# Input (1, 28, 28)
#   → Conv2d(64, 3x3, pad=1) → BN → ReLU                → (64, 28, 28)
#   → Conv2d(128, 3x3, pad=1) → BN → ReLU → MaxPool(2)  → (128, 14, 14)
#   → Conv2d(256, 3x3, pad=1) → BN → ReLU → MaxPool(2)  → (256, 7, 7)
#   → Flatten → Dense(512) → ReLU → Dropout(0.5)
#   → Dense(146)
# ```
#
# ~6.6M params (vs ~1.7M original). BatchNorm helps with the larger model
# and more diverse dataset.

# %% [code]
class CombinedNet(nn.Module):
    """CNN for combined EMNIST + BanglaLekha (146 classes)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(256 * 7 * 7, 512)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


# %% [code]
class CombinedNetMultiOutput(nn.Module):
    """Wrapper that outputs all intermediates as a tuple (for ONNX export).

    Outputs match the original model's layer names so the web visualization
    works without changes:
      conv1, relu1, conv2, relu2, pool1, conv3, relu3, pool2, dense1, relu4, output
    """

    def __init__(self, base_model: CombinedNet):
        super().__init__()
        self.base = base_model

    def forward(self, x):
        x1 = self.base.conv1(x)
        conv1_out = x1
        x1 = self.base.bn1(x1)
        x1 = self.base.relu1(x1)
        relu1_out = x1

        x1 = self.base.conv2(x1)
        conv2_out = x1
        x1 = self.base.bn2(x1)
        x1 = self.base.relu2(x1)
        relu2_out = x1
        x1 = self.base.pool1(x1)
        pool1_out = x1

        x1 = self.base.conv3(x1)
        conv3_out = x1
        x1 = self.base.bn3(x1)
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
# ## BanglaLekha Dataset
#
# Custom Dataset that loads PNG images from `Images/{1..84}/`, resizes to 28×28
# with aspect-preserving padding, and transposes to match EMNIST convention.
#
# EMNIST raw images are stored transposed — the existing model and web app
# both expect this convention. We transpose BanglaLekha images too so all data
# is in the same orientation.

# %% [code]
class BanglaLekhaDataset(Dataset):
    """BanglaLekha-Isolated dataset with preprocessing for combined training.

    Each image is:
    1. Loaded as grayscale
    2. Aspect-preserving resized to fit in 28×28 with zero-padding
    3. Transposed (rotated 90°) to match EMNIST convention
    4. Labels offset by NUM_EMNIST_CLASSES (folder 1 → idx 62, etc.)
    """

    def __init__(self, root_dir, indices=None, transform=None):
        """
        Args:
            root_dir: Path to BanglaLekha-Isolated root (containing Images/).
            indices: Optional list of sample indices to use (for train/test split).
            transform: torchvision transform applied AFTER resize+transpose.
        """
        self.transform = transform
        self.samples = []  # (path, label)

        images_dir = os.path.join(root_dir, "Images")
        for folder_num in range(1, 85):
            folder_path = os.path.join(images_dir, str(folder_num))
            if not os.path.isdir(folder_path):
                print(f"  Warning: Missing folder {folder_path}")
                continue
            label = NUM_EMNIST_CLASSES + folder_num - 1  # folder 1 → 62
            for fname in sorted(os.listdir(folder_path)):
                if fname.lower().endswith(".png"):
                    self.samples.append((os.path.join(folder_path, fname), label))

        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

        print(f"  BanglaLekha: {len(self.samples)} samples from {images_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")

        # Aspect-preserving resize to fit in 28×28 with zero-padding
        img = self._resize_pad(img, 28)

        # Transpose to match EMNIST convention (rotate 90° CW)
        img = img.transpose(Image.TRANSPOSE)

        # Apply transform (expects PIL Image, outputs tensor)
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        return img, label

    @staticmethod
    def _resize_pad(img, target_size):
        """Resize image to fit within target_size×target_size, zero-pad remainder."""
        w, h = img.size
        scale = target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)

        # Center in target_size×target_size black canvas
        padded = Image.new("L", (target_size, target_size), 0)
        x_off = (target_size - new_w) // 2
        y_off = (target_size - new_h) // 2
        padded.paste(img, (x_off, y_off))
        return padded


# %% [markdown]
# ## Load Datasets

# %% [code]
# --- EMNIST ---
emnist_train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=10),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])
emnist_test_transform = transforms.Compose([
    transforms.ToTensor(),
])

data_dir = "./data"
emnist_train = datasets.EMNIST(data_dir, split="byclass", train=True, download=True,
                                transform=emnist_train_transform)
emnist_test = datasets.EMNIST(data_dir, split="byclass", train=False, download=True,
                               transform=emnist_test_transform)

print(f"EMNIST train: {len(emnist_train)} samples")
print(f"EMNIST test:  {len(emnist_test)} samples")

# %% [code]
# --- BanglaLekha ---
# More aggressive augmentation for BanglaLekha (Bengali characters are more complex)
bangla_train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=12, translate=(0.08, 0.08), scale=(0.88, 1.12), shear=8),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
])
bangla_test_transform = transforms.Compose([
    transforms.ToTensor(),
])

# Stratified 80/20 train/test split
print("Loading BanglaLekha-Isolated...")
_full_ds = BanglaLekhaDataset(BANGLA_DIR)
_n_total = len(_full_ds)
_labels = np.array([s[1] for s in _full_ds.samples])

# Stratified split: same proportion per class
try:
    from sklearn.model_selection import train_test_split as _tts
    _train_idx, _test_idx = _tts(
        np.arange(_n_total), test_size=0.2, random_state=42, stratify=_labels
    )
except ImportError:
    # Fallback: manual stratified split (no sklearn needed)
    print("  sklearn not found, using manual stratified split")
    _rng = np.random.RandomState(42)
    _train_idx, _test_idx = [], []
    for cls in np.unique(_labels):
        cls_idx = np.where(_labels == cls)[0]
        _rng.shuffle(cls_idx)
        split = int(len(cls_idx) * 0.8)
        _train_idx.extend(cls_idx[:split])
        _test_idx.extend(cls_idx[split:])
    _train_idx, _test_idx = np.array(_train_idx), np.array(_test_idx)
print(f"  Split: {len(_train_idx)} train / {len(_test_idx)} test")

bangla_train = BanglaLekhaDataset(BANGLA_DIR, indices=_train_idx, transform=bangla_train_transform)
bangla_test = BanglaLekhaDataset(BANGLA_DIR, indices=_test_idx, transform=bangla_test_transform)

print(f"BanglaLekha train: {len(bangla_train)} samples")
print(f"BanglaLekha test:  {len(bangla_test)} samples")

# %% [code]
# --- Combined datasets ---
train_combined = ConcatDataset([emnist_train, bangla_train])
test_combined = ConcatDataset([emnist_test, bangla_test])

print(f"\nCombined train: {len(train_combined)} samples")
print(f"Combined test:  {len(test_combined)} samples")

# %% [markdown]
# ## Balanced Sampling
#
# EMNIST is ~5× larger than BanglaLekha. Use `WeightedRandomSampler` to
# oversample BanglaLekha and achieve ~60/40 EMNIST/BanglaLekha ratio per epoch
# (slight EMNIST bias since it has more classes).

# %% [code]
_n_emnist = len(emnist_train)
_n_bangla = len(bangla_train)
_n_combined = _n_emnist + _n_bangla

# Target ratio: 60% EMNIST, 40% BanglaLekha
# Weight per sample: w_emnist * n_emnist = 0.6, w_bangla * n_bangla = 0.4
_w_emnist = 0.6 / _n_emnist
_w_bangla = 0.4 / _n_bangla

_sample_weights = np.empty(_n_combined, dtype=np.float64)
_sample_weights[:_n_emnist] = _w_emnist
_sample_weights[_n_emnist:] = _w_bangla

sampler = WeightedRandomSampler(
    weights=_sample_weights,
    num_samples=_n_combined,  # same epoch size as total dataset
    replacement=True,
)

print(f"Sampler weights: EMNIST={_w_emnist:.2e}, BanglaLekha={_w_bangla:.2e}")
print(f"Expected per epoch: ~{_n_combined * 0.6:.0f} EMNIST, ~{_n_combined * 0.4:.0f} BanglaLekha")

train_loader = DataLoader(
    train_combined, batch_size=BATCH_SIZE, sampler=sampler,
    num_workers=4, pin_memory=True, persistent_workers=True,
)
test_loader = DataLoader(
    test_combined, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True, persistent_workers=True,
)

# %% [markdown]
# ## Helper Functions

# %% [code]
def evaluate(model, loader):
    """Evaluate model on a DataLoader, returning loss and accuracy."""
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


def evaluate_per_script(model, emnist_loader, bangla_loader):
    """Evaluate accuracy separately for EMNIST and BanglaLekha."""
    model.eval()
    results = {}
    with torch.no_grad(), torch.amp.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
        for name, loader in [("emnist", emnist_loader), ("bangla", bangla_loader)]:
            correct = 0
            total = 0
            for X, y in loader:
                X, y = X.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                out = model(X)
                correct += (out.argmax(1) == y).sum().item()
                total += X.size(0)
            results[name] = correct / total if total > 0 else 0.0
    return results


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
        _onnx_multi = CombinedNetMultiOutput(base).to(DEVICE)
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
    """Unwrap DataParallel model to get the base CombinedNet."""
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
#
# Key differences from the EMNIST-only script:
# - **Label smoothing** (0.1): helps with 146-class classification
# - **75 epochs**: longer training for the harder combined task
# - **OneCycleLR** with slightly higher max_lr warmup

# %% [code]
model = CombinedNet().to(DEVICE)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Multi-GPU: wrap with DataParallel
if NUM_GPUS > 1:
    model = nn.DataParallel(model)
    print(f"Using DataParallel across {NUM_GPUS} GPUs")

# Use TF32 for faster matmul on Ampere+ GPUs
if DEVICE.type == "cuda":
    torch.set_float32_matmul_precision("high")

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4,
                       fused=DEVICE.type == "cuda")
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Mixed precision for ~2x speedup on GPU
scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, epochs=EPOCHS,
    steps_per_epoch=len(train_loader),
    pct_start=0.15, anneal_strategy="cos",
    div_factor=25, final_div_factor=1000,
)

history = {
    "loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [],
    "emnist_val_accuracy": [], "bangla_val_accuracy": [],
}
weight_snapshots = {}
snapshot_epochs = {0, 1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 74}

# Separate test loaders for per-script evaluation
_emnist_test_loader = DataLoader(
    emnist_test, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=2, pin_memory=True, persistent_workers=True,
)
_bangla_test_loader = DataLoader(
    bangla_test, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=2, pin_memory=True, persistent_workers=True,
)

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

    # Validate (combined)
    val_loss, val_acc = evaluate(model, test_loader)

    # Per-script validation
    per_script = evaluate_per_script(model, _emnist_test_loader, _bangla_test_loader)

    history["loss"].append(train_loss)
    history["accuracy"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_acc)
    history["emnist_val_accuracy"].append(per_script["emnist"])
    history["bangla_val_accuracy"].append(per_script["bangla"])

    epoch_bar.set_postfix(
        loss=f"{train_loss:.4f}",
        acc=f"{train_acc:.3f}",
        val=f"{val_acc:.3f}",
        en=f"{per_script['emnist']:.3f}",
        bn=f"{per_script['bangla']:.3f}",
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

# Save class mapping
mapping_path = os.path.join(TRAINING_DIR, "class-mapping.json")
with open(mapping_path, "w") as f:
    json.dump(CLASS_MAPPING, f, indent=2, ensure_ascii=False)
print(f"Class mapping saved: {mapping_path}")

# Final stats
final_val_loss, final_val_acc = evaluate(model, test_loader)
final_per_script = evaluate_per_script(model, _emnist_test_loader, _bangla_test_loader)
print(f"\nFinal validation accuracy:")
print(f"  Combined:    {final_val_acc:.4f}")
print(f"  EMNIST:      {final_per_script['emnist']:.4f}")
print(f"  BanglaLekha: {final_per_script['bangla']:.4f}")

# %% [markdown]
# ## Package for Download
#
# Creates a zip file with the folder structure the web app expects:
# ```
# combined-export/
#   models/
#     combined-cnn/model.onnx          ← main model (multi-output)
#     checkpoints/epoch-00/model.onnx
#     checkpoints/epoch-01/model.onnx
#     ...
#     checkpoints/epoch-74/model.onnx
#   training/
#     history.json
#     weight-snapshots.json
#     class-mapping.json               ← maps index → {label, script, unicode}
# ```
#
# After downloading, extract into `public/` in the project:
# ```bash
# unzip combined-export.zip -d public/
# ```

# %% [code]
# Create zip for easy download
zip_path = "./combined-export"
shutil.make_archive(zip_path, "zip", ".", "combined-export")
zip_size = os.path.getsize(f"{zip_path}.zip") / (1024 * 1024)
print(f"\nDownload ready: combined-export.zip ({zip_size:.1f} MB)")
print("Extract into your project's public/ folder:")
print("  unzip combined-export.zip -d public/")
