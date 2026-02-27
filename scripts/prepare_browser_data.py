#!/usr/bin/env python3
"""
Extract stratified subsets from EMNIST ByClass and BanglaLekha-Isolated
into compact binary files for in-browser training with TensorFlow.js.

Binary format per file:
  Header (12 bytes):
    uint32 totalSamples
    uint32 trainSamples
    uint16 imageSize (28)
    uint16 numClasses
  Body (785 bytes per sample):
    784 uint8 pixels (row-major 28×28) + 1 uint8 label

First trainSamples entries are training data; remainder are test.
EMNIST images are transposed during extraction to match the training convention.

Usage:
  python scripts/prepare_browser_data.py \
    --emnist-dir /path/to/emnist \
    --bangla-dir /path/to/BanglaLekha-Isolated \
    --output-dir public/data \
    --emnist-samples 10000 \
    --bangla-samples 8000 \
    --train-ratio 0.8 \
    --seed 42
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def load_emnist_byclass(emnist_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load EMNIST ByClass from .npz or raw binary files."""
    npz = emnist_dir / "emnist-byclass.npz"
    if npz.exists():
        data = np.load(npz)
        return data["images"], data["labels"]

    # Try standard gzip binary format
    import gzip

    def read_idx(path: Path) -> np.ndarray:
        with gzip.open(path, "rb") as f:
            magic = struct.unpack(">I", f.read(4))[0]
            ndim = magic & 0xFF
            shape = [struct.unpack(">I", f.read(4))[0] for _ in range(ndim)]
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

    train_images = read_idx(emnist_dir / "emnist-byclass-train-images-idx3-ubyte.gz")
    train_labels = read_idx(emnist_dir / "emnist-byclass-train-labels-idx1-ubyte.gz")
    test_images = read_idx(emnist_dir / "emnist-byclass-test-images-idx3-ubyte.gz")
    test_labels = read_idx(emnist_dir / "emnist-byclass-test-labels-idx1-ubyte.gz")

    images = np.concatenate([train_images, test_images])
    labels = np.concatenate([train_labels, test_labels])
    return images, labels


def load_bangla(bangla_dir: Path, image_size: int = 28) -> tuple[np.ndarray, np.ndarray]:
    """Load BanglaLekha-Isolated dataset from folder structure."""
    from PIL import Image

    images_list = []
    labels_list = []

    folders = sorted(
        [d for d in bangla_dir.iterdir() if d.is_dir()],
        key=lambda d: int(d.name),
    )

    for label_idx, folder in enumerate(folders):
        for img_path in sorted(folder.glob("*.png"))[:500]:  # cap per class
            img = Image.open(img_path).convert("L").resize(
                (image_size, image_size), Image.BILINEAR
            )
            images_list.append(np.array(img))
            labels_list.append(label_idx)

    return np.array(images_list), np.array(labels_list)


def stratified_sample(
    images: np.ndarray,
    labels: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a stratified random subset, balanced across classes."""
    classes = np.unique(labels)
    per_class = max(1, n_samples // len(classes))
    remainder = n_samples - per_class * len(classes)

    selected_idx = []
    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        n = min(per_class, len(cls_idx))
        selected_idx.append(rng.choice(cls_idx, size=n, replace=False))

    selected_idx = np.concatenate(selected_idx)

    # Fill remainder from remaining samples
    if remainder > 0 and len(selected_idx) < n_samples:
        remaining = np.setdiff1d(np.arange(len(labels)), selected_idx)
        extra = rng.choice(remaining, size=min(remainder, len(remaining)), replace=False)
        selected_idx = np.concatenate([selected_idx, extra])

    rng.shuffle(selected_idx)
    return images[selected_idx], labels[selected_idx]


def write_binary(
    path: Path,
    images: np.ndarray,
    labels: np.ndarray,
    train_count: int,
    num_classes: int,
    transpose: bool = False,
):
    """Write dataset as compact binary file."""
    n, h, w = images.shape
    assert h == w == 28

    with open(path, "wb") as f:
        # Header: totalSamples(u32), trainSamples(u32), imageSize(u16), numClasses(u16)
        f.write(struct.pack("<IIhh", n, train_count, 28, num_classes))

        for i in range(n):
            img = images[i]
            if transpose:
                img = img.T  # EMNIST transpose convention
            f.write(img.astype(np.uint8).tobytes())
            f.write(struct.pack("B", int(labels[i])))

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  Written {path} — {n} samples, {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Prepare browser training data")
    parser.add_argument("--emnist-dir", type=Path, required=True)
    parser.add_argument("--bangla-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("public/data"))
    parser.add_argument("--emnist-samples", type=int, default=10000)
    parser.add_argument("--bangla-samples", type=int, default=8000)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- EMNIST ---
    print("Loading EMNIST ByClass...")
    emnist_images, emnist_labels = load_emnist_byclass(args.emnist_dir)
    print(f"  Loaded {len(emnist_images)} images, {len(np.unique(emnist_labels))} classes")

    print(f"Sampling {args.emnist_samples} stratified EMNIST samples...")
    e_imgs, e_lbls = stratified_sample(emnist_images, emnist_labels, args.emnist_samples, rng)
    e_train = int(len(e_imgs) * args.train_ratio)
    num_emnist_classes = int(emnist_labels.max()) + 1

    write_binary(
        args.output_dir / "emnist-subset.bin",
        e_imgs, e_lbls, e_train, num_emnist_classes,
        transpose=True,
    )

    # --- BanglaLekha ---
    if args.bangla_dir and args.bangla_dir.exists():
        print("Loading BanglaLekha-Isolated...")
        bangla_images, bangla_labels = load_bangla(args.bangla_dir)
        print(f"  Loaded {len(bangla_images)} images, {len(np.unique(bangla_labels))} classes")

        n_bangla = min(args.bangla_samples, len(bangla_images))
        print(f"Sampling {n_bangla} stratified Bengali samples...")
        b_imgs, b_lbls = stratified_sample(bangla_images, bangla_labels, n_bangla, rng)
        b_train = int(len(b_imgs) * args.train_ratio)
        num_bangla_classes = int(bangla_labels.max()) + 1

        write_binary(
            args.output_dir / "bangla-subset.bin",
            b_imgs, b_lbls, b_train, num_bangla_classes,
            transpose=False,
        )
    else:
        print("Skipping BanglaLekha (--bangla-dir not provided or doesn't exist)")

    print("Done!")


if __name__ == "__main__":
    main()
