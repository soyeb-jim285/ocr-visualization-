#!/usr/bin/env python3
"""
Upload EMNIST + BanglaLekha as a unified HuggingFace Dataset.

Creates a single dataset at soyeb-jim285/ocr-handwriting-data with:
  - 146 classes (0-61 EMNIST, 62-145 BanglaLekha)
  - Columns: image, label, character, script, category, unicode, source
  - Train/test splits (EMNIST native; BanglaLekha stratified 80/20)

Requires: pip install datasets huggingface_hub Pillow numpy tqdm scikit-learn
Requires: huggingface-cli login (or HF_TOKEN env var)
"""

import gzip
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, Features, Image, Value
from PIL import Image as PILImage
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_EMNIST_CLASSES = 62
NUM_BANGLA_CLASSES = 84
NUM_CLASSES = NUM_EMNIST_CLASSES + NUM_BANGLA_CLASSES  # 146

HF_REPO = "soyeb-jim285/ocr-handwriting-data"

BANGLALEKHA_ROOT = Path(__file__).resolve().parent.parent / "BanglaLekha-Isolated"

# ---------------------------------------------------------------------------
# Label metadata — exact copy from train_combined.py
# ---------------------------------------------------------------------------

_EMNIST_LABELS = (
    [str(d) for d in range(10)]
    + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    + [chr(c) for c in range(ord("a"), ord("z") + 1)]
)

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
    # Signs/modifiers (folders 47-50)
    ("ৎ", "U+09CE", "sign"),
    ("ং", "U+0982", "sign"),
    ("ঃ", "U+0983", "sign"),
    ("ঁ", "U+0981", "sign"),
    # Digits (folders 51-60)
    ("০", "U+09E6", "bn_digit"),
    ("১", "U+09E7", "bn_digit"),
    ("২", "U+09E8", "bn_digit"),
    ("৩", "U+09E9", "bn_digit"),
    ("৪", "U+09EA", "bn_digit"),
    ("৫", "U+09EB", "bn_digit"),
    ("৬", "U+09EC", "bn_digit"),
    ("৭", "U+09ED", "bn_digit"),
    ("৮", "U+09EE", "bn_digit"),
    ("৯", "U+09EF", "bn_digit"),
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

assert len(_BANGLA_CHARS) == 84


def _emnist_category(idx: int) -> str:
    if idx < 10:
        return "digit"
    if idx < 36:
        return "uppercase"
    return "lowercase"


# Build full label info lookup: index → (character, script, category, unicode)
LABEL_INFO: dict[int, tuple[str, str, str, str]] = {}
for i, label in enumerate(_EMNIST_LABELS):
    LABEL_INFO[i] = (label, "latin", _emnist_category(i), f"U+{ord(label):04X}")
for i, (char, uni, cat) in enumerate(_BANGLA_CHARS):
    idx = NUM_EMNIST_CLASSES + i
    LABEL_INFO[idx] = (char, "bengali", cat, uni)


# ---------------------------------------------------------------------------
# BanglaLekha processing (from train_combined.py)
# ---------------------------------------------------------------------------

def resize_pad(img: PILImage.Image, target_size: int = 28) -> PILImage.Image:
    """Aspect-preserving resize to fit in target_size×target_size, zero-pad remainder."""
    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), PILImage.LANCZOS)
    padded = PILImage.new("L", (target_size, target_size), 0)
    x_off = (target_size - new_w) // 2
    y_off = (target_size - new_h) // 2
    padded.paste(img, (x_off, y_off))
    return padded


def load_banglalekha() -> tuple[list[PILImage.Image], list[int]]:
    """Load all BanglaLekha PNGs, preprocess to 28×28 transposed PIL images."""
    images_dir = BANGLALEKHA_ROOT / "Images"
    if not images_dir.is_dir():
        print(f"ERROR: BanglaLekha not found at {images_dir}")
        sys.exit(1)

    images: list[PILImage.Image] = []
    labels: list[int] = []

    for folder_num in range(1, 85):
        folder_path = images_dir / str(folder_num)
        if not folder_path.is_dir():
            print(f"  Warning: Missing folder {folder_path}")
            continue
        label = NUM_EMNIST_CLASSES + folder_num - 1  # folder 1 → 62

        pngs = sorted(f for f in folder_path.iterdir() if f.suffix.lower() == ".png")
        for fpath in tqdm(pngs, desc=f"Folder {folder_num:2d} ({LABEL_INFO[label][0]})", leave=False):
            img = PILImage.open(fpath).convert("L")
            img = resize_pad(img, 28)
            img = img.transpose(PILImage.TRANSPOSE)
            images.append(img)
            labels.append(label)

    print(f"BanglaLekha: {len(images)} images loaded")
    return images, labels


def _read_idx(path: Path) -> np.ndarray:
    """Read an IDX file into a numpy array. Handles both raw and gzipped."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        dtype_code = (magic >> 8) & 0xFF
        ndim = magic & 0xFF
        dtype = {0x08: np.uint8, 0x09: np.int8, 0x0D: np.float32}[dtype_code]
        dims = [struct.unpack(">I", f.read(4))[0] for _ in range(ndim)]
        return np.frombuffer(f.read(), dtype=dtype).reshape(dims)


EMNIST_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "EMNIST" / "raw"

# HuggingFace mirror of EMNIST ByClass (NIST URL returns 403)
EMNIST_HF_REPO = "Royc30ne/emnist-byclass"
EMNIST_FILES = [
    "emnist-byclass-train-images-idx3-ubyte.gz",
    "emnist-byclass-train-labels-idx1-ubyte.gz",
    "emnist-byclass-test-images-idx3-ubyte.gz",
    "emnist-byclass-test-labels-idx1-ubyte.gz",
]


def _ensure_emnist() -> Path:
    """Download EMNIST ByClass from HuggingFace mirror if not present locally."""
    from huggingface_hub import hf_hub_download

    # Check if uncompressed files already exist
    expected = EMNIST_RAW_DIR / "emnist-byclass-train-images-idx3-ubyte"
    if expected.exists() and expected.stat().st_size > 500_000_000:
        print("EMNIST already present locally.")
        return EMNIST_RAW_DIR

    # Download gzipped files from HF mirror, decompress to EMNIST_RAW_DIR
    EMNIST_RAW_DIR.mkdir(parents=True, exist_ok=True)
    for fname in EMNIST_FILES:
        out_path = EMNIST_RAW_DIR / fname.removesuffix(".gz")
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"  {out_path.name} already exists, skipping.")
            continue
        print(f"Downloading {fname} from {EMNIST_HF_REPO}...")
        gz_path = Path(hf_hub_download(
            repo_id=EMNIST_HF_REPO, filename=fname, repo_type="dataset"
        ))
        print(f"  Decompressing to {out_path.name}...")
        with gzip.open(gz_path, "rb") as gz_in, open(out_path, "wb") as f_out:
            while chunk := gz_in.read(1 << 20):
                f_out.write(chunk)
    print("EMNIST download complete.")
    return EMNIST_RAW_DIR


def load_emnist() -> tuple[list[PILImage.Image], list[int], list[PILImage.Image], list[int]]:
    """Load EMNIST ByClass from raw IDX files, downloading from HF if needed."""
    raw_dir = _ensure_emnist()

    results = []
    for split in ("train", "test"):
        print(f"Reading EMNIST ByClass {split}...")
        images_arr = _read_idx(raw_dir / f"emnist-byclass-{split}-images-idx3-ubyte")
        labels_arr = _read_idx(raw_dir / f"emnist-byclass-{split}-labels-idx1-ubyte")

        # images_arr shape: (N, 28, 28), stored in transposed orientation (EMNIST convention)
        # We keep them as-is — the model expects this transposed form
        images = []
        for i in tqdm(range(len(images_arr)), desc=f"Converting {split}", leave=False):
            images.append(PILImage.fromarray(images_arr[i], mode="L"))
        labels = labels_arr.tolist()

        print(f"  {split}: {len(images)} images")
        results.append((images, labels))

    train_images, train_labels = results[0]
    test_images, test_labels = results[1]
    print(f"EMNIST total: {len(train_images)} train + {len(test_images)} test")
    return train_images, train_labels, test_images, test_labels


# ---------------------------------------------------------------------------
# Build HF Dataset
# ---------------------------------------------------------------------------

def build_rows(images: list[PILImage.Image], labels: list[int], source: str) -> dict:
    """Build column dict for HF Dataset from lists of PIL images + labels."""
    characters = []
    scripts = []
    categories = []
    unicodes = []

    for label in labels:
        char, script, category, uni = LABEL_INFO[label]
        characters.append(char)
        scripts.append(script)
        categories.append(category)
        unicodes.append(uni)

    return {
        "image": images,
        "label": labels,
        "character": characters,
        "script": scripts,
        "category": categories,
        "unicode": unicodes,
        "source": [source] * len(images),
    }


def merge_rows(a: dict, b: dict) -> dict:
    """Merge two column dicts."""
    return {k: a[k] + b[k] for k in a}


def main():
    # --- Load EMNIST ---
    emnist_train_imgs, emnist_train_labels, emnist_test_imgs, emnist_test_labels = load_emnist()

    # --- Load BanglaLekha ---
    bangla_imgs, bangla_labels = load_banglalekha()

    # --- Stratified 80/20 split for BanglaLekha ---
    print("Splitting BanglaLekha 80/20 stratified...")
    indices = list(range(len(bangla_imgs)))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=bangla_labels
    )
    bangla_train_imgs = [bangla_imgs[i] for i in train_idx]
    bangla_train_labels = [bangla_labels[i] for i in train_idx]
    bangla_test_imgs = [bangla_imgs[i] for i in test_idx]
    bangla_test_labels = [bangla_labels[i] for i in test_idx]
    print(f"  BanglaLekha split: {len(bangla_train_imgs)} train + {len(bangla_test_imgs)} test")

    # --- Build rows ---
    print("Building dataset rows...")
    train_emnist = build_rows(emnist_train_imgs, emnist_train_labels, "emnist")
    test_emnist = build_rows(emnist_test_imgs, emnist_test_labels, "emnist")
    train_bangla = build_rows(bangla_train_imgs, bangla_train_labels, "banglalekha")
    test_bangla = build_rows(bangla_test_imgs, bangla_test_labels, "banglalekha")

    train_rows = merge_rows(train_emnist, train_bangla)
    test_rows = merge_rows(test_emnist, test_bangla)

    # --- Create HF Dataset ---
    features = Features({
        "image": Image(),
        "label": Value("int32"),
        "character": Value("string"),
        "script": Value("string"),
        "category": Value("string"),
        "unicode": Value("string"),
        "source": Value("string"),
    })

    print(f"Creating HF Dataset: {len(train_rows['label'])} train, {len(test_rows['label'])} test")
    ds = DatasetDict({
        "train": Dataset.from_dict(train_rows, features=features),
        "test": Dataset.from_dict(test_rows, features=features),
    })
    print(ds)

    # --- Save label mapping as standalone JSON ---
    label_mapping = {}
    for idx, (char, script, category, uni) in LABEL_INFO.items():
        label_mapping[str(idx)] = {
            "label": char,
            "script": script,
            "category": category,
            "unicode": uni,
        }
    mapping_path = Path(__file__).resolve().parent / "label_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)
    print(f"Label mapping saved to {mapping_path}")

    # --- Push to HuggingFace Hub ---
    print(f"\nPushing to {HF_REPO}...")
    ds.push_to_hub(HF_REPO)
    print(f"Done! Dataset available at https://huggingface.co/datasets/{HF_REPO}")

    # --- Quick verification ---
    print("\n--- Verification ---")
    print(f"Train: {ds['train'].num_rows} rows")
    print(f"Test:  {ds['test'].num_rows} rows")
    print(f"Columns: {ds['train'].column_names}")
    # Spot-check first and last labels
    print(f"Label 0: {LABEL_INFO[0][0]} (should be '0')")
    print(f"Label 62: {LABEL_INFO[62][0]} (should be 'অ')")
    print(f"Label 145: {LABEL_INFO[145][0]} (should be 'হ্ম')")


if __name__ == "__main__":
    main()
