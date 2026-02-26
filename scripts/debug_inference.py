#!/usr/bin/env python3
"""Debug script: test combined ONNX model on individual images.

Usage:
    # Single BanglaLekha image (auto-detects expected class from folder name)
    python scripts/debug_inference.py /path/to/BanglaLekha-Isolated/Images/5/img001.png

    # Any image
    python scripts/debug_inference.py some_image.png

    # Batch test: random sample from each BanglaLekha folder
    python scripts/debug_inference.py --batch /path/to/BanglaLekha-Isolated

    # Use a specific model
    python scripts/debug_inference.py --model public/models/checkpoints/epoch-74/model.onnx image.png
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

# ---------------------------------------------------------------------------
# Class mapping (matches lib/model/classes.ts)
# ---------------------------------------------------------------------------

EMNIST_LABELS = (
    [str(d) for d in range(10)]
    + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    + [chr(c) for c in range(ord("a"), ord("z") + 1)]
)

# Correct BanglaLekha folder order (verified against dataset images):
#   Folders 1-11:  Vowels
#   Folders 12-50: Consonants + signs
#   Folders 51-60: Digits
#   Folders 61-84: Compounds
BANGLA_CHARS = [
    # Vowels (folders 1-11)
    "অ", "আ", "ই", "ঈ", "উ", "ঊ", "ঋ", "এ", "ঐ", "ও", "ঔ",
    # Consonants + signs (folders 12-50)
    "ক", "খ", "গ", "ঘ", "ঙ", "চ", "ছ", "জ", "ঝ", "ঞ",
    "ট", "ঠ", "ড", "ঢ", "ণ", "ত", "থ", "দ", "ধ", "ন",
    "প", "ফ", "ব", "ভ", "ম", "য", "র", "ল", "শ", "ষ",
    "স", "হ", "ড়", "ঢ়", "য়",
    "ৎ", "ং", "ঃ", "ঁ",
    # Digits (folders 51-60)
    "০", "১", "২", "৩", "৪", "৫", "৬", "৭", "৮", "৯",
    # Compounds (folders 61-84)
    "ক্ষ", "জ্ঞ", "ঞ্চ", "ঞ্ছ", "ঞ্জ", "ত্ত", "ত্র", "দ্ধ", "দ্ব", "ন্ত",
    "ন্দ", "ন্ধ", "ম্প", "ল্ক", "ষ্ট", "স্ত", "ক্ত", "ক্র", "ক্ম", "গ্ন",
    "ঙ্ক", "ঙ্গ", "ণ্ড", "হ্ম",
]

ALL_LABELS = EMNIST_LABELS + BANGLA_CHARS  # 146 total

BYMERGE_MERGED = {38, 44, 45, 46, 47, 48, 50, 51, 54, 56, 57, 58, 59, 60, 61}

# ---------------------------------------------------------------------------
# Preprocessing (matches training script exactly)
# ---------------------------------------------------------------------------

def preprocess_bangla(img_path: str) -> np.ndarray:
    """Preprocess a BanglaLekha image the same way as training.

    1. Load as grayscale
    2. Aspect-preserving resize to fit 28x28, zero-pad
    3. Transpose (rotate 90°) to match EMNIST convention
    4. Normalize to [0, 1] (ToTensor equivalent)
    Returns: np.ndarray shape [1, 1, 28, 28] float32
    """
    img = Image.open(img_path).convert("L")
    img = _resize_pad(img, 28)
    img = img.transpose(Image.TRANSPOSE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.reshape(1, 1, 28, 28)


def preprocess_raw(img_path: str) -> np.ndarray:
    """Preprocess any image (no transpose, no padding — just resize).

    For testing non-BanglaLekha images that are already 28x28 or similar.
    """
    img = Image.open(img_path).convert("L")
    img = img.resize((28, 28), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.reshape(1, 1, 28, 28)


def preprocess_web_style(img_path: str) -> np.ndarray:
    """Preprocess like the web app does (for comparison).

    1. Load as grayscale
    2. Resize to 28x28 (no aspect-preserve padding)
    3. Transpose: tensor[r][c] = resized[c][r]
    4. Normalize to [0, 1]
    """
    img = Image.open(img_path).convert("L")
    img = _resize_pad(img, 28)
    arr = np.array(img, dtype=np.float32) / 255.0
    # Transpose like the web does
    arr = arr.T
    return arr.reshape(1, 1, 28, 28)


def _resize_pad(img: Image.Image, target_size: int) -> Image.Image:
    """Aspect-preserving resize to fit within target_size x target_size, zero-pad."""
    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    padded = Image.new("L", (target_size, target_size), 0)
    x_off = (target_size - new_w) // 2
    y_off = (target_size - new_h) // 2
    padded.paste(img, (x_off, y_off))
    return padded


# ---------------------------------------------------------------------------
# Softmax with ByMerge masking
# ---------------------------------------------------------------------------

def softmax_masked(logits: np.ndarray) -> np.ndarray:
    """Softmax with ByMerge merged indices masked to 0."""
    x = logits.copy()
    for i in BYMERGE_MERGED:
        if i < len(x):
            x[i] = -1e9
    x = x - x.max()
    e = np.exp(x)
    for i in BYMERGE_MERGED:
        if i < len(e):
            e[i] = 0.0
    return e / e.sum()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(session: ort.InferenceSession, tensor: np.ndarray, top_k: int = 10):
    """Run model and return top-k predictions."""
    results = session.run(None, {"input": tensor})
    # Last output is the logits
    logits = results[-1].flatten()
    probs = softmax_masked(logits)

    top_indices = np.argsort(probs)[::-1][:top_k]
    return [(int(i), ALL_LABELS[i], float(probs[i]), float(logits[i])) for i in top_indices]


def guess_expected_class(img_path: str) -> tuple[int, str] | None:
    """Try to determine expected class from BanglaLekha folder structure.

    BanglaLekha: .../Images/{1..84}/xxx.png  → class index = 62 + folder - 1
    """
    parts = Path(img_path).parts
    for i, part in enumerate(parts):
        if part == "Images" and i + 1 < len(parts):
            try:
                folder_num = int(parts[i + 1])
                if 1 <= folder_num <= 84:
                    idx = 62 + folder_num - 1
                    return idx, ALL_LABELS[idx]
            except ValueError:
                pass
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def test_single(session: ort.InferenceSession, img_path: str, preprocess_mode: str = "bangla"):
    """Test a single image."""
    if preprocess_mode == "bangla":
        tensor = preprocess_bangla(img_path)
    elif preprocess_mode == "web":
        tensor = preprocess_web_style(img_path)
    else:
        tensor = preprocess_raw(img_path)

    predictions = run_inference(session, tensor)
    expected = guess_expected_class(img_path)

    # Print results
    print(f"\n{'='*60}")
    print(f"Image: {img_path}")
    if expected:
        print(f"Expected: [{expected[0]}] {expected[1]}")
    print(f"Preprocess: {preprocess_mode}")
    print(f"{'─'*60}")
    print(f"{'Rank':<6} {'Idx':<6} {'Label':<8} {'Prob':>8} {'Logit':>8}")
    print(f"{'─'*60}")

    correct = False
    for rank, (idx, label, prob, logit) in enumerate(predictions):
        marker = ""
        if expected and idx == expected[0]:
            marker = " ✓"
            if rank == 0:
                correct = True
        print(f"  {rank+1:<4} {idx:<6} {label:<8} {prob*100:>7.2f}% {logit:>8.2f}{marker}")

    if expected:
        # Check if expected class is anywhere in top 10
        top_indices = [p[0] for p in predictions]
        if expected[0] in top_indices:
            rank = top_indices.index(expected[0]) + 1
            if rank > 1:
                print(f"\n  ⚠ Expected class at rank {rank}")
        else:
            print(f"\n  ✗ Expected class NOT in top 10")

    return correct, expected is not None


def test_batch(session: ort.InferenceSession, bangla_dir: str, samples_per_class: int = 3):
    """Test random samples from each BanglaLekha folder."""
    images_dir = os.path.join(bangla_dir, "Images")
    if not os.path.isdir(images_dir):
        print(f"Error: {images_dir} not found")
        sys.exit(1)

    total = 0
    correct = 0
    per_class = {}

    for folder_num in range(1, 85):
        folder_path = os.path.join(images_dir, str(folder_num))
        if not os.path.isdir(folder_path):
            continue

        files = [f for f in os.listdir(folder_path) if f.lower().endswith(".png")]
        if not files:
            continue

        sample = random.sample(files, min(samples_per_class, len(files)))
        class_correct = 0
        class_total = 0

        for fname in sample:
            img_path = os.path.join(folder_path, fname)
            is_correct, has_expected = test_single(session, img_path)
            if has_expected:
                total += 1
                class_total += 1
                if is_correct:
                    correct += 1
                    class_correct += 1

        idx = 62 + folder_num - 1
        label = ALL_LABELS[idx]
        per_class[folder_num] = (class_correct, class_total, label)

    # Summary
    print(f"\n{'='*60}")
    print(f"BATCH SUMMARY")
    print(f"{'='*60}")
    print(f"Overall: {correct}/{total} correct ({correct/total*100:.1f}%)" if total > 0 else "No samples tested")
    print()

    # Per-class breakdown
    print(f"{'Folder':<8} {'Label':<8} {'Correct':<10} {'Accuracy':>8}")
    print(f"{'─'*40}")
    for folder_num in sorted(per_class):
        c, t, label = per_class[folder_num]
        acc = f"{c/t*100:.0f}%" if t > 0 else "N/A"
        marker = "✓" if c == t else "✗" if c == 0 else "~"
        print(f"  {folder_num:<6} {label:<8} {c}/{t:<8} {acc:>6}  {marker}")


def main():
    parser = argparse.ArgumentParser(description="Debug combined ONNX model inference")
    parser.add_argument("image", nargs="?", help="Path to image file")
    parser.add_argument("--model", default="public/models/combined-cnn/model.onnx",
                        help="Path to ONNX model (default: public/models/combined-cnn/model.onnx)")
    parser.add_argument("--batch", metavar="BANGLA_DIR",
                        help="Batch test: path to BanglaLekha-Isolated root")
    parser.add_argument("--samples", type=int, default=3,
                        help="Samples per class in batch mode (default: 3)")
    parser.add_argument("--preprocess", choices=["bangla", "web", "raw"], default="bangla",
                        help="Preprocessing mode (default: bangla)")
    parser.add_argument("--compare", action="store_true",
                        help="Run both 'bangla' and 'web' preprocessing and compare")
    args = parser.parse_args()

    if not args.image and not args.batch:
        parser.error("Provide an image path or --batch BANGLA_DIR")

    if not os.path.isfile(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)

    print(f"Loading model: {args.model}")
    session = ort.InferenceSession(args.model)
    print(f"Model inputs:  {[i.name for i in session.get_inputs()]}")
    print(f"Model outputs: {[o.name for o in session.get_outputs()]}")

    if args.batch:
        test_batch(session, args.batch, args.samples)
    elif args.compare:
        print("\n>>> Preprocessing: bangla (training-style)")
        test_single(session, args.image, "bangla")
        print("\n>>> Preprocessing: web (browser-style)")
        test_single(session, args.image, "web")
    else:
        test_single(session, args.image, args.preprocess)


if __name__ == "__main__":
    main()
