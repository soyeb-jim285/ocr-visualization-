#!/usr/bin/env python3
"""
Generate demo .bin training data files for Model Lab.

Creates synthetic character images using font rendering with heavy augmentation
(multiple fonts, rotation, scaling, elastic distortion, stroke width, noise).

Output:
  public/data/emnist-subset.bin  — ~6K samples, 62 classes
  public/data/bangla-subset.bin  — ~4K samples, 84 classes

Run: python scripts/generate_demo_data.py
"""

import struct
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import rotate, zoom, gaussian_filter, map_coordinates

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Requires Pillow: pip install Pillow")
    sys.exit(1)


def find_fonts() -> list[str]:
    """Find available TrueType fonts on the system."""
    candidates = [
        # Arch / general Linux
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSerif.ttf",
        "/usr/share/fonts/TTF/DejaVuSerif-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/noto/NotoSerif-Regular.ttf",
        "/usr/share/fonts/noto/NotoSansBengali-Regular.ttf",
        "/usr/share/fonts/noto/NotoSerifBengali-Regular.ttf",
        # Debian / Ubuntu
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf",
    ]
    found = [p for p in candidates if Path(p).exists()]
    if not found:
        print("Warning: No TrueType fonts found, using default bitmap font")
    return found


def render_char(char: str, font_path: str | None, size: int = 28, font_size: int = 20) -> np.ndarray:
    """Render a character to a grayscale 28x28 numpy array."""
    # Render at 2x then downscale for antialiasing
    render_size = size * 2
    img = Image.new("L", (render_size, render_size), 0)
    draw = ImageDraw.Draw(img)

    if font_path:
        try:
            font = ImageFont.truetype(font_path, font_size * 2)
        except (OSError, IOError):
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), char, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (render_size - tw) // 2 - bbox[0]
    y = (render_size - th) // 2 - bbox[1]

    draw.text((x, y), char, fill=255, font=font)

    # Downscale to target size
    img = img.resize((size, size), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def elastic_distortion(img: np.ndarray, rng: np.random.Generator,
                       alpha: float = 4.0, sigma: float = 3.0) -> np.ndarray:
    """Apply elastic deformation to simulate handwriting variation."""
    h, w = img.shape
    dx = gaussian_filter(rng.standard_normal((h, w)) * alpha, sigma)
    dy = gaussian_filter(rng.standard_normal((h, w)) * alpha, sigma)

    y, x = np.mgrid[0:h, 0:w]
    indices = [np.clip(y + dy, 0, h - 1), np.clip(x + dx, 0, w - 1)]
    return map_coordinates(img.astype(float), indices, order=1).astype(np.uint8)


def augment(img: np.ndarray, rng: np.random.Generator, strength: float = 1.0) -> np.ndarray:
    """Apply random augmentation with configurable strength."""
    result = img.astype(float)
    h, w = result.shape

    # Random rotation (-15 to +15 degrees)
    angle = rng.uniform(-15, 15) * strength
    result = rotate(result, angle, reshape=False, order=1, mode="constant", cval=0)

    # Random scale (0.85 to 1.15)
    scale = 1.0 + rng.uniform(-0.15, 0.15) * strength
    if scale != 1.0:
        zoomed = zoom(result, scale, order=1, mode="constant", cval=0)
        zh, zw = zoomed.shape
        # Center-crop or zero-pad back to original size
        new = np.zeros((h, w))
        sy = max(0, (zh - h) // 2)
        sx = max(0, (zw - w) // 2)
        dy = max(0, (h - zh) // 2)
        dx = max(0, (w - zw) // 2)
        ch = min(h, zh)
        cw = min(w, zw)
        new[dy:dy+ch, dx:dx+cw] = zoomed[sy:sy+ch, sx:sx+cw]
        result = new

    # Random shift (-3 to +3 pixels)
    shift_x = int(rng.integers(-3, 4) * strength)
    shift_y = int(rng.integers(-3, 4) * strength)
    if shift_x != 0 or shift_y != 0:
        shifted = np.zeros_like(result)
        src_x = slice(max(0, -shift_x), min(w, w - shift_x))
        src_y = slice(max(0, -shift_y), min(h, h - shift_y))
        dst_x = slice(max(0, shift_x), min(w, w + shift_x))
        dst_y = slice(max(0, shift_y), min(h, h + shift_y))
        shifted[dst_y, dst_x] = result[src_y, src_x]
        result = shifted

    # Elastic distortion (simulates handwriting wobble)
    if rng.random() < 0.7 * strength:
        result = elastic_distortion(result.astype(np.uint8), rng,
                                     alpha=rng.uniform(2, 6), sigma=rng.uniform(2, 4))

    # Morphological thickening/thinning via dilation/erosion
    if rng.random() < 0.4:
        from scipy.ndimage import binary_dilation, binary_erosion
        mask = result > 30
        if rng.random() < 0.5:
            mask = binary_dilation(mask, iterations=1)
        else:
            mask = binary_erosion(mask, iterations=1)
        result = np.where(mask, np.clip(result, 80, 255), result * 0.1)

    # Gaussian blur (slight defocus)
    if rng.random() < 0.3:
        result = gaussian_filter(result, sigma=rng.uniform(0.3, 0.8))

    # Noise
    noise_std = rng.uniform(3, 20) * strength
    result = result + rng.normal(0, noise_std, result.shape)

    # Intensity variation
    result = result * rng.uniform(0.6, 1.0)
    # Contrast variation
    if rng.random() < 0.3:
        result = np.clip((result - 128) * rng.uniform(0.8, 1.3) + 128, 0, 255)

    return np.clip(result, 0, 255).astype(np.uint8)


def write_binary(path: Path, images: np.ndarray, labels: np.ndarray,
                 train_count: int, num_classes: int):
    """Write dataset as compact binary file."""
    n = len(images)
    with open(path, "wb") as f:
        f.write(struct.pack("<IIhh", n, train_count, 28, num_classes))
        for i in range(n):
            f.write(images[i].astype(np.uint8).tobytes())
            f.write(struct.pack("B", int(labels[i])))

    size_kb = path.stat().st_size / 1024
    print(f"  {path} — {n} samples ({train_count} train, {n - train_count} test), {size_kb:.0f} KB")


def main():
    rng = np.random.default_rng(42)
    output_dir = Path("public/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    fonts = find_fonts()
    latin_fonts = [f for f in fonts if "Bengali" not in f and "bengali" not in f.lower()]
    bangla_fonts = [f for f in fonts if "Bengali" in f or "bengali" in f.lower()]

    # Fallback: if no specific fonts, use all found fonts for both
    if not latin_fonts:
        latin_fonts = fonts or [None]
    if not bangla_fonts:
        bangla_fonts = fonts or [None]

    # --- EMNIST subset: 62 classes (0-9, A-Z, a-z) ---
    print("Generating EMNIST demo data...")
    emnist_chars = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    num_classes_emnist = len(emnist_chars)  # 62
    samples_per_class = 100

    emnist_images = []
    emnist_labels = []

    for label, char in enumerate(emnist_chars):
        # Render with multiple fonts, multiple sizes
        for _ in range(samples_per_class):
            font_path = rng.choice(latin_fonts) if latin_fonts[0] is not None else None
            font_size = int(rng.integers(16, 24))
            base = render_char(char, font_path, font_size=font_size)
            img = augment(base, rng)
            # EMNIST convention: images are transposed
            emnist_images.append(img.T)
            emnist_labels.append(label)

        if (label + 1) % 10 == 0:
            print(f"    {label + 1}/{num_classes_emnist} classes done")

    emnist_images = np.array(emnist_images)
    emnist_labels = np.array(emnist_labels, dtype=np.uint8)

    # Shuffle
    perm = rng.permutation(len(emnist_images))
    emnist_images = emnist_images[perm]
    emnist_labels = emnist_labels[perm]

    train_count_e = int(len(emnist_images) * 0.8)
    write_binary(output_dir / "emnist-subset.bin", emnist_images, emnist_labels,
                 train_count_e, num_classes_emnist)

    # --- Bangla subset: 84 classes ---
    print("Generating Bangla demo data...")
    bangla_chars_raw = list("অআইঈউঊঋএঐওঔ"
                           "কখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ")
    bangla_signs = ["ড়", "ঢ়", "য়", "ৎ", "ং", "ঃ", "ঁ"]
    bangla_digits = list("০১২৩৪৫৬৭৮৯")
    bangla_compounds = ["ক্ষ", "জ্ঞ", "ঞ্চ", "ঞ্ছ", "ঞ্জ", "ত্ত", "ত্র", "দ্ধ", "দ্ব", "ন্ত",
                        "ন্দ", "ন্ধ", "ম্প", "ল্ক", "ষ্ট", "স্ত", "ক্ত", "ক্র", "ক্ম", "গ্ন",
                        "ঙ্ক", "ঙ্গ", "ণ্ড", "হ্ম"]

    all_bangla = bangla_chars_raw + bangla_signs + bangla_digits + bangla_compounds
    all_bangla = all_bangla[:84]
    num_classes_bangla = len(all_bangla)

    samples_per_class_b = 50
    bangla_images = []
    bangla_labels = []

    for label, char in enumerate(all_bangla):
        for _ in range(samples_per_class_b):
            font_path = rng.choice(bangla_fonts) if bangla_fonts[0] is not None else None
            font_size = int(rng.integers(14, 22))
            base = render_char(char, font_path, font_size=font_size)
            img = augment(base, rng)
            bangla_images.append(img)
            bangla_labels.append(label)

        if (label + 1) % 10 == 0:
            print(f"    {label + 1}/{num_classes_bangla} classes done")

    bangla_images = np.array(bangla_images)
    bangla_labels = np.array(bangla_labels, dtype=np.uint8)

    perm = rng.permutation(len(bangla_images))
    bangla_images = bangla_images[perm]
    bangla_labels = bangla_labels[perm]

    train_count_b = int(len(bangla_images) * 0.8)
    write_binary(output_dir / "bangla-subset.bin", bangla_images, bangla_labels,
                 train_count_b, num_classes_bangla)

    print("Done!")


if __name__ == "__main__":
    main()
