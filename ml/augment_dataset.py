#!/usr/bin/env python3
"""
TremorTrace — Offline Dataset Augmentation

Generates physically augmented copies of training images to expand
the dataset from ~204 to ~1200+ images. Only training images are
augmented — test images stay untouched for honest evaluation.

Usage:
    cd "ctrlhackdel project/ml"
    python augment_dataset.py

Output:
    src/data/drawings_augmented/   — augmented dataset (same directory structure)
"""

import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ============================================================================
# Paths
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, 'src')
ORIGINAL_ROOT = os.path.join(SRC_DIR, 'data', 'drawings')
AUGMENTED_ROOT = os.path.join(SRC_DIR, 'data', 'drawings_augmented')

# Number of augmented copies per original training image
COPIES_PER_IMAGE = 5

IMG_SIZE = 224


# ============================================================================
# Augmentation Functions
# ============================================================================

def random_rotation(img, max_degrees):
    """Rotate image by a random angle within [-max_degrees, max_degrees]."""
    angle = random.uniform(-max_degrees, max_degrees)
    return img.rotate(angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255))


def random_shift(img, max_fraction=0.1):
    """Translate image randomly by up to max_fraction of its size."""
    w, h = img.size
    dx = int(w * random.uniform(-max_fraction, max_fraction))
    dy = int(h * random.uniform(-max_fraction, max_fraction))
    # Use affine transform for translation
    return img.transform(
        img.size, Image.AFFINE, (1, 0, -dx, 0, 1, -dy),
        resample=Image.BICUBIC, fillcolor=(255, 255, 255)
    )


def random_zoom(img, zoom_range=(0.85, 1.15)):
    """Zoom in or out by cropping/padding, then resize back."""
    w, h = img.size
    factor = random.uniform(*zoom_range)

    # Calculate crop box for zoom
    new_w = int(w / factor)
    new_h = int(h / factor)
    left = (w - new_w) // 2
    top = (h - new_h) // 2

    if factor > 1.0:
        # Zoom in: crop center
        cropped = img.crop((left, top, left + new_w, top + new_h))
        return cropped.resize((w, h), Image.BICUBIC)
    else:
        # Zoom out: paste onto white background
        bg = Image.new('RGB', (new_w, new_h), (255, 255, 255))
        paste_x = (new_w - w) // 2
        paste_y = (new_h - h) // 2
        bg.paste(img, (paste_x, paste_y))
        return bg.resize((w, h), Image.BICUBIC)


def random_brightness(img, range_factor=(0.8, 1.2)):
    """Adjust brightness randomly within range."""
    factor = random.uniform(*range_factor)
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def random_horizontal_flip(img):
    """Flip horizontally with 50% probability."""
    if random.random() > 0.5:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def augment_spiral(img):
    """Apply spiral-specific augmentations. Full 360 rotation, NO shear."""
    img = random_rotation(img, 360)
    img = random_shift(img, 0.1)
    img = random_zoom(img, (0.85, 1.15))
    img = random_brightness(img, (0.8, 1.2))
    return img


def augment_wave(img):
    """Apply wave-specific augmentations. Limited rotation, horizontal flip, NO shear."""
    img = random_rotation(img, 15)
    img = random_horizontal_flip(img)
    img = random_shift(img, 0.1)
    img = random_zoom(img, (0.85, 1.15))
    img = random_brightness(img, (0.8, 1.2))
    return img


# ============================================================================
# Main
# ============================================================================

def process_directory(drawing_type, split, class_name, augment_fn):
    """Process a single directory: copy originals + generate augmented copies."""
    src_dir = os.path.join(ORIGINAL_ROOT, drawing_type, split, class_name)
    dst_dir = os.path.join(AUGMENTED_ROOT, drawing_type, split, class_name)
    os.makedirs(dst_dir, exist_ok=True)

    if not os.path.isdir(src_dir):
        print(f"  Skipping (not found): {src_dir}")
        return 0, 0

    image_files = [f for f in sorted(os.listdir(src_dir))
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    originals_copied = 0
    augmented_created = 0

    for fname in image_files:
        src_path = os.path.join(src_dir, fname)
        img = Image.open(src_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

        # Always copy the original
        base_name = os.path.splitext(fname)[0]
        ext = os.path.splitext(fname)[1]
        dst_path = os.path.join(dst_dir, fname)
        img.save(dst_path)
        originals_copied += 1

        # Only augment training images
        if split == 'training':
            for copy_idx in range(COPIES_PER_IMAGE):
                aug_img = augment_fn(img.copy())
                aug_fname = f"{base_name}_aug{copy_idx + 1}{ext}"
                aug_path = os.path.join(dst_dir, aug_fname)
                aug_img.save(aug_path)
                augmented_created += 1

    return originals_copied, augmented_created


def main():
    print("=" * 60)
    print("  TremorTrace — Offline Dataset Augmentation")
    print("=" * 60)
    print(f"\nSource:      {ORIGINAL_ROOT}")
    print(f"Destination: {AUGMENTED_ROOT}")
    print(f"Copies per training image: {COPIES_PER_IMAGE}")
    print()

    augment_fns = {
        'spiral': augment_spiral,
        'wave': augment_wave,
    }

    total_originals = 0
    total_augmented = 0

    print(f"{'Type':>8} | {'Split':>10} | {'Class':>10} | {'Originals':>10} | {'Augmented':>10}")
    print("-" * 65)

    for drawing_type in ['spiral', 'wave']:
        for split in ['training', 'testing']:
            for class_name in ['healthy', 'parkinson']:
                n_orig, n_aug = process_directory(
                    drawing_type, split, class_name, augment_fns[drawing_type]
                )
                total_originals += n_orig
                total_augmented += n_aug
                print(f"{drawing_type:>8} | {split:>10} | {class_name:>10} | {n_orig:>10} | {n_aug:>10}")

    total = total_originals + total_augmented
    print(f"\n{'=' * 65}")
    print(f"  Originals copied:  {total_originals}")
    print(f"  Augmented created: {total_augmented}")
    print(f"  Total images:      {total}")
    print(f"  Output: {AUGMENTED_ROOT}")
    print(f"{'=' * 65}")


if __name__ == '__main__':
    main()
