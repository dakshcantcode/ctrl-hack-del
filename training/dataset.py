"""
Dataset loading and augmentation utilities.

CRITICAL FIX: Images are kept in [0, 255] range and preprocessed via
tf.keras.applications.mobilenet_v2.preprocess_input (→ [-1, 1]).
The previous version divided by 255 → [0, 1], completely mis-matching
MobileNetV2's expected input distribution and producing garbage features.
"""
import os
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from config import (
    TRAIN_HEALTHY_DIR, TRAIN_PARKINSON_DIR,
    TEST_HEALTHY_DIR, TEST_PARKINSON_DIR,
    WAVE_TRAIN_HEALTHY_DIR, WAVE_TRAIN_PARKINSON_DIR,
    WAVE_TEST_HEALTHY_DIR, WAVE_TEST_PARKINSON_DIR,
    USE_WAVE_DATA, IMG_SIZE, AUGMENTATION, BATCH_SIZE,
)


def _load_images_from_dir(directory: str, label: int):
    """Load all images from *directory* as grayscale → 3-channel [0, 255]."""
    images, labels = [], []
    if not os.path.isdir(directory):
        print(f"  [WARN] directory not found, skipping: {directory}")
        return images, labels

    for fname in sorted(os.listdir(directory)):
        fpath = os.path.join(directory, fname)
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue
        try:
            img = Image.open(fpath).convert("L")          # → grayscale
            img = ImageOps.autocontrast(img, cutoff=2)     # enhance contrast
            img = img.resize(IMG_SIZE, Image.LANCZOS)
            arr = np.array(img, dtype=np.float32)          # [0, 255] !!!
            arr = np.stack([arr] * 3, axis=-1)             # → (224,224,3)
            images.append(arr)
            labels.append(label)
        except Exception as exc:
            print(f"  [WARN] skipping {fpath}: {exc}")
    return images, labels


def _load_split(healthy_dir: str, parkinson_dir: str):
    """Load healthy + parkinson images from a split directory pair."""
    imgs_h, lbls_h = _load_images_from_dir(healthy_dir, label=0)
    imgs_p, lbls_p = _load_images_from_dir(parkinson_dir, label=1)
    print(f"    {len(imgs_h)} healthy + {len(imgs_p)} parkinson")
    return imgs_h + imgs_p, lbls_h + lbls_p


def load_dataset():
    """
    Returns (X_train, X_test, y_train, y_test) as numpy arrays.
    Images are in [0, 255] float32 — callers must apply preprocess_input.
    """
    print("Loading spiral training set…")
    train_imgs, train_lbls = _load_split(TRAIN_HEALTHY_DIR, TRAIN_PARKINSON_DIR)

    print("Loading spiral test set…")
    test_imgs, test_lbls = _load_split(TEST_HEALTHY_DIR, TEST_PARKINSON_DIR)

    if USE_WAVE_DATA:
        print("Loading wave training set…")
        w_train, w_lbls = _load_split(WAVE_TRAIN_HEALTHY_DIR, WAVE_TRAIN_PARKINSON_DIR)
        train_imgs.extend(w_train)
        train_lbls.extend(w_lbls)

        print("Loading wave test set…")
        w_test, w_tlbls = _load_split(WAVE_TEST_HEALTHY_DIR, WAVE_TEST_PARKINSON_DIR)
        test_imgs.extend(w_test)
        test_lbls.extend(w_tlbls)

    X_train = np.array(train_imgs, dtype=np.float32)
    y_train = np.array(train_lbls, dtype=np.float32)
    X_test = np.array(test_imgs, dtype=np.float32)
    y_test = np.array(test_lbls, dtype=np.float32)

    # Shuffle training data
    idx = np.random.RandomState(42).permutation(len(X_train))
    X_train, y_train = X_train[idx], y_train[idx]

    print(f"Total: {len(X_train)} train, {len(X_test)} test")
    n_pos = int(y_train.sum())
    print(f"  Train class balance: {len(y_train) - n_pos} healthy / {n_pos} parkinson")
    print(f"  Image value range: [{X_train.min():.0f}, {X_train.max():.0f}]")
    return X_train, X_test, y_train, y_test


def get_train_generator(X_train, y_train):
    """
    Return an augmented data generator for training.
    Augmentation operates on [0,255] images, then preprocess_input maps to [-1,1].
    """
    aug_config = dict(AUGMENTATION)
    aug_config["preprocessing_function"] = preprocess_input
    datagen = ImageDataGenerator(**aug_config)
    return datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)


def preprocess_validation(X):
    """Apply MobileNetV2 preprocessing to raw [0,255] validation images."""
    return preprocess_input(X.copy())
