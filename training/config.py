"""
Training configuration for TremorTrace spiral classifier.
"""
import os

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DRAWINGS_DIR = os.path.join(DATA_DIR, "drawings")
SPIRAL_DIR = os.path.join(DRAWINGS_DIR, "spiral")
WAVE_DIR = os.path.join(DRAWINGS_DIR, "wave")

TRAIN_HEALTHY_DIR = os.path.join(SPIRAL_DIR, "training", "healthy")
TRAIN_PARKINSON_DIR = os.path.join(SPIRAL_DIR, "training", "parkinson")
TEST_HEALTHY_DIR = os.path.join(SPIRAL_DIR, "testing", "healthy")
TEST_PARKINSON_DIR = os.path.join(SPIRAL_DIR, "testing", "parkinson")

WAVE_TRAIN_HEALTHY_DIR = os.path.join(WAVE_DIR, "training", "healthy")
WAVE_TRAIN_PARKINSON_DIR = os.path.join(WAVE_DIR, "training", "parkinson")
WAVE_TEST_HEALTHY_DIR = os.path.join(WAVE_DIR, "testing", "healthy")
WAVE_TEST_PARKINSON_DIR = os.path.join(WAVE_DIR, "testing", "parkinson")

USE_WAVE_DATA = True
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "saved_models")

# ── Image settings ─────────────────────────────────────────────
IMG_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)

# ── Training hyper-parameters ──────────────────────────────────
BATCH_SIZE = 8
EPOCHS_FROZEN = 60           # Phase 1: long head training (frozen base)
EPOCHS_FINETUNE = 40         # Phase 2: conservative fine-tune
LEARNING_RATE_FROZEN = 1e-3  # Higher LR ok since only head trains
LEARNING_RATE_FINETUNE = 1e-5
FINETUNE_LAYERS = 30         # Unfreeze last 30 MobileNetV2 layers

# steps_per_epoch computed dynamically in train.py as:
#   len(X_train) * 3 // BATCH_SIZE  (3x over-sampling via augmentation)

# ── Augmentation ───────────────────────────────────────────────
# NOTE: preprocessing_function is set in dataset.py (preprocess_input).
# Images are kept in [0,255] until preprocessing_function converts to [-1,1].
AUGMENTATION = dict(
    rotation_range=360,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
)
