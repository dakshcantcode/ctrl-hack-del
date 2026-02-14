"""
TremorTrace ML Pipeline — Configuration

Single source of truth for ALL hyperparameters, paths, and constants.
No magic numbers should exist anywhere else in the codebase.
"""

# =============================================================================
# Image Settings
# =============================================================================
IMG_SIZE = 224
BATCH_SIZE = 8
RANDOM_SEED = 42

# =============================================================================
# Phase 1: Frozen Base Training (train head only)
# =============================================================================
PHASE1_EPOCHS = 30
PHASE1_LR = 5e-4
PHASE1_PATIENCE = 10

# =============================================================================
# Phase 2: Fine-Tuning (unfreeze top layers)
# =============================================================================
PHASE2_EPOCHS = 30
PHASE2_LR = 2e-5
PHASE2_PATIENCE = 8
UNFREEZE_LAYERS = 40  # Unfreeze last N layers of MobileNetV2

# =============================================================================
# Ensemble Weights (same for both input modes — no difference between drawn/uploaded)
# =============================================================================
ENSEMBLE_WEIGHTS = {
    'spiral_cnn': 0.4,
    'wave_cnn': 0.6,
}

# =============================================================================
# Risk Tier Definitions
# (lower_bound, upper_bound, label, hex_color)
# =============================================================================
RISK_TIERS = [
    (0, 25, "Low Risk", "#27AE60"),
    (25, 45, "Mild Risk", "#F1C40F"),
    (45, 65, "Moderate Risk", "#E67E22"),
    (65, 85, "Elevated Risk", "#E74C3C"),
    (85, 100, "High Risk", "#C0392B"),
]

# =============================================================================
# Confidence Labels
# (lower_bound, upper_bound, label)
# =============================================================================
CONFIDENCE_LABELS = [
    (0.0, 0.3, "Low"),
    (0.3, 0.6, "Moderate"),
    (0.6, 1.0, "High"),
]

# =============================================================================
# Medical Disclaimer (MUST be shown to users)
# =============================================================================
DISCLAIMER = (
    "This is a screening tool only. It does not constitute a medical diagnosis. "
    "Please consult a healthcare professional for proper evaluation."
)

# =============================================================================
# Paths (set dynamically in notebook based on Colab/local environment)
# =============================================================================
DATA_ROOT = None
PROJECT_DIR = None
