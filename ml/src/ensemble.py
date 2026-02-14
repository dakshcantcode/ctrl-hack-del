"""
TremorTrace ML Pipeline — Ensemble

2-model weighted ensemble that combines Spiral CNN and Wave CNN predictions
into a single probability percentage (0–100%) with risk tier classification.

SELF-CONTAINED: This file uses only standard library imports and will be
copied directly into the backend. No TensorFlow, no relative imports.

Output is ALWAYS a percentage with a risk tier — never a binary label.
Both input modes (drawn / uploaded) are processed identically; the
input_mode field is metadata only.
"""

# =============================================================================
# Constants (duplicated from config.py for self-containment)
# =============================================================================

ENSEMBLE_WEIGHTS = {
    'spiral_cnn': 0.4,
    'wave_cnn': 0.6,
}

RISK_TIERS = [
    (0, 25, "Low Risk", "#27AE60"),
    (25, 45, "Mild Risk", "#F1C40F"),
    (45, 65, "Moderate Risk", "#E67E22"),
    (65, 85, "Elevated Risk", "#E74C3C"),
    (85, 100, "High Risk", "#C0392B"),
]

CONFIDENCE_LABELS = [
    (0.0, 0.3, "Low"),
    (0.3, 0.6, "Moderate"),
    (0.6, 1.0, "High"),
]

DISCLAIMER = (
    "This is a screening tool only. It does not constitute a medical diagnosis. "
    "Please consult a healthcare professional for proper evaluation."
)


def get_risk_tier(probability_percent):
    """
    Determines the risk tier for a given PD probability percentage.

    Risk tiers provide a human-readable interpretation of the raw probability:
        0–25%   → Low Risk (green)
        25–45%  → Mild Risk (yellow)
        45–65%  → Moderate Risk (orange)
        65–85%  → Elevated Risk (red-orange)
        85–100% → High Risk (red)

    Args:
        probability_percent: float, value between 0.0 and 100.0

    Returns:
        tuple: (tier_label, hex_color)
            - tier_label: str, e.g. "Elevated Risk"
            - hex_color: str, e.g. "#E74C3C"
    """
    for lower, upper, label, color in RISK_TIERS:
        if lower <= probability_percent < upper:
            return label, color
    # Edge case: exactly 100.0% falls into the last tier
    return RISK_TIERS[-1][2], RISK_TIERS[-1][3]


def get_confidence_label(confidence_score):
    """
    Converts a numeric confidence score to a human-readable label.

    Confidence score = abs(probability - 0.5) * 2, scaled to [0, 1].
    A score near 0 means the model is uncertain (prediction near 50%),
    while a score near 1 means the model is very confident.

    Args:
        confidence_score: float, value between 0.0 and 1.0

    Returns:
        str: "Low", "Moderate", or "High"
    """
    for lower, upper, label in CONFIDENCE_LABELS:
        if lower <= confidence_score < upper:
            return label
    # Edge case: exactly 1.0
    return CONFIDENCE_LABELS[-1][2]


def ensemble_predict(spiral_cnn_prob, wave_cnn_prob, input_mode='drawn'):
    """
    Combines predictions from Spiral CNN and Wave CNN into a single
    probability-based result with risk tier classification.

    Ensemble formula:
        final_prob = 0.4 * spiral_cnn_prob + 0.6 * wave_cnn_prob

    Wave CNN is weighted higher due to stronger balanced performance.
    The input_mode parameter is recorded for metadata/logging only —
    it does NOT affect the prediction in any way.

    Args:
        spiral_cnn_prob: float, Spiral CNN's raw sigmoid output (0.0–1.0).
            Represents P(Parkinson's) from the spiral drawing analysis.
        wave_cnn_prob: float, Wave CNN's raw sigmoid output (0.0–1.0).
            Represents P(Parkinson's) from the wave drawing analysis.
        input_mode: str, either 'drawn' or 'uploaded'.
            Metadata only — no processing difference.

    Returns:
        dict: Complete prediction result containing:
            - pd_probability_percent (float): Primary output, 0.0–100.0
            - risk_tier (str): Human-readable risk level
            - risk_color (str): Hex color for UI display
            - spiral_cnn_percent (float): Spiral model's prediction as percentage
            - wave_cnn_percent (float): Wave model's prediction as percentage
            - input_mode (str): 'drawn' or 'uploaded'
            - weights_used (dict): Ensemble weight configuration
            - model_agreement (float): 1.0 if both agree, 0.5 if split
            - unanimous (bool): Whether both models agree on class
            - confidence_score (float): 0.0–1.0, distance from decision boundary
            - confidence_label (str): 'Low', 'Moderate', or 'High'
            - disclaimer (str): Medical disclaimer text
    """
    # Weighted ensemble combination
    w_spiral = ENSEMBLE_WEIGHTS['spiral_cnn']
    w_wave = ENSEMBLE_WEIGHTS['wave_cnn']
    final_prob = w_spiral * spiral_cnn_prob + w_wave * wave_cnn_prob

    # Convert to percentage
    pd_probability_percent = round(final_prob * 100.0, 2)

    # Risk tier classification
    risk_tier, risk_color = get_risk_tier(pd_probability_percent)

    # Confidence: how far the ensemble prediction is from the 50% decision boundary
    # 0.0 = maximally uncertain (exactly 50%), 1.0 = maximally confident (0% or 100%)
    confidence_score = round(abs(final_prob - 0.5) * 2.0, 4)
    confidence_label = get_confidence_label(confidence_score)

    # Model agreement: do both models agree on which side of 50% the prediction falls?
    spiral_vote = 1 if spiral_cnn_prob >= 0.5 else 0
    wave_vote = 1 if wave_cnn_prob >= 0.5 else 0
    unanimous = (spiral_vote == wave_vote)
    model_agreement = 1.0 if unanimous else 0.5

    return {
        # Primary output
        "pd_probability_percent": pd_probability_percent,
        "risk_tier": risk_tier,
        "risk_color": risk_color,

        # Per-model breakdown (percentages)
        "spiral_cnn_percent": round(spiral_cnn_prob * 100.0, 2),
        "wave_cnn_percent": round(wave_cnn_prob * 100.0, 2),

        # Ensemble metadata
        "input_mode": input_mode,
        "weights_used": {
            "spiral_cnn": w_spiral,
            "wave_cnn": w_wave,
        },
        "model_agreement": model_agreement,
        "unanimous": unanimous,

        # Confidence & interpretation
        "confidence_score": confidence_score,
        "confidence_label": confidence_label,
        "disclaimer": DISCLAIMER,
    }
