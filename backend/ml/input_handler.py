"""
TremorTrace ML Pipeline — Input Handler

Main entry point for the backend. Handles both input modes (drawn and uploaded)
identically — only the image matters, not how it was created.

SELF-CONTAINED: This file uses only absolute imports and will be copied
directly into the backend alongside ensemble.py and gradcam.py.

The backend only needs to call process_input() — everything else is handled
internally. See exports/README.md for integration instructions.

Processing pipeline:
    base64 PNG → decode → resize 224×224 → RGB → normalize [0,1]
    → detect/fix color inversion → CNN predict → ensemble → Grad-CAM → result dict
"""

import io
import base64
import numpy as np
from PIL import Image

# Import from sibling files within the ml/ package
from ml.ensemble import ensemble_predict
from ml.gradcam import generate_gradcam, image_array_to_base64

# Image size must match the training configuration
_IMG_SIZE = 224


def detect_and_fix_inversion(image_array):
    """
    Detects and corrects color-inverted images (dark background).

    HTML5 canvas drawings typically have a dark/black background with light
    strokes, but the training data has dark strokes on white backgrounds.
    This function detects the inversion by checking the mean pixel intensity
    and flips it if needed.

    Args:
        image_array: np.ndarray, shape (224, 224, 3), float32, values [0, 1]

    Returns:
        np.ndarray: corrected image array, same shape and dtype.
            If mean pixel < 0.5 (dark background), the image is inverted
            so that it becomes dark-on-white (matching training data).
            Otherwise, returned unchanged.
    """
    mean_pixel = np.mean(image_array)
    if mean_pixel < 0.5:
        # Dark background detected — invert to match training data
        # (dark lines on white background)
        image_array = 1.0 - image_array
    return image_array


def preprocess_image_from_base64(base64_str):
    """
    Decodes a base64-encoded image and preprocesses it for CNN inference.

    Handles both raw base64 strings and data URI strings
    (e.g. "data:image/png;base64,iVBOR...").

    Processing steps:
        1. Strip data URI prefix if present
        2. Decode base64 → raw bytes
        3. Open as PIL Image
        4. Convert to RGB (handles RGBA, grayscale, palette, etc.)
        5. Resize to 224×224 using Lanczos resampling
        6. Convert to float32 numpy array in [0, 1] range
        7. Detect and fix color inversion

    Args:
        base64_str: str, base64-encoded image data.
            May include a data URI prefix like "data:image/png;base64,"

    Returns:
        np.ndarray: preprocessed image, shape (224, 224, 3), float32, values [0, 1]

    Raises:
        ValueError: if the base64 string is empty or cannot be decoded
        Exception: if the image cannot be opened or processed
    """
    if not base64_str or not isinstance(base64_str, str):
        raise ValueError("base64_str must be a non-empty string")

    # Strip data URI prefix if present (e.g. "data:image/png;base64,")
    if 'base64,' in base64_str:
        base64_str = base64_str.split('base64,', 1)[1]

    # Decode base64 to bytes
    try:
        raw_bytes = base64.b64decode(base64_str)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 string: {e}")

    # Open as PIL Image
    pil_image = Image.open(io.BytesIO(raw_bytes))

    # Convert to RGB (handles RGBA transparency, grayscale, palette modes)
    pil_image = pil_image.convert('RGB')

    # Resize to model input size using high-quality resampling
    pil_image = pil_image.resize((_IMG_SIZE, _IMG_SIZE), Image.LANCZOS)

    # Convert to float32 numpy array normalized to [0, 1]
    image_array = np.array(pil_image, dtype=np.float32) / 255.0

    # Fix color inversion if image has dark background
    image_array = detect_and_fix_inversion(image_array)

    return image_array


def process_input(
    spiral_image_base64,
    wave_image_base64,
    spiral_cnn_model=None,
    wave_cnn_model=None,
    input_mode='drawn',
):
    """
    Main entry point for TremorTrace prediction.

    Both input modes (drawn and uploaded) are processed identically — only
    images are used. The input_mode parameter is recorded in the output
    for metadata/logging only; it has NO effect on processing.

    This is the ONLY function the backend needs to call. It handles:
        1. Image preprocessing (decode, resize, normalize, inversion fix)
        2. CNN inference (both spiral and wave models)
        3. Ensemble combination (weighted average → probability percentage)
        4. Grad-CAM generation (explainability heatmaps as base64)
        5. Result assembly (full output dictionary)

    Args:
        spiral_image_base64: str, base64-encoded PNG of the spiral drawing.
            Can include a data URI prefix.
        wave_image_base64: str, base64-encoded PNG of the wave drawing.
            Can include a data URI prefix.
        spiral_cnn_model: loaded TensorFlow/Keras model for spiral classification.
            Must accept input shape (None, 224, 224, 3).
        wave_cnn_model: loaded TensorFlow/Keras model for wave classification.
            Must accept input shape (None, 224, 224, 3).
        input_mode: str, either 'drawn' or 'uploaded'.
            Metadata only — no processing difference between modes.

    Returns:
        dict: Complete prediction result containing:
            - All fields from ensemble_predict() (see ensemble.py)
            - spiral_gradcam_base64 (str): Grad-CAM overlay for spiral as data URI
            - wave_gradcam_base64 (str): Grad-CAM overlay for wave as data URI

        On error, returns:
            {"error": str, "error_type": str}
    """
    try:
        # Validate inputs
        if not spiral_image_base64:
            raise ValueError("spiral_image_base64 is required")
        if not wave_image_base64:
            raise ValueError("wave_image_base64 is required")
        if spiral_cnn_model is None:
            raise ValueError("spiral_cnn_model is required")
        if wave_cnn_model is None:
            raise ValueError("wave_cnn_model is required")

        # 1. Preprocess both images
        spiral_img = preprocess_image_from_base64(spiral_image_base64)
        wave_img = preprocess_image_from_base64(wave_image_base64)

        # 2. Run CNN inference
        # Add batch dimension: (224,224,3) → (1,224,224,3)
        spiral_prob = float(
            spiral_cnn_model.predict(spiral_img[np.newaxis, ...], verbose=0)[0][0]
        )
        wave_prob = float(
            wave_cnn_model.predict(wave_img[np.newaxis, ...], verbose=0)[0][0]
        )

        # 3. Ensemble prediction
        result = ensemble_predict(spiral_prob, wave_prob, input_mode)

        # 4. Generate Grad-CAM heatmaps for explainability
        try:
            _, spiral_overlay = generate_gradcam(spiral_cnn_model, spiral_img)
            result['spiral_gradcam_base64'] = image_array_to_base64(spiral_overlay)
        except Exception:
            result['spiral_gradcam_base64'] = None

        try:
            _, wave_overlay = generate_gradcam(wave_cnn_model, wave_img)
            result['wave_gradcam_base64'] = image_array_to_base64(wave_overlay)
        except Exception:
            result['wave_gradcam_base64'] = None

        return result

    except Exception as e:
        return {
            "error": str(e),
            "error_type": type(e).__name__,
        }
