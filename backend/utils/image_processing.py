"""
Image processing utilities.
Core rule: everything is converted to **grayscale first**, then
replicated to 3 channels so MobileNetV2 can consume it.
"""
import base64
import io
import numpy as np
from PIL import Image


def decode_base64_image(data_url: str) -> Image.Image:
    """Decode a data-URL (or raw base64) string → PIL Image."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    return Image.open(io.BytesIO(raw))


def preprocess_for_model(pil_image: Image.Image, size: tuple = (224, 224)) -> np.ndarray:
    """
    Convert to grayscale → resize → replicate to 3 channels → preprocess_input.
    Returns (H, W, 3) float32 array in [-1, 1] (MobileNetV2 expected range).
    """
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from PIL import ImageOps

    gray = pil_image.convert("L")                # grayscale
    gray = ImageOps.autocontrast(gray, cutoff=2)
    gray = gray.resize(size, Image.LANCZOS)
    arr = np.array(gray, dtype=np.float32)        # [0, 255]
    arr_3ch = np.stack([arr] * 3, axis=-1)        # (224, 224, 3)
    arr_3ch = preprocess_input(arr_3ch)           # → [-1, 1]
    return arr_3ch


def image_to_base64(image) -> str:
    """Convert a numpy array or PIL Image to a data-URL base64 string."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(np.uint8(image))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"
