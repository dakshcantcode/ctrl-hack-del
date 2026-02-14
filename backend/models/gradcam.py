"""
Grad-CAM visualisation for the MobileNetV2 classifier.
Highlights the regions of the spiral that most influenced the prediction.
"""
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2


def generate_gradcam(model, image_array: np.ndarray, layer_name: str | None = None):
    """
    Generate a Grad-CAM heatmap for the given image.

    Parameters
    ----------
    model : tf.keras.Model
    image_array : ndarray  (224, 224, 3) float32 in [0,1]
    layer_name : optional target conv layer; auto-detected if None

    Returns
    -------
    heatmap : ndarray (224, 224) float32 in [0,1]
    overlay : ndarray (224, 224, 3) uint8  — coloured overlay on original image
    """
    if layer_name is None:
        # Find last Conv2D layer
        for layer in reversed(model.layers):
            if "conv" in layer.name.lower():
                try:
                    shape = layer.output.shape
                    if len(shape) == 4:
                        layer_name = layer.name
                        break
                except Exception:
                    continue
        # Fall-through: deepest layer of base model
        if layer_name is None:
            for layer in reversed(model.layers):
                if hasattr(layer, "layers"):
                    for sub in reversed(layer.layers):
                        if "conv" in sub.name.lower():
                            layer_name = sub.name
                            break
                    if layer_name:
                        break

    if layer_name is None:
        # Return a blank heatmap if we can't find a conv layer
        blank = np.zeros((224, 224), dtype=np.float32)
        overlay = np.uint8(np.clip((image_array + 1) * 127.5, 0, 255))
        return blank, overlay

    try:
        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[model.get_layer(layer_name).output, model.output],
        )
    except ValueError:
        # Layer may be nested inside base model
        base = None
        for layer in model.layers:
            if hasattr(layer, "layers"):
                base = layer
                break
        if base is None:
            blank = np.zeros((224, 224), dtype=np.float32)
            overlay = (image_array * 255).astype(np.uint8)
            return blank, overlay

        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[base.get_layer(layer_name).output, model.output],
        )

    inp = tf.convert_to_tensor(image_array[np.newaxis], dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(inp)
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_out)
    if grads is None:
        blank = np.zeros((224, 224), dtype=np.float32)
        overlay = np.uint8(np.clip((image_array + 1) * 127.5, 0, 255))
        return blank, overlay

    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.reduce_sum(weights * conv_out, axis=-1).numpy()[0]
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    # Resize to image size
    heatmap = cv2.resize(cam, (224, 224))

    # Coloured overlay
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    # image_array is in [-1, 1] (MobileNetV2 preprocess_input output)
    original_uint8 = np.uint8(np.clip((image_array + 1) * 127.5, 0, 255))
    overlay = cv2.addWeighted(original_uint8, 0.6, heatmap_color, 0.4, 0)

    return heatmap, overlay
