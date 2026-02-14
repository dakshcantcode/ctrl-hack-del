"""
TremorTrace ML Pipeline — Grad-CAM

Gradient-weighted Class Activation Mapping for visual explainability.
Shows which regions of the input drawing the CNN focused on when making
its prediction, helping users understand what the model "sees."

SELF-CONTAINED: This file uses only absolute imports and will be copied
directly into the backend. No relative imports from src/.

Implementation uses tf.GradientTape for gradient computation, targeting
the last convolutional activation layer of MobileNetV2 ('out_relu').
"""

import os
import io
import base64
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import matplotlib.pyplot as plt


def find_target_layer(model):
    """
    Finds the appropriate convolutional layer for Grad-CAM visualization.

    For MobileNetV2, targets the 'out_relu' layer (the final ReLU activation
    after the last convolutional block). Falls back to the last Conv2D layer
    if 'out_relu' is not found.

    Args:
        model: Keras Model, the full model (base + head)

    Returns:
        str: name of the target layer

    Raises:
        ValueError: if no suitable convolutional layer is found
    """
    # First priority: MobileNetV2's final activation layer
    for layer in reversed(model.layers):
        if layer.name == 'out_relu':
            return layer.name

    # Fallback: find the last Conv2D layer
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

    raise ValueError(
        "Could not find a suitable convolutional layer for Grad-CAM. "
        "Expected 'out_relu' (MobileNetV2) or any Conv2D layer."
    )


def generate_gradcam(model, image_array, layer_name=None):
    """
    Generates a Grad-CAM heatmap and overlay for a single image.

    Algorithm:
    1. Forward pass through a gradient model that outputs both the target
       conv layer activations and the final prediction
    2. Compute gradients of the prediction w.r.t. conv layer output
    3. Global average pool gradients to get per-channel importance weights
    4. Compute weighted sum of conv output channels
    5. Apply ReLU and normalize to [0, 1]
    6. Resize, colorize with JET colormap, and blend with original image

    Args:
        model: Keras Model, the trained classification model
        image_array: np.ndarray, shape (224, 224, 3), float32, values [0, 1]
        layer_name: str or None, name of target conv layer.
            If None, auto-detected via find_target_layer().

    Returns:
        tuple: (heatmap, overlay)
            - heatmap: np.ndarray, shape (224, 224), float32, values [0, 1]
            - overlay: np.ndarray, shape (224, 224, 3), uint8, RGB
    """
    if layer_name is None:
        layer_name = find_target_layer(model)

    # Build a model that outputs both conv activations and predictions
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(layer_name).output,
            model.output,
        ],
    )

    # Prepare input tensor with batch dimension
    input_tensor = tf.cast(image_array[np.newaxis, ...], tf.float32)

    # Forward pass with gradient recording
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensor)
        # For binary classification with sigmoid, the output is P(parkinson)
        loss = predictions[:, 0]

    # Gradients of the prediction w.r.t. conv layer output
    grads = tape.gradient(loss, conv_outputs)

    # Global average pool: per-channel importance weights
    # grads shape: (1, H, W, C) → pooled_grads shape: (C,)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Remove batch dimension from conv outputs
    conv_outputs = conv_outputs[0]  # shape: (H, W, C)

    # Weighted combination of channels
    # (H, W, C) @ (C, 1) → (H, W, 1) → squeeze to (H, W)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU: only keep positive activations (regions that positively
    # contribute to the predicted class)
    heatmap = tf.maximum(heatmap, 0)

    # Normalize to [0, 1] range
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    heatmap = heatmap.numpy()

    # Resize heatmap to match input image dimensions
    img_size = image_array.shape[0]  # 224
    heatmap_resized = cv2.resize(heatmap, (img_size, img_size))

    # Apply JET colormap (produces BGR)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Convert original image to uint8 for blending
    original_uint8 = np.uint8(image_array * 255.0)
    # Convert RGB to BGR for cv2 blending (original is RGB from PIL/TF)
    original_bgr = cv2.cvtColor(original_uint8, cv2.COLOR_RGB2BGR)

    # Blend: 60% original + 40% heatmap
    overlay_bgr = cv2.addWeighted(original_bgr, 0.6, heatmap_colored, 0.4, 0)

    # Convert back to RGB for display/storage
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    return heatmap_resized, overlay_rgb


def visualize_gradcam_grid(model, test_gen, name, save_dir, n_samples=8):
    """
    Creates and saves a grid visualization of Grad-CAM results.

    Each row shows: Original Image | Heatmap | Overlay
    for a test sample.

    Args:
        model: Keras Model, the trained classification model
        test_gen: Keras data generator, test data (shuffle=False)
        name: str, model display name (e.g. 'Spiral CNN')
        save_dir: str, directory to save the output grid PNG
        n_samples: int, number of samples to visualize (default 8)
    """
    plot_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    safe_name = name.replace(' ', '_')

    # Collect samples from the test generator
    test_gen.reset()
    images = []
    labels = []
    for batch_imgs, batch_labels in test_gen:
        for img, lbl in zip(batch_imgs, batch_labels):
            images.append(img)
            labels.append(lbl)
            if len(images) >= n_samples:
                break
        if len(images) >= n_samples:
            break

    n_actual = len(images)
    if n_actual == 0:
        print(f"⚠️  No test images found for Grad-CAM visualization of {name}")
        return

    # Auto-detect target layer once
    layer_name = find_target_layer(model)

    # Create figure: n_actual rows × 3 columns
    fig, axes = plt.subplots(n_actual, 3, figsize=(12, 4 * n_actual))
    fig.suptitle(f'{name} — Grad-CAM Visualization', fontsize=18, fontweight='bold')

    if n_actual == 1:
        axes = axes[np.newaxis, :]  # Ensure 2D indexing works for single row

    col_titles = ['Original', 'Heatmap', 'Overlay']
    class_names = {0: 'Healthy', 1: "Parkinson's"}

    for i in range(n_actual):
        image = images[i]
        label = int(labels[i])

        # Generate Grad-CAM
        heatmap, overlay = generate_gradcam(model, image, layer_name)

        # Original image
        axes[i, 0].imshow(image)
        axes[i, 0].set_ylabel(
            f'{class_names[label]}',
            fontsize=11, fontweight='bold', rotation=0, labelpad=70,
            va='center',
        )

        # Heatmap (use 'jet' colormap for display)
        axes[i, 1].imshow(heatmap, cmap='jet', vmin=0, vmax=1)

        # Overlay
        axes[i, 2].imshow(overlay)

        # Set column titles on first row only
        if i == 0:
            for j, title in enumerate(col_titles):
                axes[i, j].set_title(title, fontsize=14, fontweight='bold')

        # Remove tick marks from all subplots
        for j in range(3):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    plt.tight_layout()
    plt.savefig(
        os.path.join(plot_dir, f'{safe_name}_gradcam_grid.png'),
        dpi=150, bbox_inches='tight',
    )
    plt.show()
    print(f"✅ {name} Grad-CAM grid saved ({n_actual} samples)")


def image_array_to_base64(img_array):
    """
    Converts a numpy image array to a base64-encoded PNG data URI string.

    This is used to embed Grad-CAM visualizations in API responses so the
    frontend can display them directly in <img> tags.

    Args:
        img_array: np.ndarray, image data. Can be:
            - float32 with values [0, 1] → will be scaled to [0, 255]
            - uint8 with values [0, 255] → used directly

    Returns:
        str: data URI string in format "data:image/png;base64,{encoded_data}"
    """
    # Convert float images to uint8
    if img_array.dtype == np.float32 or img_array.dtype == np.float64:
        img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)

    # Create PIL Image
    pil_image = Image.fromarray(img_array)

    # Save to bytes buffer as PNG
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)

    # Encode to base64
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return f"data:image/png;base64,{encoded}"
