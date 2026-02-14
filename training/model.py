"""
MobileNetV2 transfer-learning model for spiral classification.

Architecture: MobileNetV2 (frozen) → GAP → Dropout → Dense(1, sigmoid)
Deliberately simple head to avoid overfitting on ~144 training images.
"""
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

from config import INPUT_SHAPE


def build_model() -> Model:
    """Build MobileNetV2 with a minimal binary classification head."""
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=INPUT_SHAPE,
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model


def _get_metrics():
    return [
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]


def compile_frozen(model: Model, lr: float = 1e-3):
    """Compile for Phase 1 (frozen base, train head only)."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=_get_metrics(),
    )
    return model


def unfreeze_top_layers(model: Model, num_layers_to_unfreeze: int = 30):
    """Unfreeze the top-N layers of the base for fine-tuning (Phase 2)."""
    base = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base = layer
            break

    if base is None:
        for layer in model.layers[-num_layers_to_unfreeze:]:
            layer.trainable = True
    else:
        base.trainable = True
        for layer in base.layers[:-num_layers_to_unfreeze]:
            layer.trainable = False

    return model


def compile_finetune(model: Model, lr: float = 1e-5):
    """Re-compile for Phase 2 (fine-tuning) with lower LR."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=_get_metrics(),
    )
    return model
