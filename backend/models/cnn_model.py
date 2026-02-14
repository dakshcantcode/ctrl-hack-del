"""
Load and run the MobileNetV2 CNN for spiral classification.
All images are converted to grayscale then replicated to 3 channels.
"""
import os
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "saved_models", "spiral_cnn.h5")


def load_cnn_model():
    """Load the trained Keras model and warm it up."""
    import tensorflow as tf

    if not os.path.exists(MODEL_PATH):
        print(f"[WARN] Model file not found at {MODEL_PATH} — predictions will use a dummy model")
        return _build_dummy_model()

    model = tf.keras.models.load_model(MODEL_PATH)
    # Warm-up
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    model.predict(dummy, verbose=0)
    return model


def _build_dummy_model():
    """Return a randomly-initialised model so the server can start without weights."""
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
    from tensorflow.keras.models import Model

    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base.trainable = False
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    model.predict(dummy, verbose=0)
    return model


def predict_cnn(model, image_array: np.ndarray) -> float:
    """
    Run a single prediction.
    image_array: (224, 224, 3) float32 in [-1, 1] (after preprocess_input).
    Returns probability of parkinson (0-1).
    """
    inp = np.expand_dims(image_array, axis=0)
    prob = float(model.predict(inp, verbose=0)[0][0])
    return prob
