"""
TremorTrace ML Pipeline — Model Builder

Constructs MobileNetV2-based transfer learning models for binary classification
of Parkinson's disease indicators in spiral and wave drawings.
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    BatchNormalization,
    Dense,
    Dropout,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, Precision, Recall

try:
    from src.config import IMG_SIZE, PHASE1_LR
except ImportError:
    from config import IMG_SIZE, PHASE1_LR


def build_model(name='spiral'):
    """
    Builds a MobileNetV2 transfer learning model for binary classification.

    Architecture:
        MobileNetV2 (pretrained, frozen) → GlobalAveragePooling2D →
        BatchNormalization → Dense(128, relu) → Dropout(0.5) →
        Dense(64, relu) → Dropout(0.3) → Dense(1, sigmoid)

    The base model is returned separately so the trainer can selectively
    unfreeze layers during fine-tuning (Phase 2).

    Args:
        name: str, model identifier (e.g. 'spiral', 'wave'). Used for naming layers.

    Returns:
        tuple: (model, base_model)
            - model: compiled Keras Model ready for training
            - base_model: the MobileNetV2 base for freeze/unfreeze control
    """
    # Load pretrained MobileNetV2 without classification head
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )

    # Freeze all base model layers for Phase 1 (train head only)
    for layer in base_model.layers:
        layer.trainable = False

    # Build classification head
    x = base_model.output
    x = GlobalAveragePooling2D(name=f'{name}_gap')(x)
    x = BatchNormalization(name=f'{name}_bn')(x)
    x = Dense(128, activation='relu', name=f'{name}_dense_128')(x)
    x = Dropout(0.5, name=f'{name}_dropout_05')(x)
    x = Dense(64, activation='relu', name=f'{name}_dense_64')(x)
    x = Dropout(0.3, name=f'{name}_dropout_03')(x)
    output = Dense(1, activation='sigmoid', name=f'{name}_output')(x)

    # Create full model
    model = tf.keras.Model(inputs=base_model.input, outputs=output, name=f'{name}_cnn')

    # Compile with initial learning rate
    model.compile(
        optimizer=Adam(learning_rate=PHASE1_LR),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            AUC(name='auc'),
            Precision(name='precision'),
            Recall(name='recall'),
        ],
    )

    # Print summary
    total_params = model.count_params()
    trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    non_trainable_params = total_params - trainable_params

    print(f"✅ {name.capitalize()} CNN built:")
    print(f"   Total params:         {total_params:,}")
    print(f"   Trainable params:     {trainable_params:,}")
    print(f"   Non-trainable params: {non_trainable_params:,}")

    return model, base_model
