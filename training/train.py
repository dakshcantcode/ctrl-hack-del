"""
Main training script for TremorTrace spiral classifier.

Usage:
    python train.py

Dataset layout (Kaggle "Parkinson's Drawings"):
    training/data/drawings/spiral/{training,testing}/{healthy,parkinson}
    training/data/drawings/wave/{training,testing}/{healthy,parkinson}

Outputs saved_models/spiral_cnn.h5  (also copied to ../backend/saved_models/)
"""
import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    MODEL_OUTPUT_DIR, EPOCHS_FROZEN, EPOCHS_FINETUNE,
    LEARNING_RATE_FROZEN, LEARNING_RATE_FINETUNE,
    FINETUNE_LAYERS, BATCH_SIZE,
)
from dataset import load_dataset, get_train_generator, preprocess_validation
from model import build_model, compile_frozen, unfreeze_top_layers, compile_finetune


def main():
    tf.random.set_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("TremorTrace — MobileNetV2 Spiral Classifier Training")
    print("=" * 60)

    # ── Load data (images in [0, 255]) ─────────────────────────
    X_train, X_test, y_train, y_test = load_dataset()

    # Preprocess validation/test data for direct use in evaluate()
    X_test_pp = preprocess_validation(X_test)

    # Compute steps_per_epoch: 3x oversampling via augmentation
    steps_per_epoch = max(20, len(X_train) * 3 // BATCH_SIZE)
    print(f"Steps per epoch: {steps_per_epoch} "
          f"(~{steps_per_epoch * BATCH_SIZE} augmented samples/epoch)")

    # Training generator (augmentation + preprocess_input built in)
    train_gen = get_train_generator(X_train, y_train)

    # Verify generator output range
    sample_batch = next(train_gen)
    print(f"Generator output range: [{sample_batch[0].min():.2f}, {sample_batch[0].max():.2f}]"
          f" (should be ~[-1, 1])")

    # Class weights — bias toward Parkinson to improve recall
    class_weight = {0: 0.8, 1: 1.3}
    print(f"Class weights: {class_weight}")

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_OUTPUT_DIR, "spiral_cnn.h5")

    # ── Phase 1: Frozen base ──────────────────────────────────
    print(f"\n── Phase 1: Training head ({EPOCHS_FROZEN} epochs) ──")
    model = build_model()
    model = compile_frozen(model, lr=LEARNING_RATE_FROZEN)

    trainable = sum(1 for v in model.trainable_variables)
    total_params = model.count_params()
    print(f"Trainable params: {sum(v.numpy().size for v in model.trainable_variables)}"
          f" / {total_params} total")

    callbacks_p1 = [
        EarlyStopping(
            monitor="val_auc", patience=15, mode="max",
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_auc", factor=0.5, patience=5,
            min_lr=1e-6, mode="max", verbose=1,
        ),
        ModelCheckpoint(
            model_path, monitor="val_auc", mode="max",
            save_best_only=True, verbose=1,
        ),
    ]

    model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS_FROZEN,
        validation_data=(X_test_pp, y_test),
        class_weight=class_weight,
        callbacks=callbacks_p1,
    )

    # Quick eval after Phase 1
    p1_preds = model.predict(X_test_pp, verbose=0).flatten()
    p1_acc = ((p1_preds >= 0.5).astype(int) == y_test.astype(int)).mean()
    print(f"\n  Phase 1 best — Accuracy: {p1_acc:.3f}, "
          f"Pred range: [{p1_preds.min():.3f}, {p1_preds.max():.3f}]")

    # ── Phase 2: Fine-tuning ──────────────────────────────────
    print(f"\n── Phase 2: Fine-tuning last {FINETUNE_LAYERS} layers "
          f"({EPOCHS_FINETUNE} epochs) ──")
    model = unfreeze_top_layers(model, num_layers_to_unfreeze=FINETUNE_LAYERS)
    model = compile_finetune(model, lr=LEARNING_RATE_FINETUNE)

    trainable_ft = sum(1 for v in model.trainable_variables)
    print(f"Trainable variable groups: {trainable_ft}")

    callbacks_p2 = [
        EarlyStopping(
            monitor="val_auc", patience=10, mode="max",
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_auc", factor=0.5, patience=4,
            min_lr=1e-7, mode="max", verbose=1,
        ),
        ModelCheckpoint(
            model_path, monitor="val_auc", mode="max",
            save_best_only=True, verbose=1,
        ),
    ]

    model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS_FINETUNE,
        validation_data=(X_test_pp, y_test),
        class_weight=class_weight,
        callbacks=callbacks_p2,
    )

    # ── Final Evaluation ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)
    model = tf.keras.models.load_model(model_path)
    results = model.evaluate(X_test_pp, y_test, verbose=1)
    for name, val in zip(model.metrics_names, results):
        print(f"  {name:>12s}: {val:.4f}")

    preds = model.predict(X_test_pp, verbose=0).flatten()
    pred_labels = (preds >= 0.5).astype(int)
    true_labels = y_test.astype(int)
    correct = (pred_labels == true_labels).sum()

    print(f"\n  Correct: {correct}/{len(y_test)}")
    print(f"  Predictions range: [{preds.min():.3f}, {preds.max():.3f}]")
    print(f"\nClassification Report:")
    print(classification_report(true_labels, pred_labels,
                                target_names=["Healthy", "Parkinson"]))
    print("Confusion Matrix:")
    print(confusion_matrix(true_labels, pred_labels))

    # ── Copy to backend ───────────────────────────────────────
    backend_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "backend", "saved_models",
    )
    os.makedirs(backend_dir, exist_ok=True)
    dest = os.path.join(backend_dir, "spiral_cnn.h5")
    shutil.copy2(model_path, dest)
    print(f"\nModel copied to {dest}")
    print("Done ✓")


if __name__ == "__main__":
    main()
    main()
