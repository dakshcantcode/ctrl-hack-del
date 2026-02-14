#!/usr/bin/env python3
"""
TremorTrace — Local Training Script

Trains both Spiral CNN and Wave CNN locally, evaluates them,
and exports production-ready files. Works on CPU or GPU.

Usage:
    cd "ctrlhackdel project/ml"
    python train_local.py

Output:
    outputs/           — trained models, plots, checkpoints
    exports/           — production files for backend integration
"""

import os
import sys
import time
import shutil
import json
import random
import base64
import warnings
from datetime import datetime

# Suppress TF warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# ============================================================================
# Path Setup
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, 'src')

# Add parent of src/ to path so "from src.xxx import yyy" works
sys.path.insert(0, SCRIPT_DIR)

DATA_ROOT = os.path.join(SRC_DIR, 'data', 'drawings_augmented')
ORIGINAL_DATA_ROOT = os.path.join(SRC_DIR, 'data', 'drawings')
SAVE_DIR = os.path.join(SCRIPT_DIR, 'outputs')
EXPORT_DIR = os.path.join(SCRIPT_DIR, 'exports')

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, 'plots'), exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# ============================================================================
# Imports (after path setup)
# ============================================================================
import numpy as np
import tensorflow as tf

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from src import config
from src.data_pipeline import create_generators
from src.model_builder import build_model
from src.trainer import train_two_phase
from src.evaluator import (
    evaluate_model,
    plot_training_history,
    show_sample_predictions,
    plot_prediction_distribution,
    plot_risk_tier_distribution,
    plot_model_comparison,
)
from src.gradcam import visualize_gradcam_grid
from src.ensemble import ensemble_predict

# Update config paths
config.DATA_ROOT = DATA_ROOT
config.PROJECT_DIR = SCRIPT_DIR


def print_header(text):
    """Prints a formatted section header."""
    print(f'\n{"=" * 60}')
    print(f'  {text}')
    print(f'{"=" * 60}\n')


def main():
    total_start = time.time()

    # ========================================================================
    # Step 1: Environment Check
    # ========================================================================
    print_header('Step 1: Environment Check')

    print(f'TensorFlow version: {tf.__version__}')
    print(f'Python: {sys.version.split()[0]}')
    print(f'Platform: {sys.platform}')

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f'GPU available: {gpus[0].name}')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print('No GPU detected — training on CPU (will be slower)')

    print(f'Data root: {DATA_ROOT}')
    print(f'Save dir:  {SAVE_DIR}')
    print(f'Export dir: {EXPORT_DIR}')

    # ========================================================================
    # Step 2: Verify Dataset
    # ========================================================================
    print_header('Step 2: Verify Dataset')

    PATHS = {}
    for drawing_type in ['spiral', 'wave']:
        PATHS[drawing_type] = {
            'train': os.path.join(DATA_ROOT, drawing_type, 'training'),
            'test': os.path.join(ORIGINAL_DATA_ROOT, drawing_type, 'testing'),
        }

    print(f'{"Type":>8} | {"Split":>8} | {"Healthy":>8} | {"Parkinson":>10} | {"Total":>6}')
    print('-' * 55)

    total_images = 0
    for dt in ['spiral', 'wave']:
        for split_name, split_path in PATHS[dt].items():
            healthy_dir = os.path.join(split_path, 'healthy')
            park_dir = os.path.join(split_path, 'parkinson')

            n_h = len([f for f in os.listdir(healthy_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            n_p = len([f for f in os.listdir(park_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            total_images += n_h + n_p
            print(f'{dt:>8} | {split_name:>8} | {n_h:>8} | {n_p:>10} | {n_h + n_p:>6}')

    print(f'\nTotal images: {total_images}')

    # ========================================================================
    # Step 3: Create Data Generators
    # ========================================================================
    print_header('Step 3: Create Data Generators')

    spiral_train_gen, spiral_val_gen, spiral_test_gen = create_generators(
        'spiral', PATHS['spiral']['train'], PATHS['spiral']['test'],
    )
    wave_train_gen, wave_val_gen, wave_test_gen = create_generators(
        'wave', PATHS['wave']['train'], PATHS['wave']['test'],
    )

    # ========================================================================
    # Step 4: Build Models
    # ========================================================================
    print_header('Step 4: Build Models')

    spiral_model, spiral_base = build_model('spiral')
    wave_model, wave_base = build_model('wave')

    # ========================================================================
    # Step 5: Train Spiral CNN
    # ========================================================================
    print_header('Step 5: Train Spiral CNN')

    spiral_history = train_two_phase(
        model=spiral_model,
        base_model=spiral_base,
        train_gen=spiral_train_gen,
        val_gen=spiral_val_gen,
        name='spiral',
        save_dir=SAVE_DIR,
    )

    # ========================================================================
    # Step 6: Train Wave CNN
    # ========================================================================
    print_header('Step 6: Train Wave CNN')

    wave_history = train_two_phase(
        model=wave_model,
        base_model=wave_base,
        train_gen=wave_train_gen,
        val_gen=wave_val_gen,
        name='wave',
        save_dir=SAVE_DIR,
    )

    # ========================================================================
    # Step 7: Plot Training History
    # ========================================================================
    print_header('Step 7: Training History Plots')

    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving plots

    plot_training_history(spiral_history, 'Spiral CNN', SAVE_DIR)
    plot_training_history(wave_history, 'Wave CNN', SAVE_DIR)
    print('Saved to outputs/plots/')

    # ========================================================================
    # Step 8: Evaluate Both Models
    # ========================================================================
    print_header('Step 8: Evaluate Both Models')

    from sklearn.metrics import precision_score, recall_score

    spiral_pred, spiral_true, spiral_acc, spiral_auc = evaluate_model(
        spiral_model, spiral_test_gen, 'Spiral CNN', SAVE_DIR,
    )
    plot_prediction_distribution(spiral_pred, spiral_true, 'Spiral CNN', SAVE_DIR)

    wave_pred, wave_true, wave_acc, wave_auc = evaluate_model(
        wave_model, wave_test_gen, 'Wave CNN', SAVE_DIR,
    )
    plot_prediction_distribution(wave_pred, wave_true, 'Wave CNN', SAVE_DIR)

    spiral_prec = precision_score(spiral_true, (spiral_pred > 0.5).astype(int), zero_division=0)
    spiral_rec = recall_score(spiral_true, (spiral_pred > 0.5).astype(int), zero_division=0)
    wave_prec = precision_score(wave_true, (wave_pred > 0.5).astype(int), zero_division=0)
    wave_rec = recall_score(wave_true, (wave_pred > 0.5).astype(int), zero_division=0)

    # ========================================================================
    # Step 9: Sample Predictions & Grad-CAM
    # ========================================================================
    print_header('Step 9: Sample Predictions & Grad-CAM')

    show_sample_predictions(spiral_model, spiral_test_gen, 'Spiral CNN', SAVE_DIR)
    show_sample_predictions(wave_model, wave_test_gen, 'Wave CNN', SAVE_DIR)

    visualize_gradcam_grid(spiral_model, spiral_test_gen, 'Spiral CNN', SAVE_DIR, n_samples=6)
    visualize_gradcam_grid(wave_model, wave_test_gen, 'Wave CNN', SAVE_DIR, n_samples=6)

    # ========================================================================
    # Step 10: Ensemble Test
    # ========================================================================
    print_header('Step 10: Ensemble Test')

    n_ensemble = min(len(spiral_pred), len(wave_pred))
    print(f'{"#":>3} | {"Spiral%":>8} | {"Wave%":>6} | {"Ensemble%":>9} | {"Risk Tier":>15} | {"Agree":>5}')
    print('-' * 60)

    ensemble_results = []
    for i in range(n_ensemble):
        result = ensemble_predict(float(spiral_pred[i]), float(wave_pred[i]), 'drawn')
        ensemble_results.append(result)
        print(f'{i+1:>3} | {result["spiral_cnn_percent"]:>7.1f}% | {result["wave_cnn_percent"]:>5.1f}% | '
              f'{result["pd_probability_percent"]:>8.1f}% | {result["risk_tier"]:>15} | '
              f'{"Yes" if result["unanimous"] else "No":>5}')

    # ========================================================================
    # Step 11: Risk Tier Distribution & Model Comparison
    # ========================================================================
    print_header('Step 11: Risk Tier Distribution & Model Comparison')

    plot_risk_tier_distribution(spiral_pred, spiral_true, 'Spiral CNN', SAVE_DIR)
    plot_risk_tier_distribution(wave_pred, wave_true, 'Wave CNN', SAVE_DIR)

    all_metrics = {
        'Spiral CNN': {'accuracy': spiral_acc, 'auc': spiral_auc,
                       'precision': spiral_prec, 'recall': spiral_rec},
        'Wave CNN': {'accuracy': wave_acc, 'auc': wave_auc,
                     'precision': wave_prec, 'recall': wave_rec},
    }
    plot_model_comparison(all_metrics, SAVE_DIR)

    # ========================================================================
    # Step 12: Test Input Handler End-to-End
    # ========================================================================
    print_header('Step 12: Test Input Handler (End-to-End)')

    from src.input_handler import process_input

    # Pick a test image and encode as base64
    spiral_test_dir = os.path.join(PATHS['spiral']['test'], 'parkinson')
    wave_test_dir = os.path.join(PATHS['wave']['test'], 'parkinson')
    spiral_img_path = os.path.join(spiral_test_dir, sorted(os.listdir(spiral_test_dir))[0])
    wave_img_path = os.path.join(wave_test_dir, sorted(os.listdir(wave_test_dir))[0])

    with open(spiral_img_path, 'rb') as f:
        spiral_b64 = base64.b64encode(f.read()).decode('utf-8')
    with open(wave_img_path, 'rb') as f:
        wave_b64 = base64.b64encode(f.read()).decode('utf-8')

    result = process_input(
        spiral_image_base64=spiral_b64,
        wave_image_base64=wave_b64,
        spiral_cnn_model=spiral_model,
        wave_cnn_model=wave_model,
        input_mode='drawn',
    )

    for key, value in result.items():
        if 'base64' in key:
            print(f'  {key}: [{"present" if value else "missing"}, {len(value) if value else 0} chars]')
        elif key == 'disclaimer':
            print(f'  {key}: "{value[:50]}..."')
        else:
            print(f'  {key}: {value}')

    # ========================================================================
    # Step 13: Export Production Files
    # ========================================================================
    print_header('Step 13: Export Production Files')

    # Copy models
    for name in ['spiral', 'wave']:
        src = os.path.join(SAVE_DIR, f'{name}_final.keras')
        dst = os.path.join(EXPORT_DIR, f'{name}_final.keras')
        if os.path.exists(src):
            shutil.copy2(src, dst)
            size_mb = os.path.getsize(dst) / (1024 * 1024)
            print(f'  {name}_final.keras ({size_mb:.1f} MB)')

    # Copy backend files
    for fname in ['ensemble.py', 'gradcam.py', 'input_handler.py']:
        src = os.path.join(SRC_DIR, fname)
        dst = os.path.join(EXPORT_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f'  {fname}')

    # Save metadata
    metadata = {
        'project': 'TremorTrace',
        'version': '1.0.0',
        'exported_at': datetime.now().isoformat(),
        'trained_on': 'local',
        'models': {
            'spiral_cnn': {
                'file': 'spiral_final.keras',
                'accuracy': round(spiral_acc, 4),
                'auc': round(spiral_auc, 4),
                'precision': round(spiral_prec, 4),
                'recall': round(spiral_rec, 4),
            },
            'wave_cnn': {
                'file': 'wave_final.keras',
                'accuracy': round(wave_acc, 4),
                'auc': round(wave_auc, 4),
                'precision': round(wave_prec, 4),
                'recall': round(wave_rec, 4),
            },
        },
        'ensemble': {
            'weights': {'spiral_cnn': 0.5, 'wave_cnn': 0.5},
            'output': 'probability_percentage_0_to_100',
        },
        'entry_point': 'input_handler.process_input()',
    }
    with open(os.path.join(EXPORT_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    print('  metadata.json')

    # ========================================================================
    # Final Summary
    # ========================================================================
    total_time = time.time() - total_start
    print_header('TRAINING COMPLETE')

    print(f'  Spiral CNN:  Accuracy={spiral_acc:.1%}, AUC={spiral_auc:.4f}')
    print(f'  Wave CNN:    Accuracy={wave_acc:.1%}, AUC={wave_auc:.4f}')
    print(f'  Total time:  {total_time / 60:.1f} minutes')
    print(f'  Models:      {SAVE_DIR}/')
    print(f'  Plots:       {os.path.join(SAVE_DIR, "plots")}/')
    print(f'  Exports:     {EXPORT_DIR}/')
    print(f'\n  Next: integrate exports/ into your backend')
    print(f'  Backend calls: input_handler.process_input()')


if __name__ == '__main__':
    main()
