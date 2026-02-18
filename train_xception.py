"""
Train Xception-based Deepfake Detection with Transfer Learning

Binary classification (Real vs Fake) using Xception pretrained on ImageNet.
Prevents data leakage via video-level GroupShuffleSplit.

Phase 1: Train classification head only (frozen base)
Phase 2: Fine-tune top ~30 layers of Xception with lower LR

Usage:
    python train_xception.py
    python train_xception.py --csv path/to/dataset.csv --batch_size 32
    python train_xception.py --head_epochs 5 --finetune_epochs 5
"""

import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# GPU memory growth (prevents TF from allocating all GPU memory at once)
# ---------------------------------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_SIZE = (299, 299)
AUTOTUNE = tf.data.AUTOTUNE

# Module-level augmentation layer (instantiated once)
_rotation_layer = layers.RandomRotation(0.02)  # +/- ~7 degrees


# ===================================================================
# CLI Arguments
# ===================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Xception for binary deepfake detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--csv', type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'dataset_Deepfakes_c23.csv'),
        help='Path to dataset CSV (must have image_path and label columns)')
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='Training batch size')
    parser.add_argument(
        '--head_epochs', type=int, default=10,
        help='Epochs for Phase 1 (train head only)')
    parser.add_argument(
        '--finetune_epochs', type=int, default=10,
        help='Epochs for Phase 2 (fine-tune top layers)')
    parser.add_argument(
        '--output_dir', type=str, default='xception_results',
        help='Output directory prefix (timestamp appended)')
    parser.add_argument(
        '--val_split', type=float, default=0.2,
        help='Validation set proportion')
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed')
    return parser.parse_args()


# ===================================================================
# Data Loading & Video-Level Split
# ===================================================================
def load_csv_and_split(csv_path, val_split=0.2, seed=42):
    """
    Load dataset CSV, extract video IDs, and split by video group.

    Parameters
    ----------
    csv_path : str
        Path to CSV with columns: image_path, label.
    val_split : float
        Fraction of videos held out for validation.
    seed : int
        Random state for reproducibility.

    Returns
    -------
    train_paths, train_labels, val_paths, val_labels : np.ndarray
    """
    df = pd.read_csv(csv_path, usecols=['image_path', 'label'])

    # Video ID = parent folder of each image file
    df['video_id'] = df['image_path'].apply(
        lambda p: os.path.basename(os.path.dirname(p))
    )

    paths = df['image_path'].values
    labels = df['label'].values
    groups = df['video_id'].values

    gss = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    train_idx, val_idx = next(gss.split(paths, labels, groups))

    # Verify no video leakage
    train_videos = set(groups[train_idx])
    val_videos = set(groups[val_idx])
    overlap = train_videos & val_videos
    assert len(overlap) == 0, (
        f"Data leakage detected: {len(overlap)} videos appear in both splits!"
    )

    print("=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    print(f"  CSV file:           {csv_path}")
    print(f"  Total samples:      {len(df):,}")
    print(f"  Unique videos:      {df['video_id'].nunique():,}")
    print(f"  Train samples:      {len(train_idx):,}  "
          f"({len(train_videos):,} videos)")
    print(f"  Val samples:        {len(val_idx):,}  "
          f"({len(val_videos):,} videos)")
    print(f"  Train label dist:   {np.bincount(labels[train_idx])}")
    print(f"  Val label dist:     {np.bincount(labels[val_idx])}")
    print(f"  Video overlap:      {len(overlap)} (must be 0)")
    print("=" * 80)

    return (paths[train_idx], labels[train_idx],
            paths[val_idx], labels[val_idx])


# ===================================================================
# tf.data.Dataset Pipeline
# ===================================================================
def _parse_image(path, label):
    """Load image from disk and resize to 299x299."""
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)  # float32 in [0, 255]
    return img, label


def _augment(img, label):
    """Apply spatial augmentation (training only)."""
    img = tf.image.random_flip_left_right(img)
    img = _rotation_layer(img, training=True)
    return img, label


def _finalize(img, label):
    """Apply Xception preprocessing (scales [0,255] -> [-1,1])."""
    img = preprocess_input(img)
    return img, label


def build_dataset(paths, labels, batch_size, training=False):
    """
    Create a tf.data.Dataset pipeline.

    Parameters
    ----------
    paths : np.ndarray
        Image file paths.
    labels : np.ndarray
        Binary labels (0 or 1).
    batch_size : int
        Batch size.
    training : bool
        If True, apply augmentation and shuffling.

    Returns
    -------
    tf.data.Dataset
        Batched and prefetched dataset yielding (image, label) pairs.
    """
    ds = tf.data.Dataset.from_tensor_slices(
        (paths.astype(str), labels.astype(np.float32))
    )

    if training:
        ds = ds.shuffle(buffer_size=min(len(paths), 10_000))

    ds = ds.map(_parse_image, num_parallel_calls=AUTOTUNE)

    if training:
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)

    ds = ds.map(_finalize, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds


# ===================================================================
# Model
# ===================================================================
def build_model():
    """
    Build Xception-based binary classifier.

    Architecture:
        Xception (ImageNet, frozen) -> GAP -> Dropout(0.5) -> Dense(1, sigmoid)

    Returns
    -------
    model : tf.keras.Model
    base_model : tf.keras.Model
        The Xception base (for later unfreezing).
    """
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(299, 299, 3)
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(299, 299, 3))
    # training=False keeps BatchNorm using ImageNet moving statistics
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs, name='xception_deepfake')
    return model, base_model


def compile_model(model, learning_rate):
    """Compile model with Adam, BinaryCrossentropy, Accuracy and AUC."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.AUC(name='auc'),
        ]
    )


# ===================================================================
# Callbacks
# ===================================================================
def get_callbacks(output_dir):
    """
    Create training callbacks.

    Returns
    -------
    callbacks : list
    best_weights_path : str
    """
    best_weights_path = os.path.join(output_dir, 'best_xception.keras')

    callbacks = [
        ModelCheckpoint(
            filepath=best_weights_path,
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
    ]
    return callbacks, best_weights_path


# ===================================================================
# Evaluation
# ===================================================================
def evaluate_model(model, val_ds, val_labels, output_dir):
    """
    Evaluate model on validation set. Print and save classification
    report, confusion matrix plot, and all metrics.
    """
    # Predictions (val_ds is not shuffled, so order matches val_labels)
    y_proba = model.predict(val_ds).ravel()
    y_pred = (y_proba >= 0.5).astype(int)
    y_true = val_labels

    # Keras metrics
    results = model.evaluate(val_ds, verbose=0)
    metric_names = model.metrics_names
    print()
    for name, value in zip(metric_names, results):
        print(f"  {name:12s}: {value:.4f}")

    # Calculate sklearn metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n  === Detailed Metrics ===")
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  Precision:   {precision:.4f}")
    print(f"  Recall:      {recall:.4f}")
    print(f"  F1 Score:    {f1:.4f}")

    # Classification report
    report = classification_report(y_true, y_pred,
                                   target_names=['Real', 'Fake'])
    print("\nClassification Report:")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"                Predicted Real  Predicted Fake")
    print(f"  Actual Real      {cm[0, 0]:>8,}         {cm[0, 1]:>8,}")
    print(f"  Actual Fake      {cm[1, 0]:>8,}         {cm[1, 1]:>8,}")

    # ── Plot and save confusion matrix ────────────────────────────
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix - Xception Deepfake Detection')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    cm_plot_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_plot_path, dpi=150)
    plt.close()
    print(f"\nConfusion matrix plot saved to: {cm_plot_path}")

    # ── Plot and save metrics bar chart ───────────────────────────
    plt.figure(figsize=(8, 5))
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metrics_values = [accuracy, precision, recall, f1]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    bars = plt.bar(metrics_names, metrics_values, color=colors, edgecolor='black')
    plt.ylim(0, 1.0)
    plt.ylabel('Score')
    plt.title('Model Performance Metrics')
    for bar, val in zip(bars, metrics_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    metrics_plot_path = os.path.join(output_dir, 'metrics_bar_chart.png')
    plt.savefig(metrics_plot_path, dpi=150)
    plt.close()
    print(f"Metrics bar chart saved to: {metrics_plot_path}")

    # ── Save detailed report to file ──────────────────────────────
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("XCEPTION DEEPFAKE DETECTION - EVALUATION REPORT\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write("KERAS METRICS:\n")
        for name, value in zip(metric_names, results):
            f.write(f"  {name}: {value:.4f}\n")
        f.write("\nDETAILED METRICS:\n")
        f.write(f"  Accuracy:    {accuracy:.4f}\n")
        f.write(f"  Precision:   {precision:.4f}\n")
        f.write(f"  Recall:      {recall:.4f}\n")
        f.write(f"  F1 Score:    {f1:.4f}\n")
        f.write(f"\nCLASSIFICATION REPORT:\n{report}\n")
        f.write(f"\nCONFUSION MATRIX:\n{cm}\n")
        f.write(f"\nFILES SAVED:\n")
        f.write(f"  - confusion_matrix.png\n")
        f.write(f"  - metrics_bar_chart.png\n")
        f.write(f"  - best_xception.keras\n")
    print(f"Evaluation report saved to: {report_path}")


# ===================================================================
# Training Orchestration
# ===================================================================
def train(args):
    """Execute the full two-phase training pipeline."""

    # Output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}\n")

    if not tf.config.list_physical_devices('GPU'):
        print("WARNING: No GPU detected. Training on CPU will be very slow.")
        print("Consider using Google Colab or a machine with a CUDA GPU.\n")

    # ── Load CSV and split by video ID ────────────────────────────
    train_paths, train_labels, val_paths, val_labels = load_csv_and_split(
        args.csv, args.val_split, args.seed
    )

    # ── Build tf.data pipelines ───────────────────────────────────
    train_ds = build_dataset(train_paths, train_labels,
                             args.batch_size, training=True)
    val_ds = build_dataset(val_paths, val_labels,
                           args.batch_size, training=False)

    # ── Build model ───────────────────────────────────────────────
    model, base_model = build_model()
    model.summary()

    best_weights_path = os.path.join(output_dir, 'best_xception.keras')

    # ==============================================================
    # PHASE 1: Train classification head (base frozen)
    # ==============================================================
    print("\n" + "=" * 80)
    print("PHASE 1: Training classification head (base frozen)")
    print("=" * 80 + "\n")

    compile_model(model, learning_rate=1e-3)
    callbacks_p1, _ = get_callbacks(output_dir)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.head_epochs,
        callbacks=callbacks_p1,
        verbose=1
    )

    # ==============================================================
    # PHASE 2: Fine-tune top layers of Xception
    # ==============================================================
    print("\n" + "=" * 80)
    print("PHASE 2: Fine-tuning top layers of Xception")
    print("=" * 80 + "\n")

    # Unfreeze the top ~30 layers
    base_model.trainable = True
    num_layers = len(base_model.layers)
    freeze_until = num_layers - 30
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    frozen_count = sum(1 for l in base_model.layers if not l.trainable)
    print(f"  Base model layers:  {num_layers}")
    print(f"  Frozen layers:      {frozen_count}")
    print(f"  Trainable layers:   {trainable_count}\n")

    # Re-compile with lower learning rate
    compile_model(model, learning_rate=1e-5)

    # Fresh callbacks (reset EarlyStopping patience)
    callbacks_p2, _ = get_callbacks(output_dir)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.finetune_epochs,
        callbacks=callbacks_p2,
        verbose=1
    )

    # ==============================================================
    # Final Evaluation
    # ==============================================================
    print("\n" + "=" * 80)
    print("FINAL EVALUATION ON VALIDATION SET")
    print("=" * 80)

    evaluate_model(model, val_ds, val_labels, output_dir)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print(f"  Best model saved to: {best_weights_path}")
    print("=" * 80 + "\n")


# ===================================================================
# Entry Point
# ===================================================================
def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
