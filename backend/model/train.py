"""
LipSync — Model Training Script

Trains a spatiotemporal lip reading model (3D-CNN + BiLSTM) with CTC loss
on the GRID corpus dataset.

Usage:
    python -m backend.model.train --data_dir backend/data/processed --epochs 50
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

logger = logging.getLogger("lipsync.train")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_HEIGHT = 50
IMG_WIDTH = 100
IMG_CHANNELS = 1
MAX_FRAMES = 75          # Max sequence length (frames)
VOCAB = list("abcdefghijklmnopqrstuvwxyz0123456789 ")
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(VOCAB)}  # 0 reserved for CTC blank
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(VOCAB)}
IDX_TO_CHAR[0] = ""  # CTC blank
NUM_CLASSES = len(VOCAB) + 1  # +1 for CTC blank


# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------
def build_lip_reading_model(
    input_shape: tuple = (MAX_FRAMES, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
    num_classes: int = NUM_CLASSES,
) -> tf.keras.Model:
    """
    Build the LipSync spatiotemporal model:
        3 × (Conv3D → BatchNorm → ReLU → MaxPool3D) → BiLSTM × 2 → Dense (CTC)

    Args:
        input_shape: (frames, height, width, channels)
        num_classes: number of output classes (alphabet + CTC blank)

    Returns:
        A compiled Keras model ready for CTC training.
    """
    inputs = tf.keras.Input(shape=input_shape, name="lip_frames")

    # ---- Spatial Feature Extractor (3D CNN) ----
    x = tf.keras.layers.Conv3D(32, (3, 5, 5), padding="same", activation=None)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool3D((1, 2, 2))(x)

    x = tf.keras.layers.Conv3D(64, (3, 5, 5), padding="same", activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool3D((1, 2, 2))(x)

    x = tf.keras.layers.Conv3D(96, (3, 3, 3), padding="same", activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool3D((1, 2, 2))(x)

    # Collapse spatial dims → (batch, frames, features)
    time_steps = x.shape[1]
    feature_dim = x.shape[2] * x.shape[3] * x.shape[4]
    x = tf.keras.layers.Reshape((time_steps, feature_dim))(x)

    # ---- Temporal Sequence Model (BiLSTM) ----
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.5)
    )(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.5)
    )(x)

    # ---- Output Dense ----
    x = tf.keras.layers.Dense(num_classes, activation="softmax", name="ctc_output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=x, name="LipSyncModel")
    return model


# ---------------------------------------------------------------------------
# CTC Loss
# ---------------------------------------------------------------------------
def ctc_loss_fn(y_true, y_pred):
    """Compute CTC loss compatible with Keras model.fit()."""
    batch_size = tf.shape(y_pred)[0]
    input_length = tf.fill([batch_size, 1], tf.shape(y_pred)[1])
    label_length = tf.math.count_nonzero(y_true, axis=1, keepdims=True)
    label_length = tf.cast(label_length, dtype=tf.int32)

    loss = tf.keras.backend.ctc_batch_cost(
        y_true, y_pred, input_length, label_length
    )
    return tf.reduce_mean(loss)


# ---------------------------------------------------------------------------
# Text ↔ Encoding Helpers
# ---------------------------------------------------------------------------
def encode_text(text: str, max_label_len: int = 40) -> np.ndarray:
    """Convert a text string to a padded integer sequence."""
    encoded = [CHAR_TO_IDX.get(c, 0) for c in text.lower()]
    encoded = encoded[:max_label_len]
    padded = np.zeros(max_label_len, dtype=np.int32)
    padded[: len(encoded)] = encoded
    return padded


def decode_predictions(pred: np.ndarray) -> str:
    """CTC greedy-decode a model output to text."""
    # pred shape: (time_steps, num_classes)
    argmaxes = np.argmax(pred, axis=-1)
    # Collapse repeats and remove blanks
    chars = []
    prev = -1
    for idx in argmaxes:
        if idx != prev and idx != 0:
            chars.append(IDX_TO_CHAR.get(idx, ""))
        prev = idx
    return "".join(chars)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def train(
    data_dir: str,
    output_dir: str = "backend/model",
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    patience: int = 10,
):
    """
    Train the LipSync model on processed GRID corpus data.

    Args:
        data_dir: Path to directory containing processed .npy frame sequences
                  and alignment .txt files.
        output_dir: Where to save model checkpoints.
        epochs: Maximum training epochs.
        batch_size: Training batch size.
        learning_rate: Initial learning rate.
        patience: Early stopping patience.
    """
    logger.info("Starting training — data_dir=%s, epochs=%d", data_dir, epochs)

    # ---- Load Data ----
    from backend.utils.data_loader import create_dataset

    train_ds, val_ds, num_train, num_val = create_dataset(
        data_dir=data_dir,
        batch_size=batch_size,
        val_split=0.2,
    )

    logger.info("Training samples: %d | Validation samples: %d", num_train, num_val)

    # ---- Build Model ----
    model = build_lip_reading_model()
    model.summary(print_fn=logger.info)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=ctc_loss_fn,
    )

    # ---- Callbacks ----
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "lip_model.h5")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(output_dir, "logs"),
            histogram_freq=1,
        ),
    ]

    # ---- Train ----
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    logger.info("Training complete. Best model saved to %s", checkpoint_path)
    return history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train the LipSync lip reading model.")
    parser.add_argument("--data_dir", type=str, default="backend/data/processed",
                        help="Path to processed dataset directory.")
    parser.add_argument("--output_dir", type=str, default="backend/model",
                        help="Directory for model checkpoints.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()

