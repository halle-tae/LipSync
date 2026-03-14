"""
LipSync — Model Training Script

Trains a spatiotemporal lip reading model (3D-CNN + BiLSTM) with CTC loss
on the GRID corpus dataset.  Tracks Word Error Rate (WER) and Character
Error Rate (CER) every epoch and logs everything to TensorBoard.

Usage:
    python -m backend.model.train --data_dir backend/data/processed --epochs 50
"""

import argparse
import logging
import os
import time
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
MAX_LABEL_LEN = 40       # Max encoded label length

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
        3 × (Conv3D → BatchNorm → ReLU → MaxPool3D) → Dropout
        → BiLSTM × 2 → Dense (CTC softmax)

    Architecture follows the LipNet paper (Assael et al., 2016) with
    adjustments for our ROI dimensions and vocabulary.

    Args:
        input_shape: (frames, height, width, channels)
        num_classes: number of output classes (alphabet + CTC blank)

    Returns:
        A Keras model ready for CTC training.
    """
    inputs = tf.keras.Input(shape=input_shape, name="lip_frames")

    # ---- Spatial Feature Extractor (3D CNN) ----
    # Block 1
    x = tf.keras.layers.Conv3D(
        32, (3, 5, 5), padding="same", activation=None,
        kernel_initializer="he_normal",
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool3D((1, 2, 2))(x)
    x = tf.keras.layers.SpatialDropout3D(0.25)(x)

    # Block 2
    x = tf.keras.layers.Conv3D(
        64, (3, 5, 5), padding="same", activation=None,
        kernel_initializer="he_normal",
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool3D((1, 2, 2))(x)
    x = tf.keras.layers.SpatialDropout3D(0.25)(x)

    # Block 3
    x = tf.keras.layers.Conv3D(
        96, (3, 3, 3), padding="same", activation=None,
        kernel_initializer="he_normal",
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool3D((1, 2, 2))(x)
    x = tf.keras.layers.SpatialDropout3D(0.25)(x)

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
    x = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="ctc_output"
    )(x)

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
def encode_text(text: str, max_label_len: int = MAX_LABEL_LEN) -> np.ndarray:
    """Convert a text string to a padded integer sequence."""
    encoded = [CHAR_TO_IDX.get(c, 0) for c in text.lower()]
    encoded = encoded[:max_label_len]
    padded = np.zeros(max_label_len, dtype=np.int32)
    padded[: len(encoded)] = encoded
    return padded


def decode_predictions(pred: np.ndarray) -> str:
    """CTC greedy-decode a model output to text.

    Args:
        pred: (time_steps, num_classes) softmax output.

    Returns:
        Decoded string after collapsing repeats and removing blanks.
    """
    argmaxes = np.argmax(pred, axis=-1)
    chars = []
    prev = -1
    for idx in argmaxes:
        if idx != prev and idx != 0:
            chars.append(IDX_TO_CHAR.get(idx, ""))
        prev = idx
    return "".join(chars)


def decode_label(label: np.ndarray) -> str:
    """Decode an integer label array back to text (inverse of encode_text)."""
    return "".join(IDX_TO_CHAR.get(int(i), "") for i in label if i != 0)


# ---------------------------------------------------------------------------
# WER / CER Computation
# ---------------------------------------------------------------------------
def _edit_distance(ref: list, hyp: list) -> int:
    """Compute Levenshtein edit distance between two sequences."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[n][m]


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate = edit_distance(ref_words, hyp_words) / len(ref_words)."""
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return _edit_distance(ref_words, hyp_words) / len(ref_words)


def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate = edit_distance(ref_chars, hyp_chars) / len(ref_chars)."""
    ref_chars = list(reference.strip())
    hyp_chars = list(hypothesis.strip())
    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0
    return _edit_distance(ref_chars, hyp_chars) / len(ref_chars)


# ---------------------------------------------------------------------------
# WER/CER Callback
# ---------------------------------------------------------------------------
class WERCERCallback(tf.keras.callbacks.Callback):
    """
    Evaluate WER and CER on a validation dataset at the end of each epoch.

    Decodes model predictions using CTC greedy decoding and compares them
    against ground-truth labels.  Results are printed and logged to
    TensorBoard.
    """

    def __init__(
        self,
        val_dataset: tf.data.Dataset,
        num_samples: int = 200,
        log_dir: str | None = None,
    ):
        super().__init__()
        self.val_dataset = val_dataset
        self.num_samples = num_samples
        self._writer = None
        if log_dir:
            self._writer = tf.summary.create_file_writer(
                os.path.join(log_dir, "metrics")
            )
        self.history: dict[str, list[float]] = {"wer": [], "cer": []}

    def on_epoch_end(self, epoch, logs=None):
        total_wer = 0.0
        total_cer = 0.0
        count = 0

        for frames_batch, labels_batch in self.val_dataset:
            preds = self.model.predict(frames_batch, verbose=0)
            for i in range(preds.shape[0]):
                if count >= self.num_samples:
                    break
                hyp = decode_predictions(preds[i])
                ref = decode_label(labels_batch[i].numpy())
                total_wer += compute_wer(ref, hyp)
                total_cer += compute_cer(ref, hyp)
                count += 1
            if count >= self.num_samples:
                break

        avg_wer = total_wer / max(count, 1)
        avg_cer = total_cer / max(count, 1)

        self.history["wer"].append(avg_wer)
        self.history["cer"].append(avg_cer)

        if logs is not None:
            logs["val_wer"] = avg_wer
            logs["val_cer"] = avg_cer

        logger.info(
            "Epoch %d — val_WER: %.4f | val_CER: %.4f  (evaluated on %d samples)",
            epoch + 1, avg_wer, avg_cer, count,
        )

        # Log to TensorBoard
        if self._writer is not None:
            with self._writer.as_default():
                tf.summary.scalar("val_wer", avg_wer, step=epoch)
                tf.summary.scalar("val_cer", avg_cer, step=epoch)
                self._writer.flush()

    # Show a few sample predictions at end of training
    def on_train_end(self, logs=None):
        logger.info("\n=== Sample Predictions (last epoch) ===")
        count = 0
        for frames_batch, labels_batch in self.val_dataset:
            preds = self.model.predict(frames_batch, verbose=0)
            for i in range(preds.shape[0]):
                if count >= 10:
                    return
                hyp = decode_predictions(preds[i])
                ref = decode_label(labels_batch[i].numpy())
                logger.info("  REF: %-30s  HYP: %s", ref, hyp)
                count += 1
            if count >= 10:
                break


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

    Tracks:
        - CTC loss (train & val)
        - Word Error Rate (WER) on validation set
        - Character Error Rate (CER) on validation set
        - Learning rate schedule (reduce on plateau)

    Saves:
        - Best model checkpoint (by val_loss) as lip_model.h5
        - Training history as training_history.npz
        - TensorBoard logs to <output_dir>/logs/

    Args:
        data_dir: Path to directory containing processed .npy frame sequences
                  and alignment .txt files.
        output_dir: Where to save model checkpoints.
        epochs: Maximum training epochs.
        batch_size: Training batch size.
        learning_rate: Initial learning rate.
        patience: Early stopping patience.
    """
    logger.info("=" * 60)
    logger.info("LipSync — Phase 2: Model Training")
    logger.info("=" * 60)
    logger.info("  data_dir      : %s", data_dir)
    logger.info("  output_dir    : %s", output_dir)
    logger.info("  epochs        : %d", epochs)
    logger.info("  batch_size    : %d", batch_size)
    logger.info("  learning_rate : %e", learning_rate)
    logger.info("  patience      : %d", patience)

    # ---- GPU Check ----
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logger.info("  GPU(s) found  : %s", [g.name for g in gpus])
        # Allow memory growth to avoid OOM
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        logger.warning("  No GPU found — training will be slow on CPU.")

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
    log_dir = os.path.join(output_dir, "logs")
    checkpoint_path = os.path.join(output_dir, "lip_model.h5")

    wer_cer_cb = WERCERCallback(
        val_dataset=val_ds,
        num_samples=min(200, num_val),
        log_dir=log_dir,
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
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
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq="epoch",
        ),
        wer_cer_cb,
    ]

    # ---- Train ----
    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )
    elapsed = time.time() - start_time

    # ---- Save training history ----
    history_path = os.path.join(output_dir, "training_history.npz")
    hist_data = {k: np.array(v) for k, v in history.history.items()}
    hist_data["wer"] = np.array(wer_cer_cb.history["wer"])
    hist_data["cer"] = np.array(wer_cer_cb.history["cer"])
    np.savez(history_path, **hist_data)

    logger.info("=" * 60)
    logger.info("Training complete in %.1f seconds (%.1f min).", elapsed, elapsed / 60)
    logger.info("Best model saved to %s", checkpoint_path)
    logger.info("History saved to %s", history_path)

    if wer_cer_cb.history["wer"]:
        best_wer = min(wer_cer_cb.history["wer"])
        best_cer = min(wer_cer_cb.history["cer"])
        logger.info("Best WER: %.4f | Best CER: %.4f", best_wer, best_cer)

    logger.info("=" * 60)
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
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
