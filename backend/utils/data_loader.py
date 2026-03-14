"""
LipSync — Dataset Loading & Batching

Builds a tf.data.Dataset pipeline that loads processed lip ROI frame sequences
and their text alignment labels, applies data augmentation, and batches/shuffles
for training.

Usage (as library):
    from backend.utils.data_loader import create_dataset
    train_ds, val_ds, num_train, num_val = create_dataset("backend/data/processed")

Usage (CLI — preview dataset):
    python -m backend.utils.data_loader --data_dir backend/data/processed
"""

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

logger = logging.getLogger("lipsync.data_loader")

# ---------------------------------------------------------------------------
# Constants (must match train.py and lip_extractor.py)
# ---------------------------------------------------------------------------
MAX_FRAMES = 75
IMG_HEIGHT = 50
IMG_WIDTH = 100
IMG_CHANNELS = 1
MAX_LABEL_LEN = 40

VOCAB = list("abcdefghijklmnopqrstuvwxyz0123456789 ")
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(VOCAB)}  # 0 = CTC blank


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
def encode_label(text: str) -> np.ndarray:
    """Encode a text label to a fixed-length integer array."""
    encoded = [CHAR_TO_IDX.get(c, 0) for c in text.lower()]
    encoded = encoded[:MAX_LABEL_LEN]
    padded = np.zeros(MAX_LABEL_LEN, dtype=np.int32)
    padded[: len(encoded)] = encoded
    return padded


# ---------------------------------------------------------------------------
# Data Discovery
# ---------------------------------------------------------------------------
def discover_samples(data_dir: str | Path) -> list[tuple[str, str]]:
    """
    Discover all (frames_path, label_path) pairs in the processed data directory.

    Expected naming convention:
        <utterance_id>_frames.npy
        <utterance_id>_label.txt

    Returns:
        List of (frames_npy_path, label_txt_path) tuples.
    """
    data_dir = Path(data_dir)
    samples = []

    for frames_path in sorted(data_dir.rglob("*_frames.npy")):
        utterance_id = frames_path.stem.replace("_frames", "")
        label_path = frames_path.parent / f"{utterance_id}_label.txt"
        if label_path.exists():
            samples.append((str(frames_path), str(label_path)))
        else:
            logger.warning("Missing label for %s", frames_path)

    logger.info("Discovered %d samples in %s", len(samples), data_dir)
    return samples


# ---------------------------------------------------------------------------
# Data Augmentation (applied in tf.data map)
# ---------------------------------------------------------------------------
def augment_frames(frames: tf.Tensor) -> tf.Tensor:
    """
    Apply data augmentation to a lip ROI frame sequence.

    Augmentations:
        - Random horizontal flip (50% chance)
        - Random brightness jitter (±10%)
        - Random slight rotation (±5°) — approximated with crop/pad

    Args:
        frames: Tensor of shape (MAX_FRAMES, IMG_HEIGHT, IMG_WIDTH, 1).

    Returns:
        Augmented tensor of the same shape.
    """
    # Random horizontal flip
    if tf.random.uniform(()) > 0.5:
        frames = tf.reverse(frames, axis=[2])  # flip width axis

    # Random brightness jitter
    delta = tf.random.uniform((), -0.1, 0.1)
    frames = frames + delta
    frames = tf.clip_by_value(frames, 0.0, 1.0)

    # Random contrast jitter
    factor = tf.random.uniform((), 0.8, 1.2)
    mean = tf.reduce_mean(frames)
    frames = (frames - mean) * factor + mean
    frames = tf.clip_by_value(frames, 0.0, 1.0)

    return frames


# ---------------------------------------------------------------------------
# tf.data.Dataset Construction
# ---------------------------------------------------------------------------
def _load_sample(frames_path: str, label_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a single sample from disk (used via tf.py_function)."""
    frames = np.load(frames_path)  # (MAX_FRAMES, H, W)
    frames = frames[..., np.newaxis]  # Add channel dim → (F, H, W, 1)
    frames = frames.astype(np.float32)

    label_text = open(label_path, "r").read().strip()
    label = encode_label(label_text)

    return frames, label


def _load_sample_tf(frames_path: tf.Tensor, label_path: tf.Tensor):
    """tf.py_function wrapper for _load_sample."""
    frames, label = tf.py_function(
        func=lambda fp, lp: _load_sample(fp.numpy().decode(), lp.numpy().decode()),
        inp=[frames_path, label_path],
        Tout=(tf.float32, tf.int32),
    )
    frames.set_shape((MAX_FRAMES, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    label.set_shape((MAX_LABEL_LEN,))
    return frames, label


def create_dataset(
    data_dir: str | Path,
    batch_size: int = 32,
    val_split: float = 0.2,
    augment: bool = True,
    seed: int = 42,
    buffer_size: int = 1000,
) -> tuple[tf.data.Dataset, tf.data.Dataset, int, int]:
    """
    Create training and validation tf.data.Datasets from processed data.

    The split is performed by **speaker** (speaker-independent evaluation):
    speakers are randomly divided into train/val sets, ensuring the model
    is evaluated on unseen speakers.

    Args:
        data_dir: Path to the processed data directory.
        batch_size: Training batch size.
        val_split: Fraction of speakers for validation.
        augment: Whether to apply data augmentation to training data.
        seed: Random seed for reproducibility.
        buffer_size: Shuffle buffer size.

    Returns:
        (train_dataset, val_dataset, num_train_samples, num_val_samples)
    """
    data_dir = Path(data_dir)
    all_samples = discover_samples(data_dir)

    if len(all_samples) == 0:
        raise ValueError(
            f"No samples found in {data_dir}. "
            "Run the lip extraction pipeline first: "
            "python -m backend.utils.lip_extractor"
        )

    # Group samples by speaker
    speaker_samples: dict[str, list[tuple[str, str]]] = {}
    for fp, lp in all_samples:
        # Speaker folder is the parent of the frames file
        speaker = Path(fp).parent.name
        speaker_samples.setdefault(speaker, []).append((fp, lp))

    speakers = sorted(speaker_samples.keys())
    random.seed(seed)
    random.shuffle(speakers)

    num_val_speakers = max(1, int(len(speakers) * val_split))
    val_speakers = set(speakers[:num_val_speakers])
    train_speakers = set(speakers[num_val_speakers:])

    logger.info(
        "Speaker split — Train: %s (%d speakers) | Val: %s (%d speakers)",
        train_speakers,
        len(train_speakers),
        val_speakers,
        len(val_speakers),
    )

    train_samples = [s for sp in train_speakers for s in speaker_samples[sp]]
    val_samples = [s for sp in val_speakers for s in speaker_samples[sp]]

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    def _build_dataset(samples, augment_data=False):
        frames_paths = [s[0] for s in samples]
        label_paths = [s[1] for s in samples]

        ds = tf.data.Dataset.from_tensor_slices((frames_paths, label_paths))
        ds = ds.map(_load_sample_tf, num_parallel_calls=tf.data.AUTOTUNE)

        if augment_data:
            ds = ds.map(
                lambda f, l: (augment_frames(f), l),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        ds = ds.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = _build_dataset(train_samples, augment_data=augment)
    val_ds = _build_dataset(val_samples, augment_data=False)

    return train_ds, val_ds, len(train_samples), len(val_samples)


# ---------------------------------------------------------------------------
# CLI — Dataset Preview
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Preview the LipSync data pipeline.")
    parser.add_argument("--data_dir", type=str, default="backend/data/processed")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    train_ds, val_ds, n_train, n_val = create_dataset(
        args.data_dir, batch_size=args.batch_size
    )

    print(f"\nTrain samples: {n_train} | Val samples: {n_val}")
    print("\nSample batch shapes:")
    for frames, labels in train_ds.take(1):
        print(f"  Frames: {frames.shape}")   # (batch, 75, 50, 100, 1)
        print(f"  Labels: {labels.shape}")    # (batch, 40)
        print(f"  Frame value range: [{frames.numpy().min():.3f}, {frames.numpy().max():.3f}]")

        # Decode first label
        label_arr = labels.numpy()[0]
        from backend.model.train import IDX_TO_CHAR
        decoded = "".join(IDX_TO_CHAR.get(i, "") for i in label_arr if i != 0)
        print(f"  First label decoded: '{decoded}'")


if __name__ == "__main__":
    main()

