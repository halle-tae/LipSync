"""
LipSync — Inference Module

Loads a trained lip reading model and provides prediction functionality
for both batch and real-time inference.
"""

import logging
from pathlib import Path

import numpy as np
import tensorflow as tf

from backend.model.train import (
    MAX_FRAMES,
    IMG_HEIGHT,
    IMG_WIDTH,
    IMG_CHANNELS,
    decode_predictions,
)

logger = logging.getLogger("lipsync.predict")

# Default path to the saved model weights
DEFAULT_MODEL_PATH = Path(__file__).parent / "lip_model.h5"


class LipReadingPredictor:
    """
    Wraps the trained lip reading model for inference.

    Usage:
        predictor = LipReadingPredictor("path/to/lip_model.h5")
        text, confidence = predictor.predict(frames)
    """

    def __init__(self, model_path: str | Path | None = None):
        """
        Load the trained model from disk.

        Args:
            model_path: Path to the saved Keras model (.h5).
                        Defaults to backend/model/lip_model.h5.

        Raises:
            FileNotFoundError: If the model weights file doesn't exist.
        """
        model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {model_path}. "
                "Train the model first with: python -m backend.model.train"
            )

        from backend.model.train import build_lip_reading_model, ctc_loss_fn

        self.model = build_lip_reading_model()
        self.model.compile(loss=ctc_loss_fn)
        self.model.load_weights(str(model_path))
        logger.info("Model loaded from %s", model_path)

    def preprocess(self, frames: np.ndarray) -> np.ndarray:
        """
        Preprocess a sequence of lip ROI frames for model input.

        Args:
            frames: Array of shape (num_frames, height, width) or
                    (num_frames, height, width, channels). Values in [0, 255].

        Returns:
            Preprocessed array of shape (1, MAX_FRAMES, IMG_HEIGHT, IMG_WIDTH, 1),
            normalized to [0, 1].
        """
        # Ensure 4D: (frames, h, w, c)
        if frames.ndim == 3:
            frames = frames[..., np.newaxis]

        # Normalize to [0, 1]
        if frames.max() > 1.0:
            frames = frames.astype(np.float32) / 255.0

        # Pad or truncate to MAX_FRAMES
        num_frames = frames.shape[0]
        if num_frames < MAX_FRAMES:
            pad = np.zeros(
                (MAX_FRAMES - num_frames, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                dtype=np.float32,
            )
            frames = np.concatenate([frames, pad], axis=0)
        elif num_frames > MAX_FRAMES:
            frames = frames[:MAX_FRAMES]

        # Add batch dimension
        return frames[np.newaxis, ...]

    def predict(self, frames: np.ndarray) -> tuple[str, float]:
        """
        Run inference on a sequence of lip ROI frames.

        Args:
            frames: Lip ROI frame sequence (see preprocess() for shapes).

        Returns:
            (predicted_text, confidence_score)
        """
        processed = self.preprocess(frames)
        raw_output = self.model.predict(processed, verbose=0)  # (1, T, classes)
        pred = raw_output[0]  # (T, classes)

        text = decode_predictions(pred)
        confidence = float(np.mean(np.max(pred, axis=-1)))

        return text, confidence

    def predict_batch(self, batch: list[np.ndarray]) -> list[tuple[str, float]]:
        """
        Run inference on a batch of frame sequences.

        Args:
            batch: List of lip ROI frame sequences.

        Returns:
            List of (predicted_text, confidence_score) tuples.
        """
        processed = np.concatenate(
            [self.preprocess(frames) for frames in batch], axis=0
        )
        raw_output = self.model.predict(processed, verbose=0)

        results = []
        for pred in raw_output:
            text = decode_predictions(pred)
            confidence = float(np.mean(np.max(pred, axis=-1)))
            results.append((text, confidence))
        return results

