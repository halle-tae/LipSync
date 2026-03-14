"""
LipSync — Inference Module

Loads a trained lip reading model and provides prediction functionality
for both batch and real-time inference.  Supports both Keras (.h5) and
TFLite (.tflite) models for flexible deployment.

Usage:
    from backend.model.predict import LipReadingPredictor
    predictor = LipReadingPredictor("backend/model/lip_model.h5")
    text, confidence = predictor.predict(frames)
"""

import logging
import time
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

# Default paths
DEFAULT_MODEL_PATH = Path(__file__).parent / "lip_model.h5"
DEFAULT_TFLITE_PATH = Path(__file__).parent / "lip_model.tflite"


class LipReadingPredictor:
    """
    Wraps the trained lip reading model for inference.

    Supports two backends:
        - Keras (.h5) — full model, GPU-compatible
        - TFLite (.tflite) — optimized for CPU inference

    The backend is selected automatically based on the file extension,
    or can be forced with the `use_tflite` parameter.

    Usage:
        predictor = LipReadingPredictor("path/to/lip_model.h5")
        text, confidence = predictor.predict(frames)
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        use_tflite: bool | None = None,
    ):
        """
        Load the trained model from disk.

        Args:
            model_path: Path to the saved model (.h5 or .tflite).
                        If None, tries TFLite first, then Keras.
            use_tflite: Force TFLite (True) or Keras (False) backend.
                        If None, auto-detect from file extension.

        Raises:
            FileNotFoundError: If no model weights file can be found.
        """
        self._tflite = False
        self._interpreter = None
        self._model = None

        # Auto-detect model path
        if model_path is None:
            if DEFAULT_TFLITE_PATH.exists():
                model_path = DEFAULT_TFLITE_PATH
            elif DEFAULT_MODEL_PATH.exists():
                model_path = DEFAULT_MODEL_PATH
            else:
                raise FileNotFoundError(
                    f"No model found. Checked:\n"
                    f"  {DEFAULT_TFLITE_PATH}\n"
                    f"  {DEFAULT_MODEL_PATH}\n"
                    "Train the model first with: python -m backend.model.train"
                )

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Train the model first with: python -m backend.model.train"
            )

        # Determine backend
        if use_tflite is True or (use_tflite is None and model_path.suffix == ".tflite"):
            self._load_tflite(model_path)
        else:
            self._load_keras(model_path)

    def _load_keras(self, model_path: Path):
        """Load a Keras model."""
        from backend.model.train import build_lip_reading_model, ctc_loss_fn

        self._model = build_lip_reading_model()
        self._model.compile(loss=ctc_loss_fn)
        self._model.load_weights(str(model_path))
        self._tflite = False
        logger.info("Keras model loaded from %s (%d params)",
                     model_path, self._model.count_params())

    def _load_tflite(self, model_path: Path):
        """Load a TFLite model."""
        self._interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        self._tflite = True
        logger.info("TFLite model loaded from %s (%.1f MB)",
                     model_path, model_path.stat().st_size / (1024 * 1024))

    @property
    def backend(self) -> str:
        """Return the active backend name."""
        return "tflite" if self._tflite else "keras"

    def preprocess(self, frames: np.ndarray) -> np.ndarray:
        """
        Preprocess a sequence of lip ROI frames for model input.

        Args:
            frames: Array of shape (num_frames, height, width) or
                    (num_frames, height, width, channels). Values in [0, 255] or [0, 1].

        Returns:
            Preprocessed array of shape (1, MAX_FRAMES, IMG_HEIGHT, IMG_WIDTH, 1),
            normalized to [0, 1].
        """
        if frames.ndim == 3:
            frames = frames[..., np.newaxis]

        if frames.max() > 1.0:
            frames = frames.astype(np.float32) / 255.0
        else:
            frames = frames.astype(np.float32)

        num_frames = frames.shape[0]
        if num_frames < MAX_FRAMES:
            pad = np.zeros(
                (MAX_FRAMES - num_frames, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                dtype=np.float32,
            )
            frames = np.concatenate([frames, pad], axis=0)
        elif num_frames > MAX_FRAMES:
            frames = frames[:MAX_FRAMES]

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

        if self._tflite:
            self._interpreter.set_tensor(self._input_details[0]["index"], processed)
            self._interpreter.invoke()
            raw_output = self._interpreter.get_tensor(self._output_details[0]["index"])
        else:
            raw_output = self._model.predict(processed, verbose=0)

        pred = raw_output[0]  # (T, classes)
        text = decode_predictions(pred)
        confidence = float(np.mean(np.max(pred, axis=-1)))

        return text, confidence

    def predict_timed(self, frames: np.ndarray) -> tuple[str, float, float]:
        """
        Run inference with timing.

        Returns:
            (predicted_text, confidence_score, inference_time_ms)
        """
        t0 = time.perf_counter()
        text, confidence = self.predict(frames)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return text, confidence, elapsed_ms

    def predict_batch(self, batch: list[np.ndarray]) -> list[tuple[str, float]]:
        """
        Run inference on a batch of frame sequences.

        Note: TFLite backend processes one sample at a time (no batch).
              Keras backend uses native batching.

        Args:
            batch: List of lip ROI frame sequences.

        Returns:
            List of (predicted_text, confidence_score) tuples.
        """
        if self._tflite:
            return [self.predict(frames) for frames in batch]

        processed = np.concatenate(
            [self.preprocess(frames) for frames in batch], axis=0
        )
        raw_output = self._model.predict(processed, verbose=0)

        results = []
        for pred in raw_output:
            text = decode_predictions(pred)
            confidence = float(np.mean(np.max(pred, axis=-1)))
            results.append((text, confidence))
        return results
