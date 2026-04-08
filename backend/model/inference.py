"""
Inference Pipeline
==================
End-to-end video-only speech recognition pipeline using Auto-AVSR.

This module handles:
1. Loading a video file
2. Detecting faces and cropping mouth ROI (using Auto-AVSR's built-in pipeline)
3. Preprocessing frames (grayscale, normalize, center-crop to 88x88)
4. Running model inference (beam search decoding)
5. Returning predicted text

Two modes:
- `from_video_file()`: Process a complete video file (Phase 0)
- `from_frames()`: Process a tensor of preprocessed mouth ROI frames (Phase 1 — webcam)
"""

import logging
import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
AUTOAVSR_DIR = PROJECT_ROOT / "auto_avsr"


def _ensure_autoavsr_on_path():
    """Add Auto-AVSR repo to sys.path."""
    autoavsr_path = str(AUTOAVSR_DIR)
    if autoavsr_path not in sys.path:
        sys.path.insert(0, autoavsr_path)


class InferencePipeline:
    """
    Video-only speech recognition inference pipeline.

    Wraps the Auto-AVSR model with face detection and mouth cropping.

    Usage:
        from backend.model.inference import InferencePipeline

        pipeline = InferencePipeline()
        pipeline.load()

        # From video file
        result = pipeline.predict_from_file("path/to/video.mp4")
        print(result["text"])
        print(result["latency_ms"])

        # From preprocessed frames (for real-time webcam pipeline)
        result = pipeline.predict_from_frames(mouth_roi_tensor)
        print(result["text"])
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        detector: str = "mediapipe",
    ):
        """
        Args:
            weights_path: Path to model weights. Uses default if None.
            device: 'cuda', 'mps', or 'cpu'. Auto-detected if None.
            detector: Face detector to use ('mediapipe' or 'retinaface').
        """
        from backend.model.loader import ModelLoader

        self.loader = ModelLoader(weights_path=weights_path, device=device)
        self.detector = detector
        self.device = self.loader.device

        # These are initialized on load()
        self.model_module = None
        self.landmarks_detector = None
        self.video_process = None
        self.video_transform = None

    def load(self):
        """Load model, face detector, and video preprocessor."""
        _ensure_autoavsr_on_path()

        # Load the model
        logger.info("Loading Auto-AVSR model...")
        self.model_module = self.loader.load()

        # Initialize face detector and video processor
        logger.info(f"Initializing face detector ({self.detector})...")
        if self.detector == "mediapipe":
            from preparation.detectors.mediapipe.detector import LandmarksDetector
            from preparation.detectors.mediapipe.video_process import VideoProcess

            self.landmarks_detector = LandmarksDetector()
            self.video_process = VideoProcess(convert_gray=False)
        elif self.detector == "retinaface":
            from preparation.detectors.retinaface.detector import LandmarksDetector
            from preparation.detectors.retinaface.video_process import VideoProcess

            self.landmarks_detector = LandmarksDetector(device=self.device)
            self.video_process = VideoProcess(convert_gray=False)
        else:
            raise ValueError(f"Unknown detector: {self.detector}")

        # Initialize video transform (test-time: normalize, center-crop, grayscale)
        from datamodule.transforms import VideoTransform

        self.video_transform = VideoTransform(subset="test")

        logger.info("Inference pipeline ready.")

    def is_loaded(self) -> bool:
        """Check if the pipeline is fully loaded."""
        return self.model_module is not None and self.landmarks_detector is not None

    def predict_from_file(self, video_path: str) -> dict:
        """
        Run inference on a video file.

        Args:
            video_path: Path to the video file (.mp4, .avi, etc.)

        Returns:
            dict with keys:
                - text: Predicted transcript string
                - latency_ms: Total inference time in milliseconds
                - latency_breakdown: Dict of timing per stage
                - num_frames: Number of video frames processed
        """
        if not self.is_loaded():
            raise RuntimeError("Pipeline not loaded. Call load() first.")

        video_path = str(Path(video_path).resolve())
        assert Path(video_path).exists(), f"Video file not found: {video_path}"

        timings = {}
        total_start = time.time()

        # Step 1: Load video frames
        t0 = time.time()
        video_frames = torchvision.io.read_video(video_path, pts_unit="sec")[0].numpy()
        timings["load_video_ms"] = (time.time() - t0) * 1000
        logger.info(f"Loaded {len(video_frames)} frames from {video_path}")

        # Step 2: Detect face landmarks
        t0 = time.time()
        landmarks = self.landmarks_detector(video_frames)
        timings["face_detection_ms"] = (time.time() - t0) * 1000

        detected = sum(1 for lm in landmarks if lm is not None)
        logger.info(f"Face detected in {detected}/{len(landmarks)} frames")

        # Step 3: Crop mouth ROI
        t0 = time.time()
        cropped = self.video_process(video_frames, landmarks)
        timings["mouth_crop_ms"] = (time.time() - t0) * 1000

        if cropped is None:
            logger.warning("Could not crop mouth region — no face detected")
            return {
                "text": "",
                "latency_ms": (time.time() - total_start) * 1000,
                "latency_breakdown": timings,
                "num_frames": len(video_frames),
                "error": "No face detected in video",
            }

        # Step 4: Preprocess (to tensor, permute, transform)
        t0 = time.time()
        video_tensor = torch.tensor(cropped)
        # cropped is [T, H, W, C] — need [T, C, H, W] for transforms
        video_tensor = video_tensor.permute(0, 3, 1, 2).float()
        video_tensor = self.video_transform(video_tensor)
        timings["preprocess_ms"] = (time.time() - t0) * 1000

        logger.info(f"Preprocessed tensor shape: {video_tensor.shape}")

        # Step 5: Run model inference
        t0 = time.time()
        video_tensor = video_tensor.to(self.device)
        with torch.no_grad():
            predicted_text = self.model_module(video_tensor)
        timings["inference_ms"] = (time.time() - t0) * 1000

        total_ms = (time.time() - total_start) * 1000

        result = {
            "text": predicted_text.strip(),
            "latency_ms": total_ms,
            "latency_breakdown": timings,
            "num_frames": len(video_frames),
        }

        logger.info(f"Prediction: '{predicted_text.strip()}'")
        logger.info(f"Total latency: {total_ms:.0f}ms")

        return result

    def predict_from_frames(self, frames_tensor: torch.Tensor) -> dict:
        """
        Run inference on preprocessed mouth ROI frames.

        This is the method used by the real-time webcam pipeline (Phase 1+).
        The input should already be preprocessed: grayscale, normalized,
        center-cropped to 88x88.

        Args:
            frames_tensor: Preprocessed mouth ROI tensor [T, 1, 88, 88]

        Returns:
            dict with keys:
                - text: Predicted transcript string
                - latency_ms: Inference time in milliseconds
        """
        if not self.is_loaded():
            raise RuntimeError("Pipeline not loaded. Call load() first.")

        detailed = self.predict_from_frames_detailed(frames_tensor)
        return {
            "text": detailed["text"],
            "latency_ms": detailed["latency_ms"],
        }

    def predict_from_frames_detailed(self, frames_tensor: torch.Tensor) -> dict:
        """
        Run inference and expose beam-search score based confidence.

        Args:
            frames_tensor: Preprocessed mouth ROI tensor [T, 1, 88, 88]

        Returns:
            dict with keys:
                - text: Predicted transcript string
                - latency_ms: Inference + decode latency in milliseconds
                - confidence: Normalized confidence [0.0, 1.0]
                - score: Top hypothesis raw beam-search score
                - score_margin: Top-vs-second beam-search margin
        """
        if not self.is_loaded():
            raise RuntimeError("Pipeline not loaded. Call load() first.")

        _ensure_autoavsr_on_path()
        from lightning import get_beam_search_decoder

        t0 = time.time()

        sample = frames_tensor.to(self.device)
        model = self.model_module.model

        if not hasattr(self, "_beam_search") or self._beam_search is None:
            self._beam_search = get_beam_search_decoder(model, self.model_module.token_list)

        with torch.no_grad():
            features = model.frontend(sample.unsqueeze(0))
            features = model.proj_encoder(features)
            enc_feat, _ = model.encoder(features, None)
            enc_feat = enc_feat.squeeze(0)

            nbest_hyps = self._beam_search(enc_feat)
            nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 2)]]

            predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
            predicted_text = self.model_module.text_transform.post_process(predicted_token_id).replace("<eos>", "")

        latency_ms = (time.time() - t0) * 1000

        top_score = float(nbest_hyps[0].get("score", 0.0))
        second_score = float(nbest_hyps[1].get("score", top_score - 5.0)) if len(nbest_hyps) > 1 else top_score - 5.0
        score_margin = top_score - second_score

        token_count = max(1, len(nbest_hyps[0].get("yseq", [])) - 1)
        normalized_top_score = top_score / token_count

        margin_term = 1.0 / (1.0 + math.exp(-(score_margin - 1.0)))
        score_term = 1.0 / (1.0 + math.exp(-(normalized_top_score + 2.5)))
        confidence = max(0.0, min(1.0, 0.5 * margin_term + 0.5 * score_term))

        return {
            "text": predicted_text.strip(),
            "latency_ms": latency_ms,
            "confidence": confidence,
            "score": top_score,
            "score_margin": score_margin,
        }

    def unload(self):
        """Free model and detector resources."""
        self.loader.unload()
        self.model_module = None
        self.landmarks_detector = None
        self.video_process = None
        self.video_transform = None
        self._beam_search = None
