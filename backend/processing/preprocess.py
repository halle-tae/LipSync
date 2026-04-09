"""
Preprocessing utilities for Auto-AVSR webcam frame sequences.

IMPORTANT: This must exactly replicate Auto-AVSR's VideoTransform("test") pipeline:
    x / 255.0 → CenterCrop(88) → Grayscale → Normalize(mean=0.421, std=0.165)

Input: list of affine-aligned 96×96 BGR mouth ROI frames (from FaceMeshMouthExtractor)
Output: [T, 1, 88, 88] float32 tensor ready for model.frontend()
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
import torch
import torchvision

# Ensure Auto-AVSR is on path so we can use the *exact* same VideoTransform
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_AUTOAVSR_DIR = _PROJECT_ROOT / "auto_avsr" / "auto_avsr"  # Nested structure
if str(_AUTOAVSR_DIR) not in sys.path:
    sys.path.insert(0, str(_AUTOAVSR_DIR))

from datamodule.transforms import VideoTransform


class MouthPreprocessor:
    """
    Converts mouth ROI frame sequences to [T, 1, 88, 88] normalised tensors.

    Uses the exact same VideoTransform("test") from Auto-AVSR to guarantee
    the preprocessing matches what the model was trained with.
    """

    def __init__(
        self,
        roi_size: int = 96,
        output_size: int = 88,
    ):
        self.roi_size = roi_size
        self.output_size = output_size
        # Use the EXACT same transform pipeline the model was trained with
        self._video_transform = VideoTransform(subset="test")

    def process_frames(self, frames_bgr: Iterable[np.ndarray]) -> torch.Tensor:
        """
        Convert a sequence of BGR mouth ROI frames into the model input tensor.

        The pipeline mirrors VideoTransform("test"):
          1. Stack frames → [T, H, W, C] numpy
          2. Convert to tensor [T, C, H, W] float
          3. Apply VideoTransform: /255 → CenterCrop(88) → Grayscale → Normalize

        Args:
            frames_bgr: Iterable of (96, 96, 3) BGR numpy arrays

        Returns:
            Tensor of shape [T, 1, 88, 88]
        """
        frames: List[np.ndarray] = []
        for frame in frames_bgr:
            if frame is None or frame.size == 0:
                continue
            # Ensure correct size
            if frame.shape[0] != self.roi_size or frame.shape[1] != self.roi_size:
                frame = cv2.resize(frame, (self.roi_size, self.roi_size), interpolation=cv2.INTER_LINEAR)
            # Convert BGR → RGB (torchvision Grayscale expects RGB channel order)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        if len(frames) == 0:
            raise ValueError("No valid frames provided for preprocessing.")

        # Stack: [T, H, W, C] → tensor [T, C, H, W] float32
        stacked = np.stack(frames, axis=0)  # [T, 96, 96, 3]
        tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2).float()  # [T, 3, 96, 96]

        # Apply the EXACT same VideoTransform("test") as training:
        # /255 → CenterCrop(88) → Grayscale → Normalize(0.421, 0.165)
        tensor = self._video_transform(tensor)  # [T, 1, 88, 88]

        return tensor
