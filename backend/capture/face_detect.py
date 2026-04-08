"""
Affine-aligned mouth ROI extraction for real-time webcam frames.

This replicates the **exact** preprocessing that Auto-AVSR uses during training
(see auto_avsr/preparation/detectors/mediapipe/video_process.py).

Key steps (matching the training pipeline):
1. Detect face with MediaPipe and extract 4 keypoints:
   right eye, left eye, nose tip, mouth center
2. Estimate a similarity (affine) transform that warps the face
   onto a canonical 256×256 reference pose (20words_mean_face.npy)
3. Cut a 96×96 mouth patch from the aligned image
4. Maintain a temporal landmark buffer for smoothing (±window_margin)

Without affine alignment the model receives spatially-inconsistent crops
and hallucinates stock phrases from its training distribution.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


# ---------------------------------------------------------------------------
# Path to the canonical mean-face landmarks shipped with Auto-AVSR
# ---------------------------------------------------------------------------
_MEAN_FACE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "auto_avsr",
    "preparation",
    "detectors",
    "mediapipe",
    "20words_mean_face.npy",
)


# ---------------------------------------------------------------------------
# MediaPipe Face Mesh indices for the 4 stable reference points
# (right eye center, left eye center, nose tip, mouth center)
# These mirror the 4 keypoints returned by MediaPipe FaceDetection that
# Auto-AVSR uses in its original pipeline.
# ---------------------------------------------------------------------------
# Right eye ring (mesh indices)
_RIGHT_EYE_IDXS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
# Left eye ring
_LEFT_EYE_IDXS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
# Nose tip
_NOSE_TIP_IDX = 1
# Upper/lower lip for mouth center
_UPPER_LIP_IDX = 13
_LOWER_LIP_IDX = 14


@dataclass
class FaceMouthDetection:
    """Result of face detection + affine-aligned mouth crop."""
    mouth_roi_bgr: np.ndarray          # 96×96 BGR patch (affine-aligned)
    face_bbox: Tuple[int, int, int, int]
    mouth_bbox: Tuple[int, int, int, int]
    num_faces: int
    landmarks_4pt: np.ndarray          # The 4 keypoints used for alignment


def _cut_patch(img: np.ndarray, landmarks: np.ndarray, height: int, width: int, threshold: int = 5) -> Optional[np.ndarray]:
    """Cut a patch centred on *landmarks* (same logic as Auto-AVSR)."""
    center_x, center_y = np.mean(landmarks, axis=0)
    if abs(center_y - img.shape[0] / 2) > height + threshold:
        return None
    if abs(center_x - img.shape[1] / 2) > width + threshold:
        return None
    y_min = int(round(np.clip(center_y - height, 0, img.shape[0])))
    y_max = int(round(np.clip(center_y + height, 0, img.shape[0])))
    x_min = int(round(np.clip(center_x - width, 0, img.shape[1])))
    x_max = int(round(np.clip(center_x + width, 0, img.shape[1])))
    patch = img[y_min:y_max, x_min:x_max]
    if patch.size == 0:
        return None
    return np.copy(patch)


class FaceMeshMouthExtractor:
    """
    Extracts an affine-aligned 96×96 mouth ROI from a single webcam frame.

    Maintains an internal landmark ring-buffer so that temporal smoothing
    can be applied identically to the Auto-AVSR training pipeline.
    """

    def __init__(
        self,
        roi_size: int = 96,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        window_margin: int = 12,
        landmark_buffer_size: int = 30,
    ):
        self.roi_size = roi_size
        self.window_margin = window_margin

        # Canonical mean-face reference (68 landmarks)
        self.reference = np.load(_MEAN_FACE_PATH)

        # Stable reference points from the mean face
        # Auto-AVSR uses indices 0-3 as (right eye, left eye, nose, mouth)
        # computed from the 68-point mean face:
        self._stable_ref = np.vstack([
            np.mean(self.reference[36:42], axis=0),   # right eye
            np.mean(self.reference[42:48], axis=0),   # left eye
            np.mean(self.reference[31:36], axis=0),   # nose tip
            np.mean(self.reference[48:68], axis=0),   # mouth center
        ])

        # Mouth crop indices in the 4-point landmark array
        self._crop_start_idx = 3  # mouth center start
        self._crop_stop_idx = 4   # mouth center stop

        # MediaPipe Face Mesh
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # Temporal landmark smoothing buffer
        self._landmark_buffer: Deque[np.ndarray] = deque(maxlen=landmark_buffer_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, frame_bgr: np.ndarray) -> Optional[FaceMouthDetection]:
        """Detect face, affine-align, and return a 96×96 mouth patch."""
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._mesh.process(frame_rgb)

        if not result.multi_face_landmarks:
            return None

        # Pick the largest face
        best_face = None
        best_area = -1.0
        best_bbox = (0, 0, 0, 0)
        for face in result.multi_face_landmarks:
            pts = self._all_points(face.landmark, width, height)
            bbox = self._bbox(pts, width, height)
            area = float(max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1]))
            if area > best_area:
                best_face = face
                best_area = area
                best_bbox = bbox

        if best_face is None:
            return None

        # Extract the 4 stable keypoints from Face Mesh
        landmarks_4pt = self._extract_4pt(best_face.landmark, width, height)

        # Push into temporal buffer
        self._landmark_buffer.append(landmarks_4pt)

        # Temporal smoothing (mirror Auto-AVSR's window_margin approach)
        buf_len = len(self._landmark_buffer)
        half_win = min(self.window_margin // 2, buf_len - 1)
        if half_win > 0:
            window = list(self._landmark_buffer)[-half_win - 1:]
            smoothed = np.mean(window, axis=0)
            # Re-center: keep current frame's mean position
            smoothed += landmarks_4pt.mean(axis=0) - smoothed.mean(axis=0)
        else:
            smoothed = landmarks_4pt.copy()

        # Estimate affine transform to canonical reference
        transform_matrix = cv2.estimateAffinePartial2D(
            smoothed.astype(np.float32),
            self._stable_ref.astype(np.float32),
            method=cv2.LMEDS,
        )[0]

        if transform_matrix is None:
            return None

        # Warp the frame (BGR, no grayscale conversion yet — preprocessor handles that)
        warped = cv2.warpAffine(
            frame_bgr,
            transform_matrix,
            dsize=(256, 256),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # Transform landmarks into warped space
        warped_landmarks = (
            np.matmul(smoothed, transform_matrix[:, :2].T)
            + transform_matrix[:, 2].T
        )

        # Cut 96×96 mouth patch (half-height = half-width = 48)
        mouth_pts = warped_landmarks[self._crop_start_idx:self._crop_stop_idx]
        patch = _cut_patch(warped, mouth_pts, self.roi_size // 2, self.roi_size // 2, threshold=20)

        if patch is None:
            # Fallback: centre-crop from warped image at mouth location
            cx, cy = warped_landmarks[3]  # mouth centre
            half = self.roi_size // 2
            y1 = int(np.clip(cy - half, 0, 255))
            x1 = int(np.clip(cx - half, 0, 255))
            patch = warped[y1:y1 + self.roi_size, x1:x1 + self.roi_size]
            if patch.shape[0] != self.roi_size or patch.shape[1] != self.roi_size:
                patch = cv2.resize(patch, (self.roi_size, self.roi_size))

        if patch.shape[0] != self.roi_size or patch.shape[1] != self.roi_size:
            patch = cv2.resize(patch, (self.roi_size, self.roi_size), interpolation=cv2.INTER_LINEAR)

        # Compute mouth bbox in original image coords for metadata
        inv_transform = cv2.invertAffineTransform(transform_matrix)
        mouth_center_orig = (
            np.matmul(warped_landmarks[3:4], inv_transform[:, :2].T)
            + inv_transform[:, 2].T
        )[0]
        mhalf = 48
        mx1 = int(np.clip(mouth_center_orig[0] - mhalf, 0, width))
        my1 = int(np.clip(mouth_center_orig[1] - mhalf, 0, height))
        mx2 = int(np.clip(mouth_center_orig[0] + mhalf, 0, width))
        my2 = int(np.clip(mouth_center_orig[1] + mhalf, 0, height))

        return FaceMouthDetection(
            mouth_roi_bgr=patch,
            face_bbox=best_bbox,
            mouth_bbox=(mx1, my1, mx2, my2),
            num_faces=len(result.multi_face_landmarks),
            landmarks_4pt=landmarks_4pt,
        )

    def clear_buffer(self):
        """Reset the temporal landmark buffer (e.g. when face is lost)."""
        self._landmark_buffer.clear()

    def close(self):
        if self._mesh is not None:
            self._mesh.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _all_points(landmarks, width: int, height: int) -> np.ndarray:
        return np.array([(lm.x * width, lm.y * height) for lm in landmarks], dtype=np.float32)

    @staticmethod
    def _bbox(points: np.ndarray, width: int, height: int) -> Tuple[int, int, int, int]:
        x_min = int(np.clip(np.floor(np.min(points[:, 0])), 0, width - 1))
        y_min = int(np.clip(np.floor(np.min(points[:, 1])), 0, height - 1))
        x_max = int(np.clip(np.ceil(np.max(points[:, 0])), 0, width - 1))
        y_max = int(np.clip(np.ceil(np.max(points[:, 1])), 0, height - 1))
        return x_min, y_min, x_max, y_max

    def _extract_4pt(self, landmarks, width: int, height: int) -> np.ndarray:
        """
        Extract 4 stable keypoints from MediaPipe Face Mesh to match the
        4 keypoints that Auto-AVSR's MediaPipe FaceDetection returns:
        [right_eye_center, left_eye_center, nose_tip, mouth_center]
        """
        all_pts = self._all_points(landmarks, width, height)

        right_eye = np.mean(all_pts[_RIGHT_EYE_IDXS], axis=0)
        left_eye = np.mean(all_pts[_LEFT_EYE_IDXS], axis=0)
        nose_tip = all_pts[_NOSE_TIP_IDX]
        mouth_center = (all_pts[_UPPER_LIP_IDX] + all_pts[_LOWER_LIP_IDX]) / 2.0

        return np.array([right_eye, left_eye, nose_tip, mouth_center], dtype=np.float32)
