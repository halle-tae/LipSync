"""
LipSync — MediaPipe Face Mesh Utilities

Provides face detection, facial landmark extraction, and lip ROI cropping
using the MediaPipe FaceLandmarker Tasks API (478 landmarks per face).
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger("lipsync.face_mesh")

# ---------------------------------------------------------------------------
# Lip landmark indices (MediaPipe Face Mesh / FaceLandmarker)
# ---------------------------------------------------------------------------
# Outer lip contour
OUTER_LIP_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
]

# Inner lip contour
INNER_LIP_INDICES = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
]

# Combined lip region for ROI extraction
ALL_LIP_INDICES = sorted(set(OUTER_LIP_INDICES + INNER_LIP_INDICES))

# Default ROI dimensions
DEFAULT_ROI_WIDTH = 100
DEFAULT_ROI_HEIGHT = 50

# Default model path
DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "model" / "face_landmarker.task"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class FaceDetection:
    """Holds detection results for a single frame."""
    landmarks: np.ndarray | None  # (478, 3) — normalised x, y, z
    lip_landmarks: np.ndarray | None  # (N, 2) — pixel x, y for lip points
    lip_roi: np.ndarray | None  # Cropped lip region (H, W) grayscale
    bbox: tuple[int, int, int, int] | None  # (x, y, w, h) face bounding box
    frame_rgb: np.ndarray | None  # Original frame in RGB


# ---------------------------------------------------------------------------
# FaceMeshProcessor
# ---------------------------------------------------------------------------
class FaceMeshProcessor:
    """
    Wraps MediaPipe FaceLandmarker (Tasks API) for frame-by-frame face
    detection and lip ROI extraction.

    Usage:
        processor = FaceMeshProcessor()
        result = processor.process_frame(frame_bgr)
        if result.lip_roi is not None:
            # result.lip_roi is a (50, 100) grayscale numpy array
            ...
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        roi_width: int = DEFAULT_ROI_WIDTH,
        roi_height: int = DEFAULT_ROI_HEIGHT,
        model_path: str | Path | None = None,
    ):
        self.roi_width = roi_width
        self.roi_height = roi_height

        model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(
                f"FaceLandmarker model not found at {model_path}. "
                "Download it with:\n"
                "  wget -O backend/model/face_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/latest/face_landmarker.task"
            )

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        running_mode = (
            VisionRunningMode.IMAGE if static_image_mode else VisionRunningMode.VIDEO
        )

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=running_mode,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        self._landmarker = FaceLandmarker.create_from_options(options)
        self._static = static_image_mode
        self._frame_ts = 0  # Monotonic timestamp for VIDEO mode

    # ------------------------------------------------------------------
    # Core Processing
    # ------------------------------------------------------------------
    def process_frame(self, frame_bgr: np.ndarray) -> FaceDetection:
        """
        Detect the face, extract landmarks, and crop the lip ROI.

        Args:
            frame_bgr: BGR image as read by OpenCV.

        Returns:
            FaceDetection with all fields populated (or None fields if
            no face was detected).
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        if self._static:
            results = self._landmarker.detect(mp_image)
        else:
            self._frame_ts += 33  # ~30 FPS timestamps
            results = self._landmarker.detect_for_video(mp_image, self._frame_ts)

        if not results.face_landmarks:
            return FaceDetection(
                landmarks=None,
                lip_landmarks=None,
                lip_roi=None,
                bbox=None,
                frame_rgb=frame_rgb,
            )

        # Use the first (or largest) face
        face_lms = results.face_landmarks[0]

        # Full landmarks → numpy (normalised)
        landmarks = np.array(
            [(lm.x, lm.y, lm.z) for lm in face_lms],
            dtype=np.float32,
        )

        # Lip landmarks in pixel coordinates
        lip_landmarks = np.array(
            [
                (int(face_lms[i].x * w), int(face_lms[i].y * h))
                for i in ALL_LIP_INDICES
            ],
            dtype=np.int32,
        )

        # Bounding box for the lip region
        lip_roi, bbox = self._extract_lip_roi(frame_bgr, lip_landmarks)

        return FaceDetection(
            landmarks=landmarks,
            lip_landmarks=lip_landmarks,
            lip_roi=lip_roi,
            bbox=bbox,
            frame_rgb=frame_rgb,
        )

    # ------------------------------------------------------------------
    # Lip ROI Extraction
    # ------------------------------------------------------------------
    def _extract_lip_roi(
        self,
        frame_bgr: np.ndarray,
        lip_landmarks: np.ndarray,
    ) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
        """
        Crop, resize, and convert the lip region to a fixed-size grayscale image.

        Args:
            frame_bgr: The full BGR frame.
            lip_landmarks: (N, 2) pixel coords of lip landmarks.

        Returns:
            (lip_roi, (x, y, w, h)) — the cropped grayscale ROI and its
            bounding box in the original frame, or (None, None) on failure.
        """
        h, w = frame_bgr.shape[:2]

        x_min = max(0, int(lip_landmarks[:, 0].min()))
        x_max = min(w, int(lip_landmarks[:, 0].max()))
        y_min = max(0, int(lip_landmarks[:, 1].min()))
        y_max = min(h, int(lip_landmarks[:, 1].max()))

        # Add padding (20% of the lip bbox size)
        pad_x = int((x_max - x_min) * 0.2)
        pad_y = int((y_max - y_min) * 0.2)
        x_min = max(0, x_min - pad_x)
        x_max = min(w, x_max + pad_x)
        y_min = max(0, y_min - pad_y)
        y_max = min(h, y_max + pad_y)

        roi_w = x_max - x_min
        roi_h = y_max - y_min

        if roi_w < 5 or roi_h < 5:
            logger.warning("Lip ROI too small (%dx%d), skipping.", roi_w, roi_h)
            return None, None

        lip_crop = frame_bgr[y_min:y_max, x_min:x_max]

        # Convert to grayscale & resize to standard dimensions
        lip_gray = cv2.cvtColor(lip_crop, cv2.COLOR_BGR2GRAY)
        lip_resized = cv2.resize(
            lip_gray, (self.roi_width, self.roi_height), interpolation=cv2.INTER_AREA
        )

        return lip_resized, (x_min, y_min, roi_w, roi_h)

    # ------------------------------------------------------------------
    # Visualisation Helpers
    # ------------------------------------------------------------------
    def draw_landmarks(
        self,
        frame_bgr: np.ndarray,
        detection: FaceDetection,
        draw_lip_box: bool = True,
        draw_lip_points: bool = True,
    ) -> np.ndarray:
        """
        Draw face mesh landmarks and lip ROI bounding box on the frame.

        Args:
            frame_bgr: The original BGR frame (will be copied, not modified).
            detection: FaceDetection result from process_frame().
            draw_lip_box: Whether to draw the lip bounding box.
            draw_lip_points: Whether to draw individual lip landmark points.

        Returns:
            Annotated BGR frame.
        """
        annotated = frame_bgr.copy()

        if detection.lip_landmarks is None:
            return annotated

        if draw_lip_points:
            for x, y in detection.lip_landmarks:
                cv2.circle(annotated, (x, y), 1, (0, 255, 0), -1)

        if draw_lip_box and detection.bbox is not None:
            bx, by, bw, bh = detection.bbox
            cv2.rectangle(
                annotated, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2
            )

        return annotated

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def close(self):
        """Release MediaPipe resources."""
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
