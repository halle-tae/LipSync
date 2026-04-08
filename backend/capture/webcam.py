"""OpenCV webcam capture utilities for real-time inference."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class WebcamInfo:
	camera_index: int
	width: int
	height: int
	fps_configured: float
	fps_reported: float


class WebcamCapture:
	"""Wraps OpenCV VideoCapture with predictable timing and safety checks."""

	def __init__(
		self,
		camera_index: int = 0,
		target_fps: int = 25,
		width: Optional[int] = None,
		height: Optional[int] = None,
	):
		self.camera_index = camera_index
		self.target_fps = target_fps
		self.width = width
		self.height = height

		self._capture: Optional[cv2.VideoCapture] = None
		self._frame_period = 1.0 / float(target_fps)
		self._last_frame_ts = 0.0

	def open(self) -> WebcamInfo:
		capture = cv2.VideoCapture(self.camera_index)
		if not capture.isOpened():
			raise RuntimeError(
				"Could not open webcam. Check camera permissions and whether another app is using it."
			)

		if self.width is not None:
			capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
		if self.height is not None:
			capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
		capture.set(cv2.CAP_PROP_FPS, float(self.target_fps))

		self._capture = capture
		self._last_frame_ts = 0.0

		return WebcamInfo(
			camera_index=self.camera_index,
			width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
			height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
			fps_configured=float(self.target_fps),
			fps_reported=float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
		)

	def is_open(self) -> bool:
		return self._capture is not None and self._capture.isOpened()

	def read(self) -> Tuple[np.ndarray, float]:
		if not self.is_open():
			raise RuntimeError("Webcam is not open. Call open() first.")

		assert self._capture is not None

		ok, frame = self._capture.read()
		if not ok or frame is None:
			raise RuntimeError("Failed to read a frame from webcam.")

		now = time.time()
		if self._last_frame_ts > 0:
			elapsed = now - self._last_frame_ts
			sleep_time = self._frame_period - elapsed
			if sleep_time > 0:
				time.sleep(sleep_time)
				now = time.time()
		self._last_frame_ts = now

		return frame, now

	def release(self):
		if self._capture is not None:
			self._capture.release()
			self._capture = None
