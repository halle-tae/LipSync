"""Sliding window frame buffer utilities."""

from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Deque, List, Tuple, TypeVar


T = TypeVar("T")


class SlidingFrameBuffer:
	"""Thread-safe ring buffer that stores timestamped frames."""

	def __init__(self, max_frames: int = 75):
		if max_frames <= 0:
			raise ValueError("max_frames must be > 0")

		self.max_frames = max_frames
		self._buffer: Deque[Tuple[T, float]] = deque(maxlen=max_frames)
		self._lock = Lock()

	def append(self, frame: T, timestamp: float):
		with self._lock:
			self._buffer.append((frame, timestamp))

	def clear(self):
		with self._lock:
			self._buffer.clear()

	def __len__(self) -> int:
		with self._lock:
			return len(self._buffer)

	def get_latest(self, n_frames: int) -> List[T]:
		if n_frames <= 0:
			return []
		with self._lock:
			items = list(self._buffer)[-n_frames:]
		return [frame for frame, _ in items]

	def get_latest_with_timestamps(self, n_frames: int) -> List[Tuple[T, float]]:
		if n_frames <= 0:
			return []
		with self._lock:
			return list(self._buffer)[-n_frames:]
