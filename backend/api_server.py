#!/usr/bin/env python3
"""FastAPI bridge for real-time web inference using the Auto-AVSR backend."""

from __future__ import annotations

import base64
import logging
import os
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.capture.face_detect import FaceMeshMouthExtractor
from backend.model.inference import InferencePipeline
from backend.processing.buffer import SlidingFrameBuffer
from backend.processing.preprocess import MouthPreprocessor


def _confidence_label(confidence_pct: float) -> str:
    if confidence_pct >= 75:
        return "High"
    if confidence_pct >= 45:
        return "Medium"
    return "Low"


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9\s]", "", text.upper())
    return re.sub(r"\s+", " ", cleaned).strip()


@dataclass
class SessionState:
    session_id: str
    infer_interval_seconds: float
    frame_window_seconds: float
    target_fps: int
    frames: SlidingFrameBuffer = field(default_factory=lambda: SlidingFrameBuffer(max_frames=90))
    last_infer_ts: float = 0.0
    last_text: str = "Looking for a speaker..."
    last_confidence_pct: float = 0.0
    word_count: int = 0
    last_frame_ts: float = 0.0
    fps: float = 0.0
    latency_detect_ms: float = 0.0
    latency_preprocess_ms: float = 0.0
    latency_inference_ms: float = 0.0
    latency_total_ms: float = 0.0
    last_emitted_normalized_text: str = ""
    repeated_prediction_count: int = 0


class RuntimeState:
    def __init__(self):
        self._lock = threading.Lock()
        self._pipeline: Optional[InferencePipeline] = None
        self._extractor: Optional[FaceMeshMouthExtractor] = None
        self._preprocessor: Optional[MouthPreprocessor] = None
        self._sessions: Dict[str, SessionState] = {}

    def ensure_loaded(self):
        with self._lock:
            if self._pipeline is not None:
                return
            pipeline = InferencePipeline()
            pipeline.load()
            self._pipeline = pipeline
            self._extractor = FaceMeshMouthExtractor(roi_size=96)
            self._preprocessor = MouthPreprocessor(roi_size=96, output_size=88)

    @property
    def pipeline(self) -> InferencePipeline:
        if self._pipeline is None:
            raise RuntimeError("Runtime not loaded")
        return self._pipeline

    @property
    def extractor(self) -> FaceMeshMouthExtractor:
        if self._extractor is None:
            raise RuntimeError("Runtime not loaded")
        return self._extractor

    @property
    def preprocessor(self) -> MouthPreprocessor:
        if self._preprocessor is None:
            raise RuntimeError("Runtime not loaded")
        return self._preprocessor

    def create_session(self, infer_interval_seconds: float, frame_window_seconds: float, target_fps: int) -> SessionState:
        session = SessionState(
            session_id=str(uuid.uuid4()),
            infer_interval_seconds=max(0.2, float(infer_interval_seconds)),
            frame_window_seconds=max(1.0, float(frame_window_seconds)),
            target_fps=max(1, int(target_fps)),
        )
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> SessionState:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return session

    def delete_session(self, session_id: str):
        with self._lock:
            self._sessions.pop(session_id, None)


runtime = RuntimeState()
app = FastAPI(title="LipSync Inference API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StartSessionRequest(BaseModel):
    infer_interval_seconds: float = Field(default=1.2, ge=0.2, le=5.0)
    frame_window_seconds: float = Field(default=2.0, ge=1.0, le=5.0)
    target_fps: int = Field(default=25, ge=1, le=60)


class StartSessionResponse(BaseModel):
    session_id: str
    model_loaded: bool


class StopSessionRequest(BaseModel):
    session_id: str


class FrameRequest(BaseModel):
    session_id: str
    frame_b64: Optional[str] = None
    frames_b64: Optional[List[str]] = None
    infer_interval_seconds: Optional[float] = Field(default=None, ge=0.2, le=5.0)
    frame_window_seconds: Optional[float] = Field(default=None, ge=1.0, le=5.0)


class TranscriptEntry(BaseModel):
    id: str
    timestamp_ms: int
    text: str
    confidence: int


class FrameResponse(BaseModel):
    status: str
    face_detected: bool
    caption: str
    confidence: int
    confidence_label: str
    metrics: Dict[str, Any]
    transcript_entry: Optional[TranscriptEntry] = None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": runtime._pipeline is not None}


@app.post("/session/start", response_model=StartSessionResponse)
def start_session(payload: StartSessionRequest):
    runtime.ensure_loaded()
    session = runtime.create_session(
        infer_interval_seconds=payload.infer_interval_seconds,
        frame_window_seconds=payload.frame_window_seconds,
        target_fps=payload.target_fps,
    )
    return StartSessionResponse(session_id=session.session_id, model_loaded=True)


@app.post("/session/stop")
def stop_session(payload: StopSessionRequest):
    runtime.delete_session(payload.session_id)
    return {"ok": True}


def _decode_frame(frame_b64: str) -> np.ndarray:
    encoded = frame_b64.split(",", 1)[-1]
    try:
        frame_bytes = base64.b64decode(encoded)
    except Exception as error:
        raise HTTPException(status_code=400, detail="Invalid base64 frame payload") from error

    np_buffer = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image frame")
    return frame


@app.post("/session/frame", response_model=FrameResponse)
def process_frame(payload: FrameRequest):
    runtime.ensure_loaded()
    session = runtime.get_session(payload.session_id)

    if payload.infer_interval_seconds is not None:
        session.infer_interval_seconds = max(0.2, float(payload.infer_interval_seconds))
    if payload.frame_window_seconds is not None:
        session.frame_window_seconds = max(1.0, float(payload.frame_window_seconds))

    # Collect raw frames — batch (frames_b64) or single (frame_b64)
    if payload.frames_b64:
        raw_b64_list = payload.frames_b64
    elif payload.frame_b64:
        raw_b64_list = [payload.frame_b64]
    else:
        raise HTTPException(status_code=400, detail="No frame data provided")

    frame_start = time.time()

    # Decode and run face detection on every frame in the batch,
    # buffering each mouth ROI so the model sees ~25 fps temporal data.
    detection_start = time.time()
    face_detected = False
    any_face_in_batch = False
    for frame_b64 in raw_b64_list:
        frame_bgr = _decode_frame(frame_b64)
        detection = runtime.extractor.extract(frame_bgr)
        now = time.time()
        if detection is not None:
            session.frames.append(detection.mouth_roi_bgr, now)
            face_detected = True
            any_face_in_batch = True

    detect_ms = (time.time() - detection_start) * 1000
    session.latency_detect_ms = detect_ms

    now = time.time()
    if session.last_frame_ts > 0:
        delta = now - session.last_frame_ts
        if delta > 0:
            session.fps = len(raw_b64_list) / delta
    session.last_frame_ts = now

    # Only clear buffer if no face was found in the entire batch
    if not any_face_in_batch:
        session.frames.clear()
        runtime.extractor.clear_buffer()
        session.repeated_prediction_count = 0

    transcript_entry: Optional[TranscriptEntry] = None
    window_frames = max(1, int(session.target_fps * session.frame_window_seconds))

    should_infer = (
        face_detected
        and (now - session.last_infer_ts >= session.infer_interval_seconds)
        and (len(session.frames) >= max(12, window_frames // 2))
    )

    if should_infer:
        preprocess_start = time.time()
        mouth_frames = session.frames.get_latest(window_frames)
        input_tensor = runtime.preprocessor.process_frames(mouth_frames)
        session.latency_preprocess_ms = (time.time() - preprocess_start) * 1000

        infer_start = time.time()
        prediction = runtime.pipeline.predict_from_frames_detailed(input_tensor)
        session.latency_inference_ms = (time.time() - infer_start) * 1000

        confidence_pct = int(round(float(prediction.get("confidence", 0.0)) * 100))
        raw_text = str(prediction.get("text", "")).strip()

        normalized_text = _normalize_text(raw_text)
        if normalized_text and normalized_text == session.last_emitted_normalized_text:
            session.repeated_prediction_count += 1
        else:
            session.repeated_prediction_count = 0

        if confidence_pct >= 60 and raw_text and session.repeated_prediction_count < 2:
            session.last_text = raw_text
            session.last_emitted_normalized_text = normalized_text
            session.word_count += len(raw_text.split())
            transcript_entry = TranscriptEntry(
                id=f"{int(now * 1000)}-{uuid.uuid4().hex[:8]}",
                timestamp_ms=int(now * 1000),
                text=raw_text,
                confidence=confidence_pct,
            )
        elif session.repeated_prediction_count >= 2:
            session.last_text = ""
            session.last_confidence_pct = 0.0
            session.last_infer_ts = now
            confidence_pct = 0
        else:
            session.last_text = ""

        session.last_confidence_pct = float(confidence_pct)
        session.last_infer_ts = now

    if not face_detected:
        session.last_text = ""
        session.last_confidence_pct = 0.0

    session.latency_total_ms = (time.time() - frame_start) * 1000

    confidence_pct = int(round(session.last_confidence_pct))
    response = FrameResponse(
        status="Processing" if face_detected else "Looking for a speaker...",
        face_detected=face_detected,
        caption=session.last_text,
        confidence=confidence_pct,
        confidence_label=_confidence_label(confidence_pct),
        metrics={
            "capture_ms": 0.0,
            "detect_ms": session.latency_detect_ms,
            "preprocess_ms": session.latency_preprocess_ms,
            "inference_ms": session.latency_inference_ms,
            "total_ms": session.latency_total_ms,
            "fps": round(session.fps, 1),
            "word_count": session.word_count,
        },
        transcript_entry=transcript_entry,
    )
    return response


def _deduplicate_captions(captions: List[dict]) -> List[dict]:
    """Remove overlapping repeated words between adjacent caption chunks."""
    if len(captions) <= 1:
        return captions

    result = [captions[0]]
    for cap in captions[1:]:
        prev_words = result[-1]["text"].split()
        cur_words = cap["text"].split()
        if not prev_words or not cur_words:
            result.append(cap)
            continue

        best_overlap = 0
        max_check = min(len(prev_words), len(cur_words))
        for k in range(1, max_check + 1):
            if prev_words[-k:] == cur_words[:k]:
                best_overlap = k

        if best_overlap > 0:
            cap = dict(cap)
            cap["text"] = " ".join(cur_words[best_overlap:])
        if cap["text"].strip():
            result.append(cap)

    return result


@app.post("/upload/video")
async def upload_video(file: UploadFile = File(...)):
    """
    Accept an MP4 upload, chunk into ~4s segments with 1s overlap,
    run face detection on every frame and inference on each chunk.
    Returns per-frame bounding boxes and per-chunk timestamped captions.
    """
    runtime.ensure_loaded()
    total_start = time.time()

    # Save uploaded file to a temp location
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        contents = await file.read()
        tmp.write(contents)
        tmp.flush()
        tmp.close()

        # Read all frames + FPS
        cap = cv2.VideoCapture(tmp.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        all_frames_bgr: List[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames_bgr.append(frame)
        cap.release()
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    if not all_frames_bgr:
        raise HTTPException(status_code=400, detail="Could not read any frames from the uploaded video.")

    total_frames = len(all_frames_bgr)
    total_duration_ms = int(total_frames / fps * 1000)
    logger.info(f"Upload: {total_frames} frames, {fps:.1f} fps, {total_duration_ms}ms duration")

    # Per-frame face/mouth bounding boxes
    extractor = FaceMeshMouthExtractor(roi_size=96)
    frame_data: List[dict] = []
    for i, frame_bgr in enumerate(all_frames_bgr):
        ts_ms = int(i / fps * 1000)
        detection = extractor.extract(frame_bgr)
        if detection is not None:
            frame_data.append({
                "frame_idx": i,
                "timestamp_ms": ts_ms,
                "face_bbox": list(detection.face_bbox),
                "mouth_bbox": list(detection.mouth_bbox),
            })
        else:
            frame_data.append({
                "frame_idx": i,
                "timestamp_ms": ts_ms,
                "face_bbox": None,
                "mouth_bbox": None,
            })
    extractor.close()

    # Chunk into ~4-second windows with ~3-second stride (1s overlap)
    chunk_size = int(round(4.0 * fps))
    stride = int(round(3.0 * fps))
    if stride < 1:
        stride = 1
    if chunk_size < 1:
        chunk_size = total_frames

    # Build RGB frame array for inference
    all_frames_rgb = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in all_frames_bgr])

    captions: List[dict] = []
    start_idx = 0
    while start_idx < total_frames:
        end_idx = min(start_idx + chunk_size, total_frames)
        chunk_frames = all_frames_rgb[start_idx:end_idx]
        start_ms = int(start_idx / fps * 1000)
        end_ms = int(end_idx / fps * 1000)

        logger.info(f"Inference chunk: frames {start_idx}-{end_idx} ({start_ms}ms-{end_ms}ms)")
        result = runtime.pipeline.predict_from_numpy_frames(chunk_frames)

        text = result.get("text", "").strip()
        confidence = int(round(float(result.get("confidence", 0.0)) * 100))

        if text:
            captions.append({
                "start_ms": start_ms,
                "end_ms": end_ms,
                "text": text,
                "confidence": confidence,
            })

        start_idx += stride

    captions = _deduplicate_captions(captions)
    processing_time_ms = int((time.time() - total_start) * 1000)

    return {
        "fps": round(fps, 2),
        "total_duration_ms": total_duration_ms,
        "processing_time_ms": processing_time_ms,
        "frames": frame_data,
        "captions": captions,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)
