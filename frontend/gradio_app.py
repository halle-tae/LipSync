#!/usr/bin/env python3
"""Phase 2 Gradio UI for live lip-reading captions with accessibility controls."""

from __future__ import annotations

import time
import sys
from threading import Lock
from typing import Any, Dict, List
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.capture.face_detect import FaceMeshMouthExtractor
from backend.model.inference import InferencePipeline
from backend.processing.preprocess import MouthPreprocessor

_runtime_lock = Lock()
_runtime: Dict[str, Any] = {}


def _get_runtime() -> Dict[str, Any]:
    with _runtime_lock:
        if _runtime:
            return _runtime

        pipeline = InferencePipeline()
        pipeline.load()

        _runtime["pipeline"] = pipeline
        _runtime["extractor"] = FaceMeshMouthExtractor(roi_size=96)
        _runtime["preprocessor"] = MouthPreprocessor(roi_size=96, output_size=88)
        return _runtime


def _default_state() -> Dict[str, Any]:
    return {
        "frames": [],
        "last_infer_ts": 0.0,
        "last_text": "",
        "last_confidence": 0.0,
        "last_metrics": "Waiting for enough frames...",
    }


def _confidence_color(value: float) -> str:
    if value >= 0.75:
        return "#16a34a"
    if value >= 0.45:
        return "#ca8a04"
    return "#dc2626"


def _caption_html(
    text: str,
    confidence: float,
    font_size: int,
    font_color: str,
    bg_opacity: float,
    position: str,
    dyslexia_font: bool,
) -> str:
    if position == "top":
        pos_style = "top: 12px; left: 12px; right: 12px;"
    elif position == "bottom":
        pos_style = "bottom: 12px; left: 12px; right: 12px;"
    else:
        pos_style = "top: 50%; left: 50%; transform: translate(-50%, -50%); width: 80%;"

    font_family = "OpenDyslexic, Arial, sans-serif" if dyslexia_font else "Arial, sans-serif"
    bg_alpha = max(0.0, min(1.0, bg_opacity))
    conf_color = _confidence_color(confidence)
    caption = text if text.strip() else "Looking for speech..."

    return f"""
    <div style=\"position: relative; min-height: 220px; border: 1px solid #ddd; border-radius: 8px; background: #101010;\">
      <div style=\"position: absolute; {pos_style} background: rgba(0,0,0,{bg_alpha:.2f}); border-radius: 10px; padding: 12px 16px;\">
        <div style=\"font-size: {font_size}px; color: {font_color}; font-family: {font_family}; line-height: 1.35; font-weight: 600;\">{caption}</div>
        <div style=\"margin-top: 8px; font-size: 13px; color: {conf_color}; font-family: Arial, sans-serif;\">Confidence: {confidence * 100:.1f}%</div>
      </div>
    </div>
    """


def _confidence_html(confidence: float) -> str:
    pct = max(0.0, min(100.0, confidence * 100.0))
    color = _confidence_color(confidence)
    return f"""
    <div style=\"width:100%; background:#e5e7eb; border-radius:999px; height:18px; overflow:hidden; border:1px solid #d1d5db;\">
      <div style=\"width:{pct:.1f}%; background:{color}; height:100%;\"></div>
    </div>
    <div style=\"margin-top:6px; font-size:13px; color:#374151;\">{pct:.1f}%</div>
    """


def _draw_preview(frame_bgr: np.ndarray, detection) -> np.ndarray:
    preview = frame_bgr.copy()
    if detection is None:
        cv2.putText(
            preview,
            "Looking for a speaker...",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

    fx1, fy1, fx2, fy2 = detection.face_bbox
    mx1, my1, mx2, my2 = detection.mouth_bbox
    cv2.rectangle(preview, (fx1, fy1), (fx2, fy2), (64, 220, 80), 2)
    cv2.rectangle(preview, (mx1, my1), (mx2, my2), (64, 128, 255), 2)
    cv2.putText(
        preview,
        f"faces={detection.num_faces}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)


def _process_stream(
    frame_rgb: np.ndarray,
    state: Dict[str, Any],
    font_size: int,
    font_color: str,
    bg_opacity: float,
    position: str,
    dyslexia_font: bool,
    window_seconds: float,
    infer_interval_seconds: float,
):
    if frame_rgb is None:
        caption = _caption_html(
            state.get("last_text", ""),
            float(state.get("last_confidence", 0.0)),
            font_size,
            font_color,
            bg_opacity,
            position,
            dyslexia_font,
        )
        return None, caption, _confidence_html(float(state.get("last_confidence", 0.0))), state.get("last_metrics", ""), "Idle", state

    runtime = _get_runtime()
    extractor = runtime["extractor"]
    preprocessor = runtime["preprocessor"]
    pipeline = runtime["pipeline"]

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    t0 = time.time()
    detection = extractor.extract(frame_bgr)
    detect_ms = (time.time() - t0) * 1000

    now = time.time()
    window_frames = max(1, int(25 * float(window_seconds)))

    frames: List[np.ndarray] = state.get("frames", [])
    if detection is not None:
        frames.append(detection.mouth_roi_bgr)
        if len(frames) > window_frames:
            frames = frames[-window_frames:]
    state["frames"] = frames

    active = "Tracking"
    if now - float(state.get("last_infer_ts", 0.0)) >= float(infer_interval_seconds) and len(frames) >= max(12, window_frames // 2):
        preprocess_start = time.time()
        input_tensor = preprocessor.process_frames(frames)
        preprocess_ms = (time.time() - preprocess_start) * 1000

        infer_start = time.time()
        pred = pipeline.predict_from_frames_detailed(input_tensor)
        infer_decode_ms = (time.time() - infer_start) * 1000

        total_ms = (time.time() - t0) * 1000
        state["last_infer_ts"] = now
        state["last_text"] = pred["text"]
        state["last_confidence"] = float(pred["confidence"])
        state["last_metrics"] = (
            f"capture/stream: browser | detect={detect_ms:.1f}ms | preprocess={preprocess_ms:.1f}ms | "
            f"inference+decode={infer_decode_ms:.1f}ms | total={total_ms:.1f}ms | frames={len(frames)}"
        )
        active = "Processing"

    preview = _draw_preview(frame_bgr, detection)
    caption = _caption_html(
        state.get("last_text", ""),
        float(state.get("last_confidence", 0.0)),
        font_size,
        font_color,
        bg_opacity,
        position,
        dyslexia_font,
    )

    return (
        preview,
        caption,
        _confidence_html(float(state.get("last_confidence", 0.0))),
        state.get("last_metrics", ""),
        active,
        state,
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="LipSync Phase 2", theme=gr.themes.Soft()) as app:
        gr.Markdown("## LipSync — Live Caption Overlay (Phase 2)")

        with gr.Row():
            with gr.Column(scale=2):
                webcam = gr.Image(label="Webcam Input", sources=["webcam"], streaming=True, type="numpy")
                preview = gr.Image(label="Webcam Preview (Face + Mouth ROI)", type="numpy")

            with gr.Column(scale=1):
                gr.Markdown("### Settings")
                font_size = gr.Slider(16, 48, value=28, step=1, label="Font Size")
                font_color = gr.ColorPicker(value="#FFFFFF", label="Font Color")
                bg_opacity = gr.Slider(0.1, 1.0, value=0.65, step=0.05, label="Background Opacity")
                position = gr.Radio(["top", "bottom", "floating"], value="bottom", label="Caption Position")
                dyslexia_font = gr.Checkbox(value=False, label="Use Dyslexia-Friendly Font")
                gr.Markdown("### Runtime")
                window_seconds = gr.Slider(1.0, 3.0, value=2.0, step=0.1, label="Frame Window (seconds)")
                infer_interval_seconds = gr.Slider(0.8, 2.0, value=1.2, step=0.1, label="Inference Interval (seconds)")
                status = gr.Textbox(label="Model Status", value="Idle", interactive=False)

        caption_html = gr.HTML(label="Caption Overlay")
        with gr.Row():
            confidence = gr.HTML(label="Confidence")
            latency = gr.Textbox(label="Latency Metrics", lines=3, interactive=False)

        state = gr.State(_default_state())

        webcam.stream(
            fn=_process_stream,
            inputs=[
                webcam,
                state,
                font_size,
                font_color,
                bg_opacity,
                position,
                dyslexia_font,
                window_seconds,
                infer_interval_seconds,
            ],
            outputs=[preview, caption_html, confidence, latency, status, state],
            concurrency_limit=1,
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch()
