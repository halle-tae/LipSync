#!/usr/bin/env python3
"""Fixed Gradio UI for Gradio 6.x with proper streaming."""

import time
import sys
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

# Global runtime
pipeline = None
extractor = None
preprocessor = None
frame_buffer = []
last_infer_ts = 0.0
last_text = ""
last_confidence = 0.0


def init_model():
    global pipeline, extractor, preprocessor
    if pipeline is None:
        print("Loading model...")
        pipeline = InferencePipeline()
        pipeline.load()
        extractor = FaceMeshMouthExtractor(roi_size=96)
        preprocessor = MouthPreprocessor(roi_size=96, output_size=88)
        print("Model loaded!")


def process_frame(frame, window_seconds, infer_interval_seconds):
    global frame_buffer, last_infer_ts, last_text, last_confidence

    init_model()

    if frame is None:
        return None, f"<h2 style='color: #666;'>{last_text or 'Waiting...'}</h2>", f"Confidence: {last_confidence*100:.1f}%", "No frame"

    # Convert to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Detect face and mouth
    t0 = time.time()
    detection = extractor.extract(frame_bgr)
    detect_ms = (time.time() - t0) * 1000

    # Draw detection boxes
    preview = frame_bgr.copy()
    if detection is not None:
        fx1, fy1, fx2, fy2 = detection.face_bbox
        mx1, my1, mx2, my2 = detection.mouth_bbox
        cv2.rectangle(preview, (fx1, fy1), (fx2, fy2), (64, 220, 80), 2)
        cv2.rectangle(preview, (mx1, my1), (mx2, my2), (64, 128, 255), 2)
        cv2.putText(preview, "Face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Add mouth ROI to buffer
        frame_buffer.append(detection.mouth_roi_bgr)
        window_frames = max(1, int(25 * float(window_seconds)))
        if len(frame_buffer) > window_frames:
            frame_buffer = frame_buffer[-window_frames:]
    else:
        cv2.putText(preview, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

    # Run inference if enough time has passed and we have enough frames
    now = time.time()
    status = f"Buffered frames: {len(frame_buffer)} | Detection: {detect_ms:.1f}ms"

    min_frames = max(12, int(25 * float(window_seconds)) // 2)
    if now - last_infer_ts >= float(infer_interval_seconds) and len(frame_buffer) >= min_frames:
        t1 = time.time()
        input_tensor = preprocessor.process_frames(frame_buffer)
        preprocess_ms = (time.time() - t1) * 1000

        t2 = time.time()
        pred = pipeline.predict_from_frames_detailed(input_tensor)
        infer_ms = (time.time() - t2) * 1000

        last_infer_ts = now
        last_text = pred["text"]
        last_confidence = float(pred["confidence"])

        total_ms = (time.time() - t0) * 1000
        status = f"Inference done! detect={detect_ms:.1f}ms | preprocess={preprocess_ms:.1f}ms | infer={infer_ms:.1f}ms | total={total_ms:.1f}ms"

    # Create caption HTML
    conf_color = "#16a34a" if last_confidence >= 0.75 else "#ca8a04" if last_confidence >= 0.45 else "#dc2626"
    caption_html = f"""
    <div style='background: rgba(0,0,0,0.8); padding: 20px; border-radius: 10px; margin: 10px 0;'>
        <h1 style='color: white; margin: 0; font-size: 32px;'>{last_text or 'Waiting for speech...'}</h1>
        <p style='color: {conf_color}; margin-top: 10px; font-size: 18px;'>Confidence: {last_confidence*100:.1f}%</p>
    </div>
    """

    return preview_rgb, caption_html, f"Confidence: {last_confidence*100:.1f}%", status


def build_ui():
    with gr.Blocks(title="LipSync - Live Captions") as app:
        gr.Markdown("# LipSync — Real-Time Lip Reading")
        gr.Markdown("The webcam will start automatically. Speak clearly while facing the camera.")

        with gr.Row():
            with gr.Column(scale=2):
                webcam = gr.Image(label="Webcam", sources=["webcam"], streaming=True, type="numpy")
                preview = gr.Image(label="Preview (with detection boxes)", type="numpy")

            with gr.Column(scale=1):
                gr.Markdown("### Settings")
                window_seconds = gr.Slider(1.0, 3.0, value=2.0, step=0.1, label="Frame Window (seconds)")
                infer_interval_seconds = gr.Slider(0.8, 3.0, value=1.5, step=0.1, label="Inference Interval (seconds)")
                gr.Markdown("### Status")
                status = gr.Textbox(label="Status", lines=3, interactive=False)

        caption = gr.HTML(label="Live Captions")
        confidence = gr.Textbox(label="Model Confidence", interactive=False)

        # Stream processing
        webcam.stream(
            fn=process_frame,
            inputs=[webcam, window_seconds, infer_interval_seconds],
            outputs=[preview, caption, confidence, status],
            stream_every=0.1,  # Process 10 times per second
        )

    return app


if __name__ == "__main__":
    print("\nStarting LipSync Gradio UI...")
    print("Loading model (this may take a moment)...\n")

    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=7861, share=False)
