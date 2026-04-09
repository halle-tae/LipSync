#!/usr/bin/env python3
"""Simplified Gradio UI with manual snapshot processing."""

import time
import sys
from pathlib import Path
from collections import deque
import cv2
import gradio as gr
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.capture.face_detect import FaceMeshMouthExtractor
from backend.model.inference import InferencePipeline
from backend.processing.preprocess import MouthPreprocessor

# Global state
pipeline = None
extractor = None
preprocessor = None
frame_buffer = deque(maxlen=75)  # ~3 seconds at 25fps


def init_model():
    """Initialize model on first use."""
    global pipeline, extractor, preprocessor
    if pipeline is None:
        print("Loading model...")
        pipeline = InferencePipeline()
        pipeline.load()
        extractor = FaceMeshMouthExtractor(roi_size=96)
        preprocessor = MouthPreprocessor(roi_size=96, output_size=88)
        print("Model loaded!")
    return "Model loaded and ready!"


def process_webcam_snapshot(image):
    """Process a single webcam frame and add to buffer."""
    global frame_buffer

    print(f"DEBUG: process_webcam_snapshot called, image is None: {image is None}")
    if image is not None:
        print(f"DEBUG: Image shape: {image.shape}")

    if image is None:
        return None, "❌ No image! Click the camera icon in the webcam box first to take a photo!", "⚠️ No image received - take a photo first!"

    # Make sure model is loaded
    init_model()

    # Convert to BGR
    frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Detect face
    t0 = time.time()
    detection = extractor.extract(frame_bgr)
    detect_ms = (time.time() - t0) * 1000

    # Draw preview
    preview = frame_bgr.copy()
    if detection is not None:
        fx1, fy1, fx2, fy2 = detection.face_bbox
        mx1, my1, mx2, my2 = detection.mouth_bbox
        cv2.rectangle(preview, (fx1, fy1), (fx2, fy2), (64, 220, 80), 3)
        cv2.rectangle(preview, (mx1, my1), (mx2, my2), (255, 128, 64), 3)
        cv2.putText(preview, "FACE DETECTED", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # Add to buffer
        frame_buffer.append(detection.mouth_roi_bgr)
        status = f"✓ Face detected | Buffer: {len(frame_buffer)} frames | Detection: {detect_ms:.1f}ms"
    else:
        cv2.putText(preview, "NO FACE - Please face camera", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        status = f"✗ No face detected | Buffer: {len(frame_buffer)} frames"

    preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

    return preview_rgb, f"Buffered {len(frame_buffer)} frames. Click 'Run Lip Reading' when ready.", status


def run_inference():
    """Run inference on buffered frames."""
    global frame_buffer

    if len(frame_buffer) < 25:
        return (
            f"<h2 style='color: #dc2626;'>Need at least 25 frames (you have {len(frame_buffer)})</h2>",
            "0.0%",
            f"Not enough frames. Keep capturing! Need {25 - len(frame_buffer)} more."
        )

    # Run inference
    t0 = time.time()
    frames_list = list(frame_buffer)

    t1 = time.time()
    input_tensor = preprocessor.process_frames(frames_list[-50:])  # Use last 50 frames (~2 sec)
    preprocess_ms = (time.time() - t1) * 1000

    t2 = time.time()
    pred = pipeline.predict_from_frames_detailed(input_tensor)
    infer_ms = (time.time() - t2) * 1000

    total_ms = (time.time() - t0) * 1000

    text = pred["text"]
    confidence = float(pred["confidence"])

    # Color based on confidence
    conf_color = "#16a34a" if confidence >= 0.75 else "#ca8a04" if confidence >= 0.45 else "#dc2626"

    caption_html = f"""
    <div style='background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
                padding: 30px; border-radius: 15px; margin: 10px 0;
                border: 3px solid {conf_color}; box-shadow: 0 4px 6px rgba(0,0,0,0.3);'>
        <h1 style='color: white; margin: 0; font-size: 48px; text-align: center;
                   text-shadow: 2px 2px 4px rgba(0,0,0,0.5);'>
            "{text}"
        </h1>
        <div style='text-align: center; margin-top: 20px;'>
            <span style='color: {conf_color}; font-size: 24px; font-weight: bold;'>
                Confidence: {confidence*100:.1f}%
            </span>
        </div>
    </div>
    """

    status = f"✓ Inference complete! | Preprocess: {preprocess_ms:.1f}ms | Inference: {infer_ms:.1f}ms | Total: {total_ms:.1f}ms | Frames used: {len(frames_list[-50:])}"

    return caption_html, f"{confidence*100:.1f}%", status


def clear_buffer():
    """Clear the frame buffer."""
    global frame_buffer
    frame_buffer.clear()
    return "Buffer cleared! Start capturing fresh frames.", "Buffer: 0 frames"


def build_ui():
    with gr.Blocks(title="LipSync - Manual Mode", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎤 LipSync — Real-Time Lip Reading Assistant")
        gr.Markdown("""
        ### How to use:
        1. Click **'Capture Frame'** repeatedly to build up a buffer (need at least 25 frames, ~1 second)
        2. Keep your face visible and speak clearly
        3. When you have enough frames, click **'Run Lip Reading'** to see the prediction
        4. Click **'Clear Buffer'** to start over
        """)

        init_status = gr.Textbox(label="Model Status", value="Click below to load model", interactive=False)
        load_btn = gr.Button("🔄 Load Model", variant="primary", size="lg")

        gr.Markdown("---")

        with gr.Row():
            with gr.Column(scale=2):
                webcam = gr.Image(label="📷 Webcam", sources=["webcam"], type="numpy", streaming=False)
                capture_btn = gr.Button("📸 Capture Frame", variant="primary", size="lg")
                preview = gr.Image(label="👁️ Preview (with detection boxes)", type="numpy")

            with gr.Column(scale=1):
                gr.Markdown("### 🎯 Actions")
                run_btn = gr.Button("🚀 Run Lip Reading", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Clear Buffer", variant="secondary")

                gr.Markdown("### 📊 Status")
                status = gr.Textbox(label="Processing Status", lines=3, interactive=False, value="Waiting...")

        gr.Markdown("---")
        caption = gr.HTML(label="📝 Predicted Caption", value="<h2 style='color: #666; text-align: center;'>Captions will appear here</h2>")
        confidence = gr.Textbox(label="🎯 Model Confidence", interactive=False, value="0%")

        buffer_status = gr.Textbox(label="Frame Buffer", value="Buffer: 0 frames", interactive=False)

        # Event handlers
        load_btn.click(fn=init_model, outputs=[init_status])
        capture_btn.click(
            fn=process_webcam_snapshot,
            inputs=[webcam],
            outputs=[preview, buffer_status, status]
        )
        run_btn.click(
            fn=run_inference,
            outputs=[caption, confidence, status]
        )
        clear_btn.click(
            fn=clear_buffer,
            outputs=[buffer_status, status]
        )

    return app


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  LipSync Gradio UI - Manual Capture Mode")
    print("="*70)
    print("\nStarting server on http://localhost:7862")
    print("The model will load when you click 'Load Model' button\n")

    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=7862, share=False)
