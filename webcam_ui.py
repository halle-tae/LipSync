#!/usr/bin/env python3
"""Simple OpenCV-based webcam UI for LipSync - guaranteed to work!"""

import sys
import time
from pathlib import Path
from collections import deque
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.capture.face_detect import FaceMeshMouthExtractor
from backend.model.inference import InferencePipeline
from backend.processing.preprocess import MouthPreprocessor

print("\n" + "="*70)
print("  LipSync - OpenCV Webcam UI")
print("="*70)

# Initialize components
print("\n[1/3] Loading model...")
pipeline = InferencePipeline()
pipeline.load()

print("[2/3] Initializing face detector...")
extractor = FaceMeshMouthExtractor(roi_size=96)
preprocessor = MouthPreprocessor(roi_size=96, output_size=88)

print("[3/3] Opening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam!")
    sys.exit(1)

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# State
frame_buffer = deque(maxlen=75)
last_prediction = ""
last_confidence = 0.0
last_infer_time = 0
capturing = False
infer_interval = 2.0  # seconds between inferences

print("\n" + "="*70)
print("  CONTROLS")
print("="*70)
print("  SPACE    - Start/Stop capturing frames")
print("  R        - Run inference on buffered frames")
print("  C        - Clear buffer")
print("  Q / ESC  - Quit")
print("="*70)
print("\nPress SPACE to start capturing, then speak a phrase slowly.")
print("When you have ~50 frames, press R to run lip reading.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Create display frame
    display = frame.copy()
    h, w = display.shape[:2]

    # Detect face
    detection = extractor.extract(frame)

    # Draw detection boxes
    if detection is not None:
        fx1, fy1, fx2, fy2 = detection.face_bbox
        mx1, my1, mx2, my2 = detection.mouth_bbox

        # Draw boxes
        cv2.rectangle(display, (fx1, fy1), (fx2, fy2), (64, 220, 80), 2)
        cv2.rectangle(display, (mx1, my1), (mx2, my2), (255, 128, 64), 2)

        # Add to buffer if capturing
        if capturing:
            frame_buffer.append(detection.mouth_roi_bgr)
    else:
        cv2.putText(display, "NO FACE DETECTED", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Draw status panel at top
    status_bg = np.zeros((150, w, 3), dtype=np.uint8)

    # Capturing status
    if capturing:
        status_text = "CAPTURING - Keep speaking!"
        status_color = (0, 255, 0)
    else:
        status_text = "READY - Press SPACE to start"
        status_color = (128, 128, 128)

    cv2.putText(status_bg, status_text, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    # Buffer info
    buffer_text = f"Buffer: {len(frame_buffer)} frames (need 25+)"
    cv2.putText(status_bg, buffer_text, (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Last prediction
    if last_prediction:
        pred_text = f'Prediction: "{last_prediction}"'
        conf_color = (0, 255, 0) if last_confidence > 0.75 else (0, 200, 255) if last_confidence > 0.45 else (0, 0, 255)
        cv2.putText(status_bg, pred_text, (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(status_bg, f"Confidence: {last_confidence*100:.1f}%", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2)

    # Combine status and video
    combined = np.vstack([status_bg, display])

    cv2.imshow('LipSync - Press SPACE to capture, R to read lips, Q to quit', combined)

    # Handle keyboard
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == 27:  # Q or ESC
        break

    elif key == ord(' '):  # SPACE - toggle capturing
        capturing = not capturing
        if capturing:
            print(f"\n▶ Started capturing frames (buffer: {len(frame_buffer)})")
        else:
            print(f"⏸ Stopped capturing (buffer: {len(frame_buffer)})")

    elif key == ord('c'):  # C - clear buffer
        frame_buffer.clear()
        last_prediction = ""
        last_confidence = 0.0
        print("\n🗑 Buffer cleared")

    elif key == ord('r'):  # R - run inference
        if len(frame_buffer) < 25:
            print(f"\n❌ Need at least 25 frames (you have {len(frame_buffer)})")
        else:
            print(f"\n🚀 Running inference on {len(frame_buffer)} frames...")

            t0 = time.time()
            frames_list = list(frame_buffer)[-50:]  # Use last 50 frames

            # Preprocess
            input_tensor = preprocessor.process_frames(frames_list)
            preprocess_ms = (time.time() - t0) * 1000

            # Inference
            t1 = time.time()
            pred = pipeline.predict_from_frames_detailed(input_tensor)
            infer_ms = (time.time() - t1) * 1000

            total_ms = (time.time() - t0) * 1000

            last_prediction = pred["text"]
            last_confidence = float(pred["confidence"])

            print(f"✓ Prediction: \"{last_prediction}\"")
            print(f"  Confidence: {last_confidence*100:.1f}%")
            print(f"  Time: preprocess={preprocess_ms:.1f}ms, infer={infer_ms:.1f}ms, total={total_ms:.1f}ms")

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*70)
print("  LipSync closed")
print("="*70)
