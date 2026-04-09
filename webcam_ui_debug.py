#!/usr/bin/env python3
"""Debug version that saves frames and shows detailed info."""

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
print("  LipSync - DEBUG MODE")
print("="*70)

# Create debug output directory
debug_dir = Path("debug_output")
debug_dir.mkdir(exist_ok=True)
print(f"\nDebug outputs will be saved to: {debug_dir.absolute()}")

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

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# State
frame_buffer = deque(maxlen=75)
mouth_rois = deque(maxlen=75)  # Store mouth ROIs for debugging
last_prediction = ""
last_confidence = 0.0
capturing = False
frame_count = 0

print("\n" + "="*70)
print("  CONTROLS")
print("="*70)
print("  SPACE    - Start/Stop capturing frames")
print("  R        - Run inference + save debug info")
print("  S        - Save current mouth ROI to disk")
print("  C        - Clear buffer")
print("  Q / ESC  - Quit")
print("="*70)
print("\nDEBUG FEATURES:")
print("- Shows real-time confidence scores")
print("- Saves mouth ROI images when you press S")
print("- Saves all frames and preprocessed data when you press R")
print("- Shows detailed preprocessing info")
print()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    h, w = display.shape[:2]

    # Detect face
    detection = extractor.extract(frame)
    detection_good = detection is not None

    # Draw detection
    if detection is not None:
        fx1, fy1, fx2, fy2 = detection.face_bbox
        mx1, my1, mx2, my2 = detection.mouth_bbox

        cv2.rectangle(display, (fx1, fy1), (fx2, fy2), (64, 220, 80), 2)
        cv2.rectangle(display, (mx1, my1), (mx2, my2), (255, 128, 64), 2)

        # Show mouth ROI in corner
        mouth_roi = detection.mouth_roi_bgr
        roi_display = cv2.resize(mouth_roi, (150, 150))
        display[10:160, w-160:w-10] = roi_display

        if capturing:
            frame_buffer.append(detection.mouth_roi_bgr)
            mouth_rois.append(detection.mouth_roi_bgr.copy())
    else:
        cv2.putText(display, "NO FACE DETECTED", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Status panel
    status_bg = np.zeros((180, w, 3), dtype=np.uint8)

    # Capturing status
    if capturing:
        status = f"CAPTURING - Frames: {len(frame_buffer)}"
        color = (0, 255, 0)
    else:
        status = "READY - Press SPACE to start"
        color = (128, 128, 128)

    cv2.putText(status_bg, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(status_bg, f"Detection: {'OK' if detection_good else 'FAIL'}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if detection_good else (0,0,255), 2)
    cv2.putText(status_bg, f"Buffer: {len(frame_buffer)} frames", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Last prediction
    if last_prediction:
        conf_color = (0, 255, 0) if last_confidence > 0.75 else (0, 200, 255) if last_confidence > 0.45 else (0, 0, 255)
        cv2.putText(status_bg, f'Pred: "{last_prediction}"', (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(status_bg, f"Confidence: {last_confidence*100:.1f}%", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_color, 2)

    combined = np.vstack([status_bg, display])
    cv2.imshow('LipSync DEBUG - SPACE=capture, R=infer+save, S=save ROI, Q=quit', combined)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == 27:
        break

    elif key == ord(' '):
        capturing = not capturing
        if capturing:
            print(f"\n▶ Started capturing (buffer: {len(frame_buffer)})")
        else:
            print(f"⏸ Stopped capturing (buffer: {len(frame_buffer)})")

    elif key == ord('s'):  # Save current mouth ROI
        if detection is not None:
            timestamp = int(time.time() * 1000)
            roi_path = debug_dir / f"mouth_roi_{timestamp}.jpg"
            cv2.imwrite(str(roi_path), detection.mouth_roi_bgr)
            print(f"\n💾 Saved mouth ROI to: {roi_path}")
        else:
            print("\n❌ No face detected, can't save ROI")

    elif key == ord('c'):
        frame_buffer.clear()
        mouth_rois.clear()
        last_prediction = ""
        last_confidence = 0.0
        print("\n🗑 Buffer cleared")

    elif key == ord('r'):  # Run inference with debug output
        if len(frame_buffer) < 25:
            print(f"\n❌ Need at least 25 frames (you have {len(frame_buffer)})")
        else:
            print(f"\n{'='*60}")
            print(f"🚀 RUNNING INFERENCE WITH DEBUG OUTPUT")
            print(f"{'='*60}")

            timestamp = int(time.time() * 1000)
            session_dir = debug_dir / f"session_{timestamp}"
            session_dir.mkdir(exist_ok=True)

            frames_list = list(frame_buffer)[-50:]
            print(f"\n1. Using last {len(frames_list)} frames from buffer")

            # Save all mouth ROIs
            print(f"2. Saving {len(frames_list)} mouth ROI images...")
            for i, roi in enumerate(frames_list):
                cv2.imwrite(str(session_dir / f"roi_{i:03d}.jpg"), roi)

            # Preprocess
            print(f"3. Preprocessing frames...")
            t0 = time.time()
            input_tensor = preprocessor.process_frames(frames_list)
            preprocess_ms = (time.time() - t0) * 1000
            print(f"   - Tensor shape: {input_tensor.shape}")
            print(f"   - Dtype: {input_tensor.dtype}")
            print(f"   - Time: {preprocess_ms:.1f}ms")

            # Save first preprocessed frame as image
            first_frame = input_tensor[0, 0].numpy()  # [H, W]
            first_frame_uint8 = ((first_frame * 0.165 + 0.421) * 255).astype(np.uint8)
            cv2.imwrite(str(session_dir / "preprocessed_frame_0.jpg"), first_frame_uint8)

            # Inference
            print(f"4. Running model inference...")
            t1 = time.time()
            pred = pipeline.predict_from_frames_detailed(input_tensor)
            infer_ms = (time.time() - t1) * 1000

            last_prediction = pred["text"]
            last_confidence = float(pred["confidence"])

            print(f"\n{'='*60}")
            print(f"RESULTS:")
            print(f"{'='*60}")
            print(f"Prediction:  \"{last_prediction}\"")
            print(f"Confidence:  {last_confidence*100:.1f}%")
            print(f"Preprocess:  {preprocess_ms:.1f}ms")
            print(f"Inference:   {infer_ms:.1f}ms")
            print(f"Total:       {(preprocess_ms + infer_ms):.1f}ms")
            print(f"\nDebug files saved to: {session_dir.absolute()}")
            print(f"{'='*60}\n")

            # Save results
            with open(session_dir / "results.txt", "w") as f:
                f.write(f"Prediction: {last_prediction}\n")
                f.write(f"Confidence: {last_confidence*100:.1f}%\n")
                f.write(f"Frames used: {len(frames_list)}\n")
                f.write(f"Tensor shape: {input_tensor.shape}\n")
                f.write(f"Preprocess time: {preprocess_ms:.1f}ms\n")
                f.write(f"Inference time: {infer_ms:.1f}ms\n")

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*70)
print(f"  Debug outputs saved to: {debug_dir.absolute()}")
print("="*70)
