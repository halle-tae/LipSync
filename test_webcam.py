#!/usr/bin/env python3
"""Quick webcam test to verify face detection is working."""

import cv2
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.capture.face_detect import FaceMeshMouthExtractor

print("Opening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam")
    sys.exit(1)

print("Initializing face detector...")
extractor = FaceMeshMouthExtractor(roi_size=96)

print("\nWebcam test running!")
print("- Green box = face detected")
print("- Blue box = mouth region")
print("Press 'q' to quit\n")

frame_count = 0
detection_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_count += 1

    # Run detection
    detection = extractor.extract(frame)

    # Draw boxes
    if detection is not None:
        detection_count += 1
        fx1, fy1, fx2, fy2 = detection.face_bbox
        mx1, my1, mx2, my2 = detection.mouth_bbox

        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (64, 220, 80), 2)
        cv2.rectangle(frame, (mx1, my1), (mx2, my2), (64, 128, 255), 2)
        cv2.putText(frame, f"Face detected! ({detection_count}/{frame_count})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No face detected",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(frame, "Press 'q' to quit",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('LipSync Webcam Test', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nTest complete!")
print(f"Frames processed: {frame_count}")
print(f"Face detected in: {detection_count} frames ({detection_count/max(1,frame_count)*100:.1f}%)")
