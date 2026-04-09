#!/usr/bin/env python3
"""Main entry point for LipSync video-file and real-time webcam inference."""

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _print_banner():
    print()
    print("=" * 50)
    print("  LipSync - Real-Time Lip Reading Assistant")
    print("=" * 50)
    print()


def _run_video_mode(args, logger):
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        logger.error("To download a test video: python scripts/setup.py --download-demo")
        sys.exit(1)

    from backend.model.inference import InferencePipeline

    logger.info("Loading model...")
    pipeline = InferencePipeline(
        weights_path=args.weights,
        device=args.device,
        detector=args.detector,
    )

    t0 = time.time()
    pipeline.load()
    logger.info(f"Model loaded in {(time.time() - t0) * 1000:.0f}ms")

    logger.info(f"Running inference on: {video_path}")
    result = pipeline.predict_from_file(str(video_path))

    print()
    print("-" * 50)
    if "error" in result:
        print(f"  Error: {result['error']}")
    else:
        print(f"  Prediction: \"{result['text']}\"")
        print(f"  Latency:    {result['latency_ms']:.0f}ms")
        print(f"  Frames:     {result['num_frames']}")
        if result["num_frames"] > 0:
            fps = result["num_frames"] / (result["latency_ms"] / 1000)
            print(f"  Speed:      {fps:.1f} frames/sec")
    print("-" * 50)
    print()

    pipeline.unload()


def _run_webcam_mode(args, logger):
    try:
        cv2 = __import__("cv2")
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for webcam mode. Install with: pip install opencv-python") from exc

    from backend.capture import FaceMeshMouthExtractor, WebcamCapture
    from backend.model.inference import InferencePipeline
    from backend.processing import MouthPreprocessor, SlidingFrameBuffer

    target_fps = max(1, int(args.fps))
    window_frames = max(1, int(args.window_seconds * target_fps))
    infer_interval = max(0.2, float(args.infer_interval_seconds))
    show_preview = not args.no_preview

    pipeline = InferencePipeline(
        weights_path=args.weights,
        device=args.device,
        detector=args.detector,
    )
    webcam = WebcamCapture(
        camera_index=args.camera_index,
        target_fps=target_fps,
        width=args.camera_width,
        height=args.camera_height,
    )
    extractor = FaceMeshMouthExtractor(roi_size=args.roi_size)
    preprocessor = MouthPreprocessor(roi_size=args.roi_size, output_size=88)
    frame_buffer = SlidingFrameBuffer(max_frames=window_frames)

    logger.info("Loading model...")
    t0 = time.time()
    pipeline.load()
    logger.info(f"Model loaded in {(time.time() - t0) * 1000:.0f}ms")

    info = webcam.open()
    logger.info(
        "Webcam opened: index=%s resolution=%sx%s fps(target=%.1f, reported=%.1f)",
        info.camera_index,
        info.width,
        info.height,
        info.fps_configured,
        info.fps_reported,
    )

    logger.info(
        "Starting webcam inference: window=%d frames (%.1fs), interval=%.1fs",
        window_frames,
        args.window_seconds,
        infer_interval,
    )
    if show_preview:
        logger.info("Press 'q' or ESC in the preview window to stop.")
    else:
        logger.info("Running without preview. Use Ctrl+C to stop.")

    last_infer_ts = 0.0
    last_capture_ts = 0.0
    last_text = ""
    recent_detection_ms = []

    try:
        while True:
            frame, capture_ts = webcam.read()

            if last_capture_ts > 0:
                capture_latency_ms = (capture_ts - last_capture_ts) * 1000
            else:
                capture_latency_ms = 0.0
            last_capture_ts = capture_ts

            det_start = time.time()
            detection = extractor.extract(frame)
            det_ms = (time.time() - det_start) * 1000
            recent_detection_ms.append(det_ms)
            if len(recent_detection_ms) > window_frames:
                recent_detection_ms = recent_detection_ms[-window_frames:]

            preview_frame = frame
            if detection is not None:
                frame_buffer.append(detection.mouth_roi_bgr, capture_ts)

                if show_preview:
                    fx1, fy1, fx2, fy2 = detection.face_bbox
                    mx1, my1, mx2, my2 = detection.mouth_bbox
                    preview_frame = frame.copy()
                    cv2.rectangle(preview_frame, (fx1, fy1), (fx2, fy2), (80, 180, 80), 2)
                    cv2.rectangle(preview_frame, (mx1, my1), (mx2, my2), (80, 80, 220), 2)
                    cv2.putText(
                        preview_frame,
                        f"faces={detection.num_faces} buffered={len(frame_buffer)}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
            elif show_preview:
                preview_frame = frame.copy()
                cv2.putText(
                    preview_frame,
                    "Looking for a speaker...",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (60, 60, 255),
                    2,
                    cv2.LINE_AA,
                )

            now = time.time()
            if now - last_infer_ts >= infer_interval and len(frame_buffer) >= max(12, window_frames // 2):
                infer_total_start = time.time()

                preprocess_start = time.time()
                mouth_frames = frame_buffer.get_latest(window_frames)
                input_tensor = preprocessor.process_frames(mouth_frames)
                preprocess_ms = (time.time() - preprocess_start) * 1000

                prediction = pipeline.predict_from_frames(input_tensor)
                infer_ms = float(prediction["latency_ms"])
                total_ms = (time.time() - infer_total_start) * 1000
                avg_det_ms = sum(recent_detection_ms) / max(1, len(recent_detection_ms))

                last_text = prediction["text"]
                timestamp = time.strftime("%H:%M:%S")
                print(
                    f"[{timestamp}] text=\"{last_text}\" | "
                    f"capture={capture_latency_ms:.1f}ms detect={avg_det_ms:.1f}ms "
                    f"preprocess={preprocess_ms:.1f}ms inference+decode={infer_ms:.1f}ms "
                    f"total={total_ms:.1f}ms frames={len(mouth_frames)}"
                )
                last_infer_ts = now

            if show_preview:
                if last_text:
                    cv2.putText(
                        preview_frame,
                        f"Text: {last_text}",
                        (10, max(40, preview_frame.shape[0] - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.imshow("LipSync Webcam", preview_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        logger.info("Stopping webcam inference...")
    finally:
        webcam.release()
        extractor.close()
        pipeline.unload()
        if show_preview:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="LipSync — Real-Time Lip Reading Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Run on test video
  python main.py --video my_video.mp4               # Run on custom video
  python main.py --webcam                            # Run real-time webcam mode
  python main.py --device cpu                       # Force CPU inference
  python main.py --verbose                          # Enable debug logging
        """,
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Run real-time webcam inference mode",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=str(PROJECT_ROOT / "tests" / "test_video.mp4"),
        help="Path to video file for inference (default: tests/test_video.mp4)",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam index for OpenCV VideoCapture (default: 0)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Target webcam capture FPS (default: 25)",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=2.0,
        help="Sliding window length in seconds for model context (default: 2.0)",
    )
    parser.add_argument(
        "--infer-interval-seconds",
        type=float,
        default=1.5,
        help="Inference interval in seconds for webcam mode (default: 1.5)",
    )
    parser.add_argument(
        "--roi-size",
        type=int,
        default=96,
        help="Mouth ROI size before center-crop preprocessing (default: 96)",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=None,
        help="Optional webcam capture width",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=None,
        help="Optional webcam capture height",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable OpenCV preview window (Ctrl+C to stop)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to model weights (.pth file)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", "mps"],
        help="Device to run on (auto-detected if not specified)",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="mediapipe",
        choices=["mediapipe", "retinaface"],
        help="Detector used by Auto-AVSR internals for model loading context",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s" if args.verbose else "%(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("lipsync")

    _print_banner()
    if args.webcam:
        _run_webcam_mode(args, logger)
    else:
        _run_video_mode(args, logger)


if __name__ == "__main__":
    main()
