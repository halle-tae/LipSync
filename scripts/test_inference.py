#!/usr/bin/env python3
"""
Test Inference Script
=====================
Runs the Auto-AVSR model on a test video and prints the predicted text.

This is the Phase 0 validation script — confirms the model works end-to-end.

Usage:
    python scripts/test_inference.py                          # Use default test video
    python scripts/test_inference.py --video path/to/video.mp4  # Use custom video
    python scripts/test_inference.py --weights path/to/model.pth # Use specific weights
    python scripts/test_inference.py --verbose                  # Enable debug logging

Example output:
    ============================================================
      LipSync — Test Inference
    ============================================================

    [1] Loading inference pipeline...
      ✓ Model loaded (250.0M parameters)
      ✓ Device: cpu
    [2] Running inference on tests/test_video.mp4...
      ✓ Frames: 75
      ✓ Prediction: "HELLO HOW ARE YOU DOING TODAY"
      ✓ Total latency: 2340ms
    [3] Latency breakdown:
      • Load video:     120ms
      • Face detection:  890ms
      • Mouth crop:      45ms
      • Preprocessing:   12ms
      • Model inference: 1273ms
    ============================================================
      ✓ Phase 0 inference test PASSED
    ============================================================
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_header(text: str):
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}\n")


def print_step(step: int, text: str):
    print(f"  [{step}] {text}")


def print_ok(text: str):
    print(f"  [OK] {text}")


def print_error(text: str):
    print(f"  [FAIL] {text}")


def main():
    parser = argparse.ArgumentParser(description="Test Auto-AVSR inference")
    parser.add_argument(
        "--video",
        type=str,
        default=str(PROJECT_ROOT / "tests" / "test_video.mp4"),
        help="Path to test video file",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to model weights (.pth file)",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="mediapipe",
        choices=["mediapipe", "retinaface"],
        help="Face detector to use (default: mediapipe)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", "mps"],
        help="Device to run on (auto-detected if not specified)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print_header("LipSync — Test Inference")

    # Check video file exists
    video_path = Path(args.video)
    if not video_path.exists():
        print_error(f"Video file not found: {video_path}")
        print(f"\n  To download the demo video:")
        print(f"  python scripts/setup.py --download-demo")
        print(f"\n  Or specify a different video:")
        print(f"  python scripts/test_inference.py --video path/to/video.mp4")
        sys.exit(1)

    # Load pipeline
    print_step(1, "Loading inference pipeline...")
    t0 = time.time()

    try:
        from backend.model.inference import InferencePipeline

        pipeline = InferencePipeline(
            weights_path=args.weights,
            device=args.device,
            detector=args.detector,
        )
        pipeline.load()
    except FileNotFoundError as e:
        print_error(str(e))
        sys.exit(1)
    except RuntimeError as e:
        print_error(str(e))
        sys.exit(1)
    except Exception as e:
        print_error(f"Failed to load pipeline: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    load_time = (time.time() - t0) * 1000
    num_params = sum(p.numel() for p in pipeline.model_module.parameters())
    print_ok(f"Model loaded ({num_params / 1e6:.1f}M parameters) in {load_time:.0f}ms")
    print_ok(f"Device: {pipeline.device}")

    # Run inference
    print_step(2, f"Running inference on {video_path}...")

    try:
        result = pipeline.predict_from_file(str(video_path))
    except Exception as e:
        print_error(f"Inference failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    if "error" in result:
        print_error(f"Inference error: {result['error']}")
        sys.exit(1)

    # Print results
    print_ok(f"Frames: {result['num_frames']}")
    print_ok(f'Prediction: "{result["text"]}"')
    print_ok(f"Total latency: {result['latency_ms']:.0f}ms")

    # Latency breakdown
    print_step(3, "Latency breakdown:")
    breakdown = result.get("latency_breakdown", {})
    labels = {
        "load_video_ms": "Load video",
        "face_detection_ms": "Face detection",
        "mouth_crop_ms": "Mouth crop",
        "preprocess_ms": "Preprocessing",
        "inference_ms": "Model inference",
    }
    for key, label in labels.items():
        if key in breakdown:
            print(f"  • {label + ':':20s} {breakdown[key]:.0f}ms")

    # Calculate FPS equivalent
    if result["num_frames"] > 0:
        fps = result["num_frames"] / (result["latency_ms"] / 1000)
        print(f"\n  Processing speed: {fps:.1f} frames/sec")

    # Summary
    if result["text"]:
        print_header("Phase 0 inference test PASSED")
        print(f'  Predicted text: "{result["text"]}"\n')
    else:
        print_header("WARNING: Inference produced empty output")
        print("  This may happen if no speech is detected in the video.")
        print("  Try with a different video containing clear frontal speech.\n")

    # Cleanup
    pipeline.unload()


if __name__ == "__main__":
    main()
