"""
LipSync — Phase 1 Pipeline Validation Script

Runs end-to-end checks on the data pipeline to verify:
1. Face Mesh processor initialises correctly
2. Lip ROI extraction produces correct output shapes
3. Alignment parsing works
4. Data loader outputs correct shapes and label alignment
5. Model architecture builds with the expected input/output shapes

Usage:
    python -m backend.validate_pipeline
"""

import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("validate")

PASS = "✅"
FAIL = "❌"
SKIP = "⏭️ "
results = []


def check(name: str, passed: bool, detail: str = ""):
    status = PASS if passed else FAIL
    results.append((name, passed))
    msg = f"  {status} {name}"
    if detail:
        msg += f" — {detail}"
    logger.info(msg)
    return passed


def main():
    logger.info("=" * 60)
    logger.info("LipSync — Phase 1 Pipeline Validation")
    logger.info("=" * 60)

    project_root = Path(__file__).parent.parent
    data_dir = project_root / "backend" / "data"
    processed_dir = data_dir / "processed"

    # ------------------------------------------------------------------
    # 1. Face Mesh Processor
    # ------------------------------------------------------------------
    logger.info("\n1. Face Mesh Processor")
    try:
        from backend.utils.face_mesh import FaceMeshProcessor, FaceDetection

        processor = FaceMeshProcessor(static_image_mode=True)
        check("FaceMeshProcessor initialises", True)

        # Test with a synthetic image (blank frame)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detection = processor.process_frame(dummy_frame)
        check(
            "process_frame returns FaceDetection",
            isinstance(detection, FaceDetection),
            f"type={type(detection).__name__}",
        )
        check(
            "No face in blank frame",
            detection.lip_roi is None,
            "lip_roi is None as expected",
        )

        # Test with a synthetic face-like image (won't detect, but shouldn't crash)
        noisy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detection2 = processor.process_frame(noisy)
        check(
            "process_frame handles noisy input",
            isinstance(detection2, FaceDetection),
        )

        processor.close()
        check("FaceMeshProcessor closes cleanly", True)
    except Exception as e:
        check("FaceMeshProcessor", False, str(e))

    # ------------------------------------------------------------------
    # 2. Lip ROI Extraction Constants
    # ------------------------------------------------------------------
    logger.info("\n2. Lip Extractor")
    try:
        from backend.utils.lip_extractor import (
            ROI_WIDTH,
            ROI_HEIGHT,
            MAX_FRAMES,
            parse_alignment,
        )

        check("ROI dimensions", ROI_WIDTH == 100 and ROI_HEIGHT == 50,
              f"{ROI_WIDTH}x{ROI_HEIGHT}")
        check("MAX_FRAMES", MAX_FRAMES == 75, f"{MAX_FRAMES}")
    except Exception as e:
        check("Lip extractor imports", False, str(e))

    # ------------------------------------------------------------------
    # 3. Alignment Parsing
    # ------------------------------------------------------------------
    logger.info("\n3. Alignment Parsing")
    try:
        from backend.utils.lip_extractor import parse_alignment

        # Create a test alignment file
        test_align = data_dir / "_test_validate.align"
        test_align.write_text(
            "0 5000 sil\n"
            "5000 10000 bin\n"
            "10000 15000 blue\n"
            "15000 20000 sp\n"
            "20000 25000 at\n"
            "25000 30000 f\n"
            "30000 35000 two\n"
            "35000 40000 now\n"
            "40000 45000 sil\n"
        )

        label = parse_alignment(test_align)
        check(
            "parse_alignment extracts words",
            label == "bin blue at f two now",
            f"'{label}'",
        )

        # Clean up
        test_align.unlink()
    except Exception as e:
        check("Alignment parsing", False, str(e))

    # ------------------------------------------------------------------
    # 4. Text Encoding/Decoding
    # ------------------------------------------------------------------
    logger.info("\n4. Text Encoding / Decoding")
    try:
        from backend.model.train import (
            encode_text,
            decode_predictions,
            CHAR_TO_IDX,
            IDX_TO_CHAR,
            NUM_CLASSES,
        )

        # Encode
        encoded = encode_text("bin blue at f two now")
        check(
            "encode_text produces int array",
            encoded.dtype == np.int32 and encoded.shape == (40,),
            f"shape={encoded.shape}, dtype={encoded.dtype}",
        )
        check(
            "Encoded values are valid",
            all(0 <= v <= NUM_CLASSES for v in encoded),
        )

        # Round-trip decode via one-hot
        # CTC greedy decode collapses consecutive identical characters,
        # so we insert a blank (idx=0) between repeated chars.
        text = "hello"
        # CTC encoding: h, e, l, <blank>, l, o (to keep both l's)
        ctc_indices = [
            CHAR_TO_IDX["h"],
            CHAR_TO_IDX["e"],
            CHAR_TO_IDX["l"],
            0,  # blank to separate repeated l's
            CHAR_TO_IDX["l"],
            CHAR_TO_IDX["o"],
        ]
        seq_len = 20
        fake_pred = np.zeros((seq_len, NUM_CLASSES), dtype=np.float32)
        for i, idx in enumerate(ctc_indices):
            fake_pred[i, idx] = 1.0
        # Fill remaining with blank
        for i in range(len(ctc_indices), seq_len):
            fake_pred[i, 0] = 1.0

        decoded = decode_predictions(fake_pred)
        check("decode_predictions round-trip", decoded == text, f"'{decoded}'")

    except Exception as e:
        check("Text encoding/decoding", False, str(e))

    # ------------------------------------------------------------------
    # 5. Model Architecture
    # ------------------------------------------------------------------
    logger.info("\n5. Model Architecture")
    try:
        from backend.model.train import build_lip_reading_model, MAX_FRAMES as MF

        model = build_lip_reading_model()
        check("Model builds successfully", True, model.name)

        expected_input = (None, 75, 50, 100, 1)
        actual_input = tuple(model.input_shape)
        check(
            "Input shape matches",
            actual_input == expected_input,
            f"expected {expected_input}, got {actual_input}",
        )

        # Output should be (None, time_steps, num_classes)
        out_shape = model.output_shape
        check(
            "Output shape is (batch, time, classes)",
            len(out_shape) == 3 and out_shape[-1] > 1,
            f"{out_shape}",
        )

        # Test forward pass with dummy data
        dummy = np.random.rand(1, 75, 50, 100, 1).astype(np.float32)
        pred = model.predict(dummy, verbose=0)
        check(
            "Forward pass works",
            pred.shape[0] == 1 and pred.shape[-1] == out_shape[-1],
            f"output shape: {pred.shape}",
        )

        # Check parameter count
        params = model.count_params()
        check(
            "Model parameter count reasonable",
            100_000 < params < 50_000_000,
            f"{params:,} parameters",
        )

    except Exception as e:
        check("Model architecture", False, str(e))

    # ------------------------------------------------------------------
    # 6. Data Loader (only if processed data exists)
    # ------------------------------------------------------------------
    logger.info("\n6. Data Loader")
    if processed_dir.exists() and list(processed_dir.rglob("*_frames.npy")):
        try:
            from backend.utils.data_loader import create_dataset

            train_ds, val_ds, n_train, n_val = create_dataset(
                str(processed_dir), batch_size=2
            )
            check("create_dataset succeeds", True, f"{n_train} train, {n_val} val")

            for frames, labels in train_ds.take(1):
                check(
                    "Batch frame shape",
                    frames.shape[1:] == (75, 50, 100, 1),
                    f"{frames.shape}",
                )
                check(
                    "Batch label shape",
                    labels.shape[1] == 40,
                    f"{labels.shape}",
                )
                check(
                    "Frame values normalised",
                    0.0 <= frames.numpy().min() and frames.numpy().max() <= 1.0,
                    f"[{frames.numpy().min():.3f}, {frames.numpy().max():.3f}]",
                )
        except Exception as e:
            check("Data loader", False, str(e))
    else:
        logger.info(f"  {SKIP} Skipped — no processed data found at {processed_dir}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    passed = sum(1 for _, p in results if p)
    total = len(results)
    logger.info(f"Results: {passed}/{total} checks passed")

    if passed == total:
        logger.info(f"\n{PASS} All Phase 1 pipeline checks passed!")
    else:
        failed = [name for name, p in results if not p]
        logger.info(f"\n{FAIL} Failed checks:")
        for name in failed:
            logger.info(f"    - {name}")

    logger.info("=" * 60)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

