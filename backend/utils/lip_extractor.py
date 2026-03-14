"""
LipSync — Lip ROI Extraction Pipeline

Processes raw video files from the GRID corpus, extracts lip ROI frame sequences
using MediaPipe Face Mesh, and saves them as NumPy arrays paired with text
alignment labels.

Usage:
    python -m backend.utils.lip_extractor \
        --video_dir backend/data/raw \
        --align_dir backend/data/alignments \
        --output_dir backend/data/processed
"""

import argparse
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from backend.utils.face_mesh import FaceMeshProcessor

logger = logging.getLogger("lipsync.lip_extractor")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROI_WIDTH = 100
ROI_HEIGHT = 50
MAX_FRAMES = 75  # Target frame count per sequence


# ---------------------------------------------------------------------------
# Alignment Parsing
# ---------------------------------------------------------------------------
def parse_alignment(align_path: str | Path) -> str:
    """
    Parse a GRID corpus alignment (.align) file and return the spoken text.

    GRID alignment format (each line):
        <start_frame> <end_frame> <word>
    Words "sil" and "sp" are silence/short-pause markers and are skipped.

    Args:
        align_path: Path to the .align file.

    Returns:
        The spoken sentence as a lowercase string.
    """
    words = []
    with open(align_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                word = parts[2].lower()
                if word not in ("sil", "sp"):
                    words.append(word)
    return " ".join(words)


# ---------------------------------------------------------------------------
# Video → Lip ROI Frames
# ---------------------------------------------------------------------------
def extract_lip_frames(
    video_path: str | Path,
    processor: FaceMeshProcessor,
    max_frames: int = MAX_FRAMES,
) -> np.ndarray | None:
    """
    Extract a sequence of lip ROI frames from a video file.

    Args:
        video_path: Path to the video (.mpg, .avi, .mp4, etc.).
        processor: Initialised FaceMeshProcessor instance.
        max_frames: Maximum number of frames to keep.

    Returns:
        NumPy array of shape (num_frames, ROI_HEIGHT, ROI_WIDTH) with
        dtype float32 normalised to [0, 1], or None if extraction failed.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path)
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detection = processor.process_frame(frame)
        if detection.lip_roi is not None:
            # Normalise to [0, 1] float32
            roi = detection.lip_roi.astype(np.float32) / 255.0
            frames.append(roi)

    cap.release()

    if len(frames) == 0:
        logger.warning("No lip ROIs extracted from %s", video_path)
        return None

    # Pad / truncate to max_frames
    frames_arr = np.array(frames, dtype=np.float32)
    num = frames_arr.shape[0]

    if num < max_frames:
        pad = np.zeros((max_frames - num, ROI_HEIGHT, ROI_WIDTH), dtype=np.float32)
        frames_arr = np.concatenate([frames_arr, pad], axis=0)
    elif num > max_frames:
        frames_arr = frames_arr[:max_frames]

    return frames_arr


# ---------------------------------------------------------------------------
# Batch Processing
# ---------------------------------------------------------------------------
def process_dataset(
    video_dir: str | Path,
    align_dir: str | Path,
    output_dir: str | Path,
    max_frames: int = MAX_FRAMES,
):
    """
    Process all videos in the GRID corpus directory structure.

    Expected structure:
        video_dir/
            s1/          # Speaker 1
                bbaf2n.mpg
                bbaf3s.mpg
                ...
            s2/
                ...
        align_dir/
            s1/
                bbaf2n.align
                bbaf3s.align
                ...
            s2/
                ...

    Output structure:
        output_dir/
            s1/
                bbaf2n_frames.npy    # (75, 50, 100) float32
                bbaf2n_label.txt     # "bin blue at f two now"
                ...

    Args:
        video_dir: Root of raw video files.
        align_dir: Root of alignment files.
        output_dir: Where to write processed data.
        max_frames: Max frames per sequence.
    """
    video_dir = Path(video_dir)
    align_dir = Path(align_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover all video files
    video_files = sorted(video_dir.rglob("*.mpg")) + sorted(video_dir.rglob("*.avi"))
    if not video_files:
        video_files = sorted(video_dir.rglob("*.mp4"))

    if not video_files:
        logger.error("No video files found in %s", video_dir)
        return

    logger.info("Found %d video files to process.", len(video_files))
    start_time = time.time()
    success_count = 0
    fail_count = 0

    with FaceMeshProcessor(static_image_mode=True) as processor:
        for vpath in tqdm(video_files, desc="Extracting lip ROIs"):
            # Determine speaker folder and utterance ID
            speaker = vpath.parent.name
            utterance_id = vpath.stem

            # Find matching alignment file
            align_path = align_dir / speaker / f"{utterance_id}.align"
            if not align_path.exists():
                logger.warning("Alignment not found for %s, skipping.", vpath.name)
                fail_count += 1
                continue

            # Parse alignment
            label = parse_alignment(align_path)
            if not label:
                logger.warning("Empty alignment for %s, skipping.", vpath.name)
                fail_count += 1
                continue

            # Extract lip frames
            frames = extract_lip_frames(vpath, processor, max_frames)
            if frames is None:
                fail_count += 1
                continue

            # Save
            speaker_out = output_dir / speaker
            speaker_out.mkdir(parents=True, exist_ok=True)

            np.save(speaker_out / f"{utterance_id}_frames.npy", frames)
            (speaker_out / f"{utterance_id}_label.txt").write_text(label)

            success_count += 1

    elapsed = time.time() - start_time
    logger.info(
        "Processing complete: %d succeeded, %d failed, %.1f seconds total.",
        success_count,
        fail_count,
        elapsed,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extract lip ROI frame sequences from GRID corpus videos."
    )
    parser.add_argument(
        "--video_dir", type=str, default="backend/data/raw",
        help="Root directory of raw video files.",
    )
    parser.add_argument(
        "--align_dir", type=str, default="backend/data/alignments",
        help="Root directory of alignment files.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="backend/data/processed",
        help="Output directory for processed data.",
    )
    parser.add_argument(
        "--max_frames", type=int, default=MAX_FRAMES,
        help="Max frames per lip ROI sequence.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    process_dataset(
        video_dir=args.video_dir,
        align_dir=args.align_dir,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()

