"""
LipSync — GRID Corpus Download Script

Downloads the GRID audio-visual corpus for lip reading research.

The GRID corpus contains:
    - 34 speakers (s1–s34, excluding s21 which is missing)
    - 1000 sentences each
    - Video files (.mpg) + alignment files (.align)
    - Constrained grammar: <command> <color> <preposition> <letter> <digit> <adverb>

Source: http://spandh.dcs.shef.ac.uk/gridcorpus/

Usage:
    python -m backend.data.download_grid --output_dir backend/data --speakers s1 s2 s3
    python -m backend.data.download_grid --output_dir backend/data --all
"""

import argparse
import logging
import os
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from tqdm import tqdm

logger = logging.getLogger("lipsync.download")

# ---------------------------------------------------------------------------
# GRID Corpus URLs
# ---------------------------------------------------------------------------
# Video and alignment download URLs per speaker
# The GRID corpus is hosted at the University of Sheffield
GRID_BASE_URL = "https://spandh.dcs.shef.ac.uk/gridcorpus"

# Speakers available (s21 is missing from the corpus)
ALL_SPEAKERS = [f"s{i}" for i in range(1, 35) if i != 21]

# For a quick start, use a small subset
DEFAULT_SPEAKERS = ["s1", "s2", "s3", "s4"]


# ---------------------------------------------------------------------------
# Download Helpers
# ---------------------------------------------------------------------------
class DownloadProgressBar(tqdm):
    """tqdm progress bar for urlretrieve."""

    def update_to(self, blocks=1, block_size=1, total_size=None):
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def download_file(url: str, output_path: str, desc: str = "Downloading"):
    """Download a file with progress bar."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        logger.info("Already downloaded: %s", output_path)
        return output_path

    logger.info("Downloading %s → %s", url, output_path)
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=desc) as t:
        urlretrieve(url, filename=output_path, reporthook=t.update_to)
    return output_path


def extract_archive(archive_path: str, output_dir: str):
    """Extract a tar.gz or zip archive."""
    logger.info("Extracting %s → %s", archive_path, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if archive_path.endswith(".tar.gz") or archive_path.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=output_dir)
    elif archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(path=output_dir)
    else:
        logger.warning("Unknown archive format: %s", archive_path)


# ---------------------------------------------------------------------------
# GRID Corpus Download
# ---------------------------------------------------------------------------
def download_speaker_videos(speaker: str, output_dir: Path):
    """Download video files for a single GRID speaker."""
    video_url = f"{GRID_BASE_URL}/{speaker}/video/{speaker}.mpg_vcd.zip"
    archive_path = output_dir / "archives" / f"{speaker}_video.zip"
    video_dir = output_dir / "raw" / speaker

    if video_dir.exists() and any(video_dir.iterdir()):
        logger.info("Videos for %s already extracted, skipping.", speaker)
        return

    try:
        download_file(str(video_url), str(archive_path), desc=f"{speaker} videos")
        extract_archive(str(archive_path), str(video_dir))
        logger.info("Videos for %s extracted to %s", speaker, video_dir)
    except Exception as e:
        logger.error("Failed to download videos for %s: %s", speaker, e)
        logger.info(
            "You may need to download manually from: %s",
            f"{GRID_BASE_URL}/{speaker}/",
        )


def download_speaker_alignments(speaker: str, output_dir: Path):
    """Download alignment files for a single GRID speaker."""
    align_url = f"{GRID_BASE_URL}/{speaker}/align/{speaker}.tar"
    archive_path = output_dir / "archives" / f"{speaker}_align.tar"
    align_dir = output_dir / "alignments" / speaker

    if align_dir.exists() and any(align_dir.iterdir()):
        logger.info("Alignments for %s already extracted, skipping.", speaker)
        return

    try:
        # Try .tar first, then .tar.gz
        try:
            download_file(str(align_url), str(archive_path), desc=f"{speaker} alignments")
        except Exception:
            align_url = f"{GRID_BASE_URL}/{speaker}/align/{speaker}.tar.gz"
            archive_path = output_dir / "archives" / f"{speaker}_align.tar.gz"
            download_file(str(align_url), str(archive_path), desc=f"{speaker} alignments")

        os.makedirs(str(align_dir), exist_ok=True)
        if str(archive_path).endswith(".tar"):
            with tarfile.open(str(archive_path), "r") as tar:
                tar.extractall(path=str(align_dir))
        else:
            extract_archive(str(archive_path), str(align_dir))
        logger.info("Alignments for %s extracted to %s", speaker, align_dir)
    except Exception as e:
        logger.error("Failed to download alignments for %s: %s", speaker, e)
        logger.info(
            "You may need to download manually from: %s",
            f"{GRID_BASE_URL}/{speaker}/",
        )


def download_grid_corpus(
    output_dir: str = "backend/data",
    speakers: list[str] | None = None,
):
    """
    Download the GRID corpus (video + alignment files) for specified speakers.

    Args:
        output_dir: Root data directory.
        speakers: List of speaker IDs (e.g., ["s1", "s2"]). If None, downloads DEFAULT_SPEAKERS.
    """
    output_dir = Path(output_dir)
    speakers = speakers or DEFAULT_SPEAKERS

    # Create archive directory
    (output_dir / "archives").mkdir(parents=True, exist_ok=True)

    logger.info("Downloading GRID corpus for speakers: %s", speakers)

    for speaker in speakers:
        if speaker not in ALL_SPEAKERS:
            logger.warning("Unknown speaker %s, skipping.", speaker)
            continue

        logger.info("=== Processing speaker: %s ===", speaker)
        download_speaker_videos(speaker, output_dir)
        download_speaker_alignments(speaker, output_dir)

    logger.info("Download complete! Data saved to %s", output_dir)
    logger.info(
        "\nNext steps:\n"
        "  1. Verify downloads:  ls -la %s/raw/\n"
        "  2. Extract lip ROIs:  python -m backend.utils.lip_extractor\n"
        "  3. Explore data:      jupyter notebook notebooks/exploration.ipynb\n",
        output_dir,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Download the GRID audio-visual corpus."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="backend/data",
        help="Root data directory.",
    )
    parser.add_argument(
        "--speakers",
        nargs="+",
        default=None,
        help="Speaker IDs to download (e.g., s1 s2 s3).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all 33 speakers.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    speakers = ALL_SPEAKERS if args.all else args.speakers
    download_grid_corpus(output_dir=args.output_dir, speakers=speakers)


if __name__ == "__main__":
    main()

