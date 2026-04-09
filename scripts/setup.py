#!/usr/bin/env python3
"""
LipSync Setup Script
====================
Handles environment verification, Auto-AVSR repo cloning, and model weight downloads.

Usage:
    python scripts/setup.py                    # Full setup (clone repo + download weights)
    python scripts/setup.py --download-weights  # Download weights only
    python scripts/setup.py --verify            # Verify environment only
    python scripts/setup.py --clone-repo        # Clone Auto-AVSR repo only
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "backend" / "model" / "weights"
AUTOAVSR_DIR = PROJECT_ROOT / "auto_avsr"

# Model weight configurations
# Using the best VSR model: vsr_trlrs2lrs3vox2avsp_base.pth (20.3% WER)
# Fallback: vsr_trlrs3vox2_base.pth (24.6% WER) - smaller download
MODELS = {
    "vsr_trlrs2lrs3vox2avsp_base": {
        "gdrive_id": "1r1kx7l9sWnDOCnaFHIGvOtzuhFyFA88_",
        "filename": "vsr_trlrs2lrs3vox2avsp_base.pth",
        "md5_prefix": "49f77",
        "description": "Best VSR model (20.3% WER, trained on LRS2+LRS3+VoxCeleb2+AVSpeech)",
        "url": "http://www.doc.ic.ac.uk/~pm4115/autoAVSR/vsr_trlrs2lrs3vox2avsp_base.pth",
    },
    "vsr_trlrs3vox2_base": {
        "gdrive_id": "1shcWXUK2iauRhW9NbwCc25FjU1CoMm8i",
        "filename": "vsr_trlrs3vox2_base.pth",
        "md5_prefix": "774a6",
        "description": "Good VSR model (24.6% WER, trained on LRS3+VoxCeleb2)",
        "url": "http://www.doc.ic.ac.uk/~pm4115/autoAVSR/vsr_trlrs3vox2_base.pth",
    },
    "vsr_trlrs3_base": {
        "gdrive_id": "12PNM5szUsk_CuaV1yB9dL_YWvSM1zvAd",
        "filename": "vsr_trlrs3_base.pth",
        "md5_prefix": "c00a7",
        "description": "Base VSR model (36.0% WER, trained on LRS3 only - smallest download)",
        "url": "http://www.doc.ic.ac.uk/~pm4115/autoAVSR/vsr_trlrs3_base.pth",
    },
}

# Default model to download
DEFAULT_MODEL = "vsr_trlrs3vox2_base"

# Auto-AVSR demo video URL
DEMO_VIDEO_URL = "http://www.doc.ic.ac.uk/~pm4115/autoAVSR/autoavsr_demo_video.mp4"
DEMO_VIDEO_PATH = PROJECT_ROOT / "tests" / "test_video.mp4"


def print_header(text: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}\n")


def print_step(step: int, text: str):
    """Print a formatted step."""
    print(f"  [{step}] {text}")


def print_ok(text: str):
    """Print a success message."""
    print(f"  [OK] {text}")


def print_warn(text: str):
    """Print a warning message."""
    print(f"  [WARN] {text}")


def print_error(text: str):
    """Print an error message."""
    print(f"  [ERROR] {text}")


def verify_python_version():
    """Check that Python version is 3.10+."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print_ok(f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python {version.major}.{version.minor}.{version.micro} - need 3.10+")
        return False


def verify_torch():
    """Check that PyTorch is installed and report GPU availability."""
    try:
        import torch

        print_ok(f"PyTorch {torch.__version__}")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print_ok(f"CUDA available - {gpu_name}")
            print_ok(f"CUDA version: {torch.version.cuda}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print_ok("Apple MPS (Metal) available - GPU acceleration enabled")
        else:
            print_warn("No GPU detected - inference will run on CPU (slower)")

        return True
    except ImportError:
        print_error("PyTorch not installed - run: pip install -r requirements.txt")
        return False


def verify_dependencies():
    """Check that all required packages are installed."""
    required = {
        "pytorch_lightning": "pytorch-lightning",
        "sentencepiece": "sentencepiece",
        "cv2": "opencv-python",
        "mediapipe": "mediapipe",
        "torchaudio": "torchaudio",
        "torchvision": "torchvision",
        "av": "av",
        "numpy": "numpy",
        "skimage": "scikit-image",
    }

    all_ok = True
    for module, package in required.items():
        try:
            __import__(module)
            print_ok(f"{package}")
        except ImportError:
            print_error(f"{package} not installed - run: pip install {package}")
            all_ok = False

    return all_ok


def clone_auto_avsr():
    """Clone the Auto-AVSR repository for model code and SentencePiece tokenizer."""
    print_header("Cloning Auto-AVSR Repository")

    if AUTOAVSR_DIR.exists():
        print_ok(f"Auto-AVSR already cloned at {AUTOAVSR_DIR}")
        return True

    print_step(1, "Cloning mpc001/auto_avsr from GitHub...")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/mpc001/auto_avsr.git", str(AUTOAVSR_DIR)],
            check=True,
            capture_output=True,
            text=True,
        )
        print_ok(f"Cloned to {AUTOAVSR_DIR}")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to clone: {e.stderr}")
        return False
    except FileNotFoundError:
        print_error("git not found - please install git")
        return False


def download_weights(model_key: str = DEFAULT_MODEL):
    """Download pretrained VSR model weights."""
    print_header("Downloading Pretrained Model Weights")

    if model_key not in MODELS:
        print_error(f"Unknown model: {model_key}")
        print(f"  Available models: {', '.join(MODELS.keys())}")
        return False

    model_info = MODELS[model_key]
    weight_path = WEIGHTS_DIR / model_info["filename"]

    # Create weights directory
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    if weight_path.exists():
        print_ok(f"Weights already exist: {weight_path}")
        print(f"  Model: {model_info['description']}")
        return True

    print_step(1, f"Downloading {model_info['filename']}...")
    print(f"       {model_info['description']}")

    # Try direct URL download first (faster, no auth needed)
    try:
        import urllib.request

        print(f"       Downloading from {model_info['url']}...")
        urllib.request.urlretrieve(model_info["url"], str(weight_path))
        print_ok(f"Downloaded to {weight_path}")

        # Verify MD5 prefix
        md5 = hashlib.md5(open(weight_path, "rb").read()).hexdigest()
        if md5.startswith(model_info["md5_prefix"]):
            print_ok(f"MD5 checksum verified (starts with {model_info['md5_prefix']})")
        else:
            print_warn(f"MD5 mismatch - got {md5[:5]}, expected {model_info['md5_prefix']}")
            print_warn("File may be corrupted. Try re-downloading.")

        return True

    except Exception as e:
        print_warn(f"Direct download failed: {e}")
        print_step(2, "Trying Google Drive via gdown...")

    # Fallback: use gdown for Google Drive
    try:
        import gdown

        gdown.download(id=model_info["gdrive_id"], output=str(weight_path), quiet=False)
        print_ok(f"Downloaded to {weight_path}")
        return True
    except ImportError:
        print_error("gdown not installed - run: pip install gdown")
        return False
    except Exception as e:
        print_error(f"Download failed: {e}")
        print(f"\n  Manual download:")
        print(f"  1. Go to: {model_info['url']}")
        print(f"  2. Save to: {weight_path}")
        return False


def download_demo_video():
    """Download the Auto-AVSR demo video for testing."""
    print_header("Downloading Demo Video for Testing")

    if DEMO_VIDEO_PATH.exists():
        print_ok(f"Demo video already exists: {DEMO_VIDEO_PATH}")
        return True

    DEMO_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)

    print_step(1, f"Downloading demo video...")
    try:
        import urllib.request

        urllib.request.urlretrieve(DEMO_VIDEO_URL, str(DEMO_VIDEO_PATH))
        print_ok(f"Downloaded to {DEMO_VIDEO_PATH}")
        return True
    except Exception as e:
        print_warn(f"Could not download demo video: {e}")

    # Fallback: create a synthetic test video
    print_step(2, "Creating synthetic test video instead...")
    try:
        import cv2
        import numpy as np

        width, height, fps, num_frames = 640, 480, 25, 75
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(DEMO_VIDEO_PATH), fourcc, fps, (width, height))

        for i in range(num_frames):
            frame = np.ones((height, width, 3), dtype=np.uint8) * 220
            cx, cy = width // 2, height // 2
            # Face oval
            cv2.ellipse(frame, (cx, cy), (100, 130), 0, 0, 360, (200, 180, 160), -1)
            # Eyes
            cv2.ellipse(frame, (cx - 35, cy - 25), (15, 10), 0, 0, 360, (255, 255, 255), -1)
            cv2.ellipse(frame, (cx + 35, cy - 25), (15, 10), 0, 0, 360, (255, 255, 255), -1)
            cv2.circle(frame, (cx - 35, cy - 25), 6, (50, 50, 50), -1)
            cv2.circle(frame, (cx + 35, cy - 25), 6, (50, 50, 50), -1)
            # Nose
            cv2.line(frame, (cx, cy - 10), (cx - 5, cy + 15), (160, 140, 120), 2)
            # Mouth (animated)
            mouth_open = int(5 + 10 * abs(np.sin(2 * np.pi * i / 15)))
            cv2.ellipse(frame, (cx, cy + 45), (25, mouth_open), 0, 0, 360, (150, 80, 80), -1)
            # Eyebrows
            cv2.line(frame, (cx - 55, cy - 45), (cx - 15, cy - 45), (120, 100, 80), 3)
            cv2.line(frame, (cx + 15, cy - 45), (cx + 55, cy - 45), (120, 100, 80), 3)
            writer.write(frame)

        writer.release()
        print_ok(f"Created synthetic test video at {DEMO_VIDEO_PATH}")
        print_warn("Note: This is a synthetic face - predictions will not be meaningful.")
        return True
    except Exception as e2:
        print_error(f"Could not create synthetic test video: {e2}")
        print("  You can manually provide a test video:")
        print(f"  Place a video at: {DEMO_VIDEO_PATH}")
        return False


def verify_environment():
    """Run all environment checks."""
    print_header("Verifying Environment")

    results = []
    results.append(("Python version", verify_python_version()))
    results.append(("PyTorch", verify_torch()))
    results.append(("Dependencies", verify_dependencies()))

    # Check if Auto-AVSR is cloned
    if AUTOAVSR_DIR.exists():
        print_ok(f"Auto-AVSR repo found at {AUTOAVSR_DIR}")
        results.append(("Auto-AVSR repo", True))
    else:
        print_warn(f"Auto-AVSR repo not found - run: python scripts/setup.py --clone-repo")
        results.append(("Auto-AVSR repo", False))

    # Check if weights exist
    weight_files = list(WEIGHTS_DIR.glob("*.pth"))
    if weight_files:
        print_ok(f"Model weights found: {[f.name for f in weight_files]}")
        results.append(("Model weights", True))
    else:
        print_warn("No model weights found - run: python scripts/setup.py --download-weights")
        results.append(("Model weights", False))

    # Summary
    print_header("Verification Summary")
    all_ok = True
    for name, ok in results:
        status = "[OK]" if ok else "[FAIL]"
        print(f"  {status} {name}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n  🎉 Everything looks good! Ready to run inference.")
    else:
        print("\n  Some checks failed. Fix the issues above and re-run verification.")

    return all_ok


def full_setup(model_key: str = DEFAULT_MODEL):
    """Run the complete setup process."""
    print_header("LipSync - Full Setup")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Model: {MODELS[model_key]['description']}")

    # Step 1: Verify environment
    print_header("Step 1/4: Verifying Environment")
    verify_python_version()
    torch_ok = verify_torch()
    deps_ok = verify_dependencies()

    if not torch_ok or not deps_ok:
        print_warn("Some dependencies missing. Install them first:")
        print("  pip install -r requirements.txt")
        response = input("\n  Continue anyway? [y/N] ").strip().lower()
        if response != "y":
            return False

    # Step 2: Clone Auto-AVSR
    print_header("Step 2/4: Cloning Auto-AVSR")
    clone_auto_avsr()

    # Step 3: Download weights
    print_header("Step 3/4: Downloading Model Weights")
    download_weights(model_key)

    # Step 4: Download demo video
    print_header("Step 4/4: Downloading Demo Video")
    download_demo_video()

    # Final verification
    verify_environment()

    return True


def main():
    parser = argparse.ArgumentParser(description="LipSync Setup Script")
    parser.add_argument(
        "--download-weights",
        action="store_true",
        help="Download pretrained model weights only",
    )
    parser.add_argument(
        "--clone-repo",
        action="store_true",
        help="Clone the Auto-AVSR repository only",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify environment setup only",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=list(MODELS.keys()),
        help=f"Model to download (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--download-demo",
        action="store_true",
        help="Download demo video for testing",
    )

    args = parser.parse_args()

    if args.verify:
        verify_environment()
    elif args.download_weights:
        download_weights(args.model)
    elif args.clone_repo:
        clone_auto_avsr()
    elif args.download_demo:
        download_demo_video()
    else:
        full_setup(args.model)


if __name__ == "__main__":
    main()
