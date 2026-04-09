"""
Model Loader
============
Handles downloading, caching, and initializing the Auto-AVSR pretrained model.

This module wraps the Auto-AVSR `ModelModule` (from lightning.py) and `E2E` model,
providing a clean interface for loading the VSR model with pretrained weights.

The model architecture is:
    video_resnet (3D frontend) → proj_encoder (linear 512→768) → conformer_encoder → transformer_decoder

Input: grayscale mouth ROI video tensor [T, 1, 88, 88]
Output: predicted text string
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
AUTOAVSR_DIR = PROJECT_ROOT / "auto_avsr" / "auto_avsr"  # Nested structure
WEIGHTS_DIR = PROJECT_ROOT / "backend" / "model" / "weights"

# Default weight file — best balance of quality and download size
DEFAULT_WEIGHTS = "vsr_trlrs3vox2_base.pth"


def _ensure_autoavsr_on_path():
    """Add Auto-AVSR repo to sys.path so we can import its modules."""
    autoavsr_path = str(AUTOAVSR_DIR)
    if autoavsr_path not in sys.path:
        sys.path.insert(0, autoavsr_path)
        logger.debug(f"Added {autoavsr_path} to sys.path")


class ModelLoader:
    """
    Loads and initializes the Auto-AVSR model for video-only speech recognition.

    Usage:
        loader = ModelLoader()
        model_module = loader.load()
        # model_module is a ModelModule (LightningModule) in eval mode
        # Call model_module(video_tensor) to get predicted text
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            weights_path: Path to the .pth weight file. If None, uses default.
            device: Device to load model on ('cuda', 'mps', 'cpu'). Auto-detected if None.
        """
        self.weights_path = Path(weights_path) if weights_path else WEIGHTS_DIR / DEFAULT_WEIGHTS
        self.device = device or self._detect_device()
        self.model_module = None

    @staticmethod
    def _detect_device() -> str:
        """Auto-detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS can have issues with some ops — fallback to CPU if needed
            return "cpu"  # MPS support for Auto-AVSR is untested, use CPU for safety
        else:
            return "cpu"

    def _validate_weights(self) -> bool:
        """Check that the weight file exists and is valid."""
        if not self.weights_path.exists():
            logger.error(
                f"Weight file not found: {self.weights_path}\n"
                f"Run: python scripts/setup.py --download-weights"
            )
            return False

        # Basic size check — VSR model weights should be ~1GB+
        size_mb = self.weights_path.stat().st_size / (1024 * 1024)
        if size_mb < 10:
            logger.warning(f"Weight file seems too small ({size_mb:.1f} MB). May be corrupted.")

        return True

    def load(self) -> "ModelModule":
        """
        Load the Auto-AVSR model with pretrained weights.

        Returns:
            ModelModule instance in eval mode, ready for inference.

        Raises:
            FileNotFoundError: If weight file doesn't exist.
            RuntimeError: If Auto-AVSR repo is not cloned.
        """
        if self.model_module is not None:
            return self.model_module

        # Validate
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Weight file not found: {self.weights_path}\n"
                f"Run: python scripts/setup.py --download-weights"
            )

        if not AUTOAVSR_DIR.exists():
            raise RuntimeError(
                f"Auto-AVSR repo not found at {AUTOAVSR_DIR}\n"
                f"Run: python scripts/setup.py --clone-repo"
            )

        # Add Auto-AVSR to path
        _ensure_autoavsr_on_path()

        logger.info(f"Loading model from {self.weights_path}...")
        logger.info(f"Device: {self.device}")

        # Import Auto-AVSR's ModelModule
        import argparse

        from lightning import ModelModule

        # Create args namespace with required attributes
        args = argparse.Namespace()
        args.modality = "video"

        # Load checkpoint to inspect structure
        ckpt = torch.load(
            str(self.weights_path),
            map_location="cpu",
            weights_only=False,
        )

        # Initialize the model module
        model_module = ModelModule(args)

        # Load weights — Auto-AVSR stores weights as raw state_dict
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model_module.model.load_state_dict(ckpt["model_state_dict"])
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            model_module.model.load_state_dict(ckpt["state_dict"])
        else:
            # Raw state dict
            model_module.model.load_state_dict(ckpt)

        # Move to device and set eval mode
        model_module = model_module.to(self.device)
        model_module.eval()

        self.model_module = model_module
        logger.info("Model loaded successfully.")

        # Log model size
        num_params = sum(p.numel() for p in model_module.parameters())
        logger.info(f"Model parameters: {num_params / 1e6:.1f}M")

        return model_module

    def get_device(self) -> str:
        """Return the device the model is loaded on."""
        return self.device

    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.model_module is not None

    def unload(self):
        """Unload the model to free memory."""
        if self.model_module is not None:
            del self.model_module
            self.model_module = None
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logger.info("Model unloaded.")
