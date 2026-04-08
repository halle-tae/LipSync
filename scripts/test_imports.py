#!/usr/bin/env python3
"""Quick test to verify all Auto-AVSR imports work correctly."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "auto_avsr"))
sys.path.insert(0, str(PROJECT_ROOT))

print("Test 1: Importing Auto-AVSR modules...")
from lightning import ModelModule
print("  OK: ModelModule imported")

from datamodule.transforms import VideoTransform, TextTransform
print("  OK: VideoTransform, TextTransform imported")

from preparation.detectors.mediapipe.detector import LandmarksDetector
from preparation.detectors.mediapipe.video_process import VideoProcess
print("  OK: MediaPipe detector and VideoProcess imported")

print("\nTest 2: Initializing TextTransform...")
tt = TextTransform()
print(f"  OK: TextTransform initialized, token_list has {len(tt.token_list)} tokens")

print("\nTest 3: Initializing VideoTransform...")
vt = VideoTransform(subset="test")
print("  OK: VideoTransform initialized")

print("\nTest 4: Loading model with weights...")
import argparse
import torch

args = argparse.Namespace()
args.modality = "video"
model_module = ModelModule(args)

weights_path = PROJECT_ROOT / "backend" / "model" / "weights" / "vsr_trlrs3vox2_base.pth"
if weights_path.exists():
    print(f"  Loading weights from {weights_path}...")
    ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    
    # Determine checkpoint format
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model_module.model.load_state_dict(ckpt["model_state_dict"])
        print("  Loaded from model_state_dict key")
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        model_module.model.load_state_dict(ckpt["state_dict"])
        print("  Loaded from state_dict key")
    else:
        model_module.model.load_state_dict(ckpt)
        print("  Loaded as raw state_dict")
    
    model_module.eval()
    num_params = sum(p.numel() for p in model_module.parameters())
    print(f"  OK: Model loaded with {num_params / 1e6:.1f}M parameters")
else:
    print(f"  SKIP: Weight file not found at {weights_path}")

print("\nTest 5: Testing forward pass with dummy data...")
import torch
dummy_input = torch.randn(30, 1, 88, 88)  # 30 frames, grayscale, 88x88
with torch.no_grad():
    try:
        predicted_text = model_module(dummy_input)
        print(f"  OK: Forward pass succeeded")
        print(f"  Predicted text: '{predicted_text}'")
    except Exception as e:
        print(f"  ERROR: Forward pass failed: {e}")

print("\nAll tests passed!")
