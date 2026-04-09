#!/usr/bin/env python3
"""Download model using gdown with fuzzy matching for large files."""
import subprocess
import sys
from pathlib import Path

output_path = Path("backend/model/weights/vsr_trlrs3vox2_base.pth")
output_path.parent.mkdir(parents=True, exist_ok=True)

# Google Drive file ID from the Auto-AVSR README
file_id = "1shcWXUK2iauRhW9NbwCc25FjU1CoMm8i"

print("Downloading model weights...")
print(f"File ID: {file_id}")
print(f"Output: {output_path}")
print()

# Use gdown with fuzzy matching (--fuzzy) which handles large files better
cmd = [
    sys.executable, "-m", "gdown",
    "--id", file_id,
    "--output", str(output_path),
    "--fuzzy"
]

try:
    subprocess.run(cmd, check=True)
    print("\n[OK] Download complete!")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
except subprocess.CalledProcessError as e:
    print(f"\n[FAIL] Download failed: {e}")
    print("\nManual download:")
    print(f"1. Visit: https://drive.google.com/file/d/{file_id}/view")
    print("2. Download the file manually")
    print(f"3. Save as: {output_path.absolute()}")
except Exception as e:
    print(f"\n[FAIL] Error: {e}")
