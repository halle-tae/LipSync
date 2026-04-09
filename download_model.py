#!/usr/bin/env python3
"""Download model weights with SSL verification disabled."""
import ssl
import urllib.request
from pathlib import Path

# Disable SSL verification (for corporate proxy/firewall)
ssl._create_default_https_context = ssl._create_unverified_context

# Model URLs (trying multiple sources)
URLS = [
    "https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages/releases/download/v0.1/vsr_trlrs3vox2_base.pth",
    "http://www.doc.ic.ac.uk/~pm4115/autoAVSR/vsr_trlrs3vox2_base.pth",
]

output_path = Path("backend/model/weights/vsr_trlrs3vox2_base.pth")
output_path.parent.mkdir(parents=True, exist_ok=True)

for url in URLS:
    print(f"Trying: {url}")
    try:
        print("Downloading...")
        urllib.request.urlretrieve(url, str(output_path))
        print(f"✓ Downloaded to: {output_path}")
        print(f"Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
        break
    except Exception as e:
        print(f"✗ Failed: {e}")
        continue
else:
    print("\nAll download attempts failed.")
    print("Please download manually from:")
    print("https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages/releases")
