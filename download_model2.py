#!/usr/bin/env python3
"""Download model weights using requests with SSL verification disabled."""
import requests
from pathlib import Path
from tqdm import tqdm

# Disable SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Model URLs
URLS = [
    ("Google Drive (original)", "https://drive.google.com/uc?id=1shcWXUK2iauRhW9NbwCc25FjU1CoMm8i&export=download&confirm=t"),
    ("Imperial College", "http://www.doc.ic.ac.uk/~pm4115/autoAVSR/vsr_trlrs3vox2_base.pth"),
]

output_path = Path("backend/model/weights/vsr_trlrs3vox2_base.pth")
output_path.parent.mkdir(parents=True, exist_ok=True)

for name, url in URLS:
    print(f"\nTrying {name}...")
    print(f"URL: {url}")
    try:
        response = requests.get(url, stream=True, verify=False, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        print(f"Size: {total_size / 1024 / 1024:.1f} MB")

        with open(output_path, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

        print(f"[OK] Downloaded to: {output_path}")
        print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
        break
    except Exception as e:
        print(f"[FAIL] {e}")
        if output_path.exists():
            output_path.unlink()
        continue
else:
    print("\n" + "="*60)
    print("All download attempts failed!")
    print("="*60)
    print("\nManual download instructions:")
    print("1. Visit: https://github.com/mpc001/auto_avsr")
    print("2. Look for pretrained model weights download links")
    print("3. Download 'vsr_trlrs3vox2_base.pth' (or similar)")
    print(f"4. Save to: {output_path.absolute()}")
