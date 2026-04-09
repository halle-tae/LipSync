# LipSync Setup Guide

## Current Status

✅ **Completed:**
- Python 3.12.10 installed
- All Python dependencies installed (PyTorch, MediaPipe, Gradio, FastAPI, etc.)
- Auto-AVSR repository cloned to `auto_avsr/`
- Demo test video exists at `tests/test_video.mp4`

❌ **Remaining:**
- Model weights need to be downloaded manually (SSL certificate issues on corporate network)

## Download Model Weights Manually

**Required file:** `vsr_trlrs3vox2_base.pth` (~400 MB)

### Option 1: Google Drive (Recommended)
1. Visit: https://drive.google.com/file/d/1shcWXUK2iauRhW9NbwCc25FjU1CoMm8i/view
2. Click the "Download" button
3. Save the file to: `C:\Users\halleteh\repos\LipSync\backend\model\weights\vsr_trlrs3vox2_base.pth`

### Alternative Models (if you want different accuracy/size tradeoffs)

| Model | WER | Size | Google Drive Link |
|-------|-----|------|-------------------|
| vsr_trlrs3_base.pth | 36.0% | ~400MB | https://drive.google.com/file/d/12PNM5szUsk_CuaV1yB9dL_YWvSM1zvAd/view |
| vsr_trlrs3vox2_base.pth | 24.6% | ~400MB | https://drive.google.com/file/d/1shcWXUK2iauRhW9NbwCc25FjU1CoMm8i/view |
| vsr_trlrs2lrs3vox2avsp_base.pth | 20.3% | ~400MB | https://drive.google.com/file/d/1r1kx7l9sWnDOCnaFHIGvOtzuhFyFA88_/view |

Lower WER = better accuracy. The `vsr_trlrs3vox2_base.pth` model is recommended.

## Verify Download

Once downloaded, verify the file:

```bash
# Check file size (should be ~400 MB, not 2-3 KB)
ls -lh backend/model/weights/vsr_trlrs3vox2_base.pth

# Or in PowerShell:
# Get-Item backend\model\weights\vsr_trlrs3vox2_base.pth | Select-Object Length
```

The file should be approximately 400-500 MB. If it's only a few KB, it's an error page.

## Running the Application

Once the model weights are downloaded:

### 1. Test with Demo Video
```bash
python main.py --video tests/test_video.mp4
```

This will:
- Load the model
- Process the demo video
- Output predicted text and performance metrics

### 2. Real-Time Webcam Mode
```bash
python main.py --webcam
```

Features:
- Real-time lip reading from your webcam
- OpenCV preview window
- Performance metrics in terminal

Optional parameters:
```bash
# Use specific webcam
python main.py --webcam --camera-index 0

# Adjust inference interval (seconds between predictions)
python main.py --webcam --infer-interval-seconds 1.0

# Headless mode (no preview window)
python main.py --webcam --no-preview
```

### 3. Gradio Web UI
```bash
python frontend/gradio_app.py
```

Then open http://localhost:7860 in your browser.

Features:
- Live webcam preview
- Caption overlay
- Confidence indicator
- Customizable appearance (font, color, size)
- Session transcript

### 4. Full Web App (Next.js + FastAPI)

Terminal 1 - Start backend API:
```bash
python backend/api_server.py
```

Terminal 2 - Start web frontend:
```bash
cd web
npm install
npm run dev
```

Then open http://localhost:3000

## Troubleshooting

### "Weight file not found"
- Make sure the .pth file is in `backend/model/weights/`
- Verify file size is ~400 MB, not a few KB

### "Auto-AVSR repo not found"
```bash
git clone https://github.com/mpc001/auto_avsr.git auto_avsr
```

### Slow inference on CPU
- Expected behavior without GPU
- GPU inference is 5-10x faster
- Consider ONNX export for CPU optimization

### MediaPipe import errors
```bash
pip install "mediapipe>=0.10.0,<0.10.21"
```

### No webcam detected
- Check camera permissions in Windows Settings
- Try different --camera-index (0, 1, 2, etc.)

## System Info

- Python: 3.12.10
- PyTorch: 2.11.0+cpu (CPU only, no CUDA)
- Platform: Windows 11 Enterprise
- Working Directory: C:\Users\halleteh\repos\LipSync

## Next Steps

1. Download the model weights manually using the link above
2. Verify the file size is correct (~400 MB)
3. Run the demo video test: `python main.py --video tests/test_video.mp4`
4. If that works, try webcam mode: `python main.py --webcam`
5. Explore the Gradio UI: `python frontend/gradio_app.py`
