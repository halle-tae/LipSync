# LipSync — Real-Time Lip Reading Assistant for the Hard of Hearing

> A webcam-based application that generates live captions from a speaker's lip movements using computer vision and a state-of-the-art visual speech recognition model. Built for the AMD Intern Innovation Showcase 2026.

---

## Problem Statement

Millions of people who are hard of hearing rely on audio-based captioning tools — but these fail in noisy environments, across glass barriers, in silent video calls, or when audio hardware malfunctions. There is no widely available, real-time tool that generates captions purely from visual lip movement data.

**LipSync** fills this gap by providing a webcam-powered captioning overlay that works without any audio input.

## Solution Overview

LipSync uses the **Auto-AVSR** pretrained model (Apache 2.0, Imperial College London / Meta) as the visual speech recognition backbone. We extract the video-only inference pipeline and wrap it in a clean, accessible desktop application with a real-time captioning overlay.

**What we built (our contribution):**
- Real-time webcam capture and face/lip detection pipeline
- Accessible captioning overlay UI with customizable fonts, sizes, colors, and positioning
- Confidence indicator showing model certainty per caption segment
- Session transcript logging with timestamps and export functionality
- Performance dashboard showing latency, FPS, and word detection stats

**What we used (not our contribution):**
- Auto-AVSR pretrained VSR model weights and inference code (cited, Apache 2.0)
- MediaPipe or dlib for face landmark detection
- PyTorch for model inference

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | Auto-AVSR (pretrained, video-only mode) |
| Framework | PyTorch 2.x |
| Face Detection | MediaPipe Face Mesh / dlib |
| Frontend | Electron + React (or PyQt6) |
| Webcam Capture | OpenCV |
| Language | Python (backend), TypeScript/Python (frontend) |

## Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌────────────────┐
│   Webcam     │───▶│  Face Detection  │───▶│  Mouth ROI Crop  │───▶│  Auto-AVSR     │
│   Feed       │    │  (MediaPipe)     │    │  + Preprocessing │    │  Inference      │
└─────────────┘    └──────────────────┘    └──────────────────┘    └───────┬────────┘
                                                                           │
                                                                    predicted text
                                                                           │
                                                                           ▼
┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌────────────────┐
│  Transcript  │◀───│  Session Logger  │◀───│  Text Smoothing  │◀───│  Caption       │
│  Export      │    │  (timestamps)    │    │  + Buffering     │    │  Overlay UI    │
└─────────────┘    └──────────────────┘    └──────────────────┘    └────────────────┘
```

## Features

### Core
- **Live lip-reading captions** — real-time text generated from webcam video, no audio needed
- **Accessible overlay** — high-contrast, resizable caption bar that floats over other apps
- **Confidence scoring** — visual indicator of how certain the model is about each prediction

### Quality of Life
- **Customizable appearance** — font size, color, background opacity, position (top/bottom/floating)
- **Session transcripts** — full log of all captions with timestamps, exportable to `.txt` or `.srt`
- **Multi-face handling** — detects and tracks the primary speaker when multiple faces are visible
- **Latency dashboard** — real-time display of inference time, FPS, and pipeline health

### Accessibility-First Design
- WCAG 2.1 AA compliant contrast ratios
- Keyboard-navigable settings
- Screen reader compatible controls
- Dyslexia-friendly font option (OpenDyslexic)

## Project Structure

```
lipsync/
├── backend/
│   ├── model/              # Auto-AVSR model loading and inference
│   │   ├── loader.py       # Download and initialize pretrained weights
│   │   └── inference.py    # Video-only inference pipeline
│   ├── capture/
│   │   ├── webcam.py       # OpenCV webcam capture
│   │   └── face_detect.py  # MediaPipe face mesh + mouth ROI extraction
│   ├── processing/
│   │   ├── preprocess.py   # Frame normalization, grayscale, resize
│   │   └── buffer.py       # Frame buffer for temporal context
│   └── api.py              # WebSocket/IPC bridge to frontend
├── frontend/
│   ├── components/
│   │   ├── CaptionOverlay/  # Main caption display component
│   │   ├── Settings/        # Accessibility and appearance settings
│   │   ├── Dashboard/       # Performance metrics display
│   │   └── Transcript/      # Session log viewer and exporter
│   ├── App.tsx
│   └── main.ts
├── scripts/
│   ├── setup.sh            # Environment setup and model download
│   └── benchmark.py        # Latency and accuracy benchmarking
├── tests/
├── PHASES.md
├── README.md
└── requirements.txt
```

## Quick Start

```bash
# 1. Clone this repo
git clone https://github.com/YOUR_USERNAME/lipsync.git
cd lipsync

# 2. Create environment
conda create -n lipsync python=3.10 -y
conda activate lipsync

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run full setup (clones Auto-AVSR, downloads model weights + demo video)
python scripts/setup.py

# 5. Test inference on the demo video
python scripts/test_inference.py

# 6. Run the app
python main.py
```

## Setup Details

### Prerequisites
- **Python 3.10+** (required)
- **Git** (for cloning the Auto-AVSR repository)
- **GPU** (optional) — CUDA-capable GPU for faster inference. CPU works but is slower.

### Setup Script Options

```bash
# Full setup — recommended for first run
python scripts/setup.py

# Individual steps
python scripts/setup.py --clone-repo          # Clone Auto-AVSR repo only
python scripts/setup.py --download-weights    # Download model weights only
python scripts/setup.py --download-demo       # Download test video only
python scripts/setup.py --verify              # Check environment is ready

# Choose a specific model variant
python scripts/setup.py --model vsr_trlrs3_base              # Fastest download (36% WER)
python scripts/setup.py --model vsr_trlrs3vox2_base          # Good balance (24.6% WER) — default
python scripts/setup.py --model vsr_trlrs2lrs3vox2avsp_base  # Best accuracy (20.3% WER)
```

### Model Variants

| Model | WER | Training Data | Download |
|---|---|---|---|
| `vsr_trlrs3_base` | 36.0% | LRS3 (438h) | Smallest |
| `vsr_trlrs3vox2_base` | 24.6% | LRS3 + VoxCeleb2 (1759h) | **Default** |
| `vsr_trlrs2lrs3vox2avsp_base` | 20.3% | LRS2 + LRS3 + VoxCeleb2 + AVSpeech (3291h) | Largest |

### Testing Inference

```bash
# Run on the demo video (downloaded during setup)
python scripts/test_inference.py

# Run on your own video
python scripts/test_inference.py --video path/to/video.mp4

# Verbose output (debug logging)
python scripts/test_inference.py --verbose

# Force CPU inference
python scripts/test_inference.py --device cpu
```

### Real-Time Webcam Mode (Phase 1)

```bash
# Run live webcam lip-reading (OpenCV preview enabled)
python main.py --webcam

# Optional tuning for slower CPUs
python main.py --webcam --window-seconds 1.5 --infer-interval-seconds 1.0 --device cpu

# Use a specific webcam and resolution
python main.py --webcam --camera-index 1 --camera-width 1280 --camera-height 720

# Headless mode (no preview window)
python main.py --webcam --no-preview
```

Webcam mode prints per-prediction latency metrics in the terminal:
- capture latency (frame timing)
- face detect + mouth ROI extraction latency
- preprocessing latency
- model inference + decoding latency
- total latency

### Caption Overlay UI (Phase 2)

```bash
# Launch Gradio UI with live caption overlay + webcam preview
python frontend/gradio_app.py
```

Phase 2 UI includes:
- Live webcam preview with face bounding box + mouth ROI box
- Caption overlay component (top/bottom/floating)
- Confidence indicator (green/yellow/red from beam-search scores)
- Settings panel: font size, font color, background opacity, dyslexia-friendly font toggle
- Runtime controls: temporal window and inference interval

### Web Frontend (Next.js + TypeScript + Tailwind + shadcn + Firebase)

```bash
# Terminal 1: start real inference API
python backend/api_server.py

# Terminal 2: start web app
cd web
npm install
cp .env.local.example .env.local
npm run dev
```

Optional: set `NEXT_PUBLIC_LIPSYNC_API_URL` in `web/.env.local` if your API is not running on `http://127.0.0.1:8000`.

Then open http://localhost:3000.

### Phase 3 Additions

- Dark mode toggle and refined card-based layout
- Accessibility-first controls with clear labels and live caption region
- Session transcript panel with export to `.txt` and `.srt`
- Performance dashboard: capture/detect/inference latency, FPS, total words detected
- Friendly edge-state handling:
  - no camera / permission denied error panel
  - loading progress bar while model initializes
  - "Looking for a speaker..." placeholder when no face is detected
  - low-confidence fallback message instead of unreliable caption text

### Gotchas & Troubleshooting

1. **"Auto-AVSR repo not found"** — Run `python scripts/setup.py --clone-repo`
2. **"Weight file not found"** — Run `python scripts/setup.py --download-weights`
3. **MediaPipe import error / missing `mediapipe.solutions`** — Reinstall a compatible version: `pip install "mediapipe>=0.10.0,<0.10.21"`
4. **Slow inference on CPU** — Expected. GPU inference is 5-10x faster. Consider ONNX export for CPU optimization.
5. **"Cannot detect any frames"** — The video needs a clear, frontal face. Try the demo video first.
6. **macOS MPS** — MPS backend is currently disabled for compatibility. The model runs on CPU on Mac.

## Attribution

This project uses the **Auto-AVSR** pretrained model for visual speech recognition:

```
@inproceedings{ma2023auto,
  author={Ma, Pingchuan and Haliassos, Alexandros and Fernandez-Lopez, Adriana
          and Chen, Honglie and Petridis, Stavros and Pantic, Maja},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={Auto-AVSR: Audio-Visual Speech Recognition with Automatic Labels},
  year={2023},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10096889}
}
```

Repository: [github.com/mpc001/auto_avsr](https://github.com/mpc001/auto_avsr) — Apache 2.0 License.

## License

This project is licensed under the MIT License. The Auto-AVSR model weights and inference code are used under their original Apache 2.0 license.

## Authors

- **[Your Name]** — AMD Intern, [Team/Department]

Built for the **AMD Intern Innovation Showcase 2026**.
