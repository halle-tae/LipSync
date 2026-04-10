# LipSync — AI-Powered Visual Speech Recognition

> A video-based application that generates accurate captions from lip movements using computer vision and a state-of-the-art visual speech recognition model. Upload a video and watch timestamped captions appear synchronized with the speaker's lips — no audio required.

---

## Problem Statement

Millions of people who are hard of hearing rely on audio-based captioning tools — but these fail in noisy environments, across glass barriers, in silent video calls, or when audio hardware malfunctions. Additionally, archival videos often lack audio or have poor audio quality.

**LipSync** addresses this by generating captions purely from visual lip movement data, making videos accessible without relying on audio.

## Solution Overview

LipSync uses the **Auto-AVSR** pretrained model (Apache 2.0, Imperial College London / Meta) as the visual speech recognition backbone. We extract the video-only inference pipeline and wrap it in a modern web application with synchronized caption playback.

**What we built:**
- Video upload and processing pipeline with chunked inference
- Face and mouth bounding box detection and visualization for every frame
- Synchronized caption overlay that plays in real-time with uploaded videos
- Transcript export to SRT (subtitle) and TXT formats
- FastAPI backend with OpenCV-based video processing
- Next.js + React frontend with Tailwind CSS and Framer Motion
- Confidence indicators showing model certainty per caption segment

**What we used:**
- Auto-AVSR pretrained VSR model weights and inference code (Apache 2.0)
- MediaPipe Face Mesh for face landmark detection
- PyTorch for model inference
- FastAPI for REST API
- Next.js for modern web UI

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | Auto-AVSR (pretrained, video-only mode) |
| ML Framework | PyTorch 2.x |
| Face Detection | MediaPipe Face Mesh |
| Backend API | FastAPI + Uvicorn |
| Video Processing | OpenCV (cv2) |
| Frontend | Next.js 16 + React 19 + TypeScript |
| Styling | Tailwind CSS 4 + shadcn/ui components |
| Animation | Framer Motion |
| Language | Python (backend), TypeScript (frontend) |

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  User Uploads   │───▶│  FastAPI Backend │───▶│  OpenCV Video    │
│  MP4 Video      │    │  /upload/video   │    │  Frame Extraction│
└─────────────────┘    └──────────────────┘    └────────┬─────────┘
                                                         │
                                                    All frames (RGB)
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  MediaPipe Face │◀───│  Per-Frame       │    │  Chunk into      │
│  Detection      │    │  Bounding Boxes  │    │  ~4s Windows     │
└────────┬────────┘    └──────────────────┘    └────────┬─────────┘
         │                                               │
    face_bbox, mouth_bbox                          100 frames/chunk
         │                                               │
         ▼                                               ▼
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Return Frame   │    │  Auto-AVSR       │◀───│  Mouth ROI Crop  │
│  Metadata       │    │  Inference       │    │  + Preprocessing │
└────────┬────────┘    └────────┬─────────┘    └──────────────────┘
         │                      │
         │               text + confidence + timestamps
         │                      │
         └──────────┬───────────┘
                    ▼
         ┌──────────────────────┐
         │  Return JSON to      │
         │  Frontend            │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐    ┌──────────────────┐
         │  Next.js Video       │───▶│  Canvas Overlay  │
         │  Player + Captions   │    │  Face/Mouth Box  │
         └──────────────────────┘    └──────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  SRT/TXT Export      │
         └──────────────────────┘
```

## Features

### Core Functionality
- **Video upload processing** — Upload MP4 files and get accurate lip-reading captions
- **Synchronized caption playback** — Captions appear in sync with the video, like subtitles
- **Face and mouth tracking** — Visual bounding boxes show detected face and mouth regions per frame
- **High accuracy** — Processes full temporal context (~4 second chunks) for better results than real-time streaming

### Caption & Export
- **Confidence scoring** — Visual indicator of model certainty for each caption segment
- **SRT export** — Download standard subtitle files compatible with video players
- **TXT export** — Plain text transcript with timestamps
- **Transcript viewer** — Scrollable list of all detected captions with timestamps

### User Experience
- **Drag-and-drop upload** — Easy file selection with visual feedback
- **Processing progress** — See real-time updates as chunks are processed
- **Dark mode** — Clean, modern UI with dark theme
- **Responsive design** — Works on desktop and tablet screens
- **Canvas overlay** — Real-time bounding box visualization scaled to video size

## Project Structure

```
LipSync/
├── backend/
│   ├── api_server.py          # FastAPI server with /upload/video endpoint
│   ├── model/
│   │   ├── loader.py          # Auto-AVSR model initialization
│   │   ├── inference.py       # Chunked inference pipeline
│   │   └── weights/           # Model weights (.pth files)
│   ├── capture/
│   │   ├── face_detect.py     # MediaPipe face mesh + bounding box extraction
│   │   └── webcam.py          # (Legacy) webcam capture utilities
│   └── processing/
│       ├── preprocess.py      # Frame normalization, grayscale, resize to 96x96
│       └── buffer.py          # Sliding frame buffer for temporal windows
├── web/                       # Next.js frontend
│   ├── src/
│   │   └── app/
│   │       ├── page.tsx       # Main upload + video player page
│   │       ├── layout.tsx     # Root layout with metadata
│   │       └── globals.css    # Tailwind + custom styles
│   ├── public/                # Static assets
│   ├── package.json
│   └── tsconfig.json
├── auto_avsr/                 # Cloned Auto-AVSR repository (submodule/dependency)
├── scripts/
│   ├── setup.py               # Download model weights and setup Auto-AVSR
│   └── test_inference.py      # Test offline inference on videos
├── tests/
│   └── test_lipsync.mp4       # Sample test video
├── start-app.sh               # Helper script to start both frontend and backend
├── requirements.txt           # Python dependencies
├── PHASES.md                  # Development phases documentation
├── PIVOT_PLAN.md              # Video upload mode implementation plan
└── README.md                  # This file
```

## Quick Start

### Prerequisites
- **Python 3.10+**
- **Node.js 18+** (for the Next.js frontend)
- **Git**
- **GPU** (optional but recommended for faster inference; CPU works but is slower)

### Installation

```bash
# 1. Clone this repo
git clone https://github.com/YOUR_USERNAME/lipsync.git
cd lipsync

# 2. Set up Python environment
conda create -n lipsync python=3.10 -y
conda activate lipsync
pip install -r requirements.txt

# 3. Download Auto-AVSR model weights
python scripts/setup.py

# 4. Install frontend dependencies
cd web
npm install
cd ..
```

### Running the App

**Option 1: Use the startup script (recommended)**
```bash
./start-app.sh
```
This will automatically:
- Kill any processes on ports 3000 and 8000
- Start the backend (FastAPI on port 8000)
- Start the frontend (Next.js on port 3000)

**Option 2: Manual startup**
```bash
# Terminal 1: Start backend
python backend/api_server.py

# Terminal 2: Start frontend
cd web
npm run dev
```

Then open **http://localhost:3000** in your browser.

### Testing

```bash
# Test inference on a sample video
python scripts/test_inference.py

# Use your own video
python scripts/test_inference.py --video path/to/your/video.mp4
```

## How to Use

1. **Start the application** (see Quick Start above)
2. **Open http://localhost:3000** in your browser
3. **Upload a video**:
   - Click the upload area or drag and drop an MP4 file
   - Videos with clear frontal face shots work best
   - Recommended: 25 fps, good lighting, minimal head movement
4. **Wait for processing**:
   - Backend extracts frames and runs face detection
   - Model processes video in ~4-second chunks
   - Progress is shown in real-time
5. **Watch with captions**:
   - Play the video to see synchronized captions
   - Green box shows face detection, cyan box shows mouth region
   - Captions appear based on timestamp ranges
6. **Export transcript**:
   - Download SRT file for use in video editors or players
   - Download TXT file for plain text transcript

## Setup Details

### Model Setup

The default model is `vsr_trlrs3vox2_base` (24.6% WER). Run the setup script to download:

```bash
python scripts/setup.py
```

**Available model variants:**

| Model | WER | Training Data | Size |
|---|---|---|---|
| `vsr_trlrs3_base` | 36.0% | LRS3 (438h) | Smallest |
| `vsr_trlrs3vox2_base` | 24.6% | LRS3 + VoxCeleb2 (1759h) | **Default** ✓ |
| `vsr_trlrs2lrs3vox2avsp_base` | 20.3% | LRS2 + LRS3 + VoxCeleb2 + AVSpeech (3291h) | Largest |

To use a different model, specify it in the setup script:
```bash
python scripts/setup.py --model vsr_trlrs2lrs3vox2avsp_base
```

## API Reference

### Backend Endpoints

The FastAPI backend runs on **http://localhost:8000** and provides:

#### `POST /upload/video`
Upload an MP4 file for lip-reading analysis.

**Request:**
- `Content-Type: multipart/form-data`
- `file`: MP4 video file

**Response:**
```json
{
  "fps": 25.0,
  "total_duration_ms": 17000,
  "processing_time_ms": 44000,
  "frames": [
    {
      "frame_idx": 0,
      "timestamp_ms": 0,
      "face_bbox": [120, 80, 350, 380],
      "mouth_bbox": [180, 280, 280, 340]
    },
    ...
  ],
  "captions": [
    {
      "start_ms": 0,
      "end_ms": 4000,
      "text": "HELLO MY NAME IS",
      "confidence": 82
    },
    ...
  ]
}
```

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

## Performance Notes

- **Processing time**: ~8-12 seconds per 4-second chunk on CPU
- **GPU acceleration**: 5-10x faster with CUDA-compatible GPU
- **Accuracy**: Best results with:
  - Clear frontal face view
  - Good lighting
  - Minimal head movement
  - Speaker facing camera directly
- **Video format**: MP4 recommended, 25 fps ideal
- **Resolution**: 640x480 or higher recommended

## Troubleshooting

### Common Issues

1. **"Auto-AVSR repo not found"** or **"Weight file not found"**
   - Run `python scripts/setup.py` to download the model and dependencies

2. **MediaPipe import error**
   - The app requires MediaPipe <0.10.21 (versions 0.10.21+ removed the solutions API)
   - Fix: `pip install "mediapipe>=0.10.0,<0.10.21"`

3. **Port already in use**
   - Backend (8000) or frontend (3000) port conflicts
   - Use the `start-app.sh` script which automatically kills existing processes
   - Or manually: `netstat -ano | grep :8000` and kill the PID

4. **Slow processing / inference**
   - Expected on CPU: ~8-12 seconds per 4-second video chunk
   - Use a CUDA GPU for 5-10x speedup
   - Ensure PyTorch detects your GPU: `python -c "import torch; print(torch.cuda.is_available())"`

5. **No face detected in video**
   - Video needs clear, frontal face view
   - Try the test video first: `tests/test_lipsync.mp4`
   - Check lighting and ensure speaker faces camera

6. **Frontend can't connect to backend**
   - Ensure backend is running on port 8000
   - Check `NEXT_PUBLIC_LIPSYNC_API_URL` in `web/.env.local` (should be `http://localhost:8000`)

7. **npm install fails**
   - Ensure Node.js 18+ is installed: `node --version`
   - Clear npm cache: `npm cache clean --force`
   - Delete `node_modules` and `package-lock.json`, then retry

### Best Practices

- **Video quality**: Use well-lit videos with clear face visibility
- **Face angle**: Frontal or near-frontal works best (±15° is okay)
- **Video length**: Shorter videos (10-30 seconds) process faster for testing
- **Test first**: Use the provided test video to verify setup before uploading your own

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

## Future Improvements

- [ ] Real-time webcam mode (currently video upload only)
- [ ] Multi-language support (currently English only)
- [ ] GPU acceleration auto-detection and optimization
- [ ] Batch video processing
- [ ] WebM and other video format support
- [ ] Progressive loading for large videos
- [ ] Speaker diarization for multi-speaker videos
- [ ] Adjustable confidence threshold filtering

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This project would not be possible without:
- **Auto-AVSR team** at Imperial College London and Meta for the pretrained model
- **MediaPipe** team at Google for face mesh detection
- **Next.js** and **FastAPI** communities for excellent frameworks
