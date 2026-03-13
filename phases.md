# LipSync — Implementation Phases

## AMD Intern Innovation Showcase 2026

---

> Each phase builds on the previous one. By the end of Phase 6, LipSync is a complete, demo-ready, award-competitive project with a polished UI, trained model, live demo, research report, and presentation deck.

---

## Timeline Overview

| Phase | Name | Duration | Deadline Target |
|-------|------|----------|-----------------|
| **Phase 1** | Foundation & Data Pipeline | Days 1–3 | — |
| **Phase 2** | Lip Reading Model Training | Days 4–8 | — |
| **Phase 3** | Real-Time Inference Engine | Days 9–11 | — |
| **Phase 4** | Frontend & Visual Polish | Days 12–14 | — |
| **Phase 5** | Report, Abstract & Submission | Days 15–16 | **March 19, 2026** (submission deadline) |
| **Phase 6** | Presentation & Demo Prep | Post-selection | **April 3, 2026** (deck due) / **April 15, 2026** (presentation) |

---

## Phase 1: Foundation & Data Pipeline

### Objective
Set up the project structure, development environment, and a complete data pipeline that takes raw lip reading video data and produces model-ready training samples.

### Why This Matters
Past winners like **Sign Bridge** built their success on a solid ML pipeline (TensorFlow, MediaPipe, OpenCV). Without a clean, reliable data pipeline, everything downstream — model training, inference, and the live demo — breaks. This phase is the bedrock.

### Tasks

#### 1.1 Project Scaffolding
- Initialize the project repository with the following structure:
  ```
  lipsync/
  ├── backend/
  │   ├── app.py                  # Flask/FastAPI server
  │   ├── model/
  │   │   ├── train.py            # Model training script
  │   │   ├── predict.py          # Inference module
  │   │   └── lip_model.h5        # Saved model weights (after training)
  │   ├── data/
  │   │   ├── raw/                # Raw video files
  │   │   ├── processed/          # Extracted lip ROI frames
  │   │   └── alignments/         # Text alignment files (ground truth)
  │   └── utils/
  │       ├── face_mesh.py        # MediaPipe face mesh utilities
  │       ├── lip_extractor.py    # Lip ROI extraction logic
  │       └── data_loader.py      # Dataset loading & batching
  ├── frontend/
  │   ├── src/
  │   │   ├── App.jsx
  │   │   ├── components/
  │   │   │   ├── WebcamFeed.jsx
  │   │   │   ├── CaptionOverlay.jsx
  │   │   │   └── FaceMeshViz.jsx
  │   │   └── styles/
  │   └── package.json
  ├── notebooks/
  │   └── exploration.ipynb       # Data exploration & model prototyping
  ├── requirements.txt
  └── README.md
  ```
- Set up Python virtual environment with dependencies: `tensorflow`, `mediapipe`, `opencv-python`, `numpy`, `flask` (or `fastapi`), `scipy`
- Set up React frontend with `create-react-app` or Vite

#### 1.2 Dataset Acquisition & Exploration
- Download the **GRID Corpus** (primary dataset): 34 speakers, 1000 sentences each, with aligned text transcriptions
  - GRID is the standard benchmark for word-level lip reading and is well-structured for a time-boxed project
  - Optionally explore **LRW (Lip Reading in the Wild)** for extended vocabulary if time permits
- Explore the dataset in a Jupyter notebook:
  - Visualize sample videos and their alignment files
  - Understand the vocabulary (GRID uses a constrained grammar: `<command> <color> <preposition> <letter> <digit> <adverb>`)
  - Check video resolution, frame rate, and speaker diversity

#### 1.3 Face Detection & Lip ROI Extraction Pipeline
- Implement `face_mesh.py`:
  - Use **MediaPipe Face Mesh** to detect 468 facial landmarks per frame
  - Extract the **lip region of interest (ROI)** using landmarks 61-67 (outer lips) and 78-95 (inner lips)
  - Crop and resize the lip ROI to a fixed size (e.g., 100×50 pixels, grayscale)
- Implement `lip_extractor.py`:
  - Process each video in the GRID corpus → extract lip ROI frame sequences
  - Save as NumPy arrays (shape: `[num_frames, height, width, channels]`)
  - Pair each sequence with its text alignment (ground truth label)
- Implement `data_loader.py`:
  - Build a `tf.data.Dataset` or PyTorch `DataLoader` that:
    - Loads lip frame sequences and alignment labels
    - Pads/truncates sequences to a fixed length (e.g., 75 frames)
    - Applies data augmentation (random horizontal flip, brightness jitter, slight rotation)
    - Batches and shuffles data for training

#### 1.4 Validation
- Visually verify extracted lip ROIs against source videos (overlay landmarks on frames)
- Confirm data loader outputs correct shapes and label alignments
- Benchmark extraction speed (target: process full GRID corpus in <2 hours)

### Deliverables
- [ ] Working project repository with clean structure
- [ ] GRID corpus downloaded and explored
- [ ] Lip ROI extraction pipeline producing clean, aligned frame sequences
- [ ] Data loader ready for model training

---

## Phase 2: Lip Reading Model Training

### Objective
Design, train, and evaluate a deep learning model that takes a sequence of lip ROI frames and predicts the spoken text.

### Why This Matters
The model is the core technical achievement — the component that will be scrutinized most under the **Technical Achievement** judging criterion. **DeepDebug** won Best Overall by demonstrating strong AI capability. Our model must show real, measurable performance on academic benchmarks.

### Tasks

#### 2.1 Model Architecture Design
- Implement a **spatiotemporal model** with two stages:
  1. **Spatial Feature Extractor (CNN):**
     - 3D CNN or 2D CNN applied per-frame to extract spatial features from lip ROI images
     - Architecture: 3 × (Conv3D → BatchNorm → ReLU → MaxPool3D) layers
     - Input shape: `(batch, frames, height, width, channels)`
     - Output: Feature vector per frame `(batch, frames, features)`
  2. **Temporal Sequence Model (BiLSTM or Transformer):**
     - 2 × Bidirectional LSTM layers (256 units each) with dropout (0.5)
     - OR: Small Transformer encoder (4 heads, 2 layers) for a more modern approach
     - Output: Character-level predictions per time step
  3. **CTC Decoder:**
     - Use **Connectionist Temporal Classification (CTC)** loss to handle alignment between lip frame sequences and text output
     - CTC allows the model to learn without explicit frame-to-character alignment
- Alternative: Explore using a **pre-trained LipNet** architecture as a starting point and fine-tuning on GRID

#### 2.2 Training Pipeline
- Implement `train.py`:
  - Training loop with CTC loss
  - Optimizer: Adam (lr=1e-4) with learning rate scheduling (reduce on plateau)
  - Training/validation split: 80/20 by speaker (speaker-independent evaluation)
  - Metrics tracked per epoch:
    - CTC loss
    - **Word Error Rate (WER)** — primary metric
    - **Character Error Rate (CER)** — secondary metric
  - Model checkpointing (save best model by validation WER)
  - TensorBoard logging for training visualization
- Training configuration:
  - Batch size: 32 (adjust based on GPU memory)
  - Epochs: 50–100 (with early stopping, patience=10)
  - Target performance: **WER < 15%** on GRID (LipNet achieves ~4.8%, but with more compute; 15% is realistic for our timeline)

#### 2.3 Evaluation & Analysis
- Evaluate the best checkpoint on the held-out test set
- Generate a confusion matrix for predicted vs. actual characters
- Analyze failure cases:
  - Which phonemes/visemes are most confused? (e.g., "p" vs. "b" — visually identical)
  - Does performance vary by speaker?
- Document results with clear tables and charts (these go directly into the report and presentation)

#### 2.4 Model Optimization for Real-Time Inference
- Profile model inference time per sequence
- If inference is too slow (>200ms per sequence):
  - Reduce model size (fewer LSTM units, fewer CNN filters)
  - Apply TensorFlow Lite conversion or ONNX export
  - Quantize weights (INT8 post-training quantization)
- Target: **< 150ms inference latency** per 75-frame sequence on a standard laptop CPU/GPU

### Deliverables
- [ ] Trained lip reading model with documented WER/CER performance
- [ ] Training curves and evaluation metrics visualized
- [ ] Model optimized for real-time inference (<150ms per sequence)
- [ ] Saved model weights (`lip_model.h5` or `.onnx`)

---

## Phase 3: Real-Time Inference Engine

### Objective
Build a backend that captures live webcam frames, extracts lip ROIs in real-time, runs them through the trained model, and returns predicted text — all with low enough latency for live captioning.

### Why This Matters
The live demo is what separates finalists from winners. **Sign Bridge** won in part because the webcam quiz was interactive and engaging. LipSync needs to process a webcam feed and produce captions fast enough that the delay feels natural — this is the engineering challenge.

### Tasks

#### 3.1 Real-Time Face & Lip Processing Pipeline
- Implement a **streaming pipeline**:
  1. Capture webcam frames via OpenCV (`cv2.VideoCapture`)
  2. Run MediaPipe Face Mesh on each frame (this runs at ~30 FPS natively)
  3. Extract lip ROI from detected landmarks
  4. Accumulate a **sliding window** of lip ROI frames (e.g., 75 frames ≈ 2.5 seconds at 30 FPS)
  5. Feed the window to the model for prediction
  6. Apply a **sliding window with overlap** (e.g., shift by 15 frames, re-predict) for smoother captions
- Handle edge cases:
  - No face detected → display "No face detected" message
  - Multiple faces → default to the largest/closest face
  - Poor lighting → warn the user with a visual indicator

#### 3.2 Backend API (Flask/FastAPI)
- Implement `app.py` with the following endpoints:
  - `POST /predict` — accepts a batch of lip ROI frames (as base64 or binary), returns predicted text and confidence score
  - `GET /health` — health check endpoint
  - `WebSocket /stream` (preferred) — maintains a persistent connection for streaming frames and receiving predictions with minimal latency
- Load the trained model once at startup and keep it in memory
- Add **prediction smoothing**:
  - Buffer the last N predictions
  - Use a voting/averaging mechanism to reduce flickering between predictions
  - Only update the displayed caption when confidence exceeds a threshold (e.g., 0.7)

#### 3.3 Performance Optimization
- Profile the full pipeline end-to-end:
  - Frame capture → face mesh → lip extraction → model inference → response
  - Target: **< 300ms total latency** (capture to caption)
- Optimizations:
  - Run face mesh and model inference in **separate threads** (producer-consumer pattern)
  - Use a frame buffer to decouple capture rate from inference rate
  - Pre-allocate NumPy arrays to avoid repeated memory allocation
  - If using GPU: ensure TensorFlow uses GPU for inference (`tf.device('/GPU:0')`)

#### 3.4 Integration Testing
- Test with a live webcam speaking GRID vocabulary words
- Measure:
  - End-to-end latency (frame to caption)
  - Prediction accuracy in real-world conditions (lighting, angles, distances)
  - Stability over extended sessions (10+ minutes without crashes or memory leaks)

### Deliverables
- [ ] Real-time inference pipeline processing live webcam at 30 FPS
- [ ] Backend API serving predictions with <300ms latency
- [ ] Prediction smoothing eliminating caption flickering
- [ ] Stable over extended sessions

---

## Phase 4: Frontend & Visual Polish

### Objective
Build a polished, visually compelling React frontend that showcases the real-time lip reading in action — designed to win the **Best Visual Award**.

### Why This Matters
The **Best Visual Award** explicitly judges "interfaces, dashboards, visual storytelling, and overall aesthetics." The **Knowledge-Driven RAG** project won partly on strong visual presentation. Our frontend must be beautiful, intuitive, and demo-ready. The judges' first impression is visual.

### Tasks

#### 4.1 Core UI Components
- **`WebcamFeed.jsx`** — Live webcam display with:
  - Face mesh landmark overlay (translucent dots showing the 468 landmarks)
  - Lip ROI highlight (glowing bounding box around the detected lip region)
  - Smooth animations on detection (box appears/disappears with fade transitions)
- **`CaptionOverlay.jsx`** — Real-time caption display:
  - Large, high-contrast text at the bottom of the screen (like subtitles)
  - Smooth text transitions (fade in new words, slide out old ones)
  - Confidence indicator (subtle color coding: green = high confidence, yellow = medium)
  - Scrolling history of recent captions
- **`FaceMeshViz.jsx`** — Optional visualization panel:
  - 3D face mesh rendering (using Three.js or Canvas)
  - Highlights the lip region landmarks in a distinct color
  - Provides a "tech showcase" visual that impresses judges

#### 4.2 Application Layout & UX
- **Main screen layout:**
  ```
  ┌─────────────────────────────────────────────┐
  │  LipSync — Real-Time Lip Reading Assistant   │
  │  ┌──────────────────┐  ┌──────────────────┐ │
  │  │                  │  │  Face Mesh 3D     │ │
  │  │   Webcam Feed    │  │  Visualization    │ │
  │  │   + Lip ROI      │  │                   │ │
  │  │   Highlight       │  │                   │ │
  │  └──────────────────┘  └──────────────────┘ │
  │  ┌──────────────────────────────────────────┐│
  │  │  ▸ "set blue in a one again"             ││
  │  │    Caption Overlay — Live Transcription   ││
  │  └──────────────────────────────────────────┘│
  │  ┌──────────────────────────────────────────┐│
  │  │  Status: ● Connected  |  Latency: 120ms ││
  │  │  Confidence: 87%  |  FPS: 30             ││
  │  └──────────────────────────────────────────┘│
  └─────────────────────────────────────────────┘
  ```
- **Design system:**
  - Dark theme (professional, reduces glare in demo environment)
  - AMD red (#ED1C24) as accent color — subtly aligns with AMD branding
  - Clean sans-serif typography (Inter or Roboto)
  - Glassmorphism-style panels with subtle blur and transparency
  - Smooth micro-animations (loading states, transitions, hover effects)

#### 4.3 Interactive Demo Mode
- Build a **guided demo flow** for the presentation:
  1. Welcome screen with project title, team name, and "Start Demo" button
  2. Webcam permission prompt with instructions
  3. Calibration step — "Face the camera and speak slowly" with visual guides
  4. Live captioning mode with all visualizations active
  5. Results summary — show accuracy, average latency, and session stats
- This guided flow ensures the demo runs smoothly during the 8-minute presentation

#### 4.4 Visual Polish & Accessibility
- Responsive layout (works on projector resolutions for the live presentation)
- Keyboard accessibility and screen reader support (practice what we preach — this is an accessibility project)
- Loading states and error handling with friendly messages
- Subtle particle/ambient animations in the background for visual appeal
- AMD logo placement (optional, shows company alignment)

### Deliverables
- [ ] Polished React frontend with webcam feed, face mesh visualization, and caption overlay
- [ ] Interactive demo mode with guided flow
- [ ] Dark theme with AMD accent colors and glassmorphism design
- [ ] Responsive, accessible, and visually impressive

---

## Phase 5: Report, Abstract & Submission

### Objective
Produce all required submission materials — the one-page written report, one-page PowerPoint abstract, and optional source code package — and submit before the **March 19, 2026** deadline.

### Why This Matters
Per the FAQ: "You must submit a report of your project following the template attached to the email alongside an abstract summarizing your report details." The report is what the vetting team uses to select the top 10 finalists. A weak report means no presentation opportunity — regardless of how good the project is. **Sign Bridge** and **DeepDebug** both had compelling written narratives that communicated impact clearly.

### Tasks

#### 5.1 Written Report (One Page, Text Only)
- Follow the provided template exactly
- Structure the report to hit all five judging criteria:
  1. **Problem Statement** (2–3 sentences): Communication barrier for hard-of-hearing individuals in audio-absent environments. Cite WHO statistics (1.5B affected, projected 2.5B by 2050).
  2. **Solution Overview** (3–4 sentences): LipSync uses computer vision (MediaPipe) and deep learning (CNN+BiLSTM with CTC loss) to generate real-time captions from lip movements via webcam. No audio required. Runs locally for privacy.
  3. **Technical Achievement** (3–4 sentences): Model trained on GRID corpus achieving X% WER. Real-time inference pipeline at <300ms latency. Sliding window prediction with smoothing for stable captions.
  4. **Research Foundation** (2–3 sentences): Built on academic lip reading research (LipNet, AV-HuBERT). Addresses limitations of audio-dependent captioning. Explores viseme confusion patterns and speaker-independent generalization.
  5. **Impact & Future Work** (2–3 sentences): Empowers hard-of-hearing individuals in noisy/silent environments. Future: multi-language support, sentence-level prediction, AR glasses integration, healthcare applications.
- Keep it concise — the report CANNOT exceed one page of text
- Attach a second page with:
  - Screenshot of the UI with face mesh + caption overlay active
  - Architecture diagram (data flow from webcam → model → caption)
  - Key metrics table (WER, CER, latency, FPS)

#### 5.2 Abstract (One-Pager PowerPoint)
- Single PowerPoint slide summarizing the project:
  - Project name and team
  - Problem in one line
  - Solution in one line
  - Key visual (UI screenshot or architecture diagram)
  - Key metric (WER and latency)
  - Impact statement
- Design should be visually clean and immediately communicate the project's value

#### 5.3 Source Code Package (Optional but Recommended)
- Per the FAQ: "source code/Proof of Concept submission is optional. Please provide detailed instructions or notes regarding any modifications required for proper functionality."
- Package the repository with:
  - Clear `README.md` with setup instructions
  - `requirements.txt` with pinned versions
  - Pre-trained model weights (or a download script)
  - A note about any hardware requirements (GPU recommended for training, not required for inference)

#### 5.4 Submission
- Email all materials to **both** Trinity.Vo@amd.com and Catarina.Alves@amd.com
- Confirm receipt — per the FAQ: "you must receive a confirmation email acknowledging it, if you do not receive it within 24 hours make sure to check if you sent it to the right emails"
- Submit **before** March 19, 2026 — there is no accommodation for late submissions

### Deliverables
- [ ] One-page written report following the template
- [ ] One-page PowerPoint abstract
- [ ] Source code package with setup instructions
- [ ] Submission email sent and confirmation received

---

## Phase 6: Presentation & Demo Prep (Finalists Only)

### Objective
Prepare a compelling 8-minute live presentation with a working product demo, polished slide deck, and practiced Q&A — designed to win **Best Overall**, **Best Visual**, or **Best Pitch**.

### Why This Matters
The presentation is where winners are made. **Sign Bridge** used a live webcam demo during the presentation — the audience watched ASL signs being recognized in real-time. **DeepDebug** articulated a clear problem→solution→impact narrative that resonated with judges. We need both: a jaw-dropping demo AND a compelling story. Per the rules, presentations will be **live-streamed** and judged by a panel of 4 on: Technical Achievement, Presentation Skills, Quality of Solution, Evidence of Research and Intended Impact, Visual Appeal.

### Tasks

#### 6.1 Slide Deck (Due April 3, 2026)
- Structure the presentation deck (8–10 slides):
  1. **Title Slide** — Project name, team, AMD branding, one-line tagline
  2. **The Problem** — "1.5 billion people have hearing loss. Audio captioning fails in noisy, muted, and barrier environments." (Powerful stat + visual)
  3. **The Solution** — LipSync overview with architecture diagram. "Captions from lips, not sound."
  4. **How It Works** — Technical deep-dive: face mesh → lip ROI → CNN+BiLSTM → CTC → text. Clean animated diagram.
  5. **Live Demo** — Switch to the app. Speak into the webcam. Captions appear in real-time. (This is the climax of the presentation.)
  6. **Results & Metrics** — WER, CER, latency, FPS. Comparison against baselines. Visual charts.
  7. **Research Foundation** — Academic grounding: LipNet, viseme analysis, speaker-independent evaluation.
  8. **Impact & Future Work** — Who benefits, how it scales, next steps (multi-language, AR glasses, healthcare).
  9. **Thank You / Q&A** — Contact info, acknowledgments.
- Design: consistent with the app's dark theme + AMD red accents, clean typography, minimal text per slide, high-impact visuals
- Submit to Trinity.Vo@amd.com and Catarina.Alves@amd.com by **April 3, 2026**

#### 6.2 Time Allocation Strategy
- Recommended breakdown (state at the start of the presentation, per the rules):
  - **0:00–1:30** — Problem & Solution (slides 1–3)
  - **1:30–2:30** — How It Works (slide 4)
  - **2:30–4:30** — **Live Demo** (slide 5 — switch to app, 2 full minutes of live lip reading)
  - **4:30–5:30** — Results & Research (slides 6–7)
  - **5:30–6:00** — Impact & Future Work (slide 8)
  - **6:00–8:00** — Q&A (2 minutes)
- Watch for color-coded flags:
  - 🟩 Green at 5:00 = 3 minutes remaining → wrap up demo, move to results
  - 🟨 Yellow at 7:00 = 1 minute remaining → finish current answer
  - 🔴 Red at 8:00 = time's up → stop immediately

#### 6.3 Live Demo Preparation
- **Test the demo in the exact presentation environment:**
  - Lighting conditions in the 1CV Atrium
  - Webcam angle and distance from the screen
  - Network connectivity (LipSync runs locally, so no network dependency — advantage!)
  - Projector resolution compatibility
- **Prepare fallback options:**
  - Pre-recorded video of the demo in case the webcam fails
  - Screenshots of the running app as static slide backup
- **Practice the demo transition:**
  - Smooth switch from slides → app → slides
  - Know exactly what words/phrases to speak for the most impressive recognition results
  - Use GRID vocabulary words that the model handles best for the demo

#### 6.4 Mentor Sessions
- Leverage the assigned mentor (past showcase winners):
  - Review the slide deck for clarity and flow
  - Practice the full 8-minute presentation at least 3 times
  - Get feedback on storytelling, pacing, and handling Q&A
  - Ask about common judge questions and how to handle technical deep-dive questions

#### 6.5 Anticipate Q&A
- Prepare answers for likely judge questions:
  - *"How does this compare to existing audio-based captioning?"* → Works where audio captioning fails (noisy, muted, barriers). Complementary, not a replacement.
  - *"What's the accuracy in real-world conditions vs. your benchmark?"* → GRID is controlled; real-world is harder. We see ~X% WER degradation. Future work addresses this.
  - *"Can it handle multiple speakers?"* → Currently single-speaker. Multi-face detection is implemented; multi-speaker lip reading is active research.
  - *"What about privacy?"* → Entirely local. No video is recorded or transmitted. This is a core design principle.
  - *"Why not just use audio?"* → Audio isn't always available. 1.5B people with hearing loss. Glass partitions, noisy factories, muted video calls.
  - *"How would this scale to full vocabulary?"* → Current model uses GRID's constrained vocabulary. Scaling to open vocabulary requires larger datasets (LRS2/LRS3) and more compute. Transformer-based architectures show promise.

### Deliverables
- [ ] Polished slide deck submitted by April 3, 2026
- [ ] Live demo tested in presentation environment
- [ ] Fallback demo (pre-recorded video) prepared
- [ ] Full presentation rehearsed 3+ times with mentor feedback
- [ ] Q&A answers prepared for 6+ anticipated questions

---

## Summary: Phase-by-Phase Progression

```
Phase 1: Foundation & Data Pipeline
   └─→ Clean project, working data pipeline, lip ROI extraction
        │
Phase 2: Lip Reading Model Training
   └─→ Trained model with measurable WER/CER, optimized for speed
        │
Phase 3: Real-Time Inference Engine
   └─→ Live webcam → model → caption pipeline running at <300ms
        │
Phase 4: Frontend & Visual Polish
   └─→ Beautiful React UI with face mesh viz, captions, demo mode
        │
Phase 5: Report, Abstract & Submission
   └─→ All materials submitted by March 19, 2026
        │
Phase 6: Presentation & Demo Prep (if selected)
   └─→ Winning presentation delivered on April 15, 2026
```

---

> **The goal at every phase: build something that is technically impressive, visually stunning, socially impactful, and demo-ready. That combination is what wins.**

