# PHASES.md — LipSync Development Plan

> Development roadmap for building the LipSync real-time lip reading assistant.
> Timeline: March 17 → April 15, 2026 (presentation day).

---

## Phase 0: Setup & Model Integration (March 17–19)

**Goal:** Get the Auto-AVSR model running locally in video-only mode and produce text output from a test video clip. Submit the project report and abstract by March 19th deadline.

### Tasks

1. **Environment setup**
   - Install Python 3.10, PyTorch 2.x, CUDA (if GPU available)
   - Install Auto-AVSR dependencies: `pytorch-lightning`, `sentencepiece`, `torchaudio`
   - Verify GPU/CPU inference works

2. **Download pretrained weights**
   - Grab the VSR (visual speech recognition) model checkpoint from the Auto-AVSR releases
   - These are the video-only weights — do NOT use the audio-visual or audio-only variants
   - Place weights in `backend/model/weights/`

3. **Run inference on a test video**
   - Use the Auto-AVSR `infer.py` script on a sample LRS3 clip or any talking-head video
   - Confirm text output is generated from video-only input
   - Document the inference command and any gotchas

4. **Submit project report + abstract**
   - Write the one-page report using the provided template
   - Create the one-pager PPT abstract
   - Email to Trinity.Vo@amd.com and Catarina.Alves@amd.com before March 19th EOD
   - Wait for confirmation email within 24 hours

### Deliverables
- [ ] Working Auto-AVSR inference on a test video
- [ ] Submitted report and abstract
- [ ] Documented setup instructions in README

---

## Phase 1: Webcam Pipeline (March 20–25)

**Goal:** Build the real-time webcam capture → face detection → mouth crop → model inference pipeline.

### Tasks

1. **Webcam capture module**
   - Use OpenCV `VideoCapture` to grab frames at 25 FPS (Auto-AVSR expects 25 FPS)
   - Handle camera permissions, resolution detection, and graceful failure
   - Create a frame buffer (sliding window of ~1-3 seconds of frames for temporal context)

2. **Face detection and mouth ROI extraction**
   - Integrate MediaPipe Face Mesh (lightweight, runs on CPU)
   - Extract the 468 face landmarks, isolate the mouth region landmarks
   - Crop a tight mouth ROI from each frame (typically 96x96 or 128x128 grayscale)
   - Handle cases: no face detected, multiple faces (pick largest/closest), face partially out of frame

3. **Preprocessing to match Auto-AVSR input format**
   - Convert mouth ROI to grayscale
   - Resize to the expected input dimensions (check Auto-AVSR config — likely 88x88 or 96x96)
   - Normalize pixel values to match training distribution
   - Stack frames into the temporal tensor format the model expects

4. **Inference loop**
   - Feed preprocessed frame sequences to the model at regular intervals
   - Start with batch inference every 1-2 seconds (not truly frame-by-frame — the model needs temporal context)
   - Measure inference latency — target is under 500ms per prediction on GPU, under 2s on CPU
   - Handle the model's output: decode token IDs to text using the sentencepiece tokenizer

5. **End-to-end test**
   - Point webcam at yourself speaking simple sentences
   - Verify text output appears in the terminal
   - Log latency numbers for each step (capture, detection, crop, inference, decode)

### Deliverables
- [ ] Webcam → face detect → mouth crop pipeline running in real time
- [ ] Model inference producing text from live webcam input
- [ ] Latency benchmarks documented

### Known Risks
- Auto-AVSR was trained on LRS3 (broadcast TV, clean frontal faces) — webcam quality and angles may degrade accuracy. This is expected and should be documented as a limitation.
- CPU-only inference may be too slow for real-time. If so, consider ONNX export or reducing frame buffer size.

---

## Phase 2: Caption Overlay UI (March 26–April 1)

**Goal:** Build the frontend captioning interface that displays predictions in a clean, accessible overlay.

### Tasks

1. **Choose frontend approach**
   - Option A: **Electron + React** — cross-platform desktop app, rich UI, good for demo
   - Option B: **PyQt6** — simpler, stays in Python, fewer dependencies
   - Option C: **Streamlit/Gradio** — fastest to build, web-based, easiest to demo live
   - Recommendation: **Gradio or Streamlit for the initial build** (fastest path to a working demo), then upgrade to Electron if time allows

2. **Caption display component**
   - Large, high-contrast text bar at the bottom of the screen
   - Text appears word-by-word or phrase-by-phrase as predictions come in
   - Smooth fade-in animation for new text
   - Auto-scroll and auto-clear after configurable timeout

3. **Webcam preview panel**
   - Show the live webcam feed with a bounding box around the detected face
   - Optionally highlight the mouth ROI being fed to the model
   - Include an "active" indicator showing when the model is processing

4. **Confidence indicator**
   - Display a simple confidence bar or color-coded text
   - Green = high confidence, yellow = medium, red = low
   - This comes from the model's output probabilities (beam search scores)

5. **Settings panel**
   - Font size slider (16px → 48px)
   - Font color picker
   - Background opacity slider
   - Caption position toggle (top / bottom / floating)
   - Dyslexia-friendly font toggle (OpenDyslexic)

6. **Backend ↔ Frontend communication**
   - If Gradio/Streamlit: direct Python integration, no IPC needed
   - If Electron: WebSocket server in Python, React client connects and receives caption updates as JSON messages

### Deliverables
- [ ] Working caption overlay displaying live predictions
- [ ] Settings panel with font, color, and position controls
- [ ] Confidence indicator visible per prediction
- [ ] Webcam preview with face detection bounding box

---

## Phase 3: Polish & Accessibility (April 2–7)

**Goal:** Refine the UI for visual appeal and accessibility compliance. Prepare for the presentation.

### Tasks

1. **Visual polish**
   - Consistent color scheme and typography
   - Smooth animations and transitions
   - Professional layout — think "product demo" not "school project"
   - Dark mode support

2. **Accessibility audit**
   - WCAG 2.1 AA contrast ratios on all text elements
   - Keyboard navigation for all controls (tab order, focus indicators)
   - Screen reader labels on interactive elements
   - Test with Windows Narrator or NVDA

3. **Session transcript feature**
   - Log all predictions with timestamps to an in-memory list
   - "Export transcript" button saves to `.txt` or `.srt` (subtitle format)
   - Show a scrollable transcript panel alongside the live caption

4. **Performance dashboard**
   - Small stats panel showing: inference latency (ms), FPS, total words detected
   - Useful for the live demo — judges can see it's actually running in real time

5. **Error handling and edge cases**
   - No camera detected → friendly error message with instructions
   - No face in frame → "Looking for a speaker..." placeholder
   - Model loading → progress bar
   - Low confidence → visual indicator, don't display garbage text

6. **Submit presentation deck**
   - Slides due to Trinity and Catarina by **April 3rd**
   - Structure: Problem → Solution → Demo plan → Technical architecture → Research → Impact

### Deliverables
- [ ] Polished, accessible UI
- [ ] Session transcript with export
- [ ] Performance dashboard
- [ ] Presentation deck submitted by April 3rd

---

## Phase 4: Demo Prep & Presentation (April 8–15)

**Goal:** Rehearse the demo, handle hardware logistics, and present on April 15th.

### Tasks

1. **Demo rehearsal**
   - Practice the full 8-minute presentation at least 3 times
   - Decide on time split (recommendation: 5 min presentation + 3 min Q&A)
   - State the time breakdown at the start as per the rules
   - Practice with the color-coded time signals: green = 3 min left, yellow = 1 min, red = done

2. **Demo contingency plan**
   - Record a backup demo video in case of live failure (webcam issues, model crash)
   - Have screenshots ready as a fallback
   - Test on the actual presentation laptop/hardware beforehand

3. **Hardware submission (if applicable)**
   - If using any hardware components, submit by **April 10th at 3PM** to a University Relations member in 1CV
   - If in another Canada location, contact Catarina and Trinity for logistics
   - This project is software-only, so likely not applicable

4. **Mentor check-in**
   - Use your assigned mentor for feedback on presentation flow and narrative
   - They can help with structuring the research component and handling Q&A
   - Remember: mentors advise on presentation, not on technical work

5. **Presentation structure**
   - **Opening (30s):** State time breakdown. Hook — "What if you could understand someone without hearing them?"
   - **Problem (1 min):** The gap in assistive tech. Audio captioning fails in noisy environments.
   - **Solution demo (2 min):** Live webcam demo of LipSync generating captions in real time.
   - **Technical deep dive (1 min):** Architecture diagram. Why Auto-AVSR. Pipeline overview.
   - **Research component (1 min):** Model limitations, WER benchmarks, comparison to alternatives, future work (fine-tuning on specific speakers, mobile deployment, multi-language support).
   - **Impact (30s):** Who this helps. Real-world scenarios. Potential integration with AMD hardware.
   - **Q&A (2 min):** Prepared for likely questions (accuracy, latency, why not train your own model, privacy concerns).

6. **Present on April 15th**
   - Live presentation at the Atrium in 1CV, broadcast to AMD Canada
   - Bring laptop, external webcam (if better quality), charger
   - Arrive early to test A/V setup

### Deliverables
- [ ] Rehearsed presentation (3+ run-throughs)
- [ ] Backup demo video recorded
- [ ] Presentation delivered on April 15th

---

## Timeline Summary

| Week | Dates | Phase | Key Milestone |
|---|---|---|---|
| 1 | Mar 17–19 | Phase 0 | Model running + report submitted |
| 2 | Mar 20–25 | Phase 1 | Live webcam → text pipeline working |
| 3 | Mar 26–Apr 1 | Phase 2 | Caption overlay UI functional |
| 4 | Apr 2–7 | Phase 3 | Polish, accessibility, transcript feature |
| — | Apr 3 | — | **Slides due** |
| 5 | Apr 8–15 | Phase 4 | Rehearsal and presentation |
| — | Apr 10, 3PM | — | **Hardware submission deadline** (if applicable) |
| — | Apr 15 | — | **Presentation day** |

---

## Research Component Notes

For the research/implications section of your presentation, cover:

1. **Model selection rationale** — Why Auto-AVSR over AV-HuBERT, LipNet, etc. (see model comparison in project research)
2. **Current limitations of VSR** — WER is ~20% even for SOTA models; lip reading is inherently ambiguous (many words look identical on the lips — "bat" vs "pat" vs "mat"); performance degrades with non-frontal angles, poor lighting, and speakers not in training distribution
3. **Potential improvements** — Speaker-adaptive fine-tuning (see Personalized Lip Reading, AAAI 2025), multi-language support, integration with partial audio for hybrid captioning
4. **AMD hardware relevance** — GPU acceleration for real-time inference, potential for ROCm optimization, edge deployment on AMD-powered laptops
5. **Societal impact** — Accessibility for 466M people worldwide with disabling hearing loss (WHO), complementing hearing aids in environments where they struggle
