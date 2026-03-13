# LipSync — Real-Time Lip Reading Assistant for the Hard of Hearing

## AMD Intern Innovation Showcase 2026

---

## Overview

**LipSync** is a real-time lip reading assistant that uses computer vision and deep learning to generate live captions from a speaker's lip movements. Designed for the hard-of-hearing community, LipSync works in environments where traditional audio-based captioning fails — noisy rooms, muted video calls, glass partitions, and public spaces without audio infrastructure.

The application captures a speaker's face via webcam, isolates the lip region using facial landmark detection, and feeds the sequence of lip frames into a trained deep learning model that predicts spoken words and phrases. The result is displayed as a clean, real-time caption overlay — bridging the communication gap without requiring any audio input.

---

## Problem Statement

Over **1.5 billion people** worldwide experience some degree of hearing loss (WHO, 2024), and this number is projected to rise to **2.5 billion by 2050**. While hearing aids and cochlear implants help many, they are not universally accessible or effective in all environments. Audio-based captioning tools (e.g., Google Live Transcribe, Otter.ai) rely on clear microphone input and fail in:

- **Noisy environments** (restaurants, factories, transit)
- **Physical barriers** (glass partitions, masks, distance)
- **Muted or silent contexts** (across a room, through a window, silent video playback)

There is a critical need for a **visual-input-based** communication aid that works independently of audio — one that reads lips, not sound.

---

## Solution

LipSync addresses this gap by providing a **webcam-powered, audio-independent captioning system** that:

1. **Detects faces and isolates lip regions** in real-time using MediaPipe Face Mesh
2. **Processes sequences of lip frames** through a deep learning model (CNN + LSTM / Transformer-based) trained on lip reading datasets
3. **Generates live text captions** displayed as an accessible overlay on screen
4. **Runs locally** for privacy — no audio is recorded, no data leaves the device

---

## Key Features

| Feature | Description |
|---|---|
| **Real-Time Lip Detection** | MediaPipe Face Mesh extracts 468 facial landmarks, isolating the lip ROI at 30+ FPS |
| **Deep Learning Lip Reading** | A trained sequence model (CNN+BiLSTM or Transformer) predicts words/phrases from lip movement sequences |
| **Live Caption Overlay** | Clean, accessible captions displayed in real-time on a web-based interface |
| **No Audio Required** | Entirely vision-based — works in silent, noisy, or muted environments |
| **Privacy-First Design** | All processing is local; no video or data is transmitted externally |
| **Interactive Demo Mode** | A guided demo that lets users test the system live with their own webcam |

---

## Technical Stack

- **Frontend:** React.js — responsive web interface with real-time caption rendering
- **Backend:** Python (Flask/FastAPI) — serves the ML model and handles video frame processing
- **Computer Vision:** MediaPipe Face Mesh, OpenCV — face detection, landmark extraction, lip ROI cropping
- **Deep Learning:** TensorFlow/Keras — CNN for spatial feature extraction + BiLSTM/Transformer for temporal sequence modeling
- **Datasets:** GRID Corpus, LRW (Lip Reading in the Wild), LRS2/LRS3 — standard academic lip reading benchmarks
- **Deployment:** Runs locally via Python backend + React frontend; no cloud dependency

---

## Research Component

LipSync is grounded in active academic research in visual speech recognition (VSR). The research component of this project will cover:

1. **Literature Review** — Survey of state-of-the-art lip reading models (LipNet, Visual Speech Recognition frameworks, AV-HuBERT) and their architectures, accuracy benchmarks, and limitations.
2. **Dataset Analysis** — Comparison of available lip reading datasets (GRID, LRW, LRS2/LRS3) in terms of vocabulary size, speaker diversity, and real-world applicability.
3. **Model Architecture Decisions** — Justification for the chosen architecture (CNN+BiLSTM vs. Transformer-based) based on accuracy, latency, and real-time feasibility.
4. **Accessibility Impact Research** — Statistics and studies on hearing loss prevalence, the limitations of current assistive technologies, and the potential real-world impact of visual speech recognition tools.
5. **Future Directions** — Multi-language lip reading, continuous sentence-level prediction, integration with AR glasses, and potential applications in healthcare and emergency services.

---

## Intended Impact

- **Accessibility:** Empowers hard-of-hearing individuals with a tool that works where audio captioning cannot
- **Inclusivity:** Promotes communication equity in workplaces, classrooms, and public spaces
- **Privacy:** Demonstrates that assistive technology can be powerful without compromising user privacy
- **Innovation:** Advances the practical application of visual speech recognition from research into a usable product

---

## Alignment with Judging Criteria

| Criteria | How LipSync Excels |
|---|---|
| **Technical Achievement** | Real-time ML inference pipeline combining CV (MediaPipe) + deep learning (CNN+BiLSTM/Transformer) with live webcam input |
| **Quality of Solution** | Solves a real, measurable problem — communication in audio-absent environments — with a working, interactive product |
| **Evidence of Research & Intended Impact** | Backed by WHO statistics, academic lip reading research, and a clear social impact narrative on 1.5B+ affected individuals |
| **Presentation Skills** | Live demo where the presenter speaks into the webcam and captions appear in real-time — the product *is* the presentation |
| **Visual Appeal** | Clean React UI with real-time face mesh visualization, lip ROI highlighting, and smooth caption overlay animations |

---

## Inspiration

This project draws inspiration from past Innovation Showcase winners:

- **Sign Bridge (2023 Winner)** — Demonstrated that real-time ML + webcam interaction + accessibility creates a compelling, award-winning combination. LipSync extends this vision from sign language *learning* to lip reading *translation*, tackling the inverse problem.
- **DeepDebug (Best Overall Winner)** — Showed that LLM/AI-driven tools with clear, measurable impact resonate with judges. LipSync applies the same principle: AI solving a real problem with demonstrable results.
- **Knowledge-Driven RAG (2024-25)** — Proved that a strong research foundation elevates a project. LipSync is grounded in published visual speech recognition research with quantifiable benchmarks.

---

## Team

- **Project Lead:** Halleteh
- **Competition:** AMD Intern Innovation Showcase 2026
- **Submission Deadline:** March 19, 2026
- **Presentation Date:** April 15, 2026 (if selected as finalist)

---

> *LipSync is more than a captioning tool — it is a step toward a world where communication is never limited by sound.*

