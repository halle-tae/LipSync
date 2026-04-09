# LipSync Setup Session Summary
**Date:** April 8, 2026

## ✅ What's Working

### 1. Complete Setup
- ✅ Python 3.12.10 installed
- ✅ All dependencies installed: PyTorch 2.11.0 (CPU), MediaPipe, Gradio, FastAPI, OpenCV
- ✅ Auto-AVSR repository cloned to `auto_avsr/auto_avsr/` (nested structure)
- ✅ Model weights downloaded: `vsr_trlrs2lrs3vox2_base.pth` (956 MB)
- ✅ Demo test video available at `tests/test_video.mp4`

### 2. Successful Tests
- ✅ **Demo video test worked perfectly**
  - Predicted: "THANK YOU" (correct!)
  - Processed 75 frames in 21 seconds on CPU
  - Model has 250.4M parameters
  
- ✅ **Face detection works**
  - MediaPipe face mesh detecting faces correctly
  - Green boxes around face, blue boxes around mouth
  - Tested with `test_webcam.py` - detection boxes appear properly

### 3. Working Applications
- ✅ `main.py --video <path>` - Process video files
- ✅ `test_webcam.py` - Real-time face detection test
- ✅ `webcam_ui.py` - Manual capture UI with OpenCV
- ✅ `webcam_ui_debug.py` - Debug version that saves frames

### 4. Fixed Issues
- Fixed Unicode encoding errors in Windows console (replaced fancy characters with ASCII)
- Fixed Auto-AVSR import paths (nested `auto_avsr/auto_avsr/` structure)
- Replaced deprecated `torchvision.io.read_video` with PyAV
- Fixed multiple path issues in preprocessing and face detection modules

## ❌ Current Problem

### Lip Reading Accuracy is Poor
**Symptom:** The model predicts "THANK YOU" for almost every phrase spoken live

**Confidence Levels:** Unknown (need to check with debug version)

**Possible Causes:**
1. **Model bias** - "THANK YOU" was the demo video phrase, model may overfit to common training data phrases
2. **Low confidence predictions** - Model might be guessing when uncertain
3. **Mouth ROI alignment issues** - Affine transform might not be working correctly
4. **Frame rate mismatch** - Training data vs webcam capture rate differences
5. **Lighting/angle differences** - Training data conditions vs webcam conditions
6. **Insufficient frames** - Not capturing enough context (need 50+ frames, ~2 seconds)

## 🛠️ What Needs to be Done Tomorrow

### Immediate Next Steps

1. **Run Debug Session**
   ```bash
   python webcam_ui_debug.py
   ```
   - Capture 50+ frames while saying "HELLO" slowly
   - Press R to run inference and save debug output
   - Check confidence percentage shown
   - Inspect `debug_output/session_XXXXX/` folder
   - Examine mouth ROI images to verify they're clear

2. **Analyze Debug Output**
   - What confidence % is the model showing?
   - Do the mouth ROI images look correct?
   - Can you recognize the word from the saved ROI images?
   - Is the affine alignment working properly?

3. **Potential Fixes Based on Findings**

   **If confidence is LOW (< 50%):**
   - Model is just guessing
   - Try simpler words with distinct mouth shapes: "YES", "NO", "HELLO"
   - Increase frame count to 75+
   - Improve lighting
   
   **If confidence is HIGH (> 75%) but wrong prediction:**
   - Alignment issue - affine transform might be broken
   - Need to verify mean face landmarks are correct
   - Check if mouth ROI extraction is working properly
   
   **If mouth ROI images look blurry/misaligned:**
   - Face detection landmarks are jittery
   - Need to add temporal smoothing to landmarks
   - Possibly adjust the affine transform parameters

4. **Possible Solutions to Try**

   **A. Add confidence threshold filtering:**
   ```python
   if confidence < 0.6:
       return "Low confidence - speak more clearly"
   ```

   **B. Add landmark smoothing:**
   - Average landmarks over multiple frames
   - Reduce jitter in mouth ROI alignment

   **C. Try different model:**
   - Current: `vsr_trlrs3vox2_base.pth` (24.6% WER)
   - Could try: `vsr_trlrs2lrs3vox2avsp_base.pth` (20.3% WER, better accuracy)

   **D. Adjust preprocessing:**
   - Verify VideoTransform is using correct normalization
   - Check grayscale conversion
   - Verify 88x88 center crop is correct

   **E. Collect more frames:**
   - Currently using last 50 frames
   - Try using all 75 buffered frames

## 📁 Project Structure

```
C:\Users\halleteh\repos\LipSync\
├── auto_avsr/
│   └── auto_avsr/              # Nested! (Auto-AVSR code)
│       ├── datamodule/
│       ├── espnet/
│       ├── lightning.py
│       └── preparation/
├── backend/
│   ├── capture/
│   │   └── face_detect.py      # MediaPipe mouth extraction
│   ├── model/
│   │   ├── inference.py        # Inference pipeline
│   │   ├── loader.py           # Model loader
│   │   └── weights/
│   │       └── vsr_trlrs3vox2_base.pth  # 956 MB model
│   └── processing/
│       └── preprocess.py       # Frame preprocessing
├── frontend/
│   ├── gradio_app.py           # Original Gradio UI (streaming issues)
│   ├── gradio_app_fixed.py     # Fixed for Gradio 6.x (still has issues)
│   └── gradio_app_simple.py    # Manual capture version (didn't work)
├── tests/
│   └── test_video.mp4          # Demo video (works perfectly)
├── main.py                     # Main entry point (works for video files)
├── test_webcam.py              # Face detection test (works!)
├── webcam_ui.py                # Manual capture UI (works but poor accuracy)
├── webcam_ui_debug.py          # Debug version (saves frames for analysis)
├── requirements.txt
├── README.md
└── SETUP_GUIDE.md
```

## 🔧 Key Files to Know

### Working Scripts
- **`python main.py --video tests/test_video.mp4`** - Test with demo video (WORKS)
- **`python test_webcam.py`** - Test face detection (WORKS)
- **`python webcam_ui.py`** - Manual capture for live testing (WORKS but poor accuracy)
- **`python webcam_ui_debug.py`** - Debug mode with frame saving (START HERE TOMORROW)

### Gradio UIs (All have issues - skip for now)
- ~~`frontend/gradio_app.py`~~ - Streaming doesn't work in Gradio 6.x
- ~~`frontend/gradio_app_fixed.py`~~ - Port 7861, but streaming still broken
- ~~`frontend/gradio_app_simple.py`~~ - Manual mode, webcam component issues

### Important Code Files
- `backend/model/loader.py` - Model loading (fixed paths)
- `backend/model/inference.py` - Inference pipeline (fixed PyAV usage)
- `backend/capture/face_detect.py` - Mouth ROI extraction (fixed paths)
- `backend/processing/preprocess.py` - Frame preprocessing (fixed paths)

## 🐛 Known Issues

1. **Gradio 6.x streaming is broken** - Skip web UI for now, use OpenCV
2. **Windows console Unicode errors** - All fixed with ASCII replacements
3. **Auto-AVSR nested path** - All fixed, using `auto_avsr/auto_avsr/`
4. **Poor live lip reading accuracy** - Main issue to solve tomorrow

## 📊 Model Information

**Current Model:** `vsr_trlrs3vox2_base.pth`
- Word Error Rate: 24.6%
- Training data: LRS3 + VoxCeleb2 (1759 hours)
- Parameters: 250.4M
- Size: 956 MB
- Download: https://drive.google.com/file/d/1shcWXUK2iauRhW9NbwCc25FjU1CoMm8i/view

**Alternative Models (if needed):**
- `vsr_trlrs3_base.pth` - 36.0% WER (faster, less accurate)
- `vsr_trlrs2lrs3vox2avsp_base.pth` - 20.3% WER (best accuracy, same size)

## 💡 Tips for Tomorrow

1. **Start with debug mode** to understand what's happening
2. **Check confidence scores** - this will tell you if it's guessing vs alignment issue
3. **Inspect saved mouth ROI images** - verify they're clear and aligned
4. **Try simple words first** - "YES", "NO", "HELLO" have distinct mouth shapes
5. **Speak very slowly** with exaggerated mouth movements
6. **Use good lighting** and face camera directly
7. **Capture 50-75 frames** (about 2-3 seconds of speech)

## 🎯 Success Criteria

Before building a live streaming UI, we need:
- [ ] Confidence scores > 60% on average
- [ ] Can correctly predict at least 3 different simple words
- [ ] Mouth ROI images look clear and properly aligned
- [ ] Understand why "THANK YOU" appears so often (bias vs low confidence)

Once basic accuracy is acceptable, then build:
- Continuous live capture mode
- Better UI (either Gradio fix or custom web UI)
- Real-time streaming predictions

## 📝 Commands Quick Reference

```bash
# Test with demo video (WORKS PERFECTLY)
python main.py --video tests/test_video.mp4

# Test face detection (WORKS)
python test_webcam.py

# Manual capture with basic UI
python webcam_ui.py
# Controls: SPACE=capture, R=infer, C=clear, Q=quit

# Debug mode - START HERE TOMORROW
python webcam_ui_debug.py
# Controls: SPACE=capture, R=infer+save, S=save ROI, C=clear, Q=quit

# Check model is loaded
ls -lh backend/model/weights/

# View debug output
ls -la debug_output/
```

## 🔍 Debug Checklist for Tomorrow

1. Run `python webcam_ui_debug.py`
2. Press SPACE to start capturing
3. Say "HELLO" slowly (capture 50+ frames)
4. Press R to run inference
5. Note the confidence percentage
6. Open `debug_output/session_XXXXX/` folder
7. Check:
   - [ ] Are mouth ROI images clear?
   - [ ] Can you recognize "HELLO" from the images?
   - [ ] What's the confidence score?
   - [ ] Is the preprocessing correct (check preprocessed_frame_0.jpg)?
   - [ ] Are the mouth boxes stable or jittery in the live view?

## 🚀 End Goal

Build a real-time lip reading assistant that:
- Continuously captures webcam frames
- Detects faces and extracts mouth ROIs
- Runs inference every 1-2 seconds
- Displays live captions with confidence scores
- Works accurately for common phrases
- Has a nice UI (web or desktop)

**Current Status:** Setup complete, basic inference works, but live accuracy needs debugging.

---

**Next Session Start:** Run debug mode and analyze why predictions are poor!
