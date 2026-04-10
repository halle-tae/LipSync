# Video Upload Caption Pivot — Implementation Plan

## Overview

Pivot the app from live webcam lip-reading to **video upload mode**: user uploads an MP4, backend chunks it into segments, runs offline inference on each chunk, returns timestamped captions, and the frontend plays the video with captions appearing in sync — like subtitles.

## Why

The offline inference pipeline produces dramatically better results than live webcam because the model gets full temporal context per chunk (~100 frames) instead of tiny 2-second windows. We proved this with `test_lipsync.mp4`:

- **Offline result:** "HELLO MY NAME IS HEALTHY AND TODAY I WANT TO TALK ABOUT HOW THIS APP WORKS" (near-perfect)
- **Live webcam:** fragmented, low-confidence, mostly blank captions

## Step 0: Create the branch

```bash
git checkout main
git pull
git checkout -b feature/video-upload-captions
```

## Step 1: Backend — New inference method

**File:** `backend/model/inference.py`

Add a `predict_from_numpy_frames(frames: np.ndarray)` method to `InferencePipeline` that:
- Takes a raw numpy array of RGB frames `[T, H, W, 3]` directly (no file I/O)
- Runs the same pipeline: landmarks detection → mouth crop → preprocess → model inference
- Returns `{ text, confidence, latency_ms, num_frames }`

This lets the upload endpoint pass pre-sliced frame chunks without re-reading the video file.

## Step 2: Backend — Upload endpoint with bounding box data

**File:** `backend/api_server.py`

Add `POST /upload/video` endpoint:

1. Accept `multipart/form-data` with an `.mp4` file (`UploadFile` from FastAPI)
2. Save to temp file, read all frames + FPS with OpenCV
3. Chunk into **~4-second windows** (~100 frames at 25fps), with **~3-second stride** (1 second overlap so words at boundaries aren't lost)
4. For **every frame**, run face detection and return the face bounding box and mouth bounding box coordinates. The `FaceMeshMouthExtractor.extract()` method already returns both `face_bbox` and `mouth_bbox` as `(x1, y1, x2, y2)` tuples — see `backend/capture/face_detect.py` `FaceMouthDetection` dataclass.
5. Call `predict_from_numpy_frames()` on each chunk for the caption text
6. Deduplicate overlapping text between adjacent chunks
7. Return JSON:

```json
{
  "fps": 25.0,
  "total_duration_ms": 17000,
  "processing_time_ms": 44000,
  "frames": [
    { "frame_idx": 0, "timestamp_ms": 0, "face_bbox": [120, 80, 350, 380], "mouth_bbox": [180, 280, 280, 340] },
    { "frame_idx": 1, "timestamp_ms": 40, "face_bbox": [121, 81, 351, 381], "mouth_bbox": [181, 281, 281, 341] },
    ...
  ],
  "captions": [
    { "start_ms": 0, "end_ms": 4000, "text": "HELLO MY NAME IS", "confidence": 82 },
    { "start_ms": 3000, "end_ms": 7000, "text": "HALLE AND TODAY I WANT", "confidence": 75 }
  ]
}
```

The `frames` array has one entry per video frame with the bounding box coordinates. The `captions` array has one entry per inference chunk with the predicted text.

Note: `python-multipart` must be installed for FastAPI file uploads (`pip install python-multipart`).

### Bounding box pipeline

For each frame in the video:
1. Run `FaceMeshMouthExtractor.extract(frame_bgr)` — this is the same detector used in the live pipeline
2. If a face is detected, record `face_bbox` and `mouth_bbox` from the `FaceMouthDetection` result
3. If no face is detected for a frame, set both to `null`
4. The bounding boxes are in **pixel coordinates of the original video resolution** (not the 256x256 warped space), so the frontend can scale them to the displayed video size

## Step 3: Frontend — Replace webcam with upload + playback

**File:** `web/src/app/page.tsx`

### Remove
- `getUserMedia` / webcam stream logic
- Frame capture interval (canvas → base64 → buffer)
- Polling interval (`/session/frame` calls)
- Session start/stop API calls
- Webcam-only state: `streaming`, `cameraError`, `faceDetected`, `mirrorVideo`, `showReticle`, `micEnabled`

### Add
- **New state:** `videoFile`, `videoUrl` (object URL for playback), `processing`, `progress` ("Processing chunk 2/5..."), `captions` (array from backend response), `activeCaption`, `frameBoxes` (array of per-frame bounding box data), `activeBoxes` (current face_bbox + mouth_bbox for the current frame)
- **Upload zone:** Drag-and-drop area or file picker button in place of the webcam preview area
- **Processing flow:** On file select → `POST /upload/video` with `FormData` → show progress spinner → populate captions + frame boxes on completion
- **Video player:** `<video>` element with object URL as `src`, standard playback controls
- **Bounding box overlay:** An absolutely-positioned `<canvas>` or SVG layer on top of the `<video>` element, same dimensions. On each `timeupdate` (or `requestAnimationFrame` for smoother updates), look up the current frame index from `video.currentTime * fps`, find the matching entry in `frameBoxes`, and draw:
  - **Green rectangle** for the face bounding box
  - **Cyan/blue rectangle** for the mouth bounding box
  - Scale the pixel coordinates from the original video resolution to the displayed video size
- **Caption sync:** `onTimeUpdate` handler that matches `video.currentTime * 1000` to the caption whose `[start_ms, end_ms]` range contains it:

```typescript
const currentMs = video.currentTime * 1000;
const frameIdx = Math.floor(video.currentTime * fps);
const active = captions.find(c => currentMs >= c.start_ms && currentMs <= c.end_ms);
setActiveCaption(active?.text ?? "");
setActiveBoxes(frameBoxes[frameIdx] ?? null);
```

For smoother bounding box updates (since `timeupdate` only fires ~4x/sec), use a `requestAnimationFrame` loop while the video is playing to redraw the canvas overlay at display refresh rate.

### Keep as-is
- Transcript panel (populate from returned `chunks` array instead of live polling)
- SRT/TXT export functions (just change the data source)
- Dark mode, settings panel, fonts, overall `globals.css` styling
- Confidence display and animated number hook

### UI states flow

```
Idle → [Upload Video] → Processing (spinner + "Processing chunk 2/5...") → Ready
  Ready → [Play] with synced captions overlaid on video
        → [Export SRT] / [Export TXT]
        → [Upload New] to reset
```

## Step 4: Test end-to-end

```bash
# Terminal 1: backend
python backend/api_server.py

# Terminal 2: frontend
cd web
npm run dev
```

1. Open http://localhost:3000
2. Upload `tests/test_lipsync.mp4`
3. Wait ~40-50 seconds for CPU processing
4. Play video — captions should appear synced to speech
5. Test SRT export

## Step 5: Commit

```bash
git add -A
git commit -m "Pivot to video upload mode with timestamped captions"
git push -u origin feature/video-upload-captions
```

## Important context

- **Model weights** are at `backend/model/weights/vsr_trlrs3vox2_base.pth` (24.6% WER, already downloaded)
- **Auto-AVSR repo** is at `auto_avsr/auto_avsr/` (note the double nesting — paths were already fixed in `loader.py`, `inference.py`, `preprocess.py`, `face_detect.py`)
- The `torchvision.io.read_video` call was already replaced with OpenCV `cv2.VideoCapture` in `inference.py`
- Unicode print characters were already replaced with ASCII in `scripts/test_inference.py` for Windows compatibility
- Running on **CPU only** (no GPU) — inference takes ~8-10 seconds per 4-second chunk
- Existing live webcam endpoints (`/session/start`, `/session/frame`, `/session/stop`) can stay in the code — the upload endpoint is additive
- The frontend has been moved from `LipReader_Frontend+Working/LipReader/web/` to `web/` at the repo root (cleaner structure)
- The `LipReader_Frontend+Working/` folder has been deleted
- The `web/` folder needs `npm install` before running (no `node_modules` was copied)

## Time estimate

| Task | Time |
|------|------|
| Backend endpoint + chunking | ~30 min |
| Frontend upload + playback + caption sync | ~1-2 hours |
| Testing + polish | ~30 min |
| **Total** | **~2-3 hours** |
