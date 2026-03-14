"""
LipSync — FastAPI Backend Server
Serves the lip reading model and handles real-time video frame processing.
"""

import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from model.predict import LipReadingPredictor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("lipsync")

# ---------------------------------------------------------------------------
# Globals (loaded once at startup)
# ---------------------------------------------------------------------------
predictor: LipReadingPredictor | None = None


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup; release on shutdown."""
    global predictor
    try:
        predictor = LipReadingPredictor()
        logger.info("Lip reading model loaded successfully.")
    except Exception as exc:
        logger.warning("Model not available — running in skeleton mode: %s", exc)
        predictor = None
    yield
    predictor = None
    logger.info("LipSync server shut down.")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="LipSync API",
    description="Real-time lip reading inference server.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Health-check endpoint."""
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
    }


@app.post("/predict")
async def predict(payload: dict):
    """
    Accept a batch of lip ROI frames and return predicted text.

    Expected JSON body:
        {
            "frames": <base64-encoded numpy array or list of lists>,
            "num_frames": int
        }

    Returns:
        {
            "text": str,
            "confidence": float
        }
    """
    if predictor is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model not loaded. Train the model first."},
        )

    try:
        import base64
        import io

        frames_b64 = payload.get("frames")
        if frames_b64 is None:
            return JSONResponse(status_code=400, content={"error": "Missing 'frames' field."})

        frames_bytes = base64.b64decode(frames_b64)
        frames = np.load(io.BytesIO(frames_bytes), allow_pickle=False)

        text, confidence = predictor.predict(frames)
        return {"text": text, "confidence": float(confidence)}

    except Exception as exc:
        logger.exception("Prediction failed")
        return JSONResponse(status_code=500, content={"error": str(exc)})


# ---------------------------------------------------------------------------
# WebSocket Endpoint (streaming)
# ---------------------------------------------------------------------------
@app.websocket("/stream")
async def stream(ws: WebSocket):
    """
    Persistent WebSocket for streaming lip ROI frames and receiving predictions.

    Protocol:
        Client sends: binary numpy array bytes (lip ROI frame sequence)
        Server responds: JSON {"text": str, "confidence": float}
    """
    await ws.accept()
    logger.info("WebSocket client connected.")

    try:
        while True:
            data = await ws.receive_bytes()

            if predictor is None:
                await ws.send_json({"error": "Model not loaded."})
                continue

            frames = np.frombuffer(data, dtype=np.float32).reshape(-1, 50, 100, 1)
            text, confidence = predictor.predict(frames)
            await ws.send_json({"text": text, "confidence": float(confidence)})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as exc:
        logger.exception("WebSocket error")
        await ws.close(code=1011, reason=str(exc))

