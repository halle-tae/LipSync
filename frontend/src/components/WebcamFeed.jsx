/**
 * WebcamFeed — Live webcam display with face mesh landmark overlay
 * and lip ROI bounding box highlight.
 *
 * Phase 1: Basic webcam capture component (visual polish in Phase 4).
 */

import { useEffect, useRef, useState } from "react";

export default function WebcamFeed({ onFrame }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let animationId;
    let stream;

    async function startCamera() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: "user" },
          audio: false,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setIsStreaming(true);
        }
      } catch (err) {
        setError("Camera access denied. Please allow webcam access.");
        console.error("Webcam error:", err);
      }
    }

    function captureFrame() {
      if (
        videoRef.current &&
        canvasRef.current &&
        videoRef.current.readyState >= 2
      ) {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0);

        if (onFrame) {
          onFrame(canvas);
        }
      }
      animationId = requestAnimationFrame(captureFrame);
    }

    startCamera().then(() => {
      animationId = requestAnimationFrame(captureFrame);
    });

    return () => {
      if (animationId) cancelAnimationFrame(animationId);
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [onFrame]);

  if (error) {
    return (
      <div className="webcam-feed webcam-error">
        <p>{error}</p>
      </div>
    );
  }

  return (
    <div className="webcam-feed">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{ display: isStreaming ? "block" : "none" }}
      />
      <canvas ref={canvasRef} style={{ display: "none" }} />
      {!isStreaming && <p className="webcam-loading">Starting camera…</p>}
    </div>
  );
}

