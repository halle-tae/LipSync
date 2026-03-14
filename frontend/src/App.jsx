/**
 * LipSync — Main Application
 *
 * Phase 1: Basic layout with webcam feed, caption overlay, and face mesh
 * placeholder. Full visual polish comes in Phase 4.
 */

import { useState, useCallback } from "react";
import WebcamFeed from "./components/WebcamFeed";
import CaptionOverlay from "./components/CaptionOverlay";
import FaceMeshViz from "./components/FaceMeshViz";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function App() {
  const [caption, setCaption] = useState("");
  const [confidence, setConfidence] = useState(null);
  const [landmarks, setLandmarks] = useState(null);
  const [status, setStatus] = useState("disconnected");
  const [latency, setLatency] = useState(null);

  const handleFrame = useCallback(
    (_canvas) => {
      // Phase 3: Send frames to backend for real-time inference.
      // For now, this is a no-op placeholder.
    },
    []
  );

  return (
    <div className="app">
      <header className="app-header">
        <h1>
          Lip<span className="accent">Sync</span>
        </h1>
        <p className="subtitle">Real-Time Lip Reading Assistant</p>
      </header>

      <main className="app-main">
        <div className="panels">
          <div className="panel panel-webcam">
            <WebcamFeed onFrame={handleFrame} />
          </div>
          <div className="panel panel-viz">
            <FaceMeshViz landmarks={landmarks} />
          </div>
        </div>

        <CaptionOverlay caption={caption} confidence={confidence} />

        <div className="status-bar">
          <span className={`status-dot ${status}`} />
          <span>
            {status === "connected" ? "Connected" : "Disconnected"}
          </span>
          {latency != null && <span>Latency: {latency}ms</span>}
          {confidence != null && (
            <span>Confidence: {(confidence * 100).toFixed(0)}%</span>
          )}
        </div>
      </main>

      <footer className="app-footer">
        <p>LipSync — AMD Intern Innovation Showcase 2026</p>
      </footer>
    </div>
  );
}
