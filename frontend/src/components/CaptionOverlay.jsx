/**
 * CaptionOverlay — Displays real-time predicted captions from the lip reading model.
 *
 * Phase 1: Basic caption display component (animations & polish in Phase 4).
 */

import { useEffect, useRef } from "react";

export default function CaptionOverlay({ caption, confidence }) {
  const containerRef = useRef(null);

  useEffect(() => {
    // Auto-scroll to the latest caption
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [caption]);

  const confidenceColor =
    confidence >= 0.7
      ? "var(--color-confidence-high)"
      : confidence >= 0.4
        ? "var(--color-confidence-medium)"
        : "var(--color-confidence-low)";

  return (
    <div className="caption-overlay" ref={containerRef}>
      {caption ? (
        <>
          <p className="caption-text">{caption}</p>
          {confidence != null && (
            <span
              className="caption-confidence"
              style={{ color: confidenceColor }}
            >
              {(confidence * 100).toFixed(0)}% confidence
            </span>
          )}
        </>
      ) : (
        <p className="caption-placeholder">
          Waiting for lip reading predictions…
        </p>
      )}
    </div>
  );
}

