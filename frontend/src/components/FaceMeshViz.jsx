/**
 * FaceMeshViz — Placeholder for the 3D face mesh visualisation panel.
 *
 * Phase 1: Static placeholder (Three.js / Canvas implementation in Phase 4).
 */

export default function FaceMeshViz({ landmarks }) {
  return (
    <div className="face-mesh-viz">
      <h3>Face Mesh</h3>
      {landmarks ? (
        <p className="face-mesh-status face-mesh-active">
          ● Face detected — {landmarks.length} landmarks
        </p>
      ) : (
        <p className="face-mesh-status face-mesh-inactive">
          ○ No face detected
        </p>
      )}
      <div className="face-mesh-placeholder">
        <span>3D visualisation — coming in Phase 4</span>
      </div>
    </div>
  );
}

