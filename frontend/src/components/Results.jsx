import React from "react";

function pct(v) {
  return `${(v * 100).toFixed(1)}%`;
}

export default function Results({ data }) {
  const isParkinson = data.prediction === "parkinson";

  return (
    <>
      {/* ── Prediction banner ── */}
      <div className="card" style={{ textAlign: "center" }}>
        <span className={`result-badge ${isParkinson ? "badge-parkinson" : "badge-healthy"}`}>
          {isParkinson ? "⚠ Parkinson Indicators" : "✓ Healthy Range"}
        </span>
        <p style={{ marginTop: ".6rem", color: "var(--muted)", fontSize: ".9rem" }}>
          {data.label}
        </p>

        {/* probability bar */}
        <div style={{ maxWidth: 360, margin: "1rem auto 0" }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: ".8rem", color: "var(--muted)" }}>
            <span>Healthy</span>
            <span>Parkinson</span>
          </div>
          <div className="prob-bar-track">
            <div
              className="prob-bar-fill"
              style={{
                width: pct(data.probability),
                background: isParkinson ? "var(--red)" : "var(--green)",
              }}
            />
          </div>
          <p style={{ marginTop: ".35rem", fontSize: ".85rem" }}>
            CNN probability: <strong>{pct(data.cnn_probability)}</strong> &nbsp;|&nbsp;
            Confidence: <strong>{pct(data.confidence)}</strong>
          </p>
        </div>
      </div>

      {/* ── Visualisations + features ── */}
      <div className="results-grid">
        {/* Grad-CAM */}
        <div className="card">
          <h2>Grad-CAM Heatmap</h2>
          <p style={{ fontSize: ".82rem", color: "var(--muted)", marginBottom: ".5rem" }}>
            Regions the model focused on most
          </p>
          <img className="heatmap-img" src={data.heatmap} alt="Grad-CAM overlay" />
        </div>

        {/* Speed heatmap */}
        <div className="card">
          <h2>Speed Heatmap</h2>
          <p style={{ fontSize: ".82rem", color: "var(--muted)", marginBottom: ".5rem" }}>
            Green = steady, Red = erratic drawing speed
          </p>
          <img className="heatmap-img" src={data.speed_heatmap} alt="Speed heatmap" />
        </div>
      </div>
    </>
  );
}
