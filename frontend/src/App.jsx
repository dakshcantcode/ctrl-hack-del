import React, { useState } from "react";
import SpiralCanvas from "./components/SpiralCanvas.jsx";
import ImageUpload from "./components/ImageUpload.jsx";
import Results from "./components/Results.jsx";

const API_BASE = "";

export default function App() {
  const [mode, setMode] = useState("upload"); // "draw" | "upload"
  const [analysing, setAnalysing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  /* ── Called by SpiralCanvas after the user clicks "Analyse" ── */
  const handleAnalyse = async (imageDataUrl, strokes) => {
    setAnalysing(true);
    setResult(null);
    setError(null);

    try {
      const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageDataUrl, strokes }),
      });

      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Server error ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setError(err.message);
    } finally {
      setAnalysing(false);
    }
  };

  /* ── Called by ImageUpload ── */
  const handleUpload = async (file) => {
    setAnalysing(true);
    setResult(null);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Server error ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setError(err.message);
    } finally {
      setAnalysing(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setError(null);
  };

  return (
    <div className="app">
      <header>
        <h1>TremorTrace</h1>
        <p>Detect Parkinson's tremor indicators from spiral & wave drawings</p>
      </header>

      {/* Mode toggle */}
      <div className="mode-toggle">
        <button
          className={`toggle-btn ${mode === "upload" ? "active" : ""}`}
          onClick={() => { setMode("upload"); handleReset(); }}
        >
          Upload Image
        </button>
        <button
          className={`toggle-btn ${mode === "draw" ? "active" : ""}`}
          onClick={() => { setMode("draw"); handleReset(); }}
        >
          Draw Spiral
        </button>
      </div>

      {mode === "draw" ? (
        <SpiralCanvas
          onAnalyse={handleAnalyse}
          onReset={handleReset}
          analysing={analysing}
        />
      ) : (
        <ImageUpload
          onUpload={handleUpload}
          onReset={handleReset}
          analysing={analysing}
        />
      )}

      {error && (
        <div className="card" style={{ borderColor: "var(--red)" }}>
          <p style={{ color: "var(--red)" }}>⚠ {error}</p>
        </div>
      )}

      {result && <Results data={result} />}
    </div>
  );
}
