import React, { useRef, useState, useEffect, useCallback } from "react";

const CANVAS_SIZE = 500;
const LINE_WIDTH = 2.5;
const LINE_COLOR = "#111";

/**
 * Draws a faint Archimedean spiral guide on the canvas.
 */
function drawGuide(ctx) {
  const cx = CANVAS_SIZE / 2;
  const cy = CANVAS_SIZE / 2;
  ctx.save();
  ctx.strokeStyle = "rgba(180,180,220,0.35)";
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 6]);
  ctx.beginPath();
  const turns = 4;
  const maxR = CANVAS_SIZE * 0.42;
  for (let t = 0; t <= turns * 2 * Math.PI; t += 0.05) {
    const r = (t / (turns * 2 * Math.PI)) * maxR;
    const x = cx + r * Math.cos(t);
    const y = cy + r * Math.sin(t);
    if (t === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();
}

export default function SpiralCanvas({ onAnalyse, onReset, analysing }) {
  const canvasRef = useRef(null);
  const [drawing, setDrawing] = useState(false);
  const [strokes, setStrokes] = useState([]);       // {x, y, t}[]
  const [hasDrawn, setHasDrawn] = useState(false);

  /* ── Initialise canvas ── */
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    drawGuide(ctx);
  }, []);

  /* ── Pointer handlers ── */
  const getPos = useCallback((e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const scaleX = CANVAS_SIZE / rect.width;
    const scaleY = CANVAS_SIZE / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  }, []);

  const startDraw = useCallback(
    (e) => {
      e.preventDefault();
      setDrawing(true);
      setHasDrawn(true);
      const { x, y } = getPos(e);
      const ctx = canvasRef.current.getContext("2d");
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineWidth = LINE_WIDTH;
      ctx.strokeStyle = LINE_COLOR;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      setStrokes((prev) => [...prev, { x, y, t: performance.now() }]);
    },
    [getPos]
  );

  const draw = useCallback(
    (e) => {
      if (!drawing) return;
      e.preventDefault();
      const { x, y } = getPos(e);
      const ctx = canvasRef.current.getContext("2d");
      ctx.lineTo(x, y);
      ctx.stroke();
      setStrokes((prev) => [...prev, { x, y, t: performance.now() }]);
    },
    [drawing, getPos]
  );

  const endDraw = useCallback((e) => {
    e?.preventDefault();
    setDrawing(false);
  }, []);

  /* ── Clear ── */
  const handleClear = () => {
    const ctx = canvasRef.current.getContext("2d");
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    drawGuide(ctx);
    setStrokes([]);
    setHasDrawn(false);
    onReset();
  };

  /* ── Submit ── */
  const handleSubmit = () => {
    const dataUrl = canvasRef.current.toDataURL("image/png");
    onAnalyse(dataUrl, strokes);
  };

  return (
    <div className="card canvas-wrapper">
      <h2>Draw your spiral below</h2>
      <canvas
        ref={canvasRef}
        width={CANVAS_SIZE}
        height={CANVAS_SIZE}
        style={{ width: CANVAS_SIZE, height: CANVAS_SIZE }}
        onPointerDown={startDraw}
        onPointerMove={draw}
        onPointerUp={endDraw}
        onPointerLeave={endDraw}
      />
      <div className="btn-row">
        <button className="btn btn-secondary" onClick={handleClear}>
          Clear
        </button>
        <button
          className="btn btn-primary"
          disabled={!hasDrawn || analysing}
          onClick={handleSubmit}
        >
          {analysing ? (
            <>
              <span className="spinner" />
              Analysing…
            </>
          ) : (
            "Analyse Spiral"
          )}
        </button>
      </div>
    </div>
  );
}
