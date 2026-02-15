/**
 * WaveTest — Horizontal wave drawing canvas with tremor analysis
 *
 * Users draw a continuous wave along a guide sinusoid.
 * Captures high-resolution (x, y, t) coordinate arrays for
 * temporal tremor analysis and ML pipeline submission.
 *
 * @module components/tests/wave-test
 */

"use client";

import React, {
  useRef,
  useEffect,
  useState,
  useCallback,
  useImperativeHandle,
  forwardRef,
} from "react";
import { meanStd, type SpiralPoint } from "@/lib/utils";

/* ─── Types ───────────────────────────────────────────────────────────── */

export interface WaveAnalysis {
  pointCount: number;
  meanDeviation: number;
  stdDeviation: number;
  meanSpeed: number;
  speedVariance: number;
  tremorScore: number;
  drawingTime: number;
}

export interface WaveCanvasRef {
  clear: () => void;
  getAnalysis: () => WaveAnalysis | null;
  getPoints: () => SpiralPoint[];
  getImageBlob: () => Promise<Blob | null>;
  /** Package drawing as FormData for the ML pipeline */
  getFormData: () => Promise<FormData | null>;
}

interface WaveTestProps {
  width?: number;
  height?: number;
  onAnalysisUpdate?: (analysis: WaveAnalysis) => void;
  onDrawingStart?: () => void;
  onDrawingEnd?: () => void;
}

/* ─── Wave Config ─────────────────────────────────────────────────────── */

const WAVE_AMPLITUDE = 60;
const WAVE_FREQUENCY = 3; // full cycles across canvas
const GUIDE_PADDING_X = 40;

/* ─── Component ───────────────────────────────────────────────────────── */

const WaveTest = forwardRef<WaveCanvasRef, WaveTestProps>(
  (
    {
      width = 500,
      height = 300,
      onAnalysisUpdate,
      onDrawingStart,
      onDrawingEnd,
    },
    ref
  ) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const isDrawingRef = useRef(false);
    const pointsRef = useRef<SpiralPoint[]>([]);
    const [isActive, setIsActive] = useState(false);

    const cy = height / 2;

    /* ─── Guide Wave ───────────────────────────────────────── */

    const idealWaveY = useCallback(
      (x: number): number => {
        const normalizedX =
          (x - GUIDE_PADDING_X) / (width - GUIDE_PADDING_X * 2);
        const theta = normalizedX * WAVE_FREQUENCY * 2 * Math.PI;
        return cy + Math.sin(theta) * WAVE_AMPLITUDE;
      },
      [width, cy]
    );

    const drawGuide = useCallback(
      (ctx: CanvasRenderingContext2D) => {
        ctx.beginPath();
        for (let x = GUIDE_PADDING_X; x <= width - GUIDE_PADDING_X; x++) {
          const y = idealWaveY(x);
          if (x === GUIDE_PADDING_X) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = "rgba(0, 180, 255, 0.15)";
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 6]);
        ctx.stroke();
        ctx.setLineDash([]);

        // Start marker
        ctx.beginPath();
        ctx.arc(GUIDE_PADDING_X, cy, 4, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(0, 220, 255, 0.4)";
        ctx.fill();

        // End marker
        ctx.beginPath();
        ctx.arc(width - GUIDE_PADDING_X, cy, 4, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(0, 220, 255, 0.4)";
        ctx.fill();

        // Labels
        ctx.font = "10px monospace";
        ctx.fillStyle = "rgba(0, 220, 255, 0.3)";
        ctx.textAlign = "center";
        ctx.fillText("START", GUIDE_PADDING_X, cy + 24);
        ctx.fillText("END", width - GUIDE_PADDING_X, cy + 24);
      },
      [width, cy, idealWaveY]
    );

    /* ─── Redraw ───────────────────────────────────────────── */

    const redraw = useCallback(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d")!;

      ctx.fillStyle = "#0a0a0f";
      ctx.fillRect(0, 0, width, height);

      // Grid
      ctx.strokeStyle = "rgba(0, 220, 255, 0.03)";
      ctx.lineWidth = 1;
      const gridSize = 25;
      for (let x = 0; x <= width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let y = 0; y <= height; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      // Center line
      ctx.beginPath();
      ctx.moveTo(0, cy);
      ctx.lineTo(width, cy);
      ctx.strokeStyle = "rgba(0, 220, 255, 0.06)";
      ctx.lineWidth = 1;
      ctx.stroke();

      drawGuide(ctx);

      // User stroke
      const points = pointsRef.current;
      if (points.length > 1) {
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
          ctx.lineTo(points[i].x, points[i].y);
        }
        ctx.strokeStyle = "#00ffd5";
        ctx.lineWidth = 2.5;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.stroke();
      }
    }, [width, height, cy, drawGuide]);

    /* ─── Analysis ─────────────────────────────────────────── */

    const computeAnalysis = useCallback((): WaveAnalysis | null => {
      const points = pointsRef.current;
      if (points.length < 10) return null;

      const deviations: number[] = [];
      points.forEach((p) => {
        const idealY = idealWaveY(p.x);
        deviations.push(Math.abs(p.y - idealY));
      });

      const speeds: number[] = [];
      for (let i = 1; i < points.length; i++) {
        const dx = points[i].x - points[i - 1].x;
        const dy = points[i].y - points[i - 1].y;
        const dt = points[i].t - points[i - 1].t;
        if (dt > 0) {
          speeds.push(Math.sqrt(dx * dx + dy * dy) / dt);
        }
      }

      const devStats = meanStd(deviations);
      const speedStats = meanStd(speeds);

      const tremorScore = Math.min(
        100,
        Math.round(
          (devStats.std / 15) * 50 + (speedStats.std / 0.5) * 50
        )
      );

      const drawingTime =
        points.length > 1
          ? points[points.length - 1].t - points[0].t
          : 0;

      return {
        pointCount: points.length,
        meanDeviation: Math.round(devStats.mean * 100) / 100,
        stdDeviation: Math.round(devStats.std * 100) / 100,
        meanSpeed: Math.round(speedStats.mean * 1000) / 1000,
        speedVariance: Math.round(speedStats.std * 1000) / 1000,
        tremorScore,
        drawingTime: Math.round(drawingTime),
      };
    }, [idealWaveY]);

    /* ─── Pointer Handlers ─────────────────────────────────── */

    const getCanvasPoint = useCallback(
      (e: React.PointerEvent): SpiralPoint => {
        const canvas = canvasRef.current!;
        const rect = canvas.getBoundingClientRect();
        const scaleX = width / rect.width;
        const scaleY = height / rect.height;
        return {
          x: (e.clientX - rect.left) * scaleX,
          y: (e.clientY - rect.top) * scaleY,
          t: performance.now(),
        };
      },
      [width, height]
    );

    const handlePointerDown = useCallback(
      (e: React.PointerEvent) => {
        isDrawingRef.current = true;
        setIsActive(true);
        const p = getCanvasPoint(e);
        pointsRef.current = [p];
        onDrawingStart?.();
        (e.target as HTMLElement).setPointerCapture(e.pointerId);
      },
      [getCanvasPoint, onDrawingStart]
    );

    const handlePointerMove = useCallback(
      (e: React.PointerEvent) => {
        if (!isDrawingRef.current) return;
        const p = getCanvasPoint(e);
        pointsRef.current.push(p);
        redraw();

        if (pointsRef.current.length % 10 === 0) {
          const analysis = computeAnalysis();
          if (analysis) onAnalysisUpdate?.(analysis);
        }
      },
      [getCanvasPoint, redraw, computeAnalysis, onAnalysisUpdate]
    );

    const handlePointerUp = useCallback(() => {
      if (!isDrawingRef.current) return;
      isDrawingRef.current = false;
      setIsActive(false);

      const analysis = computeAnalysis();
      if (analysis) onAnalysisUpdate?.(analysis);
      onDrawingEnd?.();
    }, [computeAnalysis, onAnalysisUpdate, onDrawingEnd]);

    /* ─── Imperative Handle ────────────────────────────────── */

    useImperativeHandle(ref, () => ({
      clear: () => {
        pointsRef.current = [];
        redraw();
        onAnalysisUpdate?.(null as unknown as WaveAnalysis);
      },
      getAnalysis: computeAnalysis,
      getPoints: () => [...pointsRef.current],
      getImageBlob: () =>
        new Promise<Blob | null>((resolve) => {
          const canvas = canvasRef.current;
          if (!canvas) return resolve(null);
          canvas.toBlob((blob) => resolve(blob), "image/png");
        }),
      getFormData: async () => {
        const canvas = canvasRef.current;
        if (!canvas) return null;
        const blob = await new Promise<Blob | null>((resolve) =>
          canvas.toBlob((b) => resolve(b), "image/png")
        );
        if (!blob) return null;

        const points = pointsRef.current;
        const fd = new FormData();
        fd.append("image", blob, "wave.png");
        fd.append(
          "coordinates",
          JSON.stringify(
            points.map((p) => ({ x: p.x, y: p.y, t: p.t }))
          )
        );
        fd.append("test_type", "wave");
        return fd;
      },
    }));

    /* ─── Initial Render ───────────────────────────────────── */

    useEffect(() => {
      redraw();
    }, [redraw]);

    return (
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className={`rounded-xl border-2 transition-colors duration-300 cursor-crosshair ${
          isActive
            ? "border-cyan-400/60 shadow-lg shadow-cyan-500/20"
            : "border-zinc-800 hover:border-zinc-700"
        }`}
        style={{
          touchAction: "none",
          maxWidth: "100%",
          height: "auto",
          aspectRatio: `${width} / ${height}`,
        }}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerUp}
      />
    );
  }
);

WaveTest.displayName = "WaveTest";

export default WaveTest;
