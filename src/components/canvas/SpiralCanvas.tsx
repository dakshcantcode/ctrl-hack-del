/**
 * NeuroSketch — Archimedes Spiral Drawing Canvas
 *
 * Interactive HTML5 canvas for drawing Archimedes spirals.
 * Captures stroke data (position + timestamps), computes real-time
 * analytics (speed variance, radial deviation, tremor score), and
 * feeds results to the parent dashboard.
 *
 * @module components/canvas/SpiralCanvas
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
import {
  idealSpiralPoint,
  radialDeviation,
  drawingSpeed,
  meanStd,
  type SpiralPoint,
} from "@/lib/utils";

/* ─── Types ───────────────────────────────────────────────────────────── */

export interface SpiralAnalysis {
  pointCount: number;
  meanDeviation: number;
  stdDeviation: number;
  meanSpeed: number;
  speedVariance: number;
  tremorScore: number; // 0-100, higher = more irregular
  drawingTime: number; // ms
}

export interface SpiralCanvasRef {
  clear: () => void;
  getAnalysis: () => SpiralAnalysis | null;
  getPoints: () => SpiralPoint[];
}

interface SpiralCanvasProps {
  width?: number;
  height?: number;
  showGuide?: boolean;
  onAnalysisUpdate?: (analysis: SpiralAnalysis) => void;
  onDrawingStart?: () => void;
  onDrawingEnd?: () => void;
}

/* ─── Spiral Config ───────────────────────────────────────────────────── */

const SPIRAL_A = 5;
const SPIRAL_B = 8;
const GUIDE_TURNS = 4;
const GUIDE_STEPS = 500;

/* ─── Component ───────────────────────────────────────────────────────── */

const SpiralCanvas = forwardRef<SpiralCanvasRef, SpiralCanvasProps>(
  (
    {
      width = 500,
      height = 500,
      showGuide = true,
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

    const cx = width / 2;
    const cy = height / 2;

    /* ─── Draw Guide Spiral ────────────────────────────────── */

    const drawGuide = useCallback(
      (ctx: CanvasRenderingContext2D) => {
        ctx.beginPath();
        for (let i = 0; i <= GUIDE_STEPS; i++) {
          const theta = (i / GUIDE_STEPS) * GUIDE_TURNS * 2 * Math.PI;
          const point = idealSpiralPoint(theta, SPIRAL_A, SPIRAL_B, cx, cy);
          if (i === 0) ctx.moveTo(point.x, point.y);
          else ctx.lineTo(point.x, point.y);
        }
        ctx.strokeStyle = "rgba(0, 180, 255, 0.15)";
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 6]);
        ctx.stroke();
        ctx.setLineDash([]);

        // Center dot
        ctx.beginPath();
        ctx.arc(cx, cy, 4, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(0, 220, 255, 0.4)";
        ctx.fill();
      },
      [cx, cy]
    );

    /* ─── Redraw Canvas ────────────────────────────────────── */

    const redraw = useCallback(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d")!;

      // Clear
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

      // Guide
      if (showGuide) drawGuide(ctx);

      // Drawn points
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
    }, [width, height, showGuide, drawGuide]);

    /* ─── Analysis Computation ──────────────────────────────── */

    const computeAnalysis = useCallback((): SpiralAnalysis | null => {
      const points = pointsRef.current;
      if (points.length < 10) return null;

      // Compute radial deviations
      const deviations: number[] = [];
      points.forEach((p) => {
        const dx = p.x - cx;
        const dy = p.y - cy;
        const theta = Math.atan2(dy, dx);
        const adjustedTheta = theta >= 0 ? theta : theta + 2 * Math.PI;

        const ideal = idealSpiralPoint(
          adjustedTheta,
          SPIRAL_A,
          SPIRAL_B,
          cx,
          cy
        );
        deviations.push(radialDeviation(p, ideal));
      });

      // Compute speeds
      const speeds: number[] = [];
      for (let i = 1; i < points.length; i++) {
        speeds.push(drawingSpeed(points[i - 1], points[i]));
      }

      const devStats = meanStd(deviations);
      const speedStats = meanStd(speeds);

      // Tremor score: composite of deviation std + speed variance
      const tremorScore = Math.min(
        100,
        Math.round(
          (devStats.std / 20) * 50 + (speedStats.std / 0.5) * 50
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
    }, [cx, cy]);

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

        // Live analysis update throttled by point count
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
        onAnalysisUpdate?.(null as unknown as SpiralAnalysis);
      },
      getAnalysis: computeAnalysis,
      getPoints: () => [...pointsRef.current],
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

SpiralCanvas.displayName = "SpiralCanvas";

export default SpiralCanvas;
