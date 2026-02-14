/**
 * NeuroSketch — Utility Functions
 * Core math and helper utilities for the Parkinson's detection pipeline.
 *
 * @module lib/utils
 */

import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/* ─── Tailwind Class Merge ─────────────────────────────────────────────── */

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/* ─── 3D Projection Math ──────────────────────────────────────────────── */

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface Vec2 {
  x: number;
  y: number;
}

/**
 * Project a 3D point onto a 2D plane using perspective division.
 * @param point - The 3D point to project
 * @param focalLength - Distance from camera to projection plane
 * @param cx - Canvas center X
 * @param cy - Canvas center Y
 */
export function project3D(
  point: Vec3,
  focalLength: number,
  cx: number,
  cy: number
): Vec2 & { scale: number } {
  const perspective = focalLength / (focalLength + point.z);
  return {
    x: point.x * perspective + cx,
    y: point.y * perspective + cy,
    scale: perspective,
  };
}

/**
 * Rotate a 3D point around the Y-axis.
 */
export function rotateY(point: Vec3, angle: number): Vec3 {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return {
    x: point.x * cos - point.z * sin,
    y: point.y,
    z: point.x * sin + point.z * cos,
  };
}

/**
 * Rotate a 3D point around the X-axis.
 */
export function rotateX(point: Vec3, angle: number): Vec3 {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return {
    x: point.x,
    y: point.y * cos - point.z * sin,
    z: point.y * sin + point.z * cos,
  };
}

/* ─── Spiral Analysis (Archimedes) ─────────────────────────────────────── */

export interface SpiralPoint {
  x: number;
  y: number;
  t: number; // timestamp
}

/**
 * Compute the ideal Archimedes spiral point at angle θ.
 * r = a + b*θ
 */
export function idealSpiralPoint(
  theta: number,
  a: number,
  b: number,
  cx: number,
  cy: number
): Vec2 {
  const r = a + b * theta;
  return {
    x: cx + r * Math.cos(theta),
    y: cy + r * Math.sin(theta),
  };
}

/**
 * Calculate the radial deviation between a drawn point and the ideal spiral.
 */
export function radialDeviation(
  drawn: Vec2,
  ideal: Vec2
): number {
  return Math.sqrt(
    (drawn.x - ideal.x) ** 2 + (drawn.y - ideal.y) ** 2
  );
}

/**
 * Compute drawing speed between consecutive points (px/ms).
 */
export function drawingSpeed(
  p1: SpiralPoint,
  p2: SpiralPoint
): number {
  const dist = Math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2);
  const dt = Math.abs(p2.t - p1.t);
  return dt > 0 ? dist / dt : 0;
}

/**
 * Compute mean and standard deviation of an array of numbers.
 */
export function meanStd(values: number[]): { mean: number; std: number } {
  if (values.length === 0) return { mean: 0, std: 0 };
  const mean = values.reduce((s, v) => s + v, 0) / values.length;
  const variance =
    values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length;
  return { mean, std: Math.sqrt(variance) };
}

/**
 * Linear interpolation.
 */
export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/**
 * Clamp a value between min and max.
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/* ─── Neuron Tree Generation ───────────────────────────────────────────── */

export interface NeuronBranch {
  start: Vec3;
  end: Vec3;
  depth: number;
  children: NeuronBranch[];
}

/**
 * Recursively generate a neuron-like branching tree in 3D space.
 * Used by the HeroNeuron canvas engine.
 */
export function generateNeuronTree(
  origin: Vec3,
  direction: Vec3,
  length: number,
  depth: number,
  maxDepth: number,
  branchAngle: number = 0.5,
  lengthDecay: number = 0.72,
  branchCount: number = 2
): NeuronBranch {
  const end: Vec3 = {
    x: origin.x + direction.x * length,
    y: origin.y + direction.y * length,
    z: origin.z + direction.z * length,
  };

  const children: NeuronBranch[] = [];

  if (depth < maxDepth) {
    for (let i = 0; i < branchCount; i++) {
      const angleOffset = branchAngle * (i - (branchCount - 1) / 2);
      const jitter = (Math.random() - 0.5) * 0.3;

      const newDir = rotateY(
        rotateX(direction, angleOffset + jitter),
        angleOffset * 0.7 + jitter
      );

      // Normalize direction
      const mag = Math.sqrt(
        newDir.x ** 2 + newDir.y ** 2 + newDir.z ** 2
      );
      const normalized: Vec3 = {
        x: newDir.x / mag,
        y: newDir.y / mag,
        z: newDir.z / mag,
      };

      children.push(
        generateNeuronTree(
          end,
          normalized,
          length * lengthDecay,
          depth + 1,
          maxDepth,
          branchAngle,
          lengthDecay,
          branchCount
        )
      );
    }
  }

  return { start: origin, end, depth, children };
}
