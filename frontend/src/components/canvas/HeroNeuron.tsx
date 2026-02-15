/**
 * NeuroSketch — First-Person 3D Neuron Canvas Engine
 *
 * Immersive first-person camera that dives through a recursive 3D
 * neuron web. Features near-clipping for fly-past effects, depth-fog
 * for spatial perception, mouse-parallax look-around inside the
 * nucleus, and 3D spatial fact nodes with proximity-based glow.
 *
 * @module components/canvas/HeroNeuron
 */

"use client";

import React, {
  useRef,
  useEffect,
  useMemo,
  useState,
} from "react";
import {
  project3D,
  rotateX,
  rotateY,
  lerp,
  clamp,
  type Vec3,
} from "@/lib/utils";
import { useNeuronZoom } from "@/hooks/use-neuron-zoom";
import FactOverlay from "@/components/canvas/FactOverlay";

/* ─── Branch Node Type ─────────────────────────────────────────────────── */

interface BranchNode {
  start: Vec3;
  end: Vec3;
  depth: number;
  children: BranchNode[];
  pulsePhase: number;
}

/* ─── Neuron Tree Generator ────────────────────────────────────────────── */

function buildNeuronTree(
  origin: Vec3,
  dir: Vec3,
  length: number,
  depth: number,
  maxDepth: number
): BranchNode {
  const end: Vec3 = {
    x: origin.x + dir.x * length,
    y: origin.y + dir.y * length,
    z: origin.z + dir.z * length,
  };

  const children: BranchNode[] = [];

  if (depth < maxDepth) {
    const numBranches = depth < 2 ? 3 : 2;
    for (let i = 0; i < numBranches; i++) {
      const spread = 0.45 + depth * 0.05;
      const offset = spread * (i - (numBranches - 1) / 2);
      const jitter = (Math.random() - 0.5) * 0.25;

      let newDir = rotateY(dir, offset + jitter);
      newDir = rotateX(newDir, offset * 0.6 + jitter * 0.5);

      const mag = Math.sqrt(
        newDir.x ** 2 + newDir.y ** 2 + newDir.z ** 2
      );
      const normalized: Vec3 = {
        x: newDir.x / mag,
        y: newDir.y / mag,
        z: newDir.z / mag,
      };

      children.push(
        buildNeuronTree(
          end,
          normalized,
          length * 0.7,
          depth + 1,
          maxDepth
        )
      );
    }
  }

  return {
    start: origin,
    end,
    depth,
    children,
    pulsePhase: Math.random() * Math.PI * 2,
  };
}

/* ─── Synapse Particle ─────────────────────────────────────────────────── */

interface Particle {
  branch: BranchNode;
  t: number;
  speed: number;
  baseSpeed: number;
  size: number;
}

/* ─── Main Component ──────────────────────────────────────────────────── */

/* ─── 3D Spatial Fact Definitions ──────────────────────────────────────── */

interface SpatialFact {
  id: number;
  pos: Vec3;           // 3D world-space coordinate
  fact: {
    title: string;
    body: string;
    stat: string;
    statLabel: string;
  };
}

const SPATIAL_FACTS: SpatialFact[] = [
  {
    id: 0,
    pos: { x: -100, y: -70, z: 80 },
    fact: {
      title: "Dopamine Depletion",
      body: "Loss of dopaminergic neurons in the substantia nigra disrupts motor signal transmission, causing bradykinesia and rigidity.",
      stat: "60-80%",
      statLabel: "neurons lost before symptoms appear",
    },
  },
  {
    id: 1,
    pos: { x: 110, y: -40, z: -60 },
    fact: {
      title: "Micrographia",
      body: "Progressive handwriting shrinkage is one of the earliest motor signs, detectable years before clinical diagnosis.",
      stat: "5-8 yrs",
      statLabel: "early detection window",
    },
  },
  {
    id: 2,
    pos: { x: -60, y: 80, z: 100 },
    fact: {
      title: "Resting Tremor",
      body: "A 4-6 Hz oscillation, typically starting unilaterally in the hand, is the hallmark motor symptom of Parkinson's disease.",
      stat: "4-6 Hz",
      statLabel: "characteristic tremor frequency",
    },
  },
  {
    id: 3,
    pos: { x: 80, y: 60, z: -80 },
    fact: {
      title: "Alpha-Synuclein",
      body: "Misfolded α-synuclein aggregates into Lewy bodies, spreading through neural circuits in a prion-like cascade.",
      stat: "140 aa",
      statLabel: "amino-acid protein linked to pathology",
    },
  },
  {
    id: 4,
    pos: { x: -30, y: -90, z: 140 },
    fact: {
      title: "Early Screening",
      body: "AI-powered spiral and wave analysis can detect subtle motor biomarkers with clinical-grade sensitivity.",
      stat: "92%+",
      statLabel: "detection accuracy with ML models",
    },
  },
];

/* ─── Main Component ──────────────────────────────────────────────────── */

/* ─── Props ────────────────────────────────────────────────────────────── */

interface HeroNeuronProps {
  /** Fires when dive state changes — drives cinematic UI fade */
  onCinematicChange?: (state: { isWarping: boolean; hasReachedNucleus: boolean; zoom: number }) => void;
}

export default function HeroNeuron({ onCinematicChange }: HeroNeuronProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number>(0);
  const timeRef = useRef<number>(0);
  const warpProgressRef = useRef<number>(0);
  const mouseRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const stemPositionsRef = useRef<{ x: number; y: number }[]>([]);
  const hoverExitTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const nucleusPosRef = useRef<{ x: number; y: number; r: number }>({ x: 0, y: 0, r: 0 });
  const nucleusHoveredRef = useRef(false);

  // Screen-space stem positions exposed to React for the overlay
  const [stemPositions, setStemPositions] = useState<{ x: number; y: number }[]>([]);
  const [hoveredStemId, setHoveredStemId] = useState<number | null>(null);

  const {
    zoom,
    isWarping,
    hasReachedNucleus,
    handleWheel,
    resetWarp,
    activeStem,
    setActiveStem,
  } = useNeuronZoom({ warpThreshold: 0.85 });

  // Notify parent of cinematic state changes
  const onCinematicRef = useRef(onCinematicChange);
  onCinematicRef.current = onCinematicChange;

  useEffect(() => {
    onCinematicRef.current?.({ isWarping, hasReachedNucleus, zoom });
  }, [isWarping, hasReachedNucleus, zoom]);

  /* ─── Refs for jitter-free canvas (decoupled from render) ── */
  const hoveredStemRef = useRef<number | null>(null);
  const activeStemRef = useRef<number | null>(null);
  const zoomRef = useRef(0);
  const isWarpingRef = useRef(false);
  const hasReachedNucleusRef = useRef(false);
  const handleWheelRef = useRef(handleWheel);
  const resetWarpRef = useRef(resetWarp);

  // Keep refs in sync with latest state (synchronous, no effect lag)
  zoomRef.current = zoom;
  isWarpingRef.current = isWarping;
  hasReachedNucleusRef.current = hasReachedNucleus;
  handleWheelRef.current = handleWheel;
  resetWarpRef.current = resetWarp;
  activeStemRef.current = activeStem;

  // Build neuron tree once using multiple root directions
  const trees = useMemo(() => {
    const directions: Vec3[] = [
      { x: 0, y: -1, z: 0 },
      { x: 0.7, y: -0.7, z: 0 },
      { x: -0.7, y: -0.7, z: 0 },
      { x: 0, y: -0.7, z: 0.7 },
      { x: 0, y: -0.7, z: -0.7 },
      { x: 0, y: 1, z: 0 },
      { x: 0.5, y: 0.85, z: 0 },
      { x: -0.5, y: 0.85, z: 0 },
      { x: 0.85, y: 0.1, z: 0.5 },
      { x: -0.85, y: 0.1, z: -0.5 },
      { x: 0.3, y: -0.5, z: -0.8 },
      { x: -0.3, y: -0.5, z: 0.8 },
    ];

    return directions.map((d) =>
      buildNeuronTree({ x: 0, y: 0, z: 0 }, d, 240, 0, 6)
    );
  }, []);

  // Generate particles along branches
  const particles = useMemo(() => {
    const parts: Particle[] = [];
    function collect(node: BranchNode) {
      if (Math.random() < 0.6) {
        const spd = 0.003 + Math.random() * 0.005;
        parts.push({
          branch: node,
          t: Math.random(),
          speed: spd,
          baseSpeed: spd,
          size: 1.5 + Math.random() * 2.5,
        });
      }
      node.children.forEach(collect);
    }
    trees.forEach(collect);
    return parts;
  }, [trees]);

  /* ─── Canvas Setup & Animation Loop ────────────────────────── */

  /* ─── Hover detection on canvas (decoupled from 3D physics) ── */

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    function onMove(e: MouseEvent) {
      const rect = canvas!.getBoundingClientRect();
      mouseRef.current = { x: e.clientX - rect.left, y: e.clientY - rect.top };

      // Nucleus hover detection (always, when inside nucleus)
      if (hasReachedNucleusRef.current) {
        const np = nucleusPosRef.current;
        const ndx = mouseRef.current.x - np.x;
        const ndy = mouseRef.current.y - np.y;
        const overNucleus = ndx * ndx + ndy * ndy < np.r * np.r;
        nucleusHoveredRef.current = overNucleus;
        canvas!.style.cursor = overNucleus ? 'pointer' : 'default';
      }

      if (!hasReachedNucleusRef.current) return;

      const positions = stemPositionsRef.current;
      let found: number | null = null;
      for (let i = 0; i < positions.length; i++) {
        const dx = mouseRef.current.x - positions[i].x;
        const dy = mouseRef.current.y - positions[i].y;
        if (dx * dx + dy * dy < 35 * 35) {
          found = i;
          break;
        }
      }

      // Debounced hover exit — 150ms grace period prevents flicker
      if (found !== null) {
        if (hoverExitTimer.current) {
          clearTimeout(hoverExitTimer.current);
          hoverExitTimer.current = null;
        }
        hoveredStemRef.current = found;
        setHoveredStemId(found);
        setActiveStem(found);
      } else if (hoveredStemRef.current !== null) {
        if (!hoverExitTimer.current) {
          hoverExitTimer.current = setTimeout(() => {
            hoveredStemRef.current = null;
            setHoveredStemId(null);
            setActiveStem(null);
            hoverExitTimer.current = null;
          }, 150);
        }
      }
    }

    function onLeave() {
      if (!hoverExitTimer.current) {
        hoverExitTimer.current = setTimeout(() => {
          hoveredStemRef.current = null;
          setHoveredStemId(null);
          setActiveStem(null);
          hoverExitTimer.current = null;
        }, 150);
      }
    }

    canvas.addEventListener("mousemove", onMove);
    canvas.addEventListener("mouseleave", onLeave);

    // Nucleus click → reset to hero view
    function onClick() {
      if (!hasReachedNucleusRef.current) return;
      const np = nucleusPosRef.current;
      const ndx = mouseRef.current.x - np.x;
      const ndy = mouseRef.current.y - np.y;
      if (ndx * ndx + ndy * ndy < np.r * np.r) {
        nucleusHoveredRef.current = false;
        canvas!.style.cursor = 'default';
        // Clear any active fact overlays
        hoveredStemRef.current = null;
        setHoveredStemId(null);
        setActiveStem(null);
        // Reset camera, zoom, and warp state
        resetWarpRef.current();
      }
    }
    canvas.addEventListener("click", onClick);

    return () => {
      canvas.removeEventListener("mousemove", onMove);
      canvas.removeEventListener("mouseleave", onLeave);
      canvas.removeEventListener("click", onClick);
      if (hoverExitTimer.current) clearTimeout(hoverExitTimer.current);
    };
  }, [setActiveStem]);

  /* ─── Canvas Setup & Animation Loop ────────────────────────── */

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Invisible scroll-interceptor — always active (via ref for stable deps)
    const onWheel = (e: WheelEvent) => handleWheelRef.current(e);
    canvas.addEventListener("wheel", onWheel, { passive: false });

    const ctx = canvas.getContext("2d")!;
    let width = 0;
    let height = 0;

    function resize() {
      const dpr = window.devicePixelRatio || 1;
      width = canvas!.clientWidth;
      height = canvas!.clientHeight;
      canvas!.width = width * dpr;
      canvas!.height = height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    resize();
    window.addEventListener("resize", resize);

    /* ─── Render Branch Recursively ────────────────────────── */

    function renderBranch(
      node: BranchNode,
      time: number,
      focalLength: number,
      cx: number,
      cy: number,
      rotAngleY: number,
      rotAngleX: number,
      globalAlpha: number,
      camZ: number,
      warpFactor: number,
      maxFogDist: number
    ) {
      // Rotate points
      let s = rotateY(node.start, rotAngleY);
      s = rotateX(s, rotAngleX);
      let e = rotateY(node.end, rotAngleY);
      e = rotateX(e, rotAngleX);

      // Camera-relative Z (first-person: camera dives along Z)
      const viewZs = s.z - camZ;
      const viewZe = e.z - camZ;
      const nearClipMin = -focalLength + 50;

      // Both endpoints behind the camera eye → skip branch entirely
      if (viewZs < nearClipMin && viewZe < nearClipMin) {
        node.children.forEach((child) =>
          renderBranch(child, time, focalLength, cx, cy, rotAngleY, rotAngleX, globalAlpha, camZ, warpFactor, maxFogDist)
        );
        return;
      }

      // Near-clip fade: wider fade zone (80px) for gentle lens-dissolve
      const minViewZ = Math.min(viewZs, viewZe);
      const nearFade = minViewZ < 80
        ? clamp((minViewZ - nearClipMin) / (80 - nearClipMin), 0, 1)
        : 1;

      // Depth-fog: distant branches are dimmer
      const avgViewZ = (viewZs + viewZe) / 2;
      const fogFactor = clamp(1 - avgViewZ / maxFogDist, 0.08, 1);

      s = { ...s, z: viewZs };
      e = { ...e, z: viewZe };

      const p1 = project3D(s, focalLength, cx, cy);
      const p2 = project3D(e, focalLength, cx, cy);

      // Pulse brightness — intensified for vibrant neon
      const pulse =
        0.7 + 0.3 * Math.sin(time * 2.5 + node.pulsePhase);

      // Depth-based color: cyan core → blue tips
      const depthRatio = node.depth / 6;

      // During warp, branches get a speed-line stretch effect
      const warpStretch = 1 + warpFactor * 3 * (1 - depthRatio);

      const r = Math.round(lerp(20, 100, depthRatio));
      const g = Math.round(lerp(255, 180, depthRatio));
      const b = Math.round(lerp(255, 255, depthRatio));

      // Warp brightens; fog and near-clip modulate alpha — boosted base intensity
      const warpBright = 1 + warpFactor * 2.5;
      const lineWidth = Math.min(14, Math.max(0.8, (1 - depthRatio) * 4 * p1.scale * warpStretch));
      const alpha = clamp(
        globalAlpha * pulse * (1 - depthRatio * 0.25) * p1.scale * warpBright * nearFade * fogFactor * 1.4,
        0,
        1
      );

      if (alpha > 0.005) {
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);

        // During warp, extend lines outward from center for speed-line effect
        if (warpFactor > 0.01) {
          const dx = p2.x - cx;
          const dy = p2.y - cy;
          ctx.lineTo(
            p2.x + dx * warpFactor * 2,
            p2.y + dy * warpFactor * 2
          );
        } else {
          ctx.lineTo(p2.x, p2.y);
        }
        ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
        ctx.lineWidth = lineWidth;
        ctx.lineCap = "round";
        ctx.stroke();
      }

      // Render children
      node.children.forEach((child) =>
        renderBranch(
          child,
          time,
          focalLength,
          cx,
          cy,
          rotAngleY,
          rotAngleX,
          globalAlpha,
          camZ,
          warpFactor,
          maxFogDist
        )
      );
    }

    /* ─── Render Nucleus ───────────────────────────────────── */

    function renderNucleus(
      time: number,
      focalLength: number,
      cx: number,
      cy: number,
      rotAngleY: number,
      rotAngleX: number,
      zoomLevel: number,
      camZ: number,
      warpFactor: number,
      hoverScale: number
    ) {
      let nucleusPos: Vec3 = { x: 0, y: 0, z: 0 };
      nucleusPos = rotateY(nucleusPos, rotAngleY);
      nucleusPos = rotateX(nucleusPos, rotAngleX);

      // Camera-relative Z for first-person dive
      const viewZ = nucleusPos.z - camZ;
      const nearClipMin = -focalLength + 50;

      // Near-clip fade (nucleus approaching or passing camera)
      const nearFade = viewZ < 80
        ? clamp((viewZ - nearClipMin) / (80 - nearClipMin), 0, 1)
        : 1;

      // Skip if fully behind camera
      if (nearFade < 0.01) return;

      nucleusPos = { ...nucleusPos, z: viewZ };

      const projected = project3D(nucleusPos, focalLength, cx, cy);
      const baseRadius = 18 + zoomLevel * 40;
      const pulse = 1 + 0.2 * Math.sin(time * 3);
      // During warp, nucleus expands dramatically
      const warpExpand = 1 + warpFactor * 8;
      const radius = baseRadius * projected.scale * pulse * warpExpand * hoverScale;

      // Store projected nucleus position + hit-radius for hover/click detection
      nucleusPosRef.current = { x: projected.x, y: projected.y, r: radius * 1.5 };

      // Outer glow — near-clip fades it as camera passes through (intensified)
      const glowAlpha = (0.5 + warpFactor * 0.5) * pulse * nearFade;
      const glowGrad = ctx.createRadialGradient(
        projected.x,
        projected.y,
        radius * 0.3,
        projected.x,
        projected.y,
        radius * (2.5 + warpFactor * 4)
      );
      glowGrad.addColorStop(0, `rgba(0, 240, 255, ${clamp(glowAlpha, 0, 1)})`);
      glowGrad.addColorStop(0.5, `rgba(0, 200, 255, ${clamp(glowAlpha * 0.5, 0, 1)})`);
      glowGrad.addColorStop(1, "rgba(0, 0, 0, 0)");

      ctx.beginPath();
      ctx.arc(
        projected.x,
        projected.y,
        radius * (2.5 + warpFactor * 4),
        0,
        Math.PI * 2
      );
      ctx.fillStyle = glowGrad;
      ctx.fill();

      // Core — alpha modulated by near-clip
      const coreAlpha = nearFade;
      const coreGrad = ctx.createRadialGradient(
        projected.x,
        projected.y,
        0,
        projected.x,
        projected.y,
        radius
      );
      coreGrad.addColorStop(0, `rgba(220, 255, 255, ${0.98 * coreAlpha})`);
      coreGrad.addColorStop(0.4, `rgba(0, 240, 255, ${0.9 * coreAlpha})`);
      coreGrad.addColorStop(1, `rgba(0, 140, 220, ${0.3 * coreAlpha})`);

      ctx.beginPath();
      ctx.arc(projected.x, projected.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = coreGrad;
      ctx.fill();
    }

    /* ─── Render Particles ─────────────────────────────────── */

    function renderParticles(
      time: number,
      focalLength: number,
      cx: number,
      cy: number,
      rotAngleY: number,
      rotAngleX: number,
      camZ: number,
      warpFactor: number,
      maxFogDist: number
    ) {
      particles.forEach((p) => {
        // During warp, particles accelerate dramatically
        const speedMult = 1 + warpFactor * 15;
        p.t = (p.t + p.baseSpeed * speedMult) % 1;

        const pos: Vec3 = {
          x: lerp(p.branch.start.x, p.branch.end.x, p.t),
          y: lerp(p.branch.start.y, p.branch.end.y, p.t),
          z: lerp(p.branch.start.z, p.branch.end.z, p.t),
        };

        let rPos = rotateY(pos, rotAngleY);
        rPos = rotateX(rPos, rotAngleX);

        // Camera-relative Z
        const viewZ = rPos.z - camZ;
        const nearClipMin = -focalLength + 50;

        // Behind camera → skip
        if (viewZ < nearClipMin) return;

        // Near-clip fade
        const nearFade = viewZ < 80
          ? clamp((viewZ - nearClipMin) / (80 - nearClipMin), 0, 1)
          : 1;

        // Depth-fog: distant particles are dimmer
        const fogFactor = clamp(1 - viewZ / maxFogDist, 0.08, 1);

        rPos = { ...rPos, z: viewZ };
        const proj = project3D(rPos, focalLength, cx, cy);
        const alpha = (0.55 + 0.45 * Math.sin(time * 5 + p.branch.pulsePhase)) * nearFade * fogFactor;

        // During warp, particles get trail effect
        const trailLen = warpFactor * 6;
        if (trailLen > 0.1) {
          const dx = proj.x - cx;
          const dy = proj.y - cy;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          const nx = dx / dist;
          const ny = dy / dist;

          ctx.beginPath();
          ctx.moveTo(proj.x, proj.y);
          ctx.lineTo(proj.x + nx * trailLen * 8, proj.y + ny * trailLen * 8);
          ctx.strokeStyle = `rgba(0, 255, 220, ${clamp(alpha * proj.scale * 0.5, 0, 0.6)})`;
          ctx.lineWidth = p.size * proj.scale * 0.5;
          ctx.stroke();
        }

        // Square pixel particles with depth-fog — brighter neon
        const pxSize = Math.min(10, p.size * proj.scale * (1 + warpFactor) * 1.3);
        ctx.fillStyle = `rgba(0, 255, 220, ${clamp(
          alpha * proj.scale * (1 + warpFactor * 2) * 1.5,
          0,
          1
        )})`;
        ctx.fillRect(
          proj.x - pxSize / 2,
          proj.y - pxSize / 2,
          pxSize,
          pxSize
        );
      });
    }

    /* ─── Warp-Speed Star Lines ────────────────────────────── */

    function renderWarpLines(
      cx: number,
      cy: number,
      warpFactor: number,
      time: number
    ) {
      if (warpFactor < 0.01) return;

      const numLines = Math.floor(warpFactor * 80);
      ctx.save();
      for (let i = 0; i < numLines; i++) {
        const angle = (i / numLines) * Math.PI * 2 + time * 0.5;
        const innerR = 30 + Math.sin(time * 3 + i) * 20;
        const outerR = innerR + warpFactor * 400 * (0.5 + Math.random() * 0.5);

        const x1 = cx + Math.cos(angle) * innerR;
        const y1 = cy + Math.sin(angle) * innerR;
        const x2 = cx + Math.cos(angle) * outerR;
        const y2 = cy + Math.sin(angle) * outerR;

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = `rgba(0, 220, 255, ${clamp(warpFactor * 0.3, 0, 0.4)})`;
        ctx.lineWidth = 0.5 + warpFactor;
        ctx.stroke();
      }
      ctx.restore();
    }

    /* ─── Animation Frame ──────────────────────────────────── */

    let currentZoom = 0;
    let postGlowExpand = 1;   // eased expansion after glow
    let cameraZPos = -320;    // first-person camera Z position (closer = tree fills viewport)
    let parallaxX = 0;        // smoothed mouse parallax yaw
    let parallaxY = 0;        // smoothed mouse parallax pitch
    let autoTriggered = false; // one-shot proximity trigger after warp

    function animate() {
      const time = (timeRef.current += 0.016);

      // Smooth zoom interpolation
      currentZoom = lerp(currentZoom, zoomRef.current, 0.08);

      // Warp progress: ramps up during isWarping, decays after
      if (isWarpingRef.current) {
        warpProgressRef.current = lerp(warpProgressRef.current, 1, 0.04);
      } else {
        warpProgressRef.current = lerp(warpProgressRef.current, 0, 0.06);
      }
      const warpFactor = warpProgressRef.current;

      // Post-glow spatial expansion — ease stems outward 25% after warp
      // Uses a slower lerp rate (0.018) for buttery-smooth expansion
      const expandTarget = hasReachedNucleusRef.current ? 1.25 : 1;
      postGlowExpand = lerp(postGlowExpand, expandTarget, 0.018);

      ctx.clearRect(0, 0, width, height);

      // Background — shifts to deep blue during warp
      const bgGrad = ctx.createRadialGradient(
        width / 2,
        height / 2,
        0,
        width / 2,
        height / 2,
        width * 0.7
      );

      const bgCenterR = Math.round(lerp(5, 0, warpFactor));
      const bgCenterG = Math.round(lerp(15, 10, warpFactor));
      const bgCenterB = Math.round(lerp(25, 40, warpFactor));
      bgGrad.addColorStop(0, `rgba(${bgCenterR}, ${bgCenterG}, ${bgCenterB}, 1)`);
      bgGrad.addColorStop(1, "rgba(0, 0, 0, 1)");
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, width, height);

      const cx = width / 2;
      const cy = height / 2;
      const focalLength = 350;
      const maxFogDist = 1100;

      // ── First-Person Camera Z ────────────────────────────
      // Camera dives from z=-500 toward origin, blasts through during warp,
      // then settles at z=0 (inside the nucleus).
      let targetCamZ: number;
      if (isWarpingRef.current) {
        targetCamZ = 600;   // blast through the neural web
      } else if (hasReachedNucleusRef.current) {
        targetCamZ = 0;     // settle at the nucleus center
      } else {
        targetCamZ = -320 + currentZoom * 376; // zoom 0→0.85 maps -320→0
      }
      const camLerp = isWarpingRef.current ? 0.035 : 0.06;
      cameraZPos = lerp(cameraZPos, targetCamZ, camLerp);

      // ── Camera Rotation ──────────────────────────────────
      // Pre-nucleus: gentle auto-spin dampened by zoom
      // Post-nucleus: mouse parallax look-around
      let rotY: number;
      let rotXAngle: number;

      if (hasReachedNucleusRef.current) {
        // Inside nucleus — mouse parallax replaces auto-rotation
        const mx = width > 0 ? (mouseRef.current.x - cx) / (width * 0.5) : 0;
        const my = height > 0 ? (mouseRef.current.y - cy) / (height * 0.5) : 0;
        parallaxX = lerp(parallaxX, mx * 0.85, 0.06);
        parallaxY = lerp(parallaxY, my * 0.65, 0.06);
        rotY = parallaxX;
        rotXAngle = parallaxY;
      } else {
        // Pre-nucleus: auto-rotation dampened by zoom depth
        const rotDampen = Math.max(0, 1 - currentZoom * 1.4);
        const rotSpeed = 0.15 * rotDampen;
        rotY = time * rotSpeed;
        rotXAngle = Math.sin(time * 0.1) * 0.15 * rotDampen;
      }

      // Render warp-speed lines behind everything
      renderWarpLines(cx, cy, warpFactor, time);

      // Render neuron branches (with near-clipping + depth-fog) — boosted globalAlpha
      trees.forEach((tree) => {
        renderBranch(tree, time, focalLength, cx, cy, rotY, rotXAngle, 1.0, cameraZPos, warpFactor, maxFogDist);
      });

      // Render nucleus (hover-scale + near-clip fades as camera passes through)
      const nucleusHoverScale = nucleusHoveredRef.current ? 1.2 : 1;
      renderNucleus(time, focalLength, cx, cy, rotY, rotXAngle, currentZoom, cameraZPos, warpFactor, nucleusHoverScale);

      // Render synapse particles (square pixels + depth-fog)
      renderParticles(time, focalLength, cx, cy, rotY, rotXAngle, cameraZPos, warpFactor, maxFogDist);

      // ── 3D Spatial Fact Nodes (after warp completes) ──────
      if (hasReachedNucleusRef.current) {
        const stemTime = time;
        const positions: { x: number; y: number }[] = [];
        let closestIdx: number | null = null;
        let closestScreenDist = Infinity;

        SPATIAL_FACTS.forEach((sf, idx) => {
          // Project fact node through the same 3D camera transform
          let worldPos = rotateY(sf.pos, rotY);
          worldPos = rotateX(worldPos, rotXAngle);
          const viewZ = worldPos.z - cameraZPos;

          // Behind camera → push offscreen placeholder
          if (viewZ < -focalLength + 50) {
            positions.push({ x: -9999, y: -9999 });
            return;
          }

          const proj = project3D({ x: worldPos.x, y: worldPos.y, z: viewZ }, focalLength, cx, cy);

          // Post-glow expansion pushes facts outward
          const stemX = cx + (proj.x - cx) * postGlowExpand;
          const stemY = cy + (proj.y - cy) * postGlowExpand;

          // 3D proximity → glow intensity
          const dist3D = Math.sqrt(worldPos.x ** 2 + worldPos.y ** 2 + viewZ ** 2);
          const proximity = clamp(1 - dist3D / 250, 0, 1);

          // Screen-center distance for auto-trigger priority
          const screenDist = Math.sqrt((stemX - cx) ** 2 + (stemY - cy) ** 2);
          if (proximity > 0.15 && screenDist < closestScreenDist) {
            closestScreenDist = screenDist;
            closestIdx = idx;
          }

          // Hover or proximity-active state
          const isHot = hoveredStemRef.current === idx || activeStemRef.current === idx;
          const glowIntensity = Math.max(proximity, isHot ? 1 : 0);
          const baseAlpha = isHot ? 0.95 : 0.4 + proximity * 0.4 + 0.15 * Math.sin(stemTime * 2 + idx);

          // Depth-fog for the stem line
          const stemFog = clamp(1 - viewZ / 900, 0.15, 1);
          const lineAlpha = baseAlpha * stemFog;

          // Dendrite stem line from center to fact node
          const grad = ctx.createLinearGradient(cx, cy, stemX, stemY);
          grad.addColorStop(0, `rgba(0, 220, 255, ${lineAlpha * 0.3})`);
          grad.addColorStop(1, `rgba(0, 255, 200, ${lineAlpha})`);

          ctx.beginPath();
          ctx.moveTo(cx, cy);
          const cpx = (cx + stemX) / 2 + Math.sin(stemTime + idx * 1.3) * 12;
          const cpy = (cy + stemY) / 2 + Math.cos(stemTime + idx * 0.9) * 12;
          ctx.quadraticCurveTo(cpx, cpy, stemX, stemY);
          ctx.strokeStyle = grad;
          ctx.lineWidth = isHot ? 3 : 1.5 + proximity;
          ctx.lineCap = "round";
          ctx.stroke();

          // Terminal bulb — glows + pulses when camera is physically near
          const proximityPulse = proximity > 0.25
            ? 1 + 0.3 * Math.sin(stemTime * 4 + idx * 1.5)
            : 1;
          const bulbRadius = (5 + glowIntensity * 8) * proximityPulse;
          const glowSpread = bulbRadius * (2.5 + proximity * 2);
          const bulbGlow = ctx.createRadialGradient(stemX, stemY, 0, stemX, stemY, glowSpread);
          bulbGlow.addColorStop(0, `rgba(0, 255, 220, ${0.5 + glowIntensity * 0.5})`);
          bulbGlow.addColorStop(0.4, `rgba(0, 200, 255, ${0.1 + glowIntensity * 0.4})`);
          bulbGlow.addColorStop(1, "rgba(0, 0, 0, 0)");
          ctx.beginPath();
          ctx.arc(stemX, stemY, glowSpread, 0, Math.PI * 2);
          ctx.fillStyle = bulbGlow;
          ctx.fill();

          ctx.beginPath();
          ctx.arc(stemX, stemY, bulbRadius, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(180, 255, 240, ${0.6 + glowIntensity * 0.35})`;
          ctx.fill();

          // Sub-dendrites off the stem
          for (let b = 0; b < 3; b++) {
            const t = 0.35 + b * 0.2;
            const bsx = cx + (stemX - cx) * t + Math.sin(stemTime + b) * 4;
            const bsy = cy + (stemY - cy) * t + Math.cos(stemTime + b) * 4;
            const bAngle = Math.atan2(stemY - cy, stemX - cx) + (b - 1) * 0.6;
            const bLen = 18 + Math.sin(stemTime * 2 + b) * 4;
            const bex = bsx + Math.cos(bAngle) * bLen;
            const bey = bsy + Math.sin(bAngle) * bLen;

            ctx.beginPath();
            ctx.moveTo(bsx, bsy);
            ctx.lineTo(bex, bey);
            ctx.strokeStyle = `rgba(0, 220, 255, ${lineAlpha * 0.35})`;
            ctx.lineWidth = 0.8;
            ctx.stroke();
          }

          positions.push({ x: stemX, y: stemY });
        });

        // Auto-trigger: activate closest fact once after warp settles
        if (!autoTriggered && closestIdx !== null) {
          autoTriggered = true;
          hoveredStemRef.current = closestIdx;
          setHoveredStemId(closestIdx);
          setActiveStem(closestIdx);
        }

        stemPositionsRef.current = positions;
        setStemPositions([...positions]);
      }

      // White flash overlay during peak warp
      if (warpFactor > 0.7) {
        const flashAlpha = clamp((warpFactor - 0.7) / 0.3 * 0.6, 0, 0.6);
        ctx.fillStyle = `rgba(200, 255, 255, ${flashAlpha})`;
        ctx.fillRect(0, 0, width, height);
      }

      animFrameRef.current = requestAnimationFrame(animate);
    }

    animFrameRef.current = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(animFrameRef.current);
      window.removeEventListener("resize", resize);
      canvas.removeEventListener("wheel", onWheel);
    };
  }, [trees, particles]);

  return (
    <div className="relative w-full h-full">
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        style={{ touchAction: "none" }}
      />

      {/* Synaptic Fact Overlay — glassmorphic tooltip at node position */}
      {/* z-40 keeps tooltip above the hero-text z-10 layer */}
      <FactOverlay
        activeStemId={hoveredStemId}
        visible={hasReachedNucleus}
        positions={stemPositions}
        facts={SPATIAL_FACTS.map((s) => s.fact)}
        onMouseEnter={() => {
          if (hoverExitTimer.current) {
            clearTimeout(hoverExitTimer.current);
            hoverExitTimer.current = null;
          }
        }}
        onMouseLeave={() => {
          if (!hoverExitTimer.current) {
            hoverExitTimer.current = setTimeout(() => {
              hoveredStemRef.current = null;
              setHoveredStemId(null);
              setActiveStem(null);
              hoverExitTimer.current = null;
            }, 150);
          }
        }}
      />

      {/* Ghost contextual instructions — VT323 breathing text */}
      {!hasReachedNucleus && (
        <div
          className="absolute bottom-12 left-1/2 -translate-x-1/2 z-20 pointer-events-none text-center select-none"
          style={{
            fontFamily: "var(--font-vt323), monospace",
            opacity: Math.max(0, 1 - zoom * 2.5),
            transition: "opacity 0.6s ease-out",
          }}
        >
          <p
            className="text-cyan-400/60 text-lg tracking-[0.25em] uppercase"
            style={{ animation: "ghostBreathe 3.5s ease-in-out infinite" }}
          >
            [ SCROLL TO DIVE &nbsp;|&nbsp; HOVER NODES TO DECODE ]
          </p>
        </div>
      )}

      <style jsx>{`
        @keyframes ghostBreathe {
          0%, 100% { opacity: 0.35; }
          50% { opacity: 0.75; }
        }
      `}</style>
    </div>
  );
}
