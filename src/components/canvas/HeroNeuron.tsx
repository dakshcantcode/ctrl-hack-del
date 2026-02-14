/**
 * NeuroSketch — Living Neuron 3D Canvas Engine
 *
 * Recursive 3D projected neuron with invisible scroll-interceptor
 * zoom, immersive warp-speed dive, and synaptic fact-stems that
 * branch from the nucleus after the warp completes.
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

/* ─── Synaptic Stem Definitions ────────────────────────────────────────── */

interface SynapticStem {
  id: number;
  angle: number;      // radians
  length: number;     // px from center
  fact: {
    title: string;
    body: string;
    stat: string;
    statLabel: string;
  };
}

const SYNAPTIC_STEMS: SynapticStem[] = [
  {
    id: 0,
    angle: -Math.PI / 2,          // top
    length: 120,
    fact: {
      title: "Dopamine Depletion",
      body: "Loss of dopaminergic neurons in the substantia nigra disrupts motor signal transmission, causing bradykinesia and rigidity.",
      stat: "60-80%",
      statLabel: "neurons lost before symptoms appear",
    },
  },
  {
    id: 1,
    angle: -Math.PI / 2 + (2 * Math.PI) / 5,
    length: 110,
    fact: {
      title: "Micrographia",
      body: "Progressive handwriting shrinkage is one of the earliest motor signs, detectable years before clinical diagnosis.",
      stat: "5-8 yrs",
      statLabel: "early detection window",
    },
  },
  {
    id: 2,
    angle: -Math.PI / 2 + (4 * Math.PI) / 5,
    length: 115,
    fact: {
      title: "Resting Tremor",
      body: "A 4-6 Hz oscillation, typically starting unilaterally in the hand, is the hallmark motor symptom of Parkinson's disease.",
      stat: "4-6 Hz",
      statLabel: "characteristic tremor frequency",
    },
  },
  {
    id: 3,
    angle: -Math.PI / 2 + (6 * Math.PI) / 5,
    length: 105,
    fact: {
      title: "Alpha-Synuclein",
      body: "Misfolded α-synuclein aggregates into Lewy bodies, spreading through neural circuits in a prion-like cascade.",
      stat: "140 aa",
      statLabel: "amino-acid protein linked to pathology",
    },
  },
  {
    id: 4,
    angle: -Math.PI / 2 + (8 * Math.PI) / 5,
    length: 118,
    fact: {
      title: "Early Screening",
      body: "AI-powered spiral and wave analysis can detect subtle motor biomarkers with clinical-grade sensitivity.",
      stat: "92%+",
      statLabel: "detection accuracy with ML models",
    },
  },
];

/* ─── Main Component ──────────────────────────────────────────────────── */

export default function HeroNeuron() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number>(0);
  const timeRef = useRef<number>(0);
  const warpProgressRef = useRef<number>(0);
  const mouseRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const stemPositionsRef = useRef<{ x: number; y: number }[]>([]);
  const hoverExitTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Screen-space stem positions exposed to React for the overlay
  const [stemPositions, setStemPositions] = useState<{ x: number; y: number }[]>([]);
  const [hoveredStemId, setHoveredStemId] = useState<number | null>(null);

  const {
    zoom,
    isWarping,
    hasReachedNucleus,
    handleWheel,
    activeStem,
    setActiveStem,
  } = useNeuronZoom({ warpThreshold: 0.85 });

  /* ─── Refs for jitter-free canvas (decoupled from render) ── */
  const hoveredStemRef = useRef<number | null>(null);
  const activeStemRef = useRef<number | null>(null);
  const zoomRef = useRef(0);
  const isWarpingRef = useRef(false);
  const hasReachedNucleusRef = useRef(false);
  const handleWheelRef = useRef(handleWheel);

  // Keep refs in sync with latest state (synchronous, no effect lag)
  zoomRef.current = zoom;
  isWarpingRef.current = isWarping;
  hasReachedNucleusRef.current = hasReachedNucleus;
  handleWheelRef.current = handleWheel;
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
    ];

    return directions.map((d) =>
      buildNeuronTree({ x: 0, y: 0, z: 0 }, d, 120, 0, 7)
    );
  }, []);

  // Generate particles along branches
  const particles = useMemo(() => {
    const parts: Particle[] = [];
    function collect(node: BranchNode) {
      if (Math.random() < 0.3) {
        const spd = 0.002 + Math.random() * 0.004;
        parts.push({
          branch: node,
          t: Math.random(),
          speed: spd,
          baseSpeed: spd,
          size: 1 + Math.random() * 2,
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
      if (!hasReachedNucleusRef.current) return;
      const rect = canvas!.getBoundingClientRect();
      mouseRef.current = { x: e.clientX - rect.left, y: e.clientY - rect.top };

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
    return () => {
      canvas.removeEventListener("mousemove", onMove);
      canvas.removeEventListener("mouseleave", onLeave);
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
      cameraZ: number,
      warpFactor: number
    ) {
      // Rotate points
      let s = rotateY(node.start, rotAngleY);
      s = rotateX(s, rotAngleX);
      let e = rotateY(node.end, rotAngleY);
      e = rotateX(e, rotAngleX);

      // Move scene toward camera (fly into the neuron)
      s = { ...s, z: s.z + cameraZ };
      e = { ...e, z: e.z + cameraZ };

      const p1 = project3D(s, focalLength, cx, cy);
      const p2 = project3D(e, focalLength, cx, cy);

      // Pulse brightness
      const pulse =
        0.6 + 0.4 * Math.sin(time * 2 + node.pulsePhase);

      // Depth-based color: cyan core → blue tips
      const depthRatio = node.depth / 7;

      // During warp, branches get a speed-line stretch effect
      const warpStretch = 1 + warpFactor * 3 * (1 - depthRatio);

      const r = Math.round(lerp(0, 80, depthRatio));
      const g = Math.round(lerp(220, 140, depthRatio));
      const b = Math.round(lerp(255, 255, depthRatio));

      // Warp brightens everything
      const warpBright = 1 + warpFactor * 2;
      const lineWidth = Math.max(0.5, (1 - depthRatio) * 3 * p1.scale * warpStretch);
      const alpha = clamp(
        globalAlpha * pulse * (1 - depthRatio * 0.4) * p1.scale * warpBright,
        0,
        1
      );

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
          cameraZ,
          warpFactor
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
      cameraZ: number,
      warpFactor: number
    ) {
      let nucleusPos: Vec3 = { x: 0, y: 0, z: 0 };
      nucleusPos = rotateY(nucleusPos, rotAngleY);
      nucleusPos = rotateX(nucleusPos, rotAngleX);
      nucleusPos = { ...nucleusPos, z: nucleusPos.z + cameraZ };

      const projected = project3D(nucleusPos, focalLength, cx, cy);
      const baseRadius = 12 + zoomLevel * 30;
      const pulse = 1 + 0.15 * Math.sin(time * 3);
      // During warp, nucleus expands dramatically
      const warpExpand = 1 + warpFactor * 8;
      const radius = baseRadius * projected.scale * pulse * warpExpand;

      // Outer glow — amplified during warp
      const glowAlpha = (0.3 + warpFactor * 0.5) * pulse;
      const glowGrad = ctx.createRadialGradient(
        projected.x,
        projected.y,
        radius * 0.3,
        projected.x,
        projected.y,
        radius * (2.5 + warpFactor * 4)
      );
      glowGrad.addColorStop(0, `rgba(0, 220, 255, ${clamp(glowAlpha, 0, 1)})`);
      glowGrad.addColorStop(0.5, `rgba(0, 180, 255, ${clamp(glowAlpha * 0.4, 0, 1)})`);
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

      // Core
      const coreGrad = ctx.createRadialGradient(
        projected.x,
        projected.y,
        0,
        projected.x,
        projected.y,
        radius
      );
      coreGrad.addColorStop(0, "rgba(200, 255, 255, 0.95)");
      coreGrad.addColorStop(0.4, "rgba(0, 220, 255, 0.8)");
      coreGrad.addColorStop(1, "rgba(0, 100, 200, 0.2)");

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
      cameraZ: number,
      warpFactor: number
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
        rPos = { ...rPos, z: rPos.z + cameraZ };

        const proj = project3D(rPos, focalLength, cx, cy);
        const alpha = 0.4 + 0.6 * Math.sin(time * 5 + p.branch.pulsePhase);

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

        ctx.beginPath();
        ctx.arc(
          proj.x,
          proj.y,
          p.size * proj.scale * (1 + warpFactor),
          0,
          Math.PI * 2
        );
        ctx.fillStyle = `rgba(0, 255, 220, ${clamp(
          alpha * proj.scale * (1 + warpFactor * 2),
          0,
          1
        )})`;
        ctx.fill();
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
      const focalLength = 400;

      // Camera flies forward along Z toward origin (0,0,0)
      // cameraZ offsets every point's Z, pulling the scene toward the viewer
      // so the nucleus at (0,0,0) stays perfectly centered on screen.
      const warpCameraBoost = warpFactor * 600;
      const cameraZ = currentZoom * 350 + warpCameraBoost;

      // Kill rotation as zoom deepens so the camera locks onto center
      const rotDampen = Math.max(0, 1 - currentZoom * 1.4);
      const rotSpeed = 0.15 * rotDampen;
      const rotY = time * rotSpeed;
      const rotXAngle = Math.sin(time * 0.1) * 0.15 * rotDampen;

      // Render warp-speed lines behind everything
      renderWarpLines(cx, cy, warpFactor, time);

      // Render neuron branches
      trees.forEach((tree) => {
        renderBranch(tree, time, focalLength, cx, cy, rotY, rotXAngle, 0.9, cameraZ, warpFactor);
      });

      // Render nucleus
      renderNucleus(time, focalLength, cx, cy, rotY, rotXAngle, currentZoom, cameraZ, warpFactor);

      // Render synapse particles
      renderParticles(time, focalLength, cx, cy, rotY, rotXAngle, cameraZ, warpFactor);

      // ── Synaptic Stems (after warp completes) ──────────────
      if (hasReachedNucleusRef.current) {
        const stemTime = time;
        const positions: { x: number; y: number }[] = [];

        // Scale stems relative to viewport so they space evenly
        // postGlowExpand eases to 1.18 after warp, pushing stems into negative space
        const stemScale = (Math.min(width, height) / 600) * postGlowExpand;

        SYNAPTIC_STEMS.forEach((stem, idx) => {
          const wobble = Math.sin(stemTime * 1.5 + stem.angle * 3) * 3;
          const scaledLen = stem.length * stemScale;
          const endX = cx + Math.cos(stem.angle) * (scaledLen + wobble);
          const endY = cy + Math.sin(stem.angle) * (scaledLen + wobble);

          // Dendrite branch line
          const isHot = hoveredStemRef.current === idx || activeStemRef.current === idx;
          const baseAlpha = isHot ? 0.95 : 0.55 + 0.15 * Math.sin(stemTime * 2 + idx);

          // Main stem line
          const grad = ctx.createLinearGradient(cx, cy, endX, endY);
          grad.addColorStop(0, `rgba(0, 220, 255, ${baseAlpha * 0.3})`);
          grad.addColorStop(1, `rgba(0, 255, 200, ${baseAlpha})`);

          ctx.beginPath();
          ctx.moveTo(cx, cy);
          // Slight organic curve
          const cpx = (cx + endX) / 2 + Math.sin(stemTime + idx * 1.3) * 12;
          const cpy = (cy + endY) / 2 + Math.cos(stemTime + idx * 0.9) * 12;
          ctx.quadraticCurveTo(cpx, cpy, endX, endY);
          ctx.strokeStyle = grad;
          ctx.lineWidth = isHot ? 3 : 1.8;
          ctx.lineCap = "round";
          ctx.stroke();

          // Terminal bulb (synapse)
          const bulbRadius = isHot ? 10 : 6;
          const bulbGlow = ctx.createRadialGradient(endX, endY, 0, endX, endY, bulbRadius * 2.5);
          bulbGlow.addColorStop(0, `rgba(0, 255, 220, ${isHot ? 0.9 : 0.6})`);
          bulbGlow.addColorStop(0.5, `rgba(0, 200, 255, ${isHot ? 0.35 : 0.15})`);
          bulbGlow.addColorStop(1, "rgba(0, 0, 0, 0)");
          ctx.beginPath();
          ctx.arc(endX, endY, bulbRadius * 2.5, 0, Math.PI * 2);
          ctx.fillStyle = bulbGlow;
          ctx.fill();

          ctx.beginPath();
          ctx.arc(endX, endY, bulbRadius, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(180, 255, 240, ${isHot ? 0.95 : 0.7})`;
          ctx.fill();

          // Small branching dendrites off the stem
          for (let b = 0; b < 3; b++) {
            const t = 0.35 + b * 0.2;
            const branchStartX = cx + (endX - cx) * t + Math.sin(stemTime + b) * 4;
            const branchStartY = cy + (endY - cy) * t + Math.cos(stemTime + b) * 4;
            const branchAngle = stem.angle + (b - 1) * 0.6;
            const branchLen = 18 + Math.sin(stemTime * 2 + b) * 4;
            const branchEndX = branchStartX + Math.cos(branchAngle) * branchLen;
            const branchEndY = branchStartY + Math.sin(branchAngle) * branchLen;

            ctx.beginPath();
            ctx.moveTo(branchStartX, branchStartY);
            ctx.lineTo(branchEndX, branchEndY);
            ctx.strokeStyle = `rgba(0, 220, 255, ${baseAlpha * 0.35})`;
            ctx.lineWidth = 0.8;
            ctx.stroke();
          }

          positions.push({ x: endX, y: endY });
        });

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
        className="w-full h-full cursor-default"
        style={{ touchAction: "none" }}
      />

      {/* Synaptic Fact Overlay — glassmorphic tooltip at node position */}
      {/* z-40 keeps tooltip above the hero-text z-10 layer */}
      <FactOverlay
        activeStemId={hoveredStemId}
        visible={hasReachedNucleus}
        positions={stemPositions}
        facts={SYNAPTIC_STEMS.map((s) => s.fact)}
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

      {/* Scroll hint — shown before user starts scrolling */}
      {zoom < 0.03 && !isWarping && !hasReachedNucleus && (
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 z-20 pointer-events-none">
          <div className="w-5 h-8 border-2 border-cyan-500/20 rounded-full flex justify-center pt-1">
            <div className="w-1 h-2 bg-cyan-400/40 rounded-full animate-bounce" />
          </div>
        </div>
      )}
    </div>
  );
}
