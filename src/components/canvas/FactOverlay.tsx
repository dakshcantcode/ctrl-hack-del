/**
 * FactOverlay — Glassmorphic tooltip that tracks a synaptic stem node
 *
 * Uses Framer Motion AnimatePresence to slide-up from the node's
 * screen-projected position with a frosted-glass backdrop.
 *
 * @module components/canvas/FactOverlay
 */

"use client";

import React from "react";
import { motion, AnimatePresence } from "framer-motion";

interface FactData {
  title: string;
  body: string;
  stat: string;
  statLabel: string;
}

interface FactOverlayProps {
  /** Index of the active stem (null = hidden) */
  activeStemId: number | null;
  /** Whether the nucleus has been reached (stems visible) */
  visible: boolean;
  /** Screen-space positions of each stem endpoint */
  positions: { x: number; y: number }[];
  /** Fact data for each stem */
  facts: FactData[];
  /** Keep fact visible while hovering over the tooltip itself */
  onMouseEnter?: () => void;
  onMouseLeave?: () => void;
}

export default function FactOverlay({
  activeStemId,
  visible,
  positions,
  facts,
  onMouseEnter,
  onMouseLeave,
}: FactOverlayProps) {
  const isActive = visible && activeStemId !== null;
  const pos = isActive ? positions[activeStemId] : null;
  const fact = isActive ? facts[activeStemId] : null;

  return (
    <AnimatePresence mode="wait">
      {isActive && pos && fact && (
        <motion.div
          key={activeStemId}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 12 }}
          transition={{ type: "spring", stiffness: 260, damping: 22, mass: 0.8 }}
          className="absolute z-30 pointer-events-auto"
          style={{
            left: pos.x,
            top: pos.y,
            transform: "translate(-50%, -110%)",
          }}
          onMouseEnter={onMouseEnter}
          onMouseLeave={onMouseLeave}
        >
          <div className="relative max-w-[260px] w-max px-4 py-3 rounded-xl bg-white/10 backdrop-blur-md border border-white/10 shadow-[0_8px_32px_rgba(0,0,0,0.35)]">
            {/* Caret */}
            <div className="absolute left-1/2 -translate-x-1/2 -bottom-1.5 w-3 h-3 rotate-45 bg-white/10 backdrop-blur-md border-b border-r border-white/10" />

            <h3 className="text-cyan-300 font-semibold text-xs tracking-wide mb-1">
              {fact.title}
            </h3>
            <p className="text-zinc-300/80 text-[11px] leading-relaxed mb-2">
              {fact.body}
            </p>
            <div className="flex items-baseline gap-1.5">
              <span className="text-cyan-400 font-mono text-sm font-bold">
                {fact.stat}
              </span>
              <span className="text-zinc-500 text-[9px] uppercase tracking-widest">
                {fact.statLabel}
              </span>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
