/**
 * FactCarousel — Animated clinical-fact carousel
 *
 * Uses Framer Motion AnimatePresence for slide-in / slide-out
 * transitions. Current fact slides out left, next slides in right.
 *
 * @module components/ui/fact-carousel
 */

"use client";

import React, { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowRight, X, ChevronLeft, ChevronRight, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";

/* ─── Clinical Facts ──────────────────────────────────────────────────── */

const CLINICAL_FACTS = [
  {
    title: "Dopamine & Movement",
    body: "Parkinson's disease is caused by the progressive loss of dopaminergic neurons in the substantia nigra, leading to tremor, rigidity, and bradykinesia.",
    stat: "~60-80%",
    statLabel: "neurons lost before motor symptoms appear",
  },
  {
    title: "Early Detection Matters",
    body: "By the time motor symptoms appear, the majority of dopaminergic neurons have already been lost. Early screening tools like spiral analysis can detect subtle motor changes years earlier.",
    stat: "5-10 yrs",
    statLabel: "earlier detection possible with motor screening",
  },
  {
    title: "The Archimedes Spiral Test",
    body: "Drawing an Archimedes spiral is a validated clinical method. Patients with Parkinson's show increased drawing irregularity, slower speeds, and higher radial deviation.",
    stat: "94%",
    statLabel: "sensitivity in clinical spiral analysis studies",
  },
  {
    title: "Neural Communication",
    body: "Each neuron connects to up to 10,000 others through synapses. In Parkinson's, the loss of these connections in the basal ganglia disrupts the motor circuit.",
    stat: "10,000+",
    statLabel: "synaptic connections per neuron",
  },
  {
    title: "Micrographia",
    body: "Progressive reduction in handwriting size (micrographia) is one of the earliest motor signs of Parkinson's disease, often appearing years before clinical diagnosis.",
    stat: "#1",
    statLabel: "earliest detectable motor sign",
  },
];

/* ─── Slide Variants ──────────────────────────────────────────────────── */

const slideVariants = {
  enter: (direction: number) => ({
    x: direction > 0 ? 300 : -300,
    opacity: 0,
    scale: 0.95,
  }),
  center: {
    x: 0,
    opacity: 1,
    scale: 1,
  },
  exit: (direction: number) => ({
    x: direction > 0 ? -300 : 300,
    opacity: 0,
    scale: 0.95,
  }),
};

/* ─── Props ────────────────────────────────────────────────────────────── */

interface FactCarouselProps {
  /** Called when the user dismisses the carousel */
  onClose: () => void;
}

/* ─── Component ────────────────────────────────────────────────────────── */

export function FactCarousel({ onClose }: FactCarouselProps) {
  const [[page, direction], setPage] = useState([0, 0]);

  const factIndex = ((page % CLINICAL_FACTS.length) + CLINICAL_FACTS.length) % CLINICAL_FACTS.length;

  const paginate = useCallback((newDirection: number) => {
    setPage(([prev]) => [prev + newDirection, newDirection]);
  }, []);

  const fact = CLINICAL_FACTS[factIndex];

  return (
    <motion.div
      initial={{ opacity: 0, y: 40 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 40 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className="absolute inset-0 z-30 flex items-center justify-center"
    >
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/70 backdrop-blur-md" />

      {/* Carousel Card */}
      <div className="relative z-10 w-full max-w-lg mx-4">
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute -top-10 right-0 p-2 rounded-full text-zinc-400 hover:text-white hover:bg-zinc-800/50 transition-colors"
          aria-label="Close"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Card */}
        <div className="relative overflow-hidden rounded-2xl border border-cyan-500/20 bg-zinc-950/90 backdrop-blur-xl">
          {/* Header accent */}
          <div className="h-1 w-full bg-gradient-to-r from-cyan-500 via-blue-500 to-cyan-500" />

          {/* Content area */}
          <div className="relative min-h-[280px] p-8">
            <AnimatePresence initial={false} custom={direction} mode="wait">
              <motion.div
                key={page}
                custom={direction}
                variants={slideVariants}
                initial="enter"
                animate="center"
                exit="exit"
                transition={{
                  x: { type: "spring", stiffness: 300, damping: 30 },
                  opacity: { duration: 0.2 },
                  scale: { duration: 0.2 },
                }}
                className="space-y-5"
              >
                {/* Fact number badge */}
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-cyan-500/20 bg-cyan-500/5 text-cyan-400 text-xs font-mono">
                  <Sparkles className="w-3 h-3" />
                  Fact {factIndex + 1} of {CLINICAL_FACTS.length}
                </div>

                {/* Title */}
                <h3 className="text-2xl font-bold text-white tracking-tight">
                  {fact.title}
                </h3>

                {/* Body */}
                <p className="text-zinc-300 leading-relaxed text-sm">
                  {fact.body}
                </p>

                {/* Stat highlight */}
                <div className="flex items-baseline gap-3 pt-2">
                  <span className="text-3xl font-bold text-cyan-400 font-mono">
                    {fact.stat}
                  </span>
                  <span className="text-xs text-zinc-500 uppercase tracking-wider">
                    {fact.statLabel}
                  </span>
                </div>
              </motion.div>
            </AnimatePresence>
          </div>

          {/* Navigation footer */}
          <div className="flex items-center justify-between px-8 py-4 border-t border-zinc-800/50">
            {/* Dots */}
            <div className="flex items-center gap-1.5">
              {CLINICAL_FACTS.map((_, i) => (
                <div
                  key={i}
                  className={`w-1.5 h-1.5 rounded-full transition-all duration-300 ${
                    i === factIndex
                      ? "bg-cyan-400 w-4"
                      : "bg-zinc-700"
                  }`}
                />
              ))}
            </div>

            {/* Arrows */}
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => paginate(-1)}
                className="h-8 w-8 p-0"
                aria-label="Previous fact"
              >
                <ChevronLeft className="w-4 h-4" />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => paginate(1)}
                className="h-8 w-8 p-0 gap-1"
                aria-label="Next fact"
              >
                <ChevronRight className="w-4 h-4" />
              </Button>
              <div className="w-px h-4 bg-zinc-800 mx-1" />
              <Button
                variant="outline"
                size="sm"
                onClick={onClose}
                className="gap-1.5 text-xs"
              >
                Continue
                <ArrowRight className="w-3 h-3" />
              </Button>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
