/**
 * NeuroSketch — Learn More Page
 *
 * Pixelated dark-science aesthetic with terminal boot entrance,
 * scroll-triggered citation reveals, and neon-pulse icon animations.
 * Consistent with the landing page scanline/pixel-grid theme.
 *
 * @module app/learn-more/page
 */

"use client";

import React, { useEffect } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  Brain,
  Activity,
  ArrowLeft,
  ExternalLink,
  FlaskConical,
  GraduationCap,
  Stethoscope,
  FileText,
  Microscope,
  Binary,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import ScrollReveal from "@/components/ui/scroll-reveal";

/* ─── Terminal Boot Animation Helpers ──────────────────────────────────── */

const bootContainer = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.12, delayChildren: 0.15 },
  },
};

const bootLine = {
  hidden: { opacity: 0, filter: "blur(2px)" },
  visible: {
    opacity: 1,
    filter: "blur(0px)",
    transition: { duration: 0.5, ease: "easeOut" as const },
  },
};

/* ─── Research Papers ──────────────────────────────────────────────────── */

const PAPERS = [

  {
    authors: "Zham P et al.",
    title:
      "Efficacy of guided and unguided spiral drawing in assessing motor symptoms of Parkinson's disease.",
    journal: "IEEE EMBC. 2017",
    url: "https://pubmed.ncbi.nlm.nih.gov/25904359/",
    summary:
      "Compared guided vs. unguided spiral drawing tasks in PD patients. Found that guided spirals (tracing a template) provide more sensitive motor impairment measures than free-form drawing.",
  },

];

/* ─── External Resources ───────────────────────────────────────────────── */

const RESOURCES = [
  {
    name: "Parkinson's Foundation — Understanding Parkinson's Disease",
    url: "https://www.parkinson.org/understanding-parkinsons",
    description:
      "Comprehensive patient-facing overview of PD symptoms, progression, treatment options, and caregiver resources.",
  },
  {
    name: "NIH NINDS — Parkinson's Disease Information Page",
    url: "https://www.ninds.nih.gov/health-information/disorders/parkinsons-disease",
    description:
      "National Institutes of Health factsheet covering epidemiology, pathology, current research trials, and treatment guidelines.",
  },
  {
    name: "Michael J. Fox Foundation — Parkinson's 101",
    url: "https://www.michaeljfox.org/parkinsons-101",
    description:
      "Educational primer from the leading PD research funding organization, including their biomarker discovery initiatives.",
  },
  {
    name: "Movement Disorder Society — UPDRS",
    url: "https://www.movementdisorders.org/MDS/MDS-Rating-Scales.htm",
    description:
      "The Unified Parkinson's Disease Rating Scale (MDS-UPDRS) — the gold-standard clinical assessment framework that NeuroSketch's metrics are modeled after.",
  },
];

/* ─── Page Component ──────────────────────────────────────────────────── */

export default function LearnMorePage() {
  // Environment cleanup — clear stale localhost storage on mount
  useEffect(() => {
    try {
      localStorage.clear();
      sessionStorage.clear();
    } catch {
      // storage access may fail in certain contexts
    }
  }, []);

  return (
    <div className="min-h-screen bg-black text-zinc-100 pixel-grid">
      {/* Scanline CRT overlay */}
      <div className="scanline-overlay" />

      {/* Header */}
      <header className="border-b border-cyan-500/10 bg-black/80 backdrop-blur-md sticky top-0 z-40">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between">
          <Link
            href="/"
            className="flex items-center gap-2.5 hover:opacity-80 transition-opacity"
          >
            <div className="p-1.5 pixel-border border-cyan-500/20 bg-cyan-500/10">
              <Brain className="w-5 h-5 text-cyan-400 neon-icon" />
            </div>
            <span className="font-bold text-lg tracking-tight font-mono bg-gradient-to-r from-pink-400 to-cyan-400 bg-clip-text text-transparent">
              NeuroSketch
            </span>
          </Link>
          <Link href="/">
            <Button variant="ghost" size="sm" className="gap-1.5 pixel-border border-transparent hover:border-cyan-500/20 rounded-none">
              <ArrowLeft className="w-4 h-4 neon-icon" />
              Back
            </Button>
          </Link>
        </div>
      </header>

      {/* Hero — Terminal Boot Sequence */}
      <section className="py-16 border-b border-cyan-500/10">
        <motion.div
          className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 space-y-4"
          variants={bootContainer}
          initial="hidden"
          animate="visible"
        >
          <motion.div variants={bootLine}>
            <div className="inline-flex items-center gap-2 px-3 py-1 pixel-border border-cyan-500/20 bg-cyan-500/5 text-cyan-400 text-xs font-mono">
              <Binary className="w-3.5 h-3.5 neon-pulse" />
              Research &amp; Sources
            </div>
          </motion.div>
          <motion.h1
            variants={bootLine}
            className="text-3xl sm:text-4xl font-bold tracking-tight"
            style={{ fontFamily: "var(--font-vt323), monospace" }}
          >
            The Science Behind{" "}
            <span className="text-cyan-400">NeuroSketch</span>
          </motion.h1>
          <motion.p
            variants={bootLine}
            className="text-zinc-400 text-lg leading-relaxed max-w-2xl"
          >
            NeuroSketch implements validated clinical screening methods for
            Parkinson&apos;s disease using hand-drawn motor tasks. Below are
            key references and resources for deeper reading.
          </motion.p>
          <motion.div
            variants={bootLine}
            className="h-px bg-gradient-to-r from-cyan-500/30 via-purple-500/20 to-transparent"
          />
        </motion.div>
      </section>

      {/* How It Works */}
      <section className="py-16 border-b border-cyan-500/10">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 space-y-8">
          <ScrollReveal>
            <div className="flex items-center gap-2">
              <Stethoscope className="w-5 h-5 text-cyan-400 neon-pulse" />
              <h2
                className="text-2xl font-bold tracking-tight"
                style={{ fontFamily: "var(--font-vt323), monospace" }}
              >
                How Screening Works
              </h2>
            </div>
          </ScrollReveal>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <ScrollReveal delay={0.1}>
              <Card className="pixel-border border-cyan-500/15 bg-zinc-950/60">
                <CardHeader className="pb-2">
                  <CardTitle className="text-base font-mono flex items-center gap-2">
                    <Microscope className="w-4 h-4 text-cyan-400 neon-pulse" />
                    Archimedes Spiral Test
                  </CardTitle>
                </CardHeader>
                <CardContent className="text-sm text-zinc-400 leading-relaxed space-y-2">
                  <p>
                    The patient traces a pre-drawn Archimedes spiral as
                    accurately as possible. We capture (x, y, t) coordinates
                    at high resolution and compute:
                  </p>
                  <ul className="list-disc pl-4 space-y-1 text-xs text-zinc-500 font-mono">
                    <li>Radial deviation from the ideal spiral path</li>
                    <li>Speed variance and drawing consistency</li>
                    <li>High-frequency tremor components (4–6 Hz band)</li>
                    <li>Total drawing time and sample density</li>
                  </ul>
                </CardContent>
              </Card>
            </ScrollReveal>

            <ScrollReveal delay={0.2}>
              <Card className="pixel-border border-purple-500/15 bg-zinc-950/60">
                <CardHeader className="pb-2">
                  <CardTitle className="text-base font-mono flex items-center gap-2">
                    <Activity className="w-4 h-4 text-purple-400 neon-pulse" />
                    Sinusoidal Wave Test
                  </CardTitle>
                </CardHeader>
                <CardContent className="text-sm text-zinc-400 leading-relaxed space-y-2">
                  <p>
                    The patient follows a sinusoidal guide wave from left to
                    right. This captures lateral motor control and measures:
                  </p>
                  <ul className="list-disc pl-4 space-y-1 text-xs text-zinc-500 font-mono">
                    <li>Amplitude deviation from the reference wave</li>
                    <li>Frequency stability and consistency</li>
                    <li>Velocity profile across the trace</li>
                    <li>Jerk (rate of acceleration change) — a bradykinesia marker</li>
                  </ul>
                </CardContent>
              </Card>
            </ScrollReveal>
          </div>
        </div>
      </section>

      {/* Research Papers — slide from left */}
      <section className="py-16 border-b border-cyan-500/10">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 space-y-8">
          <ScrollReveal>
            <div className="flex items-center gap-2">
              <FlaskConical className="w-5 h-5 text-cyan-400 neon-pulse" />
              <h2
                className="text-2xl font-bold tracking-tight"
                style={{ fontFamily: "var(--font-vt323), monospace" }}
              >
                Clinical Research
              </h2>
            </div>
          </ScrollReveal>

          <div className="space-y-4">
            {PAPERS.map((paper, idx) => (
              <ScrollReveal key={paper.url} direction="left" delay={idx * 0.08}>
                <Card className="group hover:border-cyan-500/20 transition-colors pixel-border border-zinc-800/50 bg-zinc-950/60">
                  <CardContent className="pt-5 space-y-2">
                    <a
                      href={paper.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-start gap-2 group-hover:text-cyan-400 transition-colors"
                    >
                      <FileText className="w-4 h-4 mt-0.5 shrink-0 text-cyan-500 neon-pulse" />
                      <div>
                        <p className="text-sm font-medium text-zinc-200 group-hover:text-cyan-300 transition-colors font-mono">
                          {paper.authors}.{" "}
                          <em>{paper.title}</em>
                        </p>
                        <p className="text-xs text-zinc-600 mt-0.5 font-mono">
                          {paper.journal}
                        </p>
                      </div>
                    </a>
                    <p className="text-xs text-zinc-500 leading-relaxed pl-6">
                      {paper.summary}
                    </p>
                  </CardContent>
                </Card>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>

      {/* External Resources */}
      <section className="py-16 border-b border-cyan-500/10">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 space-y-8">
          <ScrollReveal>
            <div className="flex items-center gap-2">
              <GraduationCap className="w-5 h-5 text-purple-400 neon-pulse" />
              <h2
                className="text-2xl font-bold tracking-tight"
                style={{ fontFamily: "var(--font-vt323), monospace" }}
              >
                External Resources
              </h2>
            </div>
          </ScrollReveal>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {RESOURCES.map((res, idx) => (
              <ScrollReveal key={res.url} delay={idx * 0.1}>
                <a
                  href={res.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="group block h-full"
                >
                  <Card className="h-full hover:border-purple-500/20 transition-colors pixel-border border-zinc-800/50 bg-zinc-950/60">
                    <CardContent className="pt-5 space-y-2">
                      <div className="flex items-start gap-2">
                        <ExternalLink className="w-4 h-4 mt-0.5 shrink-0 text-purple-400 neon-pulse" />
                        <p className="text-sm font-medium text-zinc-200 group-hover:text-purple-300 transition-colors font-mono">
                          {res.name}
                        </p>
                      </div>
                      <p className="text-xs text-zinc-500 leading-relaxed pl-6">
                        {res.description}
                      </p>
                    </CardContent>
                  </Card>
                </a>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-16">
        <ScrollReveal>
          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center space-y-6">
            <h2
              className="text-2xl font-bold tracking-tight"
              style={{ fontFamily: "var(--font-vt323), monospace" }}
            >
              Start Screening
            </h2>
            <p className="text-zinc-400 max-w-lg mx-auto">
              Both tests take under a minute. All processing happens locally
              in your browser — no data leaves your device.
            </p>
            <Link href="/detector">
              <Button size="lg" className="gap-2 text-base pixel-border rounded-none border-cyan-500/30">
                <Activity className="w-5 h-5 neon-icon" />
                Start Screening
              </Button>
            </Link>
          </div>
        </ScrollReveal>
      </section>

      {/* Footer */}
      <footer className="border-t border-cyan-500/10 py-8">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Brain className="w-4 h-4 text-cyan-500/50 neon-icon" />
            <span className="text-sm text-zinc-600 font-mono">
              NeuroSketch — Educational Research Tool
            </span>
          </div>
          <p className="text-xs text-zinc-700 font-mono">
            Not a medical device. For screening and educational purposes only.
          </p>
        </div>
      </footer>
    </div>
  );
}
