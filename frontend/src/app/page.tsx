/**
 * NeuroSketch — Landing Page
 *
 * Pixelated Cyber-Science theme with retro medical-monitor aesthetic.
 * Scanline overlay, pixel-grid pattern, stepped borders, neon icon
 * glow, and scroll-triggered Framer Motion entrance animations.
 *
 * @module app/page
 */

"use client";

import React, { useState, useEffect, useCallback } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import {
  Brain,
  Activity,
  ArrowRight,
  Microscope,
  Zap,
  Shield,
  BookOpen,
  Binary,
  BrainCircuit,
  FileText,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import ScrollReveal from "@/components/ui/scroll-reveal";

// Dynamic import to avoid SSR issues with canvas
const HeroNeuron = dynamic(
  () => import("@/components/canvas/HeroNeuron"),
  { ssr: false }
);

/* ─── Feature Data ─────────────────────────────────────────────────────── */

const FEATURES = [
  {
    icon: Microscope,
    title: "Spiral Analysis",
    description:
      "Draw an Archimedes spiral and get instant motor function analysis using validated clinical metrics.",
  },
  {
    icon: Activity,
    title: "Wave Analysis",
    description:
      "Trace a sinusoidal wave to measure hand tremor frequency, speed consistency, and lateral deviation.",
  },
  {
    icon: Zap,
    title: "Real-Time Detection",
    description:
      "Live computation of radial deviation, speed variance, and tremor scoring as you draw.",
  },
  {
    icon: Shield,
    title: "Privacy First",
    description:
      "All analysis runs locally in your browser. No data is transmitted or stored on any server.",
  },
];

/* ─── Page Component ──────────────────────────────────────────────────── */

export default function HomePage() {
  const [learnOpen, setLearnOpen] = useState(false);
  const [cinematicActive, setCinematicActive] = useState(false);

  // Cinematic mode: bidirectional — fades out on dive, fades back in on zoom-out
  const handleCinematicChange = useCallback(
    (state: { isWarping: boolean; hasReachedNucleus: boolean; zoom: number }) => {
      // Activate when warp starts OR zoom crosses threshold;
      // deactivate when zoom drops back below — enables snap-back
      setCinematicActive(state.isWarping || state.hasReachedNucleus || state.zoom > 0.4);
    },
    []
  );

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

      {/* Navigation — fades out during cinematic dive */}
      <AnimatePresence>
        {!cinematicActive && (
          <motion.nav
            key="nav"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.8, ease: "easeInOut" }}
            className="fixed top-0 left-0 right-0 z-50 border-b border-cyan-500/10 bg-black/70 backdrop-blur-xl"
          >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2.5 hover:opacity-80 transition-opacity">
            <div className="p-1.5 pixel-border border-cyan-500/20 bg-cyan-500/10">
              <Brain className="w-5 h-5 text-cyan-400 neon-icon" />
            </div>
            <span className="font-bold text-lg tracking-tight font-mono bg-gradient-to-r from-pink-400 to-cyan-400 bg-clip-text text-transparent">
              NeuroSketch
            </span>
          </Link>
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="sm"
              className="gap-1.5 pixel-border border-transparent hover:border-cyan-500/20 rounded-none"
              onClick={() => setLearnOpen(true)}
            >
              <BookOpen className="w-4 h-4 neon-icon" />
              Learn
            </Button>
            <Link href="/detector">
              <Button size="sm" className="gap-1.5 pixel-border rounded-none border-cyan-500/30">
                <Activity className="w-4 h-4 neon-icon" />
                Detector
              </Button>
            </Link>
            <Link href="/citations">
              <Button
                variant="ghost"
                size="sm"
                className="gap-1.5 pixel-border border-transparent hover:border-cyan-500/20 rounded-none"
              >
                <FileText className="w-4 h-4 neon-icon" />
                Citations
              </Button>
            </Link>
          </div>
        </div>
          </motion.nav>
        )}
      </AnimatePresence>

      {/* Learn Popover Card — also fades with cinematic mode */}
      <AnimatePresence>
        {!cinematicActive && learnOpen && (
          <motion.div
            key="learn-popover"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5, ease: "easeInOut" }}
          >
            <div className="fixed inset-0 z-[55]" onClick={() => setLearnOpen(false)} />
            <div className="fixed top-16 right-4 z-[60] w-80 animate-fade-in">
              <Card className="bg-zinc-950/95 backdrop-blur-xl border-cyan-500/15 shadow-2xl pixel-border">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm text-cyan-300 flex items-center gap-2 font-mono">
                    <BrainCircuit className="w-4 h-4 neon-icon" />
                    Parkinson&apos;s Screening
                  </CardTitle>
                  <CardDescription className="text-xs">
                    Motor biomarker analysis for early detection
                  </CardDescription>
                </CardHeader>
                <CardContent className="text-xs text-zinc-400 space-y-2 pt-0">
                  <p>
                    Archimedes spiral and wave tests capture motor control
                    anomalies — tremor, bradykinesia, and micrographia — among
                    the earliest detectable signs of PD.
                  </p>
                  <p>
                    We record (x,&nbsp;y,&nbsp;t) coordinates at high resolution
                    and compute radial deviation, speed variance, and tremor
                    frequency to produce a screening score.
                  </p>
                  <p className="text-zinc-600 text-[10px] font-mono">
                    Not a medical device. For screening &amp; educational purposes only.
                  </p>
                  <Link href="/detector">
                    <Button size="sm" className="w-full gap-1.5 mt-2 pixel-border rounded-none border-cyan-500/30" onClick={() => setLearnOpen(false)}>
                      <Activity className="w-3.5 h-3.5 neon-icon" />
                      Try the Detector
                    </Button>
                  </Link>
                </CardContent>
              </Card>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Hero Section */}
      <section className="relative h-screen w-full overflow-hidden">
        {/* 3D Neuron Canvas Background */}
        <div className="absolute inset-0">
          <HeroNeuron onCinematicChange={handleCinematicChange} />
        </div>

        {/* Hero Text Overlay — fades out during cinematic dive */}
        <AnimatePresence>
          {!cinematicActive && (
            <motion.div
              key="hero-text"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.8, ease: "easeInOut" }}
              className="absolute inset-0 z-10 flex items-center justify-center pointer-events-none"
            >
          <div className="text-center space-y-6 px-4 pointer-events-auto">
            <div className="inline-flex items-center gap-2 px-3 py-1 pixel-border border-cyan-500/25 bg-black/60 text-cyan-400 text-xs font-mono animate-fade-in backdrop-blur-sm">
              <Binary className="w-3.5 h-3.5 neon-icon" />
              Early Detection Research Tool
            </div>

            <h1 className="text-5xl sm:text-7xl font-bold tracking-tight animate-slide-up font-mono drop-shadow-[0_0_20px_rgba(0,0,0,0.9)]">
              <span className="bg-gradient-to-b from-white to-zinc-300 bg-clip-text text-transparent">
                Neuro
              </span>
              <span className="bg-gradient-to-r from-pink-400 to-cyan-400 bg-clip-text text-transparent">
                Sketch
              </span>
            </h1>

            <p className="text-zinc-300 text-lg sm:text-xl max-w-xl mx-auto animate-slide-up leading-relaxed drop-shadow-[0_0_12px_rgba(0,0,0,0.8)]">
              Screen for early Parkinson&apos;s indicators using the
              Archimedes spiral test — powered by real-time motor analysis.
            </p>

            <div className="flex items-center justify-center gap-4 animate-slide-up mt-14">
              <Link href="/detector">
                <Button size="lg" className="gap-2 text-base pixel-border rounded-none border-cyan-500/30">
                  <Activity className="w-5 h-5 neon-icon" />
                  Begin Screening
                </Button>
              </Link>
              <Link href="/learn-more">
                <Button
                  variant="outline"
                  size="lg"
                  className="gap-2 text-base pixel-border rounded-none border-cyan-500/25"
                >
                  <BookOpen className="w-5 h-5 neon-icon" />
                  Learn More
                </Button>
              </Link>
            </div>
          </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Bottom gradient fade — fades with UI */}
        <AnimatePresence>
          {!cinematicActive && (
            <motion.div
              key="hero-gradient"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.8, ease: "easeInOut" }}
              className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-black to-transparent"
            />
          )}
        </AnimatePresence>
      </section>

      {/* Features Section */}
      <section className="relative z-10 py-24 bg-black pixel-grid">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <ScrollReveal>
            <div className="text-center mb-8">
              <h2 className="text-3xl sm:text-4xl font-bold tracking-tight mb-0 font-mono">
                Clinical-Grade Screening,{" "}
                <span className="text-cyan-400">Instantly</span>
              </h2>
              <p className="text-zinc-500 max-w-2xl mx-auto">
                NeuroSketch brings validated neurological screening methods to
                your browser with zero data collection and real-time results.
              </p>
            </div>
          </ScrollReveal>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {FEATURES.map((feature, idx) => (
              <ScrollReveal key={feature.title} delay={idx * 0.1}>
                <Card className="group hover:border-cyan-500/20 transition-colors duration-300 pixel-border border-zinc-800/50 bg-zinc-950/60">
                  <CardHeader>
                    <div className="p-3 pixel-border border-cyan-500/15 bg-cyan-500/10 w-fit mb-3 group-hover:bg-cyan-500/15 transition-colors">
                      <feature.icon className="w-6 h-6 text-cyan-400 neon-icon" />
                    </div>
                    <CardTitle className="text-xl font-mono">
                      {feature.title}
                    </CardTitle>
                    <CardDescription className="leading-relaxed">
                      {feature.description}
                    </CardDescription>
                  </CardHeader>
                </Card>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>

      {/* What is NeuroSketch? */}
      <section className="relative z-10 py-20 bg-gradient-to-b from-black to-zinc-950 border-t border-cyan-500/10">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <ScrollReveal>
            <div className="text-center space-y-6">
              <h2 className="text-3xl sm:text-4xl font-bold tracking-tight font-mono">
                What is{" "}
                <span className="bg-gradient-to-r from-pink-400 to-cyan-400 bg-clip-text text-transparent">
                  NeuroSketch
                </span>
                ?
              </h2>
              <p className="text-zinc-400 text-lg leading-relaxed max-w-2xl mx-auto">
                NeuroSketch is a browser-based neurological screening platform
                that uses validated spiral and wave drawing tests to detect
                early motor function anomalies associated with Parkinson&apos;s
                disease.
              </p>
              <p className="text-zinc-500 leading-relaxed max-w-2xl mx-auto">
                All analysis runs entirely in your browser — zero data
                collection, zero uploads, zero tracking. Powered by
                real-time coordinate capture and machine learning inference,
                NeuroSketch delivers clinical-grade screening without
                compromising your privacy.
              </p>
            </div>
          </ScrollReveal>
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative z-10 py-24 bg-gradient-to-t from-zinc-950 to-black border-t border-cyan-500/10">
        <ScrollReveal>
          <div className="max-w-3xl mx-auto px-4 text-center space-y-8">
            <h2 className="text-3xl sm:text-4xl font-bold tracking-tight font-mono">
              Ready to test?
            </h2>
            <p className="text-zinc-400 text-lg leading-relaxed">
              The test takes less than 30 seconds. All processing
              happens locally — your data never leaves your device.
            </p>
            <Link href="/detector">
              <Button size="lg" className="gap-2 text-base pixel-border rounded-none border-cyan-500/30">
                Start The Test
                <ArrowRight className="w-5 h-5 neon-icon" />
              </Button>
            </Link>
          </div>
        </ScrollReveal>
      </section>

      {/* Footer */}
      <footer className="border-t border-cyan-500/10 py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Brain className="w-4 h-4 text-cyan-500/50 neon-icon" />
            <span className="text-sm text-zinc-600 font-mono">
              NeuroSketch — Educational Research Tool
            </span>
          </div>
          <p className="text-xs text-zinc-700 font-mono">
            Not a medical device. For screening and educational purposes
            only.
          </p>
        </div>
      </footer>
    </div>
  );
}
