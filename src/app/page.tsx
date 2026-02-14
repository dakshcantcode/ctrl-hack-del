/**
 * NeuroSketch — Landing Page
 *
 * Hero section featuring the Living Neuron 3D canvas engine with
 * scroll-to-zoom interaction. Includes navigation to the detector
 * dashboard and feature highlights.
 *
 * @module app/page
 */

"use client";

import React, { useState } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import {
  Brain,
  Activity,
  ArrowRight,
  Microscope,
  Zap,
  Shield,
  BookOpen,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

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

  return (
    <div className="min-h-screen bg-black text-zinc-100">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 border-b border-zinc-900/50 bg-black/60 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2.5 hover:opacity-80 transition-opacity">
            <div className="p-1.5 rounded-lg bg-cyan-500/10">
              <Brain className="w-5 h-5 text-cyan-400" />
            </div>
            <span className="font-bold text-lg tracking-tight bg-gradient-to-r from-pink-400 to-cyan-400 bg-clip-text text-transparent">
              NeuroSketch
            </span>
          </Link>
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="sm"
              className="gap-1.5"
              onClick={() => setLearnOpen(true)}
            >
              <BookOpen className="w-4 h-4" />
              Learn
            </Button>
            <Link href="/detector">
              <Button size="sm" className="gap-1.5">
                <Activity className="w-4 h-4" />
                Detector
              </Button>
            </Link>
          </div>
        </div>
      </nav>

      {/* Learn Popover Card */}
      {learnOpen && (
        <>
          <div className="fixed inset-0 z-[55]" onClick={() => setLearnOpen(false)} />
          <div className="fixed top-16 right-4 z-[60] w-80 animate-fade-in">
            <Card className="bg-zinc-950/95 backdrop-blur-xl border-zinc-800 shadow-2xl">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-cyan-300 flex items-center gap-2">
                  <BookOpen className="w-4 h-4" />
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
                <p className="text-zinc-600 text-[10px]">
                  Not a medical device. For screening &amp; educational purposes only.
                </p>
                <Link href="/detector">
                  <Button size="sm" className="w-full gap-1.5 mt-2" onClick={() => setLearnOpen(false)}>
                    <Activity className="w-3.5 h-3.5" />
                    Try the Detector
                  </Button>
                </Link>
              </CardContent>
            </Card>
          </div>
        </>
      )}

      {/* Hero Section */}
      <section className="relative h-screen w-full overflow-hidden">
        {/* 3D Neuron Canvas Background */}
        <div className="absolute inset-0">
          <HeroNeuron />
        </div>

        {/* Hero Text Overlay */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="text-center space-y-6 px-4 pointer-events-auto">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-cyan-500/20 bg-cyan-500/5 text-cyan-400 text-xs font-mono animate-fade-in">
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
              Early Detection Research Tool
            </div>

            <h1 className="text-5xl sm:text-7xl font-bold tracking-tight animate-slide-up">
              <span className="bg-gradient-to-b from-white to-zinc-400 bg-clip-text text-transparent">
                Neuro
              </span>
              <span className="bg-gradient-to-r from-pink-400 to-cyan-400 bg-clip-text text-transparent">
                Sketch
              </span>
            </h1>

            <p className="text-zinc-400 text-lg sm:text-xl max-w-xl mx-auto animate-slide-up leading-relaxed">
              Screen for early Parkinson&apos;s indicators using the
              Archimedes spiral test — powered by real-time motor analysis.
            </p>

            <div className="flex items-center justify-center gap-4 animate-slide-up">
              <Link href="/detector">
                <Button size="lg" className="gap-2 text-base">
                  <Activity className="w-5 h-5" />
                  Begin Screening
                </Button>
              </Link>
              <Link href="/learn-more">
                <Button
                  variant="outline"
                  size="lg"
                  className="gap-2 text-base"
                >
                  <BookOpen className="w-5 h-5" />
                  Learn More
                </Button>
              </Link>
            </div>
          </div>
        </div>

        {/* Bottom gradient fade */}
        <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-black to-transparent" />
      </section>

      {/* Features Section */}
      <section className="relative z-10 py-24 bg-black">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold tracking-tight mb-4">
              Clinical-Grade Screening,{" "}
              <span className="text-cyan-400">Instantly</span>
            </h2>
            <p className="text-zinc-500 max-w-2xl mx-auto">
              NeuroSketch brings validated neurological screening methods to
              your browser with zero data collection and real-time results.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {FEATURES.map((feature) => (
              <Card
                key={feature.title}
                className="group hover:border-cyan-500/20 transition-colors duration-300"
              >
                <CardHeader>
                  <div className="p-3 rounded-xl bg-cyan-500/10 w-fit mb-3 group-hover:bg-cyan-500/15 transition-colors">
                    <feature.icon className="w-6 h-6 text-cyan-400" />
                  </div>
                  <CardTitle className="text-xl">
                    {feature.title}
                  </CardTitle>
                  <CardDescription className="leading-relaxed">
                    {feature.description}
                  </CardDescription>
                </CardHeader>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative z-10 py-24 bg-gradient-to-t from-zinc-950 to-black border-t border-zinc-900/50">
        <div className="max-w-3xl mx-auto px-4 text-center space-y-8">
          <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">
            Ready to test?
          </h2>
          <p className="text-zinc-400 text-lg leading-relaxed">
            The spiral test takes less than 30 seconds. All processing
            happens locally — your data never leaves your device.
          </p>
          <Link href="/detector">
            <Button size="lg" className="gap-2 text-base">
              Open Detector
              <ArrowRight className="w-5 h-5" />
            </Button>
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-zinc-900 py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Brain className="w-4 h-4 text-cyan-500/50" />
            <span className="text-sm text-zinc-600">
              NeuroSketch — Educational Research Tool
            </span>
          </div>
          <p className="text-xs text-zinc-700">
            Not a medical device. For screening and educational purposes
            only.
          </p>
        </div>
      </footer>
    </div>
  );
}
