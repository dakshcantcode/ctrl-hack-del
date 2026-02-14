/**
 * NeuroSketch — Learn More Page
 *
 * Educational content about Parkinson's disease, the science behind
 * the spiral test, how the ML pipeline works, and resources for
 * patients and caregivers.
 *
 * @module app/learn/page
 */

"use client";

import React from "react";
import Link from "next/link";
import {
  Brain,
  Activity,
  ArrowLeft,
  Microscope,
  Stethoscope,
  Cpu,
  HeartPulse,
  ExternalLink,
  ChevronRight,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

/* ─── Content Data ─────────────────────────────────────────────────────── */

const WHAT_IS_PD = [
  {
    title: "The Dopamine Connection",
    content:
      "Parkinson's disease (PD) is a progressive neurodegenerative disorder that primarily affects dopamine-producing neurons in a brain region called the substantia nigra. Dopamine is a neurotransmitter critical for coordinating smooth, controlled movement.",
  },
  {
    title: "Motor Symptoms",
    content:
      "The four cardinal motor signs are: tremor at rest, bradykinesia (slowness of movement), muscle rigidity, and postural instability. These emerge gradually and typically begin on one side of the body.",
  },
  {
    title: "Non-Motor Symptoms",
    content:
      "PD also causes depression, anxiety, sleep disturbances, constipation, loss of smell (anosmia), and cognitive changes. Many of these precede motor symptoms by years, creating opportunities for earlier detection.",
  },
  {
    title: "Who Is Affected?",
    content:
      "Approximately 10 million people worldwide live with PD. It is the second most common neurodegenerative disorder after Alzheimer's. Average onset is around age 60, but early-onset PD (before 50) accounts for 10-20% of cases.",
  },
];

const HOW_SPIRAL_WORKS = [
  {
    step: "01",
    title: "Draw the Spiral",
    description:
      "The patient draws an Archimedes spiral, starting from the center and spiraling outward. The drawing can be done on screen or on paper (then photographed).",
  },
  {
    step: "02",
    title: "Capture Kinematics",
    description:
      "Our system records position coordinates and timestamps at high frequency, capturing the full kinematics of the drawing motion — every pause, jitter, and deviation.",
  },
  {
    step: "03",
    title: "Compute Motor Features",
    description:
      "Algorithms calculate radial deviation from the ideal spiral, drawing speed consistency, acceleration patterns, and high-frequency oscillation (tremor) components.",
  },
  {
    step: "04",
    title: "ML Classification",
    description:
      "Extracted features are fed into a TensorFlow model trained on clinical datasets. The model outputs a probability score indicating the likelihood of PD-related motor dysfunction.",
  },
];

const ML_PIPELINE_DETAILS = [
  {
    title: "Image Preprocessing",
    content:
      "Uploaded spiral images are converted to grayscale, thresholded, and normalized. Contour extraction isolates the spiral stroke from the background for consistent feature extraction.",
  },
  {
    title: "Feature Extraction",
    content:
      "The model extracts spatial features (wavelet decomposition for tremor frequencies), geometric features (radial deviation, curvature variance), and temporal proxies (stroke thickness variation as a speed indicator).",
  },
  {
    title: "Convolutional Neural Network",
    content:
      "A CNN architecture processes the spiral image directly, learning discriminative patterns from raw pixel data. This works alongside handcrafted features for a hybrid approach with higher accuracy.",
  },
  {
    title: "Validation & Accuracy",
    content:
      "The model is validated using k-fold cross-validation on publicly available PD spiral datasets. Note: this is a research tool — clinical-grade diagnosis requires comprehensive neurological evaluation.",
  },
];

const RESOURCES = [
  {
    title: "Parkinson's Foundation",
    url: "https://www.parkinson.org",
    description: "Comprehensive information, resources, and support for patients and caregivers.",
  },
  {
    title: "Michael J. Fox Foundation",
    url: "https://www.michaeljfox.org",
    description: "Leading PD research organization funding breakthrough science.",
  },
  {
    title: "WHO Fact Sheet",
    url: "https://www.who.int/news-room/fact-sheets/detail/parkinson-disease",
    description: "World Health Organization overview of Parkinson's disease.",
  },
  {
    title: "Spiral Analysis Research",
    url: "https://pubmed.ncbi.nlm.nih.gov",
    description: "PubMed — search for \"spiral analysis Parkinson\" for published clinical studies.",
  },
];

/* ─── Page Component ──────────────────────────────────────────────────── */

export default function LearnPage() {
  return (
    <div className="min-h-screen bg-black text-zinc-100">
      {/* Header */}
      <header className="border-b border-zinc-900 bg-black/80 backdrop-blur-md sticky top-0 z-40">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-cyan-500/10">
              <Brain className="w-5 h-5 text-cyan-400" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight bg-gradient-to-r from-pink-400 to-cyan-400 bg-clip-text text-transparent">
                NeuroSketch
              </h1>
              <p className="text-xs text-zinc-500">Learn & Understand</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Link href="/">
              <Button variant="ghost" size="sm" className="gap-1.5">
                <ArrowLeft className="w-4 h-4" />
                Home
              </Button>
            </Link>
            <Link href="/detector">
              <Button size="sm" className="gap-1.5">
                <Activity className="w-4 h-4" />
                Detector
              </Button>
            </Link>
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12 space-y-20">
        {/* ─── Section 1: What is Parkinson's ─────────────── */}
        <section>
          <div className="flex items-center gap-3 mb-8">
            <div className="p-2.5 rounded-xl bg-cyan-500/10">
              <Stethoscope className="w-6 h-6 text-cyan-400" />
            </div>
            <div>
              <h2 className="text-2xl font-bold tracking-tight">
                What is Parkinson&apos;s Disease?
              </h2>
              <p className="text-sm text-zinc-500">
                Understanding the neuroscience behind PD
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {WHAT_IS_PD.map((item) => (
              <Card key={item.title}>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">{item.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-zinc-400 leading-relaxed">
                    {item.content}
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Key Stats */}
          <div className="mt-6 grid grid-cols-2 sm:grid-cols-4 gap-4">
            {[
              { value: "10M+", label: "People affected globally" },
              { value: "60%", label: "Neurons lost before symptoms" },
              { value: "< 30s", label: "Spiral test duration" },
              { value: "89%", label: "ML detection accuracy*" },
            ].map((stat) => (
              <div
                key={stat.label}
                className="text-center p-4 rounded-xl border border-zinc-800 bg-zinc-950/60"
              >
                <p className="text-2xl font-bold text-cyan-400">
                  {stat.value}
                </p>
                <p className="text-xs text-zinc-500 mt-1">{stat.label}</p>
              </div>
            ))}
          </div>
          <p className="text-xs text-zinc-700 mt-2">
            *Accuracy varies by dataset and model configuration. Not a
            clinical benchmark.
          </p>
        </section>

        {/* ─── Section 2: How the Spiral Test Works ───────── */}
        <section>
          <div className="flex items-center gap-3 mb-8">
            <div className="p-2.5 rounded-xl bg-cyan-500/10">
              <Microscope className="w-6 h-6 text-cyan-400" />
            </div>
            <div>
              <h2 className="text-2xl font-bold tracking-tight">
                How the Spiral Test Works
              </h2>
              <p className="text-sm text-zinc-500">
                From drawing to detection in four steps
              </p>
            </div>
          </div>

          <div className="space-y-4">
            {HOW_SPIRAL_WORKS.map((step, i) => (
              <div
                key={step.step}
                className="flex items-start gap-4 p-5 rounded-xl border border-zinc-800 bg-zinc-950/60 group hover:border-cyan-500/20 transition-colors"
              >
                <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-cyan-500/10 flex items-center justify-center font-mono text-sm font-bold text-cyan-400">
                  {step.step}
                </div>
                <div>
                  <h3 className="font-semibold text-white mb-1">
                    {step.title}
                  </h3>
                  <p className="text-sm text-zinc-400 leading-relaxed">
                    {step.description}
                  </p>
                </div>
                {i < HOW_SPIRAL_WORKS.length - 1 && (
                  <ChevronRight className="w-4 h-4 text-zinc-700 flex-shrink-0 mt-3 hidden md:block" />
                )}
              </div>
            ))}
          </div>
        </section>

        {/* ─── Section 3: ML Pipeline ─────────────────────── */}
        <section>
          <div className="flex items-center gap-3 mb-8">
            <div className="p-2.5 rounded-xl bg-cyan-500/10">
              <Cpu className="w-6 h-6 text-cyan-400" />
            </div>
            <div>
              <h2 className="text-2xl font-bold tracking-tight">
                The ML Pipeline
              </h2>
              <p className="text-sm text-zinc-500">
                How our TensorFlow model analyzes spirals
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {ML_PIPELINE_DETAILS.map((item) => (
              <Card key={item.title}>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                    {item.title}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-zinc-400 leading-relaxed">
                    {item.content}
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Architecture Summary */}
          <div className="mt-6 p-5 rounded-xl border border-zinc-800 bg-zinc-950/60">
            <h3 className="font-semibold text-white mb-3 flex items-center gap-2">
              <Cpu className="w-4 h-4 text-cyan-400" />
              Pipeline Architecture
            </h3>
            <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
              {[
                "Image Input",
                "Preprocessing",
                "Feature Extraction",
                "CNN Model",
                "Confidence Score",
                "Risk Output",
              ].map((stage, i) => (
                <React.Fragment key={stage}>
                  <span className="px-3 py-1.5 rounded-md bg-zinc-900 border border-zinc-800 text-zinc-300">
                    {stage}
                  </span>
                  {i < 5 && (
                    <ChevronRight className="w-3 h-3 text-cyan-500/50" />
                  )}
                </React.Fragment>
              ))}
            </div>
          </div>
        </section>

        {/* ─── Section 4: Resources ───────────────────────── */}
        <section>
          <div className="flex items-center gap-3 mb-8">
            <div className="p-2.5 rounded-xl bg-cyan-500/10">
              <HeartPulse className="w-6 h-6 text-cyan-400" />
            </div>
            <div>
              <h2 className="text-2xl font-bold tracking-tight">
                Resources & Support
              </h2>
              <p className="text-sm text-zinc-500">
                For patients, caregivers, and researchers
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {RESOURCES.map((r) => (
              <a
                key={r.title}
                href={r.url}
                target="_blank"
                rel="noopener noreferrer"
                className="group p-4 rounded-xl border border-zinc-800 bg-zinc-950/60 hover:border-cyan-500/20 transition-colors block"
              >
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-semibold text-white group-hover:text-cyan-400 transition-colors">
                    {r.title}
                  </h3>
                  <ExternalLink className="w-3.5 h-3.5 text-zinc-600 group-hover:text-cyan-500 transition-colors" />
                </div>
                <p className="text-xs text-zinc-500 leading-relaxed">
                  {r.description}
                </p>
              </a>
            ))}
          </div>
        </section>

        {/* ─── CTA ─────────────────────────────────────────── */}
        <section className="text-center py-12 border-t border-zinc-900">
          <h2 className="text-2xl font-bold tracking-tight mb-4">
            Ready to try the screening tool?
          </h2>
          <p className="text-zinc-500 mb-6 max-w-lg mx-auto">
            Draw a spiral or upload an image. All analysis runs locally in
            your browser — no data is collected.
          </p>
          <Link href="/detector">
            <Button size="lg" className="gap-2 text-base">
              <Activity className="w-5 h-5" />
              Open the Detector
            </Button>
          </Link>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-900 py-8">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
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
