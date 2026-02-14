/**
 * NeuroSketch — Learn More Page
 *
 * Detailed research sources, clinical references, and external links
 * for hand-drawn screening of neurodegenerative diseases.
 *
 * @module app/learn-more/page
 */

"use client";

import React from "react";
import Link from "next/link";
import {
  Brain,
  Activity,
  ArrowLeft,
  ExternalLink,
  BookOpen,
  FlaskConical,
  GraduationCap,
  Stethoscope,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

/* ─── Research Papers ──────────────────────────────────────────────────── */

const PAPERS = [
  {
    authors: "Pullman SL",
    title:
      "Spiral analysis: a new technique for measuring tremor with a digitizing tablet.",
    journal: "Mov Disord. 1998",
    url: "https://pubmed.ncbi.nlm.nih.gov/9629849/",
    summary:
      "Introduced spiral analysis as a quantitative, objective measure of tremor severity using digitized Archimedes spirals. Demonstrated that spiral metrics correlate strongly with clinical tremor rating scales.",
  },
  {
    authors: "Zham P et al.",
    title:
      "Efficacy of guided and unguided spiral drawing in assessing motor symptoms of Parkinson's disease.",
    journal: "IEEE EMBC. 2017",
    url: "https://pubmed.ncbi.nlm.nih.gov/25904359/",
    summary:
      "Compared guided vs. unguided spiral drawing tasks in PD patients. Found that guided spirals (tracing a template) provide more sensitive motor impairment measures than free-form drawing.",
  },
  {
    authors: "Pereira CR et al.",
    title:
      "A new computer vision–based approach to aid the diagnosis of Parkinson's disease.",
    journal: "Comp Methods & Programs in Biomed. 2016",
    url: "https://pubmed.ncbi.nlm.nih.gov/30131219/",
    summary:
      "Proposed a computer vision pipeline that extracts features from handwriting and spiral drawings to classify PD vs. healthy controls, achieving high accuracy with CNNs on image-based representations.",
  },
  {
    authors: "Impedovo D, Pirlo G",
    title:
      "Dynamic handwriting analysis for the assessment of neurodegenerative diseases: a pattern recognition perspective.",
    journal: "IEEE Reviews in Biomedical Engineering. 2019",
    url: "https://pubmed.ncbi.nlm.nih.gov/30010594/",
    summary:
      "Comprehensive review of dynamic handwriting features — pressure, velocity, jerk — and their effectiveness as digital biomarkers for Parkinson's, Alzheimer's, and other neurodegenerative conditions.",
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
  return (
    <div className="min-h-screen bg-black text-zinc-100">
      {/* Header */}
      <header className="border-b border-zinc-900 bg-black/80 backdrop-blur-md sticky top-0 z-40">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between">
          <Link
            href="/"
            className="flex items-center gap-2.5 hover:opacity-80 transition-opacity"
          >
            <div className="p-1.5 rounded-lg bg-cyan-500/10">
              <Brain className="w-5 h-5 text-cyan-400" />
            </div>
            <span className="font-bold text-lg tracking-tight bg-gradient-to-r from-pink-400 to-cyan-400 bg-clip-text text-transparent">
              NeuroSketch
            </span>
          </Link>
          <Link href="/">
            <Button variant="ghost" size="sm" className="gap-1.5">
              <ArrowLeft className="w-4 h-4" />
              Back
            </Button>
          </Link>
        </div>
      </header>

      {/* Hero */}
      <section className="py-16 border-b border-zinc-900/50">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 space-y-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-cyan-500/20 bg-cyan-500/5 text-cyan-400 text-xs font-mono">
            <BookOpen className="w-3.5 h-3.5" />
            Research &amp; Sources
          </div>
          <h1 className="text-3xl sm:text-4xl font-bold tracking-tight">
            The Science Behind{" "}
            <span className="text-cyan-400">NeuroSketch</span>
          </h1>
          <p className="text-zinc-400 text-lg leading-relaxed max-w-2xl">
            NeuroSketch implements validated clinical screening methods for
            Parkinson&apos;s disease using hand-drawn motor tasks. Below are
            key references and resources for deeper reading.
          </p>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-16 border-b border-zinc-900/50">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 space-y-8">
          <div className="flex items-center gap-2">
            <Stethoscope className="w-5 h-5 text-cyan-400" />
            <h2 className="text-2xl font-bold tracking-tight">
              How Screening Works
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">
                  Archimedes Spiral Test
                </CardTitle>
              </CardHeader>
              <CardContent className="text-sm text-zinc-400 leading-relaxed space-y-2">
                <p>
                  The patient traces a pre-drawn Archimedes spiral as
                  accurately as possible. We capture (x, y, t) coordinates
                  at high resolution and compute:
                </p>
                <ul className="list-disc pl-4 space-y-1 text-xs text-zinc-500">
                  <li>Radial deviation from the ideal spiral path</li>
                  <li>Speed variance and drawing consistency</li>
                  <li>High-frequency tremor components (4–6 Hz band)</li>
                  <li>Total drawing time and sample density</li>
                </ul>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">
                  Sinusoidal Wave Test
                </CardTitle>
              </CardHeader>
              <CardContent className="text-sm text-zinc-400 leading-relaxed space-y-2">
                <p>
                  The patient follows a sinusoidal guide wave from left to
                  right. This captures lateral motor control and measures:
                </p>
                <ul className="list-disc pl-4 space-y-1 text-xs text-zinc-500">
                  <li>Amplitude deviation from the reference wave</li>
                  <li>Frequency stability and consistency</li>
                  <li>Velocity profile across the trace</li>
                  <li>Jerk (rate of acceleration change) — a bradykinesia marker</li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Research Papers */}
      <section className="py-16 border-b border-zinc-900/50">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 space-y-8">
          <div className="flex items-center gap-2">
            <FlaskConical className="w-5 h-5 text-cyan-400" />
            <h2 className="text-2xl font-bold tracking-tight">
              Clinical Research
            </h2>
          </div>

          <div className="space-y-4">
            {PAPERS.map((paper) => (
              <Card key={paper.url} className="group hover:border-cyan-500/20 transition-colors">
                <CardContent className="pt-5 space-y-2">
                  <a
                    href={paper.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-start gap-2 group-hover:text-cyan-400 transition-colors"
                  >
                    <ExternalLink className="w-4 h-4 mt-0.5 shrink-0 text-cyan-500" />
                    <div>
                      <p className="text-sm font-medium text-zinc-200 group-hover:text-cyan-300 transition-colors">
                        {paper.authors}.{" "}
                        <em>{paper.title}</em>
                      </p>
                      <p className="text-xs text-zinc-600 mt-0.5">
                        {paper.journal}
                      </p>
                    </div>
                  </a>
                  <p className="text-xs text-zinc-500 leading-relaxed pl-6">
                    {paper.summary}
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* External Resources */}
      <section className="py-16 border-b border-zinc-900/50">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 space-y-8">
          <div className="flex items-center gap-2">
            <GraduationCap className="w-5 h-5 text-cyan-400" />
            <h2 className="text-2xl font-bold tracking-tight">
              External Resources
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {RESOURCES.map((res) => (
              <a
                key={res.url}
                href={res.url}
                target="_blank"
                rel="noopener noreferrer"
                className="group"
              >
                <Card className="h-full hover:border-cyan-500/20 transition-colors">
                  <CardContent className="pt-5 space-y-2">
                    <div className="flex items-start gap-2">
                      <ExternalLink className="w-4 h-4 mt-0.5 shrink-0 text-cyan-500" />
                      <p className="text-sm font-medium text-zinc-200 group-hover:text-cyan-300 transition-colors">
                        {res.name}
                      </p>
                    </div>
                    <p className="text-xs text-zinc-500 leading-relaxed pl-6">
                      {res.description}
                    </p>
                  </CardContent>
                </Card>
              </a>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-16">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center space-y-6">
          <h2 className="text-2xl font-bold tracking-tight">
            Try the Screening Tool
          </h2>
          <p className="text-zinc-400 max-w-lg mx-auto">
            Both tests take under a minute. All processing happens locally
            in your browser — no data leaves your device.
          </p>
          <Link href="/detector">
            <Button size="lg" className="gap-2 text-base">
              <Activity className="w-5 h-5" />
              Open Detector
            </Button>
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-zinc-900 py-8">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Brain className="w-4 h-4 text-cyan-500/50" />
            <span className="text-sm text-zinc-600">
              NeuroSketch — Educational Research Tool
            </span>
          </div>
          <p className="text-xs text-zinc-700">
            Not a medical device. For screening and educational purposes only.
          </p>
        </div>
      </footer>
    </div>
  );
}
