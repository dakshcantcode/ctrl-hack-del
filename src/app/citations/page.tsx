/**
 * NeuroSketch — Citations & Technical Specifications
 *
 * Interactive research page with dataset info, model architecture,
 * performance metrics, tech stack, and external research links.
 *
 * @module app/citations/page
 */

"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import {
  Brain,
  Database,
  Cpu,
  BarChart3,
  Layers,
  ExternalLink,
  ArrowLeft,
  BookOpen,
  FlaskConical,
  Percent,
  Server,
  Code2,
  FileText,
  ChevronRight,
} from "lucide-react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import ScrollReveal from "@/components/ui/scroll-reveal";

/* ─── Boot Animation Variants ─────────────────────────────────────────── */

const bootContainer = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.12 },
  },
};

const bootLine = {
  hidden: { opacity: 0, filter: "blur(4px)" },
  show: { opacity: 1, filter: "blur(0px)", transition: { duration: 0.4 } },
};

/* ─── Data ─────────────────────────────────────────────────────────────── */

const performanceData = [
  {
    model: "Spiral CNN",
    accuracy: 86.7,
    auc: 0.9511,
    color: "text-pink-400",
    barColor: "bg-pink-500",
  },
  {
    model: "Wave CNN",
    accuracy: 91.3,
    auc: 0.9627,
    color: "text-cyan-400",
    barColor: "bg-cyan-500",
  },
];

const techStack = {
  backend: [
    { name: "FastAPI", desc: "High-performance Python web framework" },
    { name: "TensorFlow 2.10", desc: "Deep learning inference engine" },
    { name: "Keras", desc: "Model definition & training API" },
  ],
  frontend: [
    { name: "Next.js 14", desc: "React framework with App Router" },
    { name: "TypeScript", desc: "Type-safe JavaScript" },
    { name: "Tailwind CSS", desc: "Utility-first styling" },
    { name: "React / Canvas API", desc: "Interactive drawing capture" },
  ],
};

const researchLinks = [
  {
    title:
      "Deep Learning for Early Detection of Parkinson\u2019s Disease via Handwriting Analysis",
    journal: "PMC / NIH",
    id: "PMC11640201",
    url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC11640201/",
  },
  {
    title:
      "AI-Based Handwriting Analysis for Neurodegenerative Disease Screening",
    journal: "ScienceDirect \u2014 Ageing Research Reviews",
    id: "S1568163724002289",
    url: "https://www.sciencedirect.com/science/article/pii/S1568163724002289",
  },
];

/* ─── Page ─────────────────────────────────────────────────────────────── */

export default function CitationsPage() {
  const [booted, setBooted] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => setBooted(true), 1400);
    return () => clearTimeout(t);
  }, []);

  return (
    <div className="min-h-screen bg-black text-zinc-100 relative overflow-x-hidden">
      {/* scanline overlay */}
      <div className="pointer-events-none fixed inset-0 z-50 scanline-overlay opacity-[0.04]" />

      {/* Header */}
      <header className="border-b border-zinc-900 bg-black/80 backdrop-blur-md sticky top-0 z-40">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between">
          <Link
            href="/"
            className="flex items-center gap-3 hover:opacity-80 transition-opacity"
          >
            <div className="p-2 rounded-lg bg-cyan-500/10">
              <Brain className="w-5 h-5 text-cyan-400" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight bg-gradient-to-r from-pink-400 to-cyan-400 bg-clip-text text-transparent">
                NeuroSketch
              </h1>
              <p className="text-xs text-zinc-500">
                Citations &amp; Technical Specs
              </p>
            </div>
          </Link>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" asChild>
              <Link href="/" className="gap-1.5">
                <ArrowLeft className="w-3.5 h-3.5" />
                Home
              </Link>
            </Button>
          </div>
        </div>
      </header>

      {/* Boot animation intro */}
      <AnimatePresence>
        {!booted && (
          <motion.section
            className="fixed inset-0 z-30 flex items-center justify-center bg-black"
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5 }}
          >
            <motion.div
              variants={bootContainer}
              initial="hidden"
              animate="show"
              className="font-mono text-sm text-cyan-400 space-y-1"
            >
              {[
                "> loading citations module...",
                "> parsing research references...",
                "> indexing model architectures...",
                "> compiling performance metrics...",
                "> rendering interface ████████ OK",
              ].map((line, i) => (
                <motion.p key={i} variants={bootLine}>
                  {line}
                </motion.p>
              ))}
            </motion.div>
          </motion.section>
        )}
      </AnimatePresence>

      {/* Main content */}
      <main className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12 space-y-16">
        {/* Dataset */}
        <ScrollReveal>
          <section className="space-y-4">
            <div className="flex items-center gap-3 mb-2">
              <Database className="w-5 h-5 text-pink-400 neon-icon" />
              <h2 className="text-2xl font-bold tracking-tight font-mono">
                Dataset
              </h2>
            </div>
            <Card className="pixel-border border-pink-500/20 bg-zinc-950">
              <CardContent className="pt-6 space-y-3">
                <p className="text-zinc-300 leading-relaxed">
                  <span className="text-pink-400 font-semibold">
                    Parkinson&apos;s Drawings
                  </span>{" "}
                  — sourced from Kaggle. Contains spiral and wave drawings
                  from healthy individuals and patients diagnosed with
                  Parkinson&apos;s disease, captured via digitizing tablets
                  with high-resolution coordinate tracking.
                </p>
                <div className="flex items-center gap-2 text-xs text-zinc-500">
                  <FileText className="w-3.5 h-3.5" />
                  <span>
                    Two classes: <code className="text-pink-400">healthy</code>{" "}
                    &amp; <code className="text-pink-400">parkinson</code>
                  </span>
                </div>
                <div className="flex items-center gap-2 text-xs text-zinc-500">
                  <Layers className="w-3.5 h-3.5" />
                  <span>
                    Two test modalities: Spiral drawings &amp; Wave drawings
                  </span>
                </div>
              </CardContent>
            </Card>
          </section>
        </ScrollReveal>

        {/* Model Architecture */}
        <ScrollReveal>
          <section className="space-y-4">
            <div className="flex items-center gap-3 mb-2">
              <Cpu className="w-5 h-5 text-cyan-400 neon-icon" />
              <h2 className="text-2xl font-bold tracking-tight font-mono">
                Model Architecture
              </h2>
            </div>
            <Card className="pixel-border border-cyan-500/20 bg-zinc-950">
              <CardContent className="pt-6 space-y-4">
                <div className="space-y-2">
                  <h3 className="text-sm font-semibold text-cyan-400 uppercase tracking-wider">
                    Backbone
                  </h3>
                  <p className="text-zinc-300 leading-relaxed">
                    <span className="font-semibold text-white">
                      MobileNetV2
                    </span>{" "}
                    pre-trained on ImageNet. Feature extraction layers are
                    frozen; only the custom fully-connected classification
                    head is fine-tuned on the Parkinson&apos;s drawing
                    dataset.
                  </p>
                </div>
                <div className="space-y-2">
                  <h3 className="text-sm font-semibold text-cyan-400 uppercase tracking-wider">
                    Classification Head
                  </h3>
                  <p className="text-zinc-300 leading-relaxed text-sm">
                    Global Average Pooling &rarr; Dense(128, ReLU) &rarr;
                    Dropout(0.3) &rarr; Dense(2, Softmax)
                  </p>
                </div>
                <div className="space-y-2">
                  <h3 className="text-sm font-semibold text-cyan-400 uppercase tracking-wider">
                    Ensemble Strategy
                  </h3>
                  <p className="text-zinc-300 leading-relaxed text-sm">
                    Weighted ensemble:{" "}
                    <span className="text-pink-400 font-mono">
                      40% Spiral
                    </span>{" "}
                    /{" "}
                    <span className="text-cyan-400 font-mono">
                      60% Wave
                    </span>{" "}
                    — weights derived from validation AUC performance.
                  </p>
                </div>
              </CardContent>
            </Card>
          </section>
        </ScrollReveal>

        {/* Performance Metrics */}
        <ScrollReveal>
          <section className="space-y-4">
            <div className="flex items-center gap-3 mb-2">
              <BarChart3 className="w-5 h-5 text-emerald-400 neon-icon" />
              <h2 className="text-2xl font-bold tracking-tight font-mono">
                Performance Metrics
              </h2>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {performanceData.map((m) => (
                <Card
                  key={m.model}
                  className="pixel-border border-zinc-800 bg-zinc-950"
                >
                  <CardHeader className="pb-2">
                    <CardTitle
                      className={`text-base font-mono ${m.color}`}
                    >
                      {m.model}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {/* Accuracy bar */}
                    <div className="space-y-1">
                      <div className="flex justify-between text-xs text-zinc-400">
                        <span className="flex items-center gap-1">
                          <Percent className="w-3 h-3" />
                          Accuracy
                        </span>
                        <span className="font-mono font-bold text-white">
                          {m.accuracy}%
                        </span>
                      </div>
                      <div className="h-2 rounded-full bg-zinc-800 overflow-hidden">
                        <motion.div
                          className={`h-full rounded-full ${m.barColor}`}
                          initial={{ width: 0 }}
                          whileInView={{ width: `${m.accuracy}%` }}
                          viewport={{ once: true }}
                          transition={{
                            duration: 1,
                            ease: "easeOut",
                            delay: 0.3,
                          }}
                        />
                      </div>
                    </div>
                    {/* AUC */}
                    <div className="flex justify-between text-xs text-zinc-400">
                      <span className="flex items-center gap-1">
                        <FlaskConical className="w-3 h-3" />
                        AUC-ROC
                      </span>
                      <span className="font-mono font-bold text-white">
                        {m.auc.toFixed(4)}
                      </span>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </section>
        </ScrollReveal>

        {/* Tech Stack */}
        <ScrollReveal>
          <section className="space-y-4">
            <div className="flex items-center gap-3 mb-2">
              <Code2 className="w-5 h-5 text-amber-400 neon-icon" />
              <h2 className="text-2xl font-bold tracking-tight font-mono">
                Tech Stack
              </h2>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {/* Backend */}
              <Card className="pixel-border border-zinc-800 bg-zinc-950">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-mono text-amber-400 flex items-center gap-2">
                    <Server className="w-4 h-4" />
                    Backend
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  {techStack.backend.map((t) => (
                    <div key={t.name} className="flex items-start gap-2">
                      <ChevronRight className="w-3 h-3 mt-1 text-amber-500/60 shrink-0" />
                      <div>
                        <span className="text-sm text-white font-medium">
                          {t.name}
                        </span>
                        <p className="text-xs text-zinc-500">{t.desc}</p>
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>

              {/* Frontend */}
              <Card className="pixel-border border-zinc-800 bg-zinc-950">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-mono text-amber-400 flex items-center gap-2">
                    <Layers className="w-4 h-4" />
                    Frontend
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  {techStack.frontend.map((t) => (
                    <div key={t.name} className="flex items-start gap-2">
                      <ChevronRight className="w-3 h-3 mt-1 text-amber-500/60 shrink-0" />
                      <div>
                        <span className="text-sm text-white font-medium">
                          {t.name}
                        </span>
                        <p className="text-xs text-zinc-500">{t.desc}</p>
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </div>
          </section>
        </ScrollReveal>

        {/* Research Papers */}
        <ScrollReveal>
          <section className="space-y-4">
            <div className="flex items-center gap-3 mb-2">
              <BookOpen className="w-5 h-5 text-violet-400 neon-icon" />
              <h2 className="text-2xl font-bold tracking-tight font-mono">
                Research Papers
              </h2>
            </div>
            <div className="space-y-3">
              {researchLinks.map((paper) => (
                <a
                  key={paper.id}
                  href={paper.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block group"
                >
                  <Card className="pixel-border border-zinc-800 bg-zinc-950 transition-colors group-hover:border-violet-500/30">
                    <CardContent className="pt-5 flex items-start gap-3">
                      <ExternalLink className="w-4 h-4 text-violet-400 mt-1 shrink-0" />
                      <div className="space-y-1">
                        <p className="text-sm text-zinc-200 group-hover:text-violet-300 transition-colors font-medium leading-snug">
                          {paper.title}
                        </p>
                        <p className="text-xs text-zinc-500">
                          {paper.journal} &middot;{" "}
                          <code className="text-violet-400/70">
                            {paper.id}
                          </code>
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                </a>
              ))}
            </div>
          </section>
        </ScrollReveal>

        {/* Disclaimer */}
        <div className="text-xs text-zinc-600 text-center leading-relaxed pb-8 border-t border-zinc-800/50 pt-8">
          <strong className="text-zinc-500">Disclaimer:</strong> NeuroSketch
          is for educational and screening purposes only. It is not a medical
          diagnostic device. Always consult a qualified healthcare
          professional for clinical evaluation.
        </div>
      </main>
    </div>
  );
}
