/**
 * NeuroSketch — Detector Dashboard
 *
 * Dual-test diagnostic dashboard featuring Archimedes Spiral and
 * Wave Analysis tabs. Captures (x, y, t) coordinates and packages
 * results as FormData with image blob + JSON coordinate array.
 *
 * @module app/detector/page
 */

"use client";

import React, { useRef, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";
import {
  Brain,
  Activity,
  RotateCcw,
  TrendingUp,
  Timer,
  Target,
  AlertTriangle,
  CheckCircle2,
  Info,
  Upload,
  ExternalLink,
  Loader2,
  Send,
} from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import SpiralTest, { type SpiralTestRef, type SpiralAnalysis } from "@/components/tests/spiral-test";
import WaveTest, { type WaveCanvasRef, type WaveAnalysis } from "@/components/tests/wave-test";

/* ─── Risk Classification ─────────────────────────────────────────────── */

function classifyRisk(score: number): {
  level: "low" | "moderate" | "high";
  label: string;
  color: string;
  bgColor: string;
  description: string;
} {
  if (score < 30) {
    return {
      level: "low",
      label: "Low Risk",
      color: "text-emerald-400",
      bgColor: "bg-emerald-500/10 border-emerald-500/20",
      description:
        "Drawing patterns appear within normal range. Consistent speed and minimal deviation detected.",
    };
  }
  if (score < 65) {
    return {
      level: "moderate",
      label: "Moderate Indicators",
      color: "text-amber-400",
      bgColor: "bg-amber-500/10 border-amber-500/20",
      description:
        "Some irregularity detected in velocity or trajectory. Consider retesting in a controlled setting.",
    };
  }
  return {
    level: "high",
    label: "Elevated Indicators",
    color: "text-red-400",
    bgColor: "bg-red-500/10 border-red-500/20",
    description:
      "Significant drawing irregularity detected. This is a screening tool, not a diagnosis. Consult a neurologist.",
  };
}

/* ─── Metric Card ──────────────────────────────────────────────────────── */

function MetricCard({
  icon: Icon,
  label,
  value,
  unit,
  subtext,
}: {
  icon: React.ElementType;
  label: string;
  value: string | number;
  unit?: string;
  subtext?: string;
}) {
  return (
    <div className="flex items-start gap-3 p-3 rounded-lg bg-zinc-900/50 border border-zinc-800/50">
      <div className="mt-0.5 p-2 rounded-md bg-cyan-500/10">
        <Icon className="w-4 h-4 text-cyan-400" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-xs text-zinc-500 font-medium uppercase tracking-wider">
          {label}
        </p>
        <p className="text-lg font-semibold text-white mt-0.5">
          {value}
          {unit && (
            <span className="text-xs text-zinc-500 ml-1">{unit}</span>
          )}
        </p>
        {subtext && (
          <p className="text-xs text-zinc-600 mt-0.5">{subtext}</p>
        )}
      </div>
    </div>
  );
}

/* ─── Unified Analysis Type ────────────────────────────────────────────── */

interface UnifiedAnalysis {
  tremorScore: number;
  meanDeviation: number;
  stdDeviation: number;
  meanSpeed: number;
  speedVariance: number;
  drawingTime: number;
  pointCount: number;
  testType: "spiral" | "wave";
}

/* ─── Main Page ────────────────────────────────────────────────────────── */

export default function DetectorPage() {
  const spiralRef = useRef<SpiralTestRef>(null);
  const waveRef = useRef<WaveCanvasRef>(null);

  const [analysis, setAnalysis] = useState<UnifiedAnalysis | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [activeTest, setActiveTest] = useState<string>("spiral");
  const [showInfoDialog, setShowInfoDialog] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [prediction, setPrediction] = useState<{ label: string; confidence: number } | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSpiralUpdate = useCallback((a: SpiralAnalysis) => {
    setAnalysis({
      tremorScore: a.tremorScore,
      meanDeviation: a.meanDeviation,
      stdDeviation: a.stdDeviation,
      meanSpeed: a.meanSpeed,
      speedVariance: a.speedVariance,
      drawingTime: a.drawingTime,
      pointCount: a.pointCount,
      testType: "spiral",
    });
  }, []);

  const handleWaveUpdate = useCallback((a: WaveAnalysis) => {
    setAnalysis({
      tremorScore: a.tremorScore,
      meanDeviation: a.meanDeviation,
      stdDeviation: a.stdDeviation,
      meanSpeed: a.meanSpeed,
      speedVariance: a.speedVariance,
      drawingTime: a.drawingTime,
      pointCount: a.pointCount,
      testType: "wave",
    });
  }, []);

  const handleReset = useCallback(() => {
    if (activeTest === "spiral") {
      spiralRef.current?.clear();
    } else {
      waveRef.current?.clear();
    }
    setAnalysis(null);
    setPrediction(null);
  }, [activeTest]);

  const handlePredict = useCallback(async () => {
    setIsPredicting(true);
    setPrediction(null);
    try {
      let fd: FormData | null = null;

      if (uploadedFile) {
        fd = new FormData();
        fd.append("image", uploadedFile, uploadedFile.name);
        fd.append("test_type", activeTest === "wave" ? "wave" : "spiral");
      } else if (activeTest === "spiral" && spiralRef.current) {
        fd = await spiralRef.current.getFormData();
      } else if (activeTest === "wave" && waveRef.current) {
        const blob = await waveRef.current.getImageBlob();
        const points = waveRef.current.getPoints();
        if (blob && points.length > 0) {
          fd = new FormData();
          fd.append("image", blob, "wave.png");
          fd.append("coordinates", JSON.stringify(points));
          fd.append("test_type", "wave");
        }
      }

      if (!fd) {
        console.warn("[NeuroSketch] No drawing data to send.");
        return;
      }

      const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: fd,
      });

      if (!res.ok) throw new Error(`Backend returned ${res.status}`);
      const data = await res.json();
      setPrediction({ label: data.label, confidence: data.confidence });
    } catch (err) {
      console.error("[NeuroSketch] Prediction failed:", err);
      setPrediction({ label: "Error — backend unreachable", confidence: 0 });
    } finally {
      setIsPredicting(false);
    }
  }, [activeTest, uploadedFile]);

  const risk = analysis ? classifyRisk(analysis.tremorScore) : null;

  return (
    <div className="min-h-screen bg-black text-zinc-100">
      {/* Header */}
      <header className="border-b border-zinc-900 bg-black/80 backdrop-blur-md sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-3 hover:opacity-80 transition-opacity">
            <div className="p-2 rounded-lg bg-cyan-500/10">
              <Brain className="w-5 h-5 text-cyan-400" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight bg-gradient-to-r from-pink-400 to-cyan-400 bg-clip-text text-transparent">
                NeuroSketch
              </h1>
              <p className="text-xs text-zinc-500">
                Motor Function Analysis
              </p>
            </div>
          </Link>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setShowInfoDialog(true)}
            >
              <Info className="w-4 h-4" />
            </Button>
            <Button variant="outline" size="sm" asChild>
              <Link href="/">Home</Link>
            </Button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h2 className="text-2xl font-bold tracking-tight mb-2">
            Diagnostic Tests
          </h2>
          <p className="text-zinc-400 max-w-2xl">
            Draw on the canvas to capture motor function data. Coordinates
            are recorded at high resolution for analysis.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Canvas Column */}
          <motion.div
            className="lg:col-span-2 space-y-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: "easeOut" }}
          >
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <Tabs value={activeTest} onValueChange={(v) => { setActiveTest(v); setAnalysis(null); setPrediction(null); setUploadedFile(null); }}>
                    <TabsList>
                      <TabsTrigger value="spiral" className="gap-1.5">
                        <Target className="w-3.5 h-3.5" />
                        Archimedes Spiral
                      </TabsTrigger>
                      <TabsTrigger value="wave" className="gap-1.5">
                        <Activity className="w-3.5 h-3.5" />
                        Wave Analysis
                      </TabsTrigger>
                    </TabsList>
                  </Tabs>

                  <div className="flex items-center gap-2">
                    {isDrawing && (
                      <span className="inline-flex items-center gap-1 text-xs bg-cyan-500/10 text-cyan-400 px-2 py-0.5 rounded-full">
                        <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
                        Recording
                      </span>
                    )}
                    {activeTest !== "upload" && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => { handleReset(); setUploadedFile(null); }}
                        className="gap-1.5"
                      >
                        <RotateCcw className="w-3.5 h-3.5" />
                        Reset
                      </Button>
                    )}
                  </div>
                </div>
                <CardDescription>
                  {activeTest === "spiral"
                    ? "Trace the Archimedes spiral guide. Stay as close to the path as possible."
                    : "Follow the sinusoidal wave guide from left to right."}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => {
                    if (e.target.files?.[0]) setUploadedFile(e.target.files[0]);
                  }}
                />

                <AnimatePresence mode="wait">
                {activeTest === "spiral" && (
                  <motion.div
                    key="spiral"
                    className="flex flex-col items-center gap-3"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    transition={{ duration: 0.3 }}
                  >
                    <SpiralTest
                      ref={spiralRef}
                      width={500}
                      height={500}
                      showGuide={true}
                      onAnalysisUpdate={handleSpiralUpdate}
                      onDrawingStart={() => setIsDrawing(true)}
                      onDrawingEnd={() => setIsDrawing(false)}
                    />
                    <div className="flex items-center gap-2">
                      <motion.div whileTap={{ scale: 0.95 }}>
                        <Button
                          variant="outline"
                          size="sm"
                          className="gap-1.5 text-xs"
                          onClick={() => fileInputRef.current?.click()}
                        >
                          <Upload className="w-3.5 h-3.5" />
                          Upload Spiral
                        </Button>
                      </motion.div>
                      {uploadedFile && (
                        <span className="text-xs text-cyan-400 truncate max-w-[160px]">
                          {uploadedFile.name}
                        </span>
                      )}
                    </div>
                  </motion.div>
                )}
                {activeTest === "wave" && (
                  <motion.div
                    key="wave"
                    className="flex flex-col items-center gap-3"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    transition={{ duration: 0.3 }}
                  >
                    <WaveTest
                      ref={waveRef}
                      width={500}
                      height={300}
                      onAnalysisUpdate={handleWaveUpdate}
                      onDrawingStart={() => setIsDrawing(true)}
                      onDrawingEnd={() => setIsDrawing(false)}
                    />
                    <div className="flex items-center gap-2">
                      <motion.div whileTap={{ scale: 0.95 }}>
                        <Button
                          variant="outline"
                          size="sm"
                          className="gap-1.5 text-xs"
                          onClick={() => fileInputRef.current?.click()}
                        >
                          <Upload className="w-3.5 h-3.5" />
                          Upload Wave
                        </Button>
                      </motion.div>
                      {uploadedFile && (
                        <span className="text-xs text-cyan-400 truncate max-w-[160px]">
                          {uploadedFile.name}
                        </span>
                      )}
                    </div>
                  </motion.div>
                )}
                </AnimatePresence>

                {/* Predict Button */}
                <AnimatePresence>
                {(analysis || uploadedFile) && (
                  <motion.div
                    className="mt-4 flex justify-end"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 10 }}
                    transition={{ duration: 0.25 }}
                  >
                    <motion.div whileTap={{ scale: 0.93 }}>
                      <Button
                        size="sm"
                        className="gap-1.5 text-xs"
                        onClick={handlePredict}
                        disabled={isPredicting}
                      >
                        {isPredicting ? (
                          <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        ) : (
                          <Send className="w-3.5 h-3.5" />
                        )}
                        {isPredicting ? "Predicting…" : "Predict"}
                      </Button>
                    </motion.div>
                  </motion.div>
                )}
                </AnimatePresence>

                {/* ML Prediction Result */}
                {prediction && (
                  <div className={`mt-3 p-3 rounded-lg border text-sm ${
                    prediction.confidence > 0
                      ? "border-cyan-500/20 bg-cyan-500/5"
                      : "border-red-500/20 bg-red-500/5"
                  }`}>
                    <p className="font-medium text-zinc-200">
                      Result: <span className="text-cyan-400">{prediction.label}</span>
                    </p>
                    {prediction.confidence > 0 && (
                      <p className="text-xs text-zinc-500 mt-1">
                        Confidence: {(prediction.confidence * 100).toFixed(1)}%
                      </p>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>

          {/* Analysis Column */}
          <motion.div
            className="space-y-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.15, ease: "easeOut" }}
          >
            {/* Risk Assessment */}
            <motion.div whileHover={{ y: -4, transition: { duration: 0.2 } }}>
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center gap-2">
                  <Activity className="w-4 h-4 text-cyan-400" />
                  <CardTitle className="text-base">Risk Assessment</CardTitle>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {analysis && risk ? (
                  <>
                    <div
                      className={`p-4 rounded-lg border ${risk.bgColor}`}
                    >
                      <div className="flex items-center gap-2 mb-2">
                        {risk.level === "low" ? (
                          <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                        ) : risk.level === "moderate" ? (
                          <AlertTriangle className="w-5 h-5 text-amber-400" />
                        ) : (
                          <AlertTriangle className="w-5 h-5 text-red-400" />
                        )}
                        <span className={`font-semibold text-sm ${risk.color}`}>
                          {risk.label}
                        </span>
                      </div>
                      <p className="text-xs text-zinc-400 leading-relaxed">
                        {risk.description}
                      </p>
                    </div>

                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-zinc-400">Tremor Score</span>
                        <span className={`font-mono font-bold ${risk.color}`}>
                          {analysis.tremorScore}/100
                        </span>
                      </div>
                      <Progress value={analysis.tremorScore} />
                    </div>

                    <div className="text-[10px] uppercase tracking-widest text-zinc-600 text-center">
                      {analysis.testType === "spiral"
                        ? "Archimedes Spiral Test"
                        : "Wave Analysis Test"}
                    </div>
                  </>
                ) : (
                  <div className="text-center py-8 text-zinc-600">
                    <Activity className="w-8 h-8 mx-auto mb-2 opacity-30" />
                    <p className="text-sm">
                      Draw to begin analysis
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
            </motion.div>

            {/* Detailed Metrics */}
            <motion.div whileHover={{ y: -4, transition: { duration: 0.2 } }}>
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-cyan-400" />
                  <CardTitle className="text-base">Metrics</CardTitle>
                </div>
              </CardHeader>
              <CardContent className="space-y-2">
                {analysis ? (
                  <>
                    <MetricCard
                      icon={Target}
                      label={analysis.testType === "spiral" ? "Radial Deviation" : "Wave Deviation"}
                      value={analysis.meanDeviation}
                      unit="px"
                      subtext={`σ = ${analysis.stdDeviation}`}
                    />
                    <MetricCard
                      icon={TrendingUp}
                      label="Drawing Speed"
                      value={analysis.meanSpeed}
                      unit="px/ms"
                      subtext={`variance = ${analysis.speedVariance}`}
                    />
                    <MetricCard
                      icon={Timer}
                      label="Duration"
                      value={(analysis.drawingTime / 1000).toFixed(1)}
                      unit="sec"
                      subtext={`${analysis.pointCount} sample points`}
                    />
                  </>
                ) : (
                  <div className="text-center py-6 text-zinc-600">
                    <p className="text-xs">No data yet</p>
                  </div>
                )}
              </CardContent>
            </Card>
            </motion.div>

            {/* Disclaimer */}
            <div className="text-xs text-zinc-600 px-2 leading-relaxed">
              <strong className="text-zinc-500">Disclaimer:</strong> This
              tool is for educational and screening purposes only. It is
              not a medical diagnostic device. Always consult a qualified
              healthcare professional for clinical evaluation.
            </div>
          </motion.div>
        </div>
      </main>

      {/* Learn More Dialog — Research Sources */}
      <Dialog open={showInfoDialog} onOpenChange={setShowInfoDialog}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>Learn More — Research &amp; Sources</DialogTitle>
            <DialogDescription className="space-y-4 pt-2">
              <p>
                NeuroSketch implements validated clinical screening methods
                for Parkinson&apos;s disease. Below are key references and
                external resources for deeper reading.
              </p>

              <div className="space-y-3">
                <h4 className="text-zinc-200 font-semibold text-sm">Research Papers</h4>
                <ul className="space-y-2 text-zinc-400 text-xs">

                  <li className="flex items-start gap-2">
                    <ExternalLink className="w-3.5 h-3.5 mt-0.5 shrink-0 text-cyan-500" />
                    <a href="https://pubmed.ncbi.nlm.nih.gov/25904359/" target="_blank" rel="noopener noreferrer" className="hover:text-cyan-400 transition-colors">
                      Zham P et al. <em>Efficacy of guided and unguided spiral drawing in assessing motor symptoms of Parkinson&apos;s disease.</em> IEEE EMBC. 2017.
                    </a>
                  </li>
                  <li className="flex items-start gap-2">
                    <ExternalLink className="w-3.5 h-3.5 mt-0.5 shrink-0 text-cyan-500" />
                    <a href="https://pubmed.ncbi.nlm.nih.gov/30131219/" target="_blank" rel="noopener noreferrer" className="hover:text-cyan-400 transition-colors">
                      Pereira CR et al. <em>A new computer vision–based approach to aid the diagnosis of Parkinson&apos;s disease.</em> Comp Methods &amp; Programs in Biomed. 2016.
                    </a>
                  </li>
                </ul>
              </div>

              <div className="space-y-3">
                <h4 className="text-zinc-200 font-semibold text-sm">External Resources</h4>
                <ul className="space-y-2 text-zinc-400 text-xs">
                  <li className="flex items-start gap-2">
                    <ExternalLink className="w-3.5 h-3.5 mt-0.5 shrink-0 text-cyan-500" />
                    <a href="https://www.parkinson.org/understanding-parkinsons" target="_blank" rel="noopener noreferrer" className="hover:text-cyan-400 transition-colors">
                      Parkinson&apos;s Foundation — Understanding Parkinson&apos;s Disease
                    </a>
                  </li>
                  <li className="flex items-start gap-2">
                    <ExternalLink className="w-3.5 h-3.5 mt-0.5 shrink-0 text-cyan-500" />
                    <a href="https://www.ninds.nih.gov/health-information/disorders/parkinsons-disease" target="_blank" rel="noopener noreferrer" className="hover:text-cyan-400 transition-colors">
                      NIH NINDS — Parkinson&apos;s Disease Information Page
                    </a>
                  </li>
                  <li className="flex items-start gap-2">
                    <ExternalLink className="w-3.5 h-3.5 mt-0.5 shrink-0 text-cyan-500" />
                    <a href="https://www.michaeljfox.org/parkinsons-101" target="_blank" rel="noopener noreferrer" className="hover:text-cyan-400 transition-colors">
                      Michael J. Fox Foundation — Parkinson&apos;s 101
                    </a>
                  </li>
                </ul>
              </div>

              <p className="text-zinc-600 text-[11px] border-t border-zinc-800 pt-3">
                This tool is for educational and screening purposes only.
                It is not a medical diagnostic device.
              </p>
            </DialogDescription>
          </DialogHeader>
        </DialogContent>
      </Dialog>
    </div>
  );
}
