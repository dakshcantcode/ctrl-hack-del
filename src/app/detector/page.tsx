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
import GradCAMOverlay from "@/components/canvas/GradCAMOverlay";
import PredictionResults from "@/components/results/PredictionResults";
import { blobToBase64, fileToBase64, validateImageFile } from "@/lib/utils";

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
  const [prediction, setPrediction] = useState<{
    probability: number;
    riskTier: string;
    riskColor: string;
    spiralPercent: number;
    wavePercent: number;
    confidence: number;
    confidenceLabel: string;
    spiralGradcam: string | null;
    waveGradcam: string | null;
    disclaimer: string;
    modelAgreement?: number;
    unanimous?: boolean;
  } | null>(null);
  const [spiralUploadedFile, setSpiralUploadedFile] = useState<File | null>(null);
  const [waveUploadedFile, setWaveUploadedFile] = useState<File | null>(null);
  const [spiralPreview, setSpiralPreview] = useState<string | null>(null);
  const [wavePreview, setWavePreview] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [backendHealth, setBackendHealth] = useState<"checking" | "healthy" | "unhealthy">("checking");
  const [validationError, setValidationError] = useState<string | null>(null);
  const spiralFileInputRef = useRef<HTMLInputElement>(null);
  const waveFileInputRef = useRef<HTMLInputElement>(null);

  // Health check on mount
  React.useEffect(() => {
    const checkHealth = async () => {
      try {
        const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
        const res = await fetch(`${API_URL}/health`, {
          method: "GET",
          signal: AbortSignal.timeout(5000) // 5 second timeout
        });
        if (res.ok) {
          setBackendHealth("healthy");
        } else {
          setBackendHealth("unhealthy");
        }
      } catch (err) {
        console.warn("[NeuroSketch] Backend health check failed:", err);
        setBackendHealth("unhealthy");
      }
    };
    checkHealth();
  }, []);

  // Validation helper
  const validateDrawing = useCallback((testType: "spiral" | "wave"): { valid: boolean; error?: string } => {
    const MIN_POINTS = 10;
    const MIN_DURATION_MS = 1000; // 1 second

    if (testType === "spiral") {
      if (!spiralRef.current) return { valid: false, error: "Canvas not ready" };

      // Check uploaded file first
      if (spiralUploadedFile) return { valid: true };

      const analysis = spiralRef.current.getAnalysis?.();
      if (!analysis) return { valid: false, error: "No drawing detected. Please draw on the spiral canvas." };

      if (analysis.pointCount < MIN_POINTS) {
        return { valid: false, error: `Please draw more. Need at least ${MIN_POINTS} points (currently ${analysis.pointCount}).` };
      }

      if (analysis.drawingTime < MIN_DURATION_MS) {
        const seconds = (MIN_DURATION_MS / 1000).toFixed(1);
        return { valid: false, error: `Please draw slower. Minimum ${seconds} seconds required.` };
      }
    } else {
      if (!waveRef.current) return { valid: false, error: "Canvas not ready" };

      // Check uploaded file first
      if (waveUploadedFile) return { valid: true };

      const analysis = waveRef.current.getAnalysis?.();
      if (!analysis) return { valid: false, error: "No drawing detected. Please draw on the wave canvas." };

      if (analysis.pointCount < MIN_POINTS) {
        return { valid: false, error: `Please draw more. Need at least ${MIN_POINTS} points (currently ${analysis.pointCount}).` };
      }

      if (analysis.drawingTime < MIN_DURATION_MS) {
        const seconds = (MIN_DURATION_MS / 1000).toFixed(1);
        return { valid: false, error: `Please draw slower. Minimum ${seconds} seconds required.` };
      }
    }

    return { valid: true };
  }, [spiralUploadedFile, waveUploadedFile]);

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
      setSpiralUploadedFile(null);
      setSpiralPreview(null);
    } else {
      waveRef.current?.clear();
      setWaveUploadedFile(null);
      setWavePreview(null);
    }
    setAnalysis(null);
    setPrediction(null);
    setUploadError(null);
  }, [activeTest]);

  const handleSpiralFileUpload = useCallback(async (file: File) => {
    setUploadError(null);

    // Validate file
    const validation = validateImageFile(file);
    if (!validation.isValid) {
      setUploadError(validation.error || "Invalid file");
      return;
    }

    setSpiralUploadedFile(file);

    // Generate preview
    try {
      const preview = await fileToBase64(file);
      setSpiralPreview(preview);
    } catch {
      setUploadError("Failed to read file");
    }
  }, []);

  const handleWaveFileUpload = useCallback(async (file: File) => {
    setUploadError(null);

    // Validate file
    const validation = validateImageFile(file);
    if (!validation.isValid) {
      setUploadError(validation.error || "Invalid file");
      return;
    }

    setWaveUploadedFile(file);

    // Generate preview
    try {
      const preview = await fileToBase64(file);
      setWavePreview(preview);
    } catch {
      setUploadError("Failed to read file");
    }
  }, []);

  const handlePredict = useCallback(async () => {
    setIsPredicting(true);
    setPrediction(null);
    setUploadError(null);
    setValidationError(null);

    try {
      // Validate both drawings/uploads
      const spiralValidation = validateDrawing("spiral");
      if (!spiralValidation.valid) {
        setValidationError(spiralValidation.error || "Invalid spiral drawing");
        return;
      }

      const waveValidation = validateDrawing("wave");
      if (!waveValidation.valid) {
        setValidationError(waveValidation.error || "Invalid wave drawing");
        return;
      }

      // Collect BOTH spiral and wave images (from canvas OR upload)
      let spiralB64: string;
      let waveB64: string;
      let inputMode: "drawn" | "uploaded" | "mixed" = "drawn";

      // Determine spiral source (uploaded file takes priority)
      if (spiralUploadedFile) {
        spiralB64 = await fileToBase64(spiralUploadedFile);
        inputMode = "uploaded";
      } else {
        const spiralBlob = await spiralRef.current?.getImageBlob();
        if (!spiralBlob) {
          setUploadError("Please draw or upload a spiral image");
          return;
        }
        spiralB64 = await blobToBase64(spiralBlob);
      }

      // Determine wave source (uploaded file takes priority)
      if (waveUploadedFile) {
        waveB64 = await fileToBase64(waveUploadedFile);
        if (inputMode === "drawn") inputMode = "uploaded";
        else if (inputMode !== "uploaded") inputMode = "mixed";
      } else {
        const waveBlob = await waveRef.current?.getImageBlob();
        if (!waveBlob) {
          setUploadError("Please draw or upload a wave image");
          return;
        }
        waveB64 = await blobToBase64(waveBlob);
        if (inputMode === "uploaded") inputMode = "mixed";
      }

      // Send JSON request with both images (with timeout)
      const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

      try {
        const res = await fetch(`${API_URL}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            spiral_image: spiralB64,
            wave_image: waveB64,
            input_mode: inputMode
          }),
          signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!res.ok) {
          const errorData = await res.json().catch(() => null);
          const errorType = errorData?.detail?.error_type || "unknown";
          const errorMsg = errorData?.detail?.error || `Server error (${res.status})`;

          // Specific error handling
          if (errorType === "models_loading") {
            throw new Error("Models are still loading. Please wait a moment and try again.");
          } else if (errorType === "invalid_image") {
            throw new Error("Invalid image format. Please ensure both images are valid PNG/JPG files.");
          } else {
            throw new Error(errorMsg);
          }
        }

        // Parse backend's actual response
        const data = await res.json();
        setPrediction({
          probability: data.pd_probability_percent,
          riskTier: data.risk_tier,
          riskColor: data.risk_color,
          spiralPercent: data.spiral_cnn_percent,
          wavePercent: data.wave_cnn_percent,
          confidence: data.confidence_score,
          confidenceLabel: data.confidence_label,
          spiralGradcam: data.spiral_gradcam_base64,
          waveGradcam: data.wave_gradcam_base64,
          disclaimer: data.disclaimer,
          modelAgreement: data.model_agreement,
          unanimous: data.unanimous
        });
      } catch (fetchErr) {
        clearTimeout(timeoutId);
        if (fetchErr instanceof Error && fetchErr.name === "AbortError") {
          throw new Error("Request timed out. The server may be processing. Please try again.");
        }
        throw fetchErr;
      }
    } catch (err) {
      console.error("[NeuroSketch] Prediction failed:", err);
      const errorMsg = err instanceof Error ? err.message : "Unknown error";
      setUploadError(errorMsg);
    } finally {
      setIsPredicting(false);
    }
  }, [spiralUploadedFile, waveUploadedFile, validateDrawing]);

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
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8" role="main">
        {/* Backend Health Status */}
        {backendHealth === "unhealthy" && (
          <div className="mb-6 p-4 rounded-lg bg-red-500/10 border border-red-500/20" role="alert">
            <div className="flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-red-400">Backend Unavailable</p>
                <p className="text-xs text-zinc-400 mt-1">
                  Unable to connect to the ML backend. Please ensure the server is running on port 8000.
                </p>
              </div>
            </div>
          </div>
        )}

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
          <div className="lg:col-span-2 space-y-4">
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <Tabs value={activeTest} onValueChange={(v) => { setActiveTest(v); setAnalysis(null); setPrediction(null); }}>
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
                        onClick={handleReset}
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
                {/* Hidden File Inputs */}
                <input
                  ref={spiralFileInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => {
                    if (e.target.files?.[0]) handleSpiralFileUpload(e.target.files[0]);
                  }}
                />
                <input
                  ref={waveFileInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => {
                    if (e.target.files?.[0]) handleWaveFileUpload(e.target.files[0]);
                  }}
                />

                {/* Spiral Canvas - always mounted, hidden when inactive */}
                <div className={`flex flex-col items-center gap-3 ${activeTest === "spiral" ? "" : "hidden"}`}
                     role="region"
                     aria-label="Spiral drawing test">
                  <SpiralTest
                    ref={spiralRef}
                    width={500}
                    height={500}
                    showGuide={true}
                    onAnalysisUpdate={handleSpiralUpdate}
                    onDrawingStart={() => {
                      setIsDrawing(true);
                                            setValidationError(null);
                    }}
                    onDrawingEnd={() => {
                      setIsDrawing(false);
                    }}
                  />
                  <div className="space-y-2">
                    <Button
                      variant="outline"
                      size="sm"
                      className="gap-1.5 text-xs w-full"
                      onClick={() => spiralFileInputRef.current?.click()}
                    >
                      <Upload className="w-3.5 h-3.5" />
                      {spiralUploadedFile ? "Change File" : "Upload Spiral Image"}
                    </Button>

                    {/* Spiral Upload Preview */}
                    {spiralUploadedFile && spiralPreview && (
                      <div className="relative p-2 rounded-lg border border-cyan-500/20 bg-cyan-500/5">
                        <button
                          onClick={() => {
                            setSpiralUploadedFile(null);
                            setSpiralPreview(null);
                            if (spiralFileInputRef.current) spiralFileInputRef.current.value = "";
                          }}
                          className="absolute -top-2 -right-2 p-1 rounded-full bg-zinc-900 border border-zinc-700 text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors z-10"
                        >
                          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                          </svg>
                        </button>
                        <div className="flex items-center gap-2">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={spiralPreview}
                            alt="Spiral preview"
                            className="w-16 h-16 rounded object-contain bg-black"
                          />
                          <div className="flex-1 min-w-0">
                            <p className="text-xs font-medium text-zinc-300 truncate">
                              {spiralUploadedFile.name}
                            </p>
                            <p className="text-[10px] text-zinc-500">
                              {(spiralUploadedFile.size / 1024).toFixed(1)} KB
                            </p>
                          </div>
                          <CheckCircle2 className="w-4 h-4 text-cyan-400 shrink-0" />
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Wave Canvas - always mounted, hidden when inactive */}
                <div className={`flex flex-col items-center gap-3 ${activeTest === "wave" ? "" : "hidden"}`}
                     role="region"
                     aria-label="Wave drawing test">
                  <WaveTest
                    ref={waveRef}
                    width={500}
                    height={300}
                    onAnalysisUpdate={handleWaveUpdate}
                    onDrawingStart={() => {
                      setIsDrawing(true);
                                            setValidationError(null);
                    }}
                    onDrawingEnd={() => {
                      setIsDrawing(false);
                    }}
                  />
                  <div className="space-y-2">
                    <Button
                      variant="outline"
                      size="sm"
                      className="gap-1.5 text-xs w-full"
                      onClick={() => waveFileInputRef.current?.click()}
                    >
                      <Upload className="w-3.5 h-3.5" />
                      {waveUploadedFile ? "Change File" : "Upload Wave Image"}
                    </Button>

                    {/* Wave Upload Preview */}
                    {waveUploadedFile && wavePreview && (
                      <div className="relative p-2 rounded-lg border border-cyan-500/20 bg-cyan-500/5">
                        <button
                          onClick={() => {
                            setWaveUploadedFile(null);
                            setWavePreview(null);
                            if (waveFileInputRef.current) waveFileInputRef.current.value = "";
                          }}
                          className="absolute -top-2 -right-2 p-1 rounded-full bg-zinc-900 border border-zinc-700 text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors z-10"
                        >
                          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                          </svg>
                        </button>
                        <div className="flex items-center gap-2">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={wavePreview}
                            alt="Wave preview"
                            className="w-16 h-16 rounded object-contain bg-black"
                          />
                          <div className="flex-1 min-w-0">
                            <p className="text-xs font-medium text-zinc-300 truncate">
                              {waveUploadedFile.name}
                            </p>
                            <p className="text-[10px] text-zinc-500">
                              {(waveUploadedFile.size / 1024).toFixed(1)} KB
                            </p>
                          </div>
                          <CheckCircle2 className="w-4 h-4 text-cyan-400 shrink-0" />
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Validation Error Display */}
                {validationError && (
                  <div className="mt-3 p-3 rounded-lg bg-amber-500/10 border border-amber-500/20" role="alert">
                    <div className="flex items-start gap-2">
                      <Info className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                      <div className="flex-1">
                        <p className="text-xs font-medium text-amber-400">Drawing Incomplete</p>
                        <p className="text-xs text-zinc-400 mt-1">{validationError}</p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Upload/Prediction Error Display */}
                {uploadError && (
                  <div className="mt-3 p-3 rounded-lg bg-red-500/10 border border-red-500/20" role="alert">
                    <div className="flex items-start gap-2">
                      <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                      <div className="flex-1">
                        <p className="text-xs font-medium text-red-400">Prediction Failed</p>
                        <p className="text-xs text-zinc-400 mt-1">{uploadError}</p>
                        <Button
                          variant="outline"
                          size="sm"
                          className="mt-2 h-7 text-xs"
                          onClick={handlePredict}
                          disabled={isPredicting}
                        >
                          <RotateCcw className="w-3 h-3 mr-1.5" />
                          Retry
                        </Button>
                      </div>
                    </div>
                  </div>
                )}

                {/* Predict Button - Show if ANY content available (drawn or uploaded) */}
                {(analysis || spiralUploadedFile || waveUploadedFile) && (
                  <div className="mt-4 flex justify-end">
                    <Button
                      size="sm"
                      className="gap-1.5 text-xs"
                      onClick={handlePredict}
                      disabled={isPredicting || backendHealth === "unhealthy"}
                      aria-label={isPredicting ? "Running prediction analysis" : "Run Parkinson's disease screening prediction"}
                      aria-busy={isPredicting}
                    >
                      {isPredicting ? (
                        <Loader2 className="w-3.5 h-3.5 animate-spin" aria-hidden="true" />
                      ) : (
                        <Send className="w-3.5 h-3.5" aria-hidden="true" />
                      )}
                      {isPredicting ? "Predicting…" : "Predict"}
                    </Button>
                  </div>
                )}

                {/* ML Prediction Result - Enhanced Component */}
                {prediction && (
                  <div className="mt-3">
                    <PredictionResults prediction={prediction} />
                  </div>
                )}

                {/* Grad-CAM Explainability Visualization */}
                {prediction && (prediction.spiralGradcam || prediction.waveGradcam) && (
                  <div className="mt-4 pt-4 border-t border-zinc-800">
                    <div className="mb-3 flex items-center gap-2">
                      <Brain className="w-4 h-4 text-cyan-400" />
                      <h3 className="text-sm font-medium text-zinc-300">
                        Model Explainability
                      </h3>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {prediction.spiralGradcam && (
                        <GradCAMOverlay
                          gradcamImage={prediction.spiralGradcam}
                          testLabel="Spiral"
                          width={240}
                          height={240}
                        />
                      )}
                      {prediction.waveGradcam && (
                        <GradCAMOverlay
                          gradcamImage={prediction.waveGradcam}
                          testLabel="Wave"
                          width={240}
                          height={144}
                        />
                      )}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Analysis Column */}
          <div className="space-y-4">
            {/* Risk Assessment */}
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center gap-2">
                  <Activity className="w-4 h-4 text-cyan-400" />
                  <CardTitle className="text-base">Risk Assessment</CardTitle>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {prediction ? (
                  <>
                    {/* ML-Based Risk (when prediction available) */}
                    <div
                      className="p-4 rounded-lg border"
                      style={{
                        backgroundColor: `${prediction.riskColor}10`,
                        borderColor: `${prediction.riskColor}30`
                      }}
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <Brain className="w-5 h-5" style={{ color: prediction.riskColor }} />
                        <span className="font-semibold text-sm" style={{ color: prediction.riskColor }}>
                          {prediction.riskTier}
                        </span>
                      </div>
                      <p className="text-xs text-zinc-400 leading-relaxed">
                        ML-based assessment from dual CNN analysis (Spiral + Wave)
                      </p>
                    </div>

                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-zinc-400">PD Probability</span>
                        <span className="font-mono font-bold" style={{ color: prediction.riskColor }}>
                          {prediction.probability.toFixed(1)}%
                        </span>
                      </div>
                      <Progress value={prediction.probability} />
                    </div>

                    <div className="text-[10px] uppercase tracking-widest text-zinc-600 text-center">
                      Neural Network Prediction
                    </div>
                  </>
                ) : analysis && risk ? (
                  <>
                    {/* Local Tremor Analysis (fallback when no prediction) */}
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

            {/* Detailed Metrics */}
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

            {/* Disclaimer */}
            <div className="text-xs text-zinc-600 px-2 leading-relaxed">
              <strong className="text-zinc-500">Disclaimer:</strong> This
              tool is for educational and screening purposes only. It is
              not a medical diagnostic device. Always consult a qualified
              healthcare professional for clinical evaluation.
            </div>
          </div>
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
                    <a href="https://pubmed.ncbi.nlm.nih.gov/9629849/" target="_blank" rel="noopener noreferrer" className="hover:text-cyan-400 transition-colors">
                      Pullman SL. <em>Spiral analysis: a new technique for measuring tremor with a digitizing tablet.</em> Mov Disord. 1998.
                    </a>
                  </li>
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
