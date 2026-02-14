/**
 * NeuroSketch — Spiral Image Upload Component
 *
 * Drag-and-drop / click-to-upload component for spiral images.
 * Sends the image to the backend ML pipeline for TensorFlow-based
 * Parkinson's screening analysis.
 *
 * @module components/canvas/SpiralUpload
 */

"use client";

import React, { useRef, useState, useCallback } from "react";
import {
  Upload,
  ImageIcon,
  X,
  Loader2,
  CheckCircle2,
  AlertTriangle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

/* ─── Types ───────────────────────────────────────────────────────────── */

export interface UploadResult {
  confidence: number; // 0-1
  prediction: "healthy" | "parkinson";
  details?: string;
}

interface SpiralUploadProps {
  onUploadComplete?: (result: UploadResult) => void;
  onUploadStart?: () => void;
  apiEndpoint?: string;
}

/* ─── Component ───────────────────────────────────────────────────────── */

export default function SpiralUpload({
  onUploadComplete,
  onUploadStart,
  apiEndpoint = "/api/predict",
}: SpiralUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<UploadResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  /* ─── File Handling ──────────────────────────────────────── */

  const processFile = useCallback((f: File) => {
    // Validate file type
    if (!f.type.startsWith("image/")) {
      setError("Please upload an image file (PNG, JPG, etc.)");
      return;
    }

    // Validate size (max 10MB)
    if (f.size > 10 * 1024 * 1024) {
      setError("File size must be under 10MB");
      return;
    }

    setError(null);
    setResult(null);
    setFile(f);

    // Generate preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target?.result as string);
    };
    reader.readAsDataURL(f);
  }, []);

  /* ─── Drag & Drop ────────────────────────────────────────── */

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);

      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile) processFile(droppedFile);
    },
    [processFile]
  );

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const selected = e.target.files?.[0];
      if (selected) processFile(selected);
    },
    [processFile]
  );

  /* ─── Upload to API ──────────────────────────────────────── */

  const handleUpload = useCallback(async () => {
    if (!file) return;

    setUploading(true);
    setError(null);
    onUploadStart?.();

    try {
      const formData = new FormData();
      formData.append("image", file);

      const res = await fetch(apiEndpoint, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Server responded with ${res.status}`);
      }

      const data: UploadResult = await res.json();
      setResult(data);
      onUploadComplete?.(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? `Upload failed: ${err.message}. Make sure the backend API is running.`
          : "Upload failed. Please try again."
      );
    } finally {
      setUploading(false);
    }
  }, [file, apiEndpoint, onUploadStart, onUploadComplete]);

  /* ─── Clear ──────────────────────────────────────────────── */

  const handleClear = useCallback(() => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  }, []);

  /* ─── Render ─────────────────────────────────────────────── */

  return (
    <div className="space-y-4">
      {/* Drop Zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !preview && inputRef.current?.click()}
        className={cn(
          "relative rounded-xl border-2 border-dashed transition-all duration-300 cursor-pointer",
          "flex flex-col items-center justify-center min-h-[280px]",
          isDragging
            ? "border-cyan-400 bg-cyan-500/10 scale-[1.02]"
            : preview
            ? "border-zinc-700 bg-zinc-900/30"
            : "border-zinc-800 bg-zinc-950/50 hover:border-zinc-700 hover:bg-zinc-900/30"
        )}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          className="hidden"
        />

        {preview ? (
          /* ─── Preview State ────────────────── */
          <div className="relative w-full h-full p-4">
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleClear();
              }}
              className="absolute top-2 right-2 z-10 p-1 rounded-full bg-zinc-900/80 border border-zinc-700 text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
            <div className="flex justify-center">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={preview}
                alt="Spiral preview"
                className="max-h-[240px] rounded-lg object-contain"
              />
            </div>
          </div>
        ) : (
          /* ─── Empty State ──────────────────── */
          <div className="flex flex-col items-center gap-3 p-6 text-center">
            <div className="p-3 rounded-xl bg-cyan-500/10">
              <Upload className="w-6 h-6 text-cyan-400" />
            </div>
            <div>
              <p className="text-sm font-medium text-zinc-300">
                Upload a spiral drawing
              </p>
              <p className="text-xs text-zinc-500 mt-1">
                Drag & drop or click to browse
              </p>
            </div>
            <div className="flex items-center gap-2 text-xs text-zinc-600">
              <ImageIcon className="w-3.5 h-3.5" />
              PNG, JPG, WEBP — Max 10MB
            </div>
          </div>
        )}
      </div>

      {/* Action Buttons */}
      {preview && !result && (
        <div className="flex items-center gap-2">
          <Button
            onClick={handleUpload}
            disabled={uploading}
            className="flex-1 gap-2"
          >
            {uploading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Upload className="w-4 h-4" />
                Send for Analysis
              </>
            )}
          </Button>
          <Button variant="ghost" size="sm" onClick={handleClear}>
            Clear
          </Button>
        </div>
      )}

      {/* Result Card */}
      {result && (
        <div
          className={cn(
            "p-4 rounded-lg border",
            result.prediction === "healthy"
              ? "bg-emerald-500/10 border-emerald-500/20"
              : "bg-red-500/10 border-red-500/20"
          )}
        >
          <div className="flex items-center gap-2 mb-2">
            {result.prediction === "healthy" ? (
              <CheckCircle2 className="w-5 h-5 text-emerald-400" />
            ) : (
              <AlertTriangle className="w-5 h-5 text-red-400" />
            )}
            <span
              className={cn(
                "font-semibold text-sm",
                result.prediction === "healthy"
                  ? "text-emerald-400"
                  : "text-red-400"
              )}
            >
              {result.prediction === "healthy"
                ? "Normal Pattern Detected"
                : "Indicators Detected"}
            </span>
          </div>
          <div className="flex items-center justify-between text-xs text-zinc-400 mb-2">
            <span>Model Confidence</span>
            <span className="font-mono">
              {(result.confidence * 100).toFixed(1)}%
            </span>
          </div>
          <div className="w-full h-2 rounded-full bg-zinc-800 overflow-hidden">
            <div
              className={cn(
                "h-full rounded-full transition-all duration-500",
                result.prediction === "healthy"
                  ? "bg-emerald-400"
                  : "bg-red-400"
              )}
              style={{ width: `${result.confidence * 100}%` }}
            />
          </div>
          {result.details && (
            <p className="text-xs text-zinc-500 mt-2">{result.details}</p>
          )}
          <Button
            variant="ghost"
            size="sm"
            className="mt-3 w-full"
            onClick={handleClear}
          >
            Upload Another
          </Button>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-xs text-red-400">
          {error}
        </div>
      )}
    </div>
  );
}
