/**
 * GradCAMOverlay — Gradient-weighted Class Activation Mapping Visualization
 *
 * Displays explainability heatmaps showing which regions of a drawing
 * the CNN focused on to make its prediction. The heatmap uses a jet
 * colormap where red indicates high activation (important regions) and
 * blue indicates low activation.
 *
 * @module components/canvas/GradCAMOverlay
 */

"use client";

import React, { useState } from "react";
import { Info, Eye, EyeOff } from "lucide-react";
import { Button } from "@/components/ui/button";

/* ─── Types ───────────────────────────────────────────────────────────── */

interface GradCAMOverlayProps {
  /** Base64-encoded PNG data URI from backend */
  gradcamImage: string | null;
  /** Original canvas image for side-by-side comparison */
  originalImage?: string | null;
  /** Label for the test (e.g., "Spiral" or "Wave") */
  testLabel: string;
  /** Optional width constraint */
  width?: number;
  /** Optional height constraint */
  height?: number;
}

/* ─── Component ───────────────────────────────────────────────────────── */

export default function GradCAMOverlay({
  gradcamImage,
  originalImage,
  testLabel,
  width = 300,
  height = 300,
}: GradCAMOverlayProps) {
  const [showOverlay, setShowOverlay] = useState(true);
  const [showInfo, setShowInfo] = useState(false);

  if (!gradcamImage) {
    return (
      <div
        className="flex items-center justify-center rounded-lg border border-zinc-800 bg-zinc-900/50"
        style={{ width, height }}
      >
        <div className="text-center text-zinc-600 p-4">
          <Eye className="w-8 h-8 mx-auto mb-2 opacity-30" />
          <p className="text-xs">No Grad-CAM available</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h4 className="text-sm font-medium text-zinc-300">
            {testLabel} Explainability
          </h4>
          <button
            onClick={() => setShowInfo(!showInfo)}
            className="text-zinc-500 hover:text-cyan-400 transition-colors"
            aria-label="Show Grad-CAM info"
          >
            <Info className="w-3.5 h-3.5" />
          </button>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setShowOverlay(!showOverlay)}
          className="h-7 px-2 text-xs gap-1.5"
        >
          {showOverlay ? (
            <>
              <Eye className="w-3 h-3" />
              Heatmap
            </>
          ) : (
            <>
              <EyeOff className="w-3 h-3" />
              Original
            </>
          )}
        </Button>
      </div>

      {/* Info Tooltip */}
      {showInfo && (
        <div className="p-3 rounded-lg bg-cyan-500/5 border border-cyan-500/20 text-xs text-zinc-400 leading-relaxed">
          <p className="font-medium text-cyan-400 mb-1">
            What is Grad-CAM?
          </p>
          <p>
            Gradient-weighted Class Activation Mapping highlights which regions
            of your drawing the neural network focused on to make its prediction.
          </p>
          <ul className="mt-2 space-y-1 list-disc list-inside text-[11px]">
            <li>
              <span className="text-red-400">Red areas</span>: High importance
              (model focused here)
            </li>
            <li>
              <span className="text-blue-400">Blue areas</span>: Low importance
              (model ignored)
            </li>
            <li>Model examines tremor, line consistency, and spatial patterns</li>
          </ul>
        </div>
      )}

      {/* Image Display */}
      <div className="relative rounded-lg overflow-hidden border border-zinc-800 bg-black">
        <img
          src={showOverlay ? gradcamImage : (originalImage || gradcamImage)}
          alt={`${testLabel} Grad-CAM heatmap`}
          className="w-full h-auto"
          style={{ maxWidth: width, maxHeight: height }}
        />

        {/* Color Scale Legend */}
        {showOverlay && (
          <div className="absolute bottom-2 right-2 flex items-center gap-1.5 bg-black/80 backdrop-blur-sm px-2 py-1 rounded text-[10px] text-zinc-400 border border-zinc-700/50">
            <span>Low</span>
            <div
              className="h-2 w-16 rounded-sm"
              style={{
                background: "linear-gradient(to right, #0000FF, #00FFFF, #00FF00, #FFFF00, #FF0000)"
              }}
            />
            <span>High</span>
          </div>
        )}
      </div>

      {/* Model Focus Summary */}
      <div className="text-[10px] text-zinc-600 text-center">
        {showOverlay
          ? `Showing ${testLabel} CNN activation heatmap`
          : "Showing original drawing"}
      </div>
    </div>
  );
}
