/**
 * PredictionResults — ML Prediction Display Component
 *
 * Professional medical-grade UI for displaying Parkinson's disease
 * screening predictions from the dual CNN ensemble model.
 * Includes probability, risk tier, model breakdown, confidence,
 * agreement indicators, and medical disclaimer.
 *
 * @module components/results/PredictionResults
 */

"use client";

import React from "react";
import { motion } from "framer-motion";
import { Brain, Activity, CheckCircle2, AlertCircle, TrendingUp } from "lucide-react";
import { Progress } from "@/components/ui/progress";

/* ─── Types ───────────────────────────────────────────────────────────── */

export interface PredictionData {
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
}

interface PredictionResultsProps {
  prediction: PredictionData;
}

/* ─── Animation Variants ──────────────────────────────────────────────── */

const containerVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.4,
      staggerChildren: 0.1,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, x: -20 },
  visible: {
    opacity: 1,
    x: 0,
    transition: { duration: 0.3 },
  },
};

const pulseVariants = {
  initial: { scale: 1 },
  animate: {
    scale: [1, 1.05, 1],
    transition: {
      duration: 2,
      repeat: Infinity,
      repeatType: "reverse" as const,
    },
  },
};

/* ─── Component ───────────────────────────────────────────────────────── */

export default function PredictionResults({ prediction }: PredictionResultsProps) {
  const {
    probability,
    riskTier,
    riskColor,
    spiralPercent,
    wavePercent,
    confidence,
    confidenceLabel,
    disclaimer,
    modelAgreement,
    unanimous,
  } = prediction;

  // Determine confidence color
  const confidenceColor =
    confidence < 0.3
      ? "#EF4444" // red
      : confidence < 0.6
      ? "#F59E0B" // amber
      : "#10B981"; // green

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="space-y-4"
    >
      {/* Main Prediction Card */}
      <motion.div
        variants={itemVariants}
        className="p-5 rounded-xl border border-zinc-800 bg-gradient-to-br from-zinc-900 via-zinc-900/95 to-zinc-900/90"
      >
        {/* Header */}
        <div className="flex items-center gap-2 mb-4">
          <Brain className="w-5 h-5 text-cyan-400" />
          <h3 className="text-sm font-medium text-zinc-300 uppercase tracking-wide">
            ML Prediction
          </h3>
        </div>

        {/* Main Probability Display */}
        <div className="mb-4">
          <div className="flex items-baseline justify-between mb-2">
            <span className="text-xs font-medium text-zinc-500 uppercase tracking-wider">
              PD Probability
            </span>
            <motion.span
              variants={pulseVariants}
              initial="initial"
              animate="animate"
              className="text-4xl font-bold tabular-nums"
              style={{ color: riskColor }}
            >
              {probability.toFixed(1)}%
            </motion.span>
          </div>

          {/* Progress Bar */}
          <div className="mb-3">
            <Progress value={probability} className="h-2" />
          </div>

          {/* Risk Tier Badge */}
          <motion.div
            variants={itemVariants}
            className="px-4 py-2 rounded-lg text-sm font-semibold text-center"
            style={{
              backgroundColor: `${riskColor}15`,
              color: riskColor,
              border: `1.5px solid ${riskColor}30`,
            }}
          >
            {riskTier}
          </motion.div>
        </div>

        {/* Divider */}
        <div className="border-t border-zinc-800 my-4" />

        {/* Model Breakdown */}
        <motion.div variants={itemVariants} className="space-y-3">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="w-4 h-4 text-cyan-400" />
            <span className="text-xs font-medium text-zinc-400 uppercase tracking-wide">
              Model Breakdown
            </span>
          </div>

          {/* Spiral CNN */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-xs">
              <span className="text-zinc-500">Spiral CNN</span>
              <span className="text-zinc-300 font-mono font-semibold">
                {spiralPercent.toFixed(1)}%
              </span>
            </div>
            <Progress value={spiralPercent} className="h-1.5" />
          </div>

          {/* Wave CNN */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-xs">
              <span className="text-zinc-500">Wave CNN</span>
              <span className="text-zinc-300 font-mono font-semibold">
                {wavePercent.toFixed(1)}%
              </span>
            </div>
            <Progress value={wavePercent} className="h-1.5" />
          </div>
        </motion.div>

        {/* Divider */}
        <div className="border-t border-zinc-800 my-4" />

        {/* Confidence & Agreement */}
        <motion.div variants={itemVariants} className="space-y-3">
          {/* Confidence */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-3.5 h-3.5 text-zinc-500" />
              <span className="text-xs text-zinc-500">Confidence</span>
            </div>
            <div className="flex items-center gap-2">
              <span
                className="text-xs font-semibold"
                style={{ color: confidenceColor }}
              >
                {confidenceLabel}
              </span>
              <span className="text-xs text-zinc-600 font-mono">
                ({(confidence * 100).toFixed(1)}%)
              </span>
            </div>
          </div>

          {/* Model Agreement (if available) */}
          {typeof modelAgreement !== "undefined" && (
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {unanimous ? (
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                ) : (
                  <AlertCircle className="w-3.5 h-3.5 text-amber-400" />
                )}
                <span className="text-xs text-zinc-500">Model Agreement</span>
              </div>
              <span
                className={`text-xs font-semibold ${
                  unanimous ? "text-emerald-400" : "text-amber-400"
                }`}
              >
                {unanimous ? "Unanimous" : "Partial"}
              </span>
            </div>
          )}
        </motion.div>
      </motion.div>

      {/* Disclaimer */}
      <motion.div
        variants={itemVariants}
        className="p-4 rounded-lg bg-amber-500/5 border border-amber-500/20"
      >
        <div className="flex gap-3">
          <AlertCircle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
          <div className="space-y-1">
            <p className="text-xs font-medium text-amber-400">
              Medical Disclaimer
            </p>
            <p className="text-xs text-zinc-400 leading-relaxed">
              {disclaimer}
            </p>
          </div>
        </div>
      </motion.div>

      {/* Model Info Footer */}
      <motion.div
        variants={itemVariants}
        className="text-center text-[10px] text-zinc-600 uppercase tracking-widest"
      >
        Dual MobileNetV2 CNN Ensemble • Weighted 40/60
      </motion.div>
    </motion.div>
  );
}
