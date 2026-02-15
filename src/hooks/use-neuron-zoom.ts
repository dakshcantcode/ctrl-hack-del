/**
 * useNeuronZoom — Invisible scroll-interceptor zoom controller
 *
 * Manages zoom state for the 3D neuron canvas. Zoom activates
 * when the neuron center is in the viewport, intercepting scroll
 * seamlessly. No buttons needed.
 *
 * @module hooks/use-neuron-zoom
 */

"use client";

import { useState, useCallback, useRef } from "react";
import { clamp } from "@/lib/utils";

interface UseNeuronZoomOptions {
  warpThreshold?: number;
}

interface UseNeuronZoomReturn {
  zoom: number;
  isWarping: boolean;
  hasReachedNucleus: boolean;
  /** Attach to canvas wheel event */
  handleWheel: (e: WheelEvent) => void;
  /** Reset everything after fact viewing */
  resetWarp: () => void;
  /** Selected stem index (0-4), or null */
  activeStem: number | null;
  setActiveStem: (i: number | null) => void;
}

export function useNeuronZoom(
  opts: UseNeuronZoomOptions = {}
): UseNeuronZoomReturn {
  const { warpThreshold = 0.85 } = opts;

  const [zoom, setZoom] = useState(0);
  const [isWarping, setIsWarping] = useState(false);
  const [hasReachedNucleus, setHasReachedNucleus] = useState(false);
  const [activeStem, setActiveStem] = useState<number | null>(null);

  const warpTriggered = useRef(false);

  const handleWheel = useCallback(
    (e: WheelEvent) => {
      if (isWarping) return;

      e.preventDefault();
      const delta = e.deltaY * -0.003;

      // Post-nucleus: allow scroll-back to zoom out (snap-back)
      if (hasReachedNucleus) {
        if (delta < 0) {
          // Scrolling back out — reset warp state and begin zooming out
          setHasReachedNucleus(false);
          warpTriggered.current = false;
          setZoom(0.7); // start zoom-out from near-threshold (below 0.85 to avoid re-warp)
        }
        return;
      }

      setZoom((prev) => {
        const next = clamp(prev + delta, 0, 1);

        if (next >= warpThreshold && !warpTriggered.current) {
          warpTriggered.current = true;
          setIsWarping(true);
          setTimeout(() => {
            setIsWarping(false);
            setHasReachedNucleus(true);
          }, 1800);
        }

        return next;
      });
    },
    [isWarping, hasReachedNucleus, warpThreshold]
  );

  const resetWarp = useCallback(() => {
    setHasReachedNucleus(false);
    setIsWarping(false);
    setActiveStem(null);
    warpTriggered.current = false;
    setZoom(0);
  }, []);

  return {
    zoom,
    isWarping,
    hasReachedNucleus,
    handleWheel,
    resetWarp,
    activeStem,
    setActiveStem,
  };
}
