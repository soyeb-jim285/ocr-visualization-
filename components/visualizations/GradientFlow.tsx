"use client";

import { motion, useInView } from "framer-motion";
import { useRef, useMemo, useState, useEffect } from "react";
import type { WeightSnapshots } from "@/lib/training/trainingData";
import { loadWeightSnapshots } from "@/lib/training/trainingData";

const LAYERS = ["conv1", "conv2", "conv3", "dense1"];
const SNAPSHOT_EPOCHS = [0, 1, 2, 5, 10, 15, 20, 25, 30, 40, 49];

/** Recursively flatten nested arrays */
function flatten(data: unknown): number[] {
  if (typeof data === "number") return [data];
  if (Array.isArray(data)) return data.flatMap(flatten);
  return [];
}

/** Get std dev of weight changes between epochs as a proxy for gradient magnitude */
function computeGradientProxy(
  snapshots: WeightSnapshots,
  layer: string,
  epochIdx: number
): number {
  const epochs = SNAPSHOT_EPOCHS.filter((e) => snapshots[String(e)]?.[layer] !== undefined);
  if (epochIdx <= 0 || epochIdx >= epochs.length) return 0;

  const prev = snapshots[String(epochs[epochIdx - 1])]?.[layer];
  const curr = snapshots[String(epochs[epochIdx])]?.[layer];
  if (!prev || !curr) return 0;

  // For summary format, use std difference
  if (typeof prev === "object" && !Array.isArray(prev) && "std" in prev) {
    const pStd = (prev as { std: number }).std;
    const cStd = (curr as { std: number }).std;
    const pMean = (prev as { mean: number }).mean;
    const cMean = (curr as { mean: number }).mean;
    return Math.abs(cMean - pMean) + Math.abs(cStd - pStd);
  }

  // For raw weights, compute actual delta
  const prevFlat = flatten(prev);
  const currFlat = flatten(curr);
  if (prevFlat.length !== currFlat.length || prevFlat.length === 0) return 0;

  let sumSq = 0;
  for (let i = 0; i < prevFlat.length; i++) {
    const d = currFlat[i] - prevFlat[i];
    sumSq += d * d;
  }
  return Math.sqrt(sumSq / prevFlat.length);
}

export function GradientFlow() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { amount: 0.3 });
  const [snapshots, setSnapshots] = useState<WeightSnapshots | null>(null);
  const [epochIdx, setEpochIdx] = useState(1);

  useEffect(() => {
    loadWeightSnapshots().then(setSnapshots).catch(() => {});
  }, []);

  const availableEpochs = useMemo(
    () => (snapshots ? SNAPSHOT_EPOCHS.filter((e) => snapshots[String(e)] !== undefined) : []),
    [snapshots]
  );

  const gradients = useMemo(() => {
    if (!snapshots) return null;
    const values = LAYERS.map((layer) => ({
      name: layer,
      magnitude: computeGradientProxy(snapshots, layer, epochIdx),
    }));
    const maxMag = Math.max(...values.map((v) => v.magnitude), 1e-8);
    return values.map((v) => ({
      ...v,
      normalized: v.magnitude / maxMag,
    }));
  }, [snapshots, epochIdx]);

  const colors: Record<string, string> = {
    conv1: "#06b6d4",
    conv2: "#06b6d4",
    conv3: "#8b5cf6",
    dense1: "#6366f1",
  };

  if (!snapshots) {
    return (
      <div className="flex h-48 items-center justify-center rounded-xl border border-border bg-surface">
        <p className="text-foreground/30">Loading gradient data...</p>
      </div>
    );
  }

  return (
    <div ref={ref} className="flex flex-col items-center gap-6">
      {/* Epoch selector */}
      <div className="flex w-full max-w-lg flex-col items-center gap-2">
        <span className="text-xs text-foreground/40">
          Weight change between epochs {availableEpochs[epochIdx - 1] ?? "?"} → {availableEpochs[epochIdx] ?? "?"}
        </span>
        <input
          type="range"
          min={1}
          max={availableEpochs.length - 1}
          value={epochIdx}
          onChange={(e) => setEpochIdx(parseInt(e.target.value))}
          className="w-full max-w-xs accent-accent-primary"
        />
      </div>

      {/* Gradient bars */}
      <div className="flex w-full max-w-lg flex-col gap-3">
        {(gradients ?? []).map((layer, i) => (
          <div key={layer.name} className="flex items-center gap-4">
            <span className="w-20 text-right font-mono text-sm text-foreground/50">
              {layer.name}
            </span>
            <div className="relative flex-1 overflow-hidden rounded-full" style={{ height: 32, backgroundColor: "rgba(255,255,255,0.05)" }}>
              <div
                className="h-full rounded-full transition-all duration-700 ease-out"
                style={{
                  width: `${Math.max(layer.normalized * 100, 3)}%`,
                  backgroundColor: colors[layer.name],
                }}
              />
            </div>
            <span className="w-20 font-mono text-xs text-foreground/40">
              {layer.magnitude.toExponential(2)}
            </span>
          </div>
        ))}
      </div>

      {/* Direction arrow */}
      <div className="flex items-center gap-2 text-sm text-foreground/30">
        <span>Output</span>
        <svg width="100" height="16" viewBox="0 0 100 16" fill="none">
          <path
            d="M100 8H10M10 8l8-6M10 8l8 6"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <span>Input</span>
        <span className="ml-2 text-xs">(gradient flows backwards)</span>
      </div>

      <p className="max-w-md text-center text-sm text-foreground/40">
        {epochIdx <= 2
          ? "Early training — large weight updates across all layers as the network rapidly learns basic patterns."
          : epochIdx <= 5
          ? "Learning is slowing down in earlier layers as they settle on stable feature detectors."
          : "Later epochs — weight changes become small and focused, fine-tuning rather than restructuring."}
      </p>
    </div>
  );
}
