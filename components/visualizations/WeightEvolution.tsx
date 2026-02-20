"use client";

import { useState, useMemo, useRef, useEffect } from "react";
import type { WeightSnapshots } from "@/lib/training/trainingData";
import { activationColorScale, parseColor } from "@/lib/utils/colorScales";

interface WeightEvolutionProps {
  snapshots: WeightSnapshots | null;
}

const EPOCHS = [0, 1, 2, 5, 10, 15, 20, 25, 30, 40, 49];
const LAYERS = ["conv1", "conv2", "conv3", "dense1"];

/** Recursively flatten any nested array into a flat number array */
function flattenWeights(data: unknown): number[] {
  if (typeof data === "number") return [data];
  if (Array.isArray(data)) return data.flatMap(flattenWeights);
  return [];
}

/** Compute stats from raw weight data or extract from summary object */
function getStats(layerData: unknown): {
  mean: number;
  std: number;
  min: number;
  max: number;
  shape: number[];
  flatWeights: number[] | null;
} | null {
  if (!layerData) return null;

  // Summary format: { mean, std, min, max, shape }
  if (
    typeof layerData === "object" &&
    !Array.isArray(layerData) &&
    layerData !== null &&
    "mean" in layerData
  ) {
    const s = layerData as { mean: number; std: number; min: number; max: number; shape: number[] };
    return { ...s, flatWeights: null };
  }

  // Raw weight tensor (nested arrays)
  const flat = flattenWeights(layerData);
  if (flat.length === 0) return null;

  const mean = flat.reduce((a, b) => a + b, 0) / flat.length;
  const variance = flat.reduce((a, b) => a + (b - mean) ** 2, 0) / flat.length;
  return {
    mean,
    std: Math.sqrt(variance),
    min: Math.min(...flat),
    max: Math.max(...flat),
    shape: [],
    flatWeights: flat,
  };
}

/** Mini weight distribution histogram */
function WeightHistogram({ weights, maxAbs }: { weights: number[]; maxAbs: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const bins = 60;
  const width = 400;
  const height = 100;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, width, height);

    // Build histogram
    const counts = new Array(bins).fill(0);
    const range = maxAbs * 2;
    for (const w of weights) {
      const idx = Math.floor(((w + maxAbs) / range) * bins);
      const clamped = Math.max(0, Math.min(bins - 1, idx));
      counts[clamped]++;
    }
    const maxCount = Math.max(...counts, 1);

    const barW = width / bins;
    for (let i = 0; i < bins; i++) {
      const barH = (counts[i] / maxCount) * (height - 10);
      const t = i / bins;
      // Blue for negative, white for zero, red for positive
      const r = t > 0.5 ? Math.round(200 * (t - 0.5) * 2) : 60;
      const g = 60;
      const b = t < 0.5 ? Math.round(200 * (1 - t * 2)) : 60;
      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.fillRect(i * barW, height - barH, barW - 1, barH);
    }

    // Zero line
    const zeroX = (maxAbs / range) * width;
    ctx.strokeStyle = "rgba(255,255,255,0.3)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(zeroX, 0);
    ctx.lineTo(zeroX, height);
    ctx.stroke();
    ctx.setLineDash([]);
  }, [weights, maxAbs]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="w-full max-w-md rounded-lg border border-border"
      style={{ height: 100 }}
    />
  );
}

/** Mini kernel grid visualizer for conv layers */
function KernelGrid({ weights, shape }: { weights: number[]; shape: number[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // shape: [out_channels, in_channels, kH, kW] for conv
  // For conv1: [32, 1, 3, 3] → show 32 3x3 kernels
  const numFilters = Math.min(32, weights.length / 9); // assume 3x3
  const kSize = 3;
  const cellPx = 12;
  const gap = 2;
  const cols = 8;
  const rows = Math.ceil(numFilters / cols);
  const totalW = cols * (kSize * cellPx + gap);
  const totalH = rows * (kSize * cellPx + gap);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || numFilters === 0) return;
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, totalW, totalH);

    const absMax = Math.max(...weights.map(Math.abs), 0.001);

    for (let f = 0; f < numFilters; f++) {
      const fRow = Math.floor(f / cols);
      const fCol = f % cols;
      const ox = fCol * (kSize * cellPx + gap);
      const oy = fRow * (kSize * cellPx + gap);

      for (let r = 0; r < kSize; r++) {
        for (let c = 0; c < kSize; c++) {
          const idx = f * kSize * kSize + r * kSize + c;
          const val = weights[idx] ?? 0;
          const norm = (val / absMax + 1) / 2; // 0 = most negative, 1 = most positive
          const red = Math.round(norm * 220);
          const blue = Math.round((1 - norm) * 220);
          ctx.fillStyle = `rgb(${red}, 40, ${blue})`;
          ctx.fillRect(ox + c * cellPx, oy + r * cellPx, cellPx - 1, cellPx - 1);
        }
      }
    }
  }, [weights, numFilters, totalW, totalH]);

  if (numFilters === 0) return null;

  return (
    <div className="flex flex-col items-center gap-2">
      <span className="text-xs text-foreground/40">Kernel weights (diverging: blue=negative, red=positive)</span>
      <canvas
        ref={canvasRef}
        width={totalW}
        height={totalH}
        className="rounded-lg border border-border"
        style={{ imageRendering: "pixelated" }}
      />
    </div>
  );
}

export function WeightEvolution({ snapshots }: WeightEvolutionProps) {
  const [selectedEpoch, setSelectedEpoch] = useState(0);
  const [selectedLayer, setSelectedLayer] = useState("conv1");

  const availableEpochs = useMemo(
    () => (snapshots ? EPOCHS.filter((e) => snapshots[String(e)] !== undefined) : []),
    [snapshots]
  );

  const stats = useMemo(() => {
    if (!snapshots) return null;
    const epochData = snapshots[String(selectedEpoch)];
    return getStats(epochData?.[selectedLayer]);
  }, [snapshots, selectedEpoch, selectedLayer]);

  // Compute global max abs across all epochs for consistent histogram scale
  const globalMaxAbs = useMemo(() => {
    if (!snapshots) return 1;
    let maxAbs = 0;
    for (const epoch of availableEpochs) {
      const s = getStats(snapshots[String(epoch)]?.[selectedLayer]);
      if (s) maxAbs = Math.max(maxAbs, Math.abs(s.min), Math.abs(s.max));
    }
    return maxAbs || 1;
  }, [snapshots, selectedLayer, availableEpochs]);

  if (!snapshots) {
    return (
      <div className="flex h-48 items-center justify-center rounded-xl border border-border bg-surface">
        <p className="text-foreground/30">Weight snapshots not loaded</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center gap-6">
      {/* Layer selector */}
      <div className="flex gap-2">
        {LAYERS.map((layer) => (
          <button
            key={layer}
            onClick={() => setSelectedLayer(layer)}
            className={`rounded-lg border px-3 py-1.5 text-sm font-medium transition-colors ${
              selectedLayer === layer
                ? "border-accent-secondary/50 bg-accent-secondary/15 text-accent-secondary"
                : "border-border text-foreground/40 hover:text-foreground/60"
            }`}
          >
            {layer}
          </button>
        ))}
      </div>

      {/* Epoch scrubber */}
      <div className="flex w-full max-w-lg flex-col items-center gap-2">
        <div className="flex w-full justify-between text-xs text-foreground/40">
          <span>Epoch 0 (random)</span>
          <span>Epoch 49 (trained)</span>
        </div>
        <div className="flex w-full gap-1">
          {availableEpochs.map((epoch) => (
            <button
              key={epoch}
              onClick={() => setSelectedEpoch(epoch)}
              className={`flex-1 rounded py-3 text-xs font-mono transition-all ${
                selectedEpoch === epoch
                  ? "bg-accent-secondary text-background font-bold"
                  : "bg-surface text-foreground/40 hover:bg-surface-elevated"
              }`}
            >
              {epoch}
            </button>
          ))}
        </div>
      </div>

      {stats ? (
        <>
          {/* Stats cards */}
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            {[
              { label: "Mean", value: stats.mean },
              { label: "Std Dev", value: stats.std },
              { label: "Min", value: stats.min },
              { label: "Max", value: stats.max },
            ].map((stat) => (
              <div
                key={stat.label}
                className="flex flex-col items-center rounded-lg border border-border bg-surface px-4 py-3"
              >
                <span className="text-xs text-foreground/40">{stat.label}</span>
                <span className="font-mono text-lg font-bold text-accent-secondary">
                  {stat.value.toFixed(4)}
                </span>
              </div>
            ))}
          </div>

          {/* Weight distribution histogram */}
          {stats.flatWeights && (
            <div className="flex w-full flex-col items-center gap-2">
              <span className="text-xs text-foreground/40">Weight distribution</span>
              <WeightHistogram weights={stats.flatWeights} maxAbs={globalMaxAbs} />
              <div className="flex w-full max-w-md justify-between text-xs text-foreground/30">
                <span>-{globalMaxAbs.toFixed(2)}</span>
                <span>0</span>
                <span>+{globalMaxAbs.toFixed(2)}</span>
              </div>
            </div>
          )}

          {/* Kernel visualizer for conv1 (small enough to show) */}
          {stats.flatWeights && selectedLayer === "conv1" && (
            <KernelGrid weights={stats.flatWeights} shape={stats.shape} />
          )}
        </>
      ) : (
        <p className="text-sm text-foreground/30">
          No snapshot available for {selectedLayer} at epoch {selectedEpoch}
        </p>
      )}

      <p className="max-w-md text-center text-sm text-foreground/40">
        {selectedEpoch === 0
          ? "At epoch 0, weights are random — the network has no idea what it's looking at."
          : selectedEpoch < 10
          ? "Early epochs — the weights are starting to organize into meaningful patterns."
          : "The weights have converged into structured filters that detect specific features."}
      </p>
    </div>
  );
}
