"use client";

import { useState, useMemo, useRef, useEffect } from "react";
import { BarChart, Bar, XAxis, YAxis, Cell, ReferenceLine } from "recharts";
import {
  ChartContainer,
  ChartTooltip,
  type ChartConfig,
} from "@/components/ui/chart";
import type { WeightSnapshots } from "@/lib/training/trainingData";

interface WeightEvolutionProps {
  snapshots: WeightSnapshots | null;
}

const EPOCHS = [0, 1, 2, 5, 10, 15, 20, 25, 30, 40, 49];
const LAYERS = ["conv1", "conv2", "conv3", "dense1"];
const HIST_BINS = 50;

const chartConfig = {
  count: { label: "Count", color: "#8b5cf6" },
} satisfies ChartConfig;

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

  if (
    typeof layerData === "object" &&
    !Array.isArray(layerData) &&
    layerData !== null &&
    "mean" in layerData
  ) {
    const s = layerData as {
      mean: number;
      std: number;
      min: number;
      max: number;
      shape: number[];
    };
    return { ...s, flatWeights: null };
  }

  const flat = flattenWeights(layerData);
  if (flat.length === 0) return null;

  const mean = flat.reduce((a, b) => a + b, 0) / flat.length;
  const variance =
    flat.reduce((a, b) => a + (b - mean) ** 2, 0) / flat.length;
  return {
    mean,
    std: Math.sqrt(variance),
    min: Math.min(...flat),
    max: Math.max(...flat),
    shape: [],
    flatWeights: flat,
  };
}

/** Build histogram data for recharts */
function buildHistogramData(
  weights: number[],
  maxAbs: number
): { binCenter: number; count: number; isNegative: boolean }[] {
  const range = maxAbs * 2;
  const counts = new Array(HIST_BINS).fill(0);

  for (const w of weights) {
    const idx = Math.floor(((w + maxAbs) / range) * HIST_BINS);
    const clamped = Math.max(0, Math.min(HIST_BINS - 1, idx));
    counts[clamped]++;
  }

  return counts.map((count, i) => {
    const binCenter = -maxAbs + ((i + 0.5) / HIST_BINS) * range;
    return { binCenter, count, isNegative: binCenter < 0 };
  });
}

/** Mini kernel grid visualizer for conv layers */
function KernelGrid({
  weights,
}: {
  weights: number[];
  shape: number[];
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const numFilters = Math.min(32, weights.length / 9);
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
          const norm = (val / absMax + 1) / 2;
          const red = Math.round(norm * 220);
          const blue = Math.round((1 - norm) * 220);
          ctx.fillStyle = `rgb(${red}, 40, ${blue})`;
          ctx.fillRect(
            ox + c * cellPx,
            oy + r * cellPx,
            cellPx - 1,
            cellPx - 1
          );
        }
      }
    }
  }, [weights, numFilters, totalW, totalH]);

  if (numFilters === 0) return null;

  return (
    <div className="flex flex-col items-center gap-2">
      <span className="text-xs text-foreground/40">
        Kernel weights (diverging: blue=negative, red=positive)
      </span>
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
    () =>
      snapshots
        ? EPOCHS.filter((e) => snapshots[String(e)] !== undefined)
        : [],
    [snapshots]
  );

  const stats = useMemo(() => {
    if (!snapshots) return null;
    const epochData = snapshots[String(selectedEpoch)];
    return getStats(epochData?.[selectedLayer]);
  }, [snapshots, selectedEpoch, selectedLayer]);

  const globalMaxAbs = useMemo(() => {
    if (!snapshots) return 1;
    let maxAbs = 0;
    for (const epoch of availableEpochs) {
      const s = getStats(snapshots[String(epoch)]?.[selectedLayer]);
      if (s) maxAbs = Math.max(maxAbs, Math.abs(s.min), Math.abs(s.max));
    }
    return maxAbs || 1;
  }, [snapshots, selectedLayer, availableEpochs]);

  const histData = useMemo(() => {
    if (!stats?.flatWeights) return null;
    return buildHistogramData(stats.flatWeights, globalMaxAbs);
  }, [stats, globalMaxAbs]);

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
              className={`flex-1 rounded py-2 text-[10px] font-mono transition-all sm:py-3 sm:text-xs ${
                selectedEpoch === epoch
                  ? "bg-accent-secondary font-bold text-background"
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
                <span className="text-xs text-foreground/40">
                  {stat.label}
                </span>
                <span className="font-mono text-lg font-bold text-accent-secondary">
                  {stat.value.toFixed(4)}
                </span>
              </div>
            ))}
          </div>

          {/* Weight distribution histogram (recharts) */}
          {histData && (
            <div className="flex w-full flex-col items-center gap-2">
              <span className="text-xs text-foreground/40">
                Weight distribution
              </span>
              <ChartContainer
                config={chartConfig}
                className="h-[120px] w-full max-w-md rounded-lg border border-border bg-surface sm:h-[140px]"
              >
                <BarChart
                  data={histData}
                  margin={{ top: 8, right: 8, bottom: 4, left: 8 }}
                  barCategoryGap={0}
                  barGap={0}
                >
                  <XAxis
                    dataKey="binCenter"
                    type="number"
                    domain={[-globalMaxAbs, globalMaxAbs]}
                    tick={{ fill: "rgba(232,232,237,0.3)", fontSize: 9 }}
                    tickLine={false}
                    axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
                    tickCount={5}
                    tickFormatter={(v: number) => v.toFixed(2)}
                  />
                  <YAxis hide />
                  <ReferenceLine
                    x={0}
                    stroke="rgba(255,255,255,0.2)"
                    strokeDasharray="4 4"
                  />
                  <ChartTooltip
                    content={({ active, payload }) => {
                      if (!active || !payload?.length) return null;
                      const d = payload[0].payload as {
                        binCenter: number;
                        count: number;
                      };
                      return (
                        <div className="rounded border border-border bg-surface px-2 py-1 text-xs shadow-lg">
                          <span className="text-foreground/50">
                            {d.binCenter.toFixed(4)}
                          </span>
                          <span className="ml-2 font-mono text-accent-secondary">
                            {d.count}
                          </span>
                        </div>
                      );
                    }}
                  />
                  <Bar dataKey="count" radius={[1, 1, 0, 0]}>
                    {histData.map((entry, i) => (
                      <Cell
                        key={i}
                        fill={
                          entry.isNegative
                            ? "rgba(96, 165, 250, 0.7)"
                            : "rgba(239, 68, 68, 0.7)"
                        }
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ChartContainer>
            </div>
          )}

          {/* Kernel visualizer for conv1 */}
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
