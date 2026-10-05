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

const EPOCHS = [0, 1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 74];
const LAYERS = ["conv1", "conv2", "conv3", "dense1"];
const HIST_BINS = 50;

const chartConfig = {
  count: { label: "Count", color: "#f472b6" },
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
  const numFilters = Math.min(64, weights.length / 9);
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
          const red = Math.round(norm * 255);
          const blue = Math.round((1 - norm) * 255);
          ctx.fillStyle = `rgb(${red}, ${Math.round(70 + 40 * (1 - Math.abs(norm * 2 - 1)))}, ${blue})`;
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
    <div className="flex flex-col gap-2">
      <span className="caption">
        Conv1 kernels · blue = negative, red = positive
      </span>
      <div className="well w-fit max-w-full overflow-x-auto p-2 scrollbar-none">
        <canvas
          ref={canvasRef}
          width={totalW}
          height={totalH}
          style={{ imageRendering: "pixelated" }}
        />
      </div>
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
      <div className="viz-empty-state flex h-48 items-center justify-center">
        <p>Weight snapshots not loaded</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-5 md:flex-row md:items-end md:justify-between">
        {/* Layer selector */}
        <div className="flex flex-col gap-2">
          <span className="eyebrow">Layer</span>
          <div className="flex flex-wrap gap-2" role="group" aria-label="Layer">
            {LAYERS.map((layer) => (
              <button
                key={layer}
                onClick={() => setSelectedLayer(layer)}
                data-selected={selectedLayer === layer}
                className="chip"
              >
                {layer}
              </button>
            ))}
          </div>
        </div>

        {/* Epoch scrubber */}
        <div className="flex w-full flex-col gap-2 md:max-w-lg">
          <div className="flex w-full justify-between font-mono text-[11px] text-ink-3">
            <span>EPOCH 0 · random</span>
            <span>EPOCH 74 · trained</span>
          </div>
          <div className="flex w-full gap-1" role="group" aria-label="Epoch">
            {availableEpochs.map((epoch) => (
              <button
                key={epoch}
                onClick={() => setSelectedEpoch(epoch)}
                data-selected={selectedEpoch === epoch}
                className="chip !min-w-0 flex-1 !px-0 justify-center !text-[10.5px] sm:!text-[11px] max-sm:!min-h-11"
              >
                {epoch}
              </button>
            ))}
          </div>
        </div>
      </div>

      {stats ? (
        <>
          {/* Stats */}
          <dl className="grid grid-cols-2 border-y border-rule md:grid-cols-4">
            {[
              { label: "Mean", value: stats.mean },
              { label: "Std Dev", value: stats.std },
              { label: "Min", value: stats.min },
              { label: "Max", value: stats.max },
            ].map((stat) => (
              <div
                key={stat.label}
                className="flex flex-col gap-1 border-rule px-1 py-4 odd:border-r md:border-r md:px-5 md:first:pl-1 md:last:border-r-0"
              >
                <dt className="caption">{stat.label}</dt>
                <dd className="m-0 font-mono text-xl tabular-nums text-sig">
                  {stat.value.toFixed(4)}
                </dd>
              </div>
            ))}
          </dl>

          <div className="grid gap-6 md:grid-cols-[1fr_auto] md:items-stretch">
            {/* Weight distribution histogram (recharts) */}
            {histData && (
              <div className="flex min-w-0 flex-col gap-2">
                <span className="caption">Weight distribution</span>
                <ChartContainer
                  config={chartConfig}
                  className="well h-[200px] w-full flex-1 !aspect-auto sm:h-[260px] md:h-auto md:min-h-[260px]"
                >
                  <BarChart
                    data={histData}
                    margin={{ top: 8, right: 28, bottom: 4, left: 8 }}
                    barCategoryGap={0}
                    barGap={0}
                  >
                    <XAxis
                      dataKey="binCenter"
                      type="number"
                      domain={[-globalMaxAbs, globalMaxAbs]}
                      tick={{ fill: "#8896a3", fontSize: 11 }}
                      tickLine={false}
                      axisLine={{ stroke: "rgba(170,205,225,0.14)" }}
                      tickCount={5}
                      tickFormatter={(v: number) => v.toFixed(2)}
                    />
                    <YAxis hide />
                    <ReferenceLine
                      x={0}
                      stroke="rgba(170,205,225,0.3)"
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
                          <div className="rounded-[3px] border border-rule-strong bg-bg-raised px-2 py-1 font-mono text-[11px] text-ink">
                            <span className="text-ink-3">
                              {d.binCenter.toFixed(4)}
                            </span>
                            <span className="ml-2 text-sig">{d.count}</span>
                          </div>
                        );
                      }}
                    />
                    <Bar dataKey="count" radius={[1, 1, 0, 0]}>
                      {histData.map((entry, i) => (
                        <Cell
                          key={i}
                          fill={entry.isNegative ? "#7f8cff" : "#ff6b4a"}
                          fillOpacity={0.8}
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
          </div>
        </>
      ) : (
        <p className="caption">
          No snapshot available for {selectedLayer} at epoch {selectedEpoch}
        </p>
      )}

      <p className="callout">
        <span className="tag">NOTE</span>
        {selectedEpoch === 0
          ? "At epoch 0, weights are random — the network has no idea what it's looking at."
          : selectedEpoch < 10
            ? "Early epochs — the weights are starting to organize into meaningful patterns."
            : "The weights have converged into structured filters that detect specific features."}
      </p>
    </div>
  );
}
