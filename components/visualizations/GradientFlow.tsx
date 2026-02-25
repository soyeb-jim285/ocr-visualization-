"use client";

import { useRef, useMemo, useState, useEffect } from "react";
import { useInView } from "framer-motion";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Cell,
} from "recharts";
import {
  ChartContainer,
  ChartTooltip,
  type ChartConfig,
} from "@/components/ui/chart";
import type { WeightSnapshots } from "@/lib/training/trainingData";
import { loadWeightSnapshots } from "@/lib/training/trainingData";

const LAYERS = ["conv1", "conv2", "conv3", "dense1"];
const SNAPSHOT_EPOCHS = [0, 1, 2, 5, 10, 15, 20, 25, 30, 40, 49];

const LAYER_COLORS: Record<string, string> = {
  conv1: "#06b6d4",
  conv2: "#06b6d4",
  conv3: "#8b5cf6",
  dense1: "#6366f1",
};

const chartConfig = {
  magnitude: { label: "Gradient Magnitude", color: "#8b5cf6" },
} satisfies ChartConfig;

/** Recursively flatten nested arrays */
function flatten(data: unknown): number[] {
  if (typeof data === "number") return [data];
  if (Array.isArray(data)) return data.flatMap(flatten);
  return [];
}

function computeGradientProxy(
  snapshots: WeightSnapshots,
  layer: string,
  epochIdx: number
): number {
  const epochs = SNAPSHOT_EPOCHS.filter(
    (e) => snapshots[String(e)]?.[layer] !== undefined
  );
  if (epochIdx <= 0 || epochIdx >= epochs.length) return 0;

  const prev = snapshots[String(epochs[epochIdx - 1])]?.[layer];
  const curr = snapshots[String(epochs[epochIdx])]?.[layer];
  if (!prev || !curr) return 0;

  if (typeof prev === "object" && !Array.isArray(prev) && "std" in prev) {
    const pStd = (prev as { std: number }).std;
    const cStd = (curr as { std: number }).std;
    const pMean = (prev as { mean: number }).mean;
    const cMean = (curr as { mean: number }).mean;
    return Math.abs(cMean - pMean) + Math.abs(cStd - pStd);
  }

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
    () =>
      snapshots
        ? SNAPSHOT_EPOCHS.filter((e) => snapshots[String(e)] !== undefined)
        : [],
    [snapshots]
  );

  const chartData = useMemo(() => {
    if (!snapshots) return null;
    const values = LAYERS.map((layer) => ({
      name: layer,
      magnitude: computeGradientProxy(snapshots, layer, epochIdx),
      color: LAYER_COLORS[layer],
    }));
    return values;
  }, [snapshots, epochIdx]);

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
          Weight change between epochs{" "}
          {availableEpochs[epochIdx - 1] ?? "?"} →{" "}
          {availableEpochs[epochIdx] ?? "?"}
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

      {/* Gradient flow bar chart */}
      {chartData && (
        <ChartContainer
          config={chartConfig}
          className="h-[180px] w-full max-w-lg rounded-xl border border-border bg-surface sm:h-[200px]"
        >
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 8, right: 48, bottom: 8, left: 56 }}
          >
            <XAxis
              type="number"
              tick={{ fill: "rgba(232,232,237,0.3)", fontSize: 9 }}
              tickLine={false}
              axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
              tickFormatter={(v: number) => v.toExponential(1)}
            />
            <YAxis
              type="category"
              dataKey="name"
              tick={{ fill: "rgba(232,232,237,0.5)", fontSize: 11 }}
              tickLine={false}
              axisLine={false}
              width={52}
            />
            <ChartTooltip
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const d = payload[0].payload as {
                  name: string;
                  magnitude: number;
                };
                return (
                  <div className="rounded border border-border bg-surface px-2 py-1 text-xs shadow-lg">
                    <span className="font-mono text-foreground/50">
                      {d.name}
                    </span>
                    <span className="ml-2 font-mono font-bold text-foreground/80">
                      {d.magnitude.toExponential(3)}
                    </span>
                  </div>
                );
              }}
            />
            <Bar dataKey="magnitude" radius={[0, 4, 4, 0]} barSize={24}>
              {chartData.map((entry, i) => (
                <Cell key={i} fill={entry.color} fillOpacity={0.7} />
              ))}
            </Bar>
          </BarChart>
        </ChartContainer>
      )}

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
