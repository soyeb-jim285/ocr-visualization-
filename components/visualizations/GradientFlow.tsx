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

const LAYER_COLORS: Record<string, string> = {
  conv1: "#7f8cff", // --sig-conv1
  conv2: "#7f8cff",
  conv3: "#35d0e6", // --sig-conv3
  dense1: "#ffb347", // --sig-dense
};

const chartConfig = {
  magnitude: { label: "Gradient Magnitude", color: "#f472b6" },
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
  const epochs = Object.keys(snapshots)
    .map(Number)
    .sort((a, b) => a - b)
    .filter((e) => snapshots[String(e)]?.[layer] !== undefined);
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
        ? Object.keys(snapshots).map(Number).sort((a, b) => a - b)
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

  if (!snapshots || availableEpochs.length < 2) {
    return (
      <div className="viz-empty-state flex min-h-[420px] items-center justify-center" aria-busy="true">
        <p>Loading gradient data...</p>
      </div>
    );
  }

  return (
    <div ref={ref} className="grid gap-6 md:grid-cols-12 md:gap-x-6">
      {/* Epoch selector */}
      <div className="flex flex-col gap-3 md:col-span-4">
        <span className="eyebrow">Weight change between epochs</span>
        <p className="font-mono text-2xl tabular-nums text-ink">
          {availableEpochs[epochIdx - 1] ?? "?"}
          <span className="mx-2 text-ink-3">→</span>
          {availableEpochs[epochIdx] ?? "?"}
        </p>
        <input
          type="range"
          min={1}
          max={availableEpochs.length - 1}
          value={epochIdx}
          onChange={(e) => setEpochIdx(parseInt(e.target.value))}
          className="mb-2 w-full"
          aria-label="Epoch interval"
        />
        <p className="callout mt-2">
          <span className="tag">NOTE</span>
          {availableEpochs[epochIdx] <= 3
            ? "Early training — large weight updates across all layers as the network rapidly learns basic patterns."
            : availableEpochs[epochIdx] <= 10
              ? "Learning is slowing down in earlier layers as they settle on stable feature detectors."
              : "Later epochs — weight changes become small and focused, fine-tuning rather than restructuring."}
        </p>
      </div>

      <div className="flex min-w-0 flex-col gap-3 md:col-span-8">
        {/* Direction */}
        <div className="flex items-center gap-3 font-mono text-[11px] text-ink-3">
          <span>OUTPUT</span>
          <svg width="100" height="16" viewBox="0 0 100 16" fill="none" aria-hidden>
            <path
              d="M100 8H10M10 8l8-6M10 8l8 6"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <span>INPUT</span>
          <span className="ml-2 max-sm:hidden">gradient flows backwards</span>
        </div>
        {/* Gradient flow bar chart */}
        {chartData && (
          <div>
            <ChartContainer
              config={chartConfig}
              className="well h-[200px] w-full min-w-0 sm:h-[220px]"
            >
              <BarChart
                data={chartData}
                layout="vertical"
                margin={{ top: 12, right: 24, bottom: 8, left: 56 }}
              >
                <XAxis
                  type="number"
                  tick={{ fill: "#8896a3", fontSize: 11 }}
                  tickLine={false}
                  axisLine={{ stroke: "rgba(170,205,225,0.14)" }}
                  tickCount={3}
                  tickFormatter={(v: number) => v.toExponential(0)}
                />
                <YAxis
                  type="category"
                  dataKey="name"
                  tick={{ fill: "#b4c0ca", fontSize: 11 }}
                  tickLine={false}
                  axisLine={false}
                  width={52}
                />
                <ChartTooltip
                  wrapperStyle={{ pointerEvents: "none" }}
                  cursor={{ fill: "rgba(170,205,225,0.05)" }}
                  content={({ active, payload }) => {
                    if (!active || !payload?.length) return null;
                    const d = payload[0].payload as {
                      name: string;
                      magnitude: number;
                    };
                    return (
                      <div className="rounded-[3px] border border-rule-strong bg-bg-raised px-2 py-1 font-mono text-[11px] text-ink">
                        <span className="text-ink-3">{d.name}</span>
                        <span className="ml-2">{d.magnitude.toExponential(3)}</span>
                      </div>
                    );
                  }}
                />
                <Bar dataKey="magnitude" radius={[0, 2, 2, 0]} barSize={22}>
                  {chartData.map((entry, i) => (
                    <Cell key={i} fill={entry.color} fillOpacity={0.85} />
                  ))}
                </Bar>
              </BarChart>
            </ChartContainer>
          </div>
        )}
      </div>
    </div>
  );
}
