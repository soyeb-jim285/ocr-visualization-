"use client";

import { useMemo, useState } from "react";
import { SectionWrapper } from "@/components/ui/SectionWrapper";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { ActivationHeatmap } from "@/components/visualizations/ActivationHeatmap";
import { useInferenceStore } from "@/stores/inferenceStore";
import { Latex } from "@/components/ui/Latex";
import {
  ChartContainer,
  ChartTooltip,
  type ChartConfig,
} from "@/components/ui/chart";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ReferenceLine,
  ReferenceArea,
} from "recharts";

/* ── ReLU chart config & data ────────────────────────────────────── */

const chartConfig = {
  negative: { label: "Zeroed (x ≤ 0)", color: "#f87171" },
  positive: { label: "Identity (x > 0)", color: "#4ade80" },
  reference: { label: "y = x", color: "rgba(255,255,255,0.1)" },
} satisfies ChartConfig;

const reluData = (() => {
  const pts = [];
  for (let x = -3; x <= 3; x += 0.1) {
    const xv = parseFloat(x.toFixed(1));
    pts.push({
      x: xv,
      negative: xv <= 0 ? 0 : undefined,
      positive: xv >= 0 ? xv : undefined,
      reference: xv,
    });
  }
  return pts;
})();

/* ── Custom tooltip ──────────────────────────────────────────────── */

function ReluTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: { x: number } }> }) {
  if (!active || !payload?.length) return null;
  const x = payload[0].payload.x;
  const y = Math.max(0, x);
  return (
    <div className="rounded-lg border border-border/50 bg-background px-3 py-2 text-xs shadow-xl">
      <div className="flex items-center gap-3">
        <span className="text-foreground/50">x = <span className="font-mono text-foreground">{x.toFixed(1)}</span></span>
        <span className="text-foreground/50">f(x) = <span className={`font-mono font-medium ${x <= 0 ? "text-red-400" : "text-green-400"}`}>{y.toFixed(1)}</span></span>
      </div>
    </div>
  );
}

/* ── Main section ────────────────────────────────────────────────── */

export function ActivationSection() {
  const layerActivations = useInferenceStore((s) => s.layerActivations);
  const [selectedFilter, setSelectedFilter] = useState(0);

  const conv1Maps = layerActivations["conv1"] as number[][][] | undefined;
  const relu1Maps = layerActivations["relu1"] as number[][][] | undefined;

  const beforeRelu = conv1Maps?.[selectedFilter];
  const afterRelu = relu1Maps?.[selectedFilter];

  const stats = useMemo(() => {
    if (!beforeRelu) return null;
    const flat = beforeRelu.flat();
    const total = flat.length;
    const negCount = flat.filter((v) => v < 0).length;
    const negPercent = ((negCount / total) * 100).toFixed(1);
    const afterFlat = afterRelu?.flat() ?? flat.map((v) => Math.max(0, v));
    const activeNeurons = afterFlat.filter((v) => v > 0).length;
    return { negCount, negPercent, total, activeNeurons };
  }, [beforeRelu, afterRelu]);

  const numFilters = conv1Maps?.length ?? 0;
  const hasData = !!beforeRelu;

  return (
    <SectionWrapper id="activation">
      <SectionHeader
        step={3}
        title="Amplifying Signals: ReLU Activation"
        subtitle="ReLU (Rectified Linear Unit) is deceptively simple: it keeps positive values unchanged and sets all negative values to zero. This non-linearity is what allows neural networks to learn complex, non-linear patterns."
      />

      <div className="flex flex-col items-center gap-8 lg:flex-row lg:items-start lg:gap-12">
        {/* Left: theory text */}
        <div className="flex-1 space-y-4 text-center lg:text-left">
          <p className="text-base leading-relaxed text-foreground/65 sm:text-lg">
            After convolution produces raw feature maps, each value passes
            through a <em>non-linear activation function</em>. Without
            non-linearity, stacking multiple layers would collapse into a single
            linear transformation — equivalent to just one layer. ReLU breaks
            this by zeroing all negative values while keeping positive ones
            unchanged.
          </p>

          {/* Main equation — piecewise */}
          <div className="py-3">
            <Latex
              display
              math="\text{ReLU}(x) = \max(0,\, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}"
            />
          </div>

          {/* Equation legend */}
          <div className="flex flex-wrap justify-center gap-x-6 gap-y-1 text-sm text-foreground/40 lg:justify-start">
            <span><Latex math="x" /> — pre-activation value (from conv)</span>
            <span><Latex math="\max(0, x)" /> — output (always &ge; 0)</span>
          </div>

          <p className="text-sm leading-relaxed text-foreground/45">
            The gradient is equally simple:{" "}
            <Latex math="\frac{\partial}{\partial x}\text{ReLU}(x) = \mathbf{1}_{x > 0}" />
            {" "}— it passes gradients through unchanged for positive inputs and
            blocks them entirely for negative ones. This avoids the vanishing
            gradient problem that plagues sigmoid and tanh in deep networks.
            Applied element-wise, the shape is preserved:{" "}
            <Latex math="(64, 28, 28) \xrightarrow{\text{ReLU}} (64, 28, 28)" />.
            Typically 40–70% of values are zeroed, creating{" "}
            <em>sparse activations</em> that help the network focus on the
            strongest detected features.
          </p>
        </div>

        {/* Right: visualizations */}
        <div className="flex w-full shrink-0 flex-col items-center gap-5 lg:w-auto">
          {/* ReLU function chart */}
          <div className="flex flex-col items-center gap-1.5">
            <span className="text-xs text-foreground/40">
              Interactive ReLU curve
            </span>
            <ChartContainer
              config={chartConfig}
              className="h-[200px] w-[280px]"
            >
              <LineChart data={reluData} margin={{ top: 8, right: 12, bottom: 4, left: -8 }}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="rgba(255,255,255,0.05)"
                />
                <ReferenceArea
                  x1={-3}
                  x2={0}
                  fill="rgba(248, 113, 113, 0.04)"
                  fillOpacity={1}
                />
                <XAxis
                  dataKey="x"
                  type="number"
                  domain={[-3, 3]}
                  ticks={[-3, -2, -1, 0, 1, 2, 3]}
                  stroke="rgba(255,255,255,0.15)"
                  tick={{ fontSize: 10, fill: "rgba(255,255,255,0.3)" }}
                  axisLine={{ stroke: "rgba(255,255,255,0.2)" }}
                />
                <YAxis
                  domain={[-0.5, 3]}
                  ticks={[0, 1, 2, 3]}
                  stroke="rgba(255,255,255,0.15)"
                  tick={{ fontSize: 10, fill: "rgba(255,255,255,0.3)" }}
                  axisLine={{ stroke: "rgba(255,255,255,0.2)" }}
                />
                <ReferenceLine y={0} stroke="rgba(255,255,255,0.15)" />
                <ReferenceLine x={0} stroke="rgba(255,255,255,0.15)" />
                <ChartTooltip content={<ReluTooltip />} />
                <Line
                  type="monotone"
                  dataKey="reference"
                  stroke="var(--color-reference)"
                  strokeDasharray="4 4"
                  strokeWidth={1}
                  dot={false}
                  activeDot={false}
                />
                <Line
                  type="monotone"
                  dataKey="negative"
                  stroke="var(--color-negative)"
                  strokeWidth={2.5}
                  dot={false}
                  activeDot={{ r: 4, fill: "#f87171", stroke: "rgba(0,0,0,0.3)", strokeWidth: 1 }}
                />
                <Line
                  type="monotone"
                  dataKey="positive"
                  stroke="var(--color-positive)"
                  strokeWidth={2.5}
                  dot={false}
                  activeDot={{ r: 4, fill: "#4ade80", stroke: "rgba(0,0,0,0.3)", strokeWidth: 1 }}
                />
              </LineChart>
            </ChartContainer>
            <span className="text-[11px] text-foreground/30">
              Hover to see input &rarr; output mapping
            </span>
          </div>

          {hasData ? (
            <>
              {/* Before → After heatmaps */}
              <div className="flex flex-col items-center gap-5 sm:flex-row sm:gap-6">
                <div className="flex flex-col items-center gap-2">
                  <span className="text-xs font-medium text-red-400">
                    Before ReLU
                  </span>
                  <ActivationHeatmap data={beforeRelu} size={120} />
                  <span className="text-[11px] text-foreground/30">
                    Contains negatives
                  </span>
                </div>

                <div className="flex flex-col items-center gap-1">
                  <Latex
                    math="\xrightarrow{\max(0,\,x)}"
                    className="hidden text-foreground/40 sm:block"
                  />
                  <Latex
                    math="\downarrow"
                    className="text-foreground/40 sm:hidden"
                  />
                </div>

                <div className="flex flex-col items-center gap-2">
                  <span className="text-xs font-medium text-green-400">
                    After ReLU
                  </span>
                  {afterRelu ? (
                    <ActivationHeatmap data={afterRelu} size={120} />
                  ) : (
                    <div className="flex h-[120px] w-[120px] items-center justify-center rounded-md border border-border bg-surface">
                      <span className="text-xs text-foreground/20">
                        No data
                      </span>
                    </div>
                  )}
                  <span className="text-[11px] text-foreground/30">
                    Negatives zeroed
                  </span>
                </div>
              </div>

              {/* Stats — single line below heatmaps */}
              {stats && (
                <div className="flex items-center gap-4 text-sm">
                  <span className="font-mono font-semibold text-red-400">{stats.negCount}</span>
                  <span className="text-foreground/40">zeroed</span>
                  <span className="text-foreground/15">|</span>
                  <span className="font-mono font-semibold text-foreground/60">{stats.negPercent}%</span>
                  <span className="text-foreground/40">sparsity</span>
                  <span className="text-foreground/15">|</span>
                  <span className="font-mono font-semibold text-green-400">{stats.activeNeurons}</span>
                  <span className="text-foreground/40">active</span>
                </div>
              )}
            </>
          ) : (
            <div className="flex h-32 items-center justify-center">
              <p className="text-foreground/30">
                Draw a character above to see activations
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Filter selection: clickable feature map thumbnails — full width */}
      {hasData && (
        <div className="mt-6 space-y-3">
          <p className="text-center text-xs text-foreground/40">
            Select a filter — click any feature map below
          </p>
          <div className="grid grid-cols-4 gap-2 sm:grid-cols-8">
            {relu1Maps
              ? relu1Maps.map((fm, i) => (
                  <ActivationHeatmap
                    key={i}
                    data={fm}
                    size={56}
                    label={`#${i + 1}`}
                    onClick={() => setSelectedFilter(i)}
                    selected={i === selectedFilter}
                  />
                ))
              : Array.from({ length: numFilters }, (_, i) => (
                  <div
                    key={i}
                    className={`flex flex-col items-center gap-1 ${
                      i === selectedFilter ? "opacity-100" : "opacity-40"
                    }`}
                  >
                    <div
                      className="border border-border/50"
                      style={{
                        width: 56,
                        height: 56,
                        backgroundColor: "var(--surface)",
                      }}
                    />
                    <span className="text-xs text-foreground/40">#{i + 1}</span>
                  </div>
                ))}
          </div>
        </div>
      )}
    </SectionWrapper>
  );
}
