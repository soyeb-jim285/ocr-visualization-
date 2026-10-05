"use client";

import { useMemo, useState } from "react";
import { motion } from "framer-motion";
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

const reveal = (delay = 0) => ({
  initial: { opacity: 0, y: 12 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, margin: "-12% 0px" },
  transition: { duration: 0.5, delay, ease: [0.16, 1, 0.3, 1] as const },
});

/* ── ReLU chart config & data ────────────────────────────────────── */

const chartConfig = {
  negative: { label: "Zeroed (x ≤ 0)", color: "#ff6b4a" },
  positive: { label: "Identity (x > 0)", color: "#8fe3ff" },
  reference: { label: "y = x", color: "rgba(170,205,225,0.3)" },
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
    <div className="rounded-[3px] border border-rule-strong bg-bg-raised px-2 py-1 font-mono text-[11px]">
      <div className="flex items-center gap-3">
        <span className="text-ink-3">x = <span className="font-mono text-ink">{x.toFixed(1)}</span></span>
        <span className="text-ink-3">f(x) = <span className={`font-mono font-medium ${x <= 0 ? "text-annotation" : "text-phosphor"}`}>{y.toFixed(1)}</span></span>
      </div>
    </div>
  );
}

/* ── Main section ────────────────────────────────────────────────── */

export function ActivationSection() {
  const layerActivations = useInferenceStore((s) => s.layerActivations);
  const [pickedFilter, setSelectedFilter] = useState<number | null>(null);

  const conv1Maps = layerActivations["conv1"] as number[][][] | undefined;
  const relu1Maps = layerActivations["relu1"] as number[][][] | undefined;

  // default to the first filter that is not entirely dead, so the demo shows real data
  const firstLive = relu1Maps?.findIndex((fm) => fm.flat().some((v) => v > 0)) ?? -1;
  const selectedFilter = pickedFilter ?? Math.max(0, firstLive);

  const beforeRelu = conv1Maps?.[selectedFilter];
  const afterRelu = relu1Maps?.[selectedFilter];

  // BatchNorm is folded into the exported conv weights, so relu = max(0, conv) exactly.
  const stats = useMemo(() => {
    if (!afterRelu) return null;
    const flat = afterRelu.flat();
    const total = flat.length;
    const activeNeurons = flat.filter((v) => v > 0).length;
    const negCount = total - activeNeurons;
    const negPercent = ((negCount / total) * 100).toFixed(1);
    return { negCount, negPercent, total, activeNeurons };
  }, [afterRelu]);

  const numFilters = conv1Maps?.length ?? 32;

  return (
    <SectionWrapper id="activation" sig="relu" mirror>
      <SectionHeader
        step={3}
        tag="ReLU · 32 ch · 28×28"
        title="Amplifying Signals: ReLU Activation"
        subtitle="ReLU (Rectified Linear Unit) is deceptively simple: it keeps positive values unchanged and sets all negative values to zero. This non-linearity is what lets networks learn complex patterns."
      />

      <div className="grid grid-cols-[minmax(0,1fr)] items-start gap-x-6 gap-y-10 lg:grid-cols-12">
        {/* Figures first on desktop (mirrored plate) */}
        <motion.div {...reveal(0.16)} className="space-y-10 min-w-0 lg:order-1 lg:col-span-7">
          {/* ReLU curve */}
          <figure className="figure m-0">
            <span className="mb-3 block font-mono text-xs text-ink-2">
              f(x) = max(0, x)
            </span>
            <div className="well plate-marks p-2">
              <ChartContainer config={chartConfig} className="h-[200px] w-full touch-pan-y sm:h-[220px]">
                <LineChart data={reluData} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(170,205,225,0.07)" />
                  <ReferenceArea x1={-3} x2={0} fill="rgba(255,107,74,0.05)" fillOpacity={1} />
                  <XAxis
                    dataKey="x"
                    type="number"
                    domain={[-3, 3]}
                    ticks={[-3, -2, -1, 0, 1, 2, 3]}
                    stroke="rgba(170,205,225,0.3)"
                    tick={{ fontSize: 11, fill: "#8896a3" }}
                    axisLine={{ stroke: "rgba(170,205,225,0.3)" }}
                  />
                  <YAxis
                    domain={[-0.5, 3]}
                    ticks={[0, 1, 2, 3]}
                    stroke="rgba(170,205,225,0.3)"
                    tick={{ fontSize: 11, fill: "#8896a3" }}
                    axisLine={{ stroke: "rgba(170,205,225,0.3)" }}
                  />
                  <ReferenceLine y={0} stroke="rgba(170,205,225,0.3)" />
                  <ReferenceLine x={0} stroke="rgba(170,205,225,0.3)" />
                  <ChartTooltip content={<ReluTooltip />} allowEscapeViewBox={{ x: false, y: false }} />
                  <Line
                    type="monotone"
                    dataKey="reference"
                    stroke="var(--color-reference)"
                    strokeDasharray="4 3"
                    strokeWidth={1}
                    dot={false}
                    activeDot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="negative"
                    stroke="var(--color-negative)"
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 4, fill: "#ff6b4a", stroke: "#06080b", strokeWidth: 1 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="positive"
                    stroke="var(--color-positive)"
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 4, fill: "#8fe3ff", stroke: "#06080b", strokeWidth: 1 }}
                  />
                </LineChart>
              </ChartContainer>
            </div>
            <figcaption className="figcap [overflow-wrap:anywhere]">
              <b>FIG. 3.1</b> Orange: negatives are zeroed. Cyan: positives pass unchanged. Hover to
              read the mapping.
            </figcaption>
          </figure>

          {/* Before -> After */}
          <figure className="figure m-0">
            <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-5 sm:justify-start sm:gap-6">
              <div className="flex flex-col gap-2">
                <span className="font-mono text-xs text-annotation">BEFORE RELU</span>
                <div className="well plate-marks">
                  {beforeRelu ? (
                    <ActivationHeatmap data={beforeRelu} size={132} />
                  ) : (
                    <div className="viz-empty-state min-h-0 border-0 px-3 text-center text-xs sm:text-sm" style={{ width: 132, height: 132 }}>
                      Draw something to light this up
                    </div>
                  )}
                </div>
                <span className="font-mono text-xs text-ink-3">raw conv output</span>
              </div>

              <div className="text-ink-3">
                <Latex math="\xrightarrow{\max(0,\,x)}" className="max-sm:hidden" />
                <Latex math="\rightarrow" className="sm:hidden" />
              </div>

              <div className="flex flex-col gap-2">
                <span className="font-mono text-xs text-phosphor">AFTER RELU</span>
                <div className="well plate-marks">
                  {afterRelu ? (
                    <ActivationHeatmap data={afterRelu} size={132} />
                  ) : (
                    <div className="viz-empty-state min-h-0 border-0 px-3 text-center text-xs sm:text-sm" style={{ width: 132, height: 132 }}>
                      Draw something to light this up
                    </div>
                  )}
                </div>
                <span className="font-mono text-xs text-ink-3">negatives zeroed</span>
              </div>

              {stats && (
                <dl className="grid w-full grid-cols-3 gap-x-4 sm:ml-auto sm:w-auto sm:grid-cols-1 sm:gap-y-3">
                  {[
                    [stats.negCount, "zeroed", "text-annotation"],
                    [`${stats.negPercent}%`, "sparsity", "text-ink-3"],
                    [stats.activeNeurons, "active", "text-phosphor"],
                  ].map(([v, l, c]) => (
                    <div key={l as string}>
                      <dd className="font-mono text-xl tabular-nums text-ink">{v}</dd>
                      <dt className={`font-mono text-xs ${c}`}>{l}</dt>
                    </div>
                  ))}
                </dl>
              )}
            </div>
            <figcaption className="figcap">
              <b>FIG. 3.2</b> Filter {selectedFilter + 1}, before and after rectification, on a shared
              viridis scale.
            </figcaption>
          </figure>
        </motion.div>

        {/* Theory */}
        <motion.div {...reveal()} className="min-w-0 space-y-6 lg:order-2 lg:col-span-5">
          <p className="prose-body">
            After convolution produces raw feature maps, each value passes through a{" "}
            <em>non-linear activation function</em>. Without non-linearity, stacking layers would
            collapse into a single linear transformation. ReLU breaks this by zeroing all negative
            values while keeping positive ones unchanged.
          </p>

          <div className="formula">
            <Latex
              display
              math="\text{ReLU}(x) = \max(0,\, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}"
            />
            <span className="eq-no">(3)</span>
          </div>

          <dl className="grid grid-cols-[auto_1fr] items-baseline gap-x-4 gap-y-1.5 text-sm">
            <dt className="text-ink"><Latex math="x" /></dt>
            <dd className="font-mono text-xs text-ink-3">pre-activation value (conv, BatchNorm folded in)</dd>
            <dt className="text-ink"><Latex math="\max(0, x)" /></dt>
            <dd className="font-mono text-xs text-ink-3">output, always &ge; 0</dd>
          </dl>

          <p className="callout [overflow-wrap:anywhere]">
            <span className="tag">NOTE</span>
            During training a <em>BatchNorm</em> layer normalizes each conv output; for inference
            it is folded into the conv weights, so the map on the left is exactly what ReLU sees.
            Zeroed values create <em>sparse activations</em> that help the network focus on the
            strongest detected features.
          </p>

          <p className="text-sm leading-[1.65] text-ink-3">
            The gradient is just as simple:{" "}
            <Latex math="\frac{\partial}{\partial x}\text{ReLU}(x) = \mathbf{1}_{x > 0}" />. It
            passes gradients through for positive inputs and blocks them for negative ones, avoiding
            the vanishing gradient problem of sigmoid and tanh. Applied element-wise, the shape is
            preserved: <Latex math="(32, 28, 28) \xrightarrow{\text{ReLU}} (32, 28, 28)" />.
          </p>
        </motion.div>
      </div>

      {/* Filter selection */}
      <motion.figure {...reveal()} className="figure m-0 mt-16">
        <div className="grid grid-cols-[repeat(auto-fill,minmax(48px,1fr))] gap-x-2 gap-y-3">
          {relu1Maps
            ? relu1Maps.map((fm, i) => (
                <ActivationHeatmap
                  key={i}
                  data={fm}
                  size={48}
                  label={`${i + 1}`}
                  onClick={() => setSelectedFilter(i)}
                  selected={i === selectedFilter}
                />
              ))
            : Array.from({ length: numFilters }, (_, i) => (
                <div
                  key={i}
                  className={`flex flex-col items-center gap-1 focus-visible:outline focus-visible:outline-1 focus-visible:outline-sig ${
                    i === selectedFilter ? "opacity-100" : "opacity-40"
                  }`}
                  role="button"
                  tabIndex={0}
                  aria-label={`Filter ${i + 1}`}
                  onClick={() => setSelectedFilter(i)}
                  onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && setSelectedFilter(i)}
                >
                  <div className="tile cursor-pointer" style={{ width: 48, height: 48 }} />
                  <span className="font-mono text-xs text-ink-3">{i + 1}</span>
                </div>
              ))}
        </div>
        <figcaption className="figcap">
          <b>FIG. 3.3</b> All {numFilters} rectified feature maps. Select one to compare it above.
        </figcaption>
      </motion.figure>
    </SectionWrapper>
  );
}
