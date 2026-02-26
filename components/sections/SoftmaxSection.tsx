"use client";

import { useMemo } from "react";
import { BarChart, Bar, XAxis, YAxis, Cell, ReferenceLine } from "recharts";
import {
  ChartContainer,
  ChartTooltip,
  type ChartConfig,
} from "@/components/ui/chart";
import { SectionWrapper } from "@/components/ui/SectionWrapper";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { useInferenceStore } from "@/stores/inferenceStore";
import {
  EMNIST_CLASSES,
  BYMERGE_MERGED_INDICES,
  NUM_CLASSES,
} from "@/lib/model/classes";
import { Latex } from "@/components/ui/Latex";

/* ── Types ───────────────────────────────────────────────────────── */

interface ClassEntry {
  idx: number;
  label: string;
  logit: number;
  prob: number;
}

/* ── Chart configs ────────────────────────────────────────────────── */

const logitConfig = {
  logit: { label: "Logit", color: "#6366f1" },
} satisfies ChartConfig;

const probConfig = {
  prob: { label: "Probability", color: "#8b5cf6" },
} satisfies ChartConfig;

/* ── Logits → Probabilities dual bar chart ───────────────────────── */

function SoftmaxChart({
  entries,
  topIdx,
}: {
  entries: ClassEntry[];
  topIdx: number;
}) {
  return (
    <div className="flex w-full flex-col items-center gap-3">
      {/* Logit chart */}
      <div className="w-full">
        <p className="mb-1 text-center text-[11px] text-foreground/35">
          Raw logits (before softmax)
        </p>
        <ChartContainer
          config={logitConfig}
          className="h-[160px] w-full rounded-lg border border-border bg-surface"
        >
          <BarChart
            data={entries}
            margin={{ top: 8, right: 4, bottom: 4, left: 4 }}
            barCategoryGap={0}
            barGap={0}
          >
            <XAxis dataKey="label" hide />
            <YAxis
              tick={{ fill: "rgba(232,232,237,0.3)", fontSize: 9 }}
              tickLine={false}
              axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
              width={32}
            />
            <ReferenceLine y={0} stroke="rgba(255,255,255,0.15)" strokeDasharray="4 4" />
            <ChartTooltip
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const d = payload[0].payload as ClassEntry;
                return (
                  <div className="rounded-lg border border-border/70 bg-surface-elevated/95 px-3 py-2 shadow-lg backdrop-blur-sm">
                    <p className="mb-0.5 font-mono text-sm font-bold text-accent-primary">{d.label}</p>
                    <p className="font-mono text-xs">
                      <span className="text-foreground/50">logit: </span>
                      <span className={d.logit >= 0 ? "text-green-400" : "text-red-400"}>
                        {d.logit.toFixed(2)}
                      </span>
                    </p>
                    <p className="font-mono text-xs">
                      <span className="text-foreground/50">prob: </span>
                      <span className="text-accent-secondary">{(d.prob * 100).toFixed(2)}%</span>
                    </p>
                  </div>
                );
              }}
            />
            <Bar dataKey="logit" radius={[2, 2, 0, 0]}>
              {entries.map((e) => (
                <Cell
                  key={e.idx}
                  fill={
                    e.idx === topIdx
                      ? "rgba(99, 102, 241, 0.8)"
                      : e.logit >= 0
                        ? "rgba(74, 222, 128, 0.3)"
                        : "rgba(248, 113, 113, 0.25)"
                  }
                />
              ))}
            </Bar>
          </BarChart>
        </ChartContainer>
      </div>

      {/* Arrow */}
      <Latex
        math="\downarrow\;\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum e^{z_j}}"
        className="text-foreground/30"
      />

      {/* Probability chart */}
      <div className="w-full">
        <p className="mb-1 text-center text-[11px] text-foreground/35">
          Probabilities (after softmax)
        </p>
        <ChartContainer
          config={probConfig}
          className="h-[160px] w-full rounded-lg border border-border bg-surface"
        >
          <BarChart
            data={entries}
            margin={{ top: 8, right: 4, bottom: 4, left: 4 }}
            barCategoryGap={0}
            barGap={0}
          >
            <XAxis dataKey="label" hide />
            <YAxis
              tick={{ fill: "rgba(232,232,237,0.3)", fontSize: 9 }}
              tickLine={false}
              axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
              width={32}
              tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
            />
            <ChartTooltip
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const d = payload[0].payload as ClassEntry;
                return (
                  <div className="rounded-lg border border-border/70 bg-surface-elevated/95 px-3 py-2 shadow-lg backdrop-blur-sm">
                    <p className="mb-0.5 font-mono text-sm font-bold text-accent-primary">{d.label}</p>
                    <p className="font-mono text-xs">
                      <span className="text-foreground/50">logit: </span>
                      <span className={d.logit >= 0 ? "text-green-400" : "text-red-400"}>
                        {d.logit.toFixed(2)}
                      </span>
                    </p>
                    <p className="font-mono text-xs">
                      <span className="text-foreground/50">prob: </span>
                      <span className="text-accent-secondary">{(d.prob * 100).toFixed(2)}%</span>
                    </p>
                  </div>
                );
              }}
            />
            <Bar dataKey="prob" radius={[2, 2, 0, 0]}>
              {entries.map((e) => (
                <Cell
                  key={e.idx}
                  fill={
                    e.idx === topIdx
                      ? "rgba(99, 102, 241, 0.85)"
                      : e.prob > 0.01
                        ? "rgba(139, 92, 246, 0.3)"
                        : "rgba(255, 255, 255, 0.08)"
                  }
                />
              ))}
            </Bar>
          </BarChart>
        </ChartContainer>
      </div>
    </div>
  );
}

/* ── Main section ────────────────────────────────────────────────── */

export function SoftmaxSection() {
  const prediction = useInferenceStore((s) => s.prediction);
  const topPrediction = useInferenceStore((s) => s.topPrediction);
  const layerActivations = useInferenceStore((s) => s.layerActivations);
  const rawLogits = layerActivations["output"] as number[] | undefined;

  // Build sorted class entries (valid classes only, sorted by logit descending)
  const { entries, topIdx } = useMemo(() => {
    if (!rawLogits || !prediction)
      return { entries: [] as ClassEntry[], topIdx: -1 };

    const entries: ClassEntry[] = [];
    for (let i = 0; i < NUM_CLASSES; i++) {
      if (BYMERGE_MERGED_INDICES.has(i)) continue;
      entries.push({
        idx: i,
        label: EMNIST_CLASSES[i],
        logit: rawLogits[i] ?? 0,
        prob: prediction[i] ?? 0,
      });
    }
    entries.sort((a, b) => b.logit - a.logit);

    const topIdx = topPrediction?.classIndex ?? -1;
    return { entries, topIdx };
  }, [rawLogits, prediction, topPrediction]);

  const hasData = entries.length > 0;

  return (
    <SectionWrapper id="softmax">
      <SectionHeader
        step={8}
        title="Confidence: Softmax"
        subtitle="The final layer produces 146 raw scores (logits), one for each character class. Softmax converts these into probabilities that sum to 1.0 — transforming 'how much does this look like each character?' into 'what's the probability it IS each character?'"
      />

      <div className="flex flex-col items-center gap-8 lg:flex-row lg:items-start lg:gap-12">
        {/* Left: theory text */}
        <div className="flex-1 space-y-4 text-center lg:text-left">
          <p className="text-base leading-relaxed text-foreground/65 sm:text-lg">
            The dense layer outputs 146 raw <em>logits</em> — unbounded real
            numbers where larger means more confident. The{" "}
            <em>softmax function</em> normalizes these into a proper probability
            distribution: each output is positive, and they all sum to exactly
            1.0. The exponentiation amplifies differences — a logit just
            slightly larger than the rest can dominate the distribution.
          </p>

          {/* Main equation */}
          <div className="py-3">
            <Latex
              display
              math="\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\displaystyle\sum_{j=1}^{K} e^{z_j}}"
            />
          </div>

          {/* Equation legend */}
          <div className="flex flex-wrap justify-center gap-x-6 gap-y-1 text-sm text-foreground/40 lg:justify-start">
            <span>
              <Latex math="\mathbf{z}" /> — raw logits (146 values)
            </span>
            <span>
              <Latex math="K = 146" /> — number of classes
            </span>
            <span>
              <Latex math="\sigma_i" /> — probability of class{" "}
              <Latex math="i" />
            </span>
          </div>

          <p className="text-sm leading-relaxed text-foreground/45">
            In practice we subtract the max logit first for numerical stability:{" "}
            <Latex math="e^{z_i - z_{\max}}" /> instead of{" "}
            <Latex math="e^{z_i}" />. Of the 146 output neurons, 15 are masked
            (the ByMerge dataset merges ambiguous Latin pairs like C/c and O/o),
            leaving 131 valid character classes: Latin digits 0–9, uppercase A–Z,
            select lowercase letters, and 84 Bengali characters.
          </p>
        </div>

        {/* Right: before/after softmax charts */}
        <div className="flex w-full max-w-[980px] shrink-0 flex-col items-center gap-4 lg:w-1/3 lg:max-w-[560px]">
          {hasData ? (
            <SoftmaxChart entries={entries} topIdx={topIdx} />
          ) : (
            <div className="flex h-48 items-center justify-center">
              <p className="text-foreground/30">
                Draw a character above to see predictions
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Top prediction + stats — single line */}
      {prediction && topPrediction && (
        <div className="mt-8 flex flex-wrap items-center justify-center gap-4 text-sm">
          <span className="text-3xl font-bold text-accent-primary">
            {EMNIST_CLASSES[topPrediction.classIndex]}
          </span>
          <span className="text-foreground/15">|</span>
          <span className="font-mono font-semibold text-accent-secondary">
            {(topPrediction.confidence * 100).toFixed(1)}%
          </span>
          <span className="text-foreground/40">confidence</span>
          <span className="text-foreground/15">|</span>
          <span className="font-mono font-semibold text-foreground/60">
            131
          </span>
          <span className="text-foreground/40">valid classes</span>
          <span className="text-foreground/15">|</span>
          <span className="font-mono font-semibold text-foreground/60">
            {prediction.reduce((s, p) => s + p, 0).toFixed(3)}
          </span>
          <span className="text-foreground/40">sum</span>
        </div>
      )}
    </SectionWrapper>
  );
}
