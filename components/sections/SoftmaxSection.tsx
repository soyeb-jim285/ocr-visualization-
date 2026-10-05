"use client";

import { useMemo } from "react";
import { motion, useReducedMotion } from "framer-motion";
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
import { ANNOTATION, INK3, PHOSPHOR, RULE, RULE_STRONG } from "@/lib/theme";

/* ── Types ───────────────────────────────────────────────────────── */

interface ClassEntry {
  idx: number;
  label: string;
  logit: number;
  prob: number;
}

/* ── Chart configs ────────────────────────────────────────────────── */

const logitConfig = {
  logit: { label: "Logit", color: PHOSPHOR },
} satisfies ChartConfig;

const probConfig = {
  prob: { label: "Probability", color: PHOSPHOR },
} satisfies ChartConfig;

const AXIS_TICK = { fill: INK3, fontSize: 11, fontFamily: "var(--font-geist-mono), monospace" };

function useReveal(delay = 0) {
  const reduce = useReducedMotion();
  return reduce
    ? {}
    : {
        initial: { opacity: 0, y: 12 },
        whileInView: { opacity: 1, y: 0 },
        viewport: { once: true, margin: "-12% 0px" },
        transition: { duration: 0.5, delay, ease: [0.16, 1, 0.3, 1] as const },
      };
}

/* ── Tooltip ──────────────────────────────────────────────────────── */

function EntryTip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: { payload: ClassEntry }[];
}) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="rounded-[3px] border border-rule-strong bg-bg-raised px-2 py-1 font-mono text-[11px] text-ink">
      <p className="mb-0.5 font-serif text-base text-annotation">{d.label}</p>
      <p>
        <span className="text-ink-3">logit </span>
        {d.logit.toFixed(2)}
      </p>
      <p>
        <span className="text-ink-3">prob </span>
        {(d.prob * 100).toFixed(2)}%
      </p>
    </div>
  );
}

/* ── Logits → Probabilities dual bar chart ───────────────────────── */

function SoftmaxChart({
  entries,
  topIdx,
}: {
  entries: ClassEntry[];
  topIdx: number;
}) {
  return (
    <div className="flex w-full flex-col gap-2">
      {/* Logit chart */}
      <div>
        <p className="caption mb-2">RAW LOGITS · BEFORE SOFTMAX</p>
        <ChartContainer config={logitConfig} className="well plate-marks h-[180px] w-full">
          <BarChart
            data={entries}
            margin={{ top: 12, right: 8, bottom: 4, left: 4 }}
            barCategoryGap={0}
            barGap={0}
          >
            <XAxis dataKey="label" hide />
            <YAxis tick={AXIS_TICK} tickLine={false} axisLine={{ stroke: RULE }} width={34} />
            <ReferenceLine y={0} stroke={RULE_STRONG} strokeDasharray="4 3" />
            <ChartTooltip cursor={{ fill: RULE }} content={<EntryTip />} />
            <Bar dataKey="logit" radius={[1, 1, 0, 0]} isAnimationActive={false}>
              {entries.map((e) => (
                <Cell
                  key={e.idx}
                  fill={
                    e.idx === topIdx
                      ? ANNOTATION
                      : e.logit >= 0
                        ? "rgba(143,227,255,0.45)"
                        : "rgba(136,150,163,0.28)"
                  }
                />
              ))}
            </Bar>
          </BarChart>
        </ChartContainer>
      </div>

      {/* Transform */}
      <div className="flex items-center gap-4 py-2 text-ink-2">
        <span className="h-px flex-1 bg-rule" />
        <Latex math="\downarrow\;\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum e^{z_j}}" />
        <span className="h-px flex-1 bg-rule" />
      </div>

      {/* Probability chart */}
      <div>
        <p className="caption mb-2">PROBABILITIES · AFTER SOFTMAX</p>
        <ChartContainer config={probConfig} className="well plate-marks h-[180px] w-full">
          <BarChart
            data={entries}
            margin={{ top: 12, right: 8, bottom: 4, left: 4 }}
            barCategoryGap={0}
            barGap={0}
          >
            <XAxis dataKey="label" hide />
            <YAxis
              tick={AXIS_TICK}
              tickLine={false}
              axisLine={{ stroke: RULE }}
              width={34}
              tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
            />
            <ChartTooltip cursor={{ fill: RULE }} content={<EntryTip />} />
            <Bar dataKey="prob" radius={[1, 1, 0, 0]} isAnimationActive={false}>
              {entries.map((e) => (
                <Cell
                  key={e.idx}
                  fill={
                    e.idx === topIdx
                      ? ANNOTATION
                      : e.prob > 0.01
                        ? "rgba(143,227,255,0.55)"
                        : "rgba(136,150,163,0.22)"
                  }
                />
              ))}
            </Bar>
          </BarChart>
        </ChartContainer>
        <p className="caption mt-2">
          {entries.filter((e) => e.prob < 0.01).length} of {entries.length} classes below 1%
        </p>
      </div>
    </div>
  );
}

/* ── Verdict: the climactic readout ──────────────────────────────── */

function Verdict({
  letter,
  confidence,
  runnersUp,
  sum,
}: {
  letter: string;
  confidence: number;
  runnersUp: ClassEntry[];
  sum: number;
}) {
  const reveal = useReveal(0.1);
  const pct = confidence * 100;
  return (
    <motion.div className="figure mt-16 md:mt-24" {...reveal}>
      <p className="eyebrow mb-6">VERDICT</p>
      <div className="grid items-center gap-x-10 gap-y-8 md:grid-cols-12">
        <div className="flex items-end gap-6 md:col-span-5">
          <span
            key={letter}
            className="font-serif text-[clamp(7rem,18vw,13rem)] font-light leading-[0.8] text-annotation"
            style={{ textShadow: "0 0 60px rgba(255,107,74,0.35)" }}
            aria-label={`Predicted character ${letter}`}
          >
            {letter}
          </span>
        </div>

        <div className="md:col-span-7 md:border-l md:border-rule md:pl-10">
          <p className="font-serif text-[clamp(2.5rem,6vw,4.5rem)] font-light leading-none tracking-[-0.02em] text-ink tabular-nums">
            {pct.toFixed(1)}
            <span className="text-ink-3">%</span>
          </p>
          <p className="caption mt-2">CONFIDENCE</p>

          {/* confidence bar: scaleX only */}
          <div className="mt-5 h-1.5 w-full bg-rule-faint" role="presentation">
            <div
              className="h-full origin-left bg-annotation transition-transform duration-700 [transition-timing-function:var(--ease-out)] motion-reduce:transition-none"
              style={{ transform: `scaleX(${Math.min(1, Math.max(0, confidence))})` }}
            />
          </div>

          {runnersUp.length > 0 && (
            <ul className="mt-6 flex flex-wrap gap-x-8 gap-y-2">
              {runnersUp.map((e, i) => (
                <li key={e.idx} className="flex items-baseline gap-2">
                  <span className="caption">#{i + 2}</span>
                  <span className="font-serif text-2xl text-ink-2">{e.label}</span>
                  <span className="readout text-ink-3">{(e.prob * 100).toFixed(1)}%</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      <dl className="mt-10 grid grid-cols-2 gap-4 border-t border-rule pt-4 sm:grid-cols-3">
        {[
          { v: "131", l: "valid classes" },
          { v: sum.toFixed(3), l: "probabilities sum" },
          { v: "15", l: "masked outputs" },
        ].map((s) => (
          <div key={s.l}>
            <dd className="readout text-base">{s.v}</dd>
            <dt className="caption mt-1">{s.l}</dt>
          </div>
        ))}
      </dl>
    </motion.div>
  );
}

/* ── Main section ────────────────────────────────────────────────── */

export function SoftmaxSection() {
  const prediction = useInferenceStore((s) => s.prediction);
  const topPrediction = useInferenceStore((s) => s.topPrediction);
  const layerActivations = useInferenceStore((s) => s.layerActivations);
  const rawLogits = layerActivations["output"] as number[] | undefined;
  const textReveal = useReveal(0);
  const figReveal = useReveal(0.16);

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

  const runnersUp = useMemo(
    () =>
      entries
        .filter((e) => e.idx !== topIdx)
        .sort((a, b) => b.prob - a.prob)
        .slice(0, 3),
    [entries, topIdx],
  );

  return (
    <SectionWrapper id="softmax" sig="out">
      <SectionHeader
        step={8}
        tag="Softmax · 146 → 131 classes"
        title="Confidence: Softmax"
        subtitle="The final layer produces 146 raw scores (logits), one for each character class. Softmax converts these into probabilities that sum to 1.0 — transforming 'how much does this look like each character?' into 'what's the probability it IS each character?'"
      />

      <div className="grid gap-x-6 gap-y-14 lg:grid-cols-12">
        {/* Theory text */}
        <motion.div className="space-y-6 lg:col-span-5 lg:pt-4" {...textReveal}>
          <p className="prose-body">
            The dense layer outputs 146 raw <em>logits</em> — unbounded real
            numbers where larger means more confident. The{" "}
            <em>softmax function</em> normalizes these into a proper probability
            distribution: each output is positive, and they all sum to exactly
            1.0. The exponentiation amplifies differences — a logit just
            slightly larger than the rest can dominate the distribution.
          </p>

          <div className="formula">
            <Latex
              display
              math="\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\displaystyle\sum_{j=1}^{K} e^{z_j}}"
            />
            <span className="eq-no">(8)</span>
          </div>

          <ul className="caption space-y-1.5">
            <li><Latex math="\mathbf{z}" /> — raw logits (146 values)</li>
            <li><Latex math="K = 146" /> — number of classes</li>
            <li><Latex math="\sigma_i" /> — probability of class <Latex math="i" /></li>
          </ul>

          <p className="prose-body text-[0.9375rem] text-ink-3">
            In practice we subtract the max logit first for numerical stability:{" "}
            <Latex math="e^{z_i - z_{\max}}" /> instead of{" "}
            <Latex math="e^{z_i}" />. Of the 146 output neurons, 15 are masked
            (the ByMerge dataset merges ambiguous Latin pairs like C/c and O/o),
            leaving 131 valid character classes: Latin digits 0–9, uppercase A–Z,
            select lowercase letters, and 84 Bengali characters.
          </p>
        </motion.div>

        {/* Before/after softmax charts */}
        <motion.div className="figure lg:col-span-7" {...figReveal}>
          {hasData ? (
            <SoftmaxChart entries={entries} topIdx={topIdx} />
          ) : (
            <div className="flex flex-col gap-2">
              <p className="caption mb-2">RAW LOGITS · BEFORE SOFTMAX</p>
              <div className="viz-empty-state plate-marks h-[180px]">Draw something to light this up</div>
              <div className="flex items-center gap-4 py-2 text-ink-2">
                <span className="h-px flex-1 bg-rule" />
                <Latex math="\downarrow\;\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum e^{z_j}}" />
                <span className="h-px flex-1 bg-rule" />
              </div>
              <p className="caption mb-2">PROBABILITIES · AFTER SOFTMAX</p>
              <div className="viz-empty-state plate-marks h-[180px]" aria-hidden="true" />
            </div>
          )}
          <p className="figcap">
            <b>FIG. 8.1</b> Classes sorted by logit. The coral bar is the predicted class; hover for exact values.
          </p>
        </motion.div>
      </div>

      {prediction && topPrediction && (
        <Verdict
          letter={EMNIST_CLASSES[topPrediction.classIndex]}
          confidence={topPrediction.confidence}
          runnersUp={runnersUp}
          sum={prediction.reduce((s, p) => s + p, 0)}
        />
      )}
    </SectionWrapper>
  );
}
