"use client";

import { useMemo, useState, useRef, useEffect, useCallback } from "react";
import { SectionWrapper } from "@/components/ui/SectionWrapper";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { useInferenceStore } from "@/stores/inferenceStore";
import {
  EMNIST_CLASSES,
  BYMERGE_MERGED_INDICES,
} from "@/lib/model/classes";
import { Latex } from "@/components/ui/Latex";

/* ── Types ───────────────────────────────────────────────────────── */

interface ClassEntry {
  idx: number;
  label: string;
  logit: number;
  prob: number;
}

/* ── Logits → Probabilities dual bar chart ───────────────────────── */

const CHART_W = 300;
const CHART_H = 160;
const PAD = { top: 14, bottom: 20, left: 28, right: 8 };

function SoftmaxChart({
  entries,
  topIdx,
}: {
  entries: ClassEntry[];
  topIdx: number;
}) {
  const logitRef = useRef<HTMLCanvasElement>(null);
  const probRef = useRef<HTMLCanvasElement>(null);
  const [hoverI, setHoverI] = useState<number | null>(null);

  const plotW = CHART_W - PAD.left - PAD.right;
  const plotH = CHART_H - PAD.top - PAD.bottom;
  const barW = plotW / entries.length;

  // Logit range
  const { logitMin, logitMax, probMax } = useMemo(() => {
    let lMin = Infinity,
      lMax = -Infinity,
      pMax = 0;
    for (const e of entries) {
      if (e.logit < lMin) lMin = e.logit;
      if (e.logit > lMax) lMax = e.logit;
      if (e.prob > pMax) pMax = e.prob;
    }
    // Ensure zero is visible
    lMin = Math.min(lMin, 0);
    lMax = Math.max(lMax, 0);
    return { logitMin: lMin, logitMax: lMax, probMax: pMax };
  }, [entries]);

  // Draw logit chart
  useEffect(() => {
    const canvas = logitRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, CHART_W, CHART_H);

    const range = logitMax - logitMin || 1;
    const zeroY = PAD.top + ((logitMax - 0) / range) * plotH;

    // Zero line
    ctx.strokeStyle = "rgba(255,255,255,0.15)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PAD.left, zeroY);
    ctx.lineTo(PAD.left + plotW, zeroY);
    ctx.stroke();

    // Bars
    for (let i = 0; i < entries.length; i++) {
      const e = entries[i];
      const x = PAD.left + i * barW;
      const valY = PAD.top + ((logitMax - e.logit) / range) * plotH;
      const barH = Math.abs(valY - zeroY);
      const isTop = e.idx === topIdx;
      const isHover = i === hoverI;

      if (isTop) {
        ctx.fillStyle = "rgba(99, 102, 241, 0.8)";
      } else if (isHover) {
        ctx.fillStyle = "rgba(139, 92, 246, 0.6)";
      } else {
        ctx.fillStyle =
          e.logit >= 0
            ? "rgba(74, 222, 128, 0.3)"
            : "rgba(248, 113, 113, 0.25)";
      }

      ctx.fillRect(
        x + 0.5,
        e.logit >= 0 ? valY : zeroY,
        Math.max(barW - 1, 1),
        barH || 1
      );
    }

    // Hover highlight
    if (hoverI !== null) {
      const x = PAD.left + hoverI * barW;
      ctx.strokeStyle = "rgba(255,255,255,0.5)";
      ctx.lineWidth = 1;
      ctx.strokeRect(x, PAD.top, barW, plotH);
    }

    // Y axis labels
    ctx.fillStyle = "rgba(255,255,255,0.25)";
    ctx.font = "9px monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "top";
    ctx.fillText(logitMax.toFixed(1), PAD.left - 4, PAD.top);
    ctx.textBaseline = "bottom";
    ctx.fillText(logitMin.toFixed(1), PAD.left - 4, PAD.top + plotH);
    ctx.textBaseline = "middle";
    ctx.fillText("0", PAD.left - 4, zeroY);

    // Title
    ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.font = "10px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText("Raw logits (before softmax)", CHART_W / 2, 1);

    // X axis class labels for hover
    if (hoverI !== null) {
      const e = entries[hoverI];
      ctx.fillStyle = "rgba(255,255,255,0.6)";
      ctx.font = "bold 10px monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(e.label, PAD.left + hoverI * barW + barW / 2, PAD.top + plotH + 4);
    }
  }, [entries, topIdx, hoverI, logitMin, logitMax, plotW, plotH, barW]);

  // Draw probability chart
  useEffect(() => {
    const canvas = probRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, CHART_W, CHART_H);

    const pRange = probMax || 1;

    // Bars
    for (let i = 0; i < entries.length; i++) {
      const e = entries[i];
      const x = PAD.left + i * barW;
      const barH = (e.prob / pRange) * plotH;
      const isTop = e.idx === topIdx;
      const isHover = i === hoverI;

      if (isTop) {
        ctx.fillStyle = "rgba(99, 102, 241, 0.85)";
      } else if (isHover) {
        ctx.fillStyle = "rgba(139, 92, 246, 0.6)";
      } else {
        ctx.fillStyle =
          e.prob > 0.01
            ? "rgba(139, 92, 246, 0.3)"
            : "rgba(255, 255, 255, 0.08)";
      }

      ctx.fillRect(
        x + 0.5,
        PAD.top + plotH - barH,
        Math.max(barW - 1, 1),
        barH || 1
      );
    }

    // Hover highlight
    if (hoverI !== null) {
      const x = PAD.left + hoverI * barW;
      ctx.strokeStyle = "rgba(255,255,255,0.5)";
      ctx.lineWidth = 1;
      ctx.strokeRect(x, PAD.top, barW, plotH);
    }

    // Y axis
    ctx.fillStyle = "rgba(255,255,255,0.25)";
    ctx.font = "9px monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "top";
    ctx.fillText((pRange * 100).toFixed(0) + "%", PAD.left - 4, PAD.top);
    ctx.textBaseline = "bottom";
    ctx.fillText("0%", PAD.left - 4, PAD.top + plotH);

    // Title
    ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.font = "10px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText("Probabilities (after softmax)", CHART_W / 2, 1);

    // X axis class label for hover
    if (hoverI !== null) {
      const e = entries[hoverI];
      ctx.fillStyle = "rgba(255,255,255,0.6)";
      ctx.font = "bold 10px monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(e.label, PAD.left + hoverI * barW + barW / 2, PAD.top + plotH + 4);
    }
  }, [entries, topIdx, hoverI, probMax, plotW, plotH, barW]);

  // Shared hover handler
  const handleMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const rect = e.currentTarget.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * CHART_W - PAD.left;
      const i = Math.floor(x / barW);
      setHoverI(i >= 0 && i < entries.length ? i : null);
    },
    [barW, entries.length]
  );

  const clearHover = useCallback(() => setHoverI(null), []);

  const hoveredEntry = hoverI !== null ? entries[hoverI] : null;

  return (
    <div className="flex flex-col items-center gap-3">
      {/* Logit chart */}
      <canvas
        ref={logitRef}
        width={CHART_W}
        height={CHART_H}
        className="cursor-crosshair rounded-md border border-border/60"
        style={{ width: CHART_W, height: CHART_H }}
        onMouseMove={handleMove}
        onMouseLeave={clearHover}
      />

      {/* Arrow */}
      <Latex
        math="\downarrow\;\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum e^{z_j}}"
        className="text-foreground/30"
      />

      {/* Probability chart */}
      <canvas
        ref={probRef}
        width={CHART_W}
        height={CHART_H}
        className="cursor-crosshair rounded-md border border-border/60"
        style={{ width: CHART_W, height: CHART_H }}
        onMouseMove={handleMove}
        onMouseLeave={clearHover}
      />

      {/* Hover info */}
      <div className="h-5 text-center font-mono text-xs text-foreground/50">
        {hoveredEntry ? (
          <>
            <span className="text-accent-primary font-bold">
              {hoveredEntry.label}
            </span>
            {" logit "}
            <span className={hoveredEntry.logit >= 0 ? "text-green-400" : "text-red-400"}>
              {hoveredEntry.logit.toFixed(2)}
            </span>
            {" → prob "}
            <span className="text-accent-secondary">
              {(hoveredEntry.prob * 100).toFixed(2)}%
            </span>
          </>
        ) : (
          <span className="text-foreground/25">
            Hover to compare logit vs probability
          </span>
        )}
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
    for (let i = 0; i < 62; i++) {
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
        subtitle="The final layer produces 62 raw scores (logits), one for each character class. Softmax converts these into probabilities that sum to 1.0 — transforming 'how much does this look like each character?' into 'what's the probability it IS each character?'"
      />

      <div className="flex flex-col items-center gap-8 lg:flex-row lg:items-start lg:gap-12">
        {/* Left: theory text */}
        <div className="flex-1 space-y-4 text-center lg:text-left">
          <p className="text-base leading-relaxed text-foreground/65 sm:text-lg">
            The dense layer outputs 62 raw <em>logits</em> — unbounded real
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
              <Latex math="\mathbf{z}" /> — raw logits (62 values)
            </span>
            <span>
              <Latex math="K = 62" /> — number of classes
            </span>
            <span>
              <Latex math="\sigma_i" /> — probability of class{" "}
              <Latex math="i" />
            </span>
          </div>

          <p className="text-sm leading-relaxed text-foreground/45">
            In practice we subtract the max logit first for numerical stability:{" "}
            <Latex math="e^{z_i - z_{\max}}" /> instead of{" "}
            <Latex math="e^{z_i}" />. Of the 62 output neurons, 15 are masked
            (the ByMerge dataset merges ambiguous pairs like C/c and O/o),
            leaving 47 valid character classes: digits 0–9, uppercase A–Z, and
            select lowercase letters.
          </p>
        </div>

        {/* Right: before/after softmax charts */}
        <div className="flex w-full shrink-0 flex-col items-center gap-4 lg:w-auto">
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
            47
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
