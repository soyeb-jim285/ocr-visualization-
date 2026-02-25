"use client";

import { useMemo, useState, useRef, useEffect, useCallback } from "react";
import { SectionWrapper } from "@/components/ui/SectionWrapper";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { useInferenceStore } from "@/stores/inferenceStore";
import { Latex } from "@/components/ui/Latex";
import { viridis } from "@/lib/network/networkConstants";

/* ── Constants ───────────────────────────────────────────────────── */

const GRID = 16; // 16×16 = 256 neurons
const CELL = 16; // px per cell
const GRID_PX = GRID * CELL; // 256px
const STRIP_H = 14;

/* ── Flattened input strip ───────────────────────────────────────── */

function InputStrip({ pool2Maps }: { pool2Maps: number[][][] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const flatValues = useMemo(() => {
    const vals: number[] = [];
    for (const ch of pool2Maps) for (const row of ch) for (const v of row) vals.push(v);
    return vals;
  }, [pool2Maps]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    const w = GRID_PX;

    let mn = Infinity, mx = -Infinity;
    for (const v of flatValues) { if (v < mn) mn = v; if (v > mx) mx = v; }
    const range = mx - mn || 1;

    const img = ctx.createImageData(w, STRIP_H);
    const px = img.data;
    for (let x = 0; x < w; x++) {
      // Sample from the flat array
      const idx = Math.floor((x / w) * flatValues.length);
      const t = (flatValues[idx] - mn) / range;
      const [r, g, b] = viridis(t);
      for (let y = 0; y < STRIP_H; y++) {
        const i = (y * w + x) * 4;
        px[i] = r; px[i + 1] = g; px[i + 2] = b; px[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [flatValues]);

  return (
    <div className="flex flex-col items-center gap-1">
      <span className="text-[11px] text-foreground/30">
        Flattened input (6,272 values)
      </span>
      <canvas
        ref={canvasRef}
        width={GRID_PX}
        height={STRIP_H}
        className="rounded-sm border border-border/40"
        style={{ width: GRID_PX, height: STRIP_H }}
      />
    </div>
  );
}

/* ── 16×16 neuron activation grid ────────────────────────────────── */

function NeuronGrid({ activations }: { activations: number[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<{ idx: number; row: number; col: number } | null>(null);

  const { min, max, active, topNeurons } = useMemo(() => {
    let mn = Infinity, mx = -Infinity;
    let active = 0;
    for (const v of activations) {
      if (v < mn) mn = v;
      if (v > mx) mx = v;
      if (v > 0) active++;
    }
    // Top 5 most active neurons
    const indexed = activations.map((v, i) => ({ v, i }));
    indexed.sort((a, b) => b.v - a.v);
    const topNeurons = indexed.slice(0, 5);
    return { min: mn, max: mx, active, topNeurons };
  }, [activations]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    const range = max - min || 1;

    // Draw cells
    for (let r = 0; r < GRID; r++) {
      for (let c = 0; c < GRID; c++) {
        const idx = r * GRID + c;
        const v = activations[idx] ?? 0;
        const t = (v - min) / range;
        const [red, green, blue] = viridis(t);
        ctx.fillStyle = `rgb(${red},${green},${blue})`;
        ctx.fillRect(c * CELL, r * CELL, CELL, CELL);

        // Subtle grid line
        ctx.strokeStyle = "rgba(0,0,0,0.15)";
        ctx.lineWidth = 0.5;
        ctx.strokeRect(c * CELL, r * CELL, CELL, CELL);
      }
    }

    // Hover highlight
    if (hover) {
      const { row, col } = hover;
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.strokeRect(col * CELL + 1, row * CELL + 1, CELL - 2, CELL - 2);
    }

    // Highlight top 5 neurons with small corner marks
    for (let i = 0; i < topNeurons.length; i++) {
      const { i: idx } = topNeurons[i];
      const r = Math.floor(idx / GRID), c = idx % GRID;
      const isHovered = hover?.idx === idx;
      if (isHovered) continue;
      ctx.strokeStyle = i === 0 ? "#6366f1" : "rgba(99,102,241,0.4)";
      ctx.lineWidth = i === 0 ? 2 : 1;
      ctx.strokeRect(c * CELL + 1, r * CELL + 1, CELL - 2, CELL - 2);
    }
  }, [activations, min, max, hover, topNeurons]);

  const handleMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const col = Math.min(GRID - 1, Math.max(0, Math.floor(((e.clientX - rect.left) / rect.width) * GRID)));
    const row = Math.min(GRID - 1, Math.max(0, Math.floor(((e.clientY - rect.top) / rect.height) * GRID)));
    setHover({ idx: row * GRID + col, row, col });
  }, []);

  const sparsity = ((1 - active / 256) * 100).toFixed(1);

  return (
    <div className="flex flex-col items-center gap-3">
      {/* Grid */}
      <div className="flex flex-col items-center gap-1">
        <span className="text-[11px] text-foreground/30">
          Hidden layer — 256 neurons (16&times;16)
        </span>
        <canvas
          ref={canvasRef}
          width={GRID_PX}
          height={GRID_PX}
          className="cursor-crosshair rounded-md border border-border/60"
          style={{ width: GRID_PX, height: GRID_PX, imageRendering: "pixelated" }}
          onMouseMove={handleMove}
          onMouseLeave={() => setHover(null)}
        />
      </div>

      {/* Hover info */}
      <div className="h-5 text-center font-mono text-xs text-foreground/50">
        {hover ? (
          <>
            neuron <span className="text-accent-primary">#{hover.idx}</span>
            {" = "}
            <span className={activations[hover.idx] > 0 ? "text-green-400" : "text-red-400"}>
              {activations[hover.idx].toFixed(4)}
            </span>
          </>
        ) : (
          <span className="text-foreground/25">Hover to inspect neurons</span>
        )}
      </div>

      {/* Top neurons + stats */}
      <div className="flex flex-col items-center gap-2">
        <div className="flex items-center gap-4 text-sm">
          <span className="font-mono font-semibold text-green-400">{active}</span>
          <span className="text-foreground/40">active</span>
          <span className="text-foreground/15">|</span>
          <span className="font-mono font-semibold text-foreground/60">{sparsity}%</span>
          <span className="text-foreground/40">sparse</span>
          <span className="text-foreground/15">|</span>
          <span className="font-mono font-semibold text-accent-primary">1.6M</span>
          <span className="text-foreground/40">params</span>
        </div>

        {/* Top 5 neurons */}
        <div className="flex flex-wrap justify-center gap-x-3 gap-y-0.5 font-mono text-[11px] text-foreground/35">
          <span className="text-foreground/20">top:</span>
          {topNeurons.map(({ v, i }, rank) => (
            <span key={i} className={rank === 0 ? "text-accent-primary" : ""}>
              #{i}
              <span className="text-foreground/20">=</span>
              {v.toFixed(2)}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ── Main section ────────────────────────────────────────────────── */

export function FullyConnectedSection() {
  const layerActivations = useInferenceStore((s) => s.layerActivations);
  const relu4 = layerActivations["relu4"] as number[] | undefined;
  const pool2Maps = layerActivations["pool2"] as number[][][] | undefined;

  const hasData = !!relu4 && !!pool2Maps;

  return (
    <SectionWrapper id="fully-connected">
      <SectionHeader
        step={7}
        title="Making Decisions: Dense Layers"
        subtitle="The spatial features are flattened into a single vector of 6,272 values, then compressed to 256 neurons. Each neuron is connected to every input — it sees the entire character at once. The network is now making decisions about what character this is."
      />

      <div className="flex flex-col items-center gap-8 lg:flex-row lg:items-start lg:gap-12">
        {/* Left: theory text */}
        <div className="flex-1 space-y-4 text-center lg:text-left">
          <p className="text-base leading-relaxed text-foreground/65 sm:text-lg">
            Convolutional layers extract <em>where</em> features are. Dense
            layers decide <em>what</em> they mean. First, the 128 feature maps
            of size 7&times;7 are flattened into a single vector of 6,272
            values. Then a fully-connected layer maps this to 256 neurons —
            every output is a weighted sum of all 6,272 inputs plus a bias,
            followed by ReLU.
          </p>

          {/* Main equation */}
          <div className="py-3">
            <Latex
              display
              math="\mathbf{h} = \text{ReLU}\!\left(\,W\,\mathbf{x} + \mathbf{b}\,\right)"
            />
          </div>

          {/* Equation legend */}
          <div className="flex flex-wrap justify-center gap-x-6 gap-y-1 text-sm text-foreground/40 lg:justify-start">
            <span><Latex math="\mathbf{x}" /> — flattened input (6,272)</span>
            <span><Latex math="W" /> — weight matrix</span>
            <span><Latex math="\mathbf{b}" /> — bias vector</span>
            <span><Latex math="\mathbf{h}" /> — hidden activations (256)</span>
          </div>

          <p className="text-sm leading-relaxed text-foreground/45">
            The weight matrix <Latex math="W" /> has shape{" "}
            <Latex math="256 \times 6{,}272" />, giving{" "}
            <Latex math="256 \times 6{,}272 + 256 = 1{,}605{,}888" /> learnable
            parameters — far more than all convolutional layers combined. This
            is where most of the model&apos;s capacity lives. After ReLU,
            many neurons are zeroed out — the network has learned which
            abstract features matter for each character. A second dense layer
            then maps the 256 hidden units to the 62 output logits:{" "}
            <Latex math="(256) \xrightarrow{W_2} (62)" />.
          </p>
        </div>

        {/* Right: visualization */}
        <div className="flex w-full shrink-0 flex-col items-center gap-4 lg:w-auto">
          {hasData ? (
            <div className="flex flex-col items-center gap-4">
              <InputStrip pool2Maps={pool2Maps} />

              {/* Arrow: flatten + dense */}
              <div className="flex flex-col items-center gap-0.5">
                <Latex
                  math="\downarrow\; W \cdot \mathbf{x} + \mathbf{b}"
                  className="text-foreground/30"
                />
              </div>

              <NeuronGrid activations={relu4} />
            </div>
          ) : (
            <div className="flex h-48 items-center justify-center">
              <p className="text-foreground/30">
                Draw a character above to see activations
              </p>
            </div>
          )}
        </div>
      </div>
    </SectionWrapper>
  );
}
