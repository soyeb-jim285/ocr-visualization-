"use client";

import { useMemo, useState, useRef, useEffect, useCallback } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { SectionWrapper } from "@/components/ui/SectionWrapper";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { useInferenceStore } from "@/stores/inferenceStore";
import { Latex } from "@/components/ui/Latex";
import { viridis } from "@/lib/network/networkConstants";
import { BG_INSET, INK, SIG } from "@/lib/theme";

/* ── Constants ───────────────────────────────────────────────────── */

const GRID_COLS = 32; // 32×16 = 512 neurons
const GRID_ROWS = 16;
const CELL = 16; // px per cell (canvas scales to container width)
const GRID_W = GRID_COLS * CELL; // 512px
const GRID_H = GRID_ROWS * CELL; // 256px
const STRIP_W = 512;
const STRIP_H = 20;

/* ── Reveal helper ───────────────────────────────────────────────── */

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

    let mn = Infinity, mx = -Infinity;
    for (const v of flatValues) { if (v < mn) mn = v; if (v > mx) mx = v; }
    const range = mx - mn || 1;

    const w = STRIP_W;
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
    <canvas
      ref={canvasRef}
      width={STRIP_W}
      height={STRIP_H}
      className="block w-full"
      style={{ aspectRatio: `${STRIP_W} / ${STRIP_H}`, imageRendering: "pixelated" }}
      aria-label="Flattened input vector of 12,544 values"
    />
  );
}

/* ── 32×16 neuron activation grid ────────────────────────────────── */

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
    for (let r = 0; r < GRID_ROWS; r++) {
      for (let c = 0; c < GRID_COLS; c++) {
        const idx = r * GRID_COLS + c;
        const v = activations[idx] ?? 0;
        const t = (v - min) / range;
        const [red, green, blue] = viridis(t);
        ctx.fillStyle = v <= 0 ? BG_INSET : `rgb(${red},${green},${blue})`;
        ctx.fillRect(c * CELL, r * CELL, CELL, CELL);

        // Subtle grid line
        ctx.strokeStyle = "rgba(3,5,7,0.45)";
        ctx.lineWidth = 1;
        ctx.strokeRect(c * CELL + 0.5, r * CELL + 0.5, CELL - 1, CELL - 1);
      }
    }

    // Hover highlight
    if (hover) {
      const { row, col } = hover;
      ctx.strokeStyle = INK;
      ctx.lineWidth = 2;
      ctx.strokeRect(col * CELL + 1, row * CELL + 1, CELL - 2, CELL - 2);
    }

    // Highlight top 5 neurons in the dense signal color
    for (let i = 0; i < topNeurons.length; i++) {
      const { i: idx } = topNeurons[i];
      const r = Math.floor(idx / GRID_COLS), c = idx % GRID_COLS;
      if (hover?.idx === idx) continue;
      ctx.strokeStyle = SIG.dense;
      ctx.lineWidth = i === 0 ? 2.5 : 1.5;
      ctx.strokeRect(c * CELL + 1, r * CELL + 1, CELL - 2, CELL - 2);
      ctx.strokeStyle = "rgba(3,5,7,.9)";
      ctx.lineWidth = 1;
      ctx.strokeRect(c * CELL + 3, r * CELL + 3, CELL - 6, CELL - 6);
    }
  }, [activations, min, max, hover, topNeurons]);

  const handleMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const col = Math.min(GRID_COLS - 1, Math.max(0, Math.floor(((e.clientX - rect.left) / rect.width) * GRID_COLS)));
    const row = Math.min(GRID_ROWS - 1, Math.max(0, Math.floor(((e.clientY - rect.top) / rect.height) * GRID_ROWS)));
    setHover({ idx: row * GRID_COLS + col, row, col });
  }, []);

  const sparsity = ((1 - active / 512) * 100).toFixed(1);

  return (
    <div className="flex flex-col gap-4">
      <canvas
        ref={canvasRef}
        width={GRID_W}
        height={GRID_H}
        className="block w-full cursor-crosshair"
        style={{ aspectRatio: `${GRID_W} / ${GRID_H}`, imageRendering: "pixelated" }}
        onMouseMove={handleMove}
        onMouseLeave={() => setHover(null)}
        aria-label="Hidden layer activations, 512 neurons"
      />

      {/* Hover readout */}
      <div className="readout h-5 px-1 text-ink-2">
        {hover ? (
          <>
            neuron <span className="text-sig">#{hover.idx}</span>
            {" = "}
            <span className="text-ink">{activations[hover.idx].toFixed(4)}</span>
            {activations[hover.idx] <= 0 && <span className="text-ink-3"> · off</span>}
          </>
        ) : (
          <span className="text-ink-2">Hover a cell to inspect a neuron</span>
        )}
      </div>

      {/* Stats */}
      <dl className="grid grid-cols-3 gap-4 border-t border-rule pt-4">
        {[
          { v: String(active), l: "neurons firing", c: "text-sig" },
          { v: `${sparsity}%`, l: "zeroed by ReLU", c: "text-ink" },
          { v: "6.4M", l: "parameters", c: "text-ink" },
        ].map((s) => (
          <div key={s.l}>
            <dd className={`font-serif text-3xl font-light leading-none tabular-nums sm:text-4xl ${s.c}`}>{s.v}</dd>
            <dt className="caption mt-2">{s.l}</dt>
          </div>
        ))}
      </dl>

      {/* Top neurons */}
      <div>
        <p className="caption mb-2">TOP 5</p>
        <div className="grid grid-cols-5 gap-2">
          {topNeurons.map(({ v, i }, rank) => (
            <div key={i} className={`readout ${rank === 0 ? "text-sig" : "text-ink-2"}`}>
              <div>#{i}</div>
              <div className="text-ink-3">{v.toFixed(2)}</div>
            </div>
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
  const textReveal = useReveal(0);
  const figReveal = useReveal(0.16);

  return (
    <SectionWrapper id="fully-connected" sig="dense" mirror>
      <SectionHeader
        step={7}
        tag="Dense · 12,544 → 512"
        title="Making Decisions: Dense Layers"
        subtitle="The spatial features are flattened into a single vector of 12,544 values, then compressed to 512 neurons. Each neuron is connected to every input — it sees the entire character at once. The network is now making decisions about what character this is."
      />

      <div className="grid gap-x-6 gap-y-14 lg:grid-cols-12">
        {/* Figure (left on desktop, mirrored plate) */}
        <motion.div className="figure order-2 lg:col-span-7 lg:order-1" {...figReveal}>
          <p className="caption mb-2 flex items-baseline justify-between gap-4">
            <span>FLATTENED INPUT</span>
            <span>12,544 values</span>
          </p>
          <div className="well plate-marks p-1.5">
            {hasData ? (
              <InputStrip pool2Maps={pool2Maps} />
            ) : (
              <div className="w-full bg-bg-inset" style={{ aspectRatio: `${STRIP_W} / ${STRIP_H}` }} />
            )}
          </div>

          <div className="my-5 flex items-center gap-4 text-ink-3">
            <span className="h-px flex-1 bg-rule" />
            <Latex math="\downarrow\; W \cdot \mathbf{x} + \mathbf{b}" className="text-ink-2" />
            <span className="h-px flex-1 bg-rule" />
          </div>

          <p className="caption mb-2 flex items-baseline justify-between gap-4">
            <span>HIDDEN LAYER · 512 NEURONS</span>
            <span>32×16</span>
          </p>
          <div className="well plate-marks p-1.5">
            {hasData ? (
              <NeuronGrid activations={relu4} />
            ) : (
              <div className="viz-empty-state" style={{ aspectRatio: `${GRID_W} / ${GRID_H}` }}>
                Draw something to light this up
              </div>
            )}
          </div>
          <p className="figcap">
            <b>FIG. 7.1</b> Each cell is one neuron after ReLU. Dark cells are off; amber outlines mark the five strongest.
          </p>
        </motion.div>

        {/* Theory text */}
        <motion.div className="order-1 space-y-6 lg:col-span-5 lg:order-2" {...textReveal}>
          <p className="prose-body">
            Convolutional layers extract <em>where</em> features are. Dense
            layers decide <em>what</em> they mean. First, the 256 feature maps
            of size 7&times;7 are flattened into a single vector of 12,544
            values. Then a fully-connected layer maps this to 512 neurons —
            every output is a weighted sum of all 12,544 inputs plus a bias,
            followed by ReLU.
          </p>

          <div className="formula">
            <Latex
              display
              math="\mathbf{h} = \text{ReLU}\!\left(\,W\,\mathbf{x} + \mathbf{b}\,\right)"
            />
            <span className="eq-no">(7)</span>
          </div>

          <ul className="caption space-y-1.5">
            <li><Latex math="\mathbf{x}" /> — flattened input (12,544)</li>
            <li><Latex math="W" /> — weight matrix</li>
            <li><Latex math="\mathbf{b}" /> — bias vector</li>
            <li><Latex math="\mathbf{h}" /> — hidden activations (512)</li>
          </ul>

          <p className="prose-body text-[0.9375rem] text-ink-3">
            The weight matrix <Latex math="W" /> has shape{" "}
            <Latex math="512 \times 12{,}544" />, giving{" "}
            <Latex math="512 \times 12{,}544 + 512 = 6{,}423{,}040" /> learnable
            parameters — far more than all convolutional layers combined. This
            is where most of the model&apos;s capacity lives. After ReLU,
            many neurons are zeroed out — the network has learned which
            abstract features matter for each character. A second dense layer
            then maps the 512 hidden units to the 146 output logits:{" "}
            <Latex math="(512) \xrightarrow{W_2} (146)" />.
          </p>
        </motion.div>
      </div>
    </SectionWrapper>
  );
}
