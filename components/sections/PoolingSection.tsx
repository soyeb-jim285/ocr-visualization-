"use client";

import {
  useMemo,
  useState,
  useRef,
  useEffect,
  useCallback,
} from "react";
import { SectionWrapper } from "@/components/ui/SectionWrapper";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { ActivationHeatmap } from "@/components/visualizations/ActivationHeatmap";
import { useInferenceStore } from "@/stores/inferenceStore";
import { Latex } from "@/components/ui/Latex";
import { viridis } from "@/lib/network/networkConstants";
import { Reveal } from "@/components/visualizations/FeatureMapGrid";
import { PHOSPHOR, ANNOTATION } from "@/lib/theme";

/* ── Constants ───────────────────────────────────────────────────── */

const BEFORE_GRID = 28;
const AFTER_GRID = 14;
const CELL_B = 5; // 28 * 5 = 140
const CELL_A = 10; // 14 * 10 = 140
const CANVAS_SIZE = 140;

/* ── Interactive pooling heatmap pair ────────────────────────────── */

function PoolingViz({
  beforeData,
  afterData,
}: {
  beforeData: number[][];
  afterData: number[][];
}) {
  const beforeRef = useRef<HTMLCanvasElement>(null);
  const afterRef = useRef<HTMLCanvasElement>(null);
  const [hoverPool, setHoverPool] = useState<{
    row: number;
    col: number;
  } | null>(null);

  // Min/max for viridis normalization
  const bRange = useMemo(() => {
    let mn = Infinity,
      mx = -Infinity;
    for (const row of beforeData)
      for (const v of row) {
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
    return { min: mn, max: mx };
  }, [beforeData]);

  const aRange = useMemo(() => {
    let mn = Infinity,
      mx = -Infinity;
    for (const row of afterData)
      for (const v of row) {
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
    return { min: mn, max: mx };
  }, [afterData]);

  // Draw before-pooling canvas
  useEffect(() => {
    const canvas = beforeRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    const { min, max } = bRange;
    const range = max - min || 1;

    // Draw heatmap cells
    for (let r = 0; r < BEFORE_GRID; r++) {
      for (let c = 0; c < BEFORE_GRID; c++) {
        const t = (beforeData[r][c] - min) / range;
        const [red, green, blue] = viridis(t);
        ctx.fillStyle = `rgb(${red},${green},${blue})`;
        ctx.fillRect(c * CELL_B, r * CELL_B, CELL_B, CELL_B);
      }
    }

    // Hover overlay: highlight 2×2 region
    if (hoverPool) {
      const { row, col } = hoverPool;
      const x = col * 2 * CELL_B;
      const y = row * 2 * CELL_B;
      const size = 2 * CELL_B;

      ctx.globalAlpha = 0.2;
      ctx.fillStyle = PHOSPHOR;
      ctx.fillRect(x, y, size, size);
      ctx.globalAlpha = 1;
      ctx.strokeStyle = PHOSPHOR;
      ctx.lineWidth = 2;
      ctx.strokeRect(x + 1, y + 1, size - 2, size - 2);
    }
  }, [beforeData, bRange, hoverPool]);

  // Draw after-pooling canvas
  useEffect(() => {
    const canvas = afterRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    const { min, max } = aRange;
    const range = max - min || 1;

    // Draw heatmap cells
    for (let r = 0; r < AFTER_GRID; r++) {
      for (let c = 0; c < AFTER_GRID; c++) {
        const t = (afterData[r][c] - min) / range;
        const [red, green, blue] = viridis(t);
        ctx.fillStyle = `rgb(${red},${green},${blue})`;
        ctx.fillRect(c * CELL_A, r * CELL_A, CELL_A, CELL_A);
      }
    }

    // Hover overlay: highlight output cell
    if (hoverPool) {
      const { row, col } = hoverPool;
      const x = col * CELL_A;
      const y = row * CELL_A;

      ctx.globalAlpha = 0.25;
      ctx.fillStyle = ANNOTATION;
      ctx.fillRect(x, y, CELL_A, CELL_A);
      ctx.globalAlpha = 1;
      ctx.strokeStyle = ANNOTATION;
      ctx.lineWidth = 2;
      ctx.strokeRect(x + 1, y + 1, CELL_A - 2, CELL_A - 2);
    }
  }, [afterData, aRange, hoverPool]);

  // Extract 2×2 values for the pool grid
  const poolValues = useMemo(() => {
    if (!hoverPool) return null;
    const { row, col } = hoverPool;
    const r = row * 2,
      c = col * 2;
    return [
      beforeData[r]?.[c] ?? 0,
      beforeData[r]?.[c + 1] ?? 0,
      beforeData[r + 1]?.[c] ?? 0,
      beforeData[r + 1]?.[c + 1] ?? 0,
    ];
  }, [hoverPool, beforeData]);

  const maxIdx = useMemo(() => {
    if (!poolValues) return -1;
    let mi = 0;
    for (let i = 1; i < 4; i++) if (poolValues[i] > poolValues[mi]) mi = i;
    return mi;
  }, [poolValues]);

  // Mouse handlers
  const handleBeforeMove = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      const rect = e.currentTarget.getBoundingClientRect();
      const col = Math.min(
        AFTER_GRID - 1,
        Math.floor(
          ((e.clientX - rect.left) / rect.width) * BEFORE_GRID * 0.5
        )
      );
      const row = Math.min(
        AFTER_GRID - 1,
        Math.floor(
          ((e.clientY - rect.top) / rect.height) * BEFORE_GRID * 0.5
        )
      );
      setHoverPool({ row, col });
    },
    []
  );

  const handleAfterMove = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      const rect = e.currentTarget.getBoundingClientRect();
      const col = Math.min(
        AFTER_GRID - 1,
        Math.max(
          0,
          Math.floor(((e.clientX - rect.left) / rect.width) * AFTER_GRID)
        )
      );
      const row = Math.min(
        AFTER_GRID - 1,
        Math.max(
          0,
          Math.floor(((e.clientY - rect.top) / rect.height) * AFTER_GRID)
        )
      );
      setHoverPool({ row, col });
    },
    []
  );

  // touch keeps the highlight after the finger lifts
  const clearHover = useCallback(
    (e: React.PointerEvent) => e.pointerType !== "touch" && setHoverPool(null),
    [],
  );

  // Display values for the 4-cell grid
  const displayValues = poolValues ?? [0.3, 0.7, 0.1, 0.9];
  const displayMaxIdx = poolValues ? maxIdx : 3;
  const isLive = !!poolValues;

  return (
    <div className="flex flex-wrap items-center justify-center gap-4 sm:flex-nowrap sm:gap-6">
      {/* Before pooling */}
      <div className="flex flex-col items-center gap-2">
        <span className="font-mono text-[11px] tracking-[0.08em] text-sig">BEFORE</span>
        <div className="well plate-marks p-1.5">
          <canvas
            ref={beforeRef}
            width={CANVAS_SIZE}
            height={CANVAS_SIZE}
            className="block cursor-crosshair touch-pan-y"
            style={{ width: "min(140px, 38vw)", height: "min(140px, 38vw)", imageRendering: "pixelated" }}
            onPointerMove={handleBeforeMove}
            onPointerDown={handleBeforeMove}
            onPointerLeave={clearHover}
          />
        </div>
        <span className="caption min-h-[3em] whitespace-nowrap sm:min-h-[1.5em]">
          {BEFORE_GRID}&times;{BEFORE_GRID}
          {hoverPool && (
            <span className="text-phosphor">
              {" "}
              [{hoverPool.row * 2}:{hoverPool.row * 2 + 1}, {hoverPool.col * 2}
              :{hoverPool.col * 2 + 1}]
            </span>
          )}
        </span>
      </div>

      {/* 2×2 pool grid with live values */}
      <div className="order-last flex basis-full flex-col items-center gap-2 sm:order-none sm:basis-auto">
        <div
          className={`grid grid-cols-2 gap-px border p-1 transition-colors duration-150 ${
            isLive ? "border-phosphor/60" : "border-rule"
          }`}
        >
          {displayValues.map((v, i) => (
            <div
              key={i}
              className={`flex h-10 w-12 items-center justify-center font-mono text-xs transition-colors duration-150 ${
                i === displayMaxIdx
                  ? "bg-sig font-medium text-bg"
                  : "bg-bg-lift text-ink-3"
              }`}
            >
              {isLive ? v.toFixed(2) : v}
            </div>
          ))}
        </div>
        <Latex
          math="\xrightarrow{\max}"
          className="hidden text-ink-3 sm:block"
        />
        <Latex math="\uparrow" className="text-ink-3 sm:hidden" />
      </div>

      {/* After pooling */}
      <div className="flex flex-col items-center gap-2">
        <span className="font-mono text-[11px] tracking-[0.08em] text-sig">AFTER</span>
        <div className="well plate-marks p-1.5">
          <canvas
            ref={afterRef}
            width={CANVAS_SIZE}
            height={CANVAS_SIZE}
            className="block cursor-crosshair touch-pan-y"
            style={{ width: "min(140px, 38vw)", height: "min(140px, 38vw)", imageRendering: "pixelated" }}
            onPointerMove={handleAfterMove}
            onPointerDown={handleAfterMove}
            onPointerLeave={clearHover}
          />
        </div>
        <span className="caption min-h-[3em] whitespace-nowrap sm:min-h-[1.5em]">
          {AFTER_GRID}&times;{AFTER_GRID}
          {hoverPool && (
            <span className="text-annotation">
              {" "}
              [{hoverPool.row}, {hoverPool.col}]
            </span>
          )}
        </span>
      </div>
    </div>
  );
}

/* ── Main section ────────────────────────────────────────────────── */

export function PoolingSection() {
  const layerActivations = useInferenceStore((s) => s.layerActivations);
  const [picked, setPicked] = useState<{ maps: unknown; i: number } | null>(null);
  const liveMaps = layerActivations["relu2"] as number[][][] | undefined;
  // default to the highest-energy channel; a manual pick holds only for the current drawing
  const bestFilter = useMemo(() => {
    let best = 0, bestE = -1;
    liveMaps?.forEach((fm, i) => {
      let e = 0;
      for (const r of fm) for (const v of r) e += v > 0 ? v : 0;
      if (e > bestE) { bestE = e; best = i; }
    });
    return best;
  }, [liveMaps]);
  const selectedFilter = picked && picked.maps === liveMaps ? picked.i : bestFilter;
  const previewRef = useRef<HTMLDivElement>(null);
  const setSelectedFilter = (i: number) => {
    setPicked({ maps: liveMaps, i });
    // phones: preview sits a screen above the picker, so bring it into view
    if (window.matchMedia("(max-width: 639px)").matches)
      previewRef.current?.scrollIntoView({
        behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
        block: "center",
      });
  };

  // Before pooling (relu2 = 28x28x64) and after pooling (pool1 = 14x14x64)
  const relu2Maps = layerActivations["relu2"] as number[][][] | undefined;
  const pool1Maps = layerActivations["pool1"] as number[][][] | undefined;

  const beforePool = relu2Maps?.[selectedFilter];
  const afterPool = pool1Maps?.[selectedFilter];

  const numFilters = relu2Maps?.length ?? 64;
  const hasData = !!beforePool;

  const stats = useMemo(() => {
    if (!beforePool || !afterPool) return null;
    const beforeSize = beforePool.length * (beforePool[0]?.length ?? 0);
    const afterSize = afterPool.length * (afterPool[0]?.length ?? 0);
    return { beforeSize, afterSize };
  }, [beforePool, afterPool]);

  return (
    <SectionWrapper id="pooling" sig="pool">
      <SectionHeader
        step={5}
        tag="Pool1 · 64 ch · 28×28 → 14×14"
        title="Compressing Information: Max Pooling"
        subtitle="Max pooling slides a 2×2 window across each feature map and keeps only the maximum value. This halves the spatial dimensions while retaining the strongest activations, making the model more efficient and somewhat invariant to small shifts in position."
      />

      <div className="grid grid-cols-1 gap-x-6 gap-y-12 lg:grid-cols-12">
        {/* Left: theory text */}
        <Reveal className="min-w-0 space-y-5 lg:col-span-9 lg:col-start-4">
          <p className="prose-body">
            After activation, the feature maps still carry full spatial
            resolution. <em className="text-ink">Max pooling</em> compresses each map by partitioning
            it into non-overlapping 2&times;2 regions and keeping only the
            largest value from each. This achieves two goals: it reduces the
            number of parameters downstream (preventing overfitting) and
            introduces <em className="text-ink">translation invariance</em>: small shifts in the
            input produce identical pooled outputs.
          </p>

          <div className="formula !text-[0.9em] [scrollbar-width:thin] [scrollbar-color:var(--rule-strong)_transparent]">
            <Latex
              display
              math="P(i,j) = \max_{(m,n)\,\in\,R_{i,j}} A(m,n)"
            />
            <span className="eq-no">(5)</span>
          </div>

          <ul className="space-y-1 text-sm text-ink-3">
            <li><Latex math="A" /> input activation map</li>
            <li><Latex math="R_{i,j}" /> 2&times;2 pooling region</li>
            <li><Latex math="P" /> pooled output</li>
          </ul>

          <p className="prose-body !text-sm">
            With stride 2, each 2&times;2 window produces one output value,
            halving both dimensions:{" "}
            <Latex math="(64, 28, 28) \xrightarrow{2{\times}2\;\text{max pool}} (64, 14, 14)" />
            . This is a 75% reduction in spatial size, from{" "}
            <Latex math="28^2 = 784" /> to <Latex math="14^2 = 196" /> values
            per channel, with zero learnable parameters. The operation is
            purely structural: no weights, no bias, just a hard{" "}
            <Latex math="\max" />.
          </p>
        </Reveal>

        {/* Right: interactive visualization */}
        <Reveal delay={0.16} className="min-w-0 lg:col-span-9 lg:col-start-4">
          <div className="figure" ref={previewRef}>
            <div className="flex justify-center pb-1 sm:overflow-x-auto">
              {hasData && afterPool ? (
                <PoolingViz beforeData={beforePool} afterData={afterPool} />
              ) : (
                <div className="grid grid-cols-2 justify-items-center gap-4 sm:flex sm:gap-6">
                  {[["BEFORE", "28×28"], ["AFTER", "14×14"]].map(([t, d]) => (
                    <div key={t} className="flex flex-col items-center gap-2">
                      <span className="font-mono text-[11px] tracking-[0.08em] text-sig">{t}</span>
                      <div className="viz-empty-state aspect-square !min-h-0 w-[min(152px,40vw)] text-center text-xs sm:w-[152px]">
                        <span className="px-3 font-serif italic">Draw something to light this up</span>
                      </div>
                      <span className="caption">{d}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {stats && (
              <dl className="mt-6 grid grid-cols-3 border-y border-rule py-3 text-center">
                <div>
                  <dd className="font-mono text-lg tabular-nums text-ink">{stats.beforeSize}</dd>
                  <dt className="font-mono text-[11px] text-ink-3">values in</dt>
                </div>
                <div>
                  <dd className="font-mono text-lg tabular-nums text-sig">{stats.afterSize}</dd>
                  <dt className="font-mono text-[11px] text-ink-3">values out</dt>
                </div>
                <div>
                  <dd className="font-mono text-lg tabular-nums text-ink">75%</dd>
                  <dt className="font-mono text-[11px] text-ink-3">reduction</dt>
                </div>
              </dl>
            )}
            <p className="figcap">
              <b>FIG. 5.1</b> Filter {String(selectedFilter + 1).padStart(3, "0")}. Hover or tap either map to see the 2&times;2 pooling region.
            </p>
          </div>
        </Reveal>
      </div>

      {/* Specimen gallery */}
      <Reveal className="figure mt-16">
        <div className="mb-5 flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
          <p className="font-serif text-xl italic text-ink-2">
            All {numFilters} channels, pooled
          </p>
          <p className="caption">Select a filter to inspect it above.</p>
        </div>
        <div className="grid grid-cols-6 justify-items-center gap-x-2 gap-y-3 sm:grid-cols-[repeat(auto-fill,minmax(56px,1fr))]">
          {pool1Maps
            ? pool1Maps.map((fm, i) => (
                <ActivationHeatmap
                  key={i}
                  data={fm}
                  size={56}
                  label={String(i + 1).padStart(3, "0")}
                  onClick={() => setSelectedFilter(i)}
                  selected={i === selectedFilter}
                />
              ))
            : Array.from({ length: numFilters }, (_, i) => (
                <div
                  key={i}
                  aria-hidden
                  className={`tile border border-rule bg-bg-inset ${
                    i === selectedFilter ? "opacity-100" : "opacity-50"
                  }`}
                  data-selected={i === selectedFilter ? "true" : undefined}
                  style={{ width: 56, maxWidth: "100%", aspectRatio: "1" }}
                />
              ))}
        </div>
        <p className="figcap">
          <b>FIG. 5.2</b> Pool1 output, 14×14 per channel.
        </p>
      </Reveal>
    </SectionWrapper>
  );
}
