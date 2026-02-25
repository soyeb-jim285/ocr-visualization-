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

      ctx.fillStyle = "rgba(6, 182, 212, 0.2)";
      ctx.fillRect(x, y, size, size);
      ctx.strokeStyle = "#06b6d4";
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

      ctx.fillStyle = "rgba(245, 158, 11, 0.25)";
      ctx.fillRect(x, y, CELL_A, CELL_A);
      ctx.strokeStyle = "#f59e0b";
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
    (e: React.MouseEvent<HTMLCanvasElement>) => {
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
    (e: React.MouseEvent<HTMLCanvasElement>) => {
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

  const clearHover = useCallback(() => setHoverPool(null), []);

  // Display values for the 4-cell grid
  const displayValues = poolValues ?? [0.3, 0.7, 0.1, 0.9];
  const displayMaxIdx = poolValues ? maxIdx : 3;
  const isLive = !!poolValues;

  return (
    <div className="flex flex-col items-center gap-5 sm:flex-row sm:gap-6">
      {/* Before pooling */}
      <div className="flex flex-col items-center gap-2">
        <span className="text-xs font-medium text-foreground/60">
          Before Pooling
        </span>
        <canvas
          ref={beforeRef}
          width={CANVAS_SIZE}
          height={CANVAS_SIZE}
          className="cursor-crosshair rounded-md border border-border/60"
          style={{ width: CANVAS_SIZE, height: CANVAS_SIZE }}
          onMouseMove={handleBeforeMove}
          onMouseLeave={clearHover}
        />
        <span className="font-mono text-[11px] text-foreground/30">
          {BEFORE_GRID}&times;{BEFORE_GRID}
          {hoverPool && (
            <span className="text-cyan-400">
              {" "}
              [{hoverPool.row * 2}:{hoverPool.row * 2 + 1}, {hoverPool.col * 2}
              :{hoverPool.col * 2 + 1}]
            </span>
          )}
        </span>
      </div>

      {/* 2×2 pool grid with live values */}
      <div className="flex flex-col items-center gap-2">
        <div
          className={`grid grid-cols-2 gap-0.5 rounded border p-1.5 transition-colors ${
            isLive
              ? "border-cyan-400/40 bg-cyan-950/20"
              : "border-accent-tertiary/30 bg-surface"
          }`}
        >
          {displayValues.map((v, i) => (
            <div
              key={i}
              className={`flex h-7 w-7 items-center justify-center rounded-sm font-mono text-[10px] transition-colors ${
                i === displayMaxIdx
                  ? "bg-accent-tertiary font-bold text-background"
                  : "bg-border/30 text-foreground/30"
              }`}
            >
              {isLive ? v.toFixed(2) : v}
            </div>
          ))}
        </div>
        <Latex
          math="\xrightarrow{\max}"
          className="hidden text-foreground/40 sm:block"
        />
        <Latex math="\downarrow" className="text-foreground/40 sm:hidden" />
      </div>

      {/* After pooling */}
      <div className="flex flex-col items-center gap-2">
        <span className="text-xs font-medium text-foreground/60">
          After Pooling
        </span>
        <canvas
          ref={afterRef}
          width={CANVAS_SIZE}
          height={CANVAS_SIZE}
          className="cursor-crosshair rounded-md border border-border/60"
          style={{ width: CANVAS_SIZE, height: CANVAS_SIZE }}
          onMouseMove={handleAfterMove}
          onMouseLeave={clearHover}
        />
        <span className="font-mono text-[11px] text-foreground/30">
          {AFTER_GRID}&times;{AFTER_GRID}
          {hoverPool && (
            <span className="text-amber-400">
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
  const [selectedFilter, setSelectedFilter] = useState(0);

  // Before pooling (relu2 = 28x28x64) and after pooling (pool1 = 14x14x64)
  const relu2Maps = layerActivations["relu2"] as number[][][] | undefined;
  const pool1Maps = layerActivations["pool1"] as number[][][] | undefined;

  const beforePool = relu2Maps?.[selectedFilter];
  const afterPool = pool1Maps?.[selectedFilter];

  const numFilters = relu2Maps?.length ?? 0;
  const hasData = !!beforePool;

  const stats = useMemo(() => {
    if (!beforePool || !afterPool) return null;
    const beforeSize = beforePool.length * (beforePool[0]?.length ?? 0);
    const afterSize = afterPool.length * (afterPool[0]?.length ?? 0);
    return { beforeSize, afterSize };
  }, [beforePool, afterPool]);

  return (
    <SectionWrapper id="pooling">
      <SectionHeader
        step={5}
        title="Compressing Information: Max Pooling"
        subtitle="Max pooling slides a 2×2 window across each feature map and keeps only the maximum value. This halves the spatial dimensions while retaining the strongest activations — making the model more efficient and somewhat invariant to small shifts in position."
      />

      <div className="flex flex-col items-center gap-8 lg:flex-row lg:items-start lg:gap-12">
        {/* Left: theory text */}
        <div className="flex-1 space-y-4 text-center lg:text-left">
          <p className="text-base leading-relaxed text-foreground/65 sm:text-lg">
            After activation, the feature maps still carry full spatial
            resolution. <em>Max pooling</em> compresses each map by partitioning
            it into non-overlapping 2&times;2 regions and keeping only the
            largest value from each. This achieves two goals: it reduces the
            number of parameters downstream (preventing overfitting) and
            introduces <em>translation invariance</em> — small shifts in the
            input produce identical pooled outputs.
          </p>

          {/* Main equation */}
          <div className="py-3">
            <Latex
              display
              math="P(i,j) = \max_{(m,n)\,\in\,R_{i,j}} A(m,n)"
            />
          </div>

          {/* Equation legend */}
          <div className="flex flex-wrap justify-center gap-x-6 gap-y-1 text-sm text-foreground/40 lg:justify-start">
            <span>
              <Latex math="A" /> — input activation map
            </span>
            <span>
              <Latex math="R_{i,j}" /> — 2&times;2 pooling region
            </span>
            <span>
              <Latex math="P" /> — pooled output
            </span>
          </div>

          <p className="text-sm leading-relaxed text-foreground/45">
            With stride 2, each 2&times;2 window produces one output value,
            halving both dimensions:{" "}
            <Latex math="(64, 28, 28) \xrightarrow{2{\times}2\;\text{max pool}} (64, 14, 14)" />
            . This is a 75% reduction in spatial size — from{" "}
            <Latex math="28^2 = 784" /> to <Latex math="14^2 = 196" /> values
            per channel — with zero learnable parameters. The operation is
            purely structural: no weights, no bias, just a hard{" "}
            <Latex math="\max" />.
          </p>
        </div>

        {/* Right: interactive visualization */}
        <div className="flex w-full shrink-0 flex-col items-center gap-5 lg:w-auto">
          {hasData && afterPool ? (
            <>
              <PoolingViz beforeData={beforePool} afterData={afterPool} />

              {/* Stats — single line */}
              {stats && (
                <div className="flex items-center gap-4 text-sm">
                  <span className="font-mono font-semibold text-foreground/60">
                    {stats.beforeSize}
                  </span>
                  <span className="text-foreground/40">values</span>
                  <span className="text-accent-tertiary">&rarr;</span>
                  <span className="font-mono font-semibold text-accent-tertiary">
                    {stats.afterSize}
                  </span>
                  <span className="text-foreground/40">values</span>
                  <span className="text-foreground/15">|</span>
                  <span className="font-mono font-semibold text-green-400">
                    75%
                  </span>
                  <span className="text-foreground/40">reduction</span>
                </div>
              )}

              <span className="text-[11px] text-foreground/30">
                Hover either map to see the 2&times;2 pooling region
              </span>
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

      {/* Filter selection: clickable thumbnails — full width */}
      {hasData && (
        <div className="mt-6 space-y-3">
          <p className="text-center text-xs text-foreground/40">
            Select a filter — click any feature map below
          </p>
          <div className="grid grid-cols-4 gap-2 sm:grid-cols-8">
            {pool1Maps
              ? pool1Maps.map((fm, i) => (
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
                    <span className="text-xs text-foreground/40">
                      #{i + 1}
                    </span>
                  </div>
                ))}
          </div>
        </div>
      )}
    </SectionWrapper>
  );
}
