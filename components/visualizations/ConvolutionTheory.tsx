"use client";

import { useState, useRef, useEffect, useMemo, useCallback } from "react";
import { useInferenceStore } from "@/stores/inferenceStore";
import { useConv1Weights } from "@/hooks/useConv1Weights";
import { Latex } from "@/components/ui/Latex";
import { PatchGrid } from "@/components/visualizations/PatchGrid";
import { ActivationHeatmap } from "@/components/visualizations/ActivationHeatmap";
import { viridis } from "@/lib/network/networkConstants";

/** Grayscale color function for input pixel values (0-1 range) */
function grayscaleColor(val: number): [number, number, number] {
  const v = Math.round(Math.max(0, Math.min(1, val)) * 255);
  return [v, v, v];
}


/** Neutral dark background — just shows the numbers, no color encoding */
function neutralColor(): [number, number, number] {
  return [26, 26, 36]; // matches surface-elevated #1a1a24
}

/** Smart number format: scientific notation for tiny values, fixed otherwise */
function smartFormat(v: number): string {
  if (v === 0) return "0";
  const abs = Math.abs(v);
  if (abs >= 0.01) return v.toFixed(2);
  if (abs < 1e-6) return "≈0";
  return v.toExponential(0); // "5e-3" not "5.0e-3"
}

/** Extract a 3x3 patch from the input tensor with padding=1 */
function extractPatch(
  input: number[][],
  row: number,
  col: number,
  kSize: number,
  padding: number
): number[][] {
  const h = input.length;
  const w = input[0].length;
  const patch: number[][] = [];
  for (let kr = 0; kr < kSize; kr++) {
    const patchRow: number[] = [];
    for (let kc = 0; kc < kSize; kc++) {
      const ir = row - padding + kr;
      const ic = col - padding + kc;
      if (ir >= 0 && ir < h && ic >= 0 && ic < w) {
        patchRow.push(input[ir][ic]);
      } else {
        patchRow.push(0);
      }
    }
    patch.push(patchRow);
  }
  return patch;
}

/** Compute element-wise products of two 2D arrays */
function elementwiseProducts(a: number[][], b: number[][]): number[][] {
  return a.map((row, r) => row.map((val, c) => val * b[r][c]));
}

/** Viridis color bar: min-max normalization */
function ViridisLegend({ min, max }: { min: number; max: number }) {
  return (
    <div className="flex flex-col items-center gap-0.5">
      <span className="font-mono text-[9px] text-foreground/30">{max.toFixed(1)}</span>
      <div className="flex flex-col" style={{ width: 12, height: 140 }}>
        {Array.from({ length: 32 }, (_, i) => {
          const t = 1 - i / 31; // 1 at top, 0 at bottom
          const [r, g, b] = viridis(t);
          return (
            <div
              key={i}
              style={{
                flex: 1,
                backgroundColor: `rgb(${r},${g},${b})`,
              }}
            />
          );
        })}
      </div>
      <span className="font-mono text-[9px] text-foreground/30">{min.toFixed(1)}</span>
    </div>
  );
}

const GRID = 28;
const CELL = 10;
const CANVAS = GRID * CELL; // 280
const ANIM_INTERVAL = 80; // ms per step

export function ConvolutionTheory() {
  const inputTensor = useInferenceStore((s) => s.inputTensor);
  const layerActivations = useInferenceStore((s) => s.layerActivations);
  const conv1Maps = layerActivations["conv1"] as number[][][] | undefined;
  const { kernels, biases } = useConv1Weights();

  const [kernelPos, setKernelPos] = useState({ row: 5, col: 10 });
  const [selectedFilter, setSelectedFilter] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const inputCanvasRef = useRef<HTMLCanvasElement>(null);
  const outputCanvasRef = useRef<HTMLCanvasElement>(null);

  const kernel = kernels?.[selectedFilter] ?? null;
  const bias = biases?.[selectedFilter] ?? 0;
  const outputMap = conv1Maps?.[selectedFilter];

  // Min-max of the output feature map for normalization
  const outputMinMax = useMemo(() => {
    if (!outputMap) return { min: 0, max: 1 };
    let mn = Infinity, mx = -Infinity;
    for (const row of outputMap) for (const v of row) { if (v < mn) mn = v; if (v > mx) mx = v; }
    return { min: mn, max: mx };
  }, [outputMap]);

  // --- Cached base ImageData (recomputed only when underlying data changes) ---

  const inputBaseImage = useMemo(() => {
    if (!inputTensor) return null;
    const img = new ImageData(CANVAS, CANVAS);
    const px = img.data;
    for (let r = 0; r < GRID; r++) {
      for (let c = 0; c < GRID; c++) {
        const gray = Math.round(inputTensor[r][c] * 255);
        const x0 = c * CELL;
        const y0 = r * CELL;
        for (let dy = 0; dy < CELL; dy++) {
          for (let dx = 0; dx < CELL; dx++) {
            const i = ((y0 + dy) * CANVAS + x0 + dx) * 4;
            px[i] = gray;
            px[i + 1] = gray;
            px[i + 2] = gray;
            px[i + 3] = 255;
          }
        }
      }
    }
    return img;
  }, [inputTensor]);

  const outputBaseImage = useMemo(() => {
    if (!outputMap) return null;
    const { min, max } = outputMinMax;
    const range = max - min;
    const img = new ImageData(CANVAS, CANVAS);
    const px = img.data;
    for (let r = 0; r < GRID; r++) {
      for (let c = 0; c < GRID; c++) {
        const t = range > 0 ? (outputMap[r][c] - min) / range : 0;
        const [red, green, blue] = viridis(t);
        const x0 = c * CELL;
        const y0 = r * CELL;
        for (let dy = 0; dy < CELL; dy++) {
          for (let dx = 0; dx < CELL; dx++) {
            const i = ((y0 + dy) * CANVAS + x0 + dx) * 4;
            px[i] = red;
            px[i + 1] = green;
            px[i + 2] = blue;
            px[i + 3] = 255;
          }
        }
      }
    }
    return img;
  }, [outputMap, outputMinMax]);

  // Extract current patch
  const patch = useMemo(() => {
    if (!inputTensor) return null;
    return extractPatch(inputTensor, kernelPos.row, kernelPos.col, 3, 1);
  }, [inputTensor, kernelPos]);

  // Compute products and sum
  const products = useMemo(() => {
    if (!patch || !kernel) return null;
    return elementwiseProducts(patch, kernel);
  }, [patch, kernel]);

  const rawConvValue = useMemo(() => {
    if (!products) return null;
    let sum = 0;
    for (const row of products) for (const v of row) sum += v;
    return sum + bias;
  }, [products, bias]);

  const productsSum = useMemo(() => {
    if (!products) return null;
    let sum = 0;
    for (const row of products) for (const v of row) sum += v;
    return sum;
  }, [products]);

  // conv1 in the ONNX model is the raw convolution output (pre-ReLU).
  // relu1 is the separate post-ReLU output — that belongs in a later step.

  const hasData = inputTensor && kernel;

  // --- Canvas effects: putImageData(cached) + lightweight overlay ---
  // hasData gates the conditional render of the canvases. Include it as a dep
  // so the effects re-run when the canvas elements first mount in the DOM
  // (covers the case where data arrives before the weights finish loading).

  useEffect(() => {
    const canvas = inputCanvasRef.current;
    if (!canvas || !inputBaseImage) return;
    const ctx = canvas.getContext("2d")!;

    ctx.putImageData(inputBaseImage, 0, 0);

    const overlayX = (kernelPos.col - 1) * CELL;
    const overlayY = (kernelPos.row - 1) * CELL;
    ctx.fillStyle = "rgba(6, 182, 212, 0.15)";
    ctx.fillRect(overlayX, overlayY, 3 * CELL, 3 * CELL);
    ctx.strokeStyle = "#06b6d4";
    ctx.lineWidth = 2;
    ctx.strokeRect(overlayX, overlayY, 3 * CELL, 3 * CELL);
  }, [inputBaseImage, kernelPos, hasData]);

  useEffect(() => {
    const canvas = outputCanvasRef.current;
    if (!canvas || !outputBaseImage) return;
    const ctx = canvas.getContext("2d")!;

    ctx.putImageData(outputBaseImage, 0, 0);

    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 2;
    ctx.strokeRect(
      kernelPos.col * CELL,
      kernelPos.row * CELL,
      CELL,
      CELL
    );
  }, [outputBaseImage, kernelPos, hasData]);

  // Handle click on either canvas
  const handleCanvasClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const rect = e.currentTarget.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * CANVAS;
      const y = ((e.clientY - rect.top) / rect.height) * CANVAS;
      const col = Math.min(GRID - 1, Math.max(0, Math.floor(x / CELL)));
      const row = Math.min(GRID - 1, Math.max(0, Math.floor(y / CELL)));
      setKernelPos({ row, col });
      setIsPlaying(false);
    },
    []
  );

  // Animation: step to next position
  const step = useCallback(() => {
    setKernelPos((prev) => {
      let nextCol = prev.col + 1;
      let nextRow = prev.row;
      if (nextCol >= GRID) {
        nextCol = 0;
        nextRow += 1;
      }
      if (nextRow >= GRID) {
        setIsPlaying(false);
        return { row: 0, col: 0 };
      }
      return { row: nextRow, col: nextCol };
    });
  }, []);

  // RAF-based animation — syncs with browser paint cycle, no timer drift
  useEffect(() => {
    if (!isPlaying) return;
    let lastTime = 0;
    let rafId: number;

    const tick = (now: number) => {
      if (!lastTime) lastTime = now;
      if (now - lastTime >= ANIM_INTERVAL) {
        lastTime += ANIM_INTERVAL;
        step();
      }
      rafId = requestAnimationFrame(tick);
    };

    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [isPlaying, step]);

  // Output cell color: use same min-max viridis scale as the feature map
  const outputCellColorFn = useCallback(
    (val: number): [number, number, number] => {
      const { min, max } = outputMinMax;
      const range = max - min;
      const t = range > 0 ? (val - min) / range : 0;
      return viridis(t);
    },
    [outputMinMax]
  );

  return (
    <div className="flex flex-col gap-10">
      {/* Main layout: theory left, full visualization right */}
      <div className="flex flex-col items-center gap-8 lg:flex-row lg:items-start lg:gap-12">
        {/* Left: theory text */}
        <div className="flex-1 space-y-4 text-center lg:text-left">
          <p className="text-base leading-relaxed text-foreground/65 sm:text-lg">
            A convolution slides a small learned filter (called a <em>kernel</em>) across every
            position of the input image. At each position, it computes the dot product between the
            kernel weights and the overlapping image patch, producing a single output value. Our
            network uses 64 different 3&times;3 kernels — each one learns to detect a different
            pattern like edges, corners, or curves.
          </p>

          {/* Main equation */}
          <div className="py-3">
            <Latex
              display
              math="O(i,j) = \sum_{m=0}^{2}\sum_{n=0}^{2} I(i{+}m,\, j{+}n) \cdot K(m,n) + b"
            />
          </div>

          {/* Equation legend */}
          <div className="flex flex-wrap justify-center gap-x-6 gap-y-1 text-sm text-foreground/40 lg:justify-start">
            <span><Latex math="I" /> — input patch</span>
            <span><Latex math="K" /> — kernel weights</span>
            <span><Latex math="b" /> — bias</span>
            <span><Latex math="O" /> — output value</span>
          </div>

          <p className="text-sm leading-relaxed text-foreground/45">
            With <Latex math="\text{padding}=1" />, the 3&times;3 kernel can be centered on every
            pixel — including edges, where zero-padded values fill in. This preserves the spatial
            dimensions:{" "}
            <Latex math="(1, 28, 28) \xrightarrow{64\text{ filters}} (64, 28, 28)" />.
            Total parameters: <Latex math="64 \times (3 \times 3 + 1) = 640" />.
            A later activation step (ReLU) will clip negatives to zero, but here we show
            the raw convolution output.
          </p>
        </div>

        {/* Right: full visualization */}
        <div className="flex w-full shrink-0 flex-col items-center gap-4 lg:w-auto">
          {hasData ? (
            <div className="space-y-4">
              {/* Top row: input → feature map */}
              <div className="flex flex-col items-center gap-4 sm:flex-row sm:items-center sm:gap-6">
                <div className="flex flex-col items-center gap-1.5">
                  <span className="text-xs text-foreground/40">Input (28&times;28)</span>
                  <canvas
                    ref={inputCanvasRef}
                    width={CANVAS}
                    height={CANVAS}
                    className="cursor-crosshair rounded-md border border-border/60"
                    style={{
                      width: 196,
                      height: 196,
                      imageRendering: "pixelated",
                    }}
                    onClick={handleCanvasClick}
                  />
                  <span className="text-[11px] text-foreground/30">Cyan = current 3&times;3 patch</span>
                </div>

                <div className="text-foreground/25">
                  <span className="hidden sm:inline"><Latex math="\longrightarrow" /></span>
                  <span className="sm:hidden"><Latex math="\downarrow" /></span>
                </div>

                <div className="flex flex-col items-center gap-1.5">
                  <span className="text-xs text-foreground/40">
                    Feature Map #{selectedFilter + 1}
                  </span>
                  <div className="flex items-center gap-2">
                    {outputMap ? (
                      <canvas
                        ref={outputCanvasRef}
                        width={CANVAS}
                        height={CANVAS}
                        className="cursor-crosshair rounded-md border border-border/60"
                        style={{
                          width: 196,
                          height: 196,
                          imageRendering: "pixelated",
                        }}
                        onClick={handleCanvasClick}
                      />
                    ) : (
                      <div
                        className="flex items-center justify-center border border-border/50"
                        style={{ width: 196, height: 196 }}
                      >
                        <span className="text-xs text-foreground/20">No data</span>
                      </div>
                    )}
                    {outputMap && <ViridisLegend min={outputMinMax.min} max={outputMinMax.max} />}
                  </div>
                  <span className="text-[11px] text-foreground/30">
                    Orange = output at [{kernelPos.row}, {kernelPos.col}]
                  </span>
                </div>
              </div>

              {/* Inner workings label */}
              <div className="flex items-center justify-center gap-2 text-foreground/25">
                <Latex math="\downarrow" />
                <span className="text-[11px] tracking-wide">
                  Inner workings at [{kernelPos.row}, {kernelPos.col}] using filter #{selectedFilter + 1}
                </span>
                <Latex math="\downarrow" />
              </div>

              {/* Bottom row: patch math */}
              <div>
                <div className="flex flex-col items-center gap-3 sm:flex-row sm:flex-wrap sm:items-center sm:justify-center sm:gap-2.5">
                  <PatchGrid
                    data={patch!}
                    colorFn={grayscaleColor}
                    cellSize={40}
                    showValues
                    label="Input Patch"
                  />

                  <span className="text-foreground/30"><Latex math="\times" /></span>

                  <PatchGrid
                    data={kernel}
                    colorFn={neutralColor}
                    cellSize={40}
                    showValues
                    valueFormat={smartFormat}
                    label={`Kernel #${selectedFilter + 1}`}
                  />

                  <span className="text-foreground/30"><Latex math="=" /></span>

                  <PatchGrid
                    data={products!}
                    colorFn={neutralColor}
                    cellSize={40}
                    showValues
                    valueFormat={smartFormat}
                    label="Products"
                  />

                  <div className="flex flex-col items-center gap-0.5">
                    <span className="text-[11px] text-foreground/40"><Latex math="\scriptstyle\sum + b" /></span>
                    <span className="text-foreground/30"><Latex math="\longrightarrow" /></span>
                  </div>

                  <PatchGrid
                    data={[[rawConvValue ?? 0]]}
                    colorFn={outputCellColorFn}
                    cellSize={40}
                    showValues
                    valueFormat={smartFormat}
                    label="Output"
                  />
                </div>

                {/* Compact sum breakdown */}
                {rawConvValue !== null && productsSum !== null && (
                  <div className="mt-3 flex flex-col items-center gap-1 font-mono text-xs text-foreground/40">
                    <div className="flex flex-wrap items-center justify-center gap-x-2 gap-y-0.5">
                      <span>
                        <span className="text-foreground/25">Σ(products)</span>{" "}
                        = {smartFormat(productsSum)}
                      </span>
                      <span className="text-foreground/25">+</span>
                      <span>
                        <span className="text-foreground/25">bias</span>{" "}
                        {smartFormat(bias)}
                      </span>
                      <span className="text-foreground/25">=</span>
                      <span className="text-accent-primary">{smartFormat(rawConvValue)}</span>
                    </div>
                  </div>
                )}
              </div>

              {/* Controls */}
              <div className="flex items-center justify-center gap-3">
                <button
                  onClick={() => setIsPlaying(!isPlaying)}
                  className="flex items-center gap-1.5 rounded-lg border border-accent-tertiary/30 bg-accent-tertiary/10 px-3 py-1.5 text-sm font-medium text-accent-tertiary transition-colors hover:bg-accent-tertiary/20"
                >
                  {isPlaying ? (
                    <>
                      <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor"><rect x="2" y="1" width="4" height="12" rx="1" /><rect x="8" y="1" width="4" height="12" rx="1" /></svg>
                      Pause
                    </>
                  ) : (
                    <>
                      <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor"><path d="M3 1.5v11l9-5.5z" /></svg>
                      Play
                    </>
                  )}
                </button>
                <button
                  onClick={step}
                  className="rounded-lg border border-border px-3 py-1.5 text-sm text-foreground/60 transition-colors hover:border-foreground/40"
                >
                  Step
                </button>
                <button
                  onClick={() => {
                    setKernelPos({ row: 0, col: 0 });
                    setIsPlaying(false);
                  }}
                  className="rounded-lg border border-border px-3 py-1.5 text-sm text-foreground/60 transition-colors hover:border-foreground/40"
                >
                  Reset
                </button>
                <span className="font-mono text-[10px] text-foreground/30">
                  [{kernelPos.row}, {kernelPos.col}]
                </span>
              </div>
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

      {/* Filter selection: clickable feature map thumbnails — full width */}
      {hasData && (
        <div className="space-y-3">
          <p className="text-center text-xs text-foreground/40">
            Select a filter — click any feature map below
          </p>
          <div className="grid grid-cols-4 gap-2 sm:grid-cols-8">
            {conv1Maps
              ? conv1Maps.map((fm, i) => (
                  <ActivationHeatmap
                    key={i}
                    data={fm}
                    size={56}
                    label={`#${i + 1}`}
                    onClick={() => setSelectedFilter(i)}
                    selected={i === selectedFilter}
                  />
                ))
              : Array.from({ length: 64 }, (_, i) => (
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
    </div>
  );
}
