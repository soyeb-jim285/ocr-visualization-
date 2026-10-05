"use client";

import { motion } from "framer-motion";
import { useState, useRef, useEffect, useMemo, useCallback } from "react";
import { useInferenceStore } from "@/stores/inferenceStore";
import { useConv1Weights } from "@/hooks/useConv1Weights";
import { Latex } from "@/components/ui/Latex";
import { PatchGrid } from "@/components/visualizations/PatchGrid";
import { ActivationHeatmap } from "@/components/visualizations/ActivationHeatmap";
import { viridis } from "@/lib/network/networkConstants";
import { PHOSPHOR, ANNOTATION } from "@/lib/theme";

/** Grayscale color function for input pixel values (0-1 range) */
function grayscaleColor(val: number): [number, number, number] {
  const v = Math.round(Math.max(0, Math.min(1, val)) * 255);
  return [v, v, v];
}


/** Neutral dark background — just shows the numbers, no color encoding */
function neutralColor(): [number, number, number] {
  return [17, 24, 33]; // matches bg-lift #111821
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

/** Legend label: 2 decimals, 3 when the range is tiny; no "-0.00" */
function fmtLegend(v: number, range: number) {
  const s = v.toFixed(range < 0.05 ? 3 : 2);
  return /^-0\.0+$/.test(s) ? s.slice(1) : s;
}

/** Viridis color bar: min-max normalization */
function ViridisLegend({ min, max }: { min: number; max: number }) {
  return (
    <div className="flex flex-col items-start gap-0.5">
      <span className="font-mono text-[10px] text-ink-3">{fmtLegend(max, max - min)}</span>
      <div className="flex flex-col" style={{ width: 6, height: 164 }}>
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
      <span className="font-mono text-[10px] text-ink-3">{fmtLegend(min, max - min)}</span>
    </div>
  );
}

const reveal = (delay = 0) => ({
  initial: { opacity: 0, y: 12 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, margin: "-12% 0px" },
  transition: { duration: 0.5, delay, ease: [0.16, 1, 0.3, 1] as const },
});

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
    ctx.fillStyle = "rgba(143, 227, 255, 0.15)";
    ctx.fillRect(overlayX, overlayY, 3 * CELL, 3 * CELL);
    ctx.strokeStyle = PHOSPHOR;
    ctx.lineWidth = 2;
    ctx.strokeRect(overlayX, overlayY, 3 * CELL, 3 * CELL);
  }, [inputBaseImage, kernelPos, hasData]);

  useEffect(() => {
    const canvas = outputCanvasRef.current;
    if (!canvas || !outputBaseImage) return;
    const ctx = canvas.getContext("2d")!;

    ctx.putImageData(outputBaseImage, 0, 0);

    ctx.strokeStyle = ANNOTATION;
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

  const op = (c: string) => <span className="px-1 text-ink-3">{c}</span>;

  return (
    <div className="flex flex-col gap-16">
      <div className="grid grid-cols-[minmax(0,1fr)] gap-12 lg:grid-cols-12 lg:gap-x-6">
        {/* Left: theory */}
        <motion.div {...reveal()} className="min-w-0 space-y-6 lg:col-span-5">
          <p className="prose-body">
            A convolution slides a small learned filter, called a <em>kernel</em>, across every
            position of the input image. At each position it computes the dot product between the
            kernel weights and the overlapping image patch, producing a single output value. Our
            network uses 64 different 3&times;3 kernels, and each one learns to detect a different
            pattern like edges, corners, or curves.
          </p>

          <div className="formula">
            <Latex
              display
              math="O(i,j) = \sum_{m=0}^{2}\sum_{n=0}^{2} I(i{+}m,\, j{+}n) \cdot K(m,n) + b"
            />
            <span className="eq-no">(2)</span>
          </div>

          <dl className="grid grid-cols-[auto_1fr] items-baseline gap-x-4 gap-y-1.5 text-sm">
            {[
              ["I", "input patch"],
              ["K", "kernel weights"],
              ["b", "bias"],
              ["O", "output value"],
            ].map(([sym, desc]) => (
              <div key={sym} className="contents">
                <dt className="text-ink">
                  <Latex math={sym} />
                </dt>
                <dd className="font-mono text-[11px] text-ink-3">{desc}</dd>
              </div>
            ))}
          </dl>

          <p className="callout [overflow-wrap:anywhere]">
            <span className="tag">NOTE</span>
            With <Latex math="\text{padding}=1" /> the kernel centers on every pixel, edges
            included, so the spatial size is preserved. Raw output is shown here; ReLU comes next.
          </p>

          <p className="text-sm leading-[1.65] text-ink-3">
            <Latex math="(1, 28, 28) \xrightarrow{64\text{ filters}} (64, 28, 28)" />. Total
            parameters:{" "}
            <span className="inline-block max-w-full overflow-x-auto align-bottom">
              <Latex math="64 \times (3 \times 3 + 1) = 640" />
            </span>
            .
          </p>
        </motion.div>

        {/* Right: walkthrough figure */}
        <motion.figure {...reveal(0.16)} className="figure m-0 min-w-0 lg:col-span-7">
          {hasData ? (
            <div className="space-y-8">
              <div className="flex flex-col items-start gap-4 sm:flex-row sm:gap-5">
                <div className="flex flex-col gap-2">
                  <span className="font-mono text-[11px] text-ink-2">
                    INPUT <span className="text-ink-3">28&times;28</span>
                  </span>
                  <div className="well plate-marks">
                    <canvas
                      ref={inputCanvasRef}
                      width={CANVAS}
                      height={CANVAS}
                      className="block cursor-crosshair"
                      style={{ width: 196, height: 196, imageRendering: "pixelated" }}
                      onClick={handleCanvasClick}
                    />
                  </div>
                  <span className="font-mono text-[11px] text-ink-3">3&times;3 patch</span>
                </div>

                <div className="self-center text-ink-3 sm:pt-6">
                  <span className="hidden sm:inline"><Latex math="\longrightarrow" /></span>
                  <span className="sm:hidden"><Latex math="\downarrow" /></span>
                </div>

                <div className="flex flex-col gap-2">
                  <span className="font-mono text-[11px] text-ink-2">
                    FEATURE MAP {String(selectedFilter + 1).padStart(2, "0")}{" "}
                    <span className="text-ink-3">28&times;28</span>
                  </span>
                  <div className="flex items-start gap-2.5">
                    {outputMap ? (
                      <div className="well plate-marks">
                        <canvas
                          ref={outputCanvasRef}
                          width={CANVAS}
                          height={CANVAS}
                          className="block cursor-crosshair"
                          style={{ width: 196, height: 196, imageRendering: "pixelated" }}
                          onClick={handleCanvasClick}
                        />
                      </div>
                    ) : (
                      <div className="viz-empty-state" style={{ width: 196, height: 196 }}>
                        No data
                      </div>
                    )}
                    {outputMap && <ViridisLegend min={outputMinMax.min} max={outputMinMax.max} />}
                  </div>
                  <span className="font-mono text-[11px] text-ink-3">
                    output [{kernelPos.row}, {kernelPos.col}]
                  </span>
                </div>
              </div>

              {/* Inner workings */}
              <div className="border-t border-rule-faint pt-5">
                <p className="mb-4 font-mono text-[11px] tracking-[0.04em] text-ink-3">
                  <span className="text-sig">INNER WORKINGS</span> at [{kernelPos.row}, {kernelPos.col}], filter {selectedFilter + 1}
                </p>
                <div className="flex flex-wrap items-center justify-center gap-x-2.5 gap-y-4 sm:justify-start">
                  <PatchGrid
                    data={patch!}
                    colorFn={grayscaleColor}
                    cellSize={40}
                    showValues
                    label="Input patch"
                  />
                  <span className="text-ink-3"><Latex math="\times" /></span>
                  <PatchGrid
                    data={kernel}
                    colorFn={neutralColor}
                    cellSize={40}
                    showValues
                    valueFormat={smartFormat}
                    label={`Kernel ${selectedFilter + 1}`}
                  />
                  <span className="text-ink-3"><Latex math="=" /></span>
                  <PatchGrid
                    data={products!}
                    colorFn={neutralColor}
                    cellSize={40}
                    showValues
                    valueFormat={smartFormat}
                    label="Products"
                  />
                  <div className="flex flex-col items-center gap-0.5 text-ink-3">
                    <span className="text-[11px]"><Latex math="\scriptstyle\sum + b" /></span>
                    <Latex math="\longrightarrow" />
                  </div>
                  <PatchGrid
                    data={[[rawConvValue ?? 0]]}
                    colorFn={outputCellColorFn}
                    cellSize={40}
                    showValues
                    valueFormat={smartFormat}
                    label="Output"
                    highlight
                  />
                </div>

                {rawConvValue !== null && productsSum !== null && (
                  <p className="mt-4 flex flex-wrap items-center justify-center font-mono text-xs text-ink-2 sm:justify-start">
                    <span><span className="text-ink-3">&Sigma;(products)</span> {smartFormat(productsSum)}</span>
                    {op("+")}
                    <span><span className="text-ink-3">bias</span> {smartFormat(bias)}</span>
                    {op("=")}
                    <span className="text-sig">{smartFormat(rawConvValue)}</span>
                  </p>
                )}
              </div>

              {/* Controls */}
              <div className="flex flex-wrap items-center gap-2.5">
                <button onClick={() => setIsPlaying(!isPlaying)} className="btn-primary">
                  {isPlaying ? (
                    <>
                      <svg width="12" height="12" viewBox="0 0 14 14" fill="currentColor"><rect x="2" y="1" width="4" height="12" /><rect x="8" y="1" width="4" height="12" /></svg>
                      Pause
                    </>
                  ) : (
                    <>
                      <svg width="12" height="12" viewBox="0 0 14 14" fill="currentColor"><path d="M3 1.5v11l9-5.5z" /></svg>
                      Play
                    </>
                  )}
                </button>
                <button onClick={step} className="btn-ghost">Step</button>
                <button
                  onClick={() => {
                    setKernelPos({ row: 0, col: 0 });
                    setIsPlaying(false);
                  }}
                  className="btn-ghost"
                >
                  Reset
                </button>
                <span className="readout ml-auto min-w-0 shrink">
                  [{kernelPos.row}, {kernelPos.col}]
                </span>
              </div>
            </div>
          ) : (
            <div className="viz-empty-state">Draw something to light this up</div>
          )}
          <figcaption className="figcap [overflow-wrap:anywhere]">
            <b>FIG. 2.1</b> Click either canvas to move the kernel. Cyan marks the 3&times;3 patch,
            orange the output pixel it produces.
          </figcaption>
        </motion.figure>
      </div>

      {/* Filter selection */}
      {hasData && (
        <motion.figure {...reveal()} className="figure m-0">
          <div className="grid grid-cols-[repeat(auto-fill,minmax(56px,1fr))] gap-2">
            {conv1Maps
              ? conv1Maps.map((fm, i) => (
                  <ActivationHeatmap
                    key={i}
                    data={fm}
                    size={56}
                    label={`${i + 1}`}
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
                    <div className="tile" style={{ width: 56, height: 56 }} />
                    <span className="font-mono text-[10px] text-ink-3">{i + 1}</span>
                  </div>
                ))}
          </div>
          <figcaption className="figcap">
            <b>FIG. 2.2</b> All 64 filters applied to your drawing. Select a feature map to inspect
            its kernel above.
          </figcaption>
        </motion.figure>
      )}
    </div>
  );
}
