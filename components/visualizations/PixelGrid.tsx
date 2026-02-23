"use client";

import { useRef, useEffect, useState, useCallback, useMemo } from "react";
import { useInferenceStore } from "@/stores/inferenceStore";
import { Latex } from "@/components/ui/Latex";

const SRC_SIZE = 280;
const DST_SIZE = 28;
const SCALE = SRC_SIZE / DST_SIZE; // 10
const OUTPUT_CELL = 14;
const OUTPUT_CANVAS = DST_SIZE * OUTPUT_CELL; // 392
const ZOOM_CELLS = Math.ceil(SCALE); // 10
const ZOOM_CELL_PX = 10;
const ZOOM_CANVAS = ZOOM_CELLS * ZOOM_CELL_PX; // 100

interface HoveredCell {
  row: number;
  col: number;
  value: number;
}

export function PixelGrid() {
  const inputImageData = useInferenceStore((s) => s.inputImageData);
  const inputTensor = useInferenceStore((s) => s.inputTensor);

  const sourceRef = useRef<HTMLCanvasElement>(null);
  const outputRef = useRef<HTMLCanvasElement>(null);
  const zoomRef = useRef<HTMLCanvasElement>(null);

  const [hovered, setHovered] = useState<HoveredCell | null>(null);

  // The cell shown in zoom panel — hovered cell, or default center when data exists
  const activeCell = useMemo((): HoveredCell | null => {
    if (hovered) return hovered;
    if (!inputTensor) return null;
    return { row: 14, col: 14, value: inputTensor[14][14] };
  }, [hovered, inputTensor]);

  // Cache the output grid ImageData so hover redraws only do putImageData + overlay
  const outputBaseImage = useMemo(() => {
    if (!inputTensor) return null;
    const img = new ImageData(OUTPUT_CANVAS, OUTPUT_CANVAS);
    const px = img.data;
    for (let r = 0; r < DST_SIZE; r++) {
      for (let c = 0; c < DST_SIZE; c++) {
        const gray = Math.round(inputTensor[r][c] * 255);
        const x0 = c * OUTPUT_CELL;
        const y0 = r * OUTPUT_CELL;
        for (let dy = 0; dy < OUTPUT_CELL; dy++) {
          for (let dx = 0; dx < OUTPUT_CELL; dx++) {
            const i = ((y0 + dy) * OUTPUT_CANVAS + x0 + dx) * 4;
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

  // --- Source canvas: full image + region highlight ---
  useEffect(() => {
    const canvas = sourceRef.current;
    if (!canvas || !inputImageData) return;
    const ctx = canvas.getContext("2d")!;

    // Draw full image
    ctx.putImageData(inputImageData, 0, 0);

    if (activeCell) {
      const sx = activeCell.col * SCALE;
      const sy = activeCell.row * SCALE;
      const sz = Math.ceil(SCALE);

      // Dim everything
      ctx.fillStyle = hovered ? "rgba(0, 0, 0, 0.55)" : "rgba(0, 0, 0, 0.35)";
      ctx.fillRect(0, 0, SRC_SIZE, SRC_SIZE);

      // Restore highlighted region (putImageData ignores compositing)
      ctx.putImageData(inputImageData, 0, 0, sx, sy, sz, sz);

      // Cyan border
      ctx.strokeStyle = "#06b6d4";
      ctx.lineWidth = 2;
      ctx.strokeRect(sx + 1, sy + 1, sz - 2, sz - 2);
    }
  }, [inputImageData, activeCell, hovered]);

  // --- Output canvas: 28x28 grid + highlight ---
  useEffect(() => {
    const canvas = outputRef.current;
    if (!canvas || !outputBaseImage) return;
    const ctx = canvas.getContext("2d")!;

    ctx.putImageData(outputBaseImage, 0, 0);

    // Grid lines
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= DST_SIZE; i++) {
      const p = i * OUTPUT_CELL;
      ctx.beginPath();
      ctx.moveTo(p, 0);
      ctx.lineTo(p, OUTPUT_CANVAS);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, p);
      ctx.lineTo(OUTPUT_CANVAS, p);
      ctx.stroke();
    }

    // Highlight active cell
    if (activeCell) {
      ctx.strokeStyle = "#f59e0b";
      ctx.lineWidth = 2;
      ctx.strokeRect(
        activeCell.col * OUTPUT_CELL + 1,
        activeCell.row * OUTPUT_CELL + 1,
        OUTPUT_CELL - 2,
        OUTPUT_CELL - 2
      );
    }
  }, [outputBaseImage, activeCell]);

  // --- Zoom canvas: enlarged source patch ---
  useEffect(() => {
    const canvas = zoomRef.current;
    if (!canvas || !inputImageData || !activeCell) return;
    const ctx = canvas.getContext("2d")!;

    const sx = activeCell.col * SCALE;
    const sy = activeCell.row * SCALE;

    ctx.clearRect(0, 0, ZOOM_CANVAS, ZOOM_CANVAS);

    for (let py = 0; py < ZOOM_CELLS; py++) {
      for (let px = 0; px < ZOOM_CELLS; px++) {
        const srcX = sx + px;
        const srcY = sy + py;
        if (srcX < SRC_SIZE && srcY < SRC_SIZE) {
          const idx = (srcY * SRC_SIZE + srcX) * 4;
          const v = inputImageData.data[idx];
          ctx.fillStyle = `rgb(${v},${v},${v})`;
        } else {
          ctx.fillStyle = "rgb(0,0,0)";
        }
        ctx.fillRect(px * ZOOM_CELL_PX, py * ZOOM_CELL_PX, ZOOM_CELL_PX, ZOOM_CELL_PX);
      }
    }

    // Grid lines
    ctx.strokeStyle = "rgba(6, 182, 212, 0.25)";
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= ZOOM_CELLS; i++) {
      const p = i * ZOOM_CELL_PX;
      ctx.beginPath();
      ctx.moveTo(p, 0);
      ctx.lineTo(p, ZOOM_CANVAS);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, p);
      ctx.lineTo(ZOOM_CANVAS, p);
      ctx.stroke();
    }
  }, [inputImageData, activeCell]);

  // --- Hover handlers (both canvases set the same state) ---
  const resolveCell = useCallback(
    (canvasWidth: number, nativeSize: number, e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!inputTensor) return;
      const rect = e.currentTarget.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * nativeSize;
      const y = ((e.clientY - rect.top) / rect.height) * nativeSize;
      const col = Math.min(
        DST_SIZE - 1,
        Math.max(0, Math.floor(x / (nativeSize / DST_SIZE)))
      );
      const row = Math.min(
        DST_SIZE - 1,
        Math.max(0, Math.floor(y / (nativeSize / DST_SIZE)))
      );
      setHovered({ row, col, value: inputTensor[row][col] });
    },
    [inputTensor]
  );

  const handleSourceHover = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => resolveCell(SRC_SIZE, SRC_SIZE, e),
    [resolveCell]
  );
  const handleOutputHover = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) =>
      resolveCell(OUTPUT_CANVAS, OUTPUT_CANVAS, e),
    [resolveCell]
  );
  const clearHover = useCallback(() => setHovered(null), []);

  const hasData = inputTensor && inputImageData;

  return (
    <div className="flex flex-col items-center gap-5">
      {/* Source -> Output canvases */}
      <div className="flex flex-col items-center gap-3 sm:flex-row sm:items-center sm:gap-5">
        {/* Source 280x280 */}
        <div className="flex flex-col items-center gap-1.5">
          <span className="text-xs text-foreground/40">Source (280&times;280)</span>
          <div className="relative">
            <canvas
              ref={sourceRef}
              width={SRC_SIZE}
              height={SRC_SIZE}
              className="cursor-crosshair rounded-lg border border-border/60"
              style={{ width: 180, height: 180, imageRendering: "pixelated" }}
              onMouseMove={handleSourceHover}
              onMouseLeave={clearHover}
            />
            {!hasData && (
              <div className="absolute inset-0 flex items-center justify-center rounded-lg bg-surface/80">
                <p className="text-xs text-foreground/30">
                  Draw a character above
                </p>
              </div>
            )}
          </div>
          <span className="h-4 text-[10px] text-foreground/30">
            {activeCell && hasData
              ? `Cyan = ${ZOOM_CELLS}\u00d7${ZOOM_CELLS} source region`
              : "\u00a0"}
          </span>
        </div>

        {/* Arrow */}
        <div className="text-foreground/25">
          <span className="hidden sm:inline">
            <Latex math="\xrightarrow{\;\div 10\;}" />
          </span>
          <span className="sm:hidden">
            <Latex math="\downarrow\;\div 10" />
          </span>
        </div>

        {/* Output 28x28 */}
        <div className="flex flex-col items-center gap-1.5">
          <span className="text-xs text-foreground/40">
            Downsampled (28&times;28)
          </span>
          <div className="relative">
            <canvas
              ref={outputRef}
              width={OUTPUT_CANVAS}
              height={OUTPUT_CANVAS}
              className="cursor-crosshair rounded-lg border border-border/60"
              style={{ width: 180, height: 180, imageRendering: "pixelated" }}
              onMouseMove={handleOutputHover}
              onMouseLeave={clearHover}
            />
            {!hasData && (
              <div className="absolute inset-0 flex items-center justify-center rounded-lg bg-surface/80">
                <p className="text-xs text-foreground/30">
                  Draw a character above
                </p>
              </div>
            )}
          </div>
          <span className="h-4 text-[10px] text-foreground/30">
            {activeCell && hasData
              ? `Orange = output[${activeCell.row}, ${activeCell.col}]`
              : "\u00a0"}
          </span>
        </div>
      </div>

      {/* Zoom detail panel — always visible when data exists */}
      {hasData && (
        <div className="flex items-center gap-4 rounded-xl border border-border/40 bg-surface-elevated/60 px-5 py-3 backdrop-blur-sm">
          <div className="flex flex-col items-center gap-1">
            <span className="text-[10px] text-foreground/30">
              {ZOOM_CELLS}&times;{ZOOM_CELLS} source pixels
            </span>
            <canvas
              ref={zoomRef}
              width={ZOOM_CANVAS}
              height={ZOOM_CANVAS}
              className="rounded border border-cyan-500/30"
              style={{ width: 80, height: 80, imageRendering: "pixelated" }}
            />
          </div>

          <div className="flex flex-col items-center text-foreground/25">
            <Latex math="\xrightarrow{\;\text{avg}\;}" />
          </div>

          <div className="flex flex-col items-center gap-1">
            <span className="text-[10px] text-foreground/30">result</span>
            <div
              className="flex h-12 w-12 items-center justify-center rounded border border-amber-500/40"
              style={{
                backgroundColor: activeCell
                  ? `rgb(${Math.round(activeCell.value * 255)},${Math.round(activeCell.value * 255)},${Math.round(activeCell.value * 255)})`
                  : "transparent",
              }}
            >
              {activeCell && (
                <span
                  className="font-mono text-xs font-bold"
                  style={{
                    color:
                      activeCell.value > 0.5
                        ? "rgba(0,0,0,0.8)"
                        : "rgba(255,255,255,0.7)",
                  }}
                >
                  {activeCell.value.toFixed(2)}
                </span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Coordinate readout */}
      {activeCell && hasData ? (
        <p className="font-mono text-[11px] text-foreground/35">
          output[{activeCell.row}, {activeCell.col}] ={" "}
          {activeCell.value.toFixed(4)}
        </p>
      ) : hasData ? (
        <p className="text-xs text-foreground/30">
          Hover either canvas to see which source pixels map to each output
          value
        </p>
      ) : null}
    </div>
  );
}
