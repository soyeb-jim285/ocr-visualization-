"use client";

import { useRef, useEffect, useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useInferenceStore } from "@/stores/inferenceStore";

type ViewMode = "image" | "data";

export function PixelGrid() {
  const inputTensor = useInferenceStore((s) => s.inputTensor);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("image");
  const [hoveredCell, setHoveredCell] = useState<{
    row: number;
    col: number;
    value: number;
  } | null>(null);

  const cellSize = 14;
  const gridSize = 28;
  const canvasSize = gridSize * cellSize;

  // Draw the pixel grid
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !inputTensor) return;
    const ctx = canvas.getContext("2d")!;

    ctx.clearRect(0, 0, canvasSize, canvasSize);

    for (let r = 0; r < gridSize; r++) {
      for (let c = 0; c < gridSize; c++) {
        const val = inputTensor[r][c];

        if (viewMode === "image") {
          const gray = Math.round(val * 255);
          ctx.fillStyle = `rgb(${gray}, ${gray}, ${gray})`;
          ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
        } else {
          // Data view: color-coded cells with values
          const hue = 260 - val * 200; // Purple to yellow
          const sat = 70 + val * 30;
          const light = 15 + val * 50;
          ctx.fillStyle = `hsl(${hue}, ${sat}%, ${light}%)`;
          ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);

          // Draw grid lines
          ctx.strokeStyle = "rgba(255,255,255,0.05)";
          ctx.strokeRect(c * cellSize, r * cellSize, cellSize, cellSize);

          // Show value text for non-zero cells (if cell is big enough)
          if (val > 0.01 && cellSize >= 12) {
            ctx.fillStyle =
              val > 0.5 ? "rgba(0,0,0,0.7)" : "rgba(255,255,255,0.5)";
            ctx.font = `${Math.max(8, cellSize * 0.55)}px var(--font-geist-mono)`;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(
              val.toFixed(1),
              c * cellSize + cellSize / 2,
              r * cellSize + cellSize / 2
            );
          }
        }
      }
    }
  }, [inputTensor, viewMode, cellSize, canvasSize]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!inputTensor) return;
    const rect = canvasRef.current!.getBoundingClientRect();
    const scale = canvasSize / rect.width;
    const x = (e.clientX - rect.left) * scale;
    const y = (e.clientY - rect.top) * scale;
    const col = Math.floor(x / cellSize);
    const row = Math.floor(y / cellSize);

    if (row >= 0 && row < gridSize && col >= 0 && col < gridSize) {
      setHoveredCell({ row, col, value: inputTensor[row][col] });
    } else {
      setHoveredCell(null);
    }
  };

  const placeholder = !inputTensor;

  return (
    <div className="flex flex-col items-center gap-4">
      {/* View mode toggle */}
      <div className="flex rounded-full border border-border bg-surface p-1">
        {(["image", "data"] as const).map((mode) => (
          <button
            key={mode}
            onClick={() => setViewMode(mode)}
            className={`relative rounded-full px-4 py-1.5 text-sm font-medium transition-colors ${
              viewMode === mode ? "text-background" : "text-foreground/50"
            }`}
          >
            {viewMode === mode && (
              <motion.div
                layoutId="pixelViewToggle"
                className="absolute inset-0 rounded-full bg-accent-primary"
                transition={{ type: "spring", stiffness: 300, damping: 30 }}
              />
            )}
            <span className="relative z-10 capitalize">
              {mode === "image" ? "Human View" : "Machine View"}
            </span>
          </button>
        ))}
      </div>

      {/* Canvas */}
      <div className="relative">
        <canvas
          ref={canvasRef}
          width={canvasSize}
          height={canvasSize}
          className="rounded-xl border border-border"
          style={{
            width: Math.min(canvasSize, 392),
            height: Math.min(canvasSize, 392),
            imageRendering: viewMode === "image" ? "pixelated" : "auto",
          }}
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHoveredCell(null)}
          aria-label={`28x28 pixel grid - ${viewMode} view`}
        />

        {/* Hover tooltip */}
        <AnimatePresence>
          {hoveredCell && (
            <motion.div
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="pointer-events-none absolute -top-10 left-1/2 -translate-x-1/2 rounded-md bg-surface-elevated px-3 py-1 text-xs shadow-lg"
            >
              <span className="text-foreground/50">
                [{hoveredCell.row}, {hoveredCell.col}]
              </span>{" "}
              <span className="font-mono font-bold text-accent-primary">
                {hoveredCell.value.toFixed(4)}
              </span>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Placeholder */}
        {placeholder && (
          <div className="absolute inset-0 flex items-center justify-center rounded-xl bg-surface/80">
            <p className="text-foreground/30">
              Draw a character to see pixels
            </p>
          </div>
        )}
      </div>

      <p className="max-w-sm text-center text-sm text-foreground/40">
        {viewMode === "image"
          ? "This is what you see — a character on a dark background."
          : "This is what the computer sees — a 28x28 grid of numbers between 0 and 1."}
      </p>
    </div>
  );
}
