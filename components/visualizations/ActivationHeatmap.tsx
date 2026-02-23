"use client";

import { useRef, useEffect, useMemo } from "react";
import { viridis } from "@/lib/network/networkConstants";
interface ActivationHeatmapProps {
  data: number[][];
  size?: number;
  label?: string;
  onClick?: () => void;
  selected?: boolean;
}

export function ActivationHeatmap({
  data,
  size = 80,
  label,
  onClick,
  selected,
}: ActivationHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rows = data.length;
  const cols = data[0]?.length ?? 0;

  const { min, max } = useMemo(() => {
    let mn = Infinity, mx = -Infinity;
    for (const row of data) for (const v of row) { if (v < mn) mn = v; if (v > mx) mx = v; }
    return { min: mn, max: mx };
  }, [data]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || rows === 0) return;
    const ctx = canvas.getContext("2d")!;

    const imageData = ctx.createImageData(cols, rows);
    const pixels = imageData.data;
    const range = max - min;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const t = range > 0 ? (data[r][c] - min) / range : 0;
        const [red, green, blue] = viridis(t);
        const idx = (r * cols + c) * 4;
        pixels[idx] = red;
        pixels[idx + 1] = green;
        pixels[idx + 2] = blue;
        pixels[idx + 3] = 255;
      }
    }

    ctx.putImageData(imageData, 0, 0);
  }, [data, rows, cols, min, max]);

  return (
    <div
      className={`group flex flex-col items-center gap-1 ${
        onClick ? "cursor-pointer" : ""
      }`}
      onClick={onClick}
    >
      <div
        className={`overflow-hidden rounded-md border transition-all ${
          selected
            ? "border-accent-primary shadow-lg shadow-accent-primary/20"
            : "border-border/50 group-hover:border-accent-primary/50"
        }`}
      >
        <canvas
          ref={canvasRef}
          width={cols}
          height={rows}
          style={{
            width: size,
            height: size,
            imageRendering: "pixelated",
          }}
        />
      </div>
      {label && (
        <span className="text-xs text-foreground/40 group-hover:text-foreground/60">
          {label}
        </span>
      )}
    </div>
  );
}
