"use client";

import { useRef, useEffect, useMemo, useState } from "react";
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
  const boxRef = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  const rows = data.length;
  const cols = data[0]?.length ?? 0;

  const { min, max } = useMemo(() => {
    let mn = Infinity, mx = -Infinity;
    for (const row of data) for (const v of row) { if (v < mn) mn = v; if (v > mx) mx = v; }
    return { min: mn, max: mx };
  }, [data]);

  // Dead channel: every value is zero (X-ray tile with a dashed outline)
  const dead = max === 0 && min === 0;

  // Mount the canvas only once the cell is near the viewport (one-shot).
  useEffect(() => {
    const el = boxRef.current;
    if (!el) return;
    const io = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) {
          setVisible(true);
          io.disconnect();
        }
      },
      { rootMargin: "200px" },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!visible || !canvas || rows === 0) return;
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
  }, [visible, data, rows, cols, min, max]);

  const interactive = !!onClick;

  return (
    <div
      className={`group flex max-w-full flex-col items-center gap-1 ${
        interactive
          ? "cursor-pointer touch-manipulation select-none [-webkit-tap-highlight-color:transparent] active:opacity-80"
          : ""
      }`}
      onClick={onClick}
      role={interactive ? "button" : undefined}
      tabIndex={interactive ? 0 : undefined}
      aria-pressed={interactive ? !!selected : undefined}
      onKeyDown={
        interactive
          ? (e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onClick?.();
              }
            }
          : undefined
      }
    >
      <div
        ref={boxRef}
        className={`tile overflow-hidden border border-rule transition-[border-color] duration-150 ${
          interactive && !selected ? "group-hover:border-rule-strong" : ""
        }`}
        data-selected={selected ? "true" : undefined}
        data-dead={dead ? "true" : undefined}
        style={{ width: size, maxWidth: "100%", aspectRatio: "1" }}
      >
        {visible && (
          <canvas
            ref={canvasRef}
            width={cols}
            height={rows}
            className="block h-full w-full"
            style={{ imageRendering: "pixelated" }}
          />
        )}
      </div>
      {label && (
        <span
          className={`font-mono mt-0.5 text-[11px] leading-none transition-colors duration-150 ${
            selected ? "text-sig" : "text-ink-3 group-hover:text-ink"
          }`}
        >
          {label}
        </span>
      )}
    </div>
  );
}
