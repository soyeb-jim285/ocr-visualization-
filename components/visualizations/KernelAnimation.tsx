"use client";

import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import { motion } from "framer-motion";
import { useInferenceStore } from "@/stores/inferenceStore";
import { activationColorScale, parseColor } from "@/lib/utils/colorScales";

export function KernelAnimation() {
  const inputTensor = useInferenceStore((s) => s.inputTensor);
  const layerActivations = useInferenceStore((s) => s.layerActivations);

  const inputCanvasRef = useRef<HTMLCanvasElement>(null);
  const outputCanvasRef = useRef<HTMLCanvasElement>(null);

  const [kernelPos, setKernelPos] = useState({ row: 0, col: 0 });
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(50); // ms per step
  const animRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const gridSize = 28;
  const kernelSize = 3;
  const cellSize = 10;
  const canvasSize = gridSize * cellSize;

  // Get first conv layer's first feature map as output
  const conv1Activations = layerActivations["conv1"] as
    | number[][][]
    | undefined;
  const outputMap = conv1Activations?.[0]; // First filter's output

  // Draw input with kernel overlay
  useEffect(() => {
    const canvas = inputCanvasRef.current;
    if (!canvas || !inputTensor) return;
    const ctx = canvas.getContext("2d")!;

    // Draw input image
    ctx.clearRect(0, 0, canvasSize, canvasSize);
    for (let r = 0; r < gridSize; r++) {
      for (let c = 0; c < gridSize; c++) {
        const val = Math.round(inputTensor[r][c] * 255);
        ctx.fillStyle = `rgb(${val}, ${val}, ${val})`;
        ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
      }
    }

    // Draw kernel overlay
    ctx.strokeStyle = "#06b6d4";
    ctx.lineWidth = 2;
    ctx.strokeRect(
      kernelPos.col * cellSize,
      kernelPos.row * cellSize,
      kernelSize * cellSize,
      kernelSize * cellSize
    );

    // Highlight kernel area
    ctx.fillStyle = "rgba(6, 182, 212, 0.15)";
    ctx.fillRect(
      kernelPos.col * cellSize,
      kernelPos.row * cellSize,
      kernelSize * cellSize,
      kernelSize * cellSize
    );
  }, [inputTensor, kernelPos, cellSize, canvasSize]);

  // Draw output feature map being built
  useEffect(() => {
    const canvas = outputCanvasRef.current;
    if (!canvas || !outputMap) return;
    const ctx = canvas.getContext("2d")!;

    const { max } = outputMap.flat().reduce(
      (acc, v) => ({ min: Math.min(acc.min, v), max: Math.max(acc.max, v) }),
      { min: Infinity, max: -Infinity }
    );
    const colorFn = activationColorScale(max);

    ctx.clearRect(0, 0, canvasSize, canvasSize);

    // Only draw up to current kernel position
    const currentLinear = kernelPos.row * gridSize + kernelPos.col;
    for (let r = 0; r < gridSize; r++) {
      for (let c = 0; c < gridSize; c++) {
        const linear = r * gridSize + c;
        if (linear <= currentLinear) {
          const color = colorFn(outputMap[r][c]);
          const [red, green, blue] = parseColor(color);
          ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`;
        } else {
          ctx.fillStyle = "rgba(42, 42, 58, 0.3)";
        }
        ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
      }
    }

    // Highlight current output cell
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 2;
    ctx.strokeRect(
      kernelPos.col * cellSize,
      kernelPos.row * cellSize,
      cellSize,
      cellSize
    );
  }, [outputMap, kernelPos, cellSize, canvasSize]);

  // Animation step
  const step = useCallback(() => {
    setKernelPos((prev) => {
      let nextCol = prev.col + 1;
      let nextRow = prev.row;
      if (nextCol >= gridSize) {
        nextCol = 0;
        nextRow = prev.row + 1;
      }
      if (nextRow >= gridSize) {
        setIsPlaying(false);
        return { row: 0, col: 0 };
      }
      return { row: nextRow, col: nextCol };
    });
  }, []);

  // Play/pause
  useEffect(() => {
    if (isPlaying) {
      animRef.current = setInterval(step, speed);
    } else if (animRef.current) {
      clearInterval(animRef.current);
    }
    return () => {
      if (animRef.current) clearInterval(animRef.current);
    };
  }, [isPlaying, speed, step]);

  const hasData = inputTensor && outputMap;

  return (
    <div className="flex flex-col items-center gap-6">
      <div className="flex flex-col items-center gap-6 md:flex-row md:items-start md:gap-12">
        {/* Input with kernel */}
        <div className="flex flex-col items-center gap-2">
          <span className="text-sm font-medium text-foreground/60">
            Input Image
          </span>
          <canvas
            ref={inputCanvasRef}
            width={canvasSize}
            height={canvasSize}
            className="rounded-lg border border-border"
            style={{
              width: canvasSize,
              height: canvasSize,
              imageRendering: "pixelated",
            }}
          />
        </div>

        {/* Arrow */}
        <div className="hidden items-center md:flex" style={{ marginTop: 100 }}>
          <svg
            width="48"
            height="24"
            viewBox="0 0 48 24"
            fill="none"
            className="text-accent-tertiary"
          >
            <path
              d="M0 12h40m0 0l-8-8m8 8l-8 8"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>

        {/* Output feature map */}
        <div className="flex flex-col items-center gap-2">
          <span className="text-sm font-medium text-foreground/60">
            Feature Map #1
          </span>
          <canvas
            ref={outputCanvasRef}
            width={canvasSize}
            height={canvasSize}
            className="rounded-lg border border-border"
            style={{
              width: canvasSize,
              height: canvasSize,
              imageRendering: "pixelated",
            }}
          />
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-4">
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          disabled={!hasData}
          className="flex items-center gap-2 rounded-lg border border-accent-tertiary/30 bg-accent-tertiary/10 px-4 py-2 text-sm font-medium text-accent-tertiary transition-colors hover:bg-accent-tertiary/20 disabled:opacity-30"
        >
          {isPlaying ? "Pause" : "Play"}
        </button>
        <button
          onClick={step}
          disabled={!hasData}
          className="rounded-lg border border-border px-4 py-2 text-sm text-foreground/60 transition-colors hover:border-foreground/40 disabled:opacity-30"
        >
          Step
        </button>
        <button
          onClick={() => {
            setKernelPos({ row: 0, col: 0 });
            setIsPlaying(false);
          }}
          disabled={!hasData}
          className="rounded-lg border border-border px-4 py-2 text-sm text-foreground/60 transition-colors hover:border-foreground/40 disabled:opacity-30"
        >
          Reset
        </button>
        <div className="flex items-center gap-2">
          <label className="text-xs text-foreground/40">Speed</label>
          <input
            type="range"
            min={10}
            max={200}
            value={200 - speed}
            onChange={(e) => setSpeed(200 - parseInt(e.target.value))}
            className="w-20 accent-accent-tertiary"
          />
        </div>
      </div>

      {/* Current position info */}
      <div className="font-mono text-xs text-foreground/40">
        Position: [{kernelPos.row}, {kernelPos.col}]
        {outputMap && (
          <>
            {" "}
            | Output:{" "}
            <span className="text-accent-primary">
              {outputMap[kernelPos.row]?.[kernelPos.col]?.toFixed(3) ?? "—"}
            </span>
          </>
        )}
      </div>
    </div>
  );
}
