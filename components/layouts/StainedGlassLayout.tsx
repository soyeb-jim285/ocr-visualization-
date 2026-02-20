"use client";

import React, {
  useRef,
  useState,
  useCallback,
  useMemo,
  useEffect,
} from "react";
import { useInferenceStore } from "@/stores/inferenceStore";
import { EMNIST_CLASSES } from "@/lib/model/classes";
import { useDrawingCanvas } from "@/hooks/useDrawingCanvas";
import { useInference } from "@/hooks/useInference";
import { useAnimationFrame } from "@/hooks/useAnimationFrame";
import { LayoutNav } from "@/components/layouts/LayoutNav";
import {
  loadTrainingHistory,
  type TrainingHistory,
} from "@/lib/training/trainingData";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DRAWING_CANVAS_SIZE = 200;
const DRAWING_INTERNAL_SIZE = 280;

const LAYER_KEYS = [
  "input",
  "conv1",
  "relu1",
  "conv2",
  "relu2",
  "pool1",
  "conv3",
  "relu3",
  "pool2",
] as const;

type LayerKey = (typeof LAYER_KEYS)[number];

const LAYER_DISPLAY_NAMES: Record<LayerKey, string> = {
  input: "Input",
  conv1: "Conv1",
  relu1: "ReLU1",
  conv2: "Conv2",
  relu2: "ReLU2",
  pool1: "Pool1",
  conv3: "Conv3",
  relu3: "ReLU3",
  pool2: "Pool2",
};

/** Number of channels for each layer (conv/relu layers have channels, input has 1) */
function getChannelCount(layer: LayerKey): number {
  switch (layer) {
    case "input":
      return 1;
    case "conv1":
    case "relu1":
      return 32;
    case "conv2":
    case "relu2":
      return 64;
    case "pool1":
      return 64;
    case "conv3":
    case "relu3":
      return 128;
    case "pool2":
      return 128;
  }
}

// ---------------------------------------------------------------------------
// Jewel-tone color palette
// ---------------------------------------------------------------------------

interface JewelColor {
  r: number;
  g: number;
  b: number;
}

function hexToJewel(hex: string): JewelColor {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return { r, g, b };
}

const JEWEL_STOPS: { threshold: number; color: JewelColor }[] = [
  { threshold: 0.0, color: hexToJewel("#1a237e") },   // Deep sapphire
  { threshold: 0.15, color: hexToJewel("#283593") },  // Royal blue
  { threshold: 0.3, color: hexToJewel("#1b5e20") },   // Emerald
  { threshold: 0.45, color: hexToJewel("#2e7d32") },  // Jade
  { threshold: 0.6, color: hexToJewel("#b71c1c") },   // Ruby
  { threshold: 0.75, color: hexToJewel("#c62828") },   // Garnet
  { threshold: 0.85, color: hexToJewel("#ff8f00") },   // Amber
  { threshold: 0.95, color: hexToJewel("#ffd600") },   // Gold
];

function jewelColor(t: number): JewelColor {
  const clamped = Math.max(0, Math.min(1, t));
  // Find the two stops to interpolate between
  for (let i = 0; i < JEWEL_STOPS.length - 1; i++) {
    const s0 = JEWEL_STOPS[i];
    const s1 = JEWEL_STOPS[i + 1];
    if (clamped >= s0.threshold && clamped <= s1.threshold) {
      const range = s1.threshold - s0.threshold;
      const frac = range > 0 ? (clamped - s0.threshold) / range : 0;
      return {
        r: Math.round(s0.color.r + (s1.color.r - s0.color.r) * frac),
        g: Math.round(s0.color.g + (s1.color.g - s0.color.g) * frac),
        b: Math.round(s0.color.b + (s1.color.b - s0.color.b) * frac),
      };
    }
  }
  const last = JEWEL_STOPS[JEWEL_STOPS.length - 1];
  return last.color;
}

// ---------------------------------------------------------------------------
// Seeded random
// ---------------------------------------------------------------------------

function seededRandom(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return s / 2147483647;
  };
}

// ---------------------------------------------------------------------------
// Voronoi helpers
// ---------------------------------------------------------------------------

interface VoronoiSeed {
  x: number;
  y: number;
}

function generateSeeds(
  gridW: number,
  gridH: number,
  seed: number
): VoronoiSeed[] {
  const rng = seededRandom(seed);
  let count: number;
  if (gridW >= 28) count = 50 + Math.floor(rng() * 30);
  else if (gridW >= 14) count = 25 + Math.floor(rng() * 15);
  else count = 12 + Math.floor(rng() * 4);

  const seeds: VoronoiSeed[] = [];
  for (let i = 0; i < count; i++) {
    seeds.push({ x: rng() * gridW, y: rng() * gridH });
  }
  return seeds;
}

/** Compute Voronoi cell assignments for a canvas of size (cw, ch) mapped to grid (gw, gh) */
function computeVoronoi(
  cw: number,
  ch: number,
  seeds: VoronoiSeed[],
  gw: number,
  gh: number
): { cellMap: Int16Array; seedCount: number } {
  const cellMap = new Int16Array(cw * ch);
  const scaleX = gw / cw;
  const scaleY = gh / ch;

  for (let py = 0; py < ch; py++) {
    const gy = py * scaleY;
    for (let px = 0; px < cw; px++) {
      const gx = px * scaleX;
      let minDist = Infinity;
      let minIdx = 0;
      for (let s = 0; s < seeds.length; s++) {
        const dx = gx - seeds[s].x;
        const dy = gy - seeds[s].y;
        const dist = dx * dx + dy * dy;
        if (dist < minDist) {
          minDist = dist;
          minIdx = s;
        }
      }
      cellMap[py * cw + px] = minIdx;
    }
  }
  return { cellMap, seedCount: seeds.length };
}

/** Compute average activation per cell */
function computeCellValues(
  data: number[][],
  cellMap: Int16Array,
  cw: number,
  ch: number,
  seedCount: number
): number[] {
  const sums = new Float64Array(seedCount);
  const counts = new Float64Array(seedCount);
  const gw = data[0]?.length ?? 1;
  const gh = data.length;
  const scaleX = gw / cw;
  const scaleY = gh / ch;

  for (let py = 0; py < ch; py++) {
    const gy = Math.min(Math.floor(py * scaleY), gh - 1);
    for (let px = 0; px < cw; px++) {
      const gx = Math.min(Math.floor(px * scaleX), gw - 1);
      const cellIdx = cellMap[py * cw + px];
      sums[cellIdx] += data[gy][gx];
      counts[cellIdx]++;
    }
  }

  const values: number[] = new Array(seedCount);
  for (let i = 0; i < seedCount; i++) {
    values[i] = counts[i] > 0 ? sums[i] / counts[i] : 0;
  }
  return values;
}

// ---------------------------------------------------------------------------
// 1. StainedGlassPanel (Canvas2D component)
// ---------------------------------------------------------------------------

interface StainedGlassPanelProps {
  data: number[][] | null;
  width: number;
  height: number;
  seed?: number;
  showArch?: boolean;
}

function StainedGlassPanel({
  data,
  width,
  height,
  seed = 42,
  showArch = true,
}: StainedGlassPanelProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Memoize seeds and voronoi cell map
  const gridW = data?.[0]?.length ?? 28;
  const gridH = data?.length ?? 28;

  const voronoiData = useMemo(() => {
    const seeds = generateSeeds(gridW, gridH, seed);
    const { cellMap, seedCount } = computeVoronoi(
      width,
      height,
      seeds,
      gridW,
      gridH
    );
    return { seeds, cellMap, seedCount };
  }, [gridW, gridH, seed, width, height]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear
    ctx.clearRect(0, 0, width, height);

    if (!data) {
      // Draw empty dark panel
      ctx.fillStyle = "#1a237e";
      ctx.fillRect(0, 0, width, height);
      if (showArch) drawGothicArch(ctx, width, height);
      return;
    }

    const { seeds, cellMap, seedCount } = voronoiData;

    // Normalize data to [0, 1]
    let maxVal = 0;
    for (let r = 0; r < data.length; r++) {
      for (let c = 0; c < data[r].length; c++) {
        if (data[r][c] > maxVal) maxVal = data[r][c];
      }
    }
    if (maxVal === 0) maxVal = 1;

    const normalizedData = data.map((row) => row.map((v) => v / maxVal));
    const cellValues = computeCellValues(
      normalizedData,
      cellMap,
      width,
      height,
      seedCount
    );

    // Create ImageData for pixel-level rendering
    const imageData = ctx.createImageData(width, height);
    const pixels = imageData.data;

    // Precompute cell colors
    const cellColors: JewelColor[] = cellValues.map((v) => jewelColor(v));

    // Fill pixels with cell colors
    for (let py = 0; py < height; py++) {
      for (let px = 0; px < width; px++) {
        const idx = py * width + px;
        const cellIdx = cellMap[idx];
        const color = cellColors[cellIdx];
        const pIdx = idx * 4;
        pixels[pIdx] = color.r;
        pixels[pIdx + 1] = color.g;
        pixels[pIdx + 2] = color.b;
        pixels[pIdx + 3] = 255;
      }
    }

    ctx.putImageData(imageData, 0, 0);

    // Inner glow: radial gradient for each seed
    for (let s = 0; s < seeds.length; s++) {
      const sx = (seeds[s].x / gridW) * width;
      const sy = (seeds[s].y / gridH) * height;
      const grad = ctx.createRadialGradient(sx, sy, 0, sx, sy, 20);
      grad.addColorStop(0, "rgba(255,255,255,0.12)");
      grad.addColorStop(1, "rgba(255,255,255,0)");
      ctx.fillStyle = grad;
      ctx.fillRect(
        Math.max(0, sx - 20),
        Math.max(0, sy - 20),
        40,
        40
      );
    }

    // Lead lines: detect cell boundaries
    ctx.strokeStyle = "#2a2a2a";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let py = 0; py < height - 1; py++) {
      for (let px = 0; px < width - 1; px++) {
        const idx = py * width + px;
        const c = cellMap[idx];
        // Check right neighbor
        if (cellMap[idx + 1] !== c) {
          ctx.moveTo(px + 0.5, py);
          ctx.lineTo(px + 0.5, py + 1);
        }
        // Check bottom neighbor
        if (cellMap[(py + 1) * width + px] !== c) {
          ctx.moveTo(px, py + 0.5);
          ctx.lineTo(px + 1, py + 0.5);
        }
      }
    }
    ctx.stroke();

    // Gothic arch frame
    if (showArch) {
      drawGothicArch(ctx, width, height);
    }
  }, [data, width, height, voronoiData, showArch]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{ width, height, display: "block" }}
    />
  );
}

function drawGothicArch(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number
) {
  ctx.save();
  // Build Gothic arch shape as a clip path then stroke border
  const archTop = h * 0.05;
  const archBottom = h * 0.98;
  const archLeft = w * 0.05;
  const archRight = w * 0.95;
  const centerX = w / 2;
  const archPeakY = archTop;
  const cpOffset = (archRight - archLeft) * 0.6;

  // Draw arch outline
  ctx.beginPath();
  ctx.moveTo(archLeft, archBottom);
  ctx.lineTo(archLeft, h * 0.35);
  // Left curve of pointed arch
  ctx.quadraticCurveTo(archLeft, archPeakY + cpOffset * 0.3, centerX, archPeakY);
  // Right curve of pointed arch
  ctx.quadraticCurveTo(archRight, archPeakY + cpOffset * 0.3, archRight, h * 0.35);
  ctx.lineTo(archRight, archBottom);
  ctx.closePath();

  // Stone border gradient
  const grad = ctx.createLinearGradient(0, 0, w, 0);
  grad.addColorStop(0, "#3a3a3a");
  grad.addColorStop(0.5, "#4a4a4a");
  grad.addColorStop(1, "#3a3a3a");
  ctx.strokeStyle = grad;
  ctx.lineWidth = 4;
  ctx.stroke();

  // Inner thinner border
  ctx.strokeStyle = "#2a2a2a";
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.restore();
}

// ---------------------------------------------------------------------------
// 2. RoseWindow (Canvas2D)
// ---------------------------------------------------------------------------

interface RoseWindowProps {
  prediction: number[] | null;
  topPrediction: { classIndex: number; confidence: number } | null;
  size?: number;
}

function RoseWindow({
  prediction,
  topPrediction,
  size = 300,
}: RoseWindowProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const cx = size / 2;
    const cy = size / 2;
    const innerR = size * 0.13;
    const maxOuterR = size * 0.37;
    const minOuterR = innerR;

    ctx.clearRect(0, 0, size, size);

    // Background dark circle
    ctx.beginPath();
    ctx.arc(cx, cy, size / 2 - 4, 0, Math.PI * 2);
    ctx.fillStyle = "#1a1510";
    ctx.fill();

    const numClasses = 62;
    const wedgeAngle = (Math.PI * 2) / numClasses;

    // Concentric reference circles
    const refRadii = [0.25, 0.5, 0.75];
    ctx.strokeStyle = "rgba(51,51,51,0.6)";
    ctx.lineWidth = 0.5;
    for (const frac of refRadii) {
      const r = innerR + (maxOuterR - innerR) * frac;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Draw wedges
    for (let i = 0; i < numClasses; i++) {
      const startAngle = -Math.PI / 2 + i * wedgeAngle;
      const endAngle = startAngle + wedgeAngle;
      const prob = prediction ? prediction[i] : 0;
      const outerR = minOuterR + prob * (maxOuterR - minOuterR);

      // Color based on class index
      const hue = (i / numClasses) * 360;
      const isTop = topPrediction?.classIndex === i;

      ctx.beginPath();
      ctx.moveTo(
        cx + Math.cos(startAngle) * innerR,
        cy + Math.sin(startAngle) * innerR
      );
      ctx.arc(cx, cy, outerR, startAngle, endAngle);
      ctx.lineTo(
        cx + Math.cos(endAngle) * innerR,
        cy + Math.sin(endAngle) * innerR
      );
      ctx.arc(cx, cy, innerR, endAngle, startAngle, true);
      ctx.closePath();

      ctx.fillStyle = `hsl(${hue}, 80%, 40%)`;
      ctx.fill();

      // Lead lines between wedges
      ctx.beginPath();
      ctx.moveTo(
        cx + Math.cos(startAngle) * innerR,
        cy + Math.sin(startAngle) * innerR
      );
      ctx.lineTo(
        cx + Math.cos(startAngle) * maxOuterR,
        cy + Math.sin(startAngle) * maxOuterR
      );
      ctx.strokeStyle = "#333";
      ctx.lineWidth = 0.5;
      ctx.stroke();

      // Golden glow for top prediction
      if (isTop && prediction) {
        ctx.beginPath();
        ctx.moveTo(
          cx + Math.cos(startAngle) * innerR,
          cy + Math.sin(startAngle) * innerR
        );
        ctx.arc(cx, cy, maxOuterR, startAngle, endAngle);
        ctx.lineTo(
          cx + Math.cos(endAngle) * innerR,
          cy + Math.sin(endAngle) * innerR
        );
        ctx.arc(cx, cy, innerR, endAngle, startAngle, true);
        ctx.closePath();
        ctx.strokeStyle = "#ffd600";
        ctx.lineWidth = 2.5;
        ctx.shadowColor = "#ffd600";
        ctx.shadowBlur = 8;
        ctx.stroke();
        ctx.shadowBlur = 0;
      }
    }

    // Gothic tracery - small decorative circles near outer rim
    ctx.strokeStyle = "rgba(100,100,100,0.4)";
    ctx.lineWidth = 0.5;
    for (let i = 0; i < numClasses; i += 4) {
      const angle = -Math.PI / 2 + (i + 0.5) * wedgeAngle;
      const tr = maxOuterR + 8;
      ctx.beginPath();
      ctx.arc(
        cx + Math.cos(angle) * tr,
        cy + Math.sin(angle) * tr,
        3,
        0,
        Math.PI * 2
      );
      ctx.stroke();
    }

    // Outer stone rim
    ctx.beginPath();
    ctx.arc(cx, cy, size / 2 - 4, 0, Math.PI * 2);
    ctx.strokeStyle = "#3a3a3a";
    ctx.lineWidth = 6;
    ctx.stroke();
    ctx.strokeStyle = "#4a4a4a";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Center circle
    ctx.beginPath();
    ctx.arc(cx, cy, innerR, 0, Math.PI * 2);
    ctx.fillStyle = "#1a1510";
    ctx.fill();
    ctx.strokeStyle = "#3a3a3a";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Center: predicted character
    if (topPrediction) {
      const char = EMNIST_CLASSES[topPrediction.classIndex] ?? "?";
      ctx.font = `60px Georgia, "Times New Roman", serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.shadowColor = "#ffd600";
      ctx.shadowBlur = 15;
      ctx.fillStyle = "#ffd600";
      ctx.fillText(char, cx, cy);
      ctx.shadowBlur = 0;
    }
  }, [prediction, topPrediction, size]);

  return (
    <canvas
      ref={canvasRef}
      width={size}
      height={size}
      style={{ width: size, height: size, display: "block" }}
    />
  );
}

// ---------------------------------------------------------------------------
// 4. IlluminatedTimeline (Canvas2D)
// ---------------------------------------------------------------------------

interface IlluminatedTimelineProps {
  history: TrainingHistory | null;
  width: number;
  height: number;
}

function IlluminatedTimeline({
  history,
  width,
  height,
}: IlluminatedTimelineProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const tooltipRef = useRef<{
    epoch: number;
    x: number;
    y: number;
    values: { loss: number; acc: number; vLoss: number; vAcc: number };
  } | null>(null);
  const [tooltipTick, setTooltipTick] = useState(0);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!history) return;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const mx = ((e.clientX - rect.left) / rect.width) * width;
      const my = ((e.clientY - rect.top) / rect.height) * height;

      const padL = 50;
      const padR = 50;
      const plotW = width - padL - padR;
      const epochs = history.loss.length;
      const epoch = Math.round(((mx - padL) / plotW) * (epochs - 1));
      if (epoch >= 0 && epoch < epochs) {
        tooltipRef.current = {
          epoch,
          x: mx,
          y: my,
          values: {
            loss: history.loss[epoch],
            acc: history.accuracy[epoch],
            vLoss: history.val_loss[epoch],
            vAcc: history.val_accuracy[epoch],
          },
        };
      } else {
        tooltipRef.current = null;
      }
      setTooltipTick((t) => t + 1);
    },
    [history, width, height]
  );

  const handleMouseLeave = useCallback(() => {
    tooltipRef.current = null;
    setTooltipTick((t) => t + 1);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Parchment background
    ctx.fillStyle = "#f5e6c8";
    ctx.fillRect(0, 0, width, height);

    // Subtle paper texture noise
    const noiseData = ctx.getImageData(0, 0, width, height);
    const rng = seededRandom(777);
    for (let i = 0; i < noiseData.data.length; i += 4) {
      const noise = (rng() - 0.5) * 15;
      noiseData.data[i] = Math.max(0, Math.min(255, noiseData.data[i] + noise));
      noiseData.data[i + 1] = Math.max(
        0,
        Math.min(255, noiseData.data[i + 1] + noise)
      );
      noiseData.data[i + 2] = Math.max(
        0,
        Math.min(255, noiseData.data[i + 2] + noise)
      );
    }
    ctx.putImageData(noiseData, 0, 0);

    // Decorative golden border with corner ornaments
    ctx.strokeStyle = "#c8a600";
    ctx.lineWidth = 2;
    ctx.strokeRect(4, 4, width - 8, height - 8);

    // Corner ornaments (diamond shapes)
    const corners = [
      [8, 8],
      [width - 8, 8],
      [8, height - 8],
      [width - 8, height - 8],
    ];
    ctx.fillStyle = "#c8a600";
    for (const [cx, cy] of corners) {
      ctx.beginPath();
      ctx.moveTo(cx, cy - 5);
      ctx.lineTo(cx + 5, cy);
      ctx.lineTo(cx, cy + 5);
      ctx.lineTo(cx - 5, cy);
      ctx.closePath();
      ctx.fill();
    }

    if (!history) {
      ctx.fillStyle = "#8b7d5e";
      ctx.font = 'italic 14px Georgia, "Times New Roman", serif';
      ctx.textAlign = "center";
      ctx.fillText("Awaiting training history...", width / 2, height / 2);
      return;
    }

    const epochs = history.loss.length;
    const padL = 50;
    const padR = 50;
    const padT = 25;
    const padB = 35;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;

    // Max values for axes
    const maxLoss = Math.max(...history.loss, ...history.val_loss) * 1.1;
    const maxAcc = 1.0;

    // Axes
    ctx.strokeStyle = "#8b7d5e";
    ctx.lineWidth = 1;
    // Left Y axis (loss)
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, padT + plotH);
    ctx.stroke();
    // Right Y axis (accuracy)
    ctx.beginPath();
    ctx.moveTo(padL + plotW, padT);
    ctx.lineTo(padL + plotW, padT + plotH);
    ctx.stroke();
    // X axis
    ctx.beginPath();
    ctx.moveTo(padL, padT + plotH);
    ctx.lineTo(padL + plotW, padT + plotH);
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = "#8b7d5e";
    ctx.font = '10px Georgia, "Times New Roman", serif';
    ctx.textAlign = "center";

    // X-axis tick labels
    for (let e = 0; e < epochs; e += 5) {
      const x = padL + (e / (epochs - 1)) * plotW;
      ctx.fillText(String(e), x, padT + plotH + 14);
      ctx.beginPath();
      ctx.moveTo(x, padT + plotH);
      ctx.lineTo(x, padT + plotH + 4);
      ctx.stroke();
    }

    // Left Y labels (loss)
    ctx.textAlign = "right";
    for (let i = 0; i <= 4; i++) {
      const v = (maxLoss / 4) * i;
      const y = padT + plotH - (v / maxLoss) * plotH;
      ctx.fillText(v.toFixed(2), padL - 5, y + 3);
    }

    // Right Y labels (accuracy)
    ctx.textAlign = "left";
    for (let i = 0; i <= 4; i++) {
      const v = (maxAcc / 4) * i;
      const y = padT + plotH - (v / maxAcc) * plotH;
      ctx.fillText(v.toFixed(2), padL + plotW + 5, y + 3);
    }

    // Helper to draw a curve
    const drawCurve = (
      c: CanvasRenderingContext2D,
      values: number[],
      maxV: number,
      color: string,
      dashed: boolean
    ) => {
      c.beginPath();
      c.strokeStyle = color;
      c.lineWidth = 2;
      c.shadowColor = "rgba(0,0,0,0.3)";
      c.shadowBlur = 2;
      if (dashed) c.setLineDash([6, 4]);
      else c.setLineDash([]);

      for (let i = 0; i < values.length; i++) {
        const x = padL + (i / (values.length - 1)) * plotW;
        const y = padT + plotH - (values[i] / maxV) * plotH;
        if (i === 0) c.moveTo(x, y);
        else c.lineTo(x, y);
      }
      c.stroke();
      c.shadowBlur = 0;
      c.setLineDash([]);
    };

    // Draw curves
    drawCurve(ctx, history.loss, maxLoss, "#c8a600", false); // Gold - loss
    drawCurve(ctx, history.val_loss, maxLoss, "#c8a600", true); // Gold dashed - val_loss
    drawCurve(ctx, history.accuracy, maxAcc, "#a0a0b0", false); // Silver - accuracy
    drawCurve(ctx, history.val_accuracy, maxAcc, "#a0a0b0", true); // Silver dashed - val_acc

    // Epoch markers every 5th epoch
    ctx.fillStyle = "#c8a600";
    for (let e = 0; e < epochs; e += 5) {
      const x = padL + (e / (epochs - 1)) * plotW;
      const yLoss = padT + plotH - (history.loss[e] / maxLoss) * plotH;
      ctx.beginPath();
      ctx.arc(x, yLoss, 3, 0, Math.PI * 2);
      ctx.fill();
    }

    // Tooltip
    const tt = tooltipRef.current;
    if (tt && tt.epoch >= 0 && tt.epoch < epochs) {
      const ttW = 140;
      const ttH = 68;
      let ttX = tt.x + 10;
      let ttY = tt.y - ttH - 5;
      if (ttX + ttW > width) ttX = tt.x - ttW - 10;
      if (ttY < 0) ttY = tt.y + 10;

      ctx.fillStyle = "rgba(40,35,25,0.9)";
      ctx.fillRect(ttX, ttY, ttW, ttH);
      ctx.strokeStyle = "#c8a600";
      ctx.lineWidth = 1;
      ctx.strokeRect(ttX, ttY, ttW, ttH);

      ctx.fillStyle = "#f5e6c8";
      ctx.font = 'italic 11px Georgia, "Times New Roman", serif';
      ctx.textAlign = "left";
      ctx.fillText(`Epoch ${tt.epoch}`, ttX + 6, ttY + 14);
      ctx.font = '10px Georgia, "Times New Roman", serif';
      ctx.fillText(`Loss: ${tt.values.loss.toFixed(4)}`, ttX + 6, ttY + 28);
      ctx.fillText(`Acc: ${tt.values.acc.toFixed(4)}`, ttX + 6, ttY + 40);
      ctx.fillText(`V.Loss: ${tt.values.vLoss.toFixed(4)}`, ttX + 6, ttY + 52);
      ctx.fillText(`V.Acc: ${tt.values.vAcc.toFixed(4)}`, ttX + 6, ttY + 64);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [history, width, height, tooltipTick]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{ width: "100%", height: height, display: "block" }}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
    />
  );
}

// ---------------------------------------------------------------------------
// 5. LightRayAnimation (Canvas overlay)
// ---------------------------------------------------------------------------

function LightRayAnimation({ hasData }: { hasData: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const offsetRef = useRef(0);

  const draw = useCallback(
    (_dt: number, elapsed: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      // Match viewport
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      if (canvas.width !== vw || canvas.height !== vh) {
        canvas.width = vw;
        canvas.height = vh;
      }

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.clearRect(0, 0, vw, vh);

      const cycleMs = 6000;
      const diag = Math.sqrt(vw * vw + vh * vh);
      offsetRef.current = (elapsed % cycleMs) / cycleMs;

      const alpha = hasData ? 0.12 : 0.08;
      const rayWidth = 20;
      const numRays = 4;
      const angle = -Math.PI / 6; // ~30deg

      ctx.save();
      ctx.translate(vw / 2, vh / 2);
      ctx.rotate(angle);

      for (let i = 0; i < numRays; i++) {
        const spacing = diag / numRays;
        const baseX =
          -diag / 2 +
          i * spacing +
          offsetRef.current * diag -
          diag;
        // Two passes for wrapping
        for (const shift of [0, diag]) {
          const rx = baseX + shift;
          const grad = ctx.createLinearGradient(rx, 0, rx + rayWidth, 0);
          grad.addColorStop(0, "rgba(255,215,0,0)");
          grad.addColorStop(0.5, `rgba(255,215,0,${alpha})`);
          grad.addColorStop(1, "rgba(255,215,0,0)");
          ctx.fillStyle = grad;
          ctx.fillRect(rx, -diag, rayWidth, diag * 2);
        }
      }

      ctx.restore();
    },
    [hasData]
  );

  useAnimationFrame(draw, true);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        width: "100vw",
        height: "100vh",
        pointerEvents: "none",
        zIndex: 50,
      }}
    />
  );
}

// ---------------------------------------------------------------------------
// 6. LayerSelector (styled as carved stone tabs)
// ---------------------------------------------------------------------------

interface LayerSelectorProps {
  selectedLayer: LayerKey;
  onSelectLayer: (layer: LayerKey) => void;
  selectedChannel: number;
  onSelectChannel: (ch: number) => void;
}

function LayerSelector({
  selectedLayer,
  onSelectLayer,
  selectedChannel,
  onSelectChannel,
}: LayerSelectorProps) {
  const channelCount = getChannelCount(selectedLayer);
  const maxDisplay = 16;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {/* Layer tabs */}
      <div
        style={{
          display: "flex",
          gap: 2,
          justifyContent: "center",
          flexWrap: "wrap",
        }}
      >
        {LAYER_KEYS.map((key) => {
          const active = key === selectedLayer;
          return (
            <button
              key={key}
              onClick={() => {
                onSelectLayer(key);
                onSelectChannel(0);
              }}
              style={{
                padding: "6px 14px",
                fontFamily: 'Georgia, "Times New Roman", serif',
                fontVariant: "small-caps",
                fontSize: 13,
                color: active ? "#ffd600" : "#d4c5a0",
                background: active
                  ? "linear-gradient(180deg, #3a3a3a 0%, #2a2a2a 100%)"
                  : "linear-gradient(180deg, #333 0%, #222 100%)",
                border: "none",
                borderBottom: active ? "2px solid #c8a600" : "2px solid transparent",
                borderRadius: "4px 4px 0 0",
                cursor: "pointer",
                boxShadow: active
                  ? "inset 0 1px 0 rgba(255,255,255,0.1)"
                  : "inset 0 -2px 4px rgba(0,0,0,0.3)",
                transition: "all 0.15s",
              }}
            >
              {LAYER_DISPLAY_NAMES[key]}
            </button>
          );
        })}
      </div>

      {/* Channel selector */}
      {channelCount > 1 && (
        <div
          style={{
            display: "flex",
            gap: 2,
            overflowX: "auto",
            justifyContent: "center",
            padding: "4px 0",
            maxWidth: "100%",
          }}
        >
          {Array.from({ length: Math.min(channelCount, maxDisplay) }, (_, i) => (
            <button
              key={i}
              onClick={() => onSelectChannel(i)}
              style={{
                width: 28,
                height: 24,
                fontFamily: 'Georgia, "Times New Roman", serif',
                fontSize: 11,
                color: i === selectedChannel ? "#ffd600" : "#d4c5a0",
                background:
                  i === selectedChannel
                    ? "linear-gradient(180deg, #3a3a3a, #2a2a2a)"
                    : "#252525",
                border:
                  i === selectedChannel
                    ? "1px solid #c8a600"
                    : "1px solid #3a3a3a",
                borderRadius: 3,
                cursor: "pointer",
                flexShrink: 0,
              }}
            >
              {i + 1}
            </button>
          ))}
          {channelCount > maxDisplay && (
            <span
              style={{
                color: "#8b7d5e",
                fontSize: 11,
                fontFamily: 'Georgia, "Times New Roman", serif',
                alignSelf: "center",
                paddingLeft: 4,
                whiteSpace: "nowrap",
              }}
            >
              +{channelCount - maxDisplay} more
            </span>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// 7. DrawingCanvasPanel (manuscript-style)
// ---------------------------------------------------------------------------

function DrawingCanvasPanel() {
  const { infer } = useInference();
  const onStrokeEnd = useCallback(
    (imageData: ImageData) => {
      infer(imageData);
    },
    [infer]
  );

  const { canvasRef, clear, hasDrawn, startDrawing, draw, stopDrawing } =
    useDrawingCanvas({
      width: DRAWING_INTERNAL_SIZE,
      height: DRAWING_INTERNAL_SIZE,
      lineWidth: 16,
      strokeColor: "#ffffff",
      backgroundColor: "#000000",
      onStrokeEnd,
    });

  return (
    <div
      style={{
        position: "absolute",
        top: 16,
        left: 16,
        zIndex: 40,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 6,
      }}
    >
      <span
        style={{
          fontFamily: 'Georgia, "Times New Roman", serif',
          fontStyle: "italic",
          fontSize: 13,
          color: "#d4c5a0",
        }}
      >
        Inscribe thy character
      </span>
      <div
        style={{
          border: "4px solid #c8a600",
          borderRadius: 4,
          position: "relative",
          lineHeight: 0,
          boxShadow: "0 0 12px rgba(200,166,0,0.3)",
        }}
      >
        {/* Corner decorations */}
        {[
          { top: -6, left: -6 },
          { top: -6, right: -6 },
          { bottom: -6, left: -6 },
          { bottom: -6, right: -6 },
        ].map((pos, i) => (
          <div
            key={i}
            style={{
              position: "absolute",
              width: 8,
              height: 8,
              background: "#c8a600",
              borderRadius: "50%",
              ...pos,
            } as React.CSSProperties}
          />
        ))}
        <canvas
          ref={canvasRef}
          width={DRAWING_INTERNAL_SIZE}
          height={DRAWING_INTERNAL_SIZE}
          style={{
            width: DRAWING_CANVAS_SIZE,
            height: DRAWING_CANVAS_SIZE,
            cursor: "crosshair",
            display: "block",
          }}
          onMouseDown={(e) => startDrawing(e.nativeEvent)}
          onMouseMove={(e) => draw(e.nativeEvent)}
          onMouseUp={() => stopDrawing()}
          onMouseLeave={() => stopDrawing()}
          onTouchStart={(e) => {
            e.preventDefault();
            startDrawing(e.nativeEvent as unknown as TouchEvent);
          }}
          onTouchMove={(e) => {
            e.preventDefault();
            draw(e.nativeEvent as unknown as TouchEvent);
          }}
          onTouchEnd={() => stopDrawing()}
        />
      </div>
      {hasDrawn && (
        <button
          onClick={clear}
          style={{
            fontFamily: 'Georgia, "Times New Roman", serif',
            fontStyle: "italic",
            fontSize: 12,
            color: "#d4c5a0",
            background: "transparent",
            border: "1px solid #c8a600",
            borderRadius: 4,
            padding: "3px 14px",
            cursor: "pointer",
          }}
        >
          Erase
        </button>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Helper: Extract 2D activation map for a layer + channel
// ---------------------------------------------------------------------------

function getActivationMap(
  layer: LayerKey,
  channel: number,
  inputTensor: number[][] | null,
  layerActivations: Record<string, number[][][] | number[]>
): number[][] | null {
  if (layer === "input") {
    return inputTensor;
  }
  const act = layerActivations[layer];
  if (!act) return null;

  // 3D activation: [channels][H][W]
  if (
    Array.isArray(act) &&
    act.length > 0 &&
    Array.isArray(act[0]) &&
    Array.isArray((act as number[][][])[0][0])
  ) {
    const act3d = act as number[][][];
    if (channel >= 0 && channel < act3d.length) {
      return act3d[channel];
    }
    return act3d[0];
  }

  return null;
}

// ---------------------------------------------------------------------------
// Stone pillar component
// ---------------------------------------------------------------------------

function StonePillar() {
  return (
    <div
      style={{
        width: 20,
        minHeight: "100%",
        background:
          "repeating-linear-gradient(180deg, #3a3a3a 0px, #3a3a3a 18px, #2a2a2a 18px, #2a2a2a 20px, #3e3e3e 20px, #3e3e3e 38px, #2a2a2a 38px, #2a2a2a 40px)",
        borderLeft: "1px solid #222",
        borderRight: "1px solid #222",
        flexShrink: 0,
      }}
    />
  );
}

// ---------------------------------------------------------------------------
// Side aisle: Stack of small stained glass thumbnails
// ---------------------------------------------------------------------------

interface SideAisleProps {
  layer: LayerKey;
  startChannel: number;
  endChannel: number;
  selectedChannel: number;
  onSelectChannel: (ch: number) => void;
  inputTensor: number[][] | null;
  layerActivations: Record<string, number[][][] | number[]>;
}

function SideAisle({
  layer,
  startChannel,
  endChannel,
  selectedChannel,
  onSelectChannel,
  inputTensor,
  layerActivations,
}: SideAisleProps) {
  const totalChannels = getChannelCount(layer);
  const channelIndices: number[] = [];
  for (let i = startChannel; i < Math.min(endChannel, totalChannels); i++) {
    channelIndices.push(i);
  }

  return (
    <div
      style={{
        width: 200,
        display: "flex",
        flexDirection: "column",
        gap: 10,
        padding: "10px 8px",
        overflowY: "auto",
        flexShrink: 0,
      }}
    >
      {channelIndices.map((ch) => {
        const mapData = getActivationMap(
          layer,
          ch,
          inputTensor,
          layerActivations
        );
        const isSelected = ch === selectedChannel;
        return (
          <div
            key={ch}
            onClick={() => onSelectChannel(ch)}
            style={{
              cursor: "pointer",
              border: isSelected
                ? "2px solid #c8a600"
                : "2px solid transparent",
              borderRadius: 4,
              transition: "border-color 0.2s",
              padding: 2,
            }}
          >
            <div
              style={{
                textAlign: "center",
                fontFamily: 'Georgia, "Times New Roman", serif',
                fontSize: 10,
                color: "#8b7d5e",
                marginBottom: 2,
              }}
            >
              Ch {ch + 1}
            </div>
            <StainedGlassPanel
              data={mapData}
              width={140}
              height={160}
              seed={ch * 137 + 7}
              showArch={true}
            />
          </div>
        );
      })}
      {channelIndices.length === 0 && (
        <div
          style={{
            textAlign: "center",
            fontFamily: 'Georgia, "Times New Roman", serif',
            fontStyle: "italic",
            fontSize: 12,
            color: "#5a5040",
            padding: 20,
          }}
        >
          No channels
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main export: StainedGlassLayout
// ---------------------------------------------------------------------------

export function StainedGlassLayout() {
  const [selectedLayer, setSelectedLayer] = useState<LayerKey>("conv1");
  const [selectedChannel, setSelectedChannel] = useState(0);
  const [trainingHistory, setTrainingHistory] = useState<TrainingHistory | null>(
    null
  );

  const inputTensor = useInferenceStore((s) => s.inputTensor);
  const layerActivations = useInferenceStore((s) => s.layerActivations);
  const prediction = useInferenceStore((s) => s.prediction);
  const topPrediction = useInferenceStore((s) => s.topPrediction);

  const hasData = inputTensor !== null;

  // Load training history
  useEffect(() => {
    loadTrainingHistory()
      .then(setTrainingHistory)
      .catch(() => {});
  }, []);

  // Get the active activation map for the main panel
  const mainMapData = useMemo(
    () =>
      getActivationMap(
        selectedLayer,
        selectedChannel,
        inputTensor,
        layerActivations
      ),
    [selectedLayer, selectedChannel, inputTensor, layerActivations]
  );

  return (
    <div
      style={{
        position: "relative",
        width: "100vw",
        height: "100vh",
        overflow: "hidden",
        background: "#1a1510",
        fontFamily: 'Georgia, "Times New Roman", serif',
      }}
    >
      {/* Light ray overlay */}
      <LightRayAnimation hasData={hasData} />

      {/* Drawing canvas (floating top-left) */}
      <DrawingCanvasPanel />

      {/* Main cathedral layout */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          height: "100vh",
          position: "relative",
          zIndex: 1,
        }}
      >
        {/* Gothic arch header with layer selector */}
        <div
          style={{
            padding: "12px 240px 0",
            background:
              "linear-gradient(180deg, rgba(30,25,16,1) 0%, rgba(26,21,16,0) 100%)",
            position: "relative",
            zIndex: 10,
          }}
        >
          {/* Gothic arch decoration */}
          <div
            style={{
              position: "absolute",
              top: 0,
              left: "50%",
              transform: "translateX(-50%)",
              width: 400,
              height: 6,
              background:
                "linear-gradient(90deg, transparent, #3a3a3a 20%, #4a4a4a 50%, #3a3a3a 80%, transparent)",
              borderRadius: "0 0 50% 50%",
            }}
          />
          <LayerSelector
            selectedLayer={selectedLayer}
            onSelectLayer={setSelectedLayer}
            selectedChannel={selectedChannel}
            onSelectChannel={setSelectedChannel}
          />
        </div>

        {/* Cathedral nave: 3 columns with pillars */}
        <div
          style={{
            flex: 1,
            display: "flex",
            overflow: "hidden",
            minHeight: 0,
          }}
        >
          {/* Left aisle */}
          <SideAisle
            layer={selectedLayer}
            startChannel={0}
            endChannel={8}
            selectedChannel={selectedChannel}
            onSelectChannel={setSelectedChannel}
            inputTensor={inputTensor}
            layerActivations={layerActivations}
          />

          {/* Left pillar */}
          <StonePillar />

          {/* Central nave */}
          <div
            style={{
              flex: 1,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              padding: 20,
              minWidth: 0,
            }}
          >
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 8,
              }}
            >
              <div
                style={{
                  fontVariant: "small-caps",
                  color: "#8b7d5e",
                  fontSize: 14,
                  letterSpacing: 2,
                }}
              >
                {LAYER_DISPLAY_NAMES[selectedLayer]} &mdash; Channel{" "}
                {selectedChannel + 1}
              </div>
              <StainedGlassPanel
                data={mainMapData}
                width={300}
                height={360}
                seed={selectedChannel * 137 + 7}
                showArch={true}
              />
              {!hasData && (
                <div
                  style={{
                    fontStyle: "italic",
                    color: "#5a5040",
                    fontSize: 13,
                    marginTop: 8,
                  }}
                >
                  Inscribe a character to illuminate the windows
                </div>
              )}
            </div>
          </div>

          {/* Right pillar */}
          <StonePillar />

          {/* Right aisle */}
          <SideAisle
            layer={selectedLayer}
            startChannel={8}
            endChannel={16}
            selectedChannel={selectedChannel}
            onSelectChannel={setSelectedChannel}
            inputTensor={inputTensor}
            layerActivations={layerActivations}
          />
        </div>

        {/* Bottom section: Rose window + Timeline */}
        <div
          style={{
            display: "flex",
            alignItems: "flex-end",
            padding: "0 16px 40px",
            gap: 16,
            flexShrink: 0,
          }}
        >
          {/* Rose window */}
          <div
            style={{
              flexShrink: 0,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 6,
            }}
          >
            <span
              style={{
                fontVariant: "small-caps",
                color: "#8b7d5e",
                fontSize: 12,
                letterSpacing: 1,
              }}
            >
              Rose Window of Predictions
            </span>
            <RoseWindow
              prediction={prediction}
              topPrediction={topPrediction}
              size={300}
            />
          </div>

          {/* Illuminated timeline */}
          <div
            style={{
              flex: 1,
              display: "flex",
              flexDirection: "column",
              gap: 6,
              minWidth: 0,
            }}
          >
            <span
              style={{
                fontVariant: "small-caps",
                color: "#8b7d5e",
                fontSize: 12,
                letterSpacing: 1,
              }}
            >
              Illuminated Training Chronicle
            </span>
            <IlluminatedTimeline
              history={trainingHistory}
              width={600}
              height={200}
            />
          </div>
        </div>
      </div>

      {/* Layout nav */}
      <LayoutNav />
    </div>
  );
}
