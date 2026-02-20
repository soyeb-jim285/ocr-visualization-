"use client";

import {
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

// ---------------------------------------------------------------------------
// Types & Constants
// ---------------------------------------------------------------------------

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
  "dense1",
  "relu4",
  "output",
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
  dense1: "Dense1",
  relu4: "ReLU4",
  output: "Output",
};

const LAYER_COLORS: Record<LayerKey, string> = {
  input: "#00ff41",
  conv1: "#00e5ff",
  relu1: "#18ffff",
  conv2: "#448aff",
  relu2: "#536dfe",
  pool1: "#7c4dff",
  conv3: "#b388ff",
  relu3: "#ea80fc",
  pool2: "#ff80ab",
  dense1: "#ff6e40",
  relu4: "#ffd740",
  output: "#76ff03",
};

type FlattenMode = "scanline" | "channelMean" | "channelEnergy";

const FLATTEN_MODE_LABELS: Record<FlattenMode, string> = {
  scanline: "SCAN",
  channelMean: "MEAN",
  channelEnergy: "NRG",
};

const DRAWING_CANVAS_SIZE = 180;
const DRAWING_INTERNAL_SIZE = 280;

const OSC_KEYS = ["osc1", "osc2", "osc3", "osc4"] as const;
type OscKey = (typeof OSC_KEYS)[number];

// ---------------------------------------------------------------------------
// 7-Segment Display Mapping
// ---------------------------------------------------------------------------

// Segments: top, topRight, bottomRight, bottom, bottomLeft, topLeft, middle
// indexed 0-6 respectively
const SEVEN_SEG_MAP: Record<string, boolean[]> = {
  "0": [true, true, true, true, true, true, false],
  "1": [false, true, true, false, false, false, false],
  "2": [true, true, false, true, true, false, true],
  "3": [true, true, true, true, false, false, true],
  "4": [false, true, true, false, false, true, true],
  "5": [true, false, true, true, false, true, true],
  "6": [true, false, true, true, true, true, true],
  "7": [true, true, true, false, false, false, false],
  "8": [true, true, true, true, true, true, true],
  "9": [true, true, true, true, false, true, true],
  A: [true, true, true, false, true, true, true],
  B: [false, false, true, true, true, true, true],
  C: [true, false, false, true, true, true, false],
  D: [false, true, true, true, true, false, true],
  E: [true, false, false, true, true, true, true],
  F: [true, false, false, false, true, true, true],
  G: [true, false, true, true, true, true, false],
  H: [false, true, true, false, true, true, true],
  I: [false, false, false, false, true, true, false],
  J: [false, true, true, true, false, false, false],
  K: [false, true, true, false, true, true, true], // approximate
  L: [false, false, false, true, true, true, false],
  M: [true, true, true, false, true, true, false], // approximate
  N: [false, false, true, false, true, false, true],
  O: [true, true, true, true, true, true, false],
  P: [true, true, false, false, true, true, true],
  Q: [true, true, true, false, false, true, true],
  R: [false, false, false, false, true, false, true],
  S: [true, false, true, true, false, true, true],
  T: [false, false, false, true, true, true, true],
  U: [false, true, true, true, true, true, false],
  V: [false, true, true, true, true, true, false], // same as U
  W: [false, true, true, true, true, true, false], // approximate
  X: [false, true, true, false, true, true, true], // same as H
  Y: [false, true, true, true, false, true, true],
  Z: [true, true, false, true, true, false, true], // same as 2
};

// ---------------------------------------------------------------------------
// Utility: Flatten layer activations to 1D signal
// ---------------------------------------------------------------------------

function flattenLayerToSignal(
  activations: number[][][] | number[],
  mode: FlattenMode
): number[] {
  // Check if 1D (dense layers)
  if (!Array.isArray(activations[0]) || activations.length === 0) {
    const flat = activations as number[];
    if (mode === "channelEnergy") {
      // Chunk into groups of 8 and sum
      const result: number[] = [];
      for (let i = 0; i < flat.length; i += 8) {
        let sum = 0;
        for (let j = i; j < Math.min(i + 8, flat.length); j++) {
          sum += flat[j] * flat[j];
        }
        result.push(sum);
      }
      return result;
    }
    return flat.slice();
  }

  // 3D: [C][H][W]
  const act3d = activations as number[][][];
  const C = act3d.length;
  const H = act3d[0]?.length ?? 0;
  const W = act3d[0]?.[0]?.length ?? 0;

  if (mode === "scanline") {
    // Take channel 0, flatten row-by-row
    const ch0 = act3d[0];
    if (!ch0) return [];
    const result: number[] = [];
    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        result.push(ch0[r][c]);
      }
    }
    return result;
  }

  if (mode === "channelMean") {
    // Average across channels at each spatial position
    const result: number[] = [];
    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        let sum = 0;
        for (let ch = 0; ch < C; ch++) {
          sum += act3d[ch][r][c];
        }
        result.push(sum / C);
      }
    }
    return result;
  }

  // channelEnergy: sum-of-squares per channel
  const result: number[] = [];
  for (let ch = 0; ch < C; ch++) {
    let energy = 0;
    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        const v = act3d[ch][r][c];
        energy += v * v;
      }
    }
    result.push(energy);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Utility: Flatten input tensor to 1D signal
// ---------------------------------------------------------------------------

function flattenInputToSignal(
  input: number[][] | null,
  mode: FlattenMode
): number[] | null {
  if (!input) return null;
  if (mode === "channelEnergy") {
    // Treat each row as a "channel" and compute energy
    const result: number[] = [];
    for (const row of input) {
      let energy = 0;
      for (const v of row) {
        energy += v * v;
      }
      result.push(energy);
    }
    return result;
  }
  // scanline and channelMean: just flatten row-by-row
  const result: number[] = [];
  for (const row of input) {
    for (const v of row) {
      result.push(v);
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Component: RotaryKnob
// ---------------------------------------------------------------------------

function RotaryKnob({
  value,
  onChange,
  label,
}: {
  value: number;
  onChange: (v: number) => void;
  label: string;
}) {
  const knobRef = useRef<SVGSVGElement>(null);
  const dragging = useRef(false);
  const startY = useRef(0);
  const startVal = useRef(0);

  const angle = -135 + value * 270; // from 7 o'clock (-135) to 5 o'clock (135)

  const onPointerDown = useCallback(
    (e: React.PointerEvent) => {
      dragging.current = true;
      startY.current = e.clientY;
      startVal.current = value;
      (e.target as Element).setPointerCapture(e.pointerId);
    },
    [value]
  );

  const onPointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!dragging.current) return;
      const dy = startY.current - e.clientY; // up = positive
      const newVal = Math.max(0, Math.min(1, startVal.current + dy / 150));
      onChange(newVal);
    },
    [onChange]
  );

  const onPointerUp = useCallback(() => {
    dragging.current = false;
  }, []);

  return (
    <div className="flex flex-col items-center gap-0.5">
      <svg
        ref={knobRef}
        width={40}
        height={40}
        viewBox="0 0 40 40"
        style={{ cursor: "ns-resize", touchAction: "none" }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
      >
        <defs>
          <radialGradient id="knobGrad" cx="35%" cy="35%">
            <stop offset="0%" stopColor="#555" />
            <stop offset="100%" stopColor="#1a1a1a" />
          </radialGradient>
        </defs>
        {/* Shadow */}
        <circle cx={20} cy={21} r={16} fill="rgba(0,0,0,0.5)" />
        {/* Knob body */}
        <circle
          cx={20}
          cy={20}
          r={16}
          fill="url(#knobGrad)"
          stroke="#333"
          strokeWidth={1}
        />
        {/* Outer ring */}
        <circle
          cx={20}
          cy={20}
          r={17}
          fill="none"
          stroke="#444"
          strokeWidth={0.5}
        />
        {/* Tick mark */}
        <line
          x1={20}
          y1={20}
          x2={20 + 12 * Math.sin((angle * Math.PI) / 180)}
          y2={20 - 12 * Math.cos((angle * Math.PI) / 180)}
          stroke="#ffffff"
          strokeWidth={2}
          strokeLinecap="round"
        />
        {/* Center dot */}
        <circle cx={20} cy={20} r={2} fill="#666" />
      </svg>
      <span
        className="font-mono text-[9px]"
        style={{ color: "#88ff88" }}
      >
        {label}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Component: OscilloscopeScreen
// ---------------------------------------------------------------------------

const OSC_PAD = { top: 24, right: 12, bottom: 20, left: 40 } as const;

function OscilloscopeScreen({
  signal,
  label,
  color,
  width,
  height,
}: {
  signal: number[] | null;
  label: string;
  color: string;
  width: number;
  height: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const mousePos = useRef<{ x: number; y: number } | null>(null);
  const prevSignalRef = useRef<number[] | null>(null);

  const plotW = width - OSC_PAD.left - OSC_PAD.right;
  const plotH = height - OSC_PAD.top - OSC_PAD.bottom;

  // Compute signal bounds
  const bounds = useMemo(() => {
    if (!signal || signal.length === 0) return { min: -1, max: 1 };
    let mn = Infinity;
    let mx = -Infinity;
    for (const v of signal) {
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    if (mn === mx) {
      mn -= 1;
      mx += 1;
    }
    return { min: mn, max: mx };
  }, [signal]);

  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Phosphor persistence: don't fully clear
    ctx.globalAlpha = 0.85;
    ctx.fillStyle = "#0a1a0a";
    ctx.fillRect(0, 0, width, height);
    ctx.globalAlpha = 1.0;

    // Draw rounded border
    ctx.strokeStyle = "#2a3a2a";
    ctx.lineWidth = 1;
    const borderR = 8;
    ctx.beginPath();
    ctx.moveTo(borderR, 0);
    ctx.lineTo(width - borderR, 0);
    ctx.arcTo(width, 0, width, borderR, borderR);
    ctx.lineTo(width, height - borderR);
    ctx.arcTo(width, height, width - borderR, height, borderR);
    ctx.lineTo(borderR, height);
    ctx.arcTo(0, height, 0, height - borderR, borderR);
    ctx.lineTo(0, borderR);
    ctx.arcTo(0, 0, borderR, 0, borderR);
    ctx.closePath();
    ctx.stroke();

    // Grid
    const gridDivsX = 10;
    const gridDivsY = 8;
    ctx.strokeStyle = "rgba(0, 255, 65, 0.1)";
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= gridDivsX; i++) {
      const x = OSC_PAD.left + (plotW / gridDivsX) * i;
      ctx.beginPath();
      ctx.moveTo(x, OSC_PAD.top);
      ctx.lineTo(x, OSC_PAD.top + plotH);
      ctx.stroke();
    }
    for (let i = 0; i <= gridDivsY; i++) {
      const y = OSC_PAD.top + (plotH / gridDivsY) * i;
      ctx.beginPath();
      ctx.moveTo(OSC_PAD.left, y);
      ctx.lineTo(OSC_PAD.left + plotW, y);
      ctx.stroke();
    }

    // Center crosshair lines (brighter)
    ctx.strokeStyle = "rgba(0, 255, 65, 0.2)";
    ctx.lineWidth = 0.5;
    const centerX = OSC_PAD.left + plotW / 2;
    const centerY = OSC_PAD.top + plotH / 2;
    ctx.beginPath();
    ctx.moveTo(centerX, OSC_PAD.top);
    ctx.lineTo(centerX, OSC_PAD.top + plotH);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(OSC_PAD.left, centerY);
    ctx.lineTo(OSC_PAD.left + plotW, centerY);
    ctx.stroke();

    // Label at top-left
    ctx.font = "bold 10px monospace";
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.9;
    ctx.fillText(label, OSC_PAD.left + 4, OSC_PAD.top - 8);
    ctx.globalAlpha = 1.0;

    // Scale labels
    ctx.font = "9px monospace";
    ctx.fillStyle = "rgba(0, 255, 65, 0.5)";
    ctx.textAlign = "right";
    ctx.fillText(bounds.max.toFixed(2), OSC_PAD.left - 3, OSC_PAD.top + 9);
    ctx.fillText(bounds.min.toFixed(2), OSC_PAD.left - 3, OSC_PAD.top + plotH);
    ctx.textAlign = "left";

    // Sample count on X axis
    if (signal) {
      ctx.fillStyle = "rgba(0, 255, 65, 0.4)";
      ctx.font = "8px monospace";
      ctx.textAlign = "center";
      ctx.fillText(
        `${signal.length} samples`,
        OSC_PAD.left + plotW / 2,
        height - 4
      );
      ctx.textAlign = "left";
    }

    const sig = signal ?? prevSignalRef.current;
    if (!sig || sig.length === 0) {
      // NO SIGNAL
      ctx.font = "bold 16px monospace";
      ctx.fillStyle = "rgba(0, 255, 65, 0.25)";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("NO SIGNAL", width / 2, height / 2);
      ctx.textAlign = "left";
      ctx.textBaseline = "alphabetic";
      return;
    }

    prevSignalRef.current = sig;
    const { min: sMin, max: sMax } = bounds;
    const range = sMax - sMin;

    // Helper to map signal value to canvas Y
    const toY = (v: number) =>
      OSC_PAD.top + plotH - ((v - sMin) / range) * plotH;
    const toX = (i: number) =>
      OSC_PAD.left + (i / (sig.length - 1)) * plotW;

    // Draw glow pass
    ctx.beginPath();
    ctx.moveTo(toX(0), toY(sig[0]));
    for (let i = 1; i < sig.length; i++) {
      ctx.lineTo(toX(i), toY(sig[i]));
    }
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.15;
    ctx.lineWidth = 6;
    ctx.stroke();

    // Draw main trace
    ctx.beginPath();
    ctx.moveTo(toX(0), toY(sig[0]));
    for (let i = 1; i < sig.length; i++) {
      ctx.lineTo(toX(i), toY(sig[i]));
    }
    ctx.strokeStyle = color;
    ctx.globalAlpha = 1.0;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Mouse crosshair
    const mp = mousePos.current;
    if (mp) {
      const mx = mp.x;
      const my = mp.y;

      if (
        mx >= OSC_PAD.left &&
        mx <= OSC_PAD.left + plotW &&
        my >= OSC_PAD.top &&
        my <= OSC_PAD.top + plotH
      ) {
        ctx.strokeStyle = "rgba(0, 255, 65, 0.4)";
        ctx.lineWidth = 0.5;
        ctx.setLineDash([4, 4]);

        // Vertical
        ctx.beginPath();
        ctx.moveTo(mx, OSC_PAD.top);
        ctx.lineTo(mx, OSC_PAD.top + plotH);
        ctx.stroke();

        // Horizontal
        ctx.beginPath();
        ctx.moveTo(OSC_PAD.left, my);
        ctx.lineTo(OSC_PAD.left + plotW, my);
        ctx.stroke();

        ctx.setLineDash([]);

        // Read value at cursor X
        const sampleIdx = Math.round(
          ((mx - OSC_PAD.left) / plotW) * (sig.length - 1)
        );
        const clampedIdx = Math.max(
          0,
          Math.min(sig.length - 1, sampleIdx)
        );
        const val = sig[clampedIdx];

        // Readout in top-right corner of plot area
        ctx.font = "10px monospace";
        ctx.fillStyle = color;
        const readout = `[${clampedIdx}] ${val.toFixed(3)}`;
        const textW = ctx.measureText(readout).width;
        ctx.fillStyle = "rgba(10, 26, 10, 0.8)";
        ctx.fillRect(
          OSC_PAD.left + plotW - textW - 8,
          OSC_PAD.top + 2,
          textW + 6,
          14
        );
        ctx.fillStyle = color;
        ctx.fillText(
          readout,
          OSC_PAD.left + plotW - textW - 5,
          OSC_PAD.top + 13
        );
      }
    }
  }, [signal, label, color, width, height, bounds, plotW, plotH]);

  useAnimationFrame(render, true);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return;
      mousePos.current = {
        x: ((e.clientX - rect.left) / rect.width) * width,
        y: ((e.clientY - rect.top) / rect.height) * height,
      };
    },
    [width, height]
  );

  const handleMouseLeave = useCallback(() => {
    mousePos.current = null;
  }, []);

  return (
    <div
      ref={containerRef}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      style={{
        width: "100%",
        height: "100%",
        borderRadius: 8,
        overflow: "hidden",
        boxShadow:
          "inset 0 2px 6px rgba(0,0,0,0.8), 0 0 8px rgba(0,255,65,0.05)",
        border: "1px solid #2a3a2a",
        background: "#0a1a0a",
      }}
    >
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{ width: "100%", height: "100%", display: "block" }}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Component: SpectrumAnalyzer
// ---------------------------------------------------------------------------

function SpectrumAnalyzer({
  signal,
  width,
  height,
}: {
  signal: number[] | null;
  width: number;
  height: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Compute DFT magnitudes
  const spectrum = useMemo(() => {
    if (!signal || signal.length === 0) return null;
    const N = signal.length;
    const numBins = 64;

    // Subsample signal if too long
    let samples = signal;
    if (N > 512) {
      const step = Math.floor(N / 512);
      const sub: number[] = [];
      for (let i = 0; i < N; i += step) {
        sub.push(signal[i]);
      }
      samples = sub;
    }
    const M = samples.length;

    const magnitudes: number[] = [];
    for (let k = 0; k < numBins; k++) {
      let re = 0;
      let im = 0;
      for (let n = 0; n < M; n++) {
        const angle = (2 * Math.PI * k * n) / M;
        re += samples[n] * Math.cos(angle);
        im -= samples[n] * Math.sin(angle);
      }
      magnitudes.push(Math.sqrt(re * re + im * im) / M);
    }
    return magnitudes;
  }, [signal]);

  // Render spectrum
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const pad = { top: 24, right: 8, bottom: 8, left: 8 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    // CRT background
    ctx.fillStyle = "#0a1a0a";
    ctx.fillRect(0, 0, width, height);

    // Grid
    ctx.strokeStyle = "rgba(0, 255, 65, 0.08)";
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 8; i++) {
      const y = pad.top + (plotH / 8) * i;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();
    }

    // Label
    ctx.font = "bold 10px monospace";
    ctx.fillStyle = "#00ff41";
    ctx.fillText("SPECTRUM", pad.left + 4, 14);

    if (!spectrum) {
      ctx.font = "12px monospace";
      ctx.fillStyle = "rgba(0, 255, 65, 0.2)";
      ctx.textAlign = "center";
      ctx.fillText("NO DATA", width / 2, height / 2);
      ctx.textAlign = "left";
      return;
    }

    const maxMag = Math.max(...spectrum, 0.001);
    const barW = plotW / spectrum.length;

    for (let i = 0; i < spectrum.length; i++) {
      const normalized = spectrum[i] / maxMag;
      const barH = normalized * plotH;
      const x = pad.left + i * barW;
      const y = pad.top + plotH - barH;

      // Gradient bar color based on height
      const grad = ctx.createLinearGradient(x, pad.top + plotH, x, y);
      grad.addColorStop(0, "#00ff41");
      grad.addColorStop(0.5, "#ffff00");
      grad.addColorStop(1, "#ff2200");
      ctx.fillStyle = grad;
      ctx.fillRect(x, y, Math.max(barW - 1, 1), barH);
    }
  }, [spectrum, width, height]);

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        borderRadius: 8,
        overflow: "hidden",
        boxShadow:
          "inset 0 2px 6px rgba(0,0,0,0.8), 0 0 8px rgba(0,255,65,0.05)",
        border: "1px solid #2a3a2a",
        background: "#0a1a0a",
      }}
    >
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{ width: "100%", height: "100%", display: "block" }}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Component: SignalPatchBay
// ---------------------------------------------------------------------------

function SignalPatchBay({
  routings,
  onRoute,
  selectedSource,
  onSelectSource,
}: {
  routings: Record<OscKey, LayerKey | null>;
  onRoute: (osc: OscKey, layer: LayerKey) => void;
  selectedSource: LayerKey | null;
  onSelectSource: (layer: LayerKey | null) => void;
}) {
  const sourceRowRef = useRef<HTMLDivElement>(null);
  const destRowRef = useRef<HTMLDivElement>(null);
  const [sourcePositions, setSourcePositions] = useState<
    Record<string, { x: number; y: number }>
  >({});
  const [destPositions, setDestPositions] = useState<
    Record<string, { x: number; y: number }>
  >({});
  const containerRef = useRef<HTMLDivElement>(null);

  // Compute button positions for cable drawing
  const updatePositions = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;
    const containerRect = container.getBoundingClientRect();

    const newSrc: Record<string, { x: number; y: number }> = {};
    const srcBtns = sourceRowRef.current?.querySelectorAll("[data-layer]");
    srcBtns?.forEach((btn) => {
      const key = btn.getAttribute("data-layer");
      if (!key) return;
      const rect = btn.getBoundingClientRect();
      newSrc[key] = {
        x: rect.left + rect.width / 2 - containerRect.left,
        y: rect.bottom - containerRect.top,
      };
    });
    setSourcePositions(newSrc);

    const newDst: Record<string, { x: number; y: number }> = {};
    const dstBtns = destRowRef.current?.querySelectorAll("[data-osc]");
    dstBtns?.forEach((btn) => {
      const key = btn.getAttribute("data-osc");
      if (!key) return;
      const rect = btn.getBoundingClientRect();
      newDst[key] = {
        x: rect.left + rect.width / 2 - containerRect.left,
        y: rect.top - containerRect.top,
      };
    });
    setDestPositions(newDst);
  }, []);

  useEffect(() => {
    updatePositions();
    window.addEventListener("resize", updatePositions);
    return () => window.removeEventListener("resize", updatePositions);
  }, [updatePositions]);

  // Determine which sources are routed
  const routedSources = useMemo(() => {
    const set = new Set<string>();
    for (const osc of OSC_KEYS) {
      const layer = routings[osc];
      if (layer) set.add(layer);
    }
    return set;
  }, [routings]);

  // Build cable paths
  const cables = useMemo(() => {
    const result: Array<{
      from: { x: number; y: number };
      to: { x: number; y: number };
      color: string;
      key: string;
    }> = [];
    for (const osc of OSC_KEYS) {
      const layer = routings[osc];
      if (!layer) continue;
      const from = sourcePositions[layer];
      const to = destPositions[osc];
      if (!from || !to) continue;
      result.push({
        from,
        to,
        color: LAYER_COLORS[layer],
        key: `${osc}-${layer}`,
      });
    }
    return result;
  }, [routings, sourcePositions, destPositions]);

  const handleSourceClick = (layer: LayerKey) => {
    if (selectedSource === layer) {
      onSelectSource(null);
    } else {
      onSelectSource(layer);
    }
  };

  const handleDestClick = (osc: OscKey) => {
    if (selectedSource) {
      onRoute(osc, selectedSource);
      onSelectSource(null);
    }
  };

  return (
    <div
      ref={containerRef}
      style={{
        position: "relative",
        display: "flex",
        flexDirection: "column",
        gap: 0,
        padding: "8px 12px",
        borderRadius: 8,
        border: "1px solid #2a3a2a",
        background: "linear-gradient(180deg, #1a1a20 0%, #12121a 100%)",
      }}
    >
      {/* Title */}
      <div
        style={{
          fontFamily: "monospace",
          fontSize: 9,
          color: "#88ff88",
          textAlign: "center",
          marginBottom: 4,
          letterSpacing: 2,
        }}
      >
        PATCH BAY
      </div>

      {/* Source buttons */}
      <div
        ref={sourceRowRef}
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 3,
          justifyContent: "center",
          marginBottom: 24,
        }}
      >
        {LAYER_KEYS.map((layer) => {
          const isSelected = selectedSource === layer;
          const isRouted = routedSources.has(layer);
          const clr = LAYER_COLORS[layer];
          return (
            <button
              key={layer}
              data-layer={layer}
              onClick={() => handleSourceClick(layer)}
              style={{
                fontFamily: "monospace",
                fontSize: 9,
                padding: "3px 6px",
                borderRadius: 3,
                border: `1px solid ${isSelected ? clr : "rgba(255,255,255,0.15)"}`,
                background: isSelected
                  ? `${clr}22`
                  : "rgba(0,0,0,0.4)",
                color: isSelected ? clr : "rgba(255,255,255,0.6)",
                cursor: "pointer",
                position: "relative",
                transition: "all 0.15s",
                boxShadow: isSelected ? `0 0 6px ${clr}55` : "none",
              }}
            >
              {/* LED indicator */}
              <span
                style={{
                  position: "absolute",
                  top: -2,
                  right: -2,
                  width: 5,
                  height: 5,
                  borderRadius: "50%",
                  background: isRouted ? clr : "#333",
                  boxShadow: isRouted
                    ? `0 0 4px ${clr}`
                    : "none",
                }}
              />
              {LAYER_DISPLAY_NAMES[layer]}
            </button>
          );
        })}
      </div>

      {/* SVG cables */}
      <svg
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: "100%",
          height: "100%",
          pointerEvents: "none",
          overflow: "visible",
        }}
      >
        <defs>
          <style>{`
            @keyframes dashFlow {
              to { stroke-dashoffset: -20; }
            }
          `}</style>
        </defs>
        {cables.map((cable) => {
          const midY = (cable.from.y + cable.to.y) / 2;
          const d = `M ${cable.from.x} ${cable.from.y} C ${cable.from.x} ${midY}, ${cable.to.x} ${midY}, ${cable.to.x} ${cable.to.y}`;
          return (
            <g key={cable.key}>
              {/* Cable shadow */}
              <path
                d={d}
                fill="none"
                stroke="rgba(0,0,0,0.5)"
                strokeWidth={5}
              />
              {/* Cable body */}
              <path
                d={d}
                fill="none"
                stroke={cable.color}
                strokeWidth={3}
                strokeLinecap="round"
                opacity={0.7}
              />
              {/* Animated dash overlay */}
              <path
                d={d}
                fill="none"
                stroke={cable.color}
                strokeWidth={2}
                strokeDasharray="8 12"
                strokeLinecap="round"
                style={{
                  animation: "dashFlow 0.6s linear infinite",
                }}
              />
            </g>
          );
        })}
      </svg>

      {/* Destination buttons */}
      <div
        ref={destRowRef}
        style={{
          display: "flex",
          gap: 6,
          justifyContent: "center",
        }}
      >
        {OSC_KEYS.map((osc, idx) => {
          const routed = routings[osc];
          const clr = routed ? LAYER_COLORS[routed] : "#444";
          return (
            <button
              key={osc}
              data-osc={osc}
              onClick={() => handleDestClick(osc)}
              style={{
                fontFamily: "monospace",
                fontSize: 9,
                padding: "3px 8px",
                borderRadius: 3,
                border: `1px solid ${selectedSource ? "#88ff88" : "rgba(255,255,255,0.15)"}`,
                background: routed
                  ? `${clr}15`
                  : "rgba(0,0,0,0.4)",
                color: selectedSource
                  ? "#88ff88"
                  : "rgba(255,255,255,0.5)",
                cursor: selectedSource ? "pointer" : "default",
                transition: "all 0.15s",
                boxShadow: selectedSource
                  ? "0 0 4px rgba(0,255,65,0.3)"
                  : "none",
              }}
            >
              OSC {idx + 1}
            </button>
          );
        })}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Component: SevenSegmentDisplay
// ---------------------------------------------------------------------------

function SevenSegmentDisplay({ char }: { char: string | null }) {
  const segments = char
    ? SEVEN_SEG_MAP[char.toUpperCase()] ?? [false, false, false, false, false, false, false]
    : [false, false, false, false, false, false, false];

  const onColor = "#ffaa00";
  const offColor = "#1a1200";
  const glowShadow = "0 0 6px #ffaa00, 0 0 12px #ff880088";

  // Segment positions and dimensions
  // 0: top horizontal, 1: top-right vertical, 2: bottom-right vertical
  // 3: bottom horizontal, 4: bottom-left vertical, 5: top-left vertical
  // 6: middle horizontal
  const segW = 28;
  const segH = 6;
  const vW = 6;
  const vH = 24;
  const totalW = 40;
  const totalH = 60;
  const offsetX = 6;
  const offsetY = 2;

  const segmentStyles: React.CSSProperties[] = [
    // 0: top horizontal
    {
      position: "absolute",
      left: offsetX,
      top: offsetY,
      width: segW,
      height: segH,
      borderRadius: 3,
      background: segments[0] ? onColor : offColor,
      boxShadow: segments[0] ? glowShadow : "none",
    },
    // 1: top-right vertical
    {
      position: "absolute",
      right: 0,
      top: offsetY + segH - 2,
      width: vW,
      height: vH,
      borderRadius: 3,
      background: segments[1] ? onColor : offColor,
      boxShadow: segments[1] ? glowShadow : "none",
    },
    // 2: bottom-right vertical
    {
      position: "absolute",
      right: 0,
      top: offsetY + segH + vH - 1,
      width: vW,
      height: vH,
      borderRadius: 3,
      background: segments[2] ? onColor : offColor,
      boxShadow: segments[2] ? glowShadow : "none",
    },
    // 3: bottom horizontal
    {
      position: "absolute",
      left: offsetX,
      bottom: offsetY,
      width: segW,
      height: segH,
      borderRadius: 3,
      background: segments[3] ? onColor : offColor,
      boxShadow: segments[3] ? glowShadow : "none",
    },
    // 4: bottom-left vertical
    {
      position: "absolute",
      left: 0,
      top: offsetY + segH + vH - 1,
      width: vW,
      height: vH,
      borderRadius: 3,
      background: segments[4] ? onColor : offColor,
      boxShadow: segments[4] ? glowShadow : "none",
    },
    // 5: top-left vertical
    {
      position: "absolute",
      left: 0,
      top: offsetY + segH - 2,
      width: vW,
      height: vH,
      borderRadius: 3,
      background: segments[5] ? onColor : offColor,
      boxShadow: segments[5] ? glowShadow : "none",
    },
    // 6: middle horizontal
    {
      position: "absolute",
      left: offsetX,
      top: offsetY + segH + vH - 4,
      width: segW,
      height: segH,
      borderRadius: 3,
      background: segments[6] ? onColor : offColor,
      boxShadow: segments[6] ? glowShadow : "none",
    },
  ];

  return (
    <div
      style={{
        position: "relative",
        width: totalW,
        height: totalH,
        flexShrink: 0,
      }}
    >
      {segmentStyles.map((style, i) => (
        <div key={i} style={style} />
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Component: PredictionVUMeter
// ---------------------------------------------------------------------------

function PredictionVUMeter() {
  const prediction = useInferenceStore((s) => s.prediction);
  const topPrediction = useInferenceStore((s) => s.topPrediction);
  const [peakHolds, setPeakHolds] = useState<Record<number, number>>({});

  // Top 20 predictions by probability
  const top20 = useMemo(() => {
    if (!prediction) return [];
    const indexed = prediction.map((prob, idx) => ({ prob, idx }));
    indexed.sort((a, b) => b.prob - a.prob);
    return indexed.slice(0, 20);
  }, [prediction]);

  // Peak hold decay
  useEffect(() => {
    if (!prediction) return;
    const newPeaks = { ...peakHolds };
    for (const { prob, idx } of top20) {
      const h = prob * 120;
      const current = newPeaks[idx] ?? 0;
      if (h >= current) {
        newPeaks[idx] = h;
      }
    }
    setPeakHolds(newPeaks);

    // Decay timer
    const interval = setInterval(() => {
      setPeakHolds((prev) => {
        const next = { ...prev };
        let changed = false;
        for (const key of Object.keys(next)) {
          const k = Number(key);
          const target =
            top20.find((t) => t.idx === k)?.prob ?? 0;
          const targetH = target * 120;
          if (next[k] > targetH + 1) {
            next[k] -= 1;
            changed = true;
          }
        }
        return changed ? next : prev;
      });
    }, 100);
    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prediction]);

  const predictedChar = useMemo(() => {
    if (!topPrediction) return null;
    return EMNIST_CLASSES[topPrediction.classIndex] ?? null;
  }, [topPrediction]);

  return (
    <div
      style={{
        display: "flex",
        alignItems: "flex-end",
        gap: 12,
        padding: "10px 16px",
        borderRadius: 8,
        border: "1px solid #2a3a2a",
        background: "linear-gradient(180deg, #1a1a20 0%, #0a0a12 100%)",
        height: "100%",
        width: "100%",
        overflow: "hidden",
      }}
    >
      {/* VU Bars */}
      <div
        style={{
          display: "flex",
          alignItems: "flex-end",
          gap: 3,
          flex: 1,
          height: "100%",
          paddingTop: 16,
        }}
      >
        {/* Title */}
        <div
          style={{
            position: "absolute",
            top: 4,
            left: 16,
            fontFamily: "monospace",
            fontSize: 9,
            color: "#88ff88",
            letterSpacing: 2,
          }}
        >
          VU OUTPUT
        </div>
        {top20.map(({ prob, idx }) => {
          const barH = prob * 120;
          const peakH = peakHolds[idx] ?? barH;
          const label = EMNIST_CLASSES[idx] ?? "?";
          const isTop = topPrediction?.classIndex === idx;

          return (
            <div
              key={idx}
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                position: "relative",
              }}
            >
              {/* Bar container */}
              <div
                style={{
                  width: 8,
                  height: 120,
                  position: "relative",
                  background: "rgba(0,0,0,0.3)",
                  borderRadius: 2,
                }}
              >
                {/* Filled bar */}
                <div
                  style={{
                    position: "absolute",
                    bottom: 0,
                    left: 0,
                    width: 8,
                    height: barH,
                    borderRadius: 2,
                    background:
                      prob > 0.9
                        ? "linear-gradient(to top, #00ff41, #ffff00, #ff2200)"
                        : prob > 0.6
                          ? "linear-gradient(to top, #00ff41, #ffff00)"
                          : "#00ff41",
                    boxShadow: isTop
                      ? "0 0 4px #00ff41"
                      : "none",
                    transition: "height 0.15s ease-out",
                  }}
                />
                {/* Peak hold line */}
                <div
                  style={{
                    position: "absolute",
                    bottom: peakH,
                    left: 0,
                    width: 8,
                    height: 2,
                    background: "#ffffff",
                    boxShadow: "0 0 3px #ffffff",
                    transition: "bottom 0.1s linear",
                  }}
                />
              </div>
              {/* Label (rotated) */}
              <div
                style={{
                  fontFamily: "monospace",
                  fontSize: 8,
                  color: isTop ? "#ffaa00" : "rgba(255,255,255,0.4)",
                  transform: "rotate(-90deg)",
                  transformOrigin: "center center",
                  width: 28,
                  textAlign: "center",
                  marginTop: 8,
                  fontWeight: isTop ? "bold" : "normal",
                }}
              >
                {label}
              </div>
            </div>
          );
        })}
      </div>

      {/* 7-segment display */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: 4,
          marginLeft: 12,
          padding: "8px 12px",
          borderRadius: 6,
          border: "1px solid #2a2a10",
          background: "#0a0a02",
          minWidth: 64,
        }}
      >
        <div
          style={{
            fontFamily: "monospace",
            fontSize: 8,
            color: "#88ff88",
            letterSpacing: 1,
          }}
        >
          OUTPUT
        </div>
        <SevenSegmentDisplay char={predictedChar} />
        {topPrediction && (
          <div
            style={{
              fontFamily: "monospace",
              fontSize: 9,
              color: "#ffaa00",
            }}
          >
            {(topPrediction.confidence * 100).toFixed(1)}%
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Component: OscilloscopeDrawingCanvas
// ---------------------------------------------------------------------------

function OscilloscopeDrawingCanvas({
  flattenMode,
  onFlattenModeChange,
}: {
  flattenMode: FlattenMode;
  onFlattenModeChange: (mode: FlattenMode) => void;
}) {
  const { infer } = useInference();
  const isInferring = useInferenceStore((s) => s.isInferring);
  const reset = useInferenceStore((s) => s.reset);

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

  const handleClear = useCallback(() => {
    clear();
    reset();
  }, [clear, reset]);

  // Map knob value to flatten mode
  const knobValue = useMemo(() => {
    switch (flattenMode) {
      case "scanline":
        return 0;
      case "channelMean":
        return 0.5;
      case "channelEnergy":
        return 1;
    }
  }, [flattenMode]);

  const handleKnobChange = useCallback(
    (v: number) => {
      if (v < 0.33) onFlattenModeChange("scanline");
      else if (v < 0.67) onFlattenModeChange("channelMean");
      else onFlattenModeChange("channelEnergy");
    },
    [onFlattenModeChange]
  );

  // Infer LED pulse
  const [ledOn, setLedOn] = useState(false);
  useEffect(() => {
    if (isInferring) {
      setLedOn(true);
    } else {
      const t = setTimeout(() => setLedOn(false), 300);
      return () => clearTimeout(t);
    }
  }, [isInferring]);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 6,
        padding: "10px 12px 8px",
        borderRadius: 8,
        border: "1px solid #2a3a2a",
        background:
          "linear-gradient(180deg, #252530 0%, #1a1a24 50%, #252530 100%)",
        boxShadow: "inset 0 1px 0 rgba(255,255,255,0.05)",
      }}
    >
      {/* Title bar */}
      <div
        style={{
          fontFamily: "monospace",
          fontSize: 9,
          color: "#88ff88",
          letterSpacing: 2,
          textAlign: "center",
        }}
      >
        SIGNAL SOURCE
      </div>

      {/* Canvas */}
      <div
        style={{
          borderRadius: 6,
          overflow: "hidden",
          border: "2px solid #1a2a1a",
          boxShadow:
            "inset 0 2px 8px rgba(0,0,0,0.8)",
          position: "relative",
        }}
      >
        <canvas
          ref={canvasRef}
          width={DRAWING_INTERNAL_SIZE}
          height={DRAWING_INTERNAL_SIZE}
          className="cursor-crosshair touch-none"
          style={{
            width: DRAWING_CANVAS_SIZE,
            height: DRAWING_CANVAS_SIZE,
            imageRendering: "auto",
            display: "block",
          }}
          onMouseDown={(e) => startDrawing(e.nativeEvent)}
          onMouseMove={(e) => draw(e.nativeEvent)}
          onMouseUp={() => stopDrawing()}
          onMouseLeave={() => stopDrawing()}
          onTouchStart={(e) => {
            e.preventDefault();
            startDrawing(e.nativeEvent);
          }}
          onTouchMove={(e) => {
            e.preventDefault();
            draw(e.nativeEvent);
          }}
          onTouchEnd={() => stopDrawing()}
          aria-label="Drawing canvas for character input"
        />
        {!hasDrawn && (
          <div
            style={{
              position: "absolute",
              inset: 0,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              pointerEvents: "none",
            }}
          >
            <span
              style={{
                fontFamily: "monospace",
                fontSize: 12,
                color: "rgba(136, 255, 136, 0.25)",
              }}
            >
              Draw here
            </span>
          </div>
        )}
      </div>

      {/* Control row */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          width: "100%",
        }}
      >
        {/* Clear button */}
        <button
          onClick={handleClear}
          style={{
            fontFamily: "monospace",
            fontSize: 9,
            color: "#88ff88",
            background: "rgba(0,0,0,0.4)",
            border: "1px solid #2a3a2a",
            borderRadius: 3,
            padding: "3px 8px",
            cursor: "pointer",
            transition: "all 0.15s",
          }}
        >
          CLEAR
        </button>

        {/* Mode label */}
        <div
          style={{
            fontFamily: "monospace",
            fontSize: 8,
            color: "rgba(136,255,136,0.6)",
            flex: 1,
            textAlign: "center",
          }}
        >
          {FLATTEN_MODE_LABELS[flattenMode]}
        </div>

        {/* Rotary knob for flatten mode */}
        <RotaryKnob
          value={knobValue}
          onChange={handleKnobChange}
          label="MODE"
        />

        {/* Inference LED */}
        <div
          style={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: ledOn ? "#00ff41" : "#0a2a0a",
            boxShadow: ledOn
              ? "0 0 6px #00ff41, 0 0 12px #00ff4188"
              : "none",
            transition: "all 0.15s",
          }}
        />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Layout
// ---------------------------------------------------------------------------

export function OscilloscopeLayout() {
  // Routing state: which layer is assigned to each oscilloscope
  const [routings, setRoutings] = useState<Record<OscKey, LayerKey | null>>({
    osc1: "conv1",
    osc2: "relu2",
    osc3: "pool2",
    osc4: "output",
  });
  const [selectedSource, setSelectedSource] = useState<LayerKey | null>(null);
  const [flattenMode, setFlattenMode] = useState<FlattenMode>("channelMean");

  const inputTensor = useInferenceStore((s) => s.inputTensor);
  const layerActivations = useInferenceStore((s) => s.layerActivations);

  // Compute flattened signals for all routed layers
  const signals = useMemo(() => {
    const result: Record<OscKey, number[] | null> = {
      osc1: null,
      osc2: null,
      osc3: null,
      osc4: null,
    };
    for (const osc of OSC_KEYS) {
      const layer = routings[osc];
      if (!layer) continue;
      if (layer === "input") {
        result[osc] = flattenInputToSignal(inputTensor, flattenMode);
      } else {
        const act = layerActivations[layer];
        if (act) {
          result[osc] = flattenLayerToSignal(act, flattenMode);
        }
      }
    }
    return result;
  }, [routings, inputTensor, layerActivations, flattenMode]);

  // Signal for spectrum analyzer: use the first routed oscilloscope that has data
  const spectrumSignal = useMemo(() => {
    for (const osc of OSC_KEYS) {
      if (signals[osc]) return signals[osc];
    }
    return null;
  }, [signals]);

  const handleRoute = useCallback(
    (osc: OscKey, layer: LayerKey) => {
      setRoutings((prev) => ({ ...prev, [osc]: layer }));
    },
    []
  );

  // Oscilloscope screen dimensions (will be computed from container)
  const oscW = 500;
  const oscH = 280;

  return (
    <div
      style={{
        width: "100vw",
        height: "100vh",
        background: "#0a0a12",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
        fontFamily: "monospace",
      }}
    >
      {/* ============= Top Row ============= */}
      <div
        style={{
          display: "flex",
          gap: 10,
          padding: "10px 12px 6px",
          alignItems: "stretch",
          flexShrink: 0,
        }}
      >
        {/* Drawing Canvas (left) */}
        <div style={{ width: 220, flexShrink: 0 }}>
          <OscilloscopeDrawingCanvas
            flattenMode={flattenMode}
            onFlattenModeChange={setFlattenMode}
          />
        </div>

        {/* Patch Bay (center) */}
        <div style={{ flex: 1, minWidth: 0 }}>
          <SignalPatchBay
            routings={routings}
            onRoute={handleRoute}
            selectedSource={selectedSource}
            onSelectSource={setSelectedSource}
          />
        </div>

        {/* Spectrum Analyzer (right) */}
        <div style={{ width: 300, flexShrink: 0, height: 200 }}>
          <SpectrumAnalyzer
            signal={spectrumSignal}
            width={600}
            height={400}
          />
        </div>
      </div>

      {/* ============= Center: 2x2 Oscilloscope Grid ============= */}
      <div
        style={{
          flex: 1,
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gridTemplateRows: "1fr 1fr",
          gap: 10,
          padding: "6px 12px",
          minHeight: 0,
        }}
      >
        {OSC_KEYS.map((osc, idx) => {
          const layer = routings[osc];
          const clr = layer ? LAYER_COLORS[layer] : "#00ff41";
          const lbl = layer
            ? `OSC ${idx + 1} — ${LAYER_DISPLAY_NAMES[layer]}`
            : `OSC ${idx + 1} — UNPATCHED`;
          return (
            <div key={osc} style={{ minHeight: 0, minWidth: 0 }}>
              <OscilloscopeScreen
                signal={signals[osc]}
                label={lbl}
                color={clr}
                width={oscW}
                height={oscH}
              />
            </div>
          );
        })}
      </div>

      {/* ============= Bottom: VU Meter ============= */}
      <div
        style={{
          height: 180,
          flexShrink: 0,
          padding: "0 12px 6px",
          position: "relative",
        }}
      >
        <PredictionVUMeter />
      </div>

      {/* ============= Layout Nav ============= */}
      <LayoutNav />
    </div>
  );
}
