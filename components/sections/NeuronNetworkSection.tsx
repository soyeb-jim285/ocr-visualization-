"use client";

import {
  useRef,
  useState,
  useCallback,
  useMemo,
  useEffect,
} from "react";
import { useInferenceStore } from "@/stores/inferenceStore";
import { EMNIST_CLASSES, BYMERGE_MERGED_INDICES } from "@/lib/model/classes";

// ---------------------------------------------------------------------------
// Layer definitions
// ---------------------------------------------------------------------------

interface NeuronLayerDef {
  name: string;
  displayName: string;
  type: "input" | "conv" | "relu" | "pool" | "dense" | "output";
  totalNeurons: number;
  displayNeurons: number;
  color: string;
  rgb: [number, number, number]; // pre-parsed
  description: string;
}

function parseHex(hex: string): [number, number, number] {
  return [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)];
}

const LAYERS: NeuronLayerDef[] = [
  { name: "input", displayName: "Input", type: "input", totalNeurons: 784, displayNeurons: 20, color: "#94a3b8", rgb: parseHex("#94a3b8"), description: "28×28 pixels" },
  { name: "conv1", displayName: "Conv1", type: "conv", totalNeurons: 32, displayNeurons: 16, color: "#6366f1", rgb: parseHex("#6366f1"), description: "32 channels (3×3)" },
  { name: "relu1", displayName: "ReLU1", type: "relu", totalNeurons: 32, displayNeurons: 16, color: "#818cf8", rgb: parseHex("#818cf8"), description: "32 channels" },
  { name: "conv2", displayName: "Conv2", type: "conv", totalNeurons: 64, displayNeurons: 24, color: "#a78bfa", rgb: parseHex("#a78bfa"), description: "64 channels (3×3)" },
  { name: "relu2", displayName: "ReLU2", type: "relu", totalNeurons: 64, displayNeurons: 24, color: "#8b5cf6", rgb: parseHex("#8b5cf6"), description: "64 channels" },
  { name: "pool1", displayName: "Pool1", type: "pool", totalNeurons: 64, displayNeurons: 18, color: "#06b6d4", rgb: parseHex("#06b6d4"), description: "64 ch → 14×14" },
  { name: "conv3", displayName: "Conv3", type: "conv", totalNeurons: 128, displayNeurons: 30, color: "#0891b2", rgb: parseHex("#0891b2"), description: "128 channels (3×3)" },
  { name: "relu3", displayName: "ReLU3", type: "relu", totalNeurons: 128, displayNeurons: 30, color: "#0e7490", rgb: parseHex("#0e7490"), description: "128 channels" },
  { name: "pool2", displayName: "Pool2", type: "pool", totalNeurons: 128, displayNeurons: 20, color: "#22c55e", rgb: parseHex("#22c55e"), description: "128 ch → 7×7" },
  { name: "dense1", displayName: "Dense", type: "dense", totalNeurons: 256, displayNeurons: 32, color: "#f59e0b", rgb: parseHex("#f59e0b"), description: "256 neurons" },
  { name: "relu4", displayName: "ReLU4", type: "relu", totalNeurons: 256, displayNeurons: 32, color: "#d97706", rgb: parseHex("#d97706"), description: "256 neurons" },
  { name: "output", displayName: "Output", type: "output", totalNeurons: 47, displayNeurons: 10, color: "#ef4444", rgb: parseHex("#ef4444"), description: "Top predictions" },
];

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface HoveredNeuron {
  layerIdx: number;
  neuronIdx: number;
  screenX: number;
  screenY: number;
}

// ---------------------------------------------------------------------------
// Deterministic PRNG
// ---------------------------------------------------------------------------

function makePRNG(seed: number) {
  let s = seed;
  return () => { s = (s * 16807) % 2147483647; return (s - 1) / 2147483646; };
}

// ---------------------------------------------------------------------------
// Pre-computed connections (SoA layout for cache efficiency)
// ---------------------------------------------------------------------------

const CONN_COUNT = (() => {
  let c = 0;
  for (let li = 0; li < LAYERS.length - 1; li++) {
    c += LAYERS[li].displayNeurons * Math.min(3, LAYERS[li + 1].displayNeurons);
  }
  return c;
})();

// Struct-of-arrays for connections
const connFromLayer = new Uint8Array(CONN_COUNT);
const connFromNeuron = new Uint8Array(CONN_COUNT);
const connToLayer = new Uint8Array(CONN_COUNT);
const connToNeuron = new Uint8Array(CONN_COUNT);

(() => {
  const rand = makePRNG(42);
  let ci = 0;
  for (let li = 0; li < LAYERS.length - 1; li++) {
    const fromN = LAYERS[li].displayNeurons;
    const toN = LAYERS[li + 1].displayNeurons;
    const perNeuron = Math.min(3, toN);
    for (let fi = 0; fi < fromN; fi++) {
      const targets = new Set<number>();
      targets.add(Math.floor((fi / fromN) * toN));
      while (targets.size < perNeuron) targets.add(Math.floor(rand() * toN));
      for (const ti of targets) {
        connFromLayer[ci] = li;
        connFromNeuron[ci] = fi;
        connToLayer[ci] = li + 1;
        connToNeuron[ci] = ti;
        ci++;
      }
    }
  }
})();

// Signal highlights — ~1 per 6 connections for performance
const SIGNAL_COUNT = Math.ceil(CONN_COUNT / 6);
const sigConnIdx = new Uint16Array(SIGNAL_COUNT);
const sigProgress = new Float32Array(SIGNAL_COUNT);
const sigSpeed = new Float32Array(SIGNAL_COUNT);
const sigIntensity = new Float32Array(SIGNAL_COUNT);
const sigSize = new Float32Array(SIGNAL_COUNT);

(() => {
  const rand = makePRNG(123);
  for (let i = 0; i < SIGNAL_COUNT; i++) {
    sigConnIdx[i] = Math.floor(rand() * CONN_COUNT);
    sigProgress[i] = rand();
    sigSpeed[i] = 0.003 + rand() * 0.007;
    sigIntensity[i] = 0.5 + rand() * 0.5;
    sigSize[i] = 1.5 + rand() * 1.5;
  }
})();

// ---------------------------------------------------------------------------
// Neuron position computation (flat arrays)
// ---------------------------------------------------------------------------

function computeLayout(w: number, h: number) {
  const totalNeurons = LAYERS.reduce((s, l) => s + l.displayNeurons, 0);
  const posX = new Float32Array(totalNeurons);
  const posY = new Float32Array(totalNeurons);
  const posLayerIdx = new Uint8Array(totalNeurons);
  const posNeuronIdx = new Uint8Array(totalNeurons);

  const marginX = 80, marginY = 60;
  const availW = w - marginX * 2, availH = h - marginY * 2;
  const layerSpacing = availW / (LAYERS.length - 1);
  const radius = Math.min(7, 24 * 0.32);

  // Pre-compute per-layer start indices and bottom Y for labels
  const layerStartIdx: number[] = [];
  const layerBottomY: number[] = [];
  const layerX: number[] = [];

  let idx = 0;
  for (let li = 0; li < LAYERS.length; li++) {
    layerStartIdx.push(idx);
    const n = LAYERS[li].displayNeurons;
    const x = marginX + li * layerSpacing;
    layerX.push(x);
    const spacing = Math.min(24, availH / (n + 1));
    const totalH = (n - 1) * spacing;
    const startY = h / 2 - totalH / 2;
    let maxY = 0;
    for (let ni = 0; ni < n; ni++) {
      posX[idx] = x;
      posY[idx] = startY + ni * spacing;
      posLayerIdx[idx] = li;
      posNeuronIdx[idx] = ni;
      if (posY[idx] > maxY) maxY = posY[idx];
      idx++;
    }
    layerBottomY.push(maxY);
  }

  // Build neuron lookup: [layerIdx][neuronIdx] → flat index
  const lookup: number[][] = [];
  for (let li = 0; li < LAYERS.length; li++) lookup[li] = [];
  for (let i = 0; i < totalNeurons; i++) {
    lookup[posLayerIdx[i]][posNeuronIdx[i]] = i;
  }

  return { posX, posY, posLayerIdx, posNeuronIdx, totalNeurons, radius, lookup, layerStartIdx, layerBottomY, layerX };
}

// ---------------------------------------------------------------------------
// Extract activation values (global normalization + sqrt compression)
// Sqrt compression keeps dim layers visible while dominant layer stands out.
// e.g. a value at 4% of global max → sqrt(0.04) ≈ 0.20 (visible)
//      a value at 100% of global max → sqrt(1.0) = 1.0 (brightest)
// ---------------------------------------------------------------------------

function extractActivations(
  layerActivations: Record<string, number[][][] | number[]>,
  inputTensor: number[][] | null,
  prediction: number[] | null,
): Map<string, number[]> {
  const result = new Map<string, number[]>();
  const rawValues = new Map<string, number[]>();

  // Input: 20 image segments (5 cols × 4 rows grid)
  if (inputTensor) {
    const patchCols = 5, patchRows = 4;
    const pw = 28 / patchCols, ph = 28 / patchRows;
    const patches: number[] = [];
    for (let pr = 0; pr < patchRows; pr++) {
      for (let pc = 0; pc < patchCols; pc++) {
        const r0 = Math.floor(pr * ph), r1 = Math.floor((pr + 1) * ph);
        const c0 = Math.floor(pc * pw), c1 = Math.floor((pc + 1) * pw);
        let sum = 0, count = 0;
        for (let r = r0; r < r1; r++) for (let c = c0; c < c1; c++) { sum += inputTensor[r]?.[c] ?? 0; count++; }
        patches.push(count > 0 ? sum / count : 0);
      }
    }
    rawValues.set("input", patches);
  }

  // Conv/ReLU/Pool layers (3D or 1D)
  for (const layer of LAYERS) {
    if (layer.name === "input" || layer.name === "output") continue;
    const acts = layerActivations[layer.name];
    if (!acts) continue;

    const sampled: number[] = [];
    if (Array.isArray(acts[0]) && Array.isArray((acts[0] as number[][])[0])) {
      const acts3d = acts as number[][][];
      const means: number[] = [];
      for (const ch of acts3d) {
        let sum = 0, count = 0;
        for (const row of ch) for (const v of row) { sum += Math.abs(v); count++; }
        means.push(sum / count);
      }
      for (let i = 0; i < layer.displayNeurons; i++) {
        sampled.push(means[Math.floor(i * means.length / layer.displayNeurons)]);
      }
    } else {
      const acts1d = acts as number[];
      for (let i = 0; i < layer.displayNeurons; i++) {
        sampled.push(Math.abs(acts1d[Math.floor(i * acts1d.length / layer.displayNeurons)]));
      }
    }
    rawValues.set(layer.name, sampled);
  }

  // Output: use softmax probabilities (0-1 range) instead of raw logits
  if (prediction && prediction.length > 0) {
    const valid: { val: number; idx: number }[] = [];
    for (let i = 0; i < prediction.length; i++) {
      if (!BYMERGE_MERGED_INDICES.has(i)) valid.push({ val: prediction[i], idx: i });
    }
    valid.sort((a, b) => b.val - a.val);
    rawValues.set("output", valid.slice(0, 10).map(d => d.val));
  }

  // Global normalization: find max across ALL layers
  let globalMax = 0;
  for (const [, vals] of rawValues) {
    for (const v of vals) if (v > globalMax) globalMax = v;
  }
  globalMax = Math.max(globalMax, 0.001);

  // Normalize globally then apply sqrt compression
  for (const [name, vals] of rawValues) {
    result.set(name, vals.map(v => Math.sqrt(v / globalMax)));
  }

  return result;
}

function getOutputLabels(prediction: number[] | null): string[] {
  if (!prediction || prediction.length === 0) return [];
  const valid: { val: number; idx: number }[] = [];
  for (let i = 0; i < prediction.length; i++) {
    if (!BYMERGE_MERGED_INDICES.has(i)) valid.push({ val: prediction[i], idx: i });
  }
  valid.sort((a, b) => b.val - a.val);
  return valid.slice(0, 10).map(d => EMNIST_CLASSES[d.idx]);
}

// ---------------------------------------------------------------------------
// Bezier point (inlined for perf — avoid tuple allocation)
// ---------------------------------------------------------------------------

let _bx = 0, _by = 0;
function bezierPointInline(
  x0: number, y0: number, cx1: number, cy1: number,
  cx2: number, cy2: number, x1: number, y1: number, t: number,
) {
  const u = 1 - t;
  const uu = u * u, tt = t * t;
  _bx = uu * u * x0 + 3 * uu * t * cx1 + 3 * u * tt * cx2 + tt * t * x1;
  _by = uu * u * y0 + 3 * uu * t * cy1 + 3 * u * tt * cy2 + tt * t * y1;
}

function displayToActualIndex(layerIdx: number, displayIdx: number): number {
  const layer = LAYERS[layerIdx];
  if (layer.displayNeurons >= layer.totalNeurons) return displayIdx;
  return Math.floor(displayIdx * layer.totalNeurons / layer.displayNeurons);
}

function viridis(t: number): [number, number, number] {
  t = Math.max(0, Math.min(1, t));
  return [
    Math.round(255 * Math.max(0, Math.min(1, -0.33 + 2.2 * t * t))),
    Math.round(255 * Math.max(0, Math.min(1, -0.15 + 1.5 * t - 0.5 * t * t))),
    Math.round(255 * Math.max(0, Math.min(1, 0.52 + 0.6 * t - 1.2 * t * t))),
  ];
}

// ---------------------------------------------------------------------------
// NeuronNetworkCanvas — OPTIMIZED: no React state in render loop
// ---------------------------------------------------------------------------

function NeuronNetworkCanvas({
  width,
  height,
  activationMapRef,
  outputLabelsRef,
  hoveredLayerRef,
  hoveredNeuronRef,
  waveRef,
  onHoverLayer,
  onHoverNeuron,
  onClickLayer,
}: {
  width: number;
  height: number;
  activationMapRef: React.RefObject<Map<string, number[]>>;
  outputLabelsRef: React.RefObject<string[]>;
  hoveredLayerRef: React.RefObject<number | null>;
  hoveredNeuronRef: React.RefObject<HoveredNeuron | null>;
  waveRef: React.RefObject<number>;
  onHoverLayer: (li: number | null) => void;
  onHoverNeuron: (n: HoveredNeuron | null) => void;
  onClickLayer: (li: number, neuronIdx: number | null) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dpr = typeof window !== "undefined" ? Math.min(window.devicePixelRatio || 1, 2) : 1;

  const layout = useMemo(() => computeLayout(width, height), [width, height]);
  const layoutRef = useRef(layout);
  layoutRef.current = layout;

  // Pre-compute resolved connection positions for current layout
  const connResolved = useMemo(() => {
    const l = layout;
    const fromXArr = new Float32Array(CONN_COUNT);
    const fromYArr = new Float32Array(CONN_COUNT);
    const toXArr = new Float32Array(CONN_COUNT);
    const toYArr = new Float32Array(CONN_COUNT);
    for (let ci = 0; ci < CONN_COUNT; ci++) {
      const fi = l.lookup[connFromLayer[ci]]?.[connFromNeuron[ci]];
      const ti = l.lookup[connToLayer[ci]]?.[connToNeuron[ci]];
      if (fi !== undefined && ti !== undefined) {
        fromXArr[ci] = l.posX[fi];
        fromYArr[ci] = l.posY[fi];
        toXArr[ci] = l.posX[ti];
        toYArr[ci] = l.posY[ti];
      }
    }
    return { fromXArr, fromYArr, toXArr, toYArr };
  }, [layout]);

  // Animation loop — reads refs only, zero React state updates
  useEffect(() => {
    let raf = 0;
    const animate = () => {
      const canvas = canvasRef.current;
      if (!canvas) { raf = requestAnimationFrame(animate); return; }
      const ctx = canvas.getContext("2d");
      if (!ctx) { raf = requestAnimationFrame(animate); return; }

      const activationMap = activationMapRef.current;
      const outputLabels = outputLabelsRef.current;
      const hoveredLayer = hoveredLayerRef.current;
      const hoveredNeuron = hoveredNeuronRef.current;
      const waveProgress = waveRef.current;
      const l = layoutRef.current;
      const hasData = activationMap.size > 0;
      const { fromXArr, fromYArr, toXArr, toYArr } = connResolved;

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);

      // --- Connections (batched into 8 alpha buckets = 8 draw calls) ---
      {
        const ALPHA_BUCKETS = 8;
        const bucketPaths: Path2D[] = [];
        for (let b = 0; b < ALPHA_BUCKETS; b++) bucketPaths[b] = new Path2D();

        for (let ci = 0; ci < CONN_COUNT; ci++) {
          const fl = connFromLayer[ci], tl = connToLayer[ci];
          const fn = connFromNeuron[ci], tn = connToNeuron[ci];
          const fromActs = activationMap.get(LAYERS[fl].name);
          const toActs = activationMap.get(LAYERS[tl].name);
          const connStrength = ((fromActs?.[fn] ?? 0) + (toActs?.[tn] ?? 0)) * 0.5;

          const fromWave = waveProgress - fl;
          const toWave = waveProgress - tl;
          const waveAlpha = Math.min(
            fromWave < 0 ? 0 : fromWave > 1 ? 1 : fromWave,
            toWave < 0 ? 0 : toWave > 1 ? 1 : toWave,
          );

          let baseAlpha: number;
          if (hasData) {
            baseAlpha = 0.03 + connStrength * 0.55;
            if (hoveredLayer !== null && (fl === hoveredLayer || tl === hoveredLayer)) {
              baseAlpha = 0.15 + connStrength * 0.7;
            }
          } else {
            baseAlpha = hoveredLayer !== null && (fl === hoveredLayer || tl === hoveredLayer) ? 0.08 : 0.025;
          }
          const alpha = hasData ? baseAlpha * waveAlpha : baseAlpha;
          if (alpha < 0.005) continue;

          const bucket = Math.min((alpha * ALPHA_BUCKETS) | 0, ALPHA_BUCKETS - 1);
          const fx = fromXArr[ci], fy = fromYArr[ci], tx = toXArr[ci], ty = toYArr[ci];
          const dx = (tx - fx) * 0.4;
          const p = bucketPaths[bucket];
          p.moveTo(fx, fy);
          p.bezierCurveTo(fx + dx, fy, tx - dx, ty, tx, ty);
        }

        ctx.lineWidth = 0.6;
        for (let b = 0; b < ALPHA_BUCKETS; b++) {
          const a = (b + 0.5) / ALPHA_BUCKETS;
          ctx.strokeStyle = `rgba(140,160,200,${a})`;
          ctx.stroke(bucketPaths[b]);
        }
      }

      // --- Signal highlights (bright segment sliding along connections) ---
      if (hasData) {
        // Batch by layer color (LAYERS.length buckets)
        const sigPaths: Path2D[] = [];
        const sigAlphas: number[] = [];
        for (let li = 0; li < LAYERS.length; li++) { sigPaths[li] = new Path2D(); sigAlphas[li] = 0; }

        for (let i = 0; i < SIGNAL_COUNT; i++) {
          sigProgress[i] += sigSpeed[i];
          if (sigProgress[i] > 1) {
            sigProgress[i] -= 1;
            sigIntensity[i] = 0.4 + Math.random() * 0.6;
          }

          const ci = sigConnIdx[i];
          const fl = connFromLayer[ci];
          const layerWave = waveProgress - fl;
          if (layerWave < 0.5) continue;

          const fromActs = activationMap.get(LAYERS[fl].name);
          const val = fromActs?.[connFromNeuron[ci]] ?? 0;
          if (val < 0.1) continue;

          const tl = connToLayer[ci];
          const fx = fromXArr[ci], fy = fromYArr[ci], tx = toXArr[ci], ty = toYArr[ci];
          const ddx = (tx - fx) * 0.4;
          const cx1 = fx + ddx, cy1 = fy, cx2 = tx - ddx, cy2 = ty;
          const a = val * sigIntensity[i] * 0.7;
          if (a > sigAlphas[tl]) sigAlphas[tl] = a;

          // 4-step segment (fast)
          const t0 = sigProgress[i];
          const p = sigPaths[tl];
          bezierPointInline(fx, fy, cx1, cy1, cx2, cy2, tx, ty, Math.min(t0, 1));
          p.moveTo(_bx, _by);
          bezierPointInline(fx, fy, cx1, cy1, cx2, cy2, tx, ty, Math.min(t0 + 0.04, 1));
          p.lineTo(_bx, _by);
          bezierPointInline(fx, fy, cx1, cy1, cx2, cy2, tx, ty, Math.min(t0 + 0.08, 1));
          p.lineTo(_bx, _by);
          bezierPointInline(fx, fy, cx1, cy1, cx2, cy2, tx, ty, Math.min(t0 + 0.12, 1));
          p.lineTo(_bx, _by);
        }

        ctx.lineWidth = 1.8;
        ctx.lineCap = "round";
        for (let li = 0; li < LAYERS.length; li++) {
          if (sigAlphas[li] < 0.01) continue;
          const rgb = LAYERS[li].rgb;
          ctx.strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${sigAlphas[li] * 0.8})`;
          ctx.stroke(sigPaths[li]);
        }
      }

      // --- Neurons (batched per layer) ---
      {
        const r = l.radius;
        const TAU = 6.2832;
        for (let li = 0; li < LAYERS.length; li++) {
          const layer = LAYERS[li];
          const rgb = layer.rgb;
          const acts = activationMap.get(layer.name);
          const layerWave = hasData ? waveProgress - li : 0;
          const wClamp = layerWave < 0 ? 0 : layerWave > 1 ? 1 : layerWave;
          const isHovered = hoveredLayer === li;
          const startI = l.layerStartIdx[li];
          const count = layer.displayNeurons;

          // Pass 1: Glow (batch all glowing neurons in this layer)
          const glowPath = new Path2D();
          let hasGlow = false;
          let maxGlowAct = 0;
          for (let ni = 0; ni < count; ni++) {
            const idx = startI + ni;
            const activation = acts?.[ni] ?? 0;
            const effectiveAct = activation * wClamp;
            if (effectiveAct > 0.15) {
              glowPath.moveTo(l.posX[idx] + r * 2.2, l.posY[idx]);
              glowPath.arc(l.posX[idx], l.posY[idx], r * 2.2, 0, TAU);
              hasGlow = true;
              if (effectiveAct > maxGlowAct) maxGlowAct = effectiveAct;
            }
          }
          if (hasGlow) {
            ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${maxGlowAct * 0.2})`;
            ctx.fill(glowPath);
          }

          // Pass 2: Neuron bodies — batch active and inactive separately
          const activePath = new Path2D();
          const dimPath = new Path2D();
          let maxBodyAct = 0;
          let hasActive = false, hasDim = false;
          for (let ni = 0; ni < count; ni++) {
            const idx = startI + ni;
            const activation = acts?.[ni] ?? 0;
            const effectiveAct = activation * wClamp;
            const x = l.posX[idx], y = l.posY[idx];
            if (effectiveAct > 0.01) {
              activePath.moveTo(x + r, y);
              activePath.arc(x, y, r, 0, TAU);
              hasActive = true;
              if (effectiveAct > maxBodyAct) maxBodyAct = effectiveAct;
            } else {
              dimPath.moveTo(x + r, y);
              dimPath.arc(x, y, r, 0, TAU);
              hasDim = true;
            }
          }
          if (hasActive) {
            ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${0.15 + maxBodyAct * 0.85})`;
            ctx.fill(activePath);
          }
          if (hasDim) {
            ctx.fillStyle = isHovered ? "rgba(255,255,255,0.08)" : "rgba(255,255,255,0.03)";
            ctx.fill(dimPath);
          }

          // Pass 3: Borders — batch per layer
          const borderPath = new Path2D();
          for (let ni = 0; ni < count; ni++) {
            const idx = startI + ni;
            const x = l.posX[idx], y = l.posY[idx];
            borderPath.moveTo(x + r, y);
            borderPath.arc(x, y, r, 0, TAU);
          }
          const hovNeuronInLayer = hoveredNeuron?.layerIdx === li;
          if (isHovered) {
            ctx.strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.6)`;
            ctx.lineWidth = 1.5;
          } else {
            ctx.strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.25)`;
            ctx.lineWidth = 0.8;
          }
          ctx.stroke(borderPath);

          // Hovered neuron highlight (individual — rare)
          if (hovNeuronInLayer && hoveredNeuron) {
            const ni = hoveredNeuron.neuronIdx;
            const idx = startI + ni;
            ctx.beginPath();
            ctx.arc(l.posX[idx], l.posY[idx], r, 0, TAU);
            ctx.strokeStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
            ctx.lineWidth = 2.5;
            ctx.stroke();
          }

          // Output labels (only 10 max, cheap)
          if (layer.type === "output") {
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            for (let ni = 0; ni < count; ni++) {
              if (!outputLabels[ni]) continue;
              const idx = startI + ni;
              const activation = acts?.[ni] ?? 0;
              const effectiveAct = activation * wClamp;
              const la = effectiveAct > 0.3 ? 0.4 + effectiveAct * 0.6 : 0.25;
              ctx.fillStyle = `rgba(255,255,255,${la})`;
              ctx.font = effectiveAct > 0.5 ? "bold 10px system-ui,sans-serif" : "10px system-ui,sans-serif";
              ctx.fillText(outputLabels[ni], l.posX[idx] + r + 6, l.posY[idx]);
            }
          }
        }
      }

      // --- Layer labels (cheap, drawn once per layer) ---
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (let li = 0; li < LAYERS.length; li++) {
        const layer = LAYERS[li];
        const x = l.layerX[li];
        const maxY = l.layerBottomY[li];
        const isHov = hoveredLayer === li;

        ctx.fillStyle = isHov ? layer.color : "rgba(255,255,255,0.4)";
        ctx.font = isHov ? "bold 10px system-ui,sans-serif" : "10px system-ui,sans-serif";
        ctx.fillText(layer.displayName, x, maxY + 16);

        const unit = (layer.type === "conv" || layer.type === "relu" || layer.type === "pool") && layer.name !== "relu4" ? "ch" : "";
        ctx.fillStyle = "rgba(255,255,255,0.15)";
        ctx.font = "8px system-ui,sans-serif";
        ctx.fillText(`${layer.totalNeurons}${unit ? " " + unit : " neurons"}`, x, maxY + 28);

        if (layer.totalNeurons > layer.displayNeurons) {
          ctx.fillStyle = "rgba(255,255,255,0.12)";
          for (let d = 0; d < 3; d++) {
            ctx.beginPath();
            ctx.arc(x - 4 + d * 4, maxY + 8, 1, 0, 6.2832);
            ctx.fill();
          }
        }
      }

      raf = requestAnimationFrame(animate);
    };

    raf = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(raf);
  }, [width, height, connResolved, activationMapRef, outputLabelsRef, hoveredLayerRef, hoveredNeuronRef, waveRef]);

  // Mouse handling
  const findNearest = useCallback((clientX: number, clientY: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return { layer: null as number | null, neuron: null as HoveredNeuron | null };
    const rect = canvas.getBoundingClientRect();
    const scaleX = width / rect.width, scaleY = height / rect.height;
    const mx = (clientX - rect.left) * scaleX, my = (clientY - rect.top) * scaleY;
    const l = layoutRef.current;

    let closestNeuron: HoveredNeuron | null = null;
    let closestLayer: number | null = null;
    let minNeuronDist = 18, minLayerDist = 40;

    for (let i = 0; i < l.totalNeurons; i++) {
      const dx = mx - l.posX[i], dy = my - l.posY[i];
      const absDx = dx < 0 ? -dx : dx;
      if (absDx < minLayerDist) { minLayerDist = absDx; closestLayer = l.posLayerIdx[i]; }
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < minNeuronDist) {
        minNeuronDist = dist;
        closestNeuron = {
          layerIdx: l.posLayerIdx[i], neuronIdx: l.posNeuronIdx[i],
          screenX: l.posX[i] / scaleX + rect.left, screenY: l.posY[i] / scaleY + rect.top,
        };
      }
    }
    return { layer: closestLayer, neuron: closestNeuron };
  }, [width, height]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const { layer, neuron } = findNearest(e.clientX, e.clientY);
    onHoverLayer(layer);
    onHoverNeuron(neuron);
  }, [findNearest, onHoverLayer, onHoverNeuron]);

  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const { layer, neuron } = findNearest(e.clientX, e.clientY);
    if (layer !== null) onClickLayer(layer, neuron?.neuronIdx ?? null);
  }, [findNearest, onClickLayer]);

  return (
    <canvas
      ref={canvasRef}
      width={width * dpr}
      height={height * dpr}
      onMouseMove={handleMouseMove}
      onMouseLeave={() => { onHoverLayer(null); onHoverNeuron(null); }}
      onClick={handleClick}
      style={{ width, height, cursor: "pointer" }}
    />
  );
}

// ---------------------------------------------------------------------------
// NeuronHeatmapTooltip
// ---------------------------------------------------------------------------

function NeuronHeatmapTooltip({
  neuron, layerActivations, inputTensor, outputLabels, containerRect,
}: {
  neuron: HoveredNeuron;
  layerActivations: Record<string, number[][][] | number[]>;
  inputTensor: number[][] | null;
  outputLabels: string[];
  containerRect: DOMRect | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const layer = LAYERS[neuron.layerIdx];
  const actualIdx = displayToActualIndex(neuron.layerIdx, neuron.neuronIdx);

  const isConv3D = layer.type === "conv" || layer.type === "relu" || layer.type === "pool";
  const isDense = layer.type === "dense" || (layer.type === "relu" && layer.name === "relu4");
  const isInput = layer.type === "input";
  const isOutput = layer.type === "output";
  const canvasSize = isInput ? 112 : (isConv3D ? 112 : 80);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const w = canvas.width, h = canvas.height;
    ctx.fillStyle = "#111118";
    ctx.fillRect(0, 0, w, h);

    if (isInput && inputTensor) {
      const cellW = w / 28, cellH = h / 28;
      for (let r = 0; r < 28; r++) for (let c = 0; c < 28; c++) {
        const gray = Math.round(inputTensor[r][c] * 255);
        ctx.fillStyle = `rgb(${gray},${gray},${gray})`;
        ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5);
      }
      const patchCols = 5, patchRows = 4;
      const pc = neuron.neuronIdx % patchCols, pr = Math.floor(neuron.neuronIdx / patchCols);
      const px = Math.floor(pc * 28 / patchCols) * cellW, py = Math.floor(pr * 28 / patchRows) * cellH;
      const pw = Math.floor(28 / patchCols) * cellW, ph = Math.floor(28 / patchRows) * cellH;
      ctx.strokeStyle = "#6366f1"; ctx.lineWidth = 2; ctx.strokeRect(px, py, pw, ph);
      ctx.fillStyle = "rgba(0,0,0,0.5)";
      ctx.fillRect(0, 0, w, py); ctx.fillRect(0, py + ph, w, h - py - ph);
      ctx.fillRect(0, py, px, ph); ctx.fillRect(px + pw, py, w - px - pw, ph);
    } else if (isConv3D && layer.name !== "relu4") {
      const acts = layerActivations[layer.name];
      if (acts && Array.isArray(acts[0]) && Array.isArray((acts[0] as number[][])[0])) {
        const acts3d = acts as number[][][];
        if (actualIdx < acts3d.length) {
          const ch = acts3d[actualIdx];
          const rows = ch.length, cols = ch[0].length;
          let maxVal = 0;
          for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) if (Math.abs(ch[r][c]) > maxVal) maxVal = Math.abs(ch[r][c]);
          maxVal = Math.max(maxVal, 0.001);
          const cellW = w / cols, cellH = h / rows;
          for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
            const [cr, cg, cb] = viridis(Math.abs(ch[r][c]) / maxVal);
            ctx.fillStyle = `rgb(${cr},${cg},${cb})`;
            ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5);
          }
        }
      }
    } else if (isDense) {
      const acts = layerActivations[layer.name];
      if (acts && !Array.isArray(acts[0])) {
        const vals = acts as number[];
        if (actualIdx < vals.length) {
          let maxVal = 0;
          for (const val of vals) if (Math.abs(val) > maxVal) maxVal = Math.abs(val);
          maxVal = Math.max(maxVal, 0.001);
          const norm = Math.abs(vals[actualIdx]) / maxVal;
          const [cr, cg, cb] = viridis(norm);
          ctx.fillStyle = `rgb(${cr},${cg},${cb})`;
          ctx.fillRect(4, h / 2 - 10, norm * (w - 8), 20);
          ctx.strokeStyle = "rgba(255,255,255,0.1)"; ctx.strokeRect(4, h / 2 - 10, w - 8, 20);
        }
      }
    } else if (isOutput) {
      const acts = layerActivations["output"];
      if (acts && !Array.isArray(acts[0])) {
        const vals = acts as number[];
        const valid: { val: number; idx: number }[] = [];
        for (let i = 0; i < vals.length; i++) if (!BYMERGE_MERGED_INDICES.has(i)) valid.push({ val: vals[i], idx: i });
        valid.sort((a, b) => b.val - a.val);
        if (neuron.neuronIdx < valid.length) {
          const d = valid[neuron.neuronIdx];
          const [cr, cg, cb] = viridis(d.val);
          ctx.fillStyle = `rgb(${cr},${cg},${cb})`;
          ctx.fillRect(4, h / 2 - 10, d.val * (w - 8), 20);
          ctx.strokeStyle = "rgba(255,255,255,0.1)"; ctx.strokeRect(4, h / 2 - 10, w - 8, 20);
        }
      }
    }
  }, [neuron, layerActivations, inputTensor, layer, actualIdx, isConv3D, isDense, isInput, isOutput, canvasSize]);

  let label = "";
  if (isInput) { const pc = neuron.neuronIdx % 5, pr = Math.floor(neuron.neuronIdx / 5); label = `Patch [${pr},${pc}]`; }
  else if (isConv3D && layer.name !== "relu4") label = `Channel ${actualIdx}`;
  else if (isDense) label = `Neuron ${actualIdx}`;
  else if (isOutput) label = outputLabels[neuron.neuronIdx] ? `Class "${outputLabels[neuron.neuronIdx]}"` : `Output ${neuron.neuronIdx}`;

  let valueText = "";
  if (isInput && inputTensor) {
    const patchCols = 5, patchRows = 4;
    const pc = neuron.neuronIdx % patchCols, pr = Math.floor(neuron.neuronIdx / patchCols);
    const r0 = Math.floor(pr * 28 / patchRows), r1 = Math.floor((pr + 1) * 28 / patchRows);
    const c0 = Math.floor(pc * 28 / patchCols), c1 = Math.floor((pc + 1) * 28 / patchCols);
    let sum = 0, count = 0;
    for (let r = r0; r < r1; r++) for (let c = c0; c < c1; c++) { sum += inputTensor[r]?.[c] ?? 0; count++; }
    valueText = `mean: ${(sum / count).toFixed(3)}`;
  } else if ((isConv3D && layer.name !== "relu4") || isDense) {
    const acts = layerActivations[layer.name];
    if (acts) {
      if (Array.isArray(acts[0])) {
        const acts3d = acts as number[][][];
        if (actualIdx < acts3d.length) {
          let sum = 0, count = 0;
          for (const row of acts3d[actualIdx]) for (const v of row) { sum += Math.abs(v); count++; }
          valueText = `mean |act|: ${(sum / count).toFixed(4)}`;
        }
      } else {
        const vals = acts as number[];
        if (actualIdx < vals.length) valueText = `value: ${vals[actualIdx].toFixed(4)}`;
      }
    }
  } else if (isOutput) {
    const acts = layerActivations["output"];
    if (acts && !Array.isArray(acts[0])) {
      const vals = acts as number[];
      const valid: { val: number; idx: number }[] = [];
      for (let i = 0; i < vals.length; i++) if (!BYMERGE_MERGED_INDICES.has(i)) valid.push({ val: vals[i], idx: i });
      valid.sort((a, b) => b.val - a.val);
      if (neuron.neuronIdx < valid.length) valueText = `prob: ${(valid[neuron.neuronIdx].val * 100).toFixed(2)}%`;
    }
  }

  if (!containerRect) return null;
  const tooltipX = neuron.screenX - containerRect.left + 20;
  const tooltipY = neuron.screenY - containerRect.top - 30;

  return (
    <div style={{
      position: "absolute", left: tooltipX, top: tooltipY, transform: "translateY(-100%)",
      background: "#15151f", border: `1px solid ${layer.color}50`, borderRadius: 8, padding: 8,
      pointerEvents: "none", zIndex: 60, boxShadow: `0 4px 20px rgba(0,0,0,0.6)`, minWidth: 120,
    }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: layer.color, marginBottom: 4 }}>{layer.displayName} — {label}</div>
      <canvas ref={canvasRef} width={canvasSize} height={isDense || isOutput ? 40 : canvasSize}
        style={{ width: canvasSize, height: isDense || isOutput ? 40 : canvasSize, borderRadius: 4, imageRendering: (isInput || isConv3D) ? "pixelated" : "auto", display: "block" }}
      />
      {valueText && <div style={{ fontSize: 10, color: "rgba(255,255,255,0.5)", marginTop: 4, fontFamily: "monospace" }}>{valueText}</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// LayerTooltip
// ---------------------------------------------------------------------------

function LayerTooltip({ layer, activationMap }: { layer: NeuronLayerDef; activationMap: Map<string, number[]> }) {
  const acts = activationMap.get(layer.name);
  const meanAct = acts ? acts.reduce((s, v) => s + v, 0) / acts.length : 0;
  return (
    <div style={{
      position: "absolute", bottom: 16, left: "50%", transform: "translateX(-50%)",
      background: "#1a1a28", border: `1px solid ${layer.color}40`, borderRadius: 8,
      padding: "10px 16px", display: "flex", alignItems: "center", gap: 12,
      pointerEvents: "none", zIndex: 50, whiteSpace: "nowrap",
    }}>
      <div style={{ width: 10, height: 10, borderRadius: "50%", background: layer.color }} />
      <div>
        <div style={{ fontSize: 13, fontWeight: 600, color: "#e8e8ed" }}>{layer.displayName}</div>
        <div style={{ fontSize: 11, color: "rgba(255,255,255,0.4)" }}>{layer.description}</div>
      </div>
      <div style={{ fontSize: 11, color: "rgba(255,255,255,0.3)", fontFamily: "monospace" }}>
        {layer.totalNeurons.toLocaleString()} {(layer.type === "conv" || (layer.type === "relu" && layer.name !== "relu4") || layer.type === "pool") ? "ch" : (layer.type === "input" ? "px" : "n")}
      </div>
      {acts && (
        <div style={{ fontSize: 11, fontFamily: "monospace" }}>
          <span style={{ color: "rgba(255,255,255,0.3)" }}>avg: </span>
          <span style={{ color: layer.color }}>{(meanAct * 100).toFixed(1)}%</span>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// InspectorPanel
// ---------------------------------------------------------------------------

function InspectorPanel({
  layer, activations, inputTensor, prediction, topPrediction, initialChannel, onClose,
}: {
  layer: NeuronLayerDef; activations: number[][][] | number[] | null; inputTensor: number[][] | null;
  prediction: number[] | null; topPrediction: { classIndex: number; confidence: number } | null;
  initialChannel: number; onClose: () => void;
}) {
  const [selectedChannel, setSelectedChannel] = useState(initialChannel);
  const mainCanvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = mainCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const w = canvas.width, h = canvas.height;
    ctx.fillStyle = "#111118"; ctx.fillRect(0, 0, w, h);

    if (layer.type === "input" && inputTensor) {
      const cellW = w / 28, cellH = h / 28;
      for (let r = 0; r < 28; r++) for (let c = 0; c < 28; c++) {
        const gray = Math.round(inputTensor[r][c] * 255);
        ctx.fillStyle = `rgb(${gray},${gray},${gray})`;
        ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5);
      }
    } else if (layer.type === "output" && prediction) {
      const sorted = prediction.map((v, i) => ({ v, i })).filter(d => !BYMERGE_MERGED_INDICES.has(d.i)).sort((a, b) => b.v - a.v);
      const barH = h / Math.min(sorted.length, 20);
      ctx.font = "11px system-ui,sans-serif";
      for (let j = 0; j < Math.min(sorted.length, 20); j++) {
        const d = sorted[j]; const barW = (d.v / Math.max(sorted[0].v, 0.001)) * (w - 60);
        const [cr, cg, cb] = viridis(d.v / Math.max(sorted[0].v, 0.001));
        ctx.fillStyle = `rgb(${cr},${cg},${cb})`; ctx.fillRect(40, j * barH + 2, barW, barH - 4);
        ctx.fillStyle = "#e8e8ed"; ctx.textAlign = "right"; ctx.fillText(EMNIST_CLASSES[d.i], 35, j * barH + barH / 2 + 4);
        ctx.textAlign = "left"; ctx.fillText(`${(d.v * 100).toFixed(1)}%`, 40 + barW + 4, j * barH + barH / 2 + 4);
      }
    } else if (activations && Array.isArray(activations[0])) {
      const acts = activations as number[][][];
      if (selectedChannel < acts.length) {
        const ch = acts[selectedChannel]; const rows = ch.length, cols = ch[0].length;
        let maxVal = 0;
        for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) if (Math.abs(ch[r][c]) > maxVal) maxVal = Math.abs(ch[r][c]);
        maxVal = Math.max(maxVal, 0.001);
        const cellW = w / cols, cellH = h / rows;
        for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
          const [cr, cg, cb] = viridis(Math.abs(ch[r][c]) / maxVal);
          ctx.fillStyle = `rgb(${cr},${cg},${cb})`; ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5);
        }
      }
    } else if (activations && !Array.isArray(activations[0])) {
      const vals = activations as number[]; const n = vals.length;
      const cols = Math.ceil(Math.sqrt(n)), rows = Math.ceil(n / cols);
      const cellW = w / cols, cellH = h / rows;
      let maxVal = 0; for (const v of vals) if (Math.abs(v) > maxVal) maxVal = Math.abs(v);
      maxVal = Math.max(maxVal, 0.001);
      for (let i = 0; i < n; i++) {
        const [cr, cg, cb] = viridis(Math.abs(vals[i]) / maxVal);
        ctx.fillStyle = `rgb(${cr},${cg},${cb})`;
        ctx.fillRect((i % cols) * cellW + 0.5, Math.floor(i / cols) * cellH + 0.5, cellW - 1, cellH - 1);
      }
    } else {
      ctx.fillStyle = "rgba(255,255,255,0.15)"; ctx.font = "14px system-ui,sans-serif";
      ctx.textAlign = "center"; ctx.fillText("Draw a character to see activations", w / 2, h / 2);
    }
  }, [layer, activations, inputTensor, selectedChannel, prediction]);

  const channelCount = activations && Array.isArray(activations[0]) ? (activations as number[][][]).length : 0;

  const stats = useMemo(() => {
    if (!activations) return null;
    if (Array.isArray(activations[0])) {
      const acts = activations as number[][][];
      let min = Infinity, max = -Infinity, sum = 0, count = 0, activeCount = 0;
      for (const ch of acts) for (const row of ch) for (const v of row) {
        if (v < min) min = v; if (v > max) max = v; sum += v; count++; if (v > 0) activeCount++;
      }
      return { min, max, mean: sum / count, activePercent: (activeCount / count) * 100 };
    } else {
      const vals = activations as number[];
      let min = Infinity, max = -Infinity, sum = 0, activeCount = 0;
      for (const v of vals) { if (v < min) min = v; if (v > max) max = v; sum += v; if (v > 0) activeCount++; }
      return { min, max, mean: sum / vals.length, activePercent: (activeCount / vals.length) * 100 };
    }
  }, [activations]);

  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(0,0,0,0.75)", backdropFilter: "blur(6px)" }}
      onClick={e => { if (e.target === e.currentTarget) onClose(); }}>
      <div style={{ background: "#13131a", border: `1px solid ${layer.color}40`, borderRadius: 12, padding: 24, maxWidth: 900, width: "90vw", maxHeight: "90vh", overflowY: "auto" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16 }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
              <div style={{ width: 12, height: 12, borderRadius: "50%", background: layer.color }} />
              <h2 style={{ fontSize: 20, fontWeight: 600, color: "#e8e8ed", margin: 0 }}>{layer.displayName}</h2>
              <span style={{ fontSize: 13, color: "rgba(255,255,255,0.35)", fontFamily: "monospace" }}>
                {layer.totalNeurons.toLocaleString()} {(layer.type === "conv" || (layer.type === "relu" && layer.name !== "relu4") || layer.type === "pool") ? "channels" : (layer.type === "input" ? "pixels" : "neurons")}
              </span>
            </div>
            <p style={{ fontSize: 13, color: "rgba(255,255,255,0.45)", margin: 0 }}>{layer.description}</p>
          </div>
          <button onClick={onClose} style={{ background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 6, color: "rgba(255,255,255,0.6)", padding: "4px 12px", cursor: "pointer", fontSize: 13 }}>Close</button>
        </div>
        {stats && (
          <div style={{ display: "flex", gap: 20, marginBottom: 16, padding: "8px 12px", background: "rgba(255,255,255,0.03)", borderRadius: 8, fontSize: 12, fontFamily: "monospace", color: "rgba(255,255,255,0.5)" }}>
            <span>Min: <span style={{ color: "#e8e8ed" }}>{stats.min.toFixed(3)}</span></span>
            <span>Max: <span style={{ color: "#e8e8ed" }}>{stats.max.toFixed(3)}</span></span>
            <span>Mean: <span style={{ color: "#e8e8ed" }}>{stats.mean.toFixed(3)}</span></span>
            <span>Active: <span style={{ color: "#22c55e" }}>{stats.activePercent.toFixed(1)}%</span></span>
          </div>
        )}
        {layer.type === "output" && topPrediction && (
          <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 16, padding: "12px 16px", background: `${layer.color}15`, borderRadius: 8, border: `1px solid ${layer.color}30` }}>
            <span style={{ fontSize: 36, fontWeight: 700, color: layer.color }}>{EMNIST_CLASSES[topPrediction.classIndex]}</span>
            <div>
              <div style={{ fontSize: 14, color: "#e8e8ed" }}>Predicted: <strong>{EMNIST_CLASSES[topPrediction.classIndex]}</strong></div>
              <div style={{ fontSize: 13, color: "rgba(255,255,255,0.5)" }}>Confidence: {(topPrediction.confidence * 100).toFixed(1)}%</div>
            </div>
          </div>
        )}
        <div style={{ display: "flex", gap: 16 }}>
          <canvas ref={mainCanvasRef} width={channelCount > 0 ? 350 : 500} height={channelCount > 0 ? 350 : (layer.type === "output" ? 400 : 300)}
            style={{ width: channelCount > 0 ? 350 : "100%", height: channelCount > 0 ? 350 : (layer.type === "output" ? 400 : 300), borderRadius: 8, imageRendering: layer.type === "input" ? "pixelated" : "auto" }}
          />
          {channelCount > 0 && (
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: 12, color: "rgba(255,255,255,0.4)", marginBottom: 8 }}>{channelCount} channels — click to inspect</div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 3, maxHeight: 340, overflowY: "auto" }}>
                {Array.from({ length: channelCount }, (_, i) => (
                  <ChannelThumb key={i} chIdx={i} activations={activations as number[][][]} selected={i === selectedChannel} color={layer.color} onClick={() => setSelectedChannel(i)} />
                ))}
              </div>
              <div style={{ fontSize: 11, color: "rgba(255,255,255,0.3)", marginTop: 6, fontFamily: "monospace" }}>Channel {selectedChannel}</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function ChannelThumb({ chIdx, activations, selected, color, onClick }: {
  chIdx: number; activations: number[][][]; selected: boolean; color: string; onClick: () => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const el = canvasRef.current;
    if (!el || chIdx >= activations.length) return;
    const ctx = el.getContext("2d");
    if (!ctx) return;
    const ch = activations[chIdx]; const rows = ch.length, cols = ch[0].length;
    let maxVal = 0;
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) if (Math.abs(ch[r][c]) > maxVal) maxVal = Math.abs(ch[r][c]);
    maxVal = Math.max(maxVal, 0.001);
    const cellW = 40 / cols, cellH = 40 / rows;
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const [cr, cg, cb] = viridis(Math.abs(ch[r][c]) / maxVal);
      ctx.fillStyle = `rgb(${cr},${cg},${cb})`; ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5);
    }
  }, [chIdx, activations]);
  return (
    <canvas ref={canvasRef} width={40} height={40} onClick={onClick}
      style={{ width: 40, height: 40, borderRadius: 3, cursor: "pointer", border: selected ? `2px solid ${color}` : "2px solid transparent", imageRendering: "pixelated" }}
    />
  );
}

// ---------------------------------------------------------------------------
// NeuronNetworkSection — main exported component
// ---------------------------------------------------------------------------

export function NeuronNetworkSection({ children }: { children?: React.ReactNode }) {
  const inputTensor = useInferenceStore(s => s.inputTensor);
  const layerActivations = useInferenceStore(s => s.layerActivations);
  const prediction = useInferenceStore(s => s.prediction);
  const topPrediction = useInferenceStore(s => s.topPrediction);

  const [inspectedLayerIdx, setInspectedLayerIdx] = useState<number | null>(null);
  const [inspectedNeuronIdx, setInspectedNeuronIdx] = useState<number | null>(null);
  // Hover state as refs — avoids re-render on every mouse move
  const [, forceRender] = useState(0);
  const hoveredLayerRef = useRef<number | null>(null);
  const hoveredNeuronRef = useRef<HoveredNeuron | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasContainerRef = useRef<HTMLDivElement>(null);
  const [containerSize, setContainerSize] = useState({ w: 1200, h: 500 });

  // Wave progress — ref only, no state, updated in RAF
  const waveRef = useRef(0);
  const waveTargetRef = useRef(0);
  const hasData = Object.keys(layerActivations).length > 0;

  useEffect(() => {
    waveTargetRef.current = hasData ? LAYERS.length + 1 : 0;
    if (!hasData) waveRef.current = 0;
  }, [hasData]);

  // Wave animation via RAF — no React state updates
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const target = waveTargetRef.current;
      const current = waveRef.current;
      if (Math.abs(target - current) > 0.01) {
        waveRef.current = target > current ? current + (target - current) * 0.035 : 0;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  // Measure — outer fills remaining viewport, canvas fills its flex area
  useEffect(() => {
    const measure = () => {
      if (containerRef.current) {
        const top = containerRef.current.getBoundingClientRect().top + window.scrollY;
        const h = Math.max(400, window.innerHeight - top);
        containerRef.current.style.height = `${h}px`;
      }
      if (canvasContainerRef.current) {
        const rect = canvasContainerRef.current.getBoundingClientRect();
        setContainerSize({ w: Math.round(rect.width), h: Math.round(rect.height) });
      }
    };
    measure();
    // Re-measure after a tick so flex layout has settled
    requestAnimationFrame(measure);
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, []);

  // Activation data as refs (canvas reads these directly)
  const activationMapRef = useRef<Map<string, number[]>>(new Map());
  const outputLabelsRef = useRef<string[]>([]);

  const activationMap = useMemo(() => extractActivations(layerActivations, inputTensor, prediction), [layerActivations, inputTensor, prediction]);
  activationMapRef.current = activationMap;

  const outputLabels = useMemo(() => getOutputLabels(prediction), [prediction]);
  outputLabelsRef.current = outputLabels;

  const onHoverLayer = useCallback((li: number | null) => {
    if (hoveredLayerRef.current !== li) {
      hoveredLayerRef.current = li;
      // Only re-render for tooltip display, not for canvas (canvas reads ref)
      forceRender(n => n + 1);
    }
  }, []);

  const onHoverNeuron = useCallback((n: HoveredNeuron | null) => {
    const prev = hoveredNeuronRef.current;
    if (prev?.layerIdx !== n?.layerIdx || prev?.neuronIdx !== n?.neuronIdx) {
      hoveredNeuronRef.current = n;
      forceRender(nn => nn + 1);
    }
  }, []);

  const onClickLayer = useCallback((li: number, ni: number | null) => {
    setInspectedLayerIdx(li);
    setInspectedNeuronIdx(ni);
  }, []);

  const getActivation = useCallback(
    (name: string) => name === "input" ? null : layerActivations[name] ?? null,
    [layerActivations],
  );

  const inspectedLayer = inspectedLayerIdx !== null ? LAYERS[inspectedLayerIdx] : null;
  const hoveredLayer = hoveredLayerRef.current;
  const hoveredNeuron = hoveredNeuronRef.current;

  return (
    <div id="neuron-network" className="relative flex w-full items-stretch" ref={containerRef}>
      {/* Drawing panel — left side */}
      {children}

      {/* Neuron canvas — fills remaining width */}
      <div className="relative min-w-0 flex-1" ref={canvasContainerRef}>
        <NeuronNetworkCanvas
          width={containerSize.w}
          height={containerSize.h}
          activationMapRef={activationMapRef}
          outputLabelsRef={outputLabelsRef}
          hoveredLayerRef={hoveredLayerRef}
          hoveredNeuronRef={hoveredNeuronRef}
          waveRef={waveRef}
          onHoverLayer={onHoverLayer}
          onHoverNeuron={onHoverNeuron}
          onClickLayer={onClickLayer}
        />

        {hoveredNeuron && hasData && (
          <NeuronHeatmapTooltip
            neuron={hoveredNeuron}
            layerActivations={layerActivations}
            inputTensor={inputTensor}
            outputLabels={outputLabels}
            containerRect={canvasContainerRef.current?.getBoundingClientRect() ?? null}
          />
        )}

        {hoveredLayer !== null && !hoveredNeuron && (
          <LayerTooltip layer={LAYERS[hoveredLayer]} activationMap={activationMap} />
        )}
      </div>

      {inspectedLayer && (
        <InspectorPanel
          layer={inspectedLayer}
          activations={getActivation(inspectedLayer.name)}
          inputTensor={inputTensor}
          prediction={prediction}
          topPrediction={topPrediction}
          initialChannel={inspectedNeuronIdx !== null ? displayToActualIndex(inspectedLayerIdx!, inspectedNeuronIdx) : 0}
          onClose={() => { setInspectedLayerIdx(null); setInspectedNeuronIdx(null); }}
        />
      )}
    </div>
  );
}
