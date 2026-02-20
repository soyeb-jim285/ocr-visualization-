import { EMNIST_CLASSES, BYMERGE_MERGED_INDICES } from "@/lib/model/classes";

// ---------------------------------------------------------------------------
// Layer definitions
// ---------------------------------------------------------------------------

export interface NeuronLayerDef {
  name: string;
  displayName: string;
  type: "input" | "conv" | "relu" | "pool" | "dense" | "output";
  totalNeurons: number;
  displayNeurons: number;
  color: string;
  rgb: [number, number, number];
  description: string;
}

export function parseHex(hex: string): [number, number, number] {
  return [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)];
}

export function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

// displayNeurons is proportional to totalNeurons so relative layer sizes are visible.
// Input is fixed (784 pixels would dominate otherwise).
const MAX_DISPLAY = 36;
const MIN_DISPLAY = 6;
const INPUT_DISPLAY = 20;

const LAYERS_RAW: Omit<NeuronLayerDef, "displayNeurons">[] = [
  { name: "input", displayName: "Input", type: "input", totalNeurons: 784, color: "#94a3b8", rgb: parseHex("#94a3b8"), description: "28×28 pixels" },
  { name: "conv1", displayName: "Conv1", type: "conv", totalNeurons: 32, color: "#6366f1", rgb: parseHex("#6366f1"), description: "32 channels (3×3)" },
  { name: "relu1", displayName: "ReLU1", type: "relu", totalNeurons: 32, color: "#818cf8", rgb: parseHex("#818cf8"), description: "32 channels" },
  { name: "conv2", displayName: "Conv2", type: "conv", totalNeurons: 64, color: "#a78bfa", rgb: parseHex("#a78bfa"), description: "64 channels (3×3)" },
  { name: "relu2", displayName: "ReLU2", type: "relu", totalNeurons: 64, color: "#8b5cf6", rgb: parseHex("#8b5cf6"), description: "64 channels" },
  { name: "pool1", displayName: "Pool1", type: "pool", totalNeurons: 64, color: "#06b6d4", rgb: parseHex("#06b6d4"), description: "64 ch → 14×14" },
  { name: "conv3", displayName: "Conv3", type: "conv", totalNeurons: 128, color: "#0891b2", rgb: parseHex("#0891b2"), description: "128 channels (3×3)" },
  { name: "relu3", displayName: "ReLU3", type: "relu", totalNeurons: 128, color: "#0e7490", rgb: parseHex("#0e7490"), description: "128 channels" },
  { name: "pool2", displayName: "Pool2", type: "pool", totalNeurons: 128, color: "#22c55e", rgb: parseHex("#22c55e"), description: "128 ch → 7×7" },
  { name: "dense1", displayName: "Dense", type: "dense", totalNeurons: 256, color: "#f59e0b", rgb: parseHex("#f59e0b"), description: "256 neurons" },
  { name: "relu4", displayName: "ReLU4", type: "relu", totalNeurons: 256, color: "#d97706", rgb: parseHex("#d97706"), description: "256 neurons" },
  { name: "output", displayName: "Output", type: "output", totalNeurons: 47, color: "#ef4444", rgb: parseHex("#ef4444"), description: "Top predictions" },
];

const MAX_TOTAL = Math.max(...LAYERS_RAW.filter(l => l.name !== "input").map(l => l.totalNeurons));

export const LAYERS: NeuronLayerDef[] = LAYERS_RAW.map((l) => ({
  ...l,
  displayNeurons: l.name === "input"
    ? INPUT_DISPLAY
    : Math.max(MIN_DISPLAY, Math.round((l.totalNeurons / MAX_TOTAL) * MAX_DISPLAY)),
}));

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface HoveredNeuron {
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

// Element-wise ops (ReLU, Pool) are 1-to-1 per channel; convs/dense mix channels
function isOneToOne(from: NeuronLayerDef, to: NeuronLayerDef): boolean {
  if (from.type === "conv" && to.type === "relu") return true;
  if (from.type === "relu" && to.type === "pool") return true;
  if (from.type === "dense" && to.type === "relu") return true;
  return false;
}

export const CONN_COUNT = (() => {
  let c = 0;
  for (let li = 0; li < LAYERS.length - 1; li++) {
    if (isOneToOne(LAYERS[li], LAYERS[li + 1])) {
      c += LAYERS[li].displayNeurons; // 1 connection per neuron
    } else {
      c += LAYERS[li].displayNeurons * Math.min(3, LAYERS[li + 1].displayNeurons);
    }
  }
  return c;
})();

export const connFromLayer = new Uint8Array(CONN_COUNT);
export const connFromNeuron = new Uint8Array(CONN_COUNT);
export const connToLayer = new Uint8Array(CONN_COUNT);
export const connToNeuron = new Uint8Array(CONN_COUNT);

(() => {
  const rand = makePRNG(42);
  let ci = 0;
  for (let li = 0; li < LAYERS.length - 1; li++) {
    const from = LAYERS[li];
    const to = LAYERS[li + 1];
    const fromN = from.displayNeurons;
    const toN = to.displayNeurons;

    if (isOneToOne(from, to)) {
      // 1-to-1: each display neuron maps to its proportional counterpart
      for (let fi = 0; fi < fromN; fi++) {
        connFromLayer[ci] = li;
        connFromNeuron[ci] = fi;
        connToLayer[ci] = li + 1;
        connToNeuron[ci] = Math.floor(fi * toN / fromN);
        ci++;
      }
    } else {
      // Many-to-many: sparse random (convs mix channels, dense is FC)
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
  }
})();

// Signal highlights — ~1 per 6 connections for performance
export const SIGNAL_COUNT = Math.ceil(CONN_COUNT / 6);
export const sigConnIdx = new Uint16Array(SIGNAL_COUNT);
export const sigProgress = new Float32Array(SIGNAL_COUNT);
export const sigSpeed = new Float32Array(SIGNAL_COUNT);
export const sigIntensity = new Float32Array(SIGNAL_COUNT);
export const sigSize = new Float32Array(SIGNAL_COUNT);

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

export function computeLayout(w: number, h: number) {
  const totalNeurons = LAYERS.reduce((s, l) => s + l.displayNeurons, 0);
  const posX = new Float32Array(totalNeurons);
  const posY = new Float32Array(totalNeurons);
  const posLayerIdx = new Uint8Array(totalNeurons);
  const posNeuronIdx = new Uint8Array(totalNeurons);

  const marginL = Math.max(12, Math.min(40, w * 0.03));
  const marginR = Math.max(40, Math.min(80, w * 0.06)); // extra for output labels
  const marginY = Math.max(30, Math.min(60, h * 0.08));
  const marginX = marginL; // used by layerX[0]
  const availW = w - marginL - marginR;
  const layerSpacing = availW / (LAYERS.length - 1);
  const radius = Math.min(w < 500 ? 4 : 7, 24 * 0.32);

  const layerStartIdx: number[] = [];
  const layerBottomY: number[] = [];
  const layerX: number[] = [];

  let idx = 0;
  for (let li = 0; li < LAYERS.length; li++) {
    layerStartIdx.push(idx);
    const n = LAYERS[li].displayNeurons;
    const x = marginX + li * layerSpacing;
    layerX.push(x);
    const availH = h - marginY * 2;
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

  const lookup: number[][] = [];
  for (let li = 0; li < LAYERS.length; li++) lookup[li] = [];
  for (let i = 0; i < totalNeurons; i++) {
    lookup[posLayerIdx[i]][posNeuronIdx[i]] = i;
  }

  return { posX, posY, posLayerIdx, posNeuronIdx, totalNeurons, radius, lookup, layerStartIdx, layerBottomY, layerX };
}

// ---------------------------------------------------------------------------
// Extract activation values (global normalization + sqrt compression)
// ---------------------------------------------------------------------------

export function extractActivations(
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
        for (const row of ch) for (const v of row) { sum += v; count++; }
        means.push(sum / count);
      }
      for (let i = 0; i < layer.displayNeurons; i++) {
        sampled.push(means[Math.floor(i * means.length / layer.displayNeurons)]);
      }
    } else {
      const acts1d = acts as number[];
      for (let i = 0; i < layer.displayNeurons; i++) {
        sampled.push(acts1d[Math.floor(i * acts1d.length / layer.displayNeurons)]);
      }
    }
    rawValues.set(layer.name, sampled);
  }

  // Output: use softmax probabilities (0-1 range)
  if (prediction && prediction.length > 0) {
    const valid: { val: number; idx: number }[] = [];
    for (let i = 0; i < prediction.length; i++) {
      if (!BYMERGE_MERGED_INDICES.has(i)) valid.push({ val: prediction[i], idx: i });
    }
    valid.sort((a, b) => b.val - a.val);
    rawValues.set("output", valid.slice(0, 10).map(d => d.val));
  }

  // Per-layer min-max normalization, squared for contrast
  for (const [name, vals] of rawValues) {
    if (name === "output") {
      result.set(name, vals);
    } else {
      let lMin = Infinity, lMax = -Infinity;
      for (const v of vals) {
        if (v < lMin) lMin = v;
        if (v > lMax) lMax = v;
      }
      const range = lMax - lMin;
      if (range < 0.001) {
        result.set(name, vals.map(() => 0.3));
      } else {
        result.set(name, vals.map(v => {
          const t = (v - lMin) / range;
          return t * t;
        }));
      }
    }
  }

  return result;
}

export function getOutputLabels(prediction: number[] | null): string[] {
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

export let _bx = 0, _by = 0;
export function bezierPointInline(
  x0: number, y0: number, cx1: number, cy1: number,
  cx2: number, cy2: number, x1: number, y1: number, t: number,
) {
  const u = 1 - t;
  const uu = u * u, tt = t * t;
  _bx = uu * u * x0 + 3 * uu * t * cx1 + 3 * u * tt * cx2 + tt * t * x1;
  _by = uu * u * y0 + 3 * uu * t * cy1 + 3 * u * tt * cy2 + tt * t * y1;
}

export function displayToActualIndex(layerIdx: number, displayIdx: number): number {
  const layer = LAYERS[layerIdx];
  if (layer.displayNeurons >= layer.totalNeurons) return displayIdx;
  return Math.floor(displayIdx * layer.totalNeurons / layer.displayNeurons);
}

export function viridis(t: number): [number, number, number] {
  t = Math.max(0, Math.min(1, t));
  return [
    Math.round(255 * Math.max(0, Math.min(1, -0.33 + 2.2 * t * t))),
    Math.round(255 * Math.max(0, Math.min(1, -0.15 + 1.5 * t - 0.5 * t * t))),
    Math.round(255 * Math.max(0, Math.min(1, 0.52 + 0.6 * t - 1.2 * t * t))),
  ];
}
