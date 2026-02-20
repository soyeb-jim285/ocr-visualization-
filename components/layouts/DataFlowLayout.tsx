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
import { useDrawingCanvas } from "@/hooks/useDrawingCanvas";
import { useInference } from "@/hooks/useInference";
import { useAnimationFrame } from "@/hooks/useAnimationFrame";
import { LayoutNav } from "@/components/layouts/LayoutNav";

// ---------------------------------------------------------------------------
// Layer pipeline — each layer maps to a column of neurons
// ---------------------------------------------------------------------------

interface NeuronLayerDef {
  name: string;
  displayName: string;
  type: "input" | "conv" | "relu" | "pool" | "dense" | "output";
  totalNeurons: number;
  displayNeurons: number; // how many circles to render
  color: string;
  description: string;
}

// totalNeurons = channel count for conv layers (each filter = one neuron),
// actual neuron count for dense, pixel count for input
const LAYERS: NeuronLayerDef[] = [
  { name: "input", displayName: "Input", type: "input", totalNeurons: 784, displayNeurons: 20, color: "#94a3b8", description: "28×28 pixels" },
  { name: "conv1", displayName: "Conv1", type: "conv", totalNeurons: 32, displayNeurons: 16, color: "#6366f1", description: "32 channels (3×3)" },
  { name: "relu1", displayName: "ReLU1", type: "relu", totalNeurons: 32, displayNeurons: 16, color: "#818cf8", description: "32 channels" },
  { name: "conv2", displayName: "Conv2", type: "conv", totalNeurons: 64, displayNeurons: 24, color: "#a78bfa", description: "64 channels (3×3)" },
  { name: "relu2", displayName: "ReLU2", type: "relu", totalNeurons: 64, displayNeurons: 24, color: "#8b5cf6", description: "64 channels" },
  { name: "pool1", displayName: "Pool1", type: "pool", totalNeurons: 64, displayNeurons: 18, color: "#06b6d4", description: "64 ch → 14×14" },
  { name: "conv3", displayName: "Conv3", type: "conv", totalNeurons: 128, displayNeurons: 30, color: "#0891b2", description: "128 channels (3×3)" },
  { name: "relu3", displayName: "ReLU3", type: "relu", totalNeurons: 128, displayNeurons: 30, color: "#0e7490", description: "128 channels" },
  { name: "pool2", displayName: "Pool2", type: "pool", totalNeurons: 128, displayNeurons: 20, color: "#22c55e", description: "128 ch → 7×7" },
  { name: "dense1", displayName: "Dense", type: "dense", totalNeurons: 256, displayNeurons: 32, color: "#f59e0b", description: "256 neurons" },
  { name: "relu4", displayName: "ReLU4", type: "relu", totalNeurons: 256, displayNeurons: 32, color: "#d97706", description: "256 neurons" },
  { name: "output", displayName: "Output", type: "output", totalNeurons: 47, displayNeurons: 10, color: "#ef4444", description: "Top predictions" },
];

// ---------------------------------------------------------------------------
// Neuron position type
// ---------------------------------------------------------------------------

interface NeuronPos {
  x: number;
  y: number;
  layerIdx: number;
  neuronIdx: number;
  radius: number;
}

// ---------------------------------------------------------------------------
// Connection between neurons in adjacent layers
// ---------------------------------------------------------------------------

interface Conn {
  fromLayer: number;
  fromNeuron: number;
  toLayer: number;
  toNeuron: number;
}

// ---------------------------------------------------------------------------
// Signal particle flowing along a connection
// ---------------------------------------------------------------------------

interface Signal {
  connIdx: number;
  progress: number; // 0→1
  speed: number;
  intensity: number;
  size: number;
}

// ---------------------------------------------------------------------------
// Deterministic pseudo-random for reproducible connection patterns
// ---------------------------------------------------------------------------

function makePRNG(seed: number) {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

// ---------------------------------------------------------------------------
// Generate connections (subset) between adjacent layers
// ---------------------------------------------------------------------------

function generateConnections(): Conn[] {
  const conns: Conn[] = [];
  const rand = makePRNG(42);

  for (let li = 0; li < LAYERS.length - 1; li++) {
    const fromN = LAYERS[li].displayNeurons;
    const toN = LAYERS[li + 1].displayNeurons;
    const perNeuron = Math.min(3, toN);

    for (let fi = 0; fi < fromN; fi++) {
      const targets = new Set<number>();
      // always connect to "corresponding" neuron if possible
      targets.add(Math.floor((fi / fromN) * toN));
      while (targets.size < perNeuron) {
        targets.add(Math.floor(rand() * toN));
      }
      for (const ti of targets) {
        conns.push({ fromLayer: li, fromNeuron: fi, toLayer: li + 1, toNeuron: ti });
      }
    }
  }
  return conns;
}

const CONNECTIONS = generateConnections();

// ---------------------------------------------------------------------------
// Compute neuron positions for given canvas dimensions
// ---------------------------------------------------------------------------

function computePositions(
  w: number,
  h: number,
): NeuronPos[] {
  const positions: NeuronPos[] = [];
  const marginX = 80;
  const marginY = 60;
  const availW = w - marginX * 2;
  const availH = h - marginY * 2;
  const layerSpacing = availW / (LAYERS.length - 1);

  for (let li = 0; li < LAYERS.length; li++) {
    const layer = LAYERS[li];
    const n = layer.displayNeurons;
    const x = marginX + li * layerSpacing;
    const maxSpacing = 24;
    const spacing = Math.min(maxSpacing, availH / (n + 1));
    const totalH = (n - 1) * spacing;
    const startY = h / 2 - totalH / 2;
    const radius = Math.min(7, spacing * 0.32);

    for (let ni = 0; ni < n; ni++) {
      positions.push({
        x,
        y: startY + ni * spacing,
        layerIdx: li,
        neuronIdx: ni,
        radius,
      });
    }
  }
  return positions;
}

// ---------------------------------------------------------------------------
// Extract sampled activation values per layer → array of 0–1 values
// ---------------------------------------------------------------------------

function extractActivations(
  layerActivations: Record<string, number[][][] | number[]>,
  inputTensor: number[][] | null,
): Map<string, number[]> {
  const result = new Map<string, number[]>();

  // --- Collect raw (un-normalized) mean activations per neuron ---
  // We'll normalize globally at the end.

  const rawValues = new Map<string, number[]>();

  // Input: 20 image segments (5 cols × 4 rows grid), mean pixel intensity per patch
  if (inputTensor) {
    const patchCols = 5;
    const patchRows = 4;
    const pw = 28 / patchCols; // 5.6
    const ph = 28 / patchRows; // 7
    const patches: number[] = [];
    for (let pr = 0; pr < patchRows; pr++) {
      for (let pc = 0; pc < patchCols; pc++) {
        const r0 = Math.floor(pr * ph);
        const r1 = Math.floor((pr + 1) * ph);
        const c0 = Math.floor(pc * pw);
        const c1 = Math.floor((pc + 1) * pw);
        let sum = 0, count = 0;
        for (let r = r0; r < r1; r++) {
          for (let c = c0; c < c1; c++) {
            sum += inputTensor[r]?.[c] ?? 0;
            count++;
          }
        }
        patches.push(count > 0 ? sum / count : 0);
      }
    }
    rawValues.set("input", patches);
  }

  // Conv/ReLU/Pool layers (3D): raw mean |activation| per channel
  for (const layer of LAYERS) {
    if (layer.name === "input" || layer.name === "output") continue;
    const acts = layerActivations[layer.name];
    if (!acts) continue;

    if (Array.isArray(acts[0]) && Array.isArray((acts[0] as number[][])[0])) {
      const acts3d = acts as number[][][];
      const means: number[] = [];
      for (const ch of acts3d) {
        let sum = 0, count = 0;
        for (const row of ch) {
          for (const v of row) { sum += Math.abs(v); count++; }
        }
        means.push(sum / count);
      }
      // Sample to displayNeurons
      const sampled: number[] = [];
      for (let i = 0; i < layer.displayNeurons; i++) {
        sampled.push(means[Math.floor(i * means.length / layer.displayNeurons)]);
      }
      rawValues.set(layer.name, sampled);
    } else {
      const acts1d = acts as number[];
      const absVals = acts1d.map(v => Math.abs(v));
      const sampled: number[] = [];
      for (let i = 0; i < layer.displayNeurons; i++) {
        sampled.push(absVals[Math.floor(i * absVals.length / layer.displayNeurons)]);
      }
      rawValues.set(layer.name, sampled);
    }
  }

  // Output: top 10 most activated classes (softmax probabilities)
  const outputActs = layerActivations["output"];
  if (outputActs && !Array.isArray(outputActs[0])) {
    const vals = outputActs as number[];
    const valid: { val: number; idx: number }[] = [];
    for (let i = 0; i < vals.length; i++) {
      if (!BYMERGE_MERGED_INDICES.has(i)) valid.push({ val: vals[i], idx: i });
    }
    valid.sort((a, b) => b.val - a.val);
    const top = valid.slice(0, 10);
    rawValues.set("output", top.map(d => d.val));
  }

  // --- Global normalization: find max across ALL layers, normalize to 0–1 ---
  let globalMax = 0;
  for (const [, vals] of rawValues) {
    for (const v of vals) {
      if (v > globalMax) globalMax = v;
    }
  }
  globalMax = Math.max(globalMax, 0.001);

  for (const [name, vals] of rawValues) {
    result.set(name, vals.map(v => v / globalMax));
  }

  return result;
}

// ---------------------------------------------------------------------------
// Get output class labels for the displayed output neurons
// ---------------------------------------------------------------------------

function getOutputLabels(
  layerActivations: Record<string, number[][][] | number[]>,
): string[] {
  const outputActs = layerActivations["output"];
  if (!outputActs || Array.isArray(outputActs[0])) return [];
  const vals = outputActs as number[];
  const valid: { val: number; idx: number }[] = [];
  for (let i = 0; i < vals.length; i++) {
    if (!BYMERGE_MERGED_INDICES.has(i)) valid.push({ val: vals[i], idx: i });
  }
  valid.sort((a, b) => b.val - a.val);
  return valid.slice(0, 10).map(d => EMNIST_CLASSES[d.idx]);
}

// ---------------------------------------------------------------------------
// Bezier curve helpers for smooth connections
// ---------------------------------------------------------------------------

function bezierPoint(
  x0: number, y0: number,
  cx1: number, cy1: number,
  cx2: number, cy2: number,
  x1: number, y1: number,
  t: number,
): [number, number] {
  const u = 1 - t;
  const x = u * u * u * x0 + 3 * u * u * t * cx1 + 3 * u * t * t * cx2 + t * t * t * x1;
  const y = u * u * u * y0 + 3 * u * u * t * cy1 + 3 * u * t * t * cy2 + t * t * t * y1;
  return [x, y];
}

// ---------------------------------------------------------------------------
// Hovered neuron info
// ---------------------------------------------------------------------------

interface HoveredNeuron {
  layerIdx: number;
  neuronIdx: number;
  screenX: number;
  screenY: number;
}

// ---------------------------------------------------------------------------
// Map display neuron index → actual data index (channel or neuron)
// ---------------------------------------------------------------------------

function displayToActualIndex(layerIdx: number, displayIdx: number): number {
  const layer = LAYERS[layerIdx];
  if (layer.displayNeurons >= layer.totalNeurons) return displayIdx;
  return Math.floor(displayIdx * layer.totalNeurons / layer.displayNeurons);
}

// ---------------------------------------------------------------------------
// Main neuron network canvas
// ---------------------------------------------------------------------------

function NeuronNetworkCanvas({
  width,
  height,
  activationMap,
  outputLabels,
  hoveredLayer,
  hoveredNeuron,
  onHoverLayer,
  onHoverNeuron,
  onClickLayer,
  waveProgress,
}: {
  width: number;
  height: number;
  activationMap: Map<string, number[]>;
  outputLabels: string[];
  hoveredLayer: number | null;
  hoveredNeuron: HoveredNeuron | null;
  onHoverLayer: (li: number | null) => void;
  onHoverNeuron: (n: HoveredNeuron | null) => void;
  onClickLayer: (li: number, neuronIdx: number | null) => void;
  waveProgress: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const signalsRef = useRef<Signal[]>([]);
  const positionsRef = useRef<NeuronPos[]>([]);
  const hasData = activationMap.size > 0;

  // Compute neuron positions
  const positions = useMemo(() => computePositions(width, height), [width, height]);
  positionsRef.current = positions;

  // Initialize signal particles
  useEffect(() => {
    const signals: Signal[] = [];
    const rand = makePRNG(123);
    for (let ci = 0; ci < CONNECTIONS.length; ci++) {
      // One signal per connection, staggered start
      signals.push({
        connIdx: ci,
        progress: rand(),
        speed: 0.004 + rand() * 0.006,
        intensity: 0.5 + rand() * 0.5,
        size: 1.5 + rand() * 1.5,
      });
    }
    signalsRef.current = signals;
  }, []);

  // Build neuron index lookup: [layerIdx][neuronIdx] → index in positions array
  const neuronLookup = useMemo(() => {
    const lookup: number[][] = [];
    for (let li = 0; li < LAYERS.length; li++) {
      lookup[li] = [];
    }
    for (let i = 0; i < positions.length; i++) {
      const p = positions[i];
      lookup[p.layerIdx][p.neuronIdx] = i;
    }
    return lookup;
  }, [positions]);

  // Animation loop
  useAnimationFrame((_dt, elapsed) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);

    // --- Draw connections ---
    for (let ci = 0; ci < CONNECTIONS.length; ci++) {
      const conn = CONNECTIONS[ci];
      const fromIdx = neuronLookup[conn.fromLayer]?.[conn.fromNeuron];
      const toIdx = neuronLookup[conn.toLayer]?.[conn.toNeuron];
      if (fromIdx === undefined || toIdx === undefined) continue;
      const from = positions[fromIdx];
      const to = positions[toIdx];

      // Get activation intensity for connection opacity
      const fromLayerName = LAYERS[conn.fromLayer].name;
      const toLayerName = LAYERS[conn.toLayer].name;
      const fromActs = activationMap.get(fromLayerName);
      const toActs = activationMap.get(toLayerName);
      const fromVal = fromActs?.[conn.fromNeuron] ?? 0;
      const toVal = toActs?.[conn.toNeuron] ?? 0;
      const connStrength = (fromVal + toVal) / 2;

      // Apply wave progress — connections only visible once both layers are "lit"
      const fromWave = Math.max(0, Math.min(1, waveProgress - conn.fromLayer));
      const toWave = Math.max(0, Math.min(1, waveProgress - conn.toLayer));
      const waveAlpha = Math.min(fromWave, toWave);

      // Determine opacity
      let baseAlpha: number;
      if (hasData) {
        baseAlpha = 0.03 + connStrength * 0.15;
      } else {
        baseAlpha = 0.025;
      }

      // Highlight on hover
      if (hoveredLayer !== null && (conn.fromLayer === hoveredLayer || conn.toLayer === hoveredLayer)) {
        baseAlpha = hasData ? 0.1 + connStrength * 0.35 : 0.08;
      }

      const alpha = hasData ? baseAlpha * waveAlpha : baseAlpha;
      if (alpha < 0.005) continue;

      // Bezier control points
      const dx = (to.x - from.x) * 0.4;
      const cx1 = from.x + dx;
      const cy1 = from.y;
      const cx2 = to.x - dx;
      const cy2 = to.y;

      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.bezierCurveTo(cx1, cy1, cx2, cy2, to.x, to.y);
      ctx.strokeStyle = `rgba(140, 160, 200, ${alpha})`;
      ctx.lineWidth = 0.6;
      ctx.stroke();
    }

    // --- Draw signal particles ---
    if (hasData) {
      for (const sig of signalsRef.current) {
        sig.progress += sig.speed;
        if (sig.progress > 1) {
          sig.progress -= 1;
          sig.intensity = 0.4 + Math.random() * 0.6;
        }

        const conn = CONNECTIONS[sig.connIdx];
        if (!conn) continue;

        // Wave check — only show signals in lit layers
        const layerWave = Math.max(0, Math.min(1, waveProgress - conn.fromLayer));
        if (layerWave < 0.5) continue;

        const fromIdx = neuronLookup[conn.fromLayer]?.[conn.fromNeuron];
        const toIdx = neuronLookup[conn.toLayer]?.[conn.toNeuron];
        if (fromIdx === undefined || toIdx === undefined) continue;
        const from = positions[fromIdx];
        const to = positions[toIdx];

        // Get connection activation
        const fromActs = activationMap.get(LAYERS[conn.fromLayer].name);
        const val = fromActs?.[conn.fromNeuron] ?? 0;
        if (val < 0.05) continue;

        const dx = (to.x - from.x) * 0.4;
        const [px, py] = bezierPoint(
          from.x, from.y,
          from.x + dx, from.y,
          to.x - dx, to.y,
          to.x, to.y,
          sig.progress,
        );

        const color = LAYERS[conn.toLayer].color;
        const alpha = val * sig.intensity * 0.8;

        // Glow
        ctx.beginPath();
        ctx.arc(px, py, sig.size * 3, 0, Math.PI * 2);
        ctx.fillStyle = hexToRGBA(color, alpha * 0.2);
        ctx.fill();

        // Core
        ctx.beginPath();
        ctx.arc(px, py, sig.size, 0, Math.PI * 2);
        ctx.fillStyle = hexToRGBA(color, alpha);
        ctx.fill();
      }
    }

    // --- Draw neurons ---
    const time = elapsed / 1000;
    for (const pos of positions) {
      const layer = LAYERS[pos.layerIdx];
      const acts = activationMap.get(layer.name);
      const activation = acts?.[pos.neuronIdx] ?? 0;

      // Wave progress — neurons light up progressively
      const layerWave = hasData ? Math.max(0, Math.min(1, waveProgress - pos.layerIdx)) : 0;

      const isHovered = hoveredLayer === pos.layerIdx;
      const isThisNeuronHovered = hoveredNeuron?.layerIdx === pos.layerIdx && hoveredNeuron?.neuronIdx === pos.neuronIdx;
      const r = pos.radius;
      const effectiveAct = activation * layerWave;

      // Pulse for highly active neurons
      const pulse = effectiveAct > 0.7
        ? 1 + Math.sin(time * 3 + pos.neuronIdx * 0.5) * 0.15
        : 1;

      // Outer glow for active neurons
      if (effectiveAct > 0.1) {
        const glowR = r * 2.5 * pulse;
        const gradient = ctx.createRadialGradient(pos.x, pos.y, r * 0.5, pos.x, pos.y, glowR);
        gradient.addColorStop(0, hexToRGBA(layer.color, effectiveAct * 0.4));
        gradient.addColorStop(1, hexToRGBA(layer.color, 0));
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, glowR, 0, Math.PI * 2);
        ctx.fillStyle = gradient;
        ctx.fill();
      }

      // Neuron body
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, r * pulse, 0, Math.PI * 2);

      if (effectiveAct > 0.01) {
        // Filled based on activation
        const brightness = 0.15 + effectiveAct * 0.85;
        ctx.fillStyle = hexToRGBA(layer.color, brightness);
        ctx.fill();
      } else {
        // Empty/dim circle
        ctx.fillStyle = isHovered ? "rgba(255,255,255,0.08)" : "rgba(255,255,255,0.03)";
        ctx.fill();
      }

      // Border — bright ring on individually hovered neuron
      if (isThisNeuronHovered) {
        ctx.strokeStyle = hexToRGBA(layer.color, 1);
        ctx.lineWidth = 2.5;
      } else {
        ctx.strokeStyle = isHovered
          ? hexToRGBA(layer.color, 0.6)
          : hexToRGBA(layer.color, effectiveAct > 0.01 ? 0.3 + effectiveAct * 0.5 : 0.12);
        ctx.lineWidth = isHovered ? 1.5 : 0.8;
      }
      ctx.stroke();

      // Output labels
      if (layer.type === "output" && outputLabels[pos.neuronIdx]) {
        ctx.fillStyle = effectiveAct > 0.3
          ? `rgba(255,255,255,${0.4 + effectiveAct * 0.6})`
          : "rgba(255,255,255,0.25)";
        ctx.font = `${effectiveAct > 0.5 ? "bold " : ""}10px system-ui, sans-serif`;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(outputLabels[pos.neuronIdx], pos.x + r + 6, pos.y);
      }
    }

    // --- Layer labels ---
    const drawnLayers = new Set<number>();
    for (const pos of positions) {
      if (drawnLayers.has(pos.layerIdx)) continue;
      drawnLayers.add(pos.layerIdx);

      const layer = LAYERS[pos.layerIdx];
      // Find bottom of this layer
      let maxY = 0;
      for (const p of positions) {
        if (p.layerIdx === pos.layerIdx && p.y > maxY) maxY = p.y;
      }

      const isHovered = hoveredLayer === pos.layerIdx;

      // Layer name
      ctx.fillStyle = isHovered ? layer.color : "rgba(255,255,255,0.4)";
      ctx.font = `${isHovered ? "bold " : ""}10px system-ui, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(layer.displayName, pos.x, maxY + 16);

      // Neuron/channel count label
      const unit = (layer.type === "conv" || layer.type === "relu" || layer.type === "pool")
        && layer.name !== "relu4" ? "ch" : "";
      ctx.fillStyle = "rgba(255,255,255,0.15)";
      ctx.font = "8px system-ui, sans-serif";
      ctx.fillText(
        layer.totalNeurons > layer.displayNeurons
          ? `${layer.totalNeurons}${unit ? " " + unit : " neurons"}`
          : `${layer.totalNeurons}${unit ? " " + unit : " neurons"}`,
        pos.x,
        maxY + 28,
      );

      // Ellipsis dots if we're showing a subset
      if (layer.totalNeurons > layer.displayNeurons) {
        ctx.fillStyle = "rgba(255,255,255,0.12)";
        for (let d = 0; d < 3; d++) {
          ctx.beginPath();
          ctx.arc(pos.x - 4 + d * 4, maxY + 8, 1, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
  }, true);

  // Mouse handling — find closest individual neuron
  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const scaleX = width / rect.width;
      const scaleY = height / rect.height;
      const mx = (e.clientX - rect.left) * scaleX;
      const my = (e.clientY - rect.top) * scaleY;

      // Find closest neuron within snap distance
      let closestNeuron: HoveredNeuron | null = null;
      let closestLayer: number | null = null;
      let minNeuronDist = 18; // pixel snap radius
      let minLayerDist = 40;

      for (const pos of positionsRef.current) {
        const dx = mx - pos.x;
        const dy = my - pos.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        // Layer-level (column) detection
        if (Math.abs(dx) < minLayerDist) {
          minLayerDist = Math.abs(dx);
          closestLayer = pos.layerIdx;
        }

        // Individual neuron detection
        if (dist < minNeuronDist) {
          minNeuronDist = dist;
          closestNeuron = {
            layerIdx: pos.layerIdx,
            neuronIdx: pos.neuronIdx,
            // Convert back to screen coords for tooltip positioning
            screenX: pos.x / scaleX + rect.left,
            screenY: pos.y / scaleY + rect.top,
          };
        }
      }

      onHoverLayer(closestLayer);
      onHoverNeuron(closestNeuron);
    },
    [width, height, onHoverLayer, onHoverNeuron],
  );

  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const scaleX = width / rect.width;
      const scaleY = height / rect.height;
      const mx = (e.clientX - rect.left) * scaleX;
      const my = (e.clientY - rect.top) * scaleY;

      // Find closest neuron
      let closestLayer: number | null = null;
      let closestNeuronIdx: number | null = null;
      let minNeuronDist = 18;
      let minLayerDist = 40;

      for (const pos of positionsRef.current) {
        const dx = mx - pos.x;
        const dy = my - pos.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (Math.abs(dx) < minLayerDist) {
          minLayerDist = Math.abs(dx);
          closestLayer = pos.layerIdx;
        }
        if (dist < minNeuronDist) {
          minNeuronDist = dist;
          closestLayer = pos.layerIdx;
          closestNeuronIdx = pos.neuronIdx;
        }
      }
      if (closestLayer !== null) onClickLayer(closestLayer, closestNeuronIdx);
    },
    [width, height, onClickLayer],
  );

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      onMouseMove={handleMouseMove}
      onMouseLeave={() => { onHoverLayer(null); onHoverNeuron(null); }}
      onClick={handleClick}
      style={{
        width: "100%",
        height: "100%",
        cursor: hoveredLayer !== null ? "pointer" : "default",
      }}
    />
  );
}

// ---------------------------------------------------------------------------
// Hex color → rgba string helper
// ---------------------------------------------------------------------------

function hexToRGBA(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// ---------------------------------------------------------------------------
// Viridis-like color scale for inspector
// ---------------------------------------------------------------------------

function viridis(t: number): [number, number, number] {
  t = Math.max(0, Math.min(1, t));
  const r = Math.round(255 * Math.max(0, Math.min(1, -0.33 + 2.2 * t * t)));
  const g = Math.round(255 * Math.max(0, Math.min(1, -0.15 + 1.5 * t - 0.5 * t * t)));
  const b = Math.round(255 * Math.max(0, Math.min(1, 0.52 + 0.6 * t - 1.2 * t * t)));
  return [r, g, b];
}

// ---------------------------------------------------------------------------
// Inspector panel — detailed layer view
// ---------------------------------------------------------------------------

function InspectorPanel({
  layer,
  activations,
  inputTensor,
  prediction,
  topPrediction,
  initialChannel,
  onClose,
}: {
  layer: NeuronLayerDef;
  activations: number[][][] | number[] | null;
  inputTensor: number[][] | null;
  prediction: number[] | null;
  topPrediction: { classIndex: number; confidence: number } | null;
  initialChannel: number;
  onClose: () => void;
}) {
  const [selectedChannel, setSelectedChannel] = useState(initialChannel);
  const mainCanvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = mainCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const w = canvas.width;
    const h = canvas.height;

    ctx.fillStyle = "#111118";
    ctx.fillRect(0, 0, w, h);

    if (layer.type === "input" && inputTensor) {
      const cellW = w / 28;
      const cellH = h / 28;
      for (let r = 0; r < 28; r++) {
        for (let c = 0; c < 28; c++) {
          const gray = Math.round(inputTensor[r][c] * 255);
          ctx.fillStyle = `rgb(${gray},${gray},${gray})`;
          ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5);
        }
      }
    } else if (layer.type === "output" && prediction) {
      const sorted = prediction
        .map((v, i) => ({ v, i }))
        .filter(d => !BYMERGE_MERGED_INDICES.has(d.i))
        .sort((a, b) => b.v - a.v);
      const barH = h / Math.min(sorted.length, 20);
      ctx.font = "11px system-ui, sans-serif";
      for (let j = 0; j < Math.min(sorted.length, 20); j++) {
        const d = sorted[j];
        const barW = (d.v / Math.max(sorted[0].v, 0.001)) * (w - 60);
        const [cr, cg, cb] = viridis(d.v / Math.max(sorted[0].v, 0.001));
        ctx.fillStyle = `rgb(${cr},${cg},${cb})`;
        ctx.fillRect(40, j * barH + 2, barW, barH - 4);
        ctx.fillStyle = "#e8e8ed";
        ctx.textAlign = "right";
        ctx.fillText(EMNIST_CLASSES[d.i], 35, j * barH + barH / 2 + 4);
        ctx.textAlign = "left";
        ctx.fillText(`${(d.v * 100).toFixed(1)}%`, 40 + barW + 4, j * barH + barH / 2 + 4);
      }
    } else if (activations && Array.isArray(activations[0])) {
      const acts = activations as number[][][];
      if (selectedChannel < acts.length) {
        const ch = acts[selectedChannel];
        const rows = ch.length;
        const cols = ch[0].length;
        let maxVal = 0;
        for (let r = 0; r < rows; r++) {
          for (let c = 0; c < cols; c++) {
            if (Math.abs(ch[r][c]) > maxVal) maxVal = Math.abs(ch[r][c]);
          }
        }
        maxVal = Math.max(maxVal, 0.001);
        const cellW = w / cols;
        const cellH = h / rows;
        for (let r = 0; r < rows; r++) {
          for (let c = 0; c < cols; c++) {
            const norm = Math.abs(ch[r][c]) / maxVal;
            const [cr, cg, cb] = viridis(norm);
            ctx.fillStyle = `rgb(${cr},${cg},${cb})`;
            ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5);
          }
        }
      }
    } else if (activations && !Array.isArray(activations[0])) {
      const vals = activations as number[];
      const n = vals.length;
      const cols = Math.ceil(Math.sqrt(n));
      const rows = Math.ceil(n / cols);
      const cellW = w / cols;
      const cellH = h / rows;
      const maxVal = Math.max(...vals.map(Math.abs), 0.001);
      for (let i = 0; i < n; i++) {
        const r = Math.floor(i / cols);
        const c = i % cols;
        const norm = Math.abs(vals[i]) / maxVal;
        const [cr, cg, cb] = viridis(norm);
        ctx.fillStyle = `rgb(${cr},${cg},${cb})`;
        ctx.fillRect(c * cellW + 0.5, r * cellH + 0.5, cellW - 1, cellH - 1);
      }
    } else {
      ctx.fillStyle = "rgba(255,255,255,0.15)";
      ctx.font = "14px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Draw a character to see activations", w / 2, h / 2);
    }
  }, [layer, activations, inputTensor, selectedChannel, prediction]);

  const channelCount = activations && Array.isArray(activations[0]) ? (activations as number[][][]).length : 0;

  // Stats
  const stats = useMemo(() => {
    if (!activations) return null;
    if (Array.isArray(activations[0])) {
      const acts = activations as number[][][];
      let min = Infinity, max = -Infinity, sum = 0, count = 0, activeCount = 0;
      for (const ch of acts) {
        for (const row of ch) {
          for (const v of row) {
            if (v < min) min = v;
            if (v > max) max = v;
            sum += v;
            count++;
            if (v > 0) activeCount++;
          }
        }
      }
      return { min, max, mean: sum / count, activePercent: (activeCount / count) * 100, count };
    } else {
      const vals = activations as number[];
      let min = Infinity, max = -Infinity, sum = 0, activeCount = 0;
      for (const v of vals) {
        if (v < min) min = v;
        if (v > max) max = v;
        sum += v;
        if (v > 0) activeCount++;
      }
      return { min, max, mean: sum / vals.length, activePercent: (activeCount / vals.length) * 100, count: vals.length };
    }
  }, [activations]);

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 100,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "rgba(0,0,0,0.75)",
        backdropFilter: "blur(6px)",
      }}
      onClick={e => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div
        style={{
          background: "#13131a",
          border: `1px solid ${layer.color}40`,
          borderRadius: 12,
          padding: 24,
          maxWidth: 900,
          width: "90vw",
          maxHeight: "90vh",
          overflowY: "auto",
          boxShadow: `0 0 60px ${layer.color}15`,
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16 }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
              <div style={{ width: 12, height: 12, borderRadius: "50%", background: layer.color }} />
              <h2 style={{ fontSize: 20, fontWeight: 600, color: "#e8e8ed", margin: 0 }}>
                {layer.displayName}
              </h2>
              <span style={{ fontSize: 13, color: "rgba(255,255,255,0.35)", fontFamily: "monospace" }}>
                {layer.totalNeurons.toLocaleString()} {(layer.type === "conv" || (layer.type === "relu" && layer.name !== "relu4") || layer.type === "pool") ? "channels" : (layer.type === "input" ? "pixels" : "neurons")}
              </span>
            </div>
            <p style={{ fontSize: 13, color: "rgba(255,255,255,0.45)", margin: 0 }}>{layer.description}</p>
          </div>
          <button
            onClick={onClose}
            style={{
              background: "rgba(255,255,255,0.05)",
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: 6,
              color: "rgba(255,255,255,0.6)",
              padding: "4px 12px",
              cursor: "pointer",
              fontSize: 13,
            }}
          >
            Close
          </button>
        </div>

        {stats && (
          <div style={{
            display: "flex",
            gap: 20,
            marginBottom: 16,
            padding: "8px 12px",
            background: "rgba(255,255,255,0.03)",
            borderRadius: 8,
            fontSize: 12,
            fontFamily: "monospace",
            color: "rgba(255,255,255,0.5)",
          }}>
            <span>Min: <span style={{ color: "#e8e8ed" }}>{stats.min.toFixed(3)}</span></span>
            <span>Max: <span style={{ color: "#e8e8ed" }}>{stats.max.toFixed(3)}</span></span>
            <span>Mean: <span style={{ color: "#e8e8ed" }}>{stats.mean.toFixed(3)}</span></span>
            <span>Active: <span style={{ color: "#22c55e" }}>{stats.activePercent.toFixed(1)}%</span></span>
          </div>
        )}

        {layer.type === "output" && topPrediction && (
          <div style={{
            display: "flex",
            alignItems: "center",
            gap: 16,
            marginBottom: 16,
            padding: "12px 16px",
            background: `${layer.color}15`,
            borderRadius: 8,
            border: `1px solid ${layer.color}30`,
          }}>
            <span style={{ fontSize: 36, fontWeight: 700, color: layer.color }}>
              {EMNIST_CLASSES[topPrediction.classIndex]}
            </span>
            <div>
              <div style={{ fontSize: 14, color: "#e8e8ed" }}>
                Predicted: <strong>{EMNIST_CLASSES[topPrediction.classIndex]}</strong>
              </div>
              <div style={{ fontSize: 13, color: "rgba(255,255,255,0.5)" }}>
                Confidence: {(topPrediction.confidence * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        )}

        <div style={{ display: "flex", gap: 16 }}>
          <canvas
            ref={mainCanvasRef}
            width={channelCount > 0 ? 350 : 500}
            height={channelCount > 0 ? 350 : (layer.type === "output" ? 400 : 300)}
            style={{
              width: channelCount > 0 ? 350 : "100%",
              height: channelCount > 0 ? 350 : (layer.type === "output" ? 400 : 300),
              borderRadius: 8,
              imageRendering: layer.type === "input" ? "pixelated" : "auto",
            }}
          />

          {channelCount > 0 && (
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: 12, color: "rgba(255,255,255,0.4)", marginBottom: 8 }}>
                {channelCount} channels — click to inspect
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 3, maxHeight: 340, overflowY: "auto" }}>
                {Array.from({ length: channelCount }, (_, i) => (
                  <ChannelThumb
                    key={i}
                    chIdx={i}
                    activations={activations as number[][][]}
                    selected={i === selectedChannel}
                    color={layer.color}
                    onClick={() => setSelectedChannel(i)}
                  />
                ))}
              </div>
              <div style={{ fontSize: 11, color: "rgba(255,255,255,0.3)", marginTop: 6, fontFamily: "monospace" }}>
                Channel {selectedChannel}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Channel thumbnail for inspector
// ---------------------------------------------------------------------------

function ChannelThumb({
  chIdx,
  activations,
  selected,
  color,
  onClick,
}: {
  chIdx: number;
  activations: number[][][];
  selected: boolean;
  color: string;
  onClick: () => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const el = canvasRef.current;
    if (!el || chIdx >= activations.length) return;
    const ctx = el.getContext("2d");
    if (!ctx) return;
    const ch = activations[chIdx];
    const rows = ch.length;
    const cols = ch[0].length;
    let maxVal = 0;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if (Math.abs(ch[r][c]) > maxVal) maxVal = Math.abs(ch[r][c]);
      }
    }
    maxVal = Math.max(maxVal, 0.001);
    const cellW = 40 / cols;
    const cellH = 40 / rows;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const norm = Math.abs(ch[r][c]) / maxVal;
        const [cr, cg, cb] = viridis(norm);
        ctx.fillStyle = `rgb(${cr},${cg},${cb})`;
        ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5);
      }
    }
  }, [chIdx, activations]);

  return (
    <canvas
      ref={canvasRef}
      width={40}
      height={40}
      onClick={onClick}
      style={{
        width: 40,
        height: 40,
        borderRadius: 3,
        cursor: "pointer",
        border: selected ? `2px solid ${color}` : "2px solid transparent",
        imageRendering: "pixelated",
      }}
    />
  );
}

// ---------------------------------------------------------------------------
// Prediction summary sidebar
// ---------------------------------------------------------------------------

function PredictionSummary({
  prediction,
  topPrediction,
}: {
  prediction: number[] | null;
  topPrediction: { classIndex: number; confidence: number } | null;
}) {
  if (!prediction || !topPrediction) return null;

  const top5 = useMemo(() => {
    return prediction
      .map((v, i) => ({ v, i }))
      .filter(d => !BYMERGE_MERGED_INDICES.has(d.i))
      .sort((a, b) => b.v - a.v)
      .slice(0, 5);
  }, [prediction]);

  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      gap: 8,
      padding: "12px 16px",
      background: "#13131a",
      borderRadius: 10,
      border: "1px solid #2a2a3a",
      minWidth: 150,
    }}>
      <span style={{ fontSize: 42, fontWeight: 700, color: "#6366f1", lineHeight: 1 }}>
        {EMNIST_CLASSES[topPrediction.classIndex]}
      </span>
      <span style={{ fontSize: 12, color: "rgba(255,255,255,0.5)" }}>
        {(topPrediction.confidence * 100).toFixed(1)}%
      </span>
      <div style={{ width: "100%", borderTop: "1px solid #2a2a3a", paddingTop: 8, marginTop: 4 }}>
        {top5.map(d => (
          <div key={d.i} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
            <span style={{ fontSize: 11, width: 16, textAlign: "right", color: "#e8e8ed", fontWeight: d.i === topPrediction.classIndex ? 600 : 400 }}>
              {EMNIST_CLASSES[d.i]}
            </span>
            <div style={{ flex: 1, height: 6, borderRadius: 3, background: "rgba(255,255,255,0.05)", overflow: "hidden" }}>
              <div style={{
                width: `${(d.v / top5[0].v) * 100}%`,
                height: "100%",
                borderRadius: 3,
                background: d.i === topPrediction.classIndex ? "#6366f1" : "#4a4a5a",
                transition: "width 0.3s",
              }} />
            </div>
            <span style={{ fontSize: 10, color: "rgba(255,255,255,0.4)", width: 36, textAlign: "right", fontFamily: "monospace" }}>
              {(d.v * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Neuron heatmap tooltip — shows the actual 2D activation map for a neuron
// ---------------------------------------------------------------------------

function NeuronHeatmapTooltip({
  neuron,
  layerActivations,
  inputTensor,
  outputLabels,
  containerRect,
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

  // Determine canvas size and content
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
    const w = canvas.width;
    const h = canvas.height;

    ctx.fillStyle = "#111118";
    ctx.fillRect(0, 0, w, h);

    if (isInput && inputTensor) {
      // Draw full input image with the hovered patch highlighted
      const cellW = w / 28;
      const cellH = h / 28;
      for (let r = 0; r < 28; r++) {
        for (let c = 0; c < 28; c++) {
          const gray = Math.round(inputTensor[r][c] * 255);
          ctx.fillStyle = `rgb(${gray},${gray},${gray})`;
          ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5);
        }
      }
      // Highlight the patch this neuron represents (5 cols × 4 rows grid)
      const patchCols = 5;
      const patchRows = 4;
      const pc = neuron.neuronIdx % patchCols;
      const pr = Math.floor(neuron.neuronIdx / patchCols);
      const px = Math.floor(pc * 28 / patchCols) * cellW;
      const py = Math.floor(pr * 28 / patchRows) * cellH;
      const pw = Math.floor(28 / patchCols) * cellW;
      const ph = Math.floor(28 / patchRows) * cellH;
      ctx.strokeStyle = "#6366f1";
      ctx.lineWidth = 2;
      ctx.strokeRect(px, py, pw, ph);
      // Dim outside the patch
      ctx.fillStyle = "rgba(0,0,0,0.5)";
      ctx.fillRect(0, 0, w, py);
      ctx.fillRect(0, py + ph, w, h - py - ph);
      ctx.fillRect(0, py, px, ph);
      ctx.fillRect(px + pw, py, w - px - pw, ph);
    } else if (isConv3D && layer.name !== "relu4") {
      // Show the 2D spatial activation map for this channel
      const acts = layerActivations[layer.name];
      if (acts && Array.isArray(acts[0]) && Array.isArray((acts[0] as number[][])[0])) {
        const acts3d = acts as number[][][];
        if (actualIdx < acts3d.length) {
          const ch = acts3d[actualIdx];
          const rows = ch.length;
          const cols = ch[0].length;
          let maxVal = 0;
          for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
              if (Math.abs(ch[r][c]) > maxVal) maxVal = Math.abs(ch[r][c]);
            }
          }
          maxVal = Math.max(maxVal, 0.001);
          const cellW = w / cols;
          const cellH = h / rows;
          for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
              const norm = Math.abs(ch[r][c]) / maxVal;
              const [cr, cg, cb] = viridis(norm);
              ctx.fillStyle = `rgb(${cr},${cg},${cb})`;
              ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5);
            }
          }
        }
      }
    } else if (isDense) {
      // Show a bar for this neuron's value
      const acts = layerActivations[layer.name];
      if (acts && !Array.isArray(acts[0])) {
        const vals = acts as number[];
        if (actualIdx < vals.length) {
          const v = Math.abs(vals[actualIdx]);
          let maxVal = 0;
          for (const val of vals) { if (Math.abs(val) > maxVal) maxVal = Math.abs(val); }
          maxVal = Math.max(maxVal, 0.001);
          const norm = v / maxVal;
          const barW = norm * (w - 8);
          const [cr, cg, cb] = viridis(norm);
          ctx.fillStyle = `rgb(${cr},${cg},${cb})`;
          ctx.fillRect(4, h / 2 - 10, barW, 20);
          ctx.strokeStyle = "rgba(255,255,255,0.1)";
          ctx.strokeRect(4, h / 2 - 10, w - 8, 20);
        }
      }
    } else if (isOutput) {
      // Show probability bar for this class
      const acts = layerActivations["output"];
      if (acts && !Array.isArray(acts[0])) {
        const vals = acts as number[];
        // Find the actual class index for this output neuron
        const valid: { val: number; idx: number }[] = [];
        for (let i = 0; i < vals.length; i++) {
          if (!BYMERGE_MERGED_INDICES.has(i)) valid.push({ val: vals[i], idx: i });
        }
        valid.sort((a, b) => b.val - a.val);
        if (neuron.neuronIdx < valid.length) {
          const d = valid[neuron.neuronIdx];
          const barW = d.val * (w - 8);
          const [cr, cg, cb] = viridis(d.val);
          ctx.fillStyle = `rgb(${cr},${cg},${cb})`;
          ctx.fillRect(4, h / 2 - 10, barW, 20);
          ctx.strokeStyle = "rgba(255,255,255,0.1)";
          ctx.strokeRect(4, h / 2 - 10, w - 8, 20);
        }
      }
    }
  }, [neuron, layerActivations, inputTensor, layer, actualIdx, isConv3D, isDense, isInput, isOutput, canvasSize]);

  // Compute label text
  let label = "";
  if (isInput) {
    const pc = neuron.neuronIdx % 5;
    const pr = Math.floor(neuron.neuronIdx / 5);
    label = `Patch [${pr},${pc}]`;
  } else if (isConv3D && layer.name !== "relu4") {
    label = `Channel ${actualIdx}`;
  } else if (isDense) {
    label = `Neuron ${actualIdx}`;
  } else if (isOutput) {
    label = outputLabels[neuron.neuronIdx]
      ? `Class "${outputLabels[neuron.neuronIdx]}"`
      : `Output ${neuron.neuronIdx}`;
  }

  // Compute value text
  let valueText = "";
  if (isInput && inputTensor) {
    // Mean of the patch
    const patchCols = 5, patchRows = 4;
    const pc = neuron.neuronIdx % patchCols;
    const pr = Math.floor(neuron.neuronIdx / patchCols);
    const r0 = Math.floor(pr * 28 / patchRows);
    const r1 = Math.floor((pr + 1) * 28 / patchRows);
    const c0 = Math.floor(pc * 28 / patchCols);
    const c1 = Math.floor((pc + 1) * 28 / patchCols);
    let sum = 0, count = 0;
    for (let r = r0; r < r1; r++) {
      for (let c = c0; c < c1; c++) {
        sum += inputTensor[r]?.[c] ?? 0;
        count++;
      }
    }
    valueText = `mean: ${(sum / count).toFixed(3)}`;
  } else if ((isConv3D && layer.name !== "relu4") || isDense) {
    const acts = layerActivations[layer.name];
    if (acts) {
      if (Array.isArray(acts[0])) {
        const acts3d = acts as number[][][];
        if (actualIdx < acts3d.length) {
          const ch = acts3d[actualIdx];
          let sum = 0, count = 0;
          for (const row of ch) { for (const v of row) { sum += Math.abs(v); count++; } }
          valueText = `mean |act|: ${(sum / count).toFixed(4)}`;
        }
      } else {
        const vals = acts as number[];
        if (actualIdx < vals.length) {
          valueText = `value: ${vals[actualIdx].toFixed(4)}`;
        }
      }
    }
  } else if (isOutput) {
    const acts = layerActivations["output"];
    if (acts && !Array.isArray(acts[0])) {
      const vals = acts as number[];
      const valid: { val: number; idx: number }[] = [];
      for (let i = 0; i < vals.length; i++) {
        if (!BYMERGE_MERGED_INDICES.has(i)) valid.push({ val: vals[i], idx: i });
      }
      valid.sort((a, b) => b.val - a.val);
      if (neuron.neuronIdx < valid.length) {
        valueText = `prob: ${(valid[neuron.neuronIdx].val * 100).toFixed(2)}%`;
      }
    }
  }

  // Position tooltip relative to container
  if (!containerRect) return null;
  const tooltipX = neuron.screenX - containerRect.left + 20;
  const tooltipY = neuron.screenY - containerRect.top - 30;

  return (
    <div
      style={{
        position: "absolute",
        left: tooltipX,
        top: tooltipY,
        transform: "translateY(-100%)",
        background: "#15151f",
        border: `1px solid ${layer.color}50`,
        borderRadius: 8,
        padding: 8,
        pointerEvents: "none",
        zIndex: 60,
        boxShadow: `0 4px 20px rgba(0,0,0,0.6), 0 0 15px ${layer.color}15`,
        minWidth: 120,
      }}
    >
      <div style={{ fontSize: 11, fontWeight: 600, color: layer.color, marginBottom: 4 }}>
        {layer.displayName} — {label}
      </div>
      <canvas
        ref={canvasRef}
        width={canvasSize}
        height={isDense || isOutput ? 40 : canvasSize}
        style={{
          width: canvasSize,
          height: isDense || isOutput ? 40 : canvasSize,
          borderRadius: 4,
          imageRendering: (isInput || isConv3D) ? "pixelated" : "auto",
          display: "block",
        }}
      />
      {valueText && (
        <div style={{ fontSize: 10, color: "rgba(255,255,255,0.5)", marginTop: 4, fontFamily: "monospace" }}>
          {valueText}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Layer info tooltip
// ---------------------------------------------------------------------------

function LayerTooltip({
  layer,
  activationMap,
}: {
  layer: NeuronLayerDef;
  activationMap: Map<string, number[]>;
}) {
  const acts = activationMap.get(layer.name);
  const meanAct = acts ? acts.reduce((s, v) => s + v, 0) / acts.length : 0;

  return (
    <div style={{
      position: "absolute",
      bottom: 56,
      left: "50%",
      transform: "translateX(-50%)",
      background: "#1a1a28",
      border: `1px solid ${layer.color}40`,
      borderRadius: 8,
      padding: "10px 16px",
      display: "flex",
      alignItems: "center",
      gap: 12,
      pointerEvents: "none",
      zIndex: 50,
      whiteSpace: "nowrap",
      boxShadow: `0 0 20px ${layer.color}10`,
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
// DataFlowLayout — main component
// ---------------------------------------------------------------------------

export function DataFlowLayout() {
  const inputTensor = useInferenceStore(s => s.inputTensor);
  const layerActivations = useInferenceStore(s => s.layerActivations);
  const prediction = useInferenceStore(s => s.prediction);
  const topPrediction = useInferenceStore(s => s.topPrediction);
  const isInferring = useInferenceStore(s => s.isInferring);
  const resetStore = useInferenceStore(s => s.reset);

  const { infer } = useInference();
  const [inspectedLayerIdx, setInspectedLayerIdx] = useState<number | null>(null);
  const [inspectedNeuronIdx, setInspectedNeuronIdx] = useState<number | null>(null);
  const [hoveredLayer, setHoveredLayer] = useState<number | null>(null);
  const [hoveredNeuron, setHoveredNeuron] = useState<HoveredNeuron | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerSize, setContainerSize] = useState({ w: 1200, h: 700 });

  // Wave animation: progresses from 0 to LAYERS.length when data arrives
  const waveRef = useRef(0);
  const waveTargetRef = useRef(0);
  const [waveProgress, setWaveProgress] = useState(0);

  const hasData = Object.keys(layerActivations).length > 0;

  // When data arrives, trigger wave animation
  useEffect(() => {
    if (hasData) {
      waveTargetRef.current = LAYERS.length + 1;
    } else {
      waveTargetRef.current = 0;
      waveRef.current = 0;
      setWaveProgress(0);
    }
  }, [hasData]);

  // Animate wave progress
  useAnimationFrame(() => {
    const target = waveTargetRef.current;
    const current = waveRef.current;
    if (Math.abs(target - current) > 0.01) {
      // Smooth approach — faster for forward, instant for reset
      if (target > current) {
        waveRef.current = current + (target - current) * 0.035;
      } else {
        waveRef.current = 0;
      }
      setWaveProgress(waveRef.current);
    }
  }, true);

  // Drawing canvas
  const onStrokeEnd = useCallback(
    (imageData: ImageData) => { infer(imageData); },
    [infer],
  );

  const { canvasRef, clear, hasDrawn, startDrawing, draw, stopDrawing } =
    useDrawingCanvas({
      width: 280,
      height: 280,
      lineWidth: 16,
      strokeColor: "#ffffff",
      backgroundColor: "#000000",
      onStrokeEnd,
    });

  const handleClear = useCallback(() => {
    clear();
    resetStore();
  }, [clear, resetStore]);

  // Measure container
  useEffect(() => {
    const measure = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setContainerSize({ w: rect.width, h: rect.height });
      }
    };
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, []);

  // Extract activation values
  const activationMap = useMemo(
    () => extractActivations(layerActivations, inputTensor),
    [layerActivations, inputTensor],
  );

  const outputLabels = useMemo(
    () => getOutputLabels(layerActivations),
    [layerActivations],
  );

  // Get raw activation for inspector
  const getActivation = useCallback(
    (name: string) => {
      if (name === "input") return null;
      return layerActivations[name] ?? null;
    },
    [layerActivations],
  );

  const inspectedLayer = inspectedLayerIdx !== null ? LAYERS[inspectedLayerIdx] : null;

  return (
    <div style={{
      width: "100vw",
      height: "100vh",
      overflow: "hidden",
      background: "#08080d",
      position: "relative",
      display: "flex",
    }}>
      {/* Left panel — Drawing Canvas */}
      <div style={{
        width: 220,
        minWidth: 220,
        height: "100%",
        borderRight: "1px solid #1a1a2a",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        padding: "16px 12px",
        gap: 12,
        background: "#0a0a12",
      }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: "rgba(255,255,255,0.6)" }}>
          Draw a character
        </div>

        <div style={{
          position: "relative",
          borderRadius: 10,
          overflow: "hidden",
          border: isInferring ? "2px solid #6366f1" : "2px solid #1a1a2a",
          transition: "border-color 0.3s",
        }}>
          <canvas
            ref={canvasRef}
            width={280}
            height={280}
            style={{ width: 190, height: 190, cursor: "crosshair", touchAction: "none", display: "block" }}
            onMouseDown={e => startDrawing(e.nativeEvent)}
            onMouseMove={e => draw(e.nativeEvent)}
            onMouseUp={() => stopDrawing()}
            onMouseLeave={() => stopDrawing()}
            onTouchStart={e => { e.preventDefault(); startDrawing(e.nativeEvent); }}
            onTouchMove={e => { e.preventDefault(); draw(e.nativeEvent); }}
            onTouchEnd={() => stopDrawing()}
          />
          {!hasDrawn && (
            <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", pointerEvents: "none" }}>
              <span style={{ fontSize: 13, color: "rgba(255,255,255,0.15)" }}>Draw here</span>
            </div>
          )}
        </div>

        <button
          onClick={handleClear}
          style={{
            background: "rgba(255,255,255,0.04)",
            border: "1px solid #1a1a2a",
            borderRadius: 6,
            color: "rgba(255,255,255,0.45)",
            padding: "6px 20px",
            cursor: "pointer",
            fontSize: 12,
          }}
        >
          Clear
        </button>

        {isInferring && (
          <div style={{ fontSize: 11, color: "#6366f1", display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{
              width: 6, height: 6, borderRadius: "50%", background: "#6366f1",
              animation: "pulse 1s infinite",
            }} />
            Processing...
          </div>
        )}

        <div style={{ flex: 1 }} />

        <PredictionSummary prediction={prediction} topPrediction={topPrediction} />
      </div>

      {/* Main neuron network area */}
      <div
        ref={containerRef}
        style={{
          flex: 1,
          position: "relative",
          height: "100%",
          overflow: "hidden",
        }}
      >
        <NeuronNetworkCanvas
          width={containerSize.w}
          height={containerSize.h}
          activationMap={activationMap}
          outputLabels={outputLabels}
          hoveredLayer={hoveredLayer}
          hoveredNeuron={hoveredNeuron}
          onHoverLayer={setHoveredLayer}
          onHoverNeuron={setHoveredNeuron}
          onClickLayer={(li, ni) => { setInspectedLayerIdx(li); setInspectedNeuronIdx(ni); }}
          waveProgress={waveProgress}
        />

        {/* Neuron heatmap tooltip */}
        {hoveredNeuron && hasData && (
          <NeuronHeatmapTooltip
            neuron={hoveredNeuron}
            layerActivations={layerActivations}
            inputTensor={inputTensor}
            outputLabels={outputLabels}
            containerRect={containerRef.current?.getBoundingClientRect() ?? null}
          />
        )}

        {/* Layer tooltip (shown when no individual neuron is hovered) */}
        {hoveredLayer !== null && !hoveredNeuron && (
          <LayerTooltip
            layer={LAYERS[hoveredLayer]}
            activationMap={activationMap}
          />
        )}

        {/* Hint */}
        {!hasData && (
          <div style={{
            position: "absolute",
            top: 20,
            left: "50%",
            transform: "translateX(-50%)",
            fontSize: 13,
            color: "rgba(255,255,255,0.15)",
            background: "rgba(0,0,0,0.3)",
            padding: "6px 16px",
            borderRadius: 20,
          }}>
            Draw a character to see neurons activate and signals flow
          </div>
        )}
      </div>

      {/* Inspector modal */}
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

      <LayoutNav />

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }
      `}</style>
    </div>
  );
}
