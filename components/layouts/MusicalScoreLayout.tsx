"use client";

import React, {
  useState,
  useEffect,
  useRef,
  useMemo,
  useCallback,
} from "react";
import { useInferenceStore } from "@/stores/inferenceStore";
import { useDrawingCanvas } from "@/hooks/useDrawingCanvas";
import { useInference } from "@/hooks/useInference";
import { LayoutNav } from "@/components/layouts/LayoutNav";
import { useAnimationFrame } from "@/hooks/useAnimationFrame";
import {
  loadTrainingHistory,
  type TrainingHistory,
} from "@/lib/training/trainingData";
import {
  EMNIST_CLASSES,
  CLASS_GROUPS,
} from "@/lib/model/classes";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const LAYER_NAMES = [
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

type LayerName = (typeof LAYER_NAMES)[number];

const STAFF_LAYERS: LayerName[] = [
  "conv1",
  "pool1",
  "conv3",
  "pool2",
  "dense1",
  "output",
];

const BG = "#0d0d1a";
const PANEL_BORDER = "1px solid #2a2a3e";
const AMBER = "#ffb300";
const AMBER_LIGHT = "#ffd54f";
const TEXT_COLOR = "#f5f5f5";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Flatten a 3D conv activation [C][H][W] by averaging channels -> HxW -> flat */
function flattenActivation(
  act: number[][][] | number[]
): number[] {
  if (!act || (Array.isArray(act) && act.length === 0)) return [];
  // 1D (dense layers)
  if (!Array.isArray(act[0])) return act as number[];
  // 3D
  const channels = act as number[][][];
  const H = channels[0].length;
  const W = channels[0][0].length;
  const flat: number[] = new Array(H * W).fill(0);
  for (let c = 0; c < channels.length; c++) {
    for (let h = 0; h < H; h++) {
      for (let w = 0; w < W; w++) {
        flat[h * W + w] += channels[c][h][w];
      }
    }
  }
  const n = channels.length;
  for (let i = 0; i < flat.length; i++) flat[i] /= n;
  return flat;
}

/** Compute summary stats for a layer's activations */
function layerStats(act: number[][][] | number[]): {
  mean: number;
  max: number;
} {
  const flat = flattenActivation(act);
  if (flat.length === 0) return { mean: 0, max: 0 };
  let sum = 0;
  let mx = -Infinity;
  for (const v of flat) {
    sum += v;
    if (v > mx) mx = v;
  }
  return { mean: sum / flat.length, max: mx };
}

/** Return class group color for an index */
function classColor(idx: number): string {
  if (idx <= CLASS_GROUPS.digits.end) return "#4fc3f7";
  if (idx <= CLASS_GROUPS.uppercase.end) return "#66bb6a";
  return "#ffa726";
}

// ---------------------------------------------------------------------------
// PianoRollDisplay (Canvas2D)
// ---------------------------------------------------------------------------

function PianoRollDisplay({
  selectedLayer,
  playheadPos,
}: {
  selectedLayer: LayerName;
  playheadPos: number; // 0..1
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const layerActivations = useInferenceStore((s) => s.layerActivations);

  const flat = useMemo(() => {
    const act = layerActivations[selectedLayer];
    if (!act) return [];
    return flattenActivation(act);
  }, [layerActivations, selectedLayer]);

  // Draw
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const W = container.clientWidth;
    const H = container.clientHeight;
    canvas.width = W;
    canvas.height = H;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Background
    ctx.fillStyle = "#0a0a18";
    ctx.fillRect(0, 0, W, H);

    // Piano keyboard on left edge (40px wide)
    const KB_W = 40;
    const octaveKeys = 12;
    const totalKeys = 24; // 2 octaves
    const keyH = H / totalKeys;
    const blackPattern = [1, 3, 6, 8, 10]; // semitones that are black keys

    for (let k = 0; k < totalKeys; k++) {
      const semitone = k % octaveKeys;
      const isBlack = blackPattern.includes(semitone);
      ctx.fillStyle = isBlack ? "#1a1a2e" : "#2a2a3e";
      ctx.fillRect(0, H - (k + 1) * keyH, KB_W, keyH - 1);
    }

    // Grid lines at octave boundaries
    ctx.strokeStyle = "#333";
    ctx.setLineDash([4, 4]);
    for (let oct = 0; oct <= 2; oct++) {
      const y = H - oct * octaveKeys * keyH;
      ctx.beginPath();
      ctx.moveTo(KB_W, y);
      ctx.lineTo(W, y);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // Draw piano roll notes
    if (flat.length > 0) {
      const maxVal = Math.max(...flat, 0.001);
      const plotW = W - KB_W;
      const noteW = Math.max(2, Math.min(4, plotW / flat.length));
      const noteH = 3;

      for (let i = 0; i < flat.length; i++) {
        const v = flat[i];
        const normalized = Math.max(0, v) / maxVal;
        const x = KB_W + (i / flat.length) * plotW;
        const y = H - normalized * (H - 4) - 2;
        const alpha = Math.min(1, normalized * 1.5);
        // Interpolate #ff8c00 to #ffd700
        const r = 255;
        const g = Math.round(140 + normalized * (215 - 140));
        const b = Math.round(0 + normalized * 0);
        ctx.fillStyle = `rgba(${r},${g},${b},${alpha})`;
        ctx.fillRect(x, y, noteW, noteH);
      }
    }

    // Playhead
    if (playheadPos > 0) {
      const px = KB_W + playheadPos * (W - KB_W);
      ctx.strokeStyle = "#ffd54f";
      ctx.lineWidth = 2;
      ctx.globalAlpha = 0.8;
      ctx.beginPath();
      ctx.moveTo(px, 0);
      ctx.lineTo(px, H);
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.lineWidth = 1;
    }
  }, [flat, playheadPos]);

  return (
    <div
      ref={containerRef}
      style={{ width: "100%", height: "100%", position: "relative" }}
    >
      <canvas
        ref={canvasRef}
        style={{ width: "100%", height: "100%", display: "block" }}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// WaveformStaff (SVG)
// ---------------------------------------------------------------------------

function WaveformStaff() {
  const layerActivations = useInferenceStore((s) => s.layerActivations);

  const staffData = useMemo(() => {
    return STAFF_LAYERS.map((name) => {
      const act = layerActivations[name];
      if (!act) return { name, notes: [] as number[] };
      const flat = flattenActivation(act);
      // Downsample to max 80 notes for readability
      const step = Math.max(1, Math.floor(flat.length / 80));
      const sampled: number[] = [];
      for (let i = 0; i < flat.length; i += step) {
        sampled.push(flat[i]);
      }
      return { name, notes: sampled };
    });
  }, [layerActivations]);

  const staffH = 60;
  const totalH = STAFF_LAYERS.length * staffH;
  const svgW = 800;

  return (
    <svg
      viewBox={`0 0 ${svgW} ${totalH}`}
      style={{ width: "100%", height: "100%" }}
      preserveAspectRatio="xMidYMid meet"
    >
      {staffData.map((sd, si) => {
        const yOff = si * staffH;
        const staffLines: React.ReactNode[] = [];
        // 5 staff lines
        for (let l = 0; l < 5; l++) {
          const ly = yOff + 10 + l * 10;
          staffLines.push(
            <line
              key={`line-${si}-${l}`}
              x1={60}
              y1={ly}
              x2={svgW - 10}
              y2={ly}
              stroke="#444"
              strokeWidth={0.5}
            />
          );
        }

        // Layer name (clef)
        const clef = (
          <text
            key={`clef-${si}`}
            x={5}
            y={yOff + 35}
            fill={AMBER}
            fontSize={10}
            fontStyle="italic"
            fontFamily="serif"
          >
            {sd.name}
          </text>
        );

        // Notes
        const noteNodes: React.ReactNode[] = [];
        if (sd.notes.length > 0) {
          const maxVal = Math.max(...sd.notes.map(Math.abs), 0.001);
          const noteSpacing = (svgW - 70) / Math.max(sd.notes.length, 1);

          for (let ni = 0; ni < sd.notes.length; ni++) {
            const v = sd.notes[ni];
            const norm = Math.max(0, v) / maxVal;
            // Map to staff position: 0=bottom line, 1=top line
            const staffY = yOff + 50 - norm * 40;
            const cx = 65 + ni * noteSpacing;
            const filled = norm > 0.4;
            noteNodes.push(
              <circle
                key={`note-${si}-${ni}`}
                cx={cx}
                cy={staffY}
                r={3}
                fill={filled ? "#ffb300" : "none"}
                stroke="#ffb300"
                strokeWidth={filled ? 0 : 1}
                opacity={0.4 + norm * 0.6}
              />
            );

            // Beam: connect consecutive high activations
            if (
              ni > 0 &&
              norm > 0.6 &&
              Math.max(0, sd.notes[ni - 1]) / maxVal > 0.6
            ) {
              const prevNorm = Math.max(0, sd.notes[ni - 1]) / maxVal;
              const prevY = yOff + 50 - prevNorm * 40;
              const prevCx = 65 + (ni - 1) * noteSpacing;
              noteNodes.push(
                <line
                  key={`beam-${si}-${ni}`}
                  x1={prevCx}
                  y1={prevY - 5}
                  x2={cx}
                  y2={staffY - 5}
                  stroke="#ffb300"
                  strokeWidth={1.5}
                  opacity={0.6}
                />
              );
            }
          }
        }

        return (
          <g key={`staff-${si}`}>
            {staffLines}
            {clef}
            {noteNodes}
          </g>
        );
      })}
    </svg>
  );
}

// ---------------------------------------------------------------------------
// MixingConsole
// ---------------------------------------------------------------------------

function MixingConsole({
  selectedLayer,
  soloLayer,
  mutedLayers,
  onSelect,
  onToggleSolo,
  onToggleMute,
}: {
  selectedLayer: LayerName;
  soloLayer: LayerName | null;
  mutedLayers: Set<LayerName>;
  onSelect: (l: LayerName) => void;
  onToggleSolo: (l: LayerName) => void;
  onToggleMute: (l: LayerName) => void;
}) {
  const layerActivations = useInferenceStore((s) => s.layerActivations);

  const summaries = useMemo(() => {
    const map: Record<string, { mean: number; max: number }> = {};
    for (const name of LAYER_NAMES) {
      const act = layerActivations[name];
      map[name] = act ? layerStats(act) : { mean: 0, max: 0 };
    }
    return map;
  }, [layerActivations]);

  // Find global max for normalization
  const globalMax = useMemo(() => {
    let mx = 0.001;
    for (const s of Object.values(summaries)) {
      if (s.max > mx) mx = s.max;
    }
    return mx;
  }, [summaries]);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 2,
        height: "100%",
        overflowY: "auto",
        padding: 4,
      }}
    >
      {LAYER_NAMES.map((name) => {
        const s = summaries[name];
        const isSelected = name === selectedLayer;
        const isSolo = name === soloLayer;
        const isMuted = mutedLayers.has(name);
        const peakNorm = s.max / globalMax;
        const meanNorm = s.mean / globalMax;

        return (
          <div
            key={name}
            onClick={() => onSelect(name)}
            style={{
              background: "linear-gradient(135deg, #1a1a2e 0%, #16162a 100%)",
              border: isSelected
                ? `1px solid ${AMBER}`
                : PANEL_BORDER,
              borderRadius: 4,
              padding: "4px 6px",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: 6,
              minHeight: 36,
            }}
          >
            {/* Label */}
            <div
              style={{
                fontSize: 9,
                color: isSelected ? AMBER : TEXT_COLOR,
                width: 42,
                fontWeight: isSelected ? 700 : 400,
                flexShrink: 0,
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
            >
              {name}
            </div>

            {/* Peak meter: LED segments */}
            <div
              style={{
                display: "flex",
                flexDirection: "column-reverse",
                gap: 1,
                width: 8,
                height: 40,
                flexShrink: 0,
              }}
            >
              {Array.from({ length: 10 }).map((_, i) => {
                const threshold = (i + 1) / 10;
                const lit = peakNorm >= threshold;
                let color: string;
                if (i >= 8) color = lit ? "#ff1744" : "#330a10";
                else if (i >= 6) color = lit ? "#ffeb3b" : "#332e0a";
                else color = lit ? "#69f0ae" : "#0a331c";
                return (
                  <div
                    key={i}
                    style={{
                      width: "100%",
                      height: 3,
                      borderRadius: 1,
                      background: color,
                    }}
                  />
                );
              })}
            </div>

            {/* Fader */}
            <div
              style={{
                position: "relative",
                width: 6,
                height: 40,
                background: "#111",
                borderRadius: 3,
                flexShrink: 0,
              }}
            >
              <div
                style={{
                  position: "absolute",
                  bottom: 0,
                  left: 0,
                  width: "100%",
                  height: `${meanNorm * 100}%`,
                  background: `linear-gradient(to top, #555, ${AMBER})`,
                  borderRadius: 3,
                }}
              />
              {/* Knob */}
              <div
                style={{
                  position: "absolute",
                  left: -2,
                  bottom: `calc(${meanNorm * 100}% - 4px)`,
                  width: 10,
                  height: 8,
                  background: "#ddd",
                  borderRadius: 2,
                  boxShadow: "0 1px 3px rgba(0,0,0,0.5)",
                }}
              />
            </div>

            {/* Solo / Mute buttons */}
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: 2,
                flexShrink: 0,
              }}
            >
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onToggleSolo(name);
                }}
                style={{
                  width: 16,
                  height: 14,
                  fontSize: 8,
                  fontWeight: 700,
                  border: "none",
                  borderRadius: 2,
                  cursor: "pointer",
                  background: isSolo ? "#ffd54f" : "#333",
                  color: isSolo ? "#000" : "#888",
                  lineHeight: "14px",
                  padding: 0,
                }}
              >
                S
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onToggleMute(name);
                }}
                style={{
                  width: 16,
                  height: 14,
                  fontSize: 8,
                  fontWeight: 700,
                  border: "none",
                  borderRadius: 2,
                  cursor: "pointer",
                  background: isMuted ? "#ff5252" : "#333",
                  color: isMuted ? "#fff" : "#888",
                  lineHeight: "14px",
                  padding: 0,
                }}
              >
                M
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// ChordDiagram (SVG)
// ---------------------------------------------------------------------------

function ChordDiagram() {
  const prediction = useInferenceStore((s) => s.prediction);
  const topPrediction = useInferenceStore((s) => s.topPrediction);

  const size = 280;
  const cx = size / 2;
  const cy = size / 2;
  const radius = size / 2 - 30;

  const nodes = useMemo(() => {
    return EMNIST_CLASSES.map((ch, i) => {
      const angle = (i / 62) * Math.PI * 2 - Math.PI / 2;
      const prob = prediction ? prediction[i] : 0;
      return {
        index: i,
        char: ch,
        angle,
        x: cx + Math.cos(angle) * radius,
        y: cy + Math.sin(angle) * radius,
        prob,
        color: classColor(i),
      };
    });
  }, [prediction]);

  // Arcs between nodes > 1%
  const arcs = useMemo(() => {
    if (!prediction) return [];
    const result: { i: number; j: number; strength: number }[] = [];
    for (let i = 0; i < 62; i++) {
      if (prediction[i] < 0.01) continue;
      for (let j = i + 1; j < 62; j++) {
        if (prediction[j] < 0.01) continue;
        result.push({
          i,
          j,
          strength: prediction[i] * prediction[j],
        });
      }
    }
    // limit to top 30 arcs
    result.sort((a, b) => b.strength - a.strength);
    return result.slice(0, 30);
  }, [prediction]);

  const maxArc = useMemo(() => {
    if (arcs.length === 0) return 0.001;
    return Math.max(...arcs.map((a) => a.strength));
  }, [arcs]);

  return (
    <svg
      viewBox={`0 0 ${size} ${size}`}
      style={{ width: "100%", height: "100%" }}
      preserveAspectRatio="xMidYMid meet"
    >
      <defs>
        <filter id="glow">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Arcs */}
      {arcs.map((arc, ai) => {
        const n1 = nodes[arc.i];
        const n2 = nodes[arc.j];
        const thickness = 0.5 + (arc.strength / maxArc) * 2;
        return (
          <line
            key={`arc-${ai}`}
            x1={n1.x}
            y1={n1.y}
            x2={n2.x}
            y2={n2.y}
            stroke={n1.color}
            strokeWidth={thickness}
            opacity={0.2 + (arc.strength / maxArc) * 0.4}
          />
        );
      })}

      {/* Nodes */}
      {nodes.map((n) => {
        const isTop = topPrediction?.classIndex === n.index;
        const r = Math.max(2, 3 + n.prob * 20);
        return (
          <g key={`node-${n.index}`}>
            <circle
              cx={n.x}
              cy={n.y}
              r={isTop ? r + 3 : r}
              fill={n.color}
              opacity={Math.max(0.2, n.prob)}
              filter={isTop ? "url(#glow)" : undefined}
              stroke={isTop ? "#ffd700" : "none"}
              strokeWidth={isTop ? 2 : 0}
            />
            <text
              x={
                cx +
                Math.cos(n.angle) * (radius + 14)
              }
              y={
                cy +
                Math.sin(n.angle) * (radius + 14)
              }
              fill={TEXT_COLOR}
              fontSize={6}
              textAnchor="middle"
              dominantBaseline="middle"
              opacity={0.6}
            >
              {n.char}
            </text>
          </g>
        );
      })}

      {/* Center prediction */}
      {topPrediction && (
        <>
          <text
            x={cx}
            y={cy - 6}
            fill={AMBER_LIGHT}
            fontSize={32}
            fontWeight={700}
            textAnchor="middle"
            dominantBaseline="middle"
          >
            {EMNIST_CLASSES[topPrediction.classIndex]}
          </text>
          <text
            x={cx}
            y={cy + 18}
            fill={TEXT_COLOR}
            fontSize={10}
            textAnchor="middle"
            opacity={0.8}
          >
            {(topPrediction.confidence * 100).toFixed(1)}%
          </text>
        </>
      )}
    </svg>
  );
}

// ---------------------------------------------------------------------------
// TempoTimeline (Canvas2D)
// ---------------------------------------------------------------------------

function TempoTimeline() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [history, setHistory] = useState<TrainingHistory | null>(null);
  const [hoverEpoch, setHoverEpoch] = useState<number | null>(null);

  useEffect(() => {
    loadTrainingHistory()
      .then(setHistory)
      .catch(() => {});
  }, []);

  // Draw
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || !history) return;

    const W = container.clientWidth;
    const H = container.clientHeight;
    canvas.width = W;
    canvas.height = H;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Background
    ctx.fillStyle = "#0e0e1c";
    ctx.fillRect(0, 0, W, H);

    // Staff-paper horizontal lines
    ctx.strokeStyle = "#1a1a2e";
    ctx.lineWidth = 0.5;
    for (let i = 0; i < 12; i++) {
      const y = (i / 12) * H;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(W, y);
      ctx.stroke();
    }

    const epochs = history.loss.length;
    const padL = 30;
    const padR = 10;
    const padT = 15;
    const padB = 20;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    const maxLoss = Math.max(...history.loss, ...history.val_loss, 0.001);

    // Helper: x position for epoch
    const xForEpoch = (e: number) => padL + (e / (epochs - 1)) * plotW;
    // Helper: y position for loss value
    const yForLoss = (v: number) => padT + (1 - v / maxLoss) * plotH;
    // Helper: y position for accuracy (0-1)
    const yForAcc = (v: number) => padT + (1 - v) * plotH;

    // ---- Loss: descending decrescendo wedge ----
    // Draw as a filled area that narrows (tall at start, narrow at end)
    ctx.beginPath();
    for (let e = 0; e < epochs; e++) {
      const x = xForEpoch(e);
      const y = yForLoss(history.loss[e]);
      if (e === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = "#ff5252";
    ctx.lineWidth = 2;
    ctx.stroke();

    // val_loss - dotted
    ctx.beginPath();
    ctx.setLineDash([4, 3]);
    for (let e = 0; e < epochs; e++) {
      const x = xForEpoch(e);
      const y = yForLoss(history.val_loss[e]);
      if (e === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = "#ff5252";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.setLineDash([]);

    // ---- Accuracy: ascending crescendo ----
    ctx.beginPath();
    for (let e = 0; e < epochs; e++) {
      const x = xForEpoch(e);
      const y = yForAcc(history.accuracy[e]);
      if (e === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = "#69f0ae";
    ctx.lineWidth = 2;
    ctx.stroke();

    // val_accuracy - dotted
    ctx.beginPath();
    ctx.setLineDash([4, 3]);
    for (let e = 0; e < epochs; e++) {
      const x = xForEpoch(e);
      const y = yForAcc(history.val_accuracy[e]);
      if (e === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = "#69f0ae";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.setLineDash([]);

    // Epoch beat dots
    for (let e = 0; e < epochs; e++) {
      const x = xForEpoch(e);
      ctx.fillStyle = "#444";
      ctx.beginPath();
      ctx.arc(x, H - padB + 8, 1.5, 0, Math.PI * 2);
      ctx.fill();
    }

    // Decrescendo / crescendo wedge symbols
    // Loss decrescendo ▷ narrowing
    ctx.fillStyle = "#ff5252";
    ctx.globalAlpha = 0.15;
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, padT + plotH * 0.3);
    ctx.lineTo(padL + plotW * 0.3, padT + plotH * 0.15);
    ctx.closePath();
    ctx.fill();

    // Accuracy crescendo ◁ expanding
    ctx.fillStyle = "#69f0ae";
    ctx.beginPath();
    ctx.moveTo(padL + plotW * 0.7, padT + plotH * 0.7);
    ctx.lineTo(padL + plotW, padT + plotH * 0.55);
    ctx.lineTo(padL + plotW, padT + plotH * 0.85);
    ctx.closePath();
    ctx.fill();
    ctx.globalAlpha = 1;

    // Axis labels
    ctx.fillStyle = TEXT_COLOR;
    ctx.font = "8px sans-serif";
    ctx.globalAlpha = 0.6;
    ctx.fillText("Epoch", W / 2 - 15, H - 2);
    ctx.fillText("0", padL - 2, H - padB + 8);
    ctx.fillText(String(epochs - 1), padL + plotW - 10, H - padB + 8);

    // Legend
    ctx.globalAlpha = 0.8;
    ctx.fillStyle = "#ff5252";
    ctx.fillRect(W - 80, 4, 8, 8);
    ctx.fillStyle = TEXT_COLOR;
    ctx.fillText("Loss", W - 68, 11);
    ctx.fillStyle = "#69f0ae";
    ctx.fillRect(W - 80, 16, 8, 8);
    ctx.fillStyle = TEXT_COLOR;
    ctx.fillText("Accuracy", W - 68, 23);
    ctx.globalAlpha = 1;

    // Hover tooltip
    if (hoverEpoch !== null && hoverEpoch >= 0 && hoverEpoch < epochs) {
      const hx = xForEpoch(hoverEpoch);
      ctx.strokeStyle = AMBER;
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.5;
      ctx.beginPath();
      ctx.moveTo(hx, padT);
      ctx.lineTo(hx, padT + plotH);
      ctx.stroke();
      ctx.globalAlpha = 1;

      // Tooltip box
      const tipW = 110;
      const tipH = 38;
      const tipX = Math.min(hx + 5, W - tipW - 5);
      const tipY = padT + 5;
      ctx.fillStyle = "rgba(20,20,40,0.9)";
      ctx.fillRect(tipX, tipY, tipW, tipH);
      ctx.strokeStyle = AMBER;
      ctx.lineWidth = 0.5;
      ctx.strokeRect(tipX, tipY, tipW, tipH);
      ctx.fillStyle = TEXT_COLOR;
      ctx.font = "8px sans-serif";
      ctx.fillText(`Epoch ${hoverEpoch}`, tipX + 4, tipY + 10);
      ctx.fillStyle = "#ff5252";
      ctx.fillText(
        `Loss: ${history.loss[hoverEpoch].toFixed(4)}`,
        tipX + 4,
        tipY + 21
      );
      ctx.fillStyle = "#69f0ae";
      ctx.fillText(
        `Acc: ${(history.accuracy[hoverEpoch] * 100).toFixed(1)}%`,
        tipX + 4,
        tipY + 32
      );
    }
  }, [history, hoverEpoch]);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas || !history) return;
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const W = rect.width;
      const padL = 30;
      const padR = 10;
      const plotW = W - padL - padR;
      const relX = (x - padL) / plotW;
      const epoch = Math.round(relX * (history.loss.length - 1));
      if (epoch >= 0 && epoch < history.loss.length) {
        setHoverEpoch(epoch);
      } else {
        setHoverEpoch(null);
      }
    },
    [history]
  );

  const handleMouseLeave = useCallback(() => {
    setHoverEpoch(null);
  }, []);

  return (
    <div
      ref={containerRef}
      style={{ width: "100%", height: "100%", position: "relative" }}
    >
      <canvas
        ref={canvasRef}
        style={{ width: "100%", height: "100%", display: "block", cursor: "crosshair" }}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// PlayheadControls
// ---------------------------------------------------------------------------

function PlayheadControls({
  isPlaying,
  speed,
  onTogglePlay,
  onSetSpeed,
  onSkipStart,
  onSkipEnd,
  onStepBack,
  onStepForward,
  playheadPos,
}: {
  isPlaying: boolean;
  speed: number;
  onTogglePlay: () => void;
  onSetSpeed: (s: number) => void;
  onSkipStart: () => void;
  onSkipEnd: () => void;
  onStepBack: () => void;
  onStepForward: () => void;
  playheadPos: number;
}) {
  const btnStyle: React.CSSProperties = {
    background: "none",
    border: "1px solid #2a2a3e",
    color: TEXT_COLOR,
    cursor: "pointer",
    fontSize: 14,
    padding: "2px 6px",
    borderRadius: 4,
    lineHeight: "20px",
  };

  const speedBtnStyle = (s: number): React.CSSProperties => ({
    ...btnStyle,
    fontSize: 10,
    padding: "1px 5px",
    background: speed === s ? AMBER : "transparent",
    color: speed === s ? "#000" : TEXT_COLOR,
    fontWeight: speed === s ? 700 : 400,
  });

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 4,
      }}
    >
      <button style={btnStyle} onClick={onSkipStart} title="Skip to start">
        {"⏮"}
      </button>
      <button style={btnStyle} onClick={onStepBack} title="Step back">
        {"⏪"}
      </button>
      <button style={btnStyle} onClick={onTogglePlay} title={isPlaying ? "Pause" : "Play"}>
        {isPlaying ? "⏸" : "▶"}
      </button>
      <button style={btnStyle} onClick={onStepForward} title="Step forward">
        {"⏩"}
      </button>
      <button style={btnStyle} onClick={onSkipEnd} title="Skip to end">
        {"⏭"}
      </button>

      <div
        style={{
          marginLeft: 8,
          display: "flex",
          gap: 2,
        }}
      >
        <button style={speedBtnStyle(1)} onClick={() => onSetSpeed(1)}>
          1x
        </button>
        <button style={speedBtnStyle(2)} onClick={() => onSetSpeed(2)}>
          2x
        </button>
        <button style={speedBtnStyle(4)} onClick={() => onSetSpeed(4)}>
          4x
        </button>
      </div>

      <div
        style={{
          marginLeft: 8,
          fontSize: 9,
          color: "#888",
          minWidth: 40,
        }}
      >
        {(playheadPos * 100).toFixed(0)}%
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// DrawingPad (REC interface)
// ---------------------------------------------------------------------------

function DrawingPad() {
  const isInferring = useInferenceStore((s) => s.isInferring);
  const { infer } = useInference();
  const { canvasRef, clear, startDrawing, draw, stopDrawing } =
    useDrawingCanvas({
      width: 200,
      height: 200,
      lineWidth: 14,
      strokeColor: "#ffffff",
      backgroundColor: "#000000",
      onStrokeEnd: (imageData) => {
        infer(imageData);
      },
    });

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 4,
      }}
    >
      {/* REC label */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          fontSize: 11,
          fontWeight: 700,
          color: TEXT_COLOR,
          letterSpacing: 2,
        }}
      >
        <div
          style={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: isInferring ? "#ff1744" : "#662222",
            boxShadow: isInferring ? "0 0 8px #ff1744" : "none",
            animation: isInferring ? "pulse 0.6s ease-in-out infinite" : "none",
          }}
        />
        REC
      </div>

      {/* Canvas */}
      <div
        style={{
          border: PANEL_BORDER,
          borderRadius: 8,
          overflow: "hidden",
          background: "#000",
          width: 120,
          height: 120,
        }}
      >
        <canvas
          ref={canvasRef}
          width={200}
          height={200}
          style={{ width: 120, height: 120, display: "block", cursor: "crosshair" }}
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

      {/* STOP / Clear button */}
      <button
        onClick={clear}
        style={{
          background: "#333",
          border: "1px solid #555",
          borderRadius: 4,
          color: TEXT_COLOR,
          fontSize: 10,
          padding: "3px 10px",
          cursor: "pointer",
          display: "flex",
          alignItems: "center",
          gap: 4,
        }}
      >
        <span
          style={{
            display: "inline-block",
            width: 8,
            height: 8,
            background: "#ff5252",
            borderRadius: 1,
          }}
        />
        STOP
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Layout
// ---------------------------------------------------------------------------

export function MusicalScoreLayout() {
  // State
  const [selectedLayer, setSelectedLayer] = useState<LayerName>("conv1");
  const [soloLayer, setSoloLayer] = useState<LayerName | null>(null);
  const [mutedLayers, setMutedLayers] = useState<Set<LayerName>>(new Set());
  const [isPlaying, setIsPlaying] = useState(false);
  const [playheadPos, setPlayheadPos] = useState(0);
  const [speed, setSpeed] = useState(1);

  // Playhead animation: use ref for speed so animCallback stays stable
  const speedRef = useRef(speed);
  useEffect(() => {
    speedRef.current = speed;
  }, [speed]);

  const animCallback = useCallback(
    (deltaTime: number) => {
      // Move playhead: full sweep in ~5 seconds at 1x
      const increment = (deltaTime / 5000) * speedRef.current;
      setPlayheadPos((prev) => {
        const next = prev + increment;
        if (next >= 1) {
          setIsPlaying(false);
          return 1;
        }
        return next;
      });
    },
    []
  );

  useAnimationFrame(animCallback, isPlaying);

  // Effective selected layer considering solo
  const effectiveLayer = soloLayer ?? selectedLayer;

  // Handlers
  const handleToggleSolo = useCallback(
    (l: LayerName) => {
      setSoloLayer((prev) => (prev === l ? null : l));
    },
    []
  );

  const handleToggleMute = useCallback(
    (l: LayerName) => {
      setMutedLayers((prev) => {
        const next = new Set(prev);
        if (next.has(l)) next.delete(l);
        else next.add(l);
        return next;
      });
    },
    []
  );

  return (
    <div
      style={{
        width: "100vw",
        height: "100vh",
        background: BG,
        color: TEXT_COLOR,
        fontFamily: "system-ui, sans-serif",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
      }}
    >
      {/* Pulse animation keyframes */}
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }
      `}</style>

      {/* ===== TOP BAR ===== */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "6px 12px",
          borderBottom: PANEL_BORDER,
          flexShrink: 0,
          height: 180,
          minHeight: 180,
        }}
      >
        {/* Left: Drawing Pad */}
        <DrawingPad />

        {/* Center: Playhead Controls */}
        <PlayheadControls
          isPlaying={isPlaying}
          speed={speed}
          onTogglePlay={() => {
            if (playheadPos >= 1) setPlayheadPos(0);
            setIsPlaying((p) => !p);
          }}
          onSetSpeed={setSpeed}
          onSkipStart={() => {
            setPlayheadPos(0);
            setIsPlaying(false);
          }}
          onSkipEnd={() => {
            setPlayheadPos(1);
            setIsPlaying(false);
          }}
          onStepBack={() =>
            setPlayheadPos((p) => Math.max(0, p - 0.05))
          }
          onStepForward={() =>
            setPlayheadPos((p) => Math.min(1, p + 0.05))
          }
          playheadPos={playheadPos}
        />

        {/* Right: Title */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "flex-end",
          }}
        >
          <div
            style={{
              fontSize: 20,
              fontWeight: 700,
              fontFamily: "serif",
              color: AMBER_LIGHT,
              letterSpacing: 1,
            }}
          >
            Neural Score
          </div>
          <div
            style={{
              fontSize: 9,
              color: "#888",
              fontStyle: "italic",
              fontFamily: "serif",
            }}
          >
            OCR Network Visualization
          </div>
        </div>
      </div>

      {/* ===== MIDDLE AREA: Left (piano roll + staves) + Right (mixing console) ===== */}
      <div
        style={{
          display: "flex",
          flex: 1,
          minHeight: 0,
          overflow: "hidden",
        }}
      >
        {/* Left column: 70% */}
        <div
          style={{
            width: "70%",
            display: "flex",
            flexDirection: "column",
            borderRight: PANEL_BORDER,
            minHeight: 0,
          }}
        >
          {/* Piano Roll: top half */}
          <div
            style={{
              flex: 1,
              minHeight: 0,
              borderBottom: PANEL_BORDER,
              position: "relative",
            }}
          >
            <div
              style={{
                position: "absolute",
                top: 4,
                left: 50,
                fontSize: 9,
                color: "#666",
                zIndex: 1,
                fontFamily: "serif",
                fontStyle: "italic",
              }}
            >
              Piano Roll: {effectiveLayer}
            </div>
            <PianoRollDisplay
              selectedLayer={effectiveLayer}
              playheadPos={playheadPos}
            />
          </div>

          {/* Waveform Staves: bottom half, scrollable */}
          <div
            style={{
              flex: 1,
              minHeight: 0,
              overflowY: "auto",
              padding: 4,
              position: "relative",
            }}
          >
            <div
              style={{
                position: "sticky",
                top: 0,
                fontSize: 9,
                color: "#666",
                fontFamily: "serif",
                fontStyle: "italic",
                paddingBottom: 2,
                background: BG,
                zIndex: 1,
              }}
            >
              Staff Notation
            </div>
            <WaveformStaff />
          </div>
        </div>

        {/* Right column: 30% Mixing Console */}
        <div
          style={{
            width: "30%",
            display: "flex",
            flexDirection: "column",
            minHeight: 0,
          }}
        >
          <div
            style={{
              fontSize: 9,
              color: "#666",
              fontFamily: "serif",
              fontStyle: "italic",
              padding: "4px 6px 0",
            }}
          >
            Mixing Console
          </div>
          <div style={{ flex: 1, minHeight: 0, overflowY: "auto" }}>
            <MixingConsole
              selectedLayer={selectedLayer}
              soloLayer={soloLayer}
              mutedLayers={mutedLayers}
              onSelect={setSelectedLayer}
              onToggleSolo={handleToggleSolo}
              onToggleMute={handleToggleMute}
            />
          </div>
        </div>
      </div>

      {/* ===== BOTTOM BAR: Chord diagram + Tempo timeline ===== */}
      <div
        style={{
          display: "flex",
          borderTop: PANEL_BORDER,
          height: 220,
          flexShrink: 0,
        }}
      >
        {/* Chord Diagram: left ~40% */}
        <div
          style={{
            width: "40%",
            borderRight: PANEL_BORDER,
            position: "relative",
            display: "flex",
            flexDirection: "column",
          }}
        >
          <div
            style={{
              fontSize: 9,
              color: "#666",
              fontFamily: "serif",
              fontStyle: "italic",
              padding: "4px 6px 0",
            }}
          >
            Output Chord
          </div>
          <div style={{ flex: 1, minHeight: 0, padding: 4 }}>
            <ChordDiagram />
          </div>
        </div>

        {/* Tempo Timeline: right ~60% */}
        <div
          style={{
            width: "60%",
            position: "relative",
            display: "flex",
            flexDirection: "column",
          }}
        >
          <div
            style={{
              fontSize: 9,
              color: "#666",
              fontFamily: "serif",
              fontStyle: "italic",
              padding: "4px 6px 0",
            }}
          >
            Training Dynamics
          </div>
          <div style={{ flex: 1, minHeight: 0 }}>
            <TempoTimeline />
          </div>
        </div>
      </div>

      {/* ===== Layout Nav ===== */}
      <LayoutNav />
    </div>
  );
}
