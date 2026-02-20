"use client";

import {
  useRef,
  useState,
  useCallback,
  useMemo,
  useEffect,
  type MouseEvent as ReactMouseEvent,
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

interface ComponentDef {
  key: LayerKey;
  label: string;
  designator: string;
  type: "input" | "conv" | "relu" | "pool" | "dense" | "output";
  shape: string;
  filterCount?: number;
}

const COMPONENTS: ComponentDef[] = [
  { key: "input", label: "Input 28x28", designator: "J1", type: "input", shape: "28x28" },
  { key: "conv1", label: "Conv 32x3x3", designator: "U1", type: "conv", shape: "32x28x28", filterCount: 32 },
  { key: "relu1", label: "ReLU", designator: "D1", type: "relu", shape: "32x28x28" },
  { key: "conv2", label: "Conv 64x3x3", designator: "U2", type: "conv", shape: "64x28x28", filterCount: 64 },
  { key: "relu2", label: "ReLU", designator: "D2", type: "relu", shape: "64x28x28" },
  { key: "pool1", label: "Pool 2x2", designator: "U3", type: "pool", shape: "64x14x14" },
  { key: "conv3", label: "Conv 128x3x3", designator: "U4", type: "conv", shape: "128x14x14", filterCount: 128 },
  { key: "relu3", label: "ReLU", designator: "D3", type: "relu", shape: "128x14x14" },
  { key: "pool2", label: "Pool 2x2", designator: "U5", type: "pool", shape: "128x7x7" },
  { key: "dense1", label: "Dense 256", designator: "U6", type: "dense", shape: "256" },
  { key: "relu4", label: "ReLU", designator: "D4", type: "relu", shape: "256" },
  { key: "output", label: "Output 62", designator: "U7", type: "output", shape: "62" },
];

// Layout constants
const COMP_W = 140;
const COMP_H = 80;
const H_GAP = 40;
const V_GAP = 50;
const ROW_HEIGHT = COMP_H + V_GAP;
const MARGIN_X = 80;
const MARGIN_Y = 60;

// Colors
const PCB_GREEN = "#1a4a2e";
const COPPER = "#c87533";
const COPPER_BRIGHT = "#ffa726";
const COMP_BODY = "#1a1a1a";
const GOLD_PULSE = "#ffab00";
const LED_OFF = "#333333";
const LED_DIM = "#1a5a1a";
const LED_MED = "#33ff33";
const LED_HIGH = "#ffab00";
const SEGMENT_ON = "#ff1744";
const SEGMENT_OFF = "#330000";

const DRAWING_CANVAS_SIZE = 180;
const DRAWING_INTERNAL_SIZE = 280;

// ---------------------------------------------------------------------------
// Layout positions: 3-row serpentine
// ---------------------------------------------------------------------------

interface CompPos {
  x: number;
  y: number;
  row: number;
}

function computePositions(): Record<LayerKey, CompPos> {
  // Row 1 (L->R): input, conv1, relu1, conv2
  // Row 2 (R->L): relu2, pool1, conv3, relu3
  // Row 3 (L->R): pool2, dense1, relu4, output
  const rows: LayerKey[][] = [
    ["input", "conv1", "relu1", "conv2"],
    ["relu2", "pool1", "conv3", "relu3"],
    ["pool2", "dense1", "relu4", "output"],
  ];

  const positions: Partial<Record<LayerKey, CompPos>> = {};

  for (let r = 0; r < rows.length; r++) {
    const row = rows[r];
    const isReversed = r === 1;
    for (let i = 0; i < row.length; i++) {
      const col = isReversed ? (row.length - 1 - i) : i;
      positions[row[i]] = {
        x: MARGIN_X + col * (COMP_W + H_GAP),
        y: MARGIN_Y + r * ROW_HEIGHT,
        row: r,
      };
    }
  }

  return positions as Record<LayerKey, CompPos>;
}

const POSITIONS = computePositions();

// Total SVG dimensions
const SVG_W = MARGIN_X * 2 + 4 * COMP_W + 3 * H_GAP;
const SVG_H = MARGIN_Y * 2 + 3 * COMP_H + 2 * V_GAP;

// ---------------------------------------------------------------------------
// Trace path computation
// ---------------------------------------------------------------------------

interface TraceDef {
  from: LayerKey;
  to: LayerKey;
  pathD: string;
  wide: boolean;
}

function computeTraces(): TraceDef[] {
  const traces: TraceDef[] = [];
  for (let i = 0; i < LAYER_KEYS.length - 1; i++) {
    const from = LAYER_KEYS[i];
    const to = LAYER_KEYS[i + 1];
    const fp = POSITIONS[from];
    const tp = POSITIONS[to];

    const fromX = fp.x + COMP_W;
    const fromY = fp.y + COMP_H / 2;
    const toX = tp.x;
    const toY = tp.y + COMP_H / 2;

    let pathD: string;

    if (fp.row === tp.row) {
      // Same row: simple horizontal or right-angle if reversed
      if (fp.row === 1) {
        // Row 2 goes R->L, so output of "from" is on the left side
        const fx = fp.x;
        const fy = fp.y + COMP_H / 2;
        const tx = tp.x + COMP_W;
        const ty = tp.y + COMP_H / 2;
        pathD = `M ${fx} ${fy} L ${tx} ${ty}`;
      } else {
        pathD = `M ${fromX} ${fromY} L ${toX} ${toY}`;
      }
    } else {
      // Different rows: right-angle routing
      if (fp.row === 0 && tp.row === 1) {
        // End of row 1 (conv2, rightmost) -> start of row 2 (relu2, rightmost)
        const midY = fp.y + COMP_H + V_GAP / 2;
        pathD = `M ${fromX} ${fromY} L ${fromX + 15} ${fromY} L ${fromX + 15} ${midY} L ${tp.x + COMP_W + 15} ${midY} L ${tp.x + COMP_W + 15} ${toY} L ${tp.x + COMP_W} ${toY}`;
      } else {
        // End of row 2 (relu3, leftmost) -> start of row 3 (pool2, leftmost)
        const fx = fp.x; // row 2 reversed, output is on left
        const fy = fp.y + COMP_H / 2;
        const midY = fp.y + COMP_H + V_GAP / 2;
        pathD = `M ${fx} ${fy} L ${fx - 15} ${fy} L ${fx - 15} ${midY} L ${tp.x - 15} ${midY} L ${tp.x - 15} ${toY} L ${toX} ${toY}`;
      }
    }

    const wide = from === "pool2"; // after flatten, wider path
    traces.push({ from, to, pathD, wide });
  }
  return traces;
}

const TRACES = computeTraces();

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

function viridisColor(t: number): string {
  // Simplified viridis: dark blue -> teal -> yellow
  const clamped = Math.max(0, Math.min(1, t));
  const r = Math.floor(clamped < 0.5 ? clamped * 2 * 50 : 50 + (clamped - 0.5) * 2 * 205);
  const g = Math.floor(clamped < 0.5 ? 20 + clamped * 2 * 100 : 120 + (clamped - 0.5) * 2 * 135);
  const b = Math.floor(clamped < 0.5 ? 80 + clamped * 2 * 60 : 140 - (clamped - 0.5) * 2 * 140);
  return `rgb(${r},${g},${b})`;
}

function meanAbsValue(act: number[][][] | number[] | undefined): number {
  if (!act) return 0;
  let sum = 0;
  let count = 0;
  if (Array.isArray(act[0]) && Array.isArray((act[0] as number[][])[0])) {
    const a = act as number[][][];
    for (const ch of a) {
      for (const row of ch) {
        for (const v of row) {
          sum += Math.abs(v);
          count++;
        }
      }
    }
  } else if (!Array.isArray(act[0])) {
    const a = act as number[];
    for (const v of a) {
      sum += Math.abs(v);
      count++;
    }
  }
  return count > 0 ? sum / count : 0;
}

/** Get a 4x4 grid of average values from first 16 channels of a 3D activation */
function getMiniGrid(act: number[][][]): number[][] {
  const numCh = Math.min(16, act.length);
  const grid: number[][] = [[], [], [], []];
  for (let i = 0; i < 16; i++) {
    const row = Math.floor(i / 4);
    if (i < numCh) {
      // Average the entire channel
      const ch = act[i];
      let sum = 0;
      let count = 0;
      for (const r of ch) {
        for (const v of r) {
          sum += Math.abs(v);
          count++;
        }
      }
      grid[row].push(count > 0 ? sum / count : 0);
    } else {
      grid[row].push(0);
    }
  }
  // Normalize to 0-1
  let maxVal = 0;
  for (const row of grid) {
    for (const v of row) {
      if (v > maxVal) maxVal = v;
    }
  }
  if (maxVal > 0) {
    for (const row of grid) {
      for (let i = 0; i < row.length; i++) {
        row[i] /= maxVal;
      }
    }
  }
  return grid;
}

/** Get subsampled 16 values from a 1D activation */
function getDenseDots(act: number[]): number[] {
  const dots: number[] = [];
  const step = Math.max(1, Math.floor(act.length / 16));
  for (let i = 0; i < 16; i++) {
    const idx = Math.min(i * step, act.length - 1);
    dots.push(Math.abs(act[idx]));
  }
  // Normalize
  let maxVal = 0;
  for (const v of dots) {
    if (v > maxVal) maxVal = v;
  }
  if (maxVal > 0) {
    for (let i = 0; i < dots.length; i++) {
      dots[i] /= maxVal;
    }
  }
  return dots;
}

// ---------------------------------------------------------------------------
// Seven Segment Display Component
// ---------------------------------------------------------------------------

/** Segment map for digits and some letters. For simplicity, we handle digits + partial letter coverage. */
const SEVEN_SEG_MAP: Record<string, boolean[]> = {
  // Segments: [top, topRight, bottomRight, bottom, bottomLeft, topLeft, middle]
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
  K: [false, false, false, false, true, true, true], // approximate
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
  W: [false, true, true, true, true, true, true], // approximate
  X: [false, true, true, false, true, true, true], // same as H
  Y: [false, true, true, true, false, true, true],
  Z: [true, true, false, true, true, false, true],
};

function SevenSegmentDisplay({
  char,
  x,
  y,
  w,
  h,
}: {
  char: string;
  x: number;
  y: number;
  w: number;
  h: number;
}) {
  const upper = char.toUpperCase();
  const segments = SEVEN_SEG_MAP[upper] ?? [false, false, false, false, false, false, true];

  const segW = w * 0.7;
  const segH = h * 0.12;
  const cx = x + w / 2;
  const cy = y + h / 2;
  const halfH = h * 0.38;
  const halfW = segW / 2;

  // Segment positions: [x, y, width, height, rotation]
  const segDefs: [number, number, number, number][] = [
    [cx - halfW, y + h * 0.05, segW, segH],                  // top
    [cx + halfW - segH, cy - halfH, segH, h * 0.4],          // topRight
    [cx + halfW - segH, cy + 2, segH, h * 0.4],              // bottomRight
    [cx - halfW, y + h - h * 0.05 - segH, segW, segH],       // bottom
    [cx - halfW, cy + 2, segH, h * 0.4],                      // bottomLeft
    [cx - halfW, cy - halfH, segH, h * 0.4],                  // topLeft
    [cx - halfW, cy - segH / 2, segW, segH],                  // middle
  ];

  return (
    <g>
      {segDefs.map((def, i) => (
        <rect
          key={i}
          x={def[0]}
          y={def[1]}
          width={def[2]}
          height={def[3]}
          rx={1}
          fill={segments[i] ? SEGMENT_ON : SEGMENT_OFF}
          opacity={segments[i] ? 1 : 0.3}
        />
      ))}
    </g>
  );
}

// ---------------------------------------------------------------------------
// ComponentBlock
// ---------------------------------------------------------------------------

function ComponentBlock({
  def,
  pos,
  isSelected,
  hasData,
  miniGrid,
  denseDots,
  predictedChar,
  onSelect,
  onHover,
  isRow2,
}: {
  def: ComponentDef;
  pos: CompPos;
  isSelected: boolean;
  hasData: boolean;
  miniGrid: number[][] | null;
  denseDots: number[] | null;
  predictedChar: string | null;
  onSelect: (key: LayerKey) => void;
  onHover: (key: LayerKey | null, e?: ReactMouseEvent) => void;
  isRow2: boolean;
}) {
  const { x, y } = pos;

  // Component shape path based on type
  const renderShape = () => {
    switch (def.type) {
      case "conv": {
        // Op-amp triangle
        const triPoints = `${x + 10},${y + 5} ${x + 10},${y + COMP_H - 5} ${x + COMP_W - 10},${y + COMP_H / 2}`;
        return (
          <g>
            <polygon
              points={triPoints}
              fill={COMP_BODY}
              stroke={isSelected ? COPPER_BRIGHT : COPPER}
              strokeWidth={2}
            />
            {/* +/- labels */}
            <text x={x + 18} y={y + 25} fill={COPPER} fontSize={10} fontFamily="monospace">+</text>
            <text x={x + 18} y={y + COMP_H - 17} fill={COPPER} fontSize={10} fontFamily="monospace">-</text>
          </g>
        );
      }
      case "relu": {
        // Diode: triangle + bar
        const triW = COMP_W * 0.5;
        const triH = COMP_H * 0.6;
        const cx = x + COMP_W / 2;
        const cy = y + COMP_H / 2;
        const triPoints = `${cx - triW / 2},${cy - triH / 2} ${cx - triW / 2},${cy + triH / 2} ${cx + triW / 2},${cy}`;
        return (
          <g>
            <rect
              x={x + 3}
              y={y + 3}
              width={COMP_W - 6}
              height={COMP_H - 6}
              rx={4}
              fill={COMP_BODY}
              stroke={isSelected ? COPPER_BRIGHT : COPPER}
              strokeWidth={2}
            />
            <polygon points={triPoints} fill="none" stroke={COPPER} strokeWidth={1.5} />
            <line
              x1={cx + triW / 2}
              y1={cy - triH / 2}
              x2={cx + triW / 2}
              y2={cy + triH / 2}
              stroke={COPPER}
              strokeWidth={1.5}
            />
          </g>
        );
      }
      case "pool": {
        // Trapezoid (multiplexer)
        const inset = 15;
        const points = `${x + inset},${y + 5} ${x + COMP_W - inset},${y + 5} ${x + COMP_W - 5},${y + COMP_H - 5} ${x + 5},${y + COMP_H - 5}`;
        return (
          <polygon
            points={points}
            fill={COMP_BODY}
            stroke={isSelected ? COPPER_BRIGHT : COPPER}
            strokeWidth={2}
          />
        );
      }
      case "dense": {
        // Wide bus connector
        return (
          <g>
            <rect
              x={x + 3}
              y={y + 3}
              width={COMP_W - 6}
              height={COMP_H - 6}
              rx={3}
              fill={COMP_BODY}
              stroke={isSelected ? COPPER_BRIGHT : COPPER}
              strokeWidth={2}
            />
            {/* Pin marks along top and bottom */}
            {Array.from({ length: 8 }).map((_, i) => (
              <g key={i}>
                <rect
                  x={x + 12 + i * 15}
                  y={y + 3}
                  width={4}
                  height={6}
                  fill={COPPER}
                  opacity={0.6}
                />
                <rect
                  x={x + 12 + i * 15}
                  y={y + COMP_H - 9}
                  width={4}
                  height={6}
                  fill={COPPER}
                  opacity={0.6}
                />
              </g>
            ))}
          </g>
        );
      }
      case "output": {
        // Seven-segment display housing
        return (
          <rect
            x={x + 3}
            y={y + 3}
            width={COMP_W - 6}
            height={COMP_H - 6}
            rx={4}
            fill={COMP_BODY}
            stroke={isSelected ? COPPER_BRIGHT : COPPER}
            strokeWidth={2}
          />
        );
      }
      default: {
        // Input: IC chip rectangle
        return (
          <rect
            x={x + 3}
            y={y + 3}
            width={COMP_W - 6}
            height={COMP_H - 6}
            rx={3}
            fill={COMP_BODY}
            stroke={isSelected ? COPPER_BRIGHT : COPPER}
            strokeWidth={2}
          />
        );
      }
    }
  };

  // Input pins (left side) and output pins (right side)
  const pinCount = def.type === "input" ? 4 : def.type === "dense" ? 6 : 3;
  const inputSide = isRow2 ? "right" : "left";
  const outputSide = isRow2 ? "left" : "right";

  const renderPins = () => {
    const pins: React.ReactElement[] = [];
    for (let i = 0; i < pinCount; i++) {
      const pinY = y + 15 + i * ((COMP_H - 30) / Math.max(1, pinCount - 1));
      // Input pins
      if (inputSide === "left") {
        pins.push(
          <rect key={`in-${i}`} x={x - 6} y={pinY - 1.5} width={8} height={3} rx={0.5} fill={COPPER} />
        );
      } else {
        pins.push(
          <rect key={`in-${i}`} x={x + COMP_W - 2} y={pinY - 1.5} width={8} height={3} rx={0.5} fill={COPPER} />
        );
      }
      // Output pins
      if (outputSide === "right") {
        pins.push(
          <rect key={`out-${i}`} x={x + COMP_W - 2} y={pinY - 1.5} width={8} height={3} rx={0.5} fill={COPPER} />
        );
      } else {
        pins.push(
          <rect key={`out-${i}`} x={x - 6} y={pinY - 1.5} width={8} height={3} rx={0.5} fill={COPPER} />
        );
      }
    }
    return pins;
  };

  // Mini activation visualization inside component
  const renderMiniVis = () => {
    if (!hasData) return null;

    if (miniGrid && (def.type === "conv" || def.type === "relu" || def.type === "pool")) {
      const gridSize = 4;
      const cellSize = 8;
      const gx = x + COMP_W / 2 - (gridSize * cellSize) / 2;
      const gy = y + COMP_H / 2 - (gridSize * cellSize) / 2 + 2;
      return (
        <g>
          {miniGrid.map((row, ri) =>
            row.map((val, ci) => (
              <rect
                key={`${ri}-${ci}`}
                x={gx + ci * cellSize}
                y={gy + ri * cellSize}
                width={cellSize - 1}
                height={cellSize - 1}
                rx={1}
                fill={viridisColor(val)}
              />
            ))
          )}
        </g>
      );
    }

    if (denseDots && def.type === "dense") {
      const dotSize = 6;
      const gap = 1;
      const totalW = 16 * (dotSize + gap);
      const startX = x + COMP_W / 2 - totalW / 2;
      const dotY = y + COMP_H / 2 - dotSize / 2 + 2;
      return (
        <g>
          {denseDots.map((val, i) => (
            <circle
              key={i}
              cx={startX + i * (dotSize + gap) + dotSize / 2}
              cy={dotY + dotSize / 2}
              r={dotSize / 2}
              fill={viridisColor(val)}
            />
          ))}
        </g>
      );
    }

    if (def.type === "output" && predictedChar) {
      return (
        <text
          x={x + COMP_W / 2}
          y={y + COMP_H / 2 + 6}
          textAnchor="middle"
          fontSize={28}
          fontFamily="monospace"
          fontWeight="bold"
          fill={GOLD_PULSE}
        >
          {predictedChar}
        </text>
      );
    }

    return null;
  };

  return (
    <g
      className="cursor-pointer"
      onClick={() => onSelect(def.key)}
      onMouseEnter={(e) => onHover(def.key, e)}
      onMouseLeave={() => onHover(null)}
    >
      {/* Selection glow */}
      {isSelected && (
        <rect
          x={x - 4}
          y={y - 4}
          width={COMP_W + 8}
          height={COMP_H + 8}
          rx={6}
          fill="none"
          stroke={COPPER_BRIGHT}
          strokeWidth={2}
          opacity={0.6}
          filter="url(#glow)"
        />
      )}

      {/* Pins */}
      {renderPins()}

      {/* Component shape */}
      {renderShape()}

      {/* Label text */}
      {!hasData && (
        <text
          x={x + COMP_W / 2}
          y={y + COMP_H / 2 + 4}
          textAnchor="middle"
          fontSize={10}
          fontFamily="monospace"
          fill={COPPER}
        >
          {def.label}
        </text>
      )}

      {/* Mini activation visualization */}
      {renderMiniVis()}

      {/* Silk screen designator (white) */}
      <text
        x={x + COMP_W / 2}
        y={y - 5}
        textAnchor="middle"
        fontSize={9}
        fontFamily="monospace"
        fill="#ffffff"
        opacity={0.7}
      >
        {def.designator}
      </text>
    </g>
  );
}

// ---------------------------------------------------------------------------
// ICChipInput
// ---------------------------------------------------------------------------

function ICChipInput({
  x,
  y,
  inputTensor,
}: {
  x: number;
  y: number;
  inputTensor: number[][] | null;
}) {
  const chipW = COMP_W - 6;
  const chipH = COMP_H - 6;
  const chipX = x + 3;
  const chipY = y + 3;

  // IC notch
  const notchR = 6;

  // Pins on left and right (14 each = 28 total for 28 rows)
  const pinCount = 7; // show 7 per side for space
  const pinSpacing = (chipH - 10) / (pinCount - 1);

  const renderPins = () => {
    const pins: React.ReactElement[] = [];
    for (let i = 0; i < pinCount; i++) {
      const pinY = chipY + 5 + i * pinSpacing;
      // Left pins
      pins.push(
        <rect key={`l-${i}`} x={chipX - 8} y={pinY - 1.5} width={10} height={3} rx={0.5} fill={COPPER} />
      );
      // Right pins
      pins.push(
        <rect key={`r-${i}`} x={chipX + chipW - 2} y={pinY - 1.5} width={10} height={3} rx={0.5} fill={COPPER} />
      );
    }
    return pins;
  };

  // Render 28x28 input as tiny grid
  const renderInputGrid = () => {
    if (!inputTensor) {
      return (
        <text
          x={chipX + chipW / 2}
          y={chipY + chipH / 2 + 3}
          textAnchor="middle"
          fontSize={6}
          fontFamily="monospace"
          fill={COPPER}
          opacity={0.5}
        >
          AWAITING INPUT
        </text>
      );
    }

    const gridW = chipW - 16;
    const gridH = chipH - 16;
    const cellW = gridW / 28;
    const cellH = gridH / 28;
    const ox = chipX + 8;
    const oy = chipY + 8;

    // For performance, sample every 2 pixels
    const rects: React.ReactElement[] = [];
    for (let r = 0; r < 28; r += 2) {
      for (let c = 0; c < 28; c += 2) {
        const val = inputTensor[r]?.[c] ?? 0;
        if (val > 0.05) {
          const gray = Math.floor(val * 255);
          rects.push(
            <rect
              key={`${r}-${c}`}
              x={ox + c * cellW}
              y={oy + r * cellH}
              width={cellW * 2}
              height={cellH * 2}
              fill={`rgb(${gray},${gray},${gray})`}
            />
          );
        }
      }
    }
    return <g>{rects}</g>;
  };

  return (
    <g>
      {renderPins()}
      <rect
        x={chipX}
        y={chipY}
        width={chipW}
        height={chipH}
        rx={3}
        fill="#0a0a0a"
        stroke={COPPER}
        strokeWidth={2}
      />
      {/* IC notch */}
      <circle
        cx={chipX + notchR + 3}
        cy={chipY + 3}
        r={notchR / 2}
        fill="none"
        stroke="#444"
        strokeWidth={1}
      />
      {renderInputGrid()}
    </g>
  );
}

// ---------------------------------------------------------------------------
// LEDBarOutput
// ---------------------------------------------------------------------------

function LEDBarOutput({
  x,
  y,
  prediction,
  topPrediction,
}: {
  x: number;
  y: number;
  prediction: number[] | null;
  topPrediction: { classIndex: number; confidence: number } | null;
}) {
  const top15 = useMemo(() => {
    if (!prediction) return [];
    const indexed = prediction.map((prob, idx) => ({ prob, idx }));
    indexed.sort((a, b) => b.prob - a.prob);
    return indexed.slice(0, 10);
  }, [prediction]);

  const ledR = 6;
  const ledGap = 3;
  const startX = x + 5;
  const startY = y + 10;

  const getLEDColor = (prob: number, isTop: boolean) => {
    if (isTop) return LED_HIGH;
    if (prob > 0.3) return LED_HIGH;
    if (prob > 0.05) return LED_MED;
    if (prob > 0.01) return LED_DIM;
    return LED_OFF;
  };

  const predictedChar = topPrediction
    ? EMNIST_CLASSES[topPrediction.classIndex] ?? "?"
    : "";

  return (
    <g>
      {/* Component body */}
      <rect
        x={x + 3}
        y={y + 3}
        width={COMP_W - 6}
        height={COMP_H - 6}
        rx={4}
        fill={COMP_BODY}
        stroke={COPPER}
        strokeWidth={2}
      />

      {/* LED dots (vertically stacked in 2 columns for space) */}
      {top15.map(({ prob, idx }, i) => {
        const col = i < 5 ? 0 : 1;
        const row = i % 5;
        const cx = startX + col * 30 + ledR;
        const cy = startY + row * (ledR * 2 + ledGap) + ledR;
        const isTop = topPrediction?.classIndex === idx;
        const color = getLEDColor(prob, isTop);
        const r = isTop ? ledR + 2 : ledR;

        return (
          <g key={idx}>
            <circle
              cx={cx}
              cy={cy}
              r={r}
              fill={color}
              opacity={prob > 0.01 ? 1 : 0.4}
              filter={isTop ? "url(#ledGlow)" : undefined}
            />
            {/* Highlight dome */}
            <circle
              cx={cx - 1}
              cy={cy - 1}
              r={r * 0.4}
              fill="white"
              opacity={prob > 0.05 ? 0.3 : 0.05}
            />
            {/* Label */}
            <text
              x={cx}
              y={cy + r + 8}
              textAnchor="middle"
              fontSize={5}
              fontFamily="monospace"
              fill={COPPER}
              opacity={0.7}
            >
              {EMNIST_CLASSES[idx]}
            </text>
          </g>
        );
      })}

      {/* Seven-segment display for predicted char */}
      {predictedChar && (
        <SevenSegmentDisplay
          char={predictedChar}
          x={x + COMP_W - 48}
          y={y + 10}
          w={36}
          h={COMP_H - 25}
        />
      )}
    </g>
  );
}

// ---------------------------------------------------------------------------
// CopperTraces
// ---------------------------------------------------------------------------

function CopperTraces({
  pathRefs,
}: {
  pathRefs: React.MutableRefObject<Record<string, SVGPathElement | null>>;
}) {
  return (
    <g>
      {TRACES.map((trace) => (
        <path
          key={`${trace.from}-${trace.to}`}
          ref={(el) => {
            pathRefs.current[`${trace.from}-${trace.to}`] = el;
          }}
          d={trace.pathD}
          fill="none"
          stroke={COPPER}
          strokeWidth={trace.wide ? 5 : 3}
          strokeLinecap="round"
          strokeLinejoin="round"
          opacity={0.8}
        />
      ))}
    </g>
  );
}

// ---------------------------------------------------------------------------
// SignalPulseAnimation
// ---------------------------------------------------------------------------

function SignalPulses({
  pathRefs,
  hasActivation,
  isInferring,
  layerMeans,
}: {
  pathRefs: React.MutableRefObject<Record<string, SVGPathElement | null>>;
  hasActivation: boolean;
  isInferring: boolean;
  layerMeans: Record<string, number>;
}) {
  const pulsesRef = useRef<Record<string, number[]>>({});
  const circlesRef = useRef<Record<string, SVGCircleElement[]>>({});
  const pathLengthsRef = useRef<Record<string, number>>({});
  const initializedRef = useRef(false);

  // Initialize pulse offsets
  useEffect(() => {
    if (!hasActivation) {
      initializedRef.current = false;
      return;
    }
    if (initializedRef.current) return;

    const pulses: Record<string, number[]> = {};
    const lengths: Record<string, number> = {};

    for (const trace of TRACES) {
      const key = `${trace.from}-${trace.to}`;
      const pathEl = pathRefs.current[key];
      if (pathEl) {
        const len = pathEl.getTotalLength();
        lengths[key] = len;
        // 4 pulses per trace
        pulses[key] = [0, 0.25, 0.5, 0.75];
      }
    }

    pathLengthsRef.current = lengths;
    pulsesRef.current = pulses;
    initializedRef.current = true;
  }, [hasActivation, pathRefs]);

  const animate = useCallback(
    (dt: number) => {
      if (!hasActivation) return;

      const speedMultiplier = isInferring ? 2 : 1;

      for (const trace of TRACES) {
        const key = `${trace.from}-${trace.to}`;
        const pathEl = pathRefs.current[key];
        const offsets = pulsesRef.current[key];
        const len = pathLengthsRef.current[key];
        const circles = circlesRef.current[key];

        if (!pathEl || !offsets || !len || !circles) continue;

        const layerMean = layerMeans[trace.from] ?? 0.1;
        const baseSpeed = 0.0003 + layerMean * 0.001;

        for (let i = 0; i < offsets.length; i++) {
          offsets[i] += baseSpeed * dt * speedMultiplier;
          if (offsets[i] > 1) offsets[i] -= 1;

          const point = pathEl.getPointAtLength(offsets[i] * len);
          const circle = circles[i];
          if (circle) {
            circle.setAttribute("cx", String(point.x));
            circle.setAttribute("cy", String(point.y));
          }
        }
      }
    },
    [hasActivation, isInferring, layerMeans, pathRefs]
  );

  useAnimationFrame(animate, hasActivation);

  if (!hasActivation) return null;

  return (
    <g>
      {TRACES.map((trace) => {
        const key = `${trace.from}-${trace.to}`;
        return Array.from({ length: 4 }).map((_, i) => (
          <circle
            key={`${key}-${i}`}
            ref={(el) => {
              if (!circlesRef.current[key]) {
                circlesRef.current[key] = [];
              }
              if (el) {
                circlesRef.current[key][i] = el;
              }
            }}
            cx={0}
            cy={0}
            r={4}
            fill={GOLD_PULSE}
            opacity={isInferring ? 1 : 0.8}
            filter="url(#pulseGlow)"
          />
        ));
      })}
    </g>
  );
}

// ---------------------------------------------------------------------------
// DetailPanel
// ---------------------------------------------------------------------------

function DetailPanel({
  layerKey,
  onClose,
  posX,
  posY,
}: {
  layerKey: LayerKey;
  onClose: () => void;
  posX: number;
  posY: number;
}) {
  const layerActivations = useInferenceStore((s) => s.layerActivations);
  const prediction = useInferenceStore((s) => s.prediction);
  const inputTensor = useInferenceStore((s) => s.inputTensor);

  const def = COMPONENTS.find((c) => c.key === layerKey);
  const act = layerKey === "input" ? null : layerActivations[layerKey];

  // Compute detail visualization data
  const detailContent = useMemo(() => {
    if (layerKey === "input") {
      return { type: "input" as const, hasData: !!inputTensor };
    }
    if (layerKey === "output" && prediction) {
      const indexed = prediction.map((prob, idx) => ({ prob, idx }));
      indexed.sort((a, b) => b.prob - a.prob);
      return { type: "output" as const, top10: indexed.slice(0, 10) };
    }
    if (!act) return { type: "empty" as const };

    // 3D activation
    if (Array.isArray(act[0]) && Array.isArray((act[0] as number[][])[0])) {
      const a = act as number[][][];
      const grid = getMiniGrid(a);
      return { type: "conv" as const, grid, channels: a.length };
    }

    // 1D activation
    if (!Array.isArray(act[0])) {
      const a = act as number[];
      // Bar chart data: subsample to 32 values
      const step = Math.max(1, Math.floor(a.length / 32));
      const bars: number[] = [];
      let maxV = 0;
      for (let i = 0; i < 32 && i * step < a.length; i++) {
        const v = Math.abs(a[i * step]);
        bars.push(v);
        if (v > maxV) maxV = v;
      }
      if (maxV > 0) {
        for (let i = 0; i < bars.length; i++) bars[i] /= maxV;
      }
      return { type: "dense" as const, bars };
    }

    return { type: "empty" as const };
  }, [layerKey, act, prediction, inputTensor]);

  if (!def) return null;

  return (
    <div
      className="fixed z-50 overflow-hidden rounded-lg"
      style={{
        left: posX,
        top: posY,
        width: 280,
        background: "#0d2818",
        border: `2px solid ${COPPER}`,
        fontFamily: "monospace",
        color: COPPER,
        boxShadow: `0 4px 20px rgba(0,0,0,0.6), 0 0 10px ${COPPER}40`,
      }}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-3 py-2"
        style={{ borderBottom: `1px solid ${COPPER}40` }}
      >
        <div>
          <div className="text-sm font-bold" style={{ color: COPPER_BRIGHT }}>
            {def.designator}: {def.label}
          </div>
          <div className="text-xs opacity-60">Shape: {def.shape}</div>
        </div>
        <button
          onClick={onClose}
          className="flex h-6 w-6 items-center justify-center rounded text-sm hover:bg-white/10"
          style={{ color: COPPER }}
        >
          x
        </button>
      </div>

      {/* Detail content */}
      <div className="p-3">
        {detailContent.type === "empty" && (
          <div className="text-center text-xs opacity-50">No activation data</div>
        )}

        {detailContent.type === "input" && (
          <div className="text-center text-xs">
            {detailContent.hasData ? "28x28 grayscale input loaded" : "Awaiting input..."}
          </div>
        )}

        {detailContent.type === "conv" && (
          <div>
            <div className="mb-2 text-xs opacity-60">
              Channel averages ({detailContent.channels} channels)
            </div>
            <div className="flex flex-wrap gap-1">
              {detailContent.grid.map((row, ri) =>
                row.map((val, ci) => (
                  <div
                    key={`${ri}-${ci}`}
                    style={{
                      width: 14,
                      height: 14,
                      borderRadius: 2,
                      background: viridisColor(val),
                    }}
                  />
                ))
              )}
            </div>
          </div>
        )}

        {detailContent.type === "dense" && (
          <div>
            <div className="mb-2 text-xs opacity-60">Neuron activations (subsampled)</div>
            <div className="flex items-end gap-px" style={{ height: 60 }}>
              {detailContent.bars.map((v, i) => (
                <div
                  key={i}
                  style={{
                    width: 6,
                    height: Math.max(2, v * 56),
                    background: viridisColor(v),
                    borderRadius: 1,
                  }}
                />
              ))}
            </div>
          </div>
        )}

        {detailContent.type === "output" && (
          <div>
            <div className="mb-2 text-xs opacity-60">Top predictions</div>
            <div className="flex flex-col gap-1">
              {detailContent.top10.map(({ prob, idx }) => (
                <div key={idx} className="flex items-center gap-2 text-xs">
                  <span className="w-6 text-right font-bold" style={{ color: COPPER_BRIGHT }}>
                    {EMNIST_CLASSES[idx]}
                  </span>
                  <div className="h-2 flex-1 rounded-full" style={{ background: "#0a1f10" }}>
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${prob * 100}%`,
                        background: prob > 0.3 ? GOLD_PULSE : prob > 0.05 ? LED_MED : LED_DIM,
                      }}
                    />
                  </div>
                  <span className="w-10 text-right opacity-60">
                    {(prob * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// DrawingCanvas (Test Point)
// ---------------------------------------------------------------------------

function PCBDrawingCanvas() {
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

  const padSize = 8;

  return (
    <div className="fixed bottom-16 left-4 z-40 flex flex-col items-center gap-2">
      {/* Label */}
      <div
        className="text-xs tracking-widest"
        style={{
          fontFamily: "monospace",
          color: COPPER,
          textShadow: `0 0 4px ${COPPER}60`,
        }}
      >
        TEST POINT
      </div>

      {/* Canvas with solder pad corners */}
      <div className="relative">
        {/* Corner pads */}
        <div
          className="absolute -left-1 -top-1"
          style={{ width: padSize, height: padSize, background: COPPER, borderRadius: 1 }}
        />
        <div
          className="absolute -right-1 -top-1"
          style={{ width: padSize, height: padSize, background: COPPER, borderRadius: 1 }}
        />
        <div
          className="absolute -bottom-1 -left-1"
          style={{ width: padSize, height: padSize, background: COPPER, borderRadius: 1 }}
        />
        <div
          className="absolute -bottom-1 -right-1"
          style={{ width: padSize, height: padSize, background: COPPER, borderRadius: 1 }}
        />

        <div
          className="overflow-hidden rounded"
          style={{
            border: `2px solid ${isInferring ? COPPER_BRIGHT : COPPER}`,
            boxShadow: isInferring
              ? `0 0 12px ${COPPER_BRIGHT}80`
              : `0 0 6px ${COPPER}40`,
            transition: "border-color 0.3s, box-shadow 0.3s",
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

          {/* Hint overlay */}
          {!hasDrawn && (
            <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
              <p
                className="text-xs"
                style={{
                  fontFamily: "monospace",
                  color: `${COPPER}60`,
                }}
              >
                Draw here
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Clear button styled as PCB button */}
      <button
        onClick={handleClear}
        className="text-xs transition-all"
        style={{
          fontFamily: "monospace",
          color: COPPER,
          background: COMP_BODY,
          border: `1px solid ${COPPER}`,
          borderRadius: 3,
          padding: "3px 12px",
          boxShadow: `0 2px 0 ${COPPER}40`,
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.color = COPPER_BRIGHT;
          e.currentTarget.style.borderColor = COPPER_BRIGHT;
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.color = COPPER;
          e.currentTarget.style.borderColor = COPPER;
        }}
      >
        CLEAR
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tooltip
// ---------------------------------------------------------------------------

function Tooltip({
  layerKey,
  mouseX,
  mouseY,
}: {
  layerKey: LayerKey;
  mouseX: number;
  mouseY: number;
}) {
  const def = COMPONENTS.find((c) => c.key === layerKey);
  if (!def) return null;

  return (
    <div
      className="pointer-events-none fixed z-50 rounded px-2 py-1"
      style={{
        left: mouseX + 12,
        top: mouseY - 10,
        background: "#0d2818ee",
        border: `1px solid ${COPPER}`,
        fontFamily: "monospace",
        fontSize: 11,
        color: COPPER,
        whiteSpace: "nowrap",
      }}
    >
      {def.designator}: {def.label} [{def.shape}]
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Layout Export
// ---------------------------------------------------------------------------

export function CircuitBoardLayout() {
  const inputTensor = useInferenceStore((s) => s.inputTensor);
  const layerActivations = useInferenceStore((s) => s.layerActivations);
  const prediction = useInferenceStore((s) => s.prediction);
  const topPrediction = useInferenceStore((s) => s.topPrediction);
  const isInferring = useInferenceStore((s) => s.isInferring);

  const [selectedComponent, setSelectedComponent] = useState<LayerKey | null>(null);
  const [hoveredComponent, setHoveredComponent] = useState<LayerKey | null>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [detailPos, setDetailPos] = useState({ x: 0, y: 0 });

  // Pan & zoom state
  const [viewTransform, setViewTransform] = useState({ x: 0, y: 0, scale: 1 });
  const isPanning = useRef(false);
  const panStart = useRef({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  const pathRefs = useRef<Record<string, SVGPathElement | null>>({});

  // Compute layer means for pulse speed
  const layerMeans = useMemo(() => {
    const means: Record<string, number> = {};
    for (const key of LAYER_KEYS) {
      if (key === "input") {
        if (inputTensor) {
          let sum = 0;
          let count = 0;
          for (const row of inputTensor) {
            for (const v of row) {
              sum += Math.abs(v);
              count++;
            }
          }
          means[key] = count > 0 ? sum / count : 0;
        } else {
          means[key] = 0;
        }
      } else {
        means[key] = meanAbsValue(
          layerActivations[key] as number[][][] | number[] | undefined
        );
      }
    }
    return means;
  }, [layerActivations, inputTensor]);

  // Mini visualization data per component
  const miniVisData = useMemo(() => {
    const data: Record<
      string,
      { miniGrid: number[][] | null; denseDots: number[] | null }
    > = {};

    for (const comp of COMPONENTS) {
      const act = layerActivations[comp.key];
      if (!act) {
        data[comp.key] = { miniGrid: null, denseDots: null };
        continue;
      }

      if (
        (comp.type === "conv" || comp.type === "relu" || comp.type === "pool") &&
        Array.isArray(act[0]) &&
        Array.isArray((act[0] as number[][])[0])
      ) {
        data[comp.key] = {
          miniGrid: getMiniGrid(act as number[][][]),
          denseDots: null,
        };
      } else if (
        (comp.type === "dense" || comp.type === "relu") &&
        !Array.isArray(act[0])
      ) {
        data[comp.key] = {
          miniGrid: null,
          denseDots: getDenseDots(act as number[]),
        };
      } else {
        data[comp.key] = { miniGrid: null, denseDots: null };
      }
    }

    return data;
  }, [layerActivations]);

  const hasActivation = Object.keys(layerActivations).length > 0;

  const predictedChar = topPrediction
    ? EMNIST_CLASSES[topPrediction.classIndex] ?? null
    : null;

  // Handler: select component
  const handleSelect = useCallback((key: LayerKey) => {
    setSelectedComponent((prev) => {
      if (prev === key) return null;
      // Position detail panel near the component
      const pos = POSITIONS[key];
      setDetailPos({
        x: Math.min(window.innerWidth - 300, pos.x + COMP_W + 30),
        y: Math.max(10, pos.y - 20),
      });
      return key;
    });
  }, []);

  // Handler: hover component
  const handleHover = useCallback(
    (key: LayerKey | null, e?: ReactMouseEvent) => {
      setHoveredComponent(key);
      if (e) {
        setMousePos({ x: e.clientX, y: e.clientY });
      }
    },
    []
  );

  // Pan handlers
  const handleMouseDown = useCallback(
    (e: ReactMouseEvent<HTMLDivElement>) => {
      if ((e.target as HTMLElement).closest(".cursor-pointer")) return;
      isPanning.current = true;
      panStart.current = {
        x: e.clientX - viewTransform.x,
        y: e.clientY - viewTransform.y,
      };
    },
    [viewTransform.x, viewTransform.y]
  );

  const handleMouseMove = useCallback(
    (e: ReactMouseEvent<HTMLDivElement>) => {
      if (!isPanning.current) return;
      setViewTransform((prev) => ({
        ...prev,
        x: e.clientX - panStart.current.x,
        y: e.clientY - panStart.current.y,
      }));
    },
    []
  );

  const handleMouseUp = useCallback(() => {
    isPanning.current = false;
  }, []);

  // Zoom handler
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      const scaleDelta = e.deltaY > 0 ? 0.9 : 1.1;
      setViewTransform((prev) => {
        const newScale = Math.max(0.3, Math.min(3, prev.scale * scaleDelta));
        // Zoom toward mouse position
        const rect = container.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const scaleChange = newScale / prev.scale;
        return {
          x: mx - (mx - prev.x) * scaleChange,
          y: my - (my - prev.y) * scaleChange,
          scale: newScale,
        };
      });
    };

    container.addEventListener("wheel", handleWheel, { passive: false });
    return () => container.removeEventListener("wheel", handleWheel);
  }, []);

  // Center the board on mount
  useEffect(() => {
    const centerBoard = () => {
      const ww = window.innerWidth;
      const wh = window.innerHeight;
      const scale = Math.min(ww / (SVG_W + 40), wh / (SVG_H + 40), 1.5);
      setViewTransform({
        x: (ww - SVG_W * scale) / 2,
        y: (wh - SVG_H * scale) / 2 - 20,
        scale,
      });
    };
    centerBoard();
    window.addEventListener("resize", centerBoard);
    return () => window.removeEventListener("resize", centerBoard);
  }, []);

  return (
    <div
      ref={containerRef}
      className="relative overflow-hidden"
      style={{
        width: "100vw",
        height: "100vh",
        background: PCB_GREEN,
        // Perfboard dot grid via CSS
        backgroundImage: `radial-gradient(circle, ${COPPER}30 1px, transparent 1px)`,
        backgroundSize: "20px 20px",
        cursor: isPanning.current ? "grabbing" : "grab",
      }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      {/* SVG board */}
      <svg
        width={SVG_W}
        height={SVG_H}
        viewBox={`0 0 ${SVG_W} ${SVG_H}`}
        style={{
          transform: `translate(${viewTransform.x}px, ${viewTransform.y}px) scale(${viewTransform.scale})`,
          transformOrigin: "0 0",
          position: "absolute",
          top: 0,
          left: 0,
        }}
      >
        {/* SVG Defs: filters */}
        <defs>
          <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <filter id="pulseGlow" x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <filter id="ledGlow" x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Board outline */}
        <rect
          x={2}
          y={2}
          width={SVG_W - 4}
          height={SVG_H - 4}
          rx={8}
          fill="none"
          stroke={COPPER}
          strokeWidth={2}
          opacity={0.3}
        />

        {/* Mounting holes in corners */}
        {[
          [16, 16],
          [SVG_W - 16, 16],
          [16, SVG_H - 16],
          [SVG_W - 16, SVG_H - 16],
        ].map(([cx, cy], i) => (
          <g key={i}>
            <circle cx={cx} cy={cy} r={6} fill="none" stroke={COPPER} strokeWidth={1.5} opacity={0.4} />
            <circle cx={cx} cy={cy} r={3} fill={PCB_GREEN} stroke={COPPER} strokeWidth={0.5} opacity={0.3} />
          </g>
        ))}

        {/* Copper traces */}
        <CopperTraces pathRefs={pathRefs} />

        {/* Signal pulses */}
        <SignalPulses
          pathRefs={pathRefs}
          hasActivation={hasActivation}
          isInferring={isInferring}
          layerMeans={layerMeans}
        />

        {/* Components */}
        {COMPONENTS.map((comp) => {
          const pos = POSITIONS[comp.key];
          const isRow2 = pos.row === 1;
          const vis = miniVisData[comp.key] ?? { miniGrid: null, denseDots: null };

          if (comp.key === "input") {
            return (
              <g
                key={comp.key}
                className="cursor-pointer"
                onClick={() => handleSelect(comp.key)}
                onMouseEnter={(e) => handleHover(comp.key, e as unknown as ReactMouseEvent)}
                onMouseLeave={() => handleHover(null)}
              >
                {/* Selection glow */}
                {selectedComponent === comp.key && (
                  <rect
                    x={pos.x - 4}
                    y={pos.y - 4}
                    width={COMP_W + 8}
                    height={COMP_H + 8}
                    rx={6}
                    fill="none"
                    stroke={COPPER_BRIGHT}
                    strokeWidth={2}
                    opacity={0.6}
                    filter="url(#glow)"
                  />
                )}
                {/* Silk screen designator */}
                <text
                  x={pos.x + COMP_W / 2}
                  y={pos.y - 5}
                  textAnchor="middle"
                  fontSize={9}
                  fontFamily="monospace"
                  fill="#ffffff"
                  opacity={0.7}
                >
                  {comp.designator}
                </text>
                <ICChipInput x={pos.x} y={pos.y} inputTensor={inputTensor} />
              </g>
            );
          }

          if (comp.key === "output") {
            return (
              <g
                key={comp.key}
                className="cursor-pointer"
                onClick={() => handleSelect(comp.key)}
                onMouseEnter={(e) => handleHover(comp.key, e as unknown as ReactMouseEvent)}
                onMouseLeave={() => handleHover(null)}
              >
                {selectedComponent === comp.key && (
                  <rect
                    x={pos.x - 4}
                    y={pos.y - 4}
                    width={COMP_W + 8}
                    height={COMP_H + 8}
                    rx={6}
                    fill="none"
                    stroke={COPPER_BRIGHT}
                    strokeWidth={2}
                    opacity={0.6}
                    filter="url(#glow)"
                  />
                )}
                <text
                  x={pos.x + COMP_W / 2}
                  y={pos.y - 5}
                  textAnchor="middle"
                  fontSize={9}
                  fontFamily="monospace"
                  fill="#ffffff"
                  opacity={0.7}
                >
                  {comp.designator}
                </text>
                <LEDBarOutput
                  x={pos.x}
                  y={pos.y}
                  prediction={prediction}
                  topPrediction={topPrediction}
                />
              </g>
            );
          }

          return (
            <ComponentBlock
              key={comp.key}
              def={comp}
              pos={pos}
              isSelected={selectedComponent === comp.key}
              hasData={!!layerActivations[comp.key]}
              miniGrid={vis.miniGrid}
              denseDots={vis.denseDots}
              predictedChar={predictedChar}
              onSelect={handleSelect}
              onHover={handleHover}
              isRow2={isRow2}
            />
          );
        })}

        {/* Board title silk screen */}
        <text
          x={SVG_W / 2}
          y={SVG_H - 12}
          textAnchor="middle"
          fontSize={10}
          fontFamily="monospace"
          fill="#ffffff"
          opacity={0.5}
        >
          OCR-NET v1.0 REV A
        </text>
      </svg>

      {/* Drawing canvas overlay */}
      <PCBDrawingCanvas />

      {/* Hover tooltip */}
      {hoveredComponent && (
        <Tooltip
          layerKey={hoveredComponent}
          mouseX={mousePos.x}
          mouseY={mousePos.y}
        />
      )}

      {/* Detail panel */}
      {selectedComponent && (
        <DetailPanel
          layerKey={selectedComponent}
          onClose={() => setSelectedComponent(null)}
          posX={detailPos.x}
          posY={detailPos.y}
        />
      )}

      {/* Layout navigation */}
      <LayoutNav />
    </div>
  );
}
