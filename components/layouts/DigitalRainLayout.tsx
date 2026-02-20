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
// Types
// ---------------------------------------------------------------------------

interface RainDrop {
  y: number;
  speed: number;
  chars: string[];
  charIndex: number;
  brightness: number;
}

interface ColumnState {
  drops: RainDrop[];
  nextSpawn: number;
}

interface CharPixel {
  x: number;
  y: number;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CHAR_SIZE = 14;
const COLUMN_WIDTH = CHAR_SIZE;
const TRAIL_LENGTH = 20;
const SPAWN_RATE_MIN = 10;
const SPAWN_RATE_MAX = 40;
const SPEED_MIN = 1.5;
const SPEED_MAX = 3.5;
const FADE_ALPHA = 0.05;
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
  relu1: "#00e5ff",
  conv2: "#448aff",
  relu2: "#448aff",
  pool1: "#536dfe",
  conv3: "#7c4dff",
  relu3: "#7c4dff",
  pool2: "#7c4dff",
  dense1: "#e040fb",
  relu4: "#e040fb",
  output: "#ffab00",
};

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

function randomChar(): string {
  return EMNIST_CLASSES[Math.floor(Math.random() * EMNIST_CLASSES.length)];
}

function randomBetween(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

function hexToRgb(hex: string): { r: number; g: number; b: number } {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  if (!result) return { r: 0, g: 255, b: 65 };
  return {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16),
  };
}

/** Average across channels for a 3D activation [C][H][W] */
function averageChannels(activation: number[][][]): number[][] {
  const channels = activation.length;
  if (channels === 0) return [];
  const h = activation[0].length;
  const w = activation[0][0].length;
  const result: number[][] = Array.from({ length: h }, () =>
    new Array(w).fill(0)
  );
  for (let c = 0; c < channels; c++) {
    for (let row = 0; row < h; row++) {
      for (let col = 0; col < w; col++) {
        result[row][col] += activation[c][row][col];
      }
    }
  }
  for (let row = 0; row < h; row++) {
    for (let col = 0; col < w; col++) {
      result[row][col] /= channels;
    }
  }
  return result;
}

/** Compute mean absolute value of an activation */
function meanAbsValue(
  act: number[][][] | number[] | undefined
): number {
  if (!act) return 0;
  let sum = 0;
  let count = 0;
  if (Array.isArray(act[0]) && Array.isArray((act[0] as number[][])[0])) {
    // 3D
    const a = act as number[][][];
    for (const ch of a) {
      for (const row of ch) {
        for (const v of row) {
          sum += Math.abs(v);
          count++;
        }
      }
    }
  } else if (Array.isArray(act[0])) {
    // Should not happen for our layers, but handle 2D
    const a = act as unknown as number[][];
    for (const row of a) {
      for (const v of row) {
        sum += Math.abs(v);
        count++;
      }
    }
  } else {
    // 1D
    const a = act as number[];
    for (const v of a) {
      sum += Math.abs(v);
      count++;
    }
  }
  return count > 0 ? sum / count : 0;
}

/** Get characters from a char pixel map using offscreen canvas */
function getCharPixels(
  char: string,
  size: number
): CharPixel[] {
  if (typeof document === "undefined") return [];
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  if (!ctx) return [];

  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, size, size);
  ctx.fillStyle = "#ffffff";
  ctx.font = `bold ${Math.floor(size * 0.8)}px monospace`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(char, size / 2, size / 2);

  const imageData = ctx.getImageData(0, 0, size, size);
  const pixels: CharPixel[] = [];
  const step = 3; // sample every 3 pixels for performance
  for (let y = 0; y < size; y += step) {
    for (let x = 0; x < size; x += step) {
      const idx = (y * size + x) * 4;
      if (imageData.data[idx] > 128) {
        pixels.push({ x, y });
      }
    }
  }
  return pixels;
}

// ---------------------------------------------------------------------------
// Sub-component: DigitalRainCanvas
// ---------------------------------------------------------------------------

function DigitalRainCanvas({
  selectedLayer,
}: {
  selectedLayer: LayerKey;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const columnsRef = useRef<ColumnState[]>([]);
  const dimensionsRef = useRef({ width: 0, height: 0 });
  const charPixelsRef = useRef<CharPixel[]>([]);
  const assemblyOffsetRef = useRef({ x: 0, y: 0 });

  const inputTensor = useInferenceStore((s) => s.inputTensor);
  const layerActivations = useInferenceStore((s) => s.layerActivations);
  const topPrediction = useInferenceStore((s) => s.topPrediction);

  // Compute the predicted character string
  const predictedChar = useMemo(() => {
    if (!topPrediction) return null;
    return EMNIST_CLASSES[topPrediction.classIndex] ?? null;
  }, [topPrediction]);

  // Compute character assembly pixels when prediction changes
  useEffect(() => {
    if (predictedChar) {
      charPixelsRef.current = getCharPixels(predictedChar, 200);
      // Center the assembly effect
      const w = dimensionsRef.current.width;
      const h = dimensionsRef.current.height;
      assemblyOffsetRef.current = {
        x: (w - 200) / 2,
        y: (h - 200) / 2,
      };
    } else {
      charPixelsRef.current = [];
    }
  }, [predictedChar]);

  // Get activation-based characters for the selected layer
  const getActivationData = useCallback((): {
    grid: number[][] | null;
    flat: number[] | null;
  } => {
    if (selectedLayer === "input") {
      return { grid: inputTensor, flat: null };
    }
    const act = layerActivations[selectedLayer];
    if (!act) return { grid: null, flat: null };

    // Check if 3D (conv/pool layers)
    if (
      Array.isArray(act[0]) &&
      Array.isArray((act[0] as number[][])[0])
    ) {
      return { grid: averageChannels(act as number[][][]), flat: null };
    }
    // 1D (dense layers)
    return { grid: null, flat: act as number[] };
  }, [selectedLayer, inputTensor, layerActivations]);

  // Handle resize
  useEffect(() => {
    function handleResize() {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const w = window.innerWidth;
      const h = window.innerHeight;
      canvas.width = w;
      canvas.height = h;
      dimensionsRef.current = { width: w, height: h };

      // Reinitialize columns
      const numCols = Math.ceil(w / COLUMN_WIDTH);
      const cols: ColumnState[] = [];
      for (let i = 0; i < numCols; i++) {
        cols.push({
          drops: [],
          nextSpawn: Math.floor(randomBetween(0, SPAWN_RATE_MAX)),
        });
      }
      columnsRef.current = cols;

      // Recenter assembly
      assemblyOffsetRef.current = {
        x: (w - 200) / 2,
        y: (h - 200) / 2,
      };
    }
    handleResize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // Get the layer color
  const layerColor = useMemo(() => {
    return hexToRgb(LAYER_COLORS[selectedLayer]);
  }, [selectedLayer]);

  // Main render loop
  const render = useCallback(
    () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const { width, height } = dimensionsRef.current;
      if (width === 0 || height === 0) return;

      const columns = columnsRef.current;
      const numCols = columns.length;
      const { r, g, b } = layerColor;

      // Fade trail effect
      ctx.fillStyle = `rgba(0, 0, 0, ${FADE_ALPHA})`;
      ctx.fillRect(0, 0, width, height);

      ctx.font = `${CHAR_SIZE}px monospace`;
      ctx.textAlign = "center";

      // Get activation data for character selection
      const { grid, flat } = getActivationData();

      // Assembly attraction data
      const charPx = charPixelsRef.current;
      const assemblyX = assemblyOffsetRef.current.x;
      const assemblyY = assemblyOffsetRef.current.y;
      const hasAssembly = charPx.length > 0 && topPrediction !== null;
      const confidence = topPrediction?.confidence ?? 0;

      // Update and render each column
      for (let col = 0; col < numCols; col++) {
        const colState = columns[col];

        // Spawn new drops
        colState.nextSpawn--;
        if (colState.nextSpawn <= 0) {
          const chars: string[] = [];
          for (let i = 0; i < TRAIL_LENGTH; i++) {
            chars.push(randomChar());
          }
          colState.drops.push({
            y: -CHAR_SIZE * TRAIL_LENGTH,
            speed: randomBetween(SPEED_MIN, SPEED_MAX),
            chars,
            charIndex: 0,
            brightness: 1.0,
          });
          colState.nextSpawn = Math.floor(
            randomBetween(SPAWN_RATE_MIN, SPAWN_RATE_MAX)
          );
        }

        // Render drops
        const x = col * COLUMN_WIDTH + COLUMN_WIDTH / 2;

        for (let d = colState.drops.length - 1; d >= 0; d--) {
          const drop = colState.drops[d];

          // If there's activation data, pick character/brightness from it
          let dropChar = drop.chars[drop.charIndex % drop.chars.length];
          let brightnessMultiplier = 1.0;

          if (grid) {
            const gridH = grid.length;
            const gridW = grid[0]?.length ?? 0;
            if (gridH > 0 && gridW > 0) {
              const gridCol = col % gridW;
              const gridRow =
                Math.floor(drop.y / CHAR_SIZE) % gridH;
              const normalizedRow =
                ((gridRow % gridH) + gridH) % gridH;
              const val = grid[normalizedRow]?.[gridCol] ?? 0;
              brightnessMultiplier = Math.min(1, Math.abs(val));
              // Map value to character index
              const charIdx = Math.floor(
                Math.abs(val) * (EMNIST_CLASSES.length - 1)
              );
              dropChar =
                EMNIST_CLASSES[
                  Math.min(charIdx, EMNIST_CLASSES.length - 1)
                ];
            }
          } else if (flat) {
            const flatIdx =
              (col * 16 + Math.floor(drop.y / CHAR_SIZE)) % flat.length;
            const normalizedIdx =
              ((flatIdx % flat.length) + flat.length) % flat.length;
            const val = flat[normalizedIdx] ?? 0;
            brightnessMultiplier = Math.min(1, Math.abs(val));
            const charIdx = Math.floor(
              Math.abs(val) * (EMNIST_CLASSES.length - 1)
            );
            dropChar =
              EMNIST_CLASSES[
                Math.min(charIdx, EMNIST_CLASSES.length - 1)
              ];
          }

          // Assembly attraction effect: slow drops and bend them toward char pixels
          let drawX = x;
          let drawY = drop.y;
          let speedMod = 1.0;

          if (hasAssembly) {
            // Find nearest char pixel
            let minDist = Infinity;
            let nearestPx: CharPixel | null = null;
            for (const px of charPx) {
              const px_x = assemblyX + px.x;
              const px_y = assemblyY + px.y;
              const dist = Math.sqrt(
                (x - px_x) ** 2 + (drop.y - px_y) ** 2
              );
              if (dist < minDist) {
                minDist = dist;
                nearestPx = px;
              }
            }

            if (nearestPx && minDist < 150) {
              const attractStrength =
                (1 - minDist / 150) * confidence * 0.3;
              drawX +=
                (assemblyX + nearestPx.x - x) * attractStrength;
              drawY +=
                (assemblyY + nearestPx.y - drop.y) *
                attractStrength *
                0.1;
              speedMod = 1.0 - attractStrength * 0.5;
            }
          }

          // Render the trail
          const headY = drawY;
          for (let t = 0; t < TRAIL_LENGTH; t++) {
            const trailY = headY - t * CHAR_SIZE;
            if (trailY < -CHAR_SIZE || trailY > height + CHAR_SIZE)
              continue;

            const trailAlpha =
              (1 - t / TRAIL_LENGTH) * brightnessMultiplier;
            if (t === 0) {
              // Head character: bright white-ish
              ctx.fillStyle = `rgba(${Math.min(255, r + 100)}, ${Math.min(255, g + 100)}, ${Math.min(255, b + 100)}, ${trailAlpha})`;
              ctx.shadowColor = `rgb(${r}, ${g}, ${b})`;
              ctx.shadowBlur = 8;
            } else {
              ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${trailAlpha * 0.7})`;
              ctx.shadowBlur = 0;
            }

            const c =
              drop.chars[
                (drop.charIndex + t) % drop.chars.length
              ];
            ctx.fillText(c || dropChar, drawX, trailY);
          }
          ctx.shadowBlur = 0;

          // Update position
          drop.y += drop.speed * speedMod;

          // Cycle characters occasionally
          if (Math.random() < 0.02) {
            drop.charIndex =
              (drop.charIndex + 1) % drop.chars.length;
            // Also randomize a char in the trail
            const ri = Math.floor(Math.random() * drop.chars.length);
            drop.chars[ri] = randomChar();
          }

          // Remove drops that have gone offscreen
          if (drop.y - TRAIL_LENGTH * CHAR_SIZE > height) {
            colState.drops.splice(d, 1);
          }
        }
      }

      // Draw the assembled character glow at center
      if (hasAssembly && predictedChar) {
        const cx = width / 2;
        const cy = height / 2;
        const glowIntensity = confidence;
        const glowColor = LAYER_COLORS.output;

        ctx.save();
        ctx.font = "bold 200px monospace";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.shadowColor = glowColor;
        ctx.shadowBlur = 40 * glowIntensity;
        ctx.fillStyle = `rgba(255, 171, 0, ${0.3 + glowIntensity * 0.7})`;
        ctx.fillText(predictedChar, cx, cy);
        // Double pass for extra glow
        ctx.shadowBlur = 20 * glowIntensity;
        ctx.fillText(predictedChar, cx, cy);
        ctx.restore();
      }
    },
    [layerColor, getActivationData, topPrediction, predictedChar]
  );

  useAnimationFrame(render, true);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0"
      style={{ width: "100%", height: "100%" }}
    />
  );
}

// ---------------------------------------------------------------------------
// Sub-component: RainLayerScrubber
// ---------------------------------------------------------------------------

function RainLayerScrubber({
  selectedLayer,
  onSelectLayer,
}: {
  selectedLayer: LayerKey;
  onSelectLayer: (layer: LayerKey) => void;
}) {
  const layerActivations = useInferenceStore((s) => s.layerActivations);
  const inputTensor = useInferenceStore((s) => s.inputTensor);

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

  // Normalize for display
  const maxMean = useMemo(() => {
    let m = 0;
    for (const key of LAYER_KEYS) {
      if (layerMeans[key] > m) m = layerMeans[key];
    }
    return m || 1;
  }, [layerMeans]);

  return (
    <div className="fixed right-3 top-1/2 z-40 flex -translate-y-1/2 flex-col gap-1.5">
      <div
        className="rounded-xl border border-white/10 p-2"
        style={{
          background: "rgba(0, 0, 0, 0.7)",
          backdropFilter: "blur(12px)",
          WebkitBackdropFilter: "blur(12px)",
        }}
      >
        <div
          className="mb-2 text-center font-mono text-[10px] tracking-wider"
          style={{ color: "#00ff41" }}
        >
          LAYERS
        </div>
        {LAYER_KEYS.map((key) => {
          const isSelected = key === selectedLayer;
          const color = LAYER_COLORS[key];
          const barWidth = (layerMeans[key] / maxMean) * 40;

          return (
            <button
              key={key}
              onClick={() => onSelectLayer(key)}
              className="flex w-full items-center gap-1.5 rounded-md px-1.5 py-1 text-left font-mono text-[10px] transition-all"
              style={{
                color: isSelected ? color : "rgba(255,255,255,0.4)",
                background: isSelected
                  ? "rgba(255,255,255,0.08)"
                  : "transparent",
                boxShadow: isSelected
                  ? `0 0 8px ${color}40, inset 0 0 8px ${color}20`
                  : "none",
                border: isSelected
                  ? `1px solid ${color}60`
                  : "1px solid transparent",
              }}
            >
              <span className="w-[42px] shrink-0 truncate">
                {LAYER_DISPLAY_NAMES[key]}
              </span>
              <div
                className="h-1.5 rounded-full"
                style={{
                  width: `${Math.max(2, barWidth)}px`,
                  background: color,
                  opacity: layerMeans[key] > 0 ? 0.8 : 0.15,
                }}
              />
            </button>
          );
        })}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-component: ConfidenceOracle
// ---------------------------------------------------------------------------

function ConfidenceOracle() {
  const prediction = useInferenceStore((s) => s.prediction);
  const topPrediction = useInferenceStore((s) => s.topPrediction);

  const top10 = useMemo(() => {
    if (!prediction) return [];
    const indexed = prediction.map((prob, idx) => ({ prob, idx }));
    indexed.sort((a, b) => b.prob - a.prob);
    return indexed.slice(0, 10);
  }, [prediction]);

  if (top10.length === 0) return null;

  const maxProb = top10[0]?.prob ?? 1;

  return (
    <div className="fixed bottom-14 left-1/2 z-40 flex -translate-x-1/2 items-end gap-3">
      <div
        className="flex items-end gap-2 rounded-xl border border-white/10 px-4 py-3"
        style={{
          background: "rgba(0, 0, 0, 0.7)",
          backdropFilter: "blur(12px)",
          WebkitBackdropFilter: "blur(12px)",
        }}
      >
        {top10.map(({ prob, idx }) => {
          const char = EMNIST_CLASSES[idx] ?? "?";
          const isTop = topPrediction?.classIndex === idx;
          const scale = 0.5 + (prob / maxProb) * 0.5;
          const fontSize = Math.max(14, Math.floor(36 * scale));
          const color = isTop ? "#ffab00" : "#00ff41";
          const pct = (prob * 100).toFixed(1);

          return (
            <div
              key={idx}
              className="flex flex-col items-center"
            >
              <span
                className="font-mono font-bold"
                style={{
                  fontSize: `${fontSize}px`,
                  color,
                  textShadow: `0 0 ${isTop ? 20 : 8}px ${color}, 0 0 ${isTop ? 40 : 16}px ${color}80`,
                }}
              >
                {char}
              </span>
              <span
                className="mt-0.5 font-mono text-[9px]"
                style={{ color: `${color}99` }}
              >
                {pct}%
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-component: MatrixDrawingCanvas
// ---------------------------------------------------------------------------

function MatrixDrawingCanvas() {
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
      strokeColor: "#00ff41",
      backgroundColor: "#000000",
      onStrokeEnd,
    });

  const handleClear = useCallback(() => {
    clear();
    reset();
  }, [clear, reset]);

  return (
    <div className="fixed left-4 top-4 z-40 flex flex-col items-center gap-2">
      <div
        className="relative overflow-hidden rounded-2xl"
        style={{
          border: `2px solid ${isInferring ? "#00ff41" : "rgba(0, 255, 65, 0.3)"}`,
          boxShadow: isInferring
            ? "0 0 20px rgba(0, 255, 65, 0.5), inset 0 0 20px rgba(0, 255, 65, 0.1)"
            : "0 0 10px rgba(0, 255, 65, 0.15)",
          background: "rgba(0, 0, 0, 0.85)",
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

        {/* Draw hint overlay */}
        {!hasDrawn && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
            <p
              className="font-mono text-sm"
              style={{ color: "rgba(0, 255, 65, 0.3)" }}
            >
              Draw here
            </p>
          </div>
        )}
      </div>

      <button
        onClick={handleClear}
        className="font-mono text-xs transition-colors"
        style={{
          color: "rgba(0, 255, 65, 0.6)",
          background: "rgba(0, 0, 0, 0.6)",
          border: "1px solid rgba(0, 255, 65, 0.2)",
          borderRadius: "6px",
          padding: "4px 12px",
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.color = "#00ff41";
          e.currentTarget.style.borderColor = "rgba(0, 255, 65, 0.5)";
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.color = "rgba(0, 255, 65, 0.6)";
          e.currentTarget.style.borderColor = "rgba(0, 255, 65, 0.2)";
        }}
      >
        [X] Clear
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main layout export
// ---------------------------------------------------------------------------

export function DigitalRainLayout() {
  const [selectedLayer, setSelectedLayer] = useState<LayerKey>("input");

  return (
    <div
      className="relative overflow-hidden"
      style={{
        width: "100vw",
        height: "100vh",
        background: "#000000",
      }}
    >
      {/* Full-screen rain canvas */}
      <DigitalRainCanvas selectedLayer={selectedLayer} />

      {/* Drawing pad - top left */}
      <MatrixDrawingCanvas />

      {/* Layer scrubber - right side */}
      <RainLayerScrubber
        selectedLayer={selectedLayer}
        onSelectLayer={setSelectedLayer}
      />

      {/* Confidence oracle - bottom center */}
      <ConfidenceOracle />

      {/* Layout navigation */}
      <LayoutNav />
    </div>
  );
}
