"use client";

import {
  useState,
  useEffect,
  useRef,
  useCallback,
  useMemo,
  type CSSProperties,
} from "react";
import { useInferenceStore } from "@/stores/inferenceStore";
import { EMNIST_CLASSES, BYMERGE_MERGED_INDICES } from "@/lib/model/classes";
import { useDrawingCanvas } from "@/hooks/useDrawingCanvas";
import { useInference } from "@/hooks/useInference";
import { LayoutNav } from "@/components/layouts/LayoutNav";

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

const PHOSPHOR_GREEN = "#33ff33";
const DARK_GREEN = "#1a5a1a";
const BG_COLOR = "#0a0a0a";
const TEXT_GLOW = `0 0 5px #33ff3355`;

const ASCII_CHARS = " .:-=+*#%@"; // 10 levels
const SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"; // 8 levels

const MONO_STYLE: CSSProperties = {
  fontFamily: "'Courier New', Courier, monospace",
  color: PHOSPHOR_GREEN,
  textShadow: TEXT_GLOW,
};

// Layers we show in the architecture diagram
const ARCH_LAYERS = [
  { key: "input", label: "Input", shape: "28x28x1" },
  { key: "conv1", label: "Conv1", shape: "28x28x32" },
  { key: "relu1", label: "ReLU1", shape: "28x28x32" },
  { key: "conv2", label: "Conv2", shape: "28x28x64" },
  { key: "relu2", label: "ReLU2", shape: "28x28x64" },
  { key: "pool1", label: "Pool1", shape: "14x14x64" },
  { key: "conv3", label: "Conv3", shape: "14x14x128" },
  { key: "relu3", label: "ReLU3", shape: "14x14x128" },
  { key: "pool2", label: "Pool2", shape: "7x7x128" },
  { key: "dense1", label: "Dense1", shape: "256" },
  { key: "relu4", label: "ReLU4", shape: "256" },
  { key: "output", label: "Output", shape: "62" },
] as const;

// ─────────────────────────────────────────────────────────────────────────────
// Utility helpers
// ─────────────────────────────────────────────────────────────────────────────

function valueToAscii(v: number): string {
  const idx = Math.min(ASCII_CHARS.length - 1, Math.max(0, Math.round(v * (ASCII_CHARS.length - 1))));
  return ASCII_CHARS[idx];
}

function valueToSparkline(v: number): string {
  const idx = Math.min(
    SPARKLINE_CHARS.length - 1,
    Math.max(0, Math.round(v * (SPARKLINE_CHARS.length - 1)))
  );
  return SPARKLINE_CHARS[idx];
}

/** Normalize a 1D array to [0,1] range */
function normalize1D(arr: number[]): number[] {
  if (arr.length === 0) return [];
  let mn = arr[0],
    mx = arr[0];
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] < mn) mn = arr[i];
    if (arr[i] > mx) mx = arr[i];
  }
  const range = mx - mn || 1;
  return arr.map((v) => (v - mn) / range);
}

/** Normalize a 2D array to [0,1] range */
function normalize2D(arr: number[][]): number[][] {
  let mn = Infinity,
    mx = -Infinity;
  for (const row of arr) {
    for (const v of row) {
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
  }
  const range = mx - mn || 1;
  return arr.map((row) => row.map((v) => (v - mn) / range));
}

/** Average a 3D [channels][h][w] tensor across channels to get [h][w] */
function averageChannels(tensor: number[][][]): number[][] {
  const channels = tensor.length;
  if (channels === 0) return [];
  const h = tensor[0].length;
  const w = tensor[0][0].length;
  const result: number[][] = Array.from({ length: h }, () => new Array<number>(w).fill(0));
  for (let c = 0; c < channels; c++) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        result[y][x] += tensor[c][y][x];
      }
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      result[y][x] /= channels;
    }
  }
  return result;
}

/** Compute per-channel mean for a 3D activation [ch][h][w] */
function channelMeans(tensor: number[][][]): number[] {
  return tensor.map((ch) => {
    let sum = 0;
    let count = 0;
    for (const row of ch) {
      for (const v of row) {
        sum += v;
        count++;
      }
    }
    return count > 0 ? sum / count : 0;
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// CRT Effect Overlay
// ─────────────────────────────────────────────────────────────────────────────

function CRTEffectOverlay() {
  return (
    <>
      <style>{`
        @keyframes crt-flicker {
          0%, 100% { filter: brightness(1.0); }
          50% { filter: brightness(0.97); }
        }
        @keyframes crt-poweron {
          0% {
            clip-path: inset(49.5% 0 49.5% 0);
            opacity: 1;
          }
          40% {
            clip-path: inset(49% 0 49% 0);
            opacity: 1;
          }
          100% {
            clip-path: inset(0 0 0 0);
            opacity: 1;
          }
        }
        .crt-container {
          animation: crt-poweron 400ms ease-out forwards, crt-flicker 4s ease-in-out infinite 400ms;
        }
      `}</style>
      <div
        style={{
          position: "fixed",
          inset: 0,
          pointerEvents: "none",
          zIndex: 1000,
          background: [
            "repeating-linear-gradient(0deg, rgba(0,0,0,0.08) 0px, rgba(0,0,0,0.08) 1px, transparent 1px, transparent 3px)",
            "radial-gradient(ellipse at center, transparent 60%, rgba(0,0,0,0.5) 100%)",
          ].join(", "),
        }}
      />
    </>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ASCII Renderer
// ─────────────────────────────────────────────────────────────────────────────

function ASCIIRenderer({
  data,
  enhanced = false,
  fontSize = 6,
}: {
  data: number[][];
  enhanced?: boolean;
  fontSize?: number;
}) {
  const normalized = useMemo(() => normalize2D(data), [data]);

  const text = useMemo(() => {
    return normalized.map((row) => row.map((v) => valueToAscii(v)).join("")).join("\n");
  }, [normalized]);

  if (!enhanced) {
    return (
      <pre
        style={{
          ...MONO_STYLE,
          fontSize: `${fontSize}px`,
          lineHeight: `${fontSize + 1}px`,
          margin: 0,
          whiteSpace: "pre",
          letterSpacing: `${Math.max(0, fontSize - 5)}px`,
        }}
      >
        {text}
      </pre>
    );
  }

  // Enhanced mode: per-character opacity
  return (
    <pre
      style={{
        ...MONO_STYLE,
        fontSize: `${fontSize}px`,
        lineHeight: `${fontSize + 1}px`,
        margin: 0,
        whiteSpace: "pre",
        letterSpacing: `${Math.max(0, fontSize - 5)}px`,
      }}
    >
      {normalized.map((row, y) => (
        <span key={y}>
          {row.map((v, x) => (
            <span key={x} style={{ opacity: 0.3 + v * 0.7 }}>
              {valueToAscii(v)}
            </span>
          ))}
          {"\n"}
        </span>
      ))}
    </pre>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Text Sparkline
// ─────────────────────────────────────────────────────────────────────────────

function TextSparkline({
  values,
  maxWidth,
}: {
  values: number[];
  maxWidth?: number;
}) {
  const display = useMemo(() => {
    // Sub-sample if too wide
    let arr = values;
    if (maxWidth && arr.length > maxWidth) {
      const step = arr.length / maxWidth;
      const sampled: number[] = [];
      for (let i = 0; i < maxWidth; i++) {
        sampled.push(arr[Math.floor(i * step)]);
      }
      arr = sampled;
    }
    const norm = normalize1D(arr);
    return norm.map((v) => valueToSparkline(v)).join("");
  }, [values, maxWidth]);

  return (
    <span style={{ ...MONO_STYLE, fontSize: "14px" }}>{display}</span>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Terminal Architecture Diagram
// ─────────────────────────────────────────────────────────────────────────────

function TerminalArchDiagram({
  selectedLayer,
  onSelectLayer,
}: {
  selectedLayer: string | null;
  onSelectLayer: (key: string) => void;
}) {
  return (
    <div
      style={{
        overflowX: "auto",
        overflowY: "hidden",
        padding: "8px 0",
        whiteSpace: "nowrap",
      }}
    >
      <style>{`
        @keyframes arrow-march {
          0% { background-position: 0 0; }
          100% { background-position: 12px 0; }
        }
        .arch-arrow {
          display: inline-block;
          width: 24px;
          height: 2px;
          background: repeating-linear-gradient(
            90deg, ${PHOSPHOR_GREEN} 0px, ${PHOSPHOR_GREEN} 4px, transparent 4px, transparent 8px
          );
          background-size: 12px 2px;
          animation: arrow-march 0.6s linear infinite;
          vertical-align: middle;
          margin: 0 2px;
          position: relative;
        }
        .arch-arrow::after {
          content: ">";
          position: absolute;
          right: -6px;
          top: -8px;
          color: ${PHOSPHOR_GREEN};
          font-size: 14px;
          text-shadow: ${TEXT_GLOW};
        }
      `}</style>
      <div style={{ display: "inline-flex", alignItems: "center" }}>
        {ARCH_LAYERS.map((layer, i) => {
          const isSelected = selectedLayer === layer.key;
          return (
            <span key={layer.key} style={{ display: "inline-flex", alignItems: "center" }}>
              <button
                onClick={() => onSelectLayer(layer.key)}
                style={{
                  ...MONO_STYLE,
                  background: isSelected ? PHOSPHOR_GREEN : "transparent",
                  color: isSelected ? "#000" : PHOSPHOR_GREEN,
                  textShadow: isSelected ? "none" : TEXT_GLOW,
                  border: `1px solid ${isSelected ? PHOSPHOR_GREEN : DARK_GREEN}`,
                  padding: "4px 8px",
                  cursor: "pointer",
                  display: "inline-block",
                  textAlign: "center",
                  lineHeight: 1.4,
                  whiteSpace: "pre",
                  fontSize: "11px",
                }}
              >
                {`${layer.label}\n${layer.shape}`}
              </button>
              {i < ARCH_LAYERS.length - 1 && <span className="arch-arrow" />}
            </span>
          );
        })}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ASCII Art Display Panel (activation visualization)
// ─────────────────────────────────────────────────────────────────────────────

function ASCIIArtPanel({ selectedLayer }: { selectedLayer: string | null }) {
  const layerActivations = useInferenceStore((s) => s.layerActivations);
  const inputTensor = useInferenceStore((s) => s.inputTensor);

  if (!selectedLayer) {
    return (
      <div style={{ ...MONO_STYLE, fontSize: "12px", opacity: 0.5, padding: "8px" }}>
        {"// Click a layer above to inspect activations"}
      </div>
    );
  }

  // Input layer: show the raw 28x28
  if (selectedLayer === "input") {
    if (!inputTensor) {
      return (
        <div style={{ ...MONO_STYLE, fontSize: "12px", opacity: 0.5, padding: "8px" }}>
          {"// No input data. Draw something first."}
        </div>
      );
    }
    return (
      <div style={{ padding: "8px" }}>
        <div style={{ ...MONO_STYLE, fontSize: "12px", marginBottom: "4px" }}>
          {"Layer: input [28x28x1]"}
        </div>
        <ASCIIRenderer data={inputTensor} enhanced fontSize={6} />
      </div>
    );
  }

  const activation = layerActivations[selectedLayer];
  if (!activation) {
    return (
      <div style={{ ...MONO_STYLE, fontSize: "12px", opacity: 0.5, padding: "8px" }}>
        {"// No activation data for " + selectedLayer + ". Run inference first."}
      </div>
    );
  }

  const layerMeta = ARCH_LAYERS.find((l) => l.key === selectedLayer);
  const shapeStr = layerMeta?.shape ?? "?";

  // 3D activation (conv/pool layers): average channels and render as 2D ASCII
  if (Array.isArray(activation[0]) && Array.isArray((activation[0] as number[][])[0])) {
    const tensor3d = activation as number[][][];
    const avg = averageChannels(tensor3d);
    return (
      <div style={{ padding: "8px" }}>
        <div style={{ ...MONO_STYLE, fontSize: "12px", marginBottom: "4px" }}>
          {"Layer: " + selectedLayer + " [" + shapeStr + "]  (channel-averaged)"}
        </div>
        <ASCIIRenderer data={avg} enhanced fontSize={6} />
      </div>
    );
  }

  // 1D activation (dense/relu4/output): render as sparkline
  const arr1d = activation as number[];
  return (
    <div style={{ padding: "8px" }}>
      <div style={{ ...MONO_STYLE, fontSize: "12px", marginBottom: "4px" }}>
        {"Layer: " + selectedLayer + " [" + shapeStr + "]"}
      </div>
      <TextSparkline values={arr1d} maxWidth={80} />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Layer Sparklines Stack (compact fingerprint of all layers)
// ─────────────────────────────────────────────────────────────────────────────

function LayerSparklineStack() {
  const layerActivations = useInferenceStore((s) => s.layerActivations);

  const lines = useMemo(() => {
    const result: { name: string; sparkValues: number[]; channels: number; maxVal: number }[] = [];
    for (const layer of ARCH_LAYERS) {
      if (layer.key === "input") continue;
      const act = layerActivations[layer.key];
      if (!act) continue;

      if (Array.isArray(act[0]) && Array.isArray((act[0] as number[][])[0])) {
        const tensor3d = act as number[][][];
        const means = channelMeans(tensor3d);
        const maxVal = Math.max(...means);
        result.push({ name: layer.key, sparkValues: means, channels: tensor3d.length, maxVal });
      } else {
        const arr1d = act as number[];
        const maxVal = Math.max(...arr1d);
        result.push({ name: layer.key, sparkValues: arr1d, channels: arr1d.length, maxVal });
      }
    }
    return result;
  }, [layerActivations]);

  if (lines.length === 0) return null;

  return (
    <div style={{ padding: "4px 8px" }}>
      {lines.map((l) => (
        <div key={l.name} style={{ ...MONO_STYLE, fontSize: "11px", lineHeight: "16px" }}>
          <span style={{ display: "inline-block", width: "60px", textAlign: "right", marginRight: "8px", opacity: 0.7 }}>
            {l.name}
          </span>
          <TextSparkline values={l.sparkValues} maxWidth={40} />
          <span style={{ opacity: 0.5, marginLeft: "8px" }}>
            ({l.channels} ch, max: {l.maxVal.toFixed(2)})
          </span>
        </div>
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Command Line Output (typing animation)
// ─────────────────────────────────────────────────────────────────────────────

interface TerminalLine {
  text: string;
  id: number;
}

function useTypingAnimation(
  fullText: string,
  trigger: number // increment to restart
) {
  const [displayed, setDisplayed] = useState("");
  const [done, setDone] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const indexRef = useRef(0);

  useEffect(() => {
    if (!fullText) {
      setDisplayed("");
      setDone(true);
      return;
    }

    // Reset
    setDisplayed("");
    setDone(false);
    indexRef.current = 0;

    if (intervalRef.current) clearInterval(intervalRef.current);

    intervalRef.current = setInterval(() => {
      indexRef.current += 1;
      const nextChunk = fullText.slice(0, indexRef.current);
      setDisplayed(nextChunk);

      if (indexRef.current >= fullText.length) {
        setDone(true);
        if (intervalRef.current) clearInterval(intervalRef.current);
      }
    }, 12);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [fullText, trigger]);

  return { displayed, done };
}

function CommandLineOutput() {
  const layerActivations = useInferenceStore((s) => s.layerActivations);
  const prediction = useInferenceStore((s) => s.prediction);
  const topPrediction = useInferenceStore((s) => s.topPrediction);
  const isInferring = useInferenceStore((s) => s.isInferring);

  const [history, setHistory] = useState<TerminalLine[]>([]);
  const [triggerCount, setTriggerCount] = useState(0);
  const outputRef = useRef<HTMLDivElement>(null);
  const lastPredRef = useRef<string | null>(null);

  // Build the full output text for the current inference
  const currentOutput = useMemo(() => {
    if (!prediction || !topPrediction) return "";

    const lines: string[] = [];
    lines.push("$ ./predict --input drawing.raw");
    lines.push("Loading model... [OK]");
    lines.push("Running inference...");
    lines.push("");

    // Layer summaries with sparklines
    for (const layer of ARCH_LAYERS) {
      if (layer.key === "input") continue;
      const act = layerActivations[layer.key];
      if (!act) continue;

      let sparkStr: string;
      let chCount: number;
      let maxVal: number;

      if (Array.isArray(act[0]) && Array.isArray((act[0] as number[][])[0])) {
        const tensor3d = act as number[][][];
        const means = channelMeans(tensor3d);
        maxVal = Math.max(...means);
        chCount = tensor3d.length;
        const norm = normalize1D(means);
        // Sample to 12 characters
        const step = norm.length / 12;
        sparkStr = "";
        for (let i = 0; i < 12; i++) {
          sparkStr += valueToSparkline(norm[Math.floor(i * step)]);
        }
      } else {
        const arr1d = act as number[];
        maxVal = Math.max(...arr1d);
        chCount = arr1d.length;
        const norm = normalize1D(arr1d);
        const step = norm.length / 12;
        sparkStr = "";
        for (let i = 0; i < 12; i++) {
          sparkStr += valueToSparkline(norm[Math.floor(i * step)]);
        }
      }

      const nameStr = ("Layer " + layer.key + ":").padEnd(14);
      const unitLabel = layer.key.startsWith("dense") || layer.key.startsWith("relu4") || layer.key === "output"
        ? "neurons"
        : "ch";
      lines.push(`${nameStr} ${sparkStr}  (${chCount} ${unitLabel}, max: ${maxVal.toFixed(2)})`);
    }

    lines.push("");

    // Top prediction
    const predChar = EMNIST_CLASSES[topPrediction.classIndex];
    const conf = (topPrediction.confidence * 100).toFixed(1);
    lines.push(`PREDICTION: "${predChar}"  confidence: ${conf}%`);
    lines.push("");

    // Top 5 predictions
    lines.push("Top 5:");
    const indexed = prediction.map((v, i) => ({ v, i }));
    indexed.sort((a, b) => b.v - a.v);
    const top5 = indexed
      .filter((p) => !BYMERGE_MERGED_INDICES.has(p.i))
      .slice(0, 5);

    const maxBar = 20;
    const topVal = top5[0]?.v ?? 1;
    for (const entry of top5) {
      const ch = EMNIST_CLASSES[entry.i].padEnd(2);
      const pct = (entry.v * 100).toFixed(1).padStart(5);
      const barLen = Math.max(1, Math.round((entry.v / topVal) * maxBar));
      // Use full block for most of the bar, half block for sub-unit
      const fullBlocks = Math.floor(barLen);
      const bar = "\u2588".repeat(fullBlocks) + (barLen > fullBlocks ? "\u258C" : "");
      lines.push(`  ${ch} ${bar.padEnd(maxBar + 1)} ${pct}%`);
    }

    lines.push("");
    return lines.join("\n");
  }, [prediction, topPrediction, layerActivations]);

  // Detect new prediction and archive old output
  useEffect(() => {
    const sig = topPrediction
      ? `${topPrediction.classIndex}-${topPrediction.confidence.toFixed(6)}`
      : null;
    if (sig && sig !== lastPredRef.current) {
      lastPredRef.current = sig;
      setTriggerCount((c) => c + 1);
    }
  }, [topPrediction]);

  const { displayed, done } = useTypingAnimation(currentOutput, triggerCount);

  // On trigger change, push previous output into history
  const prevOutputRef = useRef("");
  useEffect(() => {
    if (triggerCount > 1 && prevOutputRef.current) {
      setHistory((h) => [...h, { text: prevOutputRef.current, id: triggerCount - 1 }]);
    }
    prevOutputRef.current = currentOutput;
  }, [triggerCount, currentOutput]);

  // Auto-scroll
  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [displayed, history]);

  return (
    <div
      ref={outputRef}
      style={{
        ...MONO_STYLE,
        fontSize: "12px",
        lineHeight: "16px",
        overflowY: "auto",
        flex: 1,
        padding: "8px 12px",
        whiteSpace: "pre-wrap",
        wordBreak: "break-all",
      }}
    >
      <style>{`
        @keyframes blink-cursor {
          0%, 49% { opacity: 1; }
          50%, 100% { opacity: 0; }
        }
        .terminal-cursor {
          animation: blink-cursor 1s step-end infinite;
        }
      `}</style>

      {/* History */}
      {history.map((h) => (
        <div key={h.id} style={{ opacity: 0.5 }}>
          {h.text}
        </div>
      ))}

      {/* Current typing animation */}
      {isInferring && !currentOutput && (
        <div>
          {"$ ./predict --input drawing.raw\nLoading model... [OK]\nProcessing..."}
        </div>
      )}
      {displayed}
      {done && currentOutput && (
        <span>
          {"$ "}
          <span className="terminal-cursor">_</span>
        </span>
      )}
      {!done && currentOutput && <span className="terminal-cursor">_</span>}
      {!currentOutput && !isInferring && (
        <div>
          {"EMNIST OCR Neural Network v1.0\n"}
          {"Copyright (c) 2025 ocr-visualization\n"}
          {"Ready. Draw a character to begin inference.\n\n"}
          {"$ "}
          <span className="terminal-cursor">_</span>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Drawing Canvas Area
// ─────────────────────────────────────────────────────────────────────────────

function DrawingArea() {
  const { infer } = useInference();
  const isInferring = useInferenceStore((s) => s.isInferring);

  const onStrokeEnd = useCallback(
    (imageData: ImageData) => {
      infer(imageData);
    },
    [infer]
  );

  const { canvasRef, clear, startDrawing, draw, stopDrawing } = useDrawingCanvas({
    width: 280,
    height: 280,
    lineWidth: 16,
    strokeColor: "#ffffff",
    backgroundColor: "#000000",
    onStrokeEnd,
  });

  return (
    <div style={{ padding: "8px", flexShrink: 0, width: "220px" }}>
      <div style={{ ...MONO_STYLE, fontSize: "12px", marginBottom: "4px" }}>
        {"$ draw > [canvas]"}
      </div>
      <div
        style={{
          border: `1px solid ${DARK_GREEN}`,
          boxShadow: `0 0 8px ${DARK_GREEN}, inset 0 0 8px rgba(51,255,51,0.05)`,
          width: "200px",
          height: "200px",
          position: "relative",
        }}
      >
        <canvas
          ref={canvasRef}
          width={280}
          height={280}
          style={{
            width: "200px",
            height: "200px",
            display: "block",
            cursor: "crosshair",
            imageRendering: "pixelated",
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
      <div style={{ display: "flex", alignItems: "center", gap: "8px", marginTop: "4px" }}>
        <button
          onClick={clear}
          style={{
            ...MONO_STYLE,
            background: "transparent",
            border: `1px solid ${DARK_GREEN}`,
            padding: "2px 8px",
            cursor: "pointer",
            fontSize: "11px",
          }}
        >
          {"[Ctrl+C] Clear"}
        </button>
        {isInferring && (
          <InferringDots />
        )}
      </div>
    </div>
  );
}

function InferringDots() {
  const [dots, setDots] = useState("");

  useEffect(() => {
    const iv = setInterval(() => {
      setDots((d) => (d.length >= 3 ? "" : d + "."));
    }, 400);
    return () => clearInterval(iv);
  }, []);

  return (
    <span style={{ ...MONO_STYLE, fontSize: "11px" }}>
      {"Processing" + dots}
    </span>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Layout Export
// ─────────────────────────────────────────────────────────────────────────────

export function RetroTerminalLayout() {
  const [selectedLayer, setSelectedLayer] = useState<string | null>(null);

  return (
    <div
      className="crt-container"
      style={{
        position: "fixed",
        inset: 0,
        background: BG_COLOR,
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
        boxShadow: `inset 0 0 80px rgba(51,255,51,0.04), inset 0 0 200px rgba(51,255,51,0.02)`,
      }}
    >
      <CRTEffectOverlay />

      {/* ── Top section: Drawing canvas + Architecture diagram ────────── */}
      <div
        style={{
          display: "flex",
          borderBottom: `1px solid ${DARK_GREEN}`,
          flexShrink: 0,
        }}
      >
        {/* Drawing Canvas */}
        <DrawingArea />

        {/* Architecture diagram + Layer sparklines */}
        <div
          style={{
            flex: 1,
            overflow: "hidden",
            display: "flex",
            flexDirection: "column",
            borderLeft: `1px solid ${DARK_GREEN}`,
          }}
        >
          <div
            style={{
              ...MONO_STYLE,
              fontSize: "11px",
              padding: "4px 8px",
              borderBottom: `1px solid ${DARK_GREEN}`,
              opacity: 0.6,
            }}
          >
            {"// Network Architecture"}
          </div>
          <TerminalArchDiagram selectedLayer={selectedLayer} onSelectLayer={setSelectedLayer} />
          <LayerSparklineStack />
        </div>
      </div>

      {/* ── Middle section: Activation visualization ───────────────── */}
      <div
        style={{
          borderBottom: `1px solid ${DARK_GREEN}`,
          flexShrink: 0,
          minHeight: "40px",
          maxHeight: "240px",
          overflowY: "auto",
        }}
      >
        <ASCIIArtPanel selectedLayer={selectedLayer} />
      </div>

      {/* ── Bottom section: Terminal output ─────────────────────────── */}
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
          borderBottom: `1px solid ${DARK_GREEN}`,
          marginBottom: "48px", // room for LayoutNav
        }}
      >
        <div
          style={{
            ...MONO_STYLE,
            fontSize: "11px",
            padding: "4px 8px",
            borderBottom: `1px solid ${DARK_GREEN}`,
            opacity: 0.6,
            flexShrink: 0,
          }}
        >
          {"// Terminal Output"}
        </div>
        <CommandLineOutput />
      </div>

      {/* ── Nav ──────────────────────────────────────────────────────── */}
      <LayoutNav />
    </div>
  );
}
