"use client";

import React, {
  useState,
  useEffect,
  useRef,
  useMemo,
  useCallback,
} from "react";
import dynamic from "next/dynamic";
import { useInferenceStore } from "@/stores/inferenceStore";
import { EMNIST_CLASSES } from "@/lib/model/classes";
import { useDrawingCanvas } from "@/hooks/useDrawingCanvas";
import { useInference } from "@/hooks/useInference";
import { LayoutNav } from "@/components/layouts/LayoutNav";

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

// Erosion timeline stages with geographic metaphor labels
const EROSION_STAGES: { key: LayerKey; label: string }[] = [
  { key: "conv1", label: "Survey" },
  { key: "relu1", label: "Expose" },
  { key: "pool1", label: "Erosion" },
  { key: "conv3", label: "Survey" },
  { key: "relu3", label: "Expose" },
  { key: "pool2", label: "Erosion" },
];

// ---------------------------------------------------------------------------
// Terrain Color Utility
// ---------------------------------------------------------------------------

/** Returns RGB [0-255] for a normalized activation value */
function terrainColor(value: number, max: number): [number, number, number] {
  const t = max > 0 ? Math.min(1, Math.max(0, value / max)) : 0;

  // Color stops: deep water -> shallow water -> lowland -> highland -> hills -> mountain -> high mountain -> peak
  if (t < 0.1) {
    // Deep water: #08306b
    return [8, 48, 107];
  } else if (t < 0.25) {
    // Shallow water: lerp #08306b -> #2171b5
    const f = (t - 0.1) / 0.15;
    return [
      Math.round(8 + (33 - 8) * f),
      Math.round(48 + (113 - 48) * f),
      Math.round(107 + (181 - 107) * f),
    ];
  } else if (t < 0.4) {
    // Lowland: lerp #2171b5 -> #41ab5d
    const f = (t - 0.25) / 0.15;
    return [
      Math.round(33 + (65 - 33) * f),
      Math.round(113 + (171 - 113) * f),
      Math.round(181 + (93 - 181) * f),
    ];
  } else if (t < 0.55) {
    // Highland: lerp #41ab5d -> #78c679
    const f = (t - 0.4) / 0.15;
    return [
      Math.round(65 + (120 - 65) * f),
      Math.round(171 + (198 - 171) * f),
      Math.round(93 + (121 - 93) * f),
    ];
  } else if (t < 0.7) {
    // Hills: lerp #78c679 -> #d9f0a3
    const f = (t - 0.55) / 0.15;
    return [
      Math.round(120 + (217 - 120) * f),
      Math.round(198 + (240 - 198) * f),
      Math.round(121 + (163 - 121) * f),
    ];
  } else if (t < 0.8) {
    // Mountain: lerp #d9f0a3 -> #fee08b
    const f = (t - 0.7) / 0.1;
    return [
      Math.round(217 + (254 - 217) * f),
      Math.round(240 + (224 - 240) * f),
      Math.round(163 + (139 - 163) * f),
    ];
  } else if (t < 0.9) {
    // High mountain: lerp #fee08b -> #fc8d59
    const f = (t - 0.8) / 0.1;
    return [
      Math.round(254 + (252 - 254) * f),
      Math.round(224 + (141 - 224) * f),
      Math.round(139 + (89 - 139) * f),
    ];
  } else {
    // Peak: lerp #fc8d59 -> #d73027 -> white
    const f = (t - 0.9) / 0.1;
    if (f < 0.5) {
      const ff = f / 0.5;
      return [
        Math.round(252 + (215 - 252) * ff),
        Math.round(141 + (48 - 141) * ff),
        Math.round(89 + (39 - 89) * ff),
      ];
    } else {
      const ff = (f - 0.5) / 0.5;
      return [
        Math.round(215 + (255 - 215) * ff),
        Math.round(48 + (255 - 48) * ff),
        Math.round(39 + (255 - 39) * ff),
      ];
    }
  }
}

/** CSS color string from terrain color */
function terrainColorCSS(value: number, max: number): string {
  const [r, g, b] = terrainColor(value, max);
  return `rgb(${r},${g},${b})`;
}

// ---------------------------------------------------------------------------
// Data Helpers
// ---------------------------------------------------------------------------

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

/** Get 2D data for the given layer and channel */
function getLayerData2D(
  layerKey: LayerKey,
  channel: number,
  inputTensor: number[][] | null,
  layerActivations: Record<string, number[][][] | number[]>
): number[][] | null {
  if (layerKey === "input") {
    return inputTensor;
  }
  const act = layerActivations[layerKey];
  if (!act) return null;

  // 3D activation
  if (
    Array.isArray(act[0]) &&
    Array.isArray((act[0] as number[][])[0])
  ) {
    const act3d = act as number[][][];
    const ch = Math.min(channel, act3d.length - 1);
    return act3d[ch] ?? null;
  }

  return null;
}

/** Get all channels for a 3D activation layer */
function getLayerChannels(
  layerKey: LayerKey,
  layerActivations: Record<string, number[][][] | number[]>
): number[][][] | null {
  if (layerKey === "input") return null;
  const act = layerActivations[layerKey];
  if (!act) return null;
  if (
    Array.isArray(act[0]) &&
    Array.isArray((act[0] as number[][])[0])
  ) {
    return act as number[][][];
  }
  return null;
}

/** Max value in a 2D array */
function max2D(data: number[][]): number {
  let mx = -Infinity;
  for (const row of data) {
    for (const v of row) {
      if (v > mx) mx = v;
    }
  }
  return mx === -Infinity ? 1 : Math.max(mx, 0.001);
}

/** Mean absolute value for a 3D activation (or specific channel) */
function channelMean(data: number[][]): number {
  let sum = 0;
  let count = 0;
  for (const row of data) {
    for (const v of row) {
      sum += Math.abs(v);
      count++;
    }
  }
  return count > 0 ? sum / count : 0;
}

// ---------------------------------------------------------------------------
// ContourMap Component
// ---------------------------------------------------------------------------

function ContourMap({
  data,
  width,
  height,
  showContours = true,
}: {
  data: number[][] | null;
  width: number;
  height: number;
  showContours?: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    row: number;
    col: number;
    value: number;
  } | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data || data.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rows = data.length;
    const cols = data[0].length;
    const mx = max2D(data);

    // Render elevation colors using ImageData
    const imgData = ctx.createImageData(width, height);
    const cellW = width / cols;
    const cellH = height / rows;

    for (let py = 0; py < height; py++) {
      for (let px = 0; px < width; px++) {
        const col = Math.min(cols - 1, Math.floor(px / cellW));
        const row = Math.min(rows - 1, Math.floor(py / cellH));
        const val = data[row][col];
        const [r, g, b] = terrainColor(val, mx);
        const idx = (py * width + px) * 4;
        imgData.data[idx] = r;
        imgData.data[idx + 1] = g;
        imgData.data[idx + 2] = b;
        imgData.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);

    // Draw contour lines
    if (showContours && rows > 1 && cols > 1) {
      ctx.strokeStyle = "#00000044";
      ctx.lineWidth = 0.5;

      for (let level = 1; level <= 9; level++) {
        const threshold = (level / 10) * mx;
        ctx.beginPath();

        for (let r = 0; r < rows; r++) {
          for (let c = 0; c < cols; c++) {
            const val = data[r][c];
            // Check right neighbor
            if (c < cols - 1) {
              const rightVal = data[r][c + 1];
              if (
                (val >= threshold && rightVal < threshold) ||
                (val < threshold && rightVal >= threshold)
              ) {
                // Interpolate position
                const frac =
                  Math.abs(val - threshold) /
                  (Math.abs(val - rightVal) || 1);
                const x = (c + frac) * cellW;
                const y = r * cellH;
                ctx.moveTo(x, y);
                ctx.lineTo(x, y + cellH);
              }
            }
            // Check bottom neighbor
            if (r < rows - 1) {
              const bottomVal = data[r + 1][c];
              if (
                (val >= threshold && bottomVal < threshold) ||
                (val < threshold && bottomVal >= threshold)
              ) {
                const frac =
                  Math.abs(val - threshold) /
                  (Math.abs(val - bottomVal) || 1);
                const x = c * cellW;
                const y = (r + frac) * cellH;
                ctx.moveTo(x, y);
                ctx.lineTo(x + cellW, y);
              }
            }
          }
        }
        ctx.stroke();
      }
    }
  }, [data, width, height, showContours]);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!data || data.length === 0) return;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const scaleX = width / rect.width;
      const scaleY = height / rect.height;
      const px = (e.clientX - rect.left) * scaleX;
      const py = (e.clientY - rect.top) * scaleY;
      const rows = data.length;
      const cols = data[0].length;
      const col = Math.min(cols - 1, Math.max(0, Math.floor(px / (width / cols))));
      const row = Math.min(rows - 1, Math.max(0, Math.floor(py / (height / rows))));
      setTooltip({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
        row,
        col,
        value: data[row][col],
      });
    },
    [data, width, height]
  );

  const handleMouseLeave = useCallback(() => {
    setTooltip(null);
  }, []);

  if (!data || data.length === 0) {
    return (
      <div
        style={{
          width,
          height,
          background: "#08306b",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontFamily: "Georgia, serif",
          color: "#c8b8a0",
          fontSize: 14,
          fontStyle: "italic",
          border: "1px solid #4a3828",
          borderRadius: 4,
        }}
      >
        Awaiting survey data...
      </div>
    );
  }

  return (
    <div style={{ position: "relative", width: "fit-content" }}>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{
          width: "100%",
          maxWidth: width,
          height: "auto",
          borderRadius: 4,
          border: "1px solid #4a3828",
          cursor: "crosshair",
        }}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      />
      {tooltip && (
        <div
          style={{
            position: "absolute",
            left: tooltip.x + 12,
            top: tooltip.y - 30,
            background: "rgba(26, 21, 16, 0.95)",
            border: "1px solid #4a3828",
            borderRadius: 4,
            padding: "4px 8px",
            fontFamily: "Georgia, serif",
            fontSize: 11,
            color: "#c8b8a0",
            pointerEvents: "none",
            whiteSpace: "nowrap",
            zIndex: 50,
          }}
        >
          ({tooltip.row}, {tooltip.col}) = {tooltip.value.toFixed(4)}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// TerrainFlyover3D (R3F Component - must be dynamically imported)
// ---------------------------------------------------------------------------

function TerrainFlyover3DInner({ data }: { data: number[][] | null }) {
  // These imports are only used in this component, which is loaded client-side only
  const { Canvas } = require("@react-three/fiber") as typeof import("@react-three/fiber");
  const { OrbitControls } = require("@react-three/drei") as typeof import("@react-three/drei");
  const THREE = require("three") as typeof import("three");

  const meshRef = useRef<import("three").Mesh>(null);

  const geometry = useMemo(() => {
    if (!data || data.length === 0) {
      // Flat plane fallback
      const geo = new THREE.PlaneGeometry(10, 10, 27, 27);
      geo.rotateX(-Math.PI / 2);
      // Set all to flat green
      const colors = new Float32Array(geo.attributes.position.count * 3);
      for (let i = 0; i < colors.length; i += 3) {
        colors[i] = 0.25;
        colors[i + 1] = 0.67;
        colors[i + 2] = 0.36;
      }
      geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      return geo;
    }

    const rows = data.length;
    const cols = data[0].length;
    const mx = max2D(data);
    const geo = new THREE.PlaneGeometry(10, 10, cols - 1, rows - 1);
    geo.rotateX(-Math.PI / 2);

    const positions = geo.attributes.position;
    const colors = new Float32Array(positions.count * 3);

    for (let i = 0; i < positions.count; i++) {
      const col = i % cols;
      const row = Math.floor(i / cols);
      const val = data[row]?.[col] ?? 0;

      // Displace Y by activation value
      positions.setY(i, (val / mx) * 3);

      // Vertex color from terrain palette
      const [r, g, b] = terrainColor(val, mx);
      colors[i * 3] = r / 255;
      colors[i * 3 + 1] = g / 255;
      colors[i * 3 + 2] = b / 255;
    }

    positions.needsUpdate = true;
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geo.computeVertexNormals();

    return geo;
  }, [data, THREE]);

  if (!data || data.length === 0) {
    return (
      <div
        style={{
          width: 400,
          height: 300,
          background: "#2a2018",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          border: "1px solid #4a3828",
          borderRadius: 4,
          fontFamily: "Georgia, serif",
          color: "#c8b8a0",
          fontStyle: "italic",
          fontSize: 13,
        }}
      >
        AWAITING DATA
      </div>
    );
  }

  return (
    <div
      style={{
        width: 400,
        height: 300,
        border: "1px solid #4a3828",
        borderRadius: 4,
        overflow: "hidden",
      }}
    >
      <Canvas
        camera={{ position: [8, 6, 8], fov: 50 }}
        style={{ background: "#1a1510" }}
      >
        <fog attach="fog" args={["#1a1510", 10, 50]} />
        <ambientLight intensity={0.4} />
        <directionalLight position={[5, 10, 5]} intensity={0.8} />
        <mesh ref={meshRef} geometry={geometry}>
          <meshLambertMaterial vertexColors />
        </mesh>
        <OrbitControls
          autoRotate
          autoRotateSpeed={0.5}
          enableZoom
          maxPolarAngle={Math.PI / 2.2}
        />
      </Canvas>
    </div>
  );
}

const TerrainFlyoverDynamic = dynamic(
  () => Promise.resolve({ default: TerrainFlyover3DInner }),
  { ssr: false }
);

// ---------------------------------------------------------------------------
// ChannelAtlas Component
// ---------------------------------------------------------------------------

function ChannelAtlasThumbnail({
  data,
  size,
  selected,
  onClick,
  index,
}: {
  data: number[][];
  size: number;
  selected: boolean;
  onClick: () => void;
  index: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data || data.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rows = data.length;
    const cols = data[0].length;
    const mx = max2D(data);
    const imgData = ctx.createImageData(size, size);
    const cellW = size / cols;
    const cellH = size / rows;

    for (let py = 0; py < size; py++) {
      for (let px = 0; px < size; px++) {
        const col = Math.min(cols - 1, Math.floor(px / cellW));
        const row = Math.min(rows - 1, Math.floor(py / cellH));
        const val = data[row][col];
        const [r, g, b] = terrainColor(val, mx);
        const idx = (py * size + px) * 4;
        imgData.data[idx] = r;
        imgData.data[idx + 1] = g;
        imgData.data[idx + 2] = b;
        imgData.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }, [data, size]);

  return (
    <canvas
      ref={canvasRef}
      width={size}
      height={size}
      onClick={onClick}
      title={`Channel ${index}`}
      style={{
        width: size,
        height: size,
        cursor: "pointer",
        border: selected ? "2px solid #fff" : "1px solid #333",
        borderRadius: 2,
        boxSizing: "border-box",
      }}
    />
  );
}

function ChannelAtlas({
  layerKey,
  selectedChannel,
  onSelectChannel,
}: {
  layerKey: LayerKey;
  selectedChannel: number;
  onSelectChannel: (ch: number) => void;
}) {
  const layerActivations = useInferenceStore((s) => s.layerActivations);
  const channels = getLayerChannels(layerKey, layerActivations);

  const { gridCols, thumbSize } = useMemo(() => {
    if (!channels) return { gridCols: 8, thumbSize: 40 };
    const count = channels.length;
    if (count <= 32) return { gridCols: 8, thumbSize: 56 };
    if (count <= 64) return { gridCols: 8, thumbSize: 38 };
    return { gridCols: 16, thumbSize: 26 };
  }, [channels]);

  if (!channels || channels.length === 0) {
    return (
      <div
        style={{
          padding: 16,
          fontFamily: "Georgia, serif",
          fontSize: 13,
          color: "#c8b8a0",
          fontStyle: "italic",
          textAlign: "center",
        }}
      >
        {layerKey === "input"
          ? "Input has a single channel"
          : "No channel data available"}
      </div>
    );
  }

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: `repeat(${gridCols}, ${thumbSize}px)`,
        gap: 2,
        maxHeight: 250,
        overflowY: "auto",
        padding: 8,
      }}
    >
      {channels.map((chData, idx) => (
        <ChannelAtlasThumbnail
          key={idx}
          data={chData}
          size={thumbSize}
          selected={idx === selectedChannel}
          onClick={() => onSelectChannel(idx)}
          index={idx}
        />
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// ErosionTimeline Component
// ---------------------------------------------------------------------------

function ErosionTimelineThumbnail({
  data,
  label,
}: {
  data: number[][] | null;
  label: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const size = 72;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    if (!data || data.length === 0) {
      ctx.fillStyle = "#08306b";
      ctx.fillRect(0, 0, size, size);
      ctx.fillStyle = "#c8b8a0";
      ctx.font = "italic 9px Georgia";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("N/A", size / 2, size / 2);
      return;
    }

    const rows = data.length;
    const cols = data[0].length;
    const mx = max2D(data);
    const imgData = ctx.createImageData(size, size);
    const cellW = size / cols;
    const cellH = size / rows;

    for (let py = 0; py < size; py++) {
      for (let px = 0; px < size; px++) {
        const col = Math.min(cols - 1, Math.floor(px / cellW));
        const row = Math.min(rows - 1, Math.floor(py / cellH));
        const val = data[row][col];
        const [r, g, b] = terrainColor(val, mx);
        const idx = (py * size + px) * 4;
        imgData.data[idx] = r;
        imgData.data[idx + 1] = g;
        imgData.data[idx + 2] = b;
        imgData.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }, [data]);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 4,
      }}
    >
      <canvas
        ref={canvasRef}
        width={size}
        height={size}
        style={{
          width: size,
          height: size,
          borderRadius: 4,
          border: "1px solid #4a3828",
        }}
      />
      <span
        style={{
          fontFamily: "Georgia, serif",
          fontSize: 10,
          color: "#c8b8a0",
          fontVariant: "small-caps",
        }}
      >
        {label}
      </span>
    </div>
  );
}

function ErosionTimeline({
  selectedChannel,
}: {
  selectedChannel: number;
}) {
  const inputTensor = useInferenceStore((s) => s.inputTensor);
  const layerActivations = useInferenceStore((s) => s.layerActivations);

  // For each erosion stage, get the data for the selected channel
  // When dimensions change (e.g., pool), use channel-averaged map if selected channel exceeds range
  const stageData = useMemo(() => {
    return EROSION_STAGES.map((stage) => {
      const data2d = getLayerData2D(
        stage.key,
        selectedChannel,
        inputTensor,
        layerActivations
      );
      if (data2d) return data2d;

      // Fall back to channel-averaged map
      const act = layerActivations[stage.key];
      if (
        act &&
        Array.isArray(act[0]) &&
        Array.isArray((act[0] as number[][])[0])
      ) {
        return averageChannels(act as number[][][]);
      }
      return null;
    });
  }, [selectedChannel, inputTensor, layerActivations]);

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 4,
        overflowX: "auto",
        padding: "8px 4px",
      }}
    >
      {EROSION_STAGES.map((stage, i) => (
        <React.Fragment key={stage.key + i}>
          <ErosionTimelineThumbnail
            data={stageData[i]}
            label={LAYER_DISPLAY_NAMES[stage.key]}
          />
          {i < EROSION_STAGES.length - 1 && (
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 2,
                minWidth: 40,
              }}
            >
              <span
                style={{
                  fontFamily: "Georgia, serif",
                  fontSize: 16,
                  color:
                    EROSION_STAGES[i + 1].label === "Erosion"
                      ? "#d73027"
                      : "#78c679",
                  animation: "arrowPulse 1.5s ease-in-out infinite",
                }}
              >
                {"\u2192"}
              </span>
              <span
                style={{
                  fontFamily: "Georgia, serif",
                  fontSize: 9,
                  color: "#c8b8a0",
                  fontStyle: "italic",
                  whiteSpace: "nowrap",
                }}
              >
                {EROSION_STAGES[i + 1].label}
              </span>
            </div>
          )}
        </React.Fragment>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// SummitPrediction Component (SVG)
// ---------------------------------------------------------------------------

function SummitPrediction() {
  const prediction = useInferenceStore((s) => s.prediction);
  const topPrediction = useInferenceStore((s) => s.topPrediction);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  const svgWidth = 900;
  const svgHeight = 150;
  const baseY = svgHeight - 15; // treeline region at bottom
  const maxPeakHeight = svgHeight - 40;

  const mountainPath = useMemo(() => {
    if (!prediction) return "";
    const maxProb = Math.max(...prediction, 0.001);

    let d = `M 0 ${baseY}`;
    for (let i = 0; i < prediction.length; i++) {
      const prob = prediction[i];
      const x = (i / (prediction.length - 1)) * svgWidth;
      const peakH = (prob / maxProb) * maxPeakHeight;
      const peakY = baseY - peakH;
      const halfWidth = svgWidth / prediction.length / 2;

      // Quadratic bezier for each mountain peak
      d += ` Q ${x - halfWidth} ${baseY}, ${x} ${peakY}`;
      d += ` Q ${x + halfWidth} ${baseY}, ${x + halfWidth * 2} ${baseY}`;
    }
    d += ` L ${svgWidth} ${baseY} Z`;
    return d;
  }, [prediction, baseY, maxPeakHeight, svgWidth]);

  // Treeline path (jagged green line along bottom)
  const treelinePath = useMemo(() => {
    let d = `M 0 ${baseY}`;
    for (let x = 0; x <= svgWidth; x += 6) {
      const jag = Math.random() * 8;
      d += ` L ${x} ${baseY - jag}`;
    }
    d += ` L ${svgWidth} ${svgHeight} L 0 ${svgHeight} Z`;
    return d;
  }, [baseY, svgWidth, svgHeight]);

  if (!prediction) {
    return (
      <div
        style={{
          width: "100%",
          height: svgHeight,
          background: "linear-gradient(to bottom, #1a237e, #3e2723)",
          borderRadius: 4,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontFamily: "Georgia, serif",
          fontStyle: "italic",
          color: "#c8b8a0",
          fontSize: 14,
          border: "1px solid #4a3828",
        }}
      >
        Awaiting prediction summit...
      </div>
    );
  }

  const maxProb = Math.max(...prediction, 0.001);

  return (
    <div style={{ position: "relative", width: "100%" }}>
      <svg
        width="100%"
        viewBox={`0 0 ${svgWidth} ${svgHeight}`}
        style={{ display: "block", borderRadius: 4, border: "1px solid #4a3828" }}
      >
        {/* Sky gradient background */}
        <defs>
          <linearGradient id="skyGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#1a237e" />
            <stop offset="100%" stopColor="#3e2723" />
          </linearGradient>
          <linearGradient id="mountainGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#ffffff" />
            <stop offset="30%" stopColor="#d73027" />
            <stop offset="60%" stopColor="#8d6e63" />
            <stop offset="100%" stopColor="#3e2723" />
          </linearGradient>
        </defs>

        {/* Sky */}
        <rect
          x="0"
          y="0"
          width={svgWidth}
          height={svgHeight}
          fill="url(#skyGrad)"
        />

        {/* Mountain range */}
        <path d={mountainPath} fill="url(#mountainGrad)" opacity={0.9} />

        {/* Treeline */}
        <path d={treelinePath} fill="#2e7d32" opacity={0.6} />

        {/* Interactive hover regions and flag for top prediction */}
        {prediction.map((prob, i) => {
          const x = (i / (prediction.length - 1)) * svgWidth;
          const peakH = (prob / maxProb) * maxPeakHeight;
          const peakY = baseY - peakH;
          const isTop = topPrediction?.classIndex === i;
          const isHovered = hoveredIdx === i;

          return (
            <g key={i}>
              {/* Hover region */}
              <rect
                x={x - svgWidth / prediction.length / 2}
                y={0}
                width={svgWidth / prediction.length}
                height={svgHeight}
                fill="transparent"
                onMouseEnter={() => setHoveredIdx(i)}
                onMouseLeave={() => setHoveredIdx(null)}
                style={{ cursor: "pointer" }}
              />

              {/* Flag on the top prediction */}
              {isTop && peakH > 5 && (
                <g>
                  {/* Flag pole */}
                  <line
                    x1={x}
                    y1={peakY}
                    x2={x}
                    y2={peakY - 20}
                    stroke="#fff"
                    strokeWidth={1.5}
                  />
                  {/* Flag triangle */}
                  <polygon
                    points={`${x},${peakY - 20} ${x + 12},${peakY - 16} ${x},${peakY - 12}`}
                    fill="#d73027"
                  />
                  {/* Predicted character */}
                  <text
                    x={x}
                    y={peakY - 24}
                    textAnchor="middle"
                    fill="#fff"
                    fontFamily="Georgia, serif"
                    fontWeight="bold"
                    fontSize={12}
                  >
                    {EMNIST_CLASSES[i]}
                  </text>
                </g>
              )}

              {/* Tooltip on hover */}
              {isHovered && (
                <g>
                  <rect
                    x={Math.min(x + 5, svgWidth - 90)}
                    y={Math.max(peakY - 30, 5)}
                    width={80}
                    height={24}
                    rx={4}
                    fill="rgba(26, 21, 16, 0.95)"
                    stroke="#4a3828"
                  />
                  <text
                    x={Math.min(x + 45, svgWidth - 50)}
                    y={Math.max(peakY - 14, 21)}
                    textAnchor="middle"
                    fill="#c8b8a0"
                    fontFamily="Georgia, serif"
                    fontSize={11}
                  >
                    {EMNIST_CLASSES[i]}: {(prob * 100).toFixed(1)}%
                  </text>
                </g>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

// ---------------------------------------------------------------------------
// LayerTabs Component
// ---------------------------------------------------------------------------

function LayerTabs({
  selectedLayer,
  onSelectLayer,
}: {
  selectedLayer: LayerKey;
  onSelectLayer: (key: LayerKey) => void;
}) {
  const layerActivations = useInferenceStore((s) => s.layerActivations);
  const inputTensor = useInferenceStore((s) => s.inputTensor);

  // Compute average activation for each layer to get a terrain-color dot
  const layerMeans = useMemo(() => {
    const means: Record<string, number> = {};
    for (const key of LAYER_KEYS) {
      if (key === "input") {
        if (inputTensor) {
          means[key] = channelMean(inputTensor);
        } else {
          means[key] = 0;
        }
      } else {
        const act = layerActivations[key];
        if (act && Array.isArray(act[0]) && Array.isArray((act[0] as number[][])[0])) {
          const avg = averageChannels(act as number[][][]);
          means[key] = channelMean(avg);
        } else {
          means[key] = 0;
        }
      }
    }
    return means;
  }, [layerActivations, inputTensor]);

  const maxMean = useMemo(() => {
    let m = 0;
    for (const key of LAYER_KEYS) {
      if (layerMeans[key] > m) m = layerMeans[key];
    }
    return m || 1;
  }, [layerMeans]);

  return (
    <div
      style={{
        display: "flex",
        gap: 2,
        flexWrap: "wrap",
        justifyContent: "center",
      }}
    >
      {LAYER_KEYS.map((key) => {
        const active = key === selectedLayer;
        const dotColor = terrainColorCSS(layerMeans[key], maxMean);

        return (
          <button
            key={key}
            onClick={() => onSelectLayer(key)}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 4,
              padding: "6px 12px",
              fontFamily: "Georgia, serif",
              fontVariant: "small-caps",
              fontSize: 12,
              color: active ? "#e8d8c0" : "#8a7a6a",
              background: active ? "#3a2a1a" : "#2a2018",
              border: active ? "1px solid #6a5a4a" : "1px solid #4a3828",
              borderRadius: "6px 6px 0 0",
              cursor: "pointer",
              boxShadow: active
                ? "0 -2px 8px rgba(100, 80, 60, 0.3)"
                : "none",
              transition: "all 0.2s",
            }}
          >
            <span
              style={{
                display: "inline-block",
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: dotColor,
                border: "1px solid #5a4a3a",
              }}
            />
            {LAYER_DISPLAY_NAMES[key]}
          </button>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// SurveyorDrawingCanvas Component
// ---------------------------------------------------------------------------

function SurveyorDrawingCanvas() {
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

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 8,
      }}
    >
      {/* Label */}
      <span
        style={{
          fontFamily: "Georgia, serif",
          fontStyle: "italic",
          fontSize: 14,
          color: "#8d6e63",
        }}
      >
        Field Sample
      </span>

      {/* Canvas container styled as field notebook */}
      <div
        style={{
          position: "relative",
          border: `2px dashed #8d6e63`,
          borderRadius: 4,
          background: "#000000",
          boxShadow: isInferring
            ? "0 0 12px rgba(141, 110, 99, 0.4)"
            : "none",
          transition: "box-shadow 0.3s",
        }}
      >
        <canvas
          ref={canvasRef}
          width={DRAWING_INTERNAL_SIZE}
          height={DRAWING_INTERNAL_SIZE}
          className="touch-none"
          style={{
            width: DRAWING_CANVAS_SIZE,
            height: DRAWING_CANVAS_SIZE,
            cursor: "crosshair",
            display: "block",
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

        {/* Draw hint */}
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
                fontFamily: "Georgia, serif",
                fontStyle: "italic",
                fontSize: 14,
                color: "rgba(141, 110, 99, 0.4)",
              }}
            >
              Sketch here
            </span>
          </div>
        )}
      </div>

      {/* Clear button */}
      <button
        onClick={handleClear}
        style={{
          fontFamily: "Georgia, serif",
          fontSize: 12,
          color: "#8d6e63",
          background: "transparent",
          border: "1px solid #5a4a3a",
          borderRadius: 4,
          padding: "4px 16px",
          cursor: "pointer",
          transition: "all 0.2s",
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.color = "#c8b8a0";
          e.currentTarget.style.borderColor = "#8d6e63";
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.color = "#8d6e63";
          e.currentTarget.style.borderColor = "#5a4a3a";
        }}
      >
        New Sample
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Layout Export
// ---------------------------------------------------------------------------

export function TopographicLayout() {
  const [selectedLayer, setSelectedLayer] = useState<LayerKey>("conv1");
  const [selectedChannel, setSelectedChannel] = useState(0);

  const inputTensor = useInferenceStore((s) => s.inputTensor);
  const layerActivations = useInferenceStore((s) => s.layerActivations);
  const topPrediction = useInferenceStore((s) => s.topPrediction);
  const isInferring = useInferenceStore((s) => s.isInferring);

  // Get the 2D data for the main ContourMap
  const mainData = useMemo(() => {
    return getLayerData2D(
      selectedLayer,
      selectedChannel,
      inputTensor,
      layerActivations
    );
  }, [selectedLayer, selectedChannel, inputTensor, layerActivations]);

  // Reset channel to 0 when switching layers
  useEffect(() => {
    setSelectedChannel(0);
  }, [selectedLayer]);

  // Determine contour map size based on data dimensions
  const contourSize = useMemo(() => {
    if (!mainData || mainData.length === 0) return 500;
    const dim = Math.max(mainData.length, mainData[0].length);
    if (dim <= 7) return 350;
    if (dim <= 14) return 420;
    return 500;
  }, [mainData]);

  return (
    <div
      style={{
        width: "100vw",
        minHeight: "100vh",
        background: "#1a1510",
        color: "#c8b8a0",
        fontFamily: "Georgia, serif",
        overflow: "auto",
      }}
    >
      {/* CSS for arrow animation */}
      <style>{`
        @keyframes arrowPulse {
          0%, 100% { opacity: 0.5; transform: translateX(0); }
          50% { opacity: 1; transform: translateX(3px); }
        }
      `}</style>

      {/* Top Section */}
      <div
        style={{
          display: "flex",
          alignItems: "flex-start",
          gap: 24,
          padding: "16px 24px 8px",
          flexWrap: "wrap",
        }}
      >
        {/* Drawing Canvas - Left */}
        <div style={{ flexShrink: 0 }}>
          <SurveyorDrawingCanvas />
        </div>

        {/* Layer Tabs - Center */}
        <div
          style={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 8,
            minWidth: 300,
          }}
        >
          <LayerTabs
            selectedLayer={selectedLayer}
            onSelectLayer={setSelectedLayer}
          />
          {/* Info text */}
          <div
            style={{
              textAlign: "center",
              fontSize: 12,
              color: "#8a7a6a",
              fontStyle: "italic",
            }}
          >
            {selectedLayer === "input"
              ? "Raw terrain survey input (28\u00d728)"
              : `${LAYER_DISPLAY_NAMES[selectedLayer]} \u2014 Channel ${selectedChannel}`}
            {topPrediction && (
              <span style={{ marginLeft: 12, color: "#d73027" }}>
                Peak: &quot;{EMNIST_CLASSES[topPrediction.classIndex]}&quot; (
                {(topPrediction.confidence * 100).toFixed(1)}%)
              </span>
            )}
            {isInferring && (
              <span style={{ marginLeft: 8, color: "#78c679" }}>
                Surveying...
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Center Section: ContourMap + TerrainFlyover */}
      <div
        style={{
          display: "flex",
          gap: 16,
          padding: "8px 24px",
          flexWrap: "wrap",
          justifyContent: "center",
        }}
      >
        {/* Left: Main Contour Map + Erosion Timeline */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 12,
            flex: "1 1 55%",
            minWidth: 350,
            maxWidth: 600,
          }}
        >
          {/* Main ContourMap Panel */}
          <div
            style={{
              background: "#2a2018",
              borderRadius: 6,
              border: "1px solid #4a3828",
              padding: 12,
            }}
          >
            <div
              style={{
                fontSize: 11,
                fontVariant: "small-caps",
                color: "#8a7a6a",
                marginBottom: 8,
                letterSpacing: 1,
              }}
            >
              Topographic Survey &mdash;{" "}
              {LAYER_DISPLAY_NAMES[selectedLayer]}
              {selectedLayer !== "input" && ` / Ch.${selectedChannel}`}
            </div>
            <ContourMap
              data={mainData}
              width={contourSize}
              height={contourSize}
              showContours
            />
          </div>

          {/* Erosion Timeline */}
          <div
            style={{
              background: "#2a2018",
              borderRadius: 6,
              border: "1px solid #4a3828",
              padding: 12,
            }}
          >
            <div
              style={{
                fontSize: 11,
                fontVariant: "small-caps",
                color: "#8a7a6a",
                marginBottom: 6,
                letterSpacing: 1,
              }}
            >
              Erosion Timeline &mdash; Channel {selectedChannel}
            </div>
            <ErosionTimeline selectedChannel={selectedChannel} />
          </div>
        </div>

        {/* Right: Terrain Flyover + Channel Atlas */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 12,
            flex: "1 1 40%",
            minWidth: 300,
            maxWidth: 440,
          }}
        >
          {/* 3D Terrain Flyover */}
          <div
            style={{
              background: "#2a2018",
              borderRadius: 6,
              border: "1px solid #4a3828",
              padding: 12,
            }}
          >
            <div
              style={{
                fontSize: 11,
                fontVariant: "small-caps",
                color: "#8a7a6a",
                marginBottom: 8,
                letterSpacing: 1,
              }}
            >
              3D Terrain Flyover
            </div>
            <TerrainFlyoverDynamic data={mainData} />
          </div>

          {/* Channel Atlas */}
          <div
            style={{
              background: "#2a2018",
              borderRadius: 6,
              border: "1px solid #4a3828",
              padding: 12,
            }}
          >
            <div
              style={{
                fontSize: 11,
                fontVariant: "small-caps",
                color: "#8a7a6a",
                marginBottom: 6,
                letterSpacing: 1,
              }}
            >
              Channel Atlas &mdash;{" "}
              {LAYER_DISPLAY_NAMES[selectedLayer]}
            </div>
            <ChannelAtlas
              layerKey={selectedLayer}
              selectedChannel={selectedChannel}
              onSelectChannel={setSelectedChannel}
            />
          </div>
        </div>
      </div>

      {/* Bottom: Summit Prediction */}
      <div
        style={{
          padding: "8px 24px 60px",
        }}
      >
        <div
          style={{
            background: "#2a2018",
            borderRadius: 6,
            border: "1px solid #4a3828",
            padding: 12,
          }}
        >
          <div
            style={{
              fontSize: 11,
              fontVariant: "small-caps",
              color: "#8a7a6a",
              marginBottom: 8,
              letterSpacing: 1,
            }}
          >
            Summit Prediction Range
          </div>
          <SummitPrediction />
        </div>
      </div>

      {/* Layout Navigation */}
      <LayoutNav />
    </div>
  );
}
