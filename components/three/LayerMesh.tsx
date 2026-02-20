"use client";

import { useRef, useMemo, useEffect } from "react";
import { useFrame } from "@react-three/fiber";
import { Text } from "@react-three/drei";
import * as THREE from "three";
import type { LayerMeta } from "@/lib/model/layerInfo";

const TEX_SIZE = 256;

// --- Viridis color scale (pre-computed LUT) ---
const VIRIDIS_STOPS: [number, number, number, number][] = [
  [0.0, 68, 1, 84],
  [0.13, 72, 36, 117],
  [0.25, 59, 82, 139],
  [0.38, 44, 114, 142],
  [0.5, 33, 145, 140],
  [0.63, 42, 176, 127],
  [0.75, 94, 201, 98],
  [0.88, 170, 220, 50],
  [1.0, 253, 231, 37],
];

function viridisRGB(t: number): [number, number, number] {
  t = Math.max(0, Math.min(1, t));
  for (let i = 1; i < VIRIDIS_STOPS.length; i++) {
    if (t <= VIRIDIS_STOPS[i][0]) {
      const [pt, pr, pg, pb] = VIRIDIS_STOPS[i - 1];
      const [nt, nr, ng, nb] = VIRIDIS_STOPS[i];
      const f = (t - pt) / (nt - pt);
      return [
        Math.round(pr + f * (nr - pr)),
        Math.round(pg + f * (ng - pg)),
        Math.round(pb + f * (nb - pb)),
      ];
    }
  }
  return [253, 231, 37];
}

// --- Texture creation ---

function createConvTexture(channels: number[][][]): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = TEX_SIZE;
  canvas.height = TEX_SIZE;
  const ctx = canvas.getContext("2d")!;

  ctx.fillStyle = "#08080c";
  ctx.fillRect(0, 0, TEX_SIZE, TEX_SIZE);

  const numShow = Math.min(channels.length, 16);
  const cols = 4;
  const rows = Math.ceil(numShow / cols);

  // Sort channels by total activation (show most active first)
  const sorted = channels
    .map((ch, i) => {
      let sum = 0;
      for (const row of ch) for (const v of row) sum += Math.abs(v);
      return { i, sum };
    })
    .sort((a, b) => b.sum - a.sum);

  const gap = 3;
  const cellW = (TEX_SIZE - gap * (cols + 1)) / cols;
  const cellH = (TEX_SIZE - gap * (rows + 1)) / rows;
  const h = channels[0].length;
  const w = channels[0][0].length;

  // Global max for consistent normalization
  let maxVal = 0;
  for (let gi = 0; gi < numShow; gi++) {
    for (const row of channels[sorted[gi].i])
      for (const v of row) {
        const a = Math.abs(v);
        if (a > maxVal) maxVal = a;
      }
  }
  maxVal = Math.max(maxVal, 0.001);

  const pixW = cellW / w;
  const pixH = cellH / h;

  for (let gi = 0; gi < numShow; gi++) {
    const col = gi % cols;
    const row = Math.floor(gi / cols);
    const ox = gap + col * (cellW + gap);
    const oy = gap + row * (cellH + gap);
    const ch = channels[sorted[gi].i];

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const [r, g, b] = viridisRGB(Math.abs(ch[y][x]) / maxVal);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(
          ox + x * pixW,
          oy + y * pixH,
          Math.ceil(pixW),
          Math.ceil(pixH)
        );
      }
    }
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.needsUpdate = true;
  return tex;
}

function createDenseTexture(values: number[]): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = TEX_SIZE;
  canvas.height = TEX_SIZE;
  const ctx = canvas.getContext("2d")!;

  ctx.fillStyle = "#08080c";
  ctx.fillRect(0, 0, TEX_SIZE, TEX_SIZE);

  const maxVal = Math.max(...values.map(Math.abs), 0.001);
  const count = values.length;
  const cols = Math.ceil(Math.sqrt(count));
  const rows = Math.ceil(count / cols);
  const cellW = TEX_SIZE / cols;
  const cellH = TEX_SIZE / rows;

  for (let i = 0; i < count; i++) {
    const col = i % cols;
    const row = Math.floor(i / cols);
    const [r, g, b] = viridisRGB(Math.abs(values[i]) / maxVal);
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(col * cellW, row * cellH, cellW - 0.5, cellH - 0.5);
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.needsUpdate = true;
  return tex;
}

function createOutputTexture(values: number[]): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = TEX_SIZE;
  canvas.height = TEX_SIZE;
  const ctx = canvas.getContext("2d")!;

  ctx.fillStyle = "#08080c";
  ctx.fillRect(0, 0, TEX_SIZE, TEX_SIZE);

  const count = values.length;
  const maxVal = Math.max(...values, 0.001);
  const barH = Math.max((TEX_SIZE - 20) / count, 1);
  const topIdx = values.indexOf(Math.max(...values));

  for (let i = 0; i < count; i++) {
    const barW = Math.max(2, (values[i] / maxVal) * (TEX_SIZE - 40));
    const y = 10 + i * barH;

    if (i === topIdx) {
      ctx.fillStyle = "rgb(253, 231, 37)";
    } else {
      const [r, g, b] = viridisRGB(values[i] / maxVal);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
    }
    ctx.fillRect(20, y, barW, Math.max(barH - 0.5, 1));
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.needsUpdate = true;
  return tex;
}

/** Idle texture: subtle grid/noise pattern */
function createIdleTexture(layerType: string): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = TEX_SIZE;
  canvas.height = TEX_SIZE;
  const ctx = canvas.getContext("2d")!;

  ctx.fillStyle = "#0a0a14";
  ctx.fillRect(0, 0, TEX_SIZE, TEX_SIZE);

  // Subtle grid
  ctx.strokeStyle = "rgba(99, 102, 241, 0.08)";
  ctx.lineWidth = 0.5;
  const step = layerType === "dense" || layerType === "softmax" ? 16 : 8;
  for (let x = 0; x <= TEX_SIZE; x += step) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, TEX_SIZE);
    ctx.stroke();
  }
  for (let y = 0; y <= TEX_SIZE; y += step) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(TEX_SIZE, y);
    ctx.stroke();
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.needsUpdate = true;
  return tex;
}

// --- Main component ---

interface LayerMeshProps {
  layer: LayerMeta;
  activations: number[][][] | number[] | undefined;
  position: [number, number, number];
}

export function LayerMesh({ layer, activations, position }: LayerMeshProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const hasActivation = activations !== undefined;

  const dimensions = useMemo((): [number, number, number] => {
    const shape = layer.outputShape;
    const scale = 0.08;
    switch (layer.type) {
      case "conv2d":
      case "relu":
      case "batchnorm":
      case "maxpool": {
        const [h, w, c] = shape;
        return [w * scale, h * scale, Math.max(c * scale * 0.05, 0.15)];
      }
      case "dense":
      case "softmax": {
        const units = shape[0];
        return [Math.min(units * 0.015, 3), 0.6, 0.3];
      }
      case "flatten":
        return [2, 0.3, 0.3];
      default:
        return [1, 1, 0.2];
    }
  }, [layer]);

  // Create activation texture (or idle texture)
  const texture = useMemo(() => {
    if (!activations) return createIdleTexture(layer.type);

    // 3D array → conv/relu/pool
    if (
      Array.isArray(activations) &&
      Array.isArray(activations[0]) &&
      Array.isArray((activations as number[][][])[0][0])
    ) {
      return createConvTexture(activations as number[][][]);
    }
    // 1D array → dense or output
    if (layer.type === "softmax") {
      return createOutputTexture(activations as number[]);
    }
    return createDenseTexture(activations as number[]);
  }, [activations, layer.type]);

  // Dispose texture on change / unmount
  useEffect(() => {
    return () => {
      texture?.dispose();
    };
  }, [texture]);

  // Average intensity for glow
  const intensity = useMemo(() => {
    if (!activations) return 0;
    if (Array.isArray(activations) && typeof activations[0] === "number") {
      const arr = activations as number[];
      return Math.min(1, arr.reduce((a, b) => a + Math.abs(b), 0) / arr.length);
    }
    if (Array.isArray(activations) && Array.isArray(activations[0])) {
      const ch0 = (activations as number[][][])[0];
      const flat = ch0.flat();
      return Math.min(1, flat.reduce((a, b) => a + Math.abs(b), 0) / flat.length);
    }
    return 0;
  }, [activations]);

  useFrame((state) => {
    if (!meshRef.current) return;
    meshRef.current.rotation.y =
      Math.sin(state.clock.elapsedTime * 0.3 + position[2] * 0.5) * 0.03;
  });

  return (
    <group position={position}>
      {/* Main mesh with activation texture */}
      <mesh ref={meshRef}>
        <boxGeometry args={dimensions} />
        <meshStandardMaterial
          color={hasActivation ? "#0a0a12" : "#111118"}
          emissiveMap={texture}
          emissive="#ffffff"
          emissiveIntensity={hasActivation ? 0.7 + intensity * 0.3 : 0.15}
          transparent
          opacity={hasActivation ? 0.95 : 0.6}
        />
      </mesh>

      {/* Glow outline when activated */}
      {hasActivation && (
        <mesh>
          <boxGeometry
            args={[
              dimensions[0] + 0.08,
              dimensions[1] + 0.08,
              dimensions[2] + 0.08,
            ]}
          />
          <meshBasicMaterial
            color="#6366f1"
            transparent
            opacity={0.06 + intensity * 0.14}
            wireframe
          />
        </mesh>
      )}

      {/* Label */}
      <Text
        position={[0, dimensions[1] / 2 + 0.4, 0]}
        fontSize={0.22}
        color={hasActivation ? "#e8e8ed" : "#6a6a7a"}
        anchorX="center"
        anchorY="bottom"
        font={undefined}
      >
        {layer.displayName}
      </Text>

      {/* Shape label */}
      <Text
        position={[0, -dimensions[1] / 2 - 0.2, 0]}
        fontSize={0.15}
        color="#5a5a6a"
        anchorX="center"
        anchorY="top"
        font={undefined}
      >
        {layer.outputShape.join("\u00d7")}
      </Text>
    </group>
  );
}
