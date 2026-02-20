"use client";

import {
  useRef,
  useState,
  useCallback,
  useMemo,
  useEffect,
  type RefObject,
} from "react";
import dynamic from "next/dynamic";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Text, Line } from "@react-three/drei";
import * as THREE from "three";
import { useInferenceStore } from "@/stores/inferenceStore";
import { EMNIST_CLASSES, CLASS_GROUPS } from "@/lib/model/classes";
import { useDrawingCanvas } from "@/hooks/useDrawingCanvas";
import { useInference } from "@/hooks/useInference";
import { LayoutNav } from "@/components/layouts/LayoutNav";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type FocusLayer = "input" | "conv" | "pool" | "dense" | "output";

interface CosmosSceneProps {
  focusLayer: FocusLayer;
  setFocusLayer: (layer: FocusLayer) => void;
}

// ---------------------------------------------------------------------------
// Focus Y positions for camera targeting
// ---------------------------------------------------------------------------

const FOCUS_Y: Record<FocusLayer, number> = {
  input: 0,
  conv: 8,
  pool: 20,
  dense: 32,
  output: 40,
};

// ---------------------------------------------------------------------------
// BackgroundStars
// ---------------------------------------------------------------------------

function BackgroundStars() {
  const { positions, colors } = useMemo(() => {
    const count = 2000;
    const pos = new Float32Array(count * 3);
    const col = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const r = 40 + Math.random() * 40;
      pos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      pos[i * 3 + 2] = r * Math.cos(phi);

      const brightness = 0.5 + Math.random() * 0.5;
      col[i * 3] = brightness;
      col[i * 3 + 1] = brightness;
      col[i * 3 + 2] = brightness;
    }
    return { positions: pos, colors: col };
  }, []);

  return (
    <points>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
          count={2000}
        />
        <bufferAttribute
          attach="attributes-color"
          args={[colors, 3]}
          count={2000}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.1}
        vertexColors
        transparent
        opacity={0.8}
        depthWrite={false}
      />
    </points>
  );
}

// ---------------------------------------------------------------------------
// InputGalaxy
// ---------------------------------------------------------------------------

function InputGalaxy() {
  const inputTensor = useInferenceStore((s) => s.inputTensor);
  const pointsRef = useRef<THREE.Points>(null);
  const count = 784;

  const { positions, colors } = useMemo(() => {
    const pos = new Float32Array(count * 3);
    const col = new Float32Array(count * 3);
    const spacing = 0.3;
    const offset = (27 * spacing) / 2;

    for (let row = 0; row < 28; row++) {
      for (let c = 0; c < 28; c++) {
        const idx = row * 28 + c;
        const px = inputTensor ? inputTensor[row][c] : 0.05;
        pos[idx * 3] = c * spacing - offset;
        pos[idx * 3 + 1] = 0;
        pos[idx * 3 + 2] = row * spacing - offset;

        const brightness = inputTensor ? Math.max(px, 0.05) : 0.05;
        col[idx * 3] = 0;
        col[idx * 3 + 1] = 0.898 * brightness;
        col[idx * 3 + 2] = 1.0 * brightness;
      }
    }
    return { positions: pos, colors: col };
  }, [inputTensor]);

  useFrame(({ clock }) => {
    if (!pointsRef.current) return;
    const posAttr = pointsRef.current.geometry.attributes
      .position as THREE.BufferAttribute;
    const posArray = posAttr.array as Float32Array;
    const t = clock.getElapsedTime();
    const spacing = 0.3;
    const offset = (27 * spacing) / 2;

    for (let i = 0; i < count; i++) {
      const row = Math.floor(i / 28);
      posArray[i * 3 + 1] = Math.sin(t * 0.5 + i * 0.01) * 0.02;
      posArray[i * 3] = (i % 28) * spacing - offset;
      posArray[i * 3 + 2] = row * spacing - offset;
    }
    posAttr.needsUpdate = true;
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
          count={count}
        />
        <bufferAttribute
          attach="attributes-color"
          args={[colors, 3]}
          count={count}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.08}
        vertexColors
        transparent
        opacity={0.9}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
        sizeAttenuation
      />
    </points>
  );
}

// ---------------------------------------------------------------------------
// LayerClusters
// ---------------------------------------------------------------------------

function LayerClusters() {
  const layerActivations = useInferenceStore((s) => s.layerActivations);
  const conv1Ref = useRef<THREE.Points>(null);
  const pool1Ref = useRef<THREE.Points>(null);
  const pool2Ref = useRef<THREE.Points>(null);

  // Conv1 cluster: Y=8, 256 neurons
  const conv1Data = useMemo(() => {
    const cnt = 256;
    const pos = new Float32Array(cnt * 3);
    const col = new Float32Array(cnt * 3);
    const conv1 = layerActivations.conv1 as number[][][] | undefined;
    const spacing = 0.3;
    const offset = (27 * spacing) / 2;

    for (let i = 0; i < cnt; i++) {
      const ch = i % 8;
      const spatialIdx = Math.floor(i / 8);
      const row = Math.floor(spatialIdx / 6) % 28;
      const cIdx = spatialIdx % 28;

      pos[i * 3] = cIdx * spacing - offset + (Math.random() - 0.5) * 0.2;
      pos[i * 3 + 1] = 8 + (Math.random() - 0.5) * 0.5;
      pos[i * 3 + 2] = row * spacing - offset + (Math.random() - 0.5) * 0.2;

      let val = 0.05;
      if (conv1 && conv1[ch] && conv1[ch][row % 28]) {
        val = Math.abs(conv1[ch][row % 28][cIdx % 28]) || 0.05;
      }
      const brightness = Math.min(val, 1.0);
      col[i * 3] = 0;
      col[i * 3 + 1] = 0.898 * Math.max(brightness, 0.05);
      col[i * 3 + 2] = 1.0 * Math.max(brightness, 0.05);
    }
    return { positions: pos, colors: col, count: cnt };
  }, [layerActivations.conv1]);

  // Pool1 cluster: Y=16, 196 neurons
  const pool1Data = useMemo(() => {
    const cnt = 196;
    const pos = new Float32Array(cnt * 3);
    const col = new Float32Array(cnt * 3);
    const pool1 = layerActivations.pool1 as number[][][] | undefined;
    const gridSize = 14;
    const spacing = 0.4;
    const offset = (13 * spacing) / 2;

    for (let i = 0; i < cnt; i++) {
      const row = Math.floor(i / gridSize);
      const cIdx = i % gridSize;
      pos[i * 3] = cIdx * spacing - offset + (Math.random() - 0.5) * 0.15;
      pos[i * 3 + 1] = 16 + (Math.random() - 0.5) * 0.4;
      pos[i * 3 + 2] = row * spacing - offset + (Math.random() - 0.5) * 0.15;

      let val = 0.05;
      if (pool1) {
        let sum = 0;
        const chCount = Math.min(pool1.length, 64);
        for (let ch = 0; ch < chCount; ch++) {
          if (pool1[ch] && pool1[ch][row]) {
            sum += Math.abs(pool1[ch][row][cIdx] || 0);
          }
        }
        val = Math.min(sum / chCount, 1.0) || 0.05;
      }
      const brightness = Math.max(val, 0.05);
      col[i * 3] = 0.325 * brightness;
      col[i * 3 + 1] = 0.427 * brightness;
      col[i * 3 + 2] = 0.996 * brightness;
    }
    return { positions: pos, colors: col, count: cnt };
  }, [layerActivations.pool1]);

  // Pool2 cluster: Y=24, 128 neurons
  const pool2Data = useMemo(() => {
    const cnt = 128;
    const pos = new Float32Array(cnt * 3);
    const col = new Float32Array(cnt * 3);
    const pool2 = layerActivations.pool2 as number[][][] | undefined;
    const gridSize = 7;
    const spacing = 0.5;
    const offset = (6 * spacing) / 2;

    for (let i = 0; i < cnt; i++) {
      const spatialIdx = i % 49;
      const row = Math.floor(spatialIdx / gridSize);
      const cIdx = spatialIdx % gridSize;
      pos[i * 3] = cIdx * spacing - offset + (Math.random() - 0.5) * 0.2;
      pos[i * 3 + 1] = 24 + (Math.random() - 0.5) * 0.5;
      pos[i * 3 + 2] = row * spacing - offset + (Math.random() - 0.5) * 0.2;

      let val = 0.05;
      if (pool2) {
        const ch = Math.floor(i / 49) % Math.min(pool2.length, 128);
        if (pool2[ch] && pool2[ch][row]) {
          val = Math.min(Math.abs(pool2[ch][row][cIdx] || 0), 1.0) || 0.05;
        }
      }
      const brightness = Math.max(val, 0.05);
      col[i * 3] = 0.486 * brightness;
      col[i * 3 + 1] = 0.302 * brightness;
      col[i * 3 + 2] = 1.0 * brightness;
    }
    return { positions: pos, colors: col, count: cnt };
  }, [layerActivations.pool2]);

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    if (conv1Ref.current) conv1Ref.current.rotation.y = t * 0.05;
    if (pool1Ref.current) pool1Ref.current.rotation.y = t * 0.03;
    if (pool2Ref.current) pool2Ref.current.rotation.y = t * 0.04;
  });

  return (
    <group>
      <points ref={conv1Ref}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[conv1Data.positions, 3]} count={conv1Data.count} />
          <bufferAttribute attach="attributes-color" args={[conv1Data.colors, 3]} count={conv1Data.count} />
        </bufferGeometry>
        <pointsMaterial size={0.08} vertexColors transparent opacity={0.85} blending={THREE.AdditiveBlending} depthWrite={false} sizeAttenuation />
      </points>
      <points ref={pool1Ref}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[pool1Data.positions, 3]} count={pool1Data.count} />
          <bufferAttribute attach="attributes-color" args={[pool1Data.colors, 3]} count={pool1Data.count} />
        </bufferGeometry>
        <pointsMaterial size={0.1} vertexColors transparent opacity={0.85} blending={THREE.AdditiveBlending} depthWrite={false} sizeAttenuation />
      </points>
      <points ref={pool2Ref}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[pool2Data.positions, 3]} count={pool2Data.count} />
          <bufferAttribute attach="attributes-color" args={[pool2Data.colors, 3]} count={pool2Data.count} />
        </bufferGeometry>
        <pointsMaterial size={0.12} vertexColors transparent opacity={0.85} blending={THREE.AdditiveBlending} depthWrite={false} sizeAttenuation />
      </points>
    </group>
  );
}

// ---------------------------------------------------------------------------
// DenseNebula
// ---------------------------------------------------------------------------

function DenseNebula() {
  const layerActivations = useInferenceStore((s) => s.layerActivations);
  const pointsRef = useRef<THREE.Points>(null);
  const count = 256;

  const nebulaData = useMemo(() => {
    const pos = new Float32Array(count * 3);
    const col = new Float32Array(count * 3);
    const relu4 = layerActivations.relu4 as number[] | undefined;
    const dense1 = layerActivations.dense1 as number[] | undefined;
    const activations = relu4 || dense1;

    for (let i = 0; i < count; i++) {
      const radius = 2 + (i / 256) * 4;
      const angle = i * 0.3;
      pos[i * 3] = Math.cos(angle) * radius;
      pos[i * 3 + 1] = 32 + Math.sin(i * 0.1) * 0.5;
      pos[i * 3 + 2] = Math.sin(angle) * radius;

      let val = 0.05;
      if (activations && activations[i] !== undefined) {
        val = Math.min(Math.abs(activations[i]), 1.0) || 0.05;
      }

      // Warm amber/orange: #ff6e40 to #ffd740
      const t = Math.max(val, 0.1);
      col[i * 3] = 1.0 * t;
      col[i * 3 + 1] = (0.431 + val * 0.412) * t;
      col[i * 3 + 2] = 0.251 * t;
    }
    return { positions: pos, colors: col };
  }, [layerActivations.relu4, layerActivations.dense1]);

  useFrame(({ clock }) => {
    if (!pointsRef.current) return;
    const posAttr = pointsRef.current.geometry.attributes
      .position as THREE.BufferAttribute;
    const posArray = posAttr.array as Float32Array;
    const t = clock.getElapsedTime();

    for (let i = 0; i < count; i++) {
      const radius = 2 + (i / 256) * 4;
      const angle = i * 0.3;
      posArray[i * 3 + 1] =
        32 + Math.sin(i * 0.1) * 0.5 + Math.sin(t * 2 + i * 0.5) * 0.02;
      posArray[i * 3] = Math.cos(angle + t * 0.02) * radius;
      posArray[i * 3 + 2] = Math.sin(angle + t * 0.02) * radius;
    }
    posAttr.needsUpdate = true;
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[nebulaData.positions, 3]}
          count={count}
        />
        <bufferAttribute
          attach="attributes-color"
          args={[nebulaData.colors, 3]}
          count={count}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.12}
        vertexColors
        transparent
        opacity={0.9}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
        sizeAttenuation
      />
    </points>
  );
}

// ---------------------------------------------------------------------------
// OutputPlanets
// ---------------------------------------------------------------------------

function OutputPlanets() {
  const prediction = useInferenceStore((s) => s.prediction);
  const topPrediction = useInferenceStore((s) => s.topPrediction);
  const groupRef = useRef<THREE.Group>(null);

  const planetData = useMemo(() => {
    const data: Array<{
      index: number;
      char: string;
      x: number;
      z: number;
      scale: number;
      color: string;
      isTop: boolean;
    }> = [];

    for (let i = 0; i < 62; i++) {
      const angle = (i / 62) * Math.PI * 2;
      const radius = 6;
      const prob = prediction ? prediction[i] : 0;
      const isTop = topPrediction?.classIndex === i;

      let color: string;
      if (i >= CLASS_GROUPS.digits.start && i <= CLASS_GROUPS.digits.end) {
        color = "#4fc3f7";
      } else if (i >= CLASS_GROUPS.uppercase.start && i <= CLASS_GROUPS.uppercase.end) {
        color = "#66bb6a";
      } else {
        color = "#ffa726";
      }

      data.push({
        index: i,
        char: EMNIST_CLASSES[i],
        x: Math.cos(angle) * radius,
        z: Math.sin(angle) * radius,
        scale: 0.05 + prob * 1.5,
        color,
        isTop,
      });
    }
    return data;
  }, [prediction, topPrediction]);

  useFrame(({ clock }) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = clock.getElapsedTime() * 0.1;
    }
  });

  return (
    <group ref={groupRef} position={[0, 40, 0]}>
      {planetData.map((planet) => (
        <group key={planet.index} position={[planet.x, 0, planet.z]}>
          <mesh>
            <sphereGeometry args={[planet.scale, 16, 16]} />
            <meshStandardMaterial
              color={prediction ? planet.color : "#555555"}
              emissive={planet.isTop ? "#ffd740" : "#000000"}
              emissiveIntensity={planet.isTop ? 0.8 : 0}
              roughness={0.4}
              metalness={0.3}
            />
          </mesh>

          {planet.isTop && (
            <mesh rotation={[Math.PI / 3, 0, 0]}>
              <torusGeometry args={[planet.scale * 1.6, planet.scale * 0.15, 8, 32]} />
              <meshStandardMaterial
                color="#ffd740"
                emissive="#ffd740"
                emissiveIntensity={0.6}
                transparent
                opacity={0.7}
              />
            </mesh>
          )}

          <Text
            position={[0, planet.scale + 0.4, 0]}
            fontSize={0.3}
            color="white"
            anchorX="center"
            anchorY="bottom"
          >
            {planet.char}
          </Text>
        </group>
      ))}
    </group>
  );
}

// ---------------------------------------------------------------------------
// ConstellationLines
// ---------------------------------------------------------------------------

function ConstellationLines() {
  const layerActivations = useInferenceStore((s) => s.layerActivations);
  const prediction = useInferenceStore((s) => s.prediction);
  const inputTensor = useInferenceStore((s) => s.inputTensor);

  const lines = useMemo(() => {
    const result: Array<{
      key: string;
      points: Array<[number, number, number]>;
    }> = [];

    if (!prediction) return result;

    const relu4 = layerActivations.relu4 as number[] | undefined;
    const dense1 = layerActivations.dense1 as number[] | undefined;
    const denseActs = relu4 || dense1;

    // Top 3 output indices
    const outputIndices = prediction
      .map((p, i) => ({ p, i }))
      .sort((a, b) => b.p - a.p)
      .slice(0, 3);

    if (denseActs) {
      const topDense = denseActs
        .map((v, i) => ({ v: Math.abs(v), i }))
        .sort((a, b) => b.v - a.v)
        .slice(0, 5);

      for (const dn of topDense) {
        const dnRadius = 2 + (dn.i / 256) * 4;
        const dnAngle = dn.i * 0.3;
        const dnX = Math.cos(dnAngle) * dnRadius;
        const dnZ = Math.sin(dnAngle) * dnRadius;

        for (const op of outputIndices) {
          const opAngle = (op.i / 62) * Math.PI * 2;
          const opRadius = 6;
          result.push({
            key: `d${dn.i}-o${op.i}`,
            points: [
              [dnX, 32, dnZ],
              [Math.cos(opAngle) * opRadius, 40, Math.sin(opAngle) * opRadius],
            ],
          });
        }
      }
    }

    if (inputTensor) {
      const brightPixels: { row: number; col: number; val: number }[] = [];
      for (let r = 0; r < 28; r++) {
        for (let c = 0; c < 28; c++) {
          if (inputTensor[r][c] > 0.5) {
            brightPixels.push({ row: r, col: c, val: inputTensor[r][c] });
          }
        }
      }
      brightPixels.sort((a, b) => b.val - a.val);
      const topPixels = brightPixels.slice(0, 3);
      const spacing = 0.3;
      const offset = (27 * spacing) / 2;

      for (let pi = 0; pi < topPixels.length; pi++) {
        const px = topPixels[pi];
        const inputX = px.col * spacing - offset;
        const inputZ = px.row * spacing - offset;

        result.push({
          key: `i${px.row}_${px.col}-c1_${pi}`,
          points: [
            [inputX, 0, inputZ],
            [inputX + (Math.random() - 0.5) * 0.5, 8, inputZ + (Math.random() - 0.5) * 0.5],
          ],
        });
      }
    }

    return result;
  }, [prediction, layerActivations.relu4, layerActivations.dense1, inputTensor]);

  if (lines.length === 0) return null;

  return (
    <group>
      {lines.map((line) => (
        <Line
          key={line.key}
          points={line.points}
          color="white"
          lineWidth={1}
          transparent
          opacity={0.2}
        />
      ))}
    </group>
  );
}

// ---------------------------------------------------------------------------
// CameraController
// ---------------------------------------------------------------------------

function CameraController({
  focusLayer,
  controlsRef,
}: {
  focusLayer: FocusLayer;
  controlsRef: RefObject<{ target: THREE.Vector3; update: () => void } | null>;
}) {
  const targetY = useRef(FOCUS_Y[focusLayer]);

  useEffect(() => {
    targetY.current = FOCUS_Y[focusLayer];
  }, [focusLayer]);

  useFrame(({ camera }) => {
    if (!controlsRef.current) return;
    const controls = controlsRef.current;
    const currentTargetY = controls.target.y;
    const newTargetY = currentTargetY + (targetY.current - currentTargetY) * 0.03;
    controls.target.y = newTargetY;

    const camTargetY = newTargetY + 12;
    camera.position.y += (camTargetY - camera.position.y) * 0.03;

    controls.update();
  });

  return null;
}

// ---------------------------------------------------------------------------
// CosmosScene (the full R3F scene, will be dynamically imported)
// ---------------------------------------------------------------------------

function CosmosScene({ focusLayer, setFocusLayer }: CosmosSceneProps) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const controlsRef = useRef<any>(null);

  const layers: FocusLayer[] = ["input", "conv", "pool", "dense", "output"];
  const labels = ["Input", "Conv", "Pool", "Dense", "Output"];

  return (
    <>
      <Canvas
        style={{ width: "100%", height: "100%", background: "#000008" }}
        camera={{ position: [15, 12, 15], fov: 60, near: 0.1, far: 200 }}
        gl={{ antialias: true }}
      >
        <fog attach="fog" args={["#000008", 30, 100]} />
        <ambientLight intensity={0.2} />
        <pointLight position={[20, 10, 0]} intensity={0.8} color="#536dfe" />
        <pointLight position={[-10, -5, 20]} intensity={0.6} color="#00e5ff" />

        <BackgroundStars />
        <InputGalaxy />
        <LayerClusters />
        <DenseNebula />
        <OutputPlanets />
        <ConstellationLines />

        <OrbitControls
          ref={controlsRef}
          autoRotate
          autoRotateSpeed={0.15}
          enablePan
          minDistance={5}
          maxDistance={80}
          target={[0, 20, 0]}
        />
        <CameraController focusLayer={focusLayer} controlsRef={controlsRef} />
      </Canvas>

      {/* Layer focus buttons overlay */}
      <div
        style={{
          position: "absolute",
          bottom: 64,
          left: "50%",
          transform: "translateX(-50%)",
          display: "flex",
          gap: 6,
          zIndex: 20,
        }}
      >
        {layers.map((layer, idx) => (
          <button
            key={layer}
            onClick={() => setFocusLayer(layer)}
            style={{
              fontFamily: "monospace",
              fontSize: 11,
              color: "#00e5ff",
              background: "rgba(0, 8, 20, 0.6)",
              border: "1px solid rgba(0, 229, 255, 0.3)",
              borderRadius: 4,
              padding: "4px 10px",
              cursor: "pointer",
              backdropFilter: "blur(8px)",
              transition: "all 0.2s",
            }}
            onMouseEnter={(e) => {
              const btn = e.currentTarget;
              btn.style.background = "rgba(0, 229, 255, 0.15)";
              btn.style.borderColor = "rgba(0, 229, 255, 0.6)";
            }}
            onMouseLeave={(e) => {
              const btn = e.currentTarget;
              btn.style.background = "rgba(0, 8, 20, 0.6)";
              btn.style.borderColor = "rgba(0, 229, 255, 0.3)";
            }}
          >
            {labels[idx]}
          </button>
        ))}
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------
// Dynamic import wrapper for SSR safety
// ---------------------------------------------------------------------------

const CosmosSceneDynamic = dynamic(
  () => Promise.resolve({ default: CosmosScene }),
  { ssr: false }
);

// ---------------------------------------------------------------------------
// DrawingPanel (HTML overlay, bottom-left)
// ---------------------------------------------------------------------------

const DRAWING_CANVAS_SIZE = 180;
const DRAWING_INTERNAL_SIZE = 280;

function DrawingPanel() {
  const { infer } = useInference();
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
        position: "absolute",
        bottom: 80,
        left: 20,
        zIndex: 20,
        background: "rgba(0, 2, 12, 0.75)",
        border: "1px solid rgba(0, 229, 255, 0.3)",
        borderRadius: 8,
        padding: 14,
        backdropFilter: "blur(12px)",
        fontFamily: "monospace",
      }}
    >
      {/* Corner bracket decorations */}
      <div style={{ position: "absolute", top: -1, left: -1, width: 12, height: 12, borderTop: "2px solid #00e5ff", borderLeft: "2px solid #00e5ff" }} />
      <div style={{ position: "absolute", top: -1, right: -1, width: 12, height: 12, borderTop: "2px solid #00e5ff", borderRight: "2px solid #00e5ff" }} />
      <div style={{ position: "absolute", bottom: -1, left: -1, width: 12, height: 12, borderBottom: "2px solid #00e5ff", borderLeft: "2px solid #00e5ff" }} />
      <div style={{ position: "absolute", bottom: -1, right: -1, width: 12, height: 12, borderBottom: "2px solid #00e5ff", borderRight: "2px solid #00e5ff" }} />

      {/* Label */}
      <div
        style={{
          fontSize: 10,
          color: "#00e5ff",
          letterSpacing: 3,
          marginBottom: 8,
          textAlign: "center",
        }}
      >
        INPUT SIGNAL
      </div>

      {/* Canvas */}
      <div
        style={{
          position: "relative",
          borderRadius: 4,
          overflow: "hidden",
          border: "1px solid rgba(0, 229, 255, 0.2)",
        }}
      >
        <canvas
          ref={canvasRef}
          width={DRAWING_INTERNAL_SIZE}
          height={DRAWING_INTERNAL_SIZE}
          style={{
            width: DRAWING_CANVAS_SIZE,
            height: DRAWING_CANVAS_SIZE,
            display: "block",
            cursor: "crosshair",
            touchAction: "none",
          }}
          onMouseDown={(e) => startDrawing(e.nativeEvent)}
          onMouseMove={(e) => draw(e.nativeEvent)}
          onMouseUp={() => stopDrawing()}
          onMouseLeave={() => stopDrawing()}
          onTouchStart={(e) => { e.preventDefault(); startDrawing(e.nativeEvent); }}
          onTouchMove={(e) => { e.preventDefault(); draw(e.nativeEvent); }}
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
            <span style={{ fontSize: 11, color: "rgba(0, 229, 255, 0.25)", fontFamily: "monospace" }}>
              Draw here
            </span>
          </div>
        )}
      </div>

      {/* Clear button */}
      <button
        onClick={handleClear}
        style={{
          marginTop: 8,
          width: "100%",
          fontFamily: "monospace",
          fontSize: 10,
          color: "#00e5ff",
          background: "rgba(0, 229, 255, 0.08)",
          border: "1px solid rgba(0, 229, 255, 0.3)",
          borderRadius: 4,
          padding: "4px 0",
          cursor: "pointer",
          letterSpacing: 2,
          transition: "all 0.2s",
        }}
      >
        RESET
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// InfoDisplay (HTML overlay, top-right)
// ---------------------------------------------------------------------------

function InfoDisplay({ focusLayer }: { focusLayer: FocusLayer }) {
  const topPrediction = useInferenceStore((s) => s.topPrediction);
  const isInferring = useInferenceStore((s) => s.isInferring);

  const predictedChar = topPrediction
    ? EMNIST_CLASSES[topPrediction.classIndex] ?? "?"
    : null;

  const focusLabels: Record<FocusLayer, string> = {
    input: "INPUT GALAXY",
    conv: "CONVOLUTION CLUSTERS",
    pool: "POOLING FORMATIONS",
    dense: "DENSE NEBULA",
    output: "OUTPUT ORBITS",
  };

  return (
    <div
      style={{
        position: "absolute",
        top: 20,
        right: 20,
        zIndex: 20,
        background: "rgba(0, 2, 12, 0.75)",
        border: "1px solid rgba(0, 229, 255, 0.3)",
        borderRadius: 8,
        padding: "12px 16px",
        backdropFilter: "blur(12px)",
        fontFamily: "monospace",
        minWidth: 160,
      }}
    >
      {/* Corner brackets */}
      <div style={{ position: "absolute", top: -1, left: -1, width: 10, height: 10, borderTop: "2px solid #00e5ff", borderLeft: "2px solid #00e5ff" }} />
      <div style={{ position: "absolute", top: -1, right: -1, width: 10, height: 10, borderTop: "2px solid #00e5ff", borderRight: "2px solid #00e5ff" }} />
      <div style={{ position: "absolute", bottom: -1, left: -1, width: 10, height: 10, borderBottom: "2px solid #00e5ff", borderLeft: "2px solid #00e5ff" }} />
      <div style={{ position: "absolute", bottom: -1, right: -1, width: 10, height: 10, borderBottom: "2px solid #00e5ff", borderRight: "2px solid #00e5ff" }} />

      {/* Focus layer label */}
      <div style={{ fontSize: 9, color: "#00e5ff", letterSpacing: 2, marginBottom: 8, opacity: 0.7 }}>
        {focusLabels[focusLayer]}
      </div>

      {/* Predicted character */}
      {predictedChar ? (
        <div
          style={{
            fontSize: 42,
            fontWeight: "bold",
            color: "#ffd740",
            textAlign: "center",
            lineHeight: 1,
            textShadow: "0 0 20px rgba(255, 215, 64, 0.5)",
          }}
        >
          {predictedChar}
        </div>
      ) : (
        <div style={{ fontSize: 14, color: "rgba(255,255,255,0.2)", textAlign: "center" }}>
          ---
        </div>
      )}

      {/* Confidence */}
      {topPrediction && (
        <div style={{ fontSize: 11, color: "rgba(255, 255, 255, 0.6)", textAlign: "center", marginTop: 4 }}>
          {(topPrediction.confidence * 100).toFixed(1)}%
        </div>
      )}

      {/* Inferring indicator */}
      {isInferring && (
        <div style={{ fontSize: 9, color: "#00e5ff", textAlign: "center", marginTop: 6, opacity: 0.8 }}>
          PROCESSING...
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main exported component
// ---------------------------------------------------------------------------

export function ParticleCosmosLayout() {
  const [focusLayer, setFocusLayer] = useState<FocusLayer>("input");

  return (
    <div
      style={{
        position: "relative",
        width: "100vw",
        height: "100vh",
        background: "#000008",
        overflow: "hidden",
      }}
    >
      {/* Full viewport 3D scene */}
      <div style={{ position: "absolute", inset: 0 }}>
        <CosmosSceneDynamic
          focusLayer={focusLayer}
          setFocusLayer={setFocusLayer}
        />
      </div>

      {/* HTML overlays */}
      <DrawingPanel />
      <InfoDisplay focusLayer={focusLayer} />

      {/* Layout navigation */}
      <LayoutNav />
    </div>
  );
}
