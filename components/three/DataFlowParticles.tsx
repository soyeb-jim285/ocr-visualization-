"use client";

import { useRef, useMemo } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import type { LayerMeta } from "@/lib/model/layerInfo";

const PARTICLE_COUNT = 400;

interface DataFlowParticlesProps {
  layers: (LayerMeta & { position: [number, number, number] })[];
}

export function DataFlowParticles({ layers }: DataFlowParticlesProps) {
  const pointsRef = useRef<THREE.Points>(null);

  const totalLength = useMemo(() => {
    if (layers.length < 2) return 0;
    return layers[layers.length - 1].position[2] - layers[0].position[2];
  }, [layers]);

  const { positions, speeds, colors } = useMemo(() => {
    const positions = new Float32Array(PARTICLE_COUNT * 3);
    const speeds = new Float32Array(PARTICLE_COUNT);
    const colors = new Float32Array(PARTICLE_COUNT * 3);

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 3;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 3;
      positions[i * 3 + 2] = Math.random() * totalLength;
      speeds[i] = 0.015 + Math.random() * 0.04;

      // Gradient: cyan at start → indigo → purple at end
      const progress = positions[i * 3 + 2] / Math.max(totalLength, 1);
      colors[i * 3] = 0.02 + progress * 0.37; // R: 0.02 → 0.39
      colors[i * 3 + 1] = 0.71 - progress * 0.31; // G: 0.71 → 0.40
      colors[i * 3 + 2] = 0.83 - progress * 0.04; // B: 0.83 → 0.79
    }

    return { positions, speeds, colors };
  }, [totalLength]);

  useFrame(() => {
    if (!pointsRef.current) return;
    const posAttr = pointsRef.current.geometry.attributes
      .position as THREE.BufferAttribute;
    const colorAttr = pointsRef.current.geometry.attributes
      .color as THREE.BufferAttribute;
    const posArray = posAttr.array as Float32Array;
    const colorArray = colorAttr.array as Float32Array;

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      posArray[i * 3 + 2] += speeds[i];

      if (posArray[i * 3 + 2] > totalLength) {
        posArray[i * 3] = (Math.random() - 0.5) * 3;
        posArray[i * 3 + 1] = (Math.random() - 0.5) * 3;
        posArray[i * 3 + 2] = 0;
      }

      // Converge toward center
      posArray[i * 3] *= 0.997;
      posArray[i * 3 + 1] *= 0.997;

      // Update color based on progress
      const progress = posArray[i * 3 + 2] / Math.max(totalLength, 1);
      colorArray[i * 3] = 0.02 + progress * 0.37;
      colorArray[i * 3 + 1] = 0.71 - progress * 0.31;
      colorArray[i * 3 + 2] = 0.83 - progress * 0.04;
    }

    posAttr.needsUpdate = true;
    colorAttr.needsUpdate = true;
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
          count={PARTICLE_COUNT}
        />
        <bufferAttribute
          attach="attributes-color"
          args={[colors, 3]}
          count={PARTICLE_COUNT}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.08}
        vertexColors
        transparent
        opacity={0.8}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </points>
  );
}
