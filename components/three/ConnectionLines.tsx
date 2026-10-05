"use client";

import { useRef, useMemo, useEffect } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import type { LayerMeta } from "@/lib/model/layerInfo";

interface ConnectionLinesProps {
  layers: (LayerMeta & { position: [number, number, number] })[];
  hasData: boolean;
}

export function ConnectionLines({ layers, hasData }: ConnectionLinesProps) {
  const groupRef = useRef<THREE.Group>(null);

  const lines = useMemo(() => {
    const result: {
      geometry: THREE.BufferGeometry;
      key: string;
    }[] = [];

    for (let i = 0; i < layers.length - 1; i++) {
      const from = layers[i].position;
      const to = layers[i + 1].position;

      // Create 5 connection lines with varying spread
      for (let j = -2; j <= 2; j++) {
        const yOffset = j * 0.25;
        const xOffset = j * 0.15;
        result.push({
          key: `${i}-${j}`,
          geometry: new THREE.BufferGeometry().setFromPoints(
            new THREE.QuadraticBezierCurve3(
              new THREE.Vector3(from[0] + xOffset, from[1] + yOffset, from[2]),
              new THREE.Vector3(
                (from[0] + to[0]) / 2 + xOffset * 0.5,
                (from[1] + to[1]) / 2 + yOffset * 0.5,
                (from[2] + to[2]) / 2
              ),
              new THREE.Vector3(to[0] + xOffset, to[1] + yOffset, to[2])
            ).getPoints(30)
          ),
        });
      }
    }

    return result;
  }, [layers]);

  useEffect(() => () => lines.forEach((l) => l.geometry.dispose()), [lines]);

  // Animate opacity pulse when data is flowing
  useFrame((state) => {
    if (!groupRef.current || !hasData) return;
    const pulse =
      0.15 + Math.sin(state.clock.elapsedTime * 2) * 0.05;
    groupRef.current.children.forEach((child) => {
      const line = child as THREE.Line;
      if (line.material instanceof THREE.LineBasicMaterial) {
        line.material.opacity = pulse;
      }
    });
  });

  return (
    <group ref={groupRef}>
      {lines.map((line) => (
        <line key={line.key}>
          <primitive object={line.geometry} attach="geometry" />
          <lineBasicMaterial
            color={hasData ? "#818cf8" : "#6366f1"}
            transparent
            opacity={hasData ? 0.15 : 0.04}
          />
        </line>
      ))}
    </group>
  );
}
