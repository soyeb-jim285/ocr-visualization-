"use client";

import { useRef, useMemo } from "react";
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
      points: THREE.Vector3[];
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
          points: [
            new THREE.Vector3(from[0] + xOffset, from[1] + yOffset, from[2]),
            new THREE.Vector3(
              (from[0] + to[0]) / 2 + xOffset * 0.5,
              (from[1] + to[1]) / 2 + yOffset * 0.5,
              (from[2] + to[2]) / 2
            ),
            new THREE.Vector3(to[0] + xOffset, to[1] + yOffset, to[2]),
          ],
        });
      }
    }

    return result;
  }, [layers]);

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
      {lines.map((line) => {
        const curve = new THREE.QuadraticBezierCurve3(
          line.points[0],
          line.points[1],
          line.points[2]
        );
        const points = curve.getPoints(30);
        const geometry = new THREE.BufferGeometry().setFromPoints(points);

        return (
          <line key={line.key}>
            <primitive object={geometry} attach="geometry" />
            <lineBasicMaterial
              color={hasData ? "#818cf8" : "#6366f1"}
              transparent
              opacity={hasData ? 0.15 : 0.04}
            />
          </line>
        );
      })}
    </group>
  );
}
