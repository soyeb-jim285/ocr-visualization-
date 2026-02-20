"use client";

import { Suspense } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Environment, PerspectiveCamera } from "@react-three/drei";
import { LayerMesh } from "./LayerMesh";
import { ConnectionLines } from "./ConnectionLines";
import { DataFlowParticles } from "./DataFlowParticles";
import { useInferenceStore } from "@/stores/inferenceStore";
import { LAYER_CONFIG, getVisualizableLayers } from "@/lib/model/layerInfo";

// Position layers along Z-axis
const LAYER_SPACING = 3;

export function NetworkScene() {
  const layerActivations = useInferenceStore((s) => s.layerActivations);
  const hasData = Object.keys(layerActivations).length > 0;

  const visualizableLayers = getVisualizableLayers();

  const layerPositions = visualizableLayers.map((layer, i) => ({
    ...layer,
    position: [0, 0, i * LAYER_SPACING] as [number, number, number],
    index: i,
  }));

  return (
    <div className="h-[70vh] w-full overflow-hidden rounded-2xl border border-border bg-[#08080c]">
      <Canvas>
        <Suspense fallback={null}>
          <PerspectiveCamera
            makeDefault
            position={[8, 6, -5]}
            fov={50}
          />

          {/* Lighting */}
          <ambientLight intensity={0.3} />
          <pointLight position={[10, 10, 5]} intensity={0.6} color="#6366f1" />
          <pointLight position={[-10, -5, 15]} intensity={0.3} color="#06b6d4" />

          {/* Environment */}
          <fog attach="fog" args={["#08080c", 20, 60]} />

          {/* Layer meshes */}
          {layerPositions.map((layer) => (
            <LayerMesh
              key={layer.name}
              layer={layer}
              activations={layerActivations[layer.name]}
              position={layer.position}
            />
          ))}

          {/* Connections between layers */}
          <ConnectionLines layers={layerPositions} hasData={hasData} />

          {/* Data flow particles */}
          {hasData && <DataFlowParticles layers={layerPositions} />}

          {/* Controls */}
          <OrbitControls
            enablePan={false}
            minDistance={5}
            maxDistance={40}
            autoRotate
            autoRotateSpeed={0.3}
            target={[0, 0, (layerPositions.length * LAYER_SPACING) / 2]}
          />
        </Suspense>
      </Canvas>
    </div>
  );
}
