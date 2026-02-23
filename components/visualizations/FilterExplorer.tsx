"use client";

import { useState, useMemo, useCallback } from "react";
import { useInferenceStore } from "@/stores/inferenceStore";
import { useConv1Weights } from "@/hooks/useConv1Weights";
import { PatchGrid } from "@/components/visualizations/PatchGrid";
import { ActivationHeatmap } from "@/components/visualizations/ActivationHeatmap";
import { divergingRGB } from "@/lib/utils/colorScales";

export function FilterExplorer() {
  const layerActivations = useInferenceStore((s) => s.layerActivations);
  const conv1Maps = layerActivations["conv1"] as number[][][] | undefined;
  const { kernels, biases, loading } = useConv1Weights();

  const [selectedFilter, setSelectedFilter] = useState(0);

  // Global absolute max across all kernels for consistent coloring
  const globalAbsMax = useMemo(() => {
    if (!kernels) return 1;
    let m = 0;
    for (const k of kernels)
      for (const row of k) for (const v of row) m = Math.max(m, Math.abs(v));
    return m;
  }, [kernels]);

  const kernelColorFn = useCallback(
    (val: number): [number, number, number] => divergingRGB(val, globalAbsMax),
    [globalAbsMax]
  );

  if (loading || !kernels) {
    return (
      <div className="flex h-48 items-center justify-center">
        <p className="text-foreground/30">Loading kernel weights...</p>
      </div>
    );
  }

  const selectedKernel = kernels[selectedFilter];
  const selectedBias = biases?.[selectedFilter] ?? 0;
  const selectedMap = conv1Maps?.[selectedFilter];

  return (
    <div className="flex flex-col gap-8">
      {/* Selected kernel detail */}
      <div className="flex flex-col items-center gap-6 sm:flex-row sm:items-start sm:justify-center sm:gap-12">
        {/* Kernel with values */}
        <div className="flex flex-col items-center gap-2">
          <span className="text-sm text-foreground/50">
            Kernel #{selectedFilter + 1}
          </span>
          <PatchGrid
            data={selectedKernel}
            colorFn={kernelColorFn}
            cellSize={56}
            showValues
          />
          <span className="font-mono text-xs text-foreground/30">
            bias: {selectedBias.toFixed(4)}
          </span>
        </div>

        {/* Arrow */}
        <div className="hidden items-center pt-20 text-foreground/20 sm:flex">
          <svg width="36" height="16" viewBox="0 0 36 16" fill="none">
            <path
              d="M0 8h28m0 0l-6-6m6 6l-6 6"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>

        {/* Feature map output */}
        <div className="flex flex-col items-center gap-2">
          <span className="text-sm text-foreground/50">
            Feature Map #{selectedFilter + 1}
          </span>
          {selectedMap ? (
            <ActivationHeatmap data={selectedMap} size={168} />
          ) : (
            <div
              className="flex items-center justify-center border border-border/50"
              style={{ width: 168, height: 168 }}
            >
              <span className="text-xs text-foreground/20">No data</span>
            </div>
          )}
          <span className="font-mono text-xs text-foreground/30">
            28 &times; 28
          </span>
        </div>
      </div>

      {/* Color legend */}
      <div className="flex items-center justify-center gap-4 text-xs text-foreground/30">
        <div className="flex items-center gap-1.5">
          <div
            className="h-3 w-3"
            style={{ backgroundColor: `rgb(${divergingRGB(-1, 1).join(",")})` }}
          />
          <span>Negative</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="h-3 w-3 bg-white" />
          <span>Zero</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div
            className="h-3 w-3"
            style={{ backgroundColor: `rgb(${divergingRGB(1, 1).join(",")})` }}
          />
          <span>Positive</span>
        </div>
      </div>

      {/* All 32 feature maps as clickable selector */}
      <div className="space-y-3">
        <p className="text-center text-xs text-foreground/40">
          Click a feature map to inspect its kernel
        </p>
        <div className="grid grid-cols-4 gap-2 sm:grid-cols-8">
          {conv1Maps
            ? conv1Maps.map((fm, i) => (
                <ActivationHeatmap
                  key={i}
                  data={fm}
                  size={56}
                  label={`#${i + 1}`}
                  onClick={() => setSelectedFilter(i)}
                  selected={i === selectedFilter}
                />
              ))
            : kernels.map((k, i) => (
                <PatchGrid
                  key={i}
                  data={k}
                  colorFn={kernelColorFn}
                  cellSize={12}
                  onClick={() => setSelectedFilter(i)}
                  highlight={i === selectedFilter}
                  label={`#${i + 1}`}
                />
              ))}
        </div>
      </div>
    </div>
  );
}
