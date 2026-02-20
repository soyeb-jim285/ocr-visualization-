"use client";

import { useMemo, useState } from "react";
import { SectionWrapper } from "@/components/ui/SectionWrapper";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { ActivationHeatmap } from "@/components/visualizations/ActivationHeatmap";
import { useInferenceStore } from "@/stores/inferenceStore";

export function PoolingSection() {
  const layerActivations = useInferenceStore((s) => s.layerActivations);
  const [selectedFilter, setSelectedFilter] = useState(0);

  // Before pooling (relu2 = 28x28x64) and after pooling (pool1 = 14x14x64)
  const relu2Maps = layerActivations["relu2"] as number[][][] | undefined;
  const pool1Maps = layerActivations["pool1"] as number[][][] | undefined;

  const beforePool = relu2Maps?.[selectedFilter];
  const afterPool = pool1Maps?.[selectedFilter];

  const numFilters = relu2Maps?.length ?? 0;

  return (
    <SectionWrapper id="pooling">
      <SectionHeader
        step={4}
        title="Compressing Information: Max Pooling"
        subtitle="Max pooling slides a 2x2 window across each feature map and keeps only the maximum value. This halves the spatial dimensions (28x28 → 14x14) while retaining the strongest activations. It makes the model more efficient and somewhat invariant to small shifts in position."
      />

      <div className="flex flex-col items-center gap-8">
        {/* Filter selector */}
        {numFilters > 0 && (
          <div className="flex items-center gap-3">
            <span className="text-sm text-foreground/40">Filter:</span>
            <input
              type="range"
              min={0}
              max={numFilters - 1}
              value={selectedFilter}
              onChange={(e) => setSelectedFilter(parseInt(e.target.value))}
              className="w-32 sm:w-48 accent-accent-primary"
            />
            <span className="font-mono text-sm text-accent-primary">
              #{selectedFilter + 1}
            </span>
          </div>
        )}

        {/* Before/After comparison */}
        <div className="flex flex-col items-center gap-8 md:flex-row md:gap-16">
          <div className="flex flex-col items-center gap-3">
            <span className="text-sm font-medium text-foreground/60">
              Before Pooling
            </span>
            {beforePool ? (
              <div className="flex flex-col items-center gap-1">
                <ActivationHeatmap data={beforePool} size={160} />
                <span className="font-mono text-xs text-foreground/30">
                  {beforePool.length}x{beforePool[0]?.length ?? 0}
                </span>
              </div>
            ) : (
              <div className="flex h-[160px] w-[160px] items-center justify-center rounded-md border border-border bg-surface">
                <span className="text-xs text-foreground/20">No data</span>
              </div>
            )}
          </div>

          {/* Arrow with pool icon */}
          <div className="flex flex-col items-center gap-2">
            <div className="grid grid-cols-2 gap-0.5 rounded border border-accent-tertiary/30 bg-surface p-2">
              {[0.3, 0.7, 0.1, 0.9].map((v, i) => (
                <div
                  key={i}
                  className={`flex h-7 w-7 items-center justify-center rounded-sm text-xs font-mono ${
                    i === 3
                      ? "bg-accent-tertiary text-background font-bold"
                      : "bg-border/30 text-foreground/30"
                  }`}
                >
                  {v}
                </div>
              ))}
            </div>
            <span className="text-xs text-accent-tertiary">
              max(2x2) → keep strongest
            </span>
          </div>

          <div className="flex flex-col items-center gap-3">
            <span className="text-sm font-medium text-foreground/60">
              After Pooling
            </span>
            {afterPool ? (
              <div className="flex flex-col items-center gap-1">
                <ActivationHeatmap data={afterPool} size={160} />
                <span className="font-mono text-xs text-foreground/30">
                  {afterPool.length}x{afterPool[0]?.length ?? 0}
                </span>
              </div>
            ) : (
              <div className="flex h-[160px] w-[160px] items-center justify-center rounded-md border border-border bg-surface">
                <span className="text-xs text-foreground/20">No data</span>
              </div>
            )}
          </div>
        </div>

        {/* Size comparison */}
        {beforePool && afterPool && (
          <div className="flex flex-wrap justify-center gap-4 rounded-xl border border-border bg-surface px-4 py-3 sm:gap-6 sm:px-6 sm:py-4">
            <div className="flex flex-col items-center">
              <span className="font-mono text-2xl font-bold text-foreground/60">
                {beforePool.length * (beforePool[0]?.length ?? 0)}
              </span>
              <span className="text-xs text-foreground/40">
                values before
              </span>
            </div>
            <div className="flex flex-col items-center justify-center">
              <span className="text-2xl text-accent-tertiary">→</span>
            </div>
            <div className="flex flex-col items-center">
              <span className="font-mono text-2xl font-bold text-accent-tertiary">
                {afterPool.length * (afterPool[0]?.length ?? 0)}
              </span>
              <span className="text-xs text-foreground/40">values after</span>
            </div>
            <div className="w-px bg-border" />
            <div className="flex flex-col items-center">
              <span className="font-mono text-2xl font-bold text-accent-positive">
                75%
              </span>
              <span className="text-xs text-foreground/40">reduction</span>
            </div>
          </div>
        )}
      </div>
    </SectionWrapper>
  );
}
