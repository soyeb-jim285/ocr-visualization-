"use client";

import { useMemo, useState } from "react";
import { SectionWrapper } from "@/components/ui/SectionWrapper";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { ActivationHeatmap } from "@/components/visualizations/ActivationHeatmap";
import { useInferenceStore } from "@/stores/inferenceStore";
import { relu2D } from "@/lib/utils/mathUtils";

export function ActivationSection() {
  const layerActivations = useInferenceStore((s) => s.layerActivations);
  const [selectedFilter, setSelectedFilter] = useState(0);

  // Get pre-ReLU (conv1) and post-ReLU (relu1) activations
  const conv1Maps = layerActivations["conv1"] as number[][][] | undefined;
  const relu1Maps = layerActivations["relu1"] as number[][][] | undefined;

  const beforeRelu = conv1Maps?.[selectedFilter];
  const afterRelu = relu1Maps?.[selectedFilter];

  // Count negative values that were zeroed out
  const negativeCount = useMemo(() => {
    if (!beforeRelu) return 0;
    return beforeRelu.flat().filter((v) => v < 0).length;
  }, [beforeRelu]);

  const totalPixels = beforeRelu ? beforeRelu.length * beforeRelu[0].length : 0;
  const percentRemoved =
    totalPixels > 0 ? ((negativeCount / totalPixels) * 100).toFixed(1) : "0";

  const numFilters = conv1Maps?.length ?? 0;

  return (
    <SectionWrapper id="activation">
      <SectionHeader
        step={3}
        title="Amplifying Signals: ReLU Activation"
        subtitle="ReLU (Rectified Linear Unit) is deceptively simple: it keeps positive values unchanged and sets all negative values to zero. This non-linearity is what allows neural networks to learn complex, non-linear patterns."
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
            <span className="text-sm font-medium text-accent-negative">
              Before ReLU
            </span>
            {beforeRelu ? (
              <ActivationHeatmap data={beforeRelu} size={160} />
            ) : (
              <div className="flex h-[160px] w-[160px] items-center justify-center rounded-md border border-border bg-surface">
                <span className="text-xs text-foreground/20">No data</span>
              </div>
            )}
            <span className="text-xs text-foreground/40">
              Contains negative values
            </span>
          </div>

          {/* Arrow */}
          <div className="flex flex-col items-center gap-2">
            <span className="font-mono text-sm text-accent-warning">
              max(0, x)
            </span>
            <svg
              width="48"
              height="24"
              viewBox="0 0 48 24"
              fill="none"
              className="text-accent-warning"
            >
              <path
                d="M0 12h40m0 0l-8-8m8 8l-8 8"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
              />
            </svg>
          </div>

          <div className="flex flex-col items-center gap-3">
            <span className="text-sm font-medium text-accent-positive">
              After ReLU
            </span>
            {afterRelu ? (
              <ActivationHeatmap data={afterRelu} size={160} />
            ) : (
              <div className="flex h-[160px] w-[160px] items-center justify-center rounded-md border border-border bg-surface">
                <span className="text-xs text-foreground/20">No data</span>
              </div>
            )}
            <span className="text-xs text-foreground/40">
              Negatives zeroed out
            </span>
          </div>
        </div>

        {/* Stats */}
        {beforeRelu && (
          <div className="flex gap-4 rounded-xl border border-border bg-surface px-4 py-3 sm:gap-6 sm:px-6 sm:py-4">
            <div className="flex flex-col items-center">
              <span className="font-mono text-2xl font-bold text-accent-negative">
                {negativeCount}
              </span>
              <span className="text-xs text-foreground/40">
                values removed
              </span>
            </div>
            <div className="w-px bg-border" />
            <div className="flex flex-col items-center">
              <span className="font-mono text-2xl font-bold text-foreground/60">
                {percentRemoved}%
              </span>
              <span className="text-xs text-foreground/40">of the map</span>
            </div>
          </div>
        )}
      </div>
    </SectionWrapper>
  );
}
