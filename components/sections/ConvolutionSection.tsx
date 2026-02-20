"use client";

import { SectionWrapper } from "@/components/ui/SectionWrapper";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { KernelAnimation } from "@/components/visualizations/KernelAnimation";
import { FeatureMapGrid } from "@/components/visualizations/FeatureMapGrid";
import { useInferenceStore } from "@/stores/inferenceStore";

export function ConvolutionSection() {
  const layerActivations = useInferenceStore((s) => s.layerActivations);
  const conv1Maps = layerActivations["conv1"] as number[][][] | undefined;

  return (
    <SectionWrapper id="convolution">
      <SectionHeader
        step={2}
        title="Finding Patterns: Convolution"
        subtitle="A small 3x3 filter slides across the entire image, computing a dot product at each position. Different filters detect different features — edges, curves, corners. The first layer uses 32 filters, producing 32 feature maps."
      />

      <div className="flex flex-col gap-16">
        {/* Kernel animation */}
        <div>
          <h3 className="mb-6 text-center text-lg font-medium text-foreground/70">
            Watch the kernel slide across the input
          </h3>
          <KernelAnimation />
        </div>

        {/* All feature maps */}
        <div>
          <h3 className="mb-6 text-center text-lg font-medium text-foreground/70">
            All 32 feature maps from Convolution Layer 1
          </h3>
          {conv1Maps ? (
            <FeatureMapGrid
              featureMaps={conv1Maps}
              layerName="conv1"
              columns={8}
              cellSize={64}
            />
          ) : (
            <div className="flex h-40 items-center justify-center rounded-xl border border-border bg-surface">
              <p className="text-foreground/30">
                Draw a character to see feature maps
              </p>
            </div>
          )}
        </div>
      </div>
    </SectionWrapper>
  );
}
