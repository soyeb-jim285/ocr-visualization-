"use client";

import { SectionWrapper } from "@/components/ui/SectionWrapper";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { ConvolutionTheory } from "@/components/visualizations/ConvolutionTheory";

export function ConvolutionSection() {
  return (
    <SectionWrapper id="convolution" sig="conv1">
      <SectionHeader
        step={2}
        tag="Conv1 · 64 ch · 28×28"
        title="Finding Patterns: Convolution"
        subtitle="A small 3×3 filter slides across the entire image, computing a dot product at each position. Different filters detect different features: edges, curves, corners."
      />
      <ConvolutionTheory />
    </SectionWrapper>
  );
}
