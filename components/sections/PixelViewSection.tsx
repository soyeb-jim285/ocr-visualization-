"use client";

import { SectionWrapper } from "@/components/ui/SectionWrapper";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { PixelGrid } from "@/components/visualizations/PixelGrid";

export function PixelViewSection() {
  return (
    <SectionWrapper id="pixel-view">
      <SectionHeader
        step={1}
        title="What Does a Computer See?"
        subtitle="You see a character. The computer sees a 28x28 grid of numbers — each pixel is just a value between 0 (black) and 1 (white). Toggle between views to see the difference."
      />
      <PixelGrid />
    </SectionWrapper>
  );
}
