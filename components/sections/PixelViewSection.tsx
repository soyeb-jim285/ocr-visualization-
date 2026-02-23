"use client";

import { SectionWrapper } from "@/components/ui/SectionWrapper";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { PixelGrid } from "@/components/visualizations/PixelGrid";
import { Latex } from "@/components/ui/Latex";

export function PixelViewSection() {
  return (
    <SectionWrapper id="pixel-view">
      <SectionHeader
        step={1}
        title="What Does a Computer See?"
        subtitle="Your drawing is captured on a 280×280 canvas, then downsampled to a 28×28 grid of numbers — each pixel becomes a value between 0 and 1. Hover to see how each output pixel is computed."
      />
      <div className="flex flex-col items-center gap-8 lg:flex-row lg:items-start lg:gap-12">
        {/* Left: theory text */}
        <div className="flex-1 space-y-4 text-center lg:text-left">
          <p className="text-base leading-relaxed text-foreground/65 sm:text-lg">
            The canvas captures your strokes at 280&times;280 pixels — white
            ink on a black background. Before the neural network can process it,
            we downsample to 28&times;28 using{" "}
            <em>area-average filtering</em>. Each output pixel is a weighted
            average of the ~10&times;10 source region that maps to it,
            preserving stroke edges better than naive nearest-neighbor sampling.
          </p>

          {/* Main equation */}
          <div className="py-3">
            <Latex
              display
              math="O(i,j) = \frac{1}{A} \sum_{s \in R_i} \sum_{t \in R_j} w(s,t) \cdot I(s,t)"
            />
          </div>

          {/* Equation legend */}
          <div className="flex flex-wrap justify-center gap-x-6 gap-y-1 text-sm text-foreground/40 lg:justify-start">
            <span><Latex math="I" /> — source image (280&times;280)</span>
            <span><Latex math="O" /> — output pixel (28&times;28)</span>
            <span><Latex math="R_i, R_j" /> — source region</span>
            <span><Latex math="w" /> — fractional overlap weight</span>
            <span><Latex math="A" /> — total overlap area</span>
          </div>

          <p className="text-sm leading-relaxed text-foreground/45">
            The grayscale normalization is straightforward:{" "}
            <Latex math="I(x,y) = R(x,y) / 255" />, where{" "}
            <Latex math="R" /> is the red channel (identical to green and blue
            on a grayscale canvas). After resizing, we transpose for the EMNIST
            convention — the training data was stored column-major. The final
            shape transform:{" "}
            <Latex math="(280, 280) \xrightarrow{\text{resize}} (28, 28) \xrightarrow{\text{reshape}} (1, 1, 28, 28)" />.
            That&apos;s 784 numbers in <Latex math="[0, 1]" /> — batch,
            channel, height, width.
          </p>
        </div>

        {/* Right: PixelGrid visualization */}
        <div className="flex w-full shrink-0 flex-col items-center gap-4 lg:w-auto">
          <PixelGrid />
        </div>
      </div>
    </SectionWrapper>
  );
}
