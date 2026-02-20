"use client";

import { SectionWrapper } from "@/components/ui/SectionWrapper";
import { PixelGrid } from "@/components/visualizations/PixelGrid";

export function PixelViewSection() {
  return (
    <SectionWrapper id="pixel-view" fullHeight={false}>
      <div className="flex flex-col items-center gap-8 lg:flex-row lg:items-center lg:gap-16">
        <div className="flex-1 text-center lg:text-left">
          <span className="mb-3 inline-block rounded-full border border-accent-primary/30 bg-accent-primary/10 px-3 py-1 font-mono text-xs text-accent-primary">
            Step 1
          </span>
          <h2 className="mb-4 bg-gradient-to-br from-foreground to-foreground/60 bg-clip-text text-3xl font-bold tracking-tight text-transparent sm:text-4xl md:text-5xl">
            What Does a Computer See?
          </h2>
          <p className="mb-4 text-base text-foreground/50 sm:text-lg">
            You see a character. The computer sees a 28&times;28 grid of numbers
            &mdash; each pixel is just a value between 0 (black) and 1 (white).
            Toggle between views to see the difference.
          </p>
          <p className="text-sm leading-relaxed text-foreground/35">
            Your high-resolution canvas drawing is downsampled to a tiny
            28&times;28 pixel image &mdash; the same format the neural network
            was trained on (EMNIST). Every detail gets compressed into just 784
            numbers before the network ever sees it.
          </p>
        </div>
        <div className="w-full max-w-md shrink-0 lg:w-auto">
          <PixelGrid />
        </div>
      </div>
    </SectionWrapper>
  );
}
