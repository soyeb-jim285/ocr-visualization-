"use client";

import { motion } from "framer-motion";
import { SectionWrapper } from "@/components/ui/SectionWrapper";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { PixelGrid } from "@/components/visualizations/PixelGrid";
import { Latex } from "@/components/ui/Latex";

const reveal = (delay = 0) => ({
  initial: { opacity: 0, y: 12 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, margin: "-12% 0px" },
  transition: { duration: 0.5, delay, ease: [0.16, 1, 0.3, 1] as const },
});

const LEGEND: [string, string][] = [
  ["I", "source image, 280×280"],
  ["O", "output pixel, 28×28"],
  ["R_i, R_j", "source region"],
  ["w", "fractional overlap weight"],
  ["A", "total overlap area"],
];

export function PixelViewSection() {
  return (
    <SectionWrapper id="pixel-view" sig="input">
      <SectionHeader
        step={1}
        tag="Input · 1 ch · 28×28"
        title="What Does a Computer See?"
        subtitle="Your drawing is captured on a 280×280 canvas, then downsampled to a 28×28 grid of numbers. Each pixel becomes a value between 0 and 1. Hover to see how each output pixel is computed."
      />

      <div className="grid grid-cols-[minmax(0,1fr)] gap-12 lg:grid-cols-12 lg:gap-x-6">
        <motion.div {...reveal()} className="min-w-0 space-y-6 lg:col-span-5">
          <p className="prose-body">
            The canvas captures your strokes at 280&times;280 pixels, white ink
            on a black background. Before the network can read it, we
            downsample to 28&times;28 using <em>area-average filtering</em>.
            Each output pixel is a weighted average of the ~10&times;10 source
            region that maps to it, preserving stroke edges better than
            nearest-neighbor sampling.
          </p>

          <div className="formula">
            <Latex
              display
              math="O(i,j) = \frac{1}{A} \sum_{s \in R_i} \sum_{t \in R_j} w(s,t) \cdot I(s,t)"
            />
            <span className="eq-no">(1)</span>
          </div>

          <dl className="grid grid-cols-[auto_1fr] items-baseline gap-x-4 gap-y-1.5 text-sm text-ink-2">
            {LEGEND.map(([sym, desc]) => (
              <div key={sym} className="contents">
                <dt className="text-ink">
                  <Latex math={sym} />
                </dt>
                <dd className="font-mono text-[11px] text-ink-3">{desc}</dd>
              </div>
            ))}
          </dl>

          <p className="callout [overflow-wrap:anywhere]">
            <span className="tag">NOTE</span>
            Grayscale is <Latex math="I(x,y) = R(x,y)/255" />. We then transpose
            for the EMNIST convention, because the training data was stored
            column-major. Final shape:{" "}
            <Latex math="(1, 1, 28, 28)" />: 784 numbers in <Latex math="[0, 1]" />.
          </p>
        </motion.div>

        <motion.div {...reveal(0.16)} className="min-w-0 self-start lg:sticky lg:top-24 lg:col-span-7">
          <PixelGrid />
        </motion.div>
      </div>
    </SectionWrapper>
  );
}
