"use client";

import { SectionWrapper } from "@/components/ui/SectionWrapper";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { ProbabilityBars } from "@/components/visualizations/ProbabilityBars";
import { useInferenceStore } from "@/stores/inferenceStore";
import { EMNIST_CLASSES } from "@/lib/model/classes";

export function SoftmaxSection() {
  const prediction = useInferenceStore((s) => s.prediction);
  const topPrediction = useInferenceStore((s) => s.topPrediction);

  return (
    <SectionWrapper id="softmax">
      <SectionHeader
        step={7}
        title="Confidence: Softmax"
        subtitle="The final layer produces 62 raw scores (logits), one for each character class. Softmax converts these into probabilities that sum to 1.0 — transforming 'how much does this look like each character?' into 'what's the probability it IS each character?'"
      />

      <div className="flex flex-col items-center gap-8">
        {/* Main probability bars */}
        <div className="w-full max-w-2xl">
          <ProbabilityBars maxBars={15} />
        </div>

        {/* Summary stats */}
        {prediction && topPrediction && (
          <div className="flex flex-wrap justify-center gap-4">
            <div className="flex flex-col items-center rounded-xl border border-accent-primary/30 bg-accent-primary/5 px-6 py-4">
              <span className="text-4xl font-bold text-accent-primary">
                {EMNIST_CLASSES[topPrediction.classIndex]}
              </span>
              <span className="text-xs text-foreground/40">
                Top prediction
              </span>
            </div>
            <div className="flex flex-col items-center rounded-xl border border-border bg-surface px-6 py-4">
              <span className="font-mono text-2xl font-bold text-accent-secondary">
                {(topPrediction.confidence * 100).toFixed(1)}%
              </span>
              <span className="text-xs text-foreground/40">Confidence</span>
            </div>
            <div className="flex flex-col items-center rounded-xl border border-border bg-surface px-6 py-4">
              <span className="font-mono text-2xl font-bold text-foreground/60">
                62
              </span>
              <span className="text-xs text-foreground/40">Classes</span>
            </div>
            <div className="flex flex-col items-center rounded-xl border border-border bg-surface px-6 py-4">
              <span className="font-mono text-2xl font-bold text-foreground/60">
                {prediction.reduce((sum, p) => sum + p, 0).toFixed(3)}
              </span>
              <span className="text-xs text-foreground/40">
                Sum (should be 1.0)
              </span>
            </div>
          </div>
        )}
      </div>
    </SectionWrapper>
  );
}
