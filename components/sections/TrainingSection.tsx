"use client";

import { useEffect, useState } from "react";
import { SectionWrapper } from "@/components/ui/SectionWrapper";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { LossCurve } from "@/components/visualizations/LossCurve";
import { WeightEvolution } from "@/components/visualizations/WeightEvolution";
import { GradientFlow } from "@/components/visualizations/GradientFlow";
import { EpochPredictionTimeline } from "@/components/visualizations/EpochPredictionTimeline";
import {
  loadTrainingHistory,
  loadWeightSnapshots,
  type TrainingHistory,
  type WeightSnapshots,
} from "@/lib/training/trainingData";

export function TrainingSection() {
  const [history, setHistory] = useState<TrainingHistory | null>(null);
  const [snapshots, setSnapshots] = useState<WeightSnapshots | null>(null);
  const [loadError, setLoadError] = useState(false);

  useEffect(() => {
    Promise.all([loadTrainingHistory(), loadWeightSnapshots()])
      .then(([h, s]) => {
        setHistory(h);
        setSnapshots(s);
      })
      .catch(() => setLoadError(true));
  }, []);

  return (
    <SectionWrapper id="training" fullHeight={false}>
      <SectionHeader
        step={8}
        title="How It Learned: Training"
        subtitle="The model wasn't born smart — it started with random weights and learned by seeing millions of examples. Over 50 epochs of training, it gradually improved its ability to recognize characters. Scrub through epochs to see how your drawing would be predicted at each stage."
      />

      <div className="flex flex-col gap-20">
        {/* Epoch prediction timeline - the star feature */}
        <div>
          <h3 className="mb-6 text-center text-lg font-medium text-foreground/70">
            Your drawing through training
          </h3>
          <EpochPredictionTimeline />
        </div>

        {/* Loss curve */}
        <div>
          <h3 className="mb-6 text-center text-lg font-medium text-foreground/70">
            Training progress
          </h3>
          {loadError ? (
            <div className="flex h-48 items-center justify-center rounded-xl border border-border bg-surface">
              <p className="text-foreground/30">
                Training history not available — run the training script first
              </p>
            </div>
          ) : (
            <LossCurve history={history} />
          )}
        </div>

        {/* Weight evolution */}
        <div>
          <h3 className="mb-6 text-center text-lg font-medium text-foreground/70">
            Weight evolution
          </h3>
          <WeightEvolution snapshots={snapshots} />
        </div>

        {/* Gradient flow */}
        <div>
          <h3 className="mb-6 text-center text-lg font-medium text-foreground/70">
            Gradient flow
          </h3>
          <GradientFlow />
        </div>
      </div>
    </SectionWrapper>
  );
}
