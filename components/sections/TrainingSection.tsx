"use client";

import { useEffect, useState } from "react";
import { SectionWrapper } from "@/components/ui/SectionWrapper";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { LossCurve } from "@/components/visualizations/LossCurve";
import { WeightEvolution } from "@/components/visualizations/WeightEvolution";
import { GradientFlow } from "@/components/visualizations/GradientFlow";
import { EpochNetworkVisualization } from "@/components/visualizations/EpochNetworkVisualization";
import { Latex } from "@/components/ui/Latex";
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
        step={9}
        title="How It Learned: Training"
        subtitle="The model wasn't born smart — it started with random weights and learned by seeing millions of examples. Over 75 epochs of training, it gradually improved its ability to recognize characters. Scrub through epochs to see how your drawing would be predicted at each stage."
      />

      {/* Theory introduction */}
      <div className="mb-10 space-y-4 text-center lg:text-left">
        <p className="text-base leading-relaxed text-foreground/65 sm:text-lg">
          Training is an iterative optimization: the model sees a batch of
          labeled examples, computes how wrong its predictions are (the{" "}
          <em>loss</em>), then adjusts every weight slightly to reduce that
          error. This cycle repeats millions of times. The loss function used
          here is <em>cross-entropy</em>, which heavily penalizes confident
          wrong answers.
        </p>

        <div className="py-3">
          <Latex
            display
            math="\mathcal{L} = -\sum_{i=1}^{K} y_i \log\!\left(\hat{y}_i\right) \qquad\qquad \theta \leftarrow \theta - \eta\,\nabla_\theta \mathcal{L}"
          />
        </div>

        <div className="flex flex-wrap justify-center gap-x-6 gap-y-1 text-sm text-foreground/40 lg:justify-start">
          <span><Latex math="y_i" /> — true label (one-hot)</span>
          <span><Latex math="\hat{y}_i" /> — predicted probability</span>
          <span><Latex math="\theta" /> — all model weights</span>
          <span><Latex math="\eta" /> — learning rate</span>
          <span><Latex math="\nabla_\theta \mathcal{L}" /> — gradient</span>
        </div>

        <p className="text-sm leading-relaxed text-foreground/45">
          The gradient <Latex math="\nabla_\theta \mathcal{L}" /> tells each
          weight how to change to reduce the loss. Backpropagation computes this
          efficiently using the chain rule, flowing error signals backward
          through every layer. With Adam optimizer (
          <Latex math="\eta = 10^{-3}" />
          ), the model converges over 75 epochs on EMNIST ByMerge + BanglaLekha-Isolated —
          roughly 1M training images of handwritten characters in English and Bengali.
        </p>
      </div>

      <div className="flex flex-col gap-12 sm:gap-20">
        {/* Epoch network visualization - the star feature */}
        <div>
          <h3 className="mb-2 text-center text-lg font-medium text-foreground/70">
            Your drawing through training
          </h3>
          <EpochNetworkVisualization />
        </div>

        {/* Loss curve */}
        <div>
          <h3 className="mb-6 text-center text-lg font-medium text-foreground/70">
            Training progress
          </h3>
          {loadError ? (
            <div className="viz-empty-state h-48">
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
