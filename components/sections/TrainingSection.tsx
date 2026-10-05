"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { SectionWrapper } from "@/components/ui/SectionWrapper";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { LossCurve } from "@/components/visualizations/LossCurve";
import { WeightEvolution } from "@/components/visualizations/WeightEvolution";
import { GradientFlow } from "@/components/visualizations/GradientFlow";
import { EpochNetworkVisualization } from "@/components/visualizations/EpochNetworkVisualization";
import { Latex } from "@/components/ui/Latex";
import { TOTAL_EPOCHS } from "@/lib/model/epochModels";
import {
  loadTrainingHistory,
  loadWeightSnapshots,
  type TrainingHistory,
  type WeightSnapshots,
} from "@/lib/training/trainingData";

const reveal = {
  initial: { opacity: 0, y: 16 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, margin: "-10% 0px" },
  transition: { duration: 0.6, ease: [0.16, 1, 0.3, 1] as const },
};

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
    <SectionWrapper id="training" fullHeight={false} sig="train">
      <SectionHeader
        step={9}
        wide
        tag="40 epochs · AdamW"
        title="How It Learned: Training"
        subtitle="The model wasn't born smart — it started with random weights and, over many passes through the data, learned by seeing millions of examples. Over 40 epochs of training, it gradually improved its ability to recognize characters. Scrub through epochs to see how your drawing would be predicted at each stage."
      />

      {/* Theory introduction */}
      <motion.div
        {...reveal}
        className="mb-12 grid sm:mb-16 gap-x-6 gap-y-8 md:grid-cols-12"
      >
        <p className="prose-body min-w-0 md:col-span-6">
          Training is an iterative optimization: the model sees a batch of
          labeled examples, computes how wrong its predictions are (the{" "}
          <em>loss</em>), then adjusts every weight slightly to reduce that
          error. This cycle repeats millions of times. The loss function used
          here is <em>cross-entropy</em>, which heavily penalizes confident
          wrong answers.
        </p>
        <p className="prose-body min-w-0 md:col-span-6">
          The gradient <Latex math="\nabla_\theta \mathcal{L}" /> tells each
          weight how to change to reduce the loss. Backpropagation computes this
          efficiently using the chain rule, flowing error signals backward
          through every layer. With the AdamW optimizer on a one-cycle learning-rate schedule (the rate
          ramps up, then anneals toward zero) and an exponential moving average
          of the weights, the model converges over 40 epochs on EMNIST ByMerge + BanglaLekha-Isolated —
          hundreds of thousands of training images of handwritten characters in English and Bengali.
        </p>

        <div className="formula min-w-0 flex-wrap gap-x-10 text-[0.85em] sm:text-[1em] md:col-span-12">
          <Latex
            display
            math="\mathcal{L} = -\sum_{i=1}^{K} y_i \log\!\left(\hat{y}_i\right)"
          />
          <Latex display math="\theta \leftarrow \theta - \eta\,\nabla_\theta \mathcal{L}" />
          <span className="eq-no">(9)</span>
        </div>

        <dl className="grid grid-cols-1 gap-x-6 min-[420px]:grid-cols-2 gap-y-2 font-mono text-[11px] text-ink-3 md:col-span-12 md:grid-cols-5">
          <div><Latex math="y_i" /> true label (one-hot)</div>
          <div><Latex math="\hat{y}_i" /> predicted probability</div>
          <div><Latex math="\theta" /> all model weights</div>
          <div><Latex math="\eta" /> learning rate</div>
          <div><Latex math="\nabla_\theta \mathcal{L}" /> gradient</div>
        </dl>
      </motion.div>

      <div className="flex flex-col gap-12 sm:gap-16">
        {/* Epoch network visualization - the star feature */}
        <motion.figure {...reveal} className="figure m-0">
          <EpochNetworkVisualization />
          <figcaption className="figcap">
            <b>FIG. 9.1</b> Your drawing through training. Same pixels, {TOTAL_EPOCHS}{" "}
            checkpoints: drag the scrubber to watch the prediction sharpen.
          </figcaption>
        </motion.figure>

        {/* Loss curve */}
        <motion.figure {...reveal} className="figure m-0">
          {loadError ? (
            <div className="viz-empty-state min-h-[320px]">
              <p>Training history not available — run the training script first</p>
            </div>
          ) : (
            <LossCurve history={history} />
          )}
          <figcaption className="figcap">
            <b>FIG. 9.2</b> Training progress. Solid is train, dashed is
            validation; hover or touch for per-epoch values. Train metrics come from a clean evaluation pass (no augmentation, no dropout), so train and validation are directly comparable.
          </figcaption>
        </motion.figure>

        {/* Weight evolution */}
        <motion.figure {...reveal} className="figure m-0">
          <WeightEvolution snapshots={snapshots} />
          <figcaption className="figcap">
            <b>FIG. 9.3</b> Weight evolution. Distribution of every weight in
            the chosen layer at each snapshot epoch.
          </figcaption>
        </motion.figure>

        {/* Gradient flow */}
        <motion.figure {...reveal} className="figure m-0">
          <GradientFlow />
          <figcaption className="figcap">
            <b>FIG. 9.4</b> Gradient flow. RMS weight change per layer between
            two snapshots, a proxy for the size of the gradient updates.
          </figcaption>
        </motion.figure>
      </div>
    </SectionWrapper>
  );
}
