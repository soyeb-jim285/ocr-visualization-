"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { predictAtEpoch, prefetchAdjacentEpochs } from "@/lib/model/epochModels";
import { preprocessCanvas } from "@/lib/model/preprocess";
import { useInferenceStore } from "@/stores/inferenceStore";
import { EMNIST_CLASSES } from "@/lib/model/classes";
import { ProbabilityBars } from "./ProbabilityBars";

export function EpochPredictionTimeline() {
  const inputImageData = useInferenceStore((s) => s.inputImageData);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [epochPrediction, setEpochPrediction] = useState<number[] | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const tensorRef = useRef<Float32Array | null>(null);

  // Preprocess input once and cache the Float32Array
  useEffect(() => {
    tensorRef.current = null;
    if (inputImageData) {
      const { tensor } = preprocessCanvas(inputImageData);
      tensorRef.current = tensor;
    }
  }, [inputImageData]);

  // Run prediction at current epoch
  const predict = useCallback(async () => {
    if (!tensorRef.current) {
      setEpochPrediction(null);
      return;
    }

    setIsLoading(true);
    setError(null);
    try {
      const pred = await predictAtEpoch(tensorRef.current, currentEpoch);
      setEpochPrediction(pred);
      prefetchAdjacentEpochs(currentEpoch);
    } catch {
      setError("Model checkpoint not available");
      setEpochPrediction(null);
    } finally {
      setIsLoading(false);
    }
  }, [currentEpoch]);

  // Re-predict when epoch or input changes
  useEffect(() => {
    predict();
  }, [predict, inputImageData]);

  const topPrediction = epochPrediction
    ? (() => {
        const maxIdx = epochPrediction.indexOf(Math.max(...epochPrediction));
        return {
          label: EMNIST_CLASSES[maxIdx],
          confidence: epochPrediction[maxIdx],
        };
      })()
    : null;

  const hasInput = inputImageData !== null;

  return (
    <div className="flex flex-col items-center gap-8">
      {/* Epoch slider */}
      <div className="flex w-full max-w-xl flex-col items-center gap-3">
        <div className="flex w-full items-center justify-between">
          <span className="text-sm text-foreground/40">Epoch 0</span>
          <span className="font-mono text-lg font-bold text-accent-primary">
            Epoch {currentEpoch}
          </span>
          <span className="text-sm text-foreground/40">Epoch 49</span>
        </div>

        <input
          type="range"
          min={0}
          max={49}
          value={currentEpoch}
          onChange={(e) => setCurrentEpoch(parseInt(e.target.value))}
          className="w-full accent-accent-primary"
          disabled={!hasInput}
        />

        {/* Epoch markers */}
        <div className="flex w-full justify-between px-1">
          {[0, 10, 20, 30, 40, 49].map((e) => (
            <button
              key={e}
              onClick={() => setCurrentEpoch(e)}
              className={`rounded px-2 py-0.5 font-mono text-xs transition-colors ${
                currentEpoch === e
                  ? "bg-accent-primary/20 text-accent-primary"
                  : "text-foreground/30 hover:text-foreground/50"
              }`}
            >
              {e}
            </button>
          ))}
        </div>
      </div>

      {/* Prediction display */}
      <div className="flex flex-col items-center gap-6 md:flex-row md:items-start md:gap-12">
        {/* Big prediction */}
        <div className="flex flex-col items-center gap-2">
          <span className="text-sm text-foreground/40">Model predicts</span>
          <motion.div
            key={topPrediction?.label ?? "none"}
            initial={{ scale: 0.5, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="flex h-28 w-28 items-center justify-center rounded-2xl border-2 border-accent-primary/30 bg-surface"
          >
            {isLoading ? (
              <div className="h-6 w-6 animate-spin rounded-full border-2 border-accent-primary border-t-transparent" />
            ) : topPrediction ? (
              <span className="text-5xl font-bold text-accent-primary">
                {topPrediction.label}
              </span>
            ) : (
              <span className="text-2xl text-foreground/20">?</span>
            )}
          </motion.div>
          {topPrediction && (
            <span className="font-mono text-sm text-foreground/50">
              {(topPrediction.confidence * 100).toFixed(1)}% confident
            </span>
          )}
        </div>

        {/* Probability bars for this epoch */}
        <div className="w-72">
          <ProbabilityBars
            prediction={epochPrediction}
            compact
            maxBars={5}
          />
        </div>
      </div>

      {error && (
        <p className="text-sm text-accent-negative/60">{error}</p>
      )}

      {!hasInput && (
        <p className="text-sm text-foreground/30">
          Draw a character above to see how the model improves across training epochs
        </p>
      )}
    </div>
  );
}
