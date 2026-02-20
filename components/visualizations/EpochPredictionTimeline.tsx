"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import {
  predictAtEpoch,
  prefetchAdjacentEpochs,
  prefetchAllEpochs,
  TOTAL_EPOCHS,
  getCachedModelCount,
} from "@/lib/model/epochModels";
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
  const [loadedCount, setLoadedCount] = useState(() => getCachedModelCount());
  const tensorRef = useRef<Float32Array | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingEpochRef = useRef<number>(0);

  // Prefetch all models in background on mount — throttle progress updates
  useEffect(() => {
    let lastUpdate = 0;
    let rafId = 0;
    let latestLoaded = getCachedModelCount();

    prefetchAllEpochs((loaded, _total) => {
      latestLoaded = loaded;
      const now = Date.now();
      // Only update state at most every 500ms to avoid 50 re-renders
      if (now - lastUpdate > 500 || loaded >= _total) {
        lastUpdate = now;
        cancelAnimationFrame(rafId);
        rafId = requestAnimationFrame(() => setLoadedCount(latestLoaded));
      }
    });

    return () => cancelAnimationFrame(rafId);
  }, []);

  // Preprocess input once and cache the Float32Array
  useEffect(() => {
    tensorRef.current = null;
    if (inputImageData) {
      const { tensor } = preprocessCanvas(inputImageData);
      tensorRef.current = tensor;
    }
  }, [inputImageData]);

  // Run prediction at a given epoch
  const predictEpoch = useCallback(async (epoch: number) => {
    if (!tensorRef.current) {
      setEpochPrediction(null);
      return;
    }

    setIsLoading(true);
    setError(null);
    try {
      const pred = await predictAtEpoch(tensorRef.current, epoch);
      // Only update if this is still the latest requested epoch
      if (pendingEpochRef.current === epoch) {
        setEpochPrediction(pred);
      }
      prefetchAdjacentEpochs(epoch);
    } catch {
      if (pendingEpochRef.current === epoch) {
        setError("Model checkpoint not available");
        setEpochPrediction(null);
      }
    } finally {
      if (pendingEpochRef.current === epoch) {
        setIsLoading(false);
      }
    }
  }, []);

  // Debounced epoch change — waits 150ms after slider stops moving
  const handleEpochChange = useCallback(
    (epoch: number) => {
      setCurrentEpoch(epoch);
      pendingEpochRef.current = epoch;

      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        predictEpoch(epoch);
      }, 150);
    },
    [predictEpoch],
  );

  // Re-predict when input changes
  useEffect(() => {
    pendingEpochRef.current = currentEpoch;
    predictEpoch(currentEpoch);
  }, [inputImageData, currentEpoch, predictEpoch]);

  // Cleanup debounce on unmount
  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

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
  const allLoaded = loadedCount >= TOTAL_EPOCHS;

  return (
    <div className="flex flex-col items-center gap-8">
      {/* Download progress */}
      {!allLoaded && (
        <div className="flex w-full max-w-xl flex-col items-center gap-1">
          <div className="flex w-full items-center justify-between text-xs text-foreground/30">
            <span>Loading epoch models...</span>
            <span>{loadedCount}/{TOTAL_EPOCHS}</span>
          </div>
          <div className="h-1 w-full overflow-hidden rounded-full bg-surface-elevated">
            <div
              className="h-full rounded-full bg-accent-primary/50 transition-all duration-300"
              style={{ width: `${(loadedCount / TOTAL_EPOCHS) * 100}%` }}
            />
          </div>
        </div>
      )}

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
          onChange={(e) => handleEpochChange(parseInt(e.target.value))}
          className="w-full accent-accent-primary"
          disabled={!hasInput}
        />

        {/* Epoch markers */}
        <div className="flex w-full justify-between px-1">
          {[0, 10, 20, 30, 40, 49].map((e) => (
            <button
              key={e}
              onClick={() => handleEpochChange(e)}
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
