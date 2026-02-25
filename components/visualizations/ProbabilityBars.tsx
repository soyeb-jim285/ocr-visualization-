"use client";

import { useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useInferenceStore } from "@/stores/inferenceStore";
import { EMNIST_CLASSES } from "@/lib/model/classes";

interface ProbabilityBarsProps {
  compact?: boolean;
  maxBars?: number;
  prediction?: number[] | null;
}

export function ProbabilityBars({
  compact = false,
  maxBars = 10,
  prediction: externalPrediction,
}: ProbabilityBarsProps) {
  const storePrediction = useInferenceStore((s) => s.prediction);
  const prediction = externalPrediction ?? storePrediction;

  const sortedPredictions = useMemo(() => {
    if (!prediction) return [];
    return prediction
      .map((prob, idx) => ({
        classIndex: idx,
        label: EMNIST_CLASSES[idx],
        probability: prob,
      }))
      .sort((a, b) => b.probability - a.probability)
      .slice(0, maxBars);
  }, [prediction, maxBars]);

  if (!prediction || sortedPredictions.length === 0) {
    return (
      <div
        className={`flex items-center justify-center rounded-xl border border-border bg-surface ${
          compact ? "h-32 w-48" : "h-48"
        }`}
      >
        <p className="text-sm text-foreground/30">No prediction yet</p>
      </div>
    );
  }

  const topProb = sortedPredictions[0].probability;

  return (
    <div className={`flex flex-col gap-${compact ? "1.5" : "2"} ${compact ? "w-56" : "w-full"}`}>
      <AnimatePresence mode="popLayout">
        {sortedPredictions.map((item, i) => (
          <motion.div
            key={item.classIndex}
            layout
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.03, duration: 0.3 }}
            className="flex items-center gap-3"
          >
            {/* Class label */}
            <span
              className={`shrink-0 font-mono font-bold ${
                compact ? "w-6 text-sm" : "w-8 text-lg"
              } ${i === 0 ? "text-accent-primary" : "text-foreground/50"}`}
            >
              {item.label}
            </span>

            {/* Bar */}
            <div className="relative flex-1 overflow-hidden rounded-full bg-border/30">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${(item.probability / topProb) * 100}%` }}
                transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
                className={`rounded-full ${compact ? "h-4" : "h-6"} ${
                  i === 0
                    ? "bg-accent-primary"
                    : "bg-foreground/10"
                }`}
              />
            </div>

            {/* Percentage */}
            <span
              className={`shrink-0 font-mono ${
                compact ? "w-12 text-xs" : "w-16 text-sm"
              } ${i === 0 ? "text-accent-primary" : "text-foreground/40"}`}
            >
              {(item.probability * 100).toFixed(1)}%
            </span>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}
