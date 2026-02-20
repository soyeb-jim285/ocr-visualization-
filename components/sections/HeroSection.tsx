"use client";

import { motion, AnimatePresence } from "framer-motion";
import { DrawingCanvas } from "@/components/canvas/DrawingCanvas";
import { ImageUploader } from "@/components/canvas/ImageUploader";
import { ProbabilityBars } from "@/components/visualizations/ProbabilityBars";
import { useInferenceStore } from "@/stores/inferenceStore";
import { useUIStore } from "@/stores/uiStore";
import { EMNIST_CLASSES } from "@/lib/model/classes";

export function HeroSection() {
  const topPrediction = useInferenceStore((s) => s.topPrediction);
  const isInferring = useInferenceStore((s) => s.isInferring);
  const modelLoaded = useUIStore((s) => s.modelLoaded);

  return (
    <section
      id="hero"
      className="relative flex min-h-screen flex-col items-center justify-center px-4 py-20"
    >
      {/* Title */}
      <motion.h1
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="mb-3 text-center text-5xl font-bold tracking-tight sm:text-6xl md:text-7xl"
        style={{
          backgroundImage:
            "linear-gradient(135deg, var(--accent-primary), var(--accent-tertiary))",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
        }}
      >
        Neural Network X-Ray
      </motion.h1>

      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3, duration: 0.8 }}
        className="mb-12 max-w-lg text-center text-lg text-foreground/50"
      >
        Draw a character and watch a neural network recognize it — layer by
        layer, neuron by neuron.
      </motion.p>

      {/* Main content */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5, duration: 0.8 }}
        className="flex flex-col items-center gap-8 md:flex-row md:items-start md:gap-16"
      >
        {/* Drawing area */}
        <div className="flex flex-col items-center gap-4">
          <DrawingCanvas />
          <ImageUploader />
        </div>

        {/* Prediction result */}
        <div className="flex min-w-[240px] flex-col items-center gap-6">
          <AnimatePresence mode="wait">
            {topPrediction ? (
              <motion.div
                key={topPrediction.classIndex}
                initial={{ scale: 0.5, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.5, opacity: 0 }}
                className="flex flex-col items-center gap-2"
              >
                <span className="text-sm text-foreground/40">
                  Predicted character
                </span>
                <div className="flex h-32 w-32 items-center justify-center rounded-2xl border-2 border-accent-primary bg-accent-primary/5">
                  <span className="text-7xl font-bold text-accent-primary">
                    {EMNIST_CLASSES[topPrediction.classIndex]}
                  </span>
                </div>
                <span className="font-mono text-sm text-foreground/50">
                  {(topPrediction.confidence * 100).toFixed(1)}% confidence
                </span>
              </motion.div>
            ) : (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex h-32 w-32 items-center justify-center rounded-2xl border-2 border-dashed border-border"
              >
                <span className="text-foreground/20">?</span>
              </motion.div>
            )}
          </AnimatePresence>

          {isInferring && (
            <div className="h-5 w-5 animate-spin rounded-full border-2 border-accent-primary border-t-transparent" />
          )}

          <ProbabilityBars compact maxBars={5} />
        </div>
      </motion.div>

      {/* Scroll CTA */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.5 }}
        className="absolute bottom-8 flex flex-col items-center gap-2"
      >
        <span className="text-sm text-foreground/30">
          Scroll to see how the network thinks
        </span>
        <motion.div
          animate={{ y: [0, 6, 0] }}
          transition={{ repeat: Infinity, duration: 2 }}
        >
          <svg
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            className="text-foreground/30"
          >
            <path
              d="M12 5v14m0 0l-6-6m6 6l6-6"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </motion.div>
      </motion.div>
    </section>
  );
}
