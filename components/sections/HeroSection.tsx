"use client";

import { motion } from "framer-motion";
import { DrawingCanvas } from "@/components/canvas/DrawingCanvas";
import { ImageUploader } from "@/components/canvas/ImageUploader";

export function HeroSection() {
  return (
    <div className="flex flex-col items-center px-4 pt-14 pb-2">
      <motion.h1
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="mb-1 text-center text-3xl font-bold tracking-tight sm:text-4xl"
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
        className="mb-3 text-center text-sm text-foreground/50"
      >
        Draw a character and watch the neural network recognize it
      </motion.p>
    </div>
  );
}

/** Compact drawing input for side-by-side layout */
export function HeroDrawingPanel() {
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: 0.5, duration: 0.8 }}
      className="flex shrink-0 flex-col items-center justify-center gap-2 px-4 py-4"
    >
      <DrawingCanvas />
      <ImageUploader />
    </motion.div>
  );
}
