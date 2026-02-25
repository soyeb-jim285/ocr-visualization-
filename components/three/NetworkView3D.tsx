"use client";

import dynamic from "next/dynamic";
import { useUIStore } from "@/stores/uiStore";
import { motion, AnimatePresence } from "framer-motion";

// Dynamic import: Three.js only loads when user toggles to 3D mode
const NetworkScene = dynamic(
  () =>
    import("@/components/three/NetworkScene").then((m) => ({
      default: m.NetworkScene,
    })),
  {
    ssr: false,
    loading: () => (
      <div className="flex h-[70vh] items-center justify-center rounded-2xl border border-border bg-[#08080c]">
        <div className="flex flex-col items-center gap-3">
          <div className="h-6 w-6 animate-spin rounded-full border-2 border-accent-tertiary border-t-transparent" />
          <span className="text-sm text-foreground/30">
            Loading 3D scene...
          </span>
        </div>
      </div>
    ),
  }
);

export function NetworkView3D() {
  const viewMode = useUIStore((s) => s.viewMode);

  return (
    <AnimatePresence>
      {viewMode === "3d" && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          transition={{ duration: 0.3 }}
          className="fixed inset-0 z-40 bg-background/95 backdrop-blur-sm"
        >
          <div className="flex h-full flex-col items-center justify-center p-8">
            <h2
              className="mb-4 text-2xl font-semibold text-foreground"
            >
              3D Network Architecture
            </h2>
            <p className="mb-6 text-sm text-foreground/40">
              Orbit with mouse drag · Zoom with scroll · Draw a character to see activations
            </p>
            <div className="w-full max-w-5xl flex-1">
              <NetworkScene />
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
