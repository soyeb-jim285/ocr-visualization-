"use client";

import { motion } from "framer-motion";
import { useUIStore, type ViewMode } from "@/stores/uiStore";

export function ViewToggle() {
  const viewMode = useUIStore((s) => s.viewMode);
  const setViewMode = useUIStore((s) => s.setViewMode);

  return (
    <div className="fixed right-4 top-14 z-50 flex rounded-full border border-border bg-surface/80 p-1 backdrop-blur-md md:right-6">
      {(["2d", "3d"] as const).map((mode) => (
        <button
          key={mode}
          onClick={() => setViewMode(mode)}
          className={`relative rounded-full px-3 py-1.5 text-xs font-medium uppercase tracking-wider transition-colors ${
            viewMode === mode ? "text-background" : "text-foreground/50"
          }`}
        >
          {viewMode === mode && (
            <motion.div
              layoutId="viewTogglePill"
              className="absolute inset-0 rounded-full bg-accent-primary"
              transition={{ type: "spring", stiffness: 300, damping: 30 }}
            />
          )}
          <span className="relative z-10">{mode}</span>
        </button>
      ))}
    </div>
  );
}
