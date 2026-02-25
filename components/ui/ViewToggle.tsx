"use client";

import { motion } from "framer-motion";
import { useUIStore } from "@/stores/uiStore";

export function ViewToggle() {
  const viewMode = useUIStore((s) => s.viewMode);
  const setViewMode = useUIStore((s) => s.setViewMode);

  return (
    <div className="fixed bottom-4 left-1/2 z-50 flex -translate-x-1/2 rounded-full border border-border/60 bg-background/80 p-1 shadow-lg shadow-black/30 backdrop-blur-xl md:bottom-auto md:left-auto md:right-6 md:top-14 md:translate-x-0">
      {(["2d", "3d"] as const).map((mode) => (
        <button
          key={mode}
          onClick={() => setViewMode(mode)}
          className={`relative rounded-full px-3 py-1.5 text-[11px] font-medium uppercase tracking-[0.16em] transition-colors ${
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
