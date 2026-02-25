"use client";

import { motion, useScroll, useTransform } from "framer-motion";
import { useUIStore } from "@/stores/uiStore";
import { SECTION_IDS } from "@/hooks/useScrollSection";

const SECTION_LABELS = [
  "Hero",
  "Pixels",
  "Conv",
  "ReLU",
  "Conv2",
  "Pool",
  "Deep",
  "Dense",
  "Softmax",
  "Training",
  "Inspector",
];

export function ScrollProgress() {
  const { scrollYProgress } = useScroll();
  const activeSection = useUIStore((s) => s.activeSection);

  const scaleX = useTransform(scrollYProgress, [0, 1], [0, 1]);

  return (
    <div className="fixed left-0 right-0 top-0 z-50">
      {/* Progress bar */}
      <div className="h-0.5 w-full bg-border/20">
        <motion.div
          className="h-full origin-left bg-gradient-to-r from-accent-primary to-accent-tertiary"
          style={{ scaleX }}
        />
      </div>

      {/* Section dots - hidden on mobile */}
      <div className="hidden items-center justify-center gap-1 bg-background/80 py-2 backdrop-blur-sm md:flex">
        {SECTION_LABELS.map((label, i) => (
          <button
            key={i}
            onClick={() => {
              const el = document.getElementById(SECTION_IDS[i]);
              el?.scrollIntoView({ behavior: "smooth" });
            }}
            className={`group flex items-center gap-1.5 rounded-full px-2 py-1 text-xs transition-all ${
              activeSection === i
                ? "bg-accent-primary/10 text-accent-primary"
                : "text-foreground/30 hover:text-foreground/50"
            }`}
          >
            <div
              className={`h-1.5 w-1.5 rounded-full transition-all ${
                activeSection === i
                  ? "bg-accent-primary"
                  : "bg-foreground/20 group-hover:bg-foreground/40"
              }`}
            />
            <span className={activeSection === i ? "" : "hidden lg:inline"}>
              {label}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
