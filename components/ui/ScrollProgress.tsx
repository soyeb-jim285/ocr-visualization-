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
];

export function ScrollProgress() {
  const { scrollYProgress } = useScroll();
  const activeSection = useUIStore((s) => s.activeSection);

  const scaleX = useTransform(scrollYProgress, [0, 1], [0, 1]);
  const lastSection = SECTION_LABELS.length - 1;
  const safeActiveSection = Math.min(Math.max(activeSection, 0), lastSection);
  const activeLabel = SECTION_LABELS[safeActiveSection] ?? SECTION_LABELS[0];

  const scrollToSection = (index: number) => {
    const sectionId = SECTION_IDS[index];
    if (!sectionId) return;
    const el = document.getElementById(sectionId);
    el?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div className="fixed left-0 right-0 top-0 z-50">
      {/* Progress bar */}
      <div className="h-0.5 w-full bg-border/20">
        <motion.div
          className="h-full origin-left bg-accent-primary"
          style={{ scaleX }}
        />
      </div>

      {/* Section dots - desktop */}
      <div className="hidden items-center justify-center gap-1 bg-background/70 py-2 backdrop-blur-lg md:flex">
        {SECTION_LABELS.map((label, i) => (
          <button
            key={i}
            onClick={() => scrollToSection(i)}
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

      {/* Compact mobile section navigator */}
      <div className="mx-2 mt-1 flex items-center justify-between rounded-full border border-border/50 bg-background/72 px-2 py-1 shadow-lg shadow-black/25 backdrop-blur-xl md:hidden">
        <button
          type="button"
          onClick={() => scrollToSection(Math.max(0, safeActiveSection - 1))}
          disabled={safeActiveSection <= 0}
          className="rounded-full px-2 py-1 text-[10px] font-medium uppercase tracking-[0.16em] text-foreground/60 transition disabled:opacity-30"
        >
          Prev
        </button>

        <button
          type="button"
          onClick={() => scrollToSection(safeActiveSection)}
          className="rounded-full border border-border/60 bg-black/20 px-3 py-1 text-[10px] uppercase tracking-[0.16em] text-foreground/68"
          aria-label={`Current section: ${activeLabel}`}
        >
          {safeActiveSection + 1}/{SECTION_LABELS.length} - {activeLabel}
        </button>

        <button
          type="button"
          onClick={() => scrollToSection(Math.min(lastSection, safeActiveSection + 1))}
          disabled={safeActiveSection >= lastSection}
          className="rounded-full px-2 py-1 text-[10px] font-medium uppercase tracking-[0.16em] text-foreground/60 transition disabled:opacity-30"
        >
          Next
        </button>
      </div>
    </div>
  );
}
