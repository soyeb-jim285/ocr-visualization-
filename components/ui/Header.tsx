"use client";

import { useEffect, useRef } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
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
  "Model Lab",
];

const pad = (n: number) => String(n).padStart(2, "0");

export function Header() {
  const activeSection = useUIStore((s) => s.activeSection);
  const barRef = useRef<HTMLDivElement>(null);

  const lastSection = SECTION_LABELS.length - 1;
  const safeActiveSection = Math.min(Math.max(activeSection, 0), lastSection);
  const activeLabel = SECTION_LABELS[safeActiveSection] ?? SECTION_LABELS[0];

  // Top hairline progress: writes the transform directly, no React state per scroll
  useEffect(() => {
    let raf = 0;
    const update = () => {
      const max = document.documentElement.scrollHeight - window.innerHeight;
      const p = max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 0;
      if (barRef.current) barRef.current.style.transform = `scaleX(${p})`;
    };
    const onScroll = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(update);
    };
    update();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
    };
  }, []);

  const scrollToSection = (index: number) => {
    const sectionId = SECTION_IDS[index];
    if (!sectionId) return;
    const el = document.getElementById(sectionId);
    el?.scrollIntoView(); // smooth vs. reduced-motion handled by html scroll-behavior in globals.css
  };

  return (
    <>
      {/* Reading progress hairline */}
      <div aria-hidden className="pointer-events-none fixed inset-x-0 top-0 z-40 h-px bg-rule">
        <div ref={barRef} className="h-full origin-left bg-phosphor" style={{ transform: "scaleX(0)" }} />
      </div>

      {/* Desktop index rail */}
      <nav
        aria-label="Sections"
        className="group/rail fixed left-6 top-5 z-40 hidden md:block"
      >
        <button
          type="button"
          onClick={() => scrollToSection(0)}
          className="font-mono text-[11px] font-medium tracking-[0.12em] text-ink transition-colors duration-150 hover:text-phosphor"
        >
          NNXR
        </button>
        <ol className="mt-6 border-l border-rule">
          {SECTION_LABELS.map((label, i) => {
            const active = safeActiveSection === i;
            return (
              <li key={label}>
                <button
                  type="button"
                  onClick={() => scrollToSection(i)}
                  aria-label={label}
                  aria-current={active ? "location" : undefined}
                  className={`group relative flex h-6 items-center pl-3 font-mono text-[11px] transition-colors duration-150 hover:text-ink ${
                    active ? "text-phosphor" : "text-ink-3"
                  }`}
                >
                  <span
                    aria-hidden
                    className={`absolute left-0 top-1/2 h-px w-2 origin-left transition-transform duration-200 ${
                      active ? "scale-x-[2.5] bg-phosphor" : "bg-ink-4"
                    }`}
                  />
                  <span className="w-5 pl-1">{pad(i)}</span>
                  <span
                    className={`absolute left-10 whitespace-nowrap rounded-[2px] bg-bg/80 px-1.5 transition-opacity duration-150 group-hover/rail:opacity-100 group-focus-within/rail:opacity-100 ${
                      active ? "opacity-0 min-[1536px]:opacity-100" : "opacity-0"
                    }`}
                  >
                    {label}
                  </span>
                </button>
              </li>
            );
          })}
        </ol>
      </nav>

      {/* Compact mobile section navigator */}
      <div className="fixed inset-x-4 top-[max(12px,env(safe-area-inset-top))] z-40 flex h-11 items-center justify-between rounded-[4px] border border-rule bg-bg-raised/90 backdrop-blur-md md:hidden">
        <button
          type="button"
          onClick={() => scrollToSection(Math.max(0, safeActiveSection - 1))}
          disabled={safeActiveSection <= 0}
          aria-label="Previous section"
          className="flex size-11 items-center justify-center text-ink-2 transition-colors disabled:opacity-30"
        >
          <ChevronLeft className="size-4" />
        </button>

        <button
          type="button"
          onClick={() => scrollToSection(safeActiveSection)}
          className="h-full flex-1 font-mono text-[11px] uppercase tracking-[0.08em] text-ink"
          aria-label={`Current section: ${activeLabel}`}
        >
          {pad(safeActiveSection + 1)}/{SECTION_LABELS.length} · {activeLabel}
        </button>

        <button
          type="button"
          onClick={() => scrollToSection(Math.min(lastSection, safeActiveSection + 1))}
          disabled={safeActiveSection >= lastSection}
          aria-label="Next section"
          className="flex size-11 items-center justify-center text-ink-2 transition-colors disabled:opacity-30"
        >
          <ChevronRight className="size-4" />
        </button>
      </div>
    </>
  );
}
