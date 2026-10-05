"use client";

import { useState, useEffect } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { useUIStore } from "@/stores/uiStore";

const scripts = ["English", "বাংলা", "123৪৫৬", "সংযুক্ত"];

function Specimen() {
  const reduce = useReducedMotion();
  const [idx, setIdx] = useState(0);
  const [displayed, setDisplayed] = useState("");
  const [isDeleting, setIsDeleting] = useState(false);

  useEffect(() => {
    if (reduce) return;
    const word = scripts[idx];
    let timer: ReturnType<typeof setTimeout>;

    if (!isDeleting && displayed.length < word.length) {
      timer = setTimeout(() => setDisplayed(word.slice(0, displayed.length + 1)), 90);
    } else if (!isDeleting && displayed.length === word.length) {
      timer = setTimeout(() => setIsDeleting(true), 1400);
    } else if (isDeleting && displayed.length > 0) {
      timer = setTimeout(() => setDisplayed(displayed.slice(0, -1)), 50);
    } else {
      timer = setTimeout(() => {
        setIsDeleting(false);
        setIdx((i) => (i + 1) % scripts.length);
      }, 0);
    }

    return () => clearTimeout(timer);
  }, [displayed, isDeleting, idx, reduce]);

  return (
    <p className="h-5 font-mono text-[11px] tracking-[0.06em] text-ink-3">
      SPECIMEN ·{" "}
      <span className="text-ink-2">{reduce ? scripts[0] : displayed}</span>
      {!reduce && <span className="animate-pulse text-phosphor">|</span>}
    </p>
  );
}

export function HeroHeader() {
  const modelLoaded = useUIStore((s) => s.modelLoaded);
  const reduce = useReducedMotion();

  // Staggered rise-in on mount; content is visible by default under reduced motion
  const rise = (delay: number) =>
    reduce
      ? {}
      : {
          initial: { opacity: 0, y: 14 },
          animate: { opacity: 1, y: 0 },
          transition: { duration: 0.7, delay, ease: [0.16, 1, 0.3, 1] as const },
        };

  return (
    <div className="flex max-w-[44rem] flex-col items-start">
      <motion.p {...rise(0)} className="flex items-center gap-2 font-mono text-[11px] tracking-[0.08em] text-ink-3">
        <span className="inline-block size-1.5 shrink-0 rounded-full bg-annotation" aria-hidden />
        {modelLoaded ? "MODEL COMBINED-CNN v2 · 13 LAYERS · 146 CLASSES" : "LOADING WEIGHTS"}
      </motion.p>

      <motion.h1
        {...rise(0.08)}
        className="mt-3 text-balance font-serif text-[clamp(1.9rem,8.5vw,2.4rem)] font-light leading-[1.02] tracking-[-0.02em] text-ink/90 sm:mt-6 sm:text-[clamp(2.6rem,6.2vw,5.25rem)] sm:leading-[0.98]"
      >
        Watch a network <em className="italic text-ink">read</em> your handwriting.
      </motion.h1>

      <motion.p {...rise(0.18)} className="lead mt-3 max-w-[62ch] font-serif text-base leading-[1.45] text-ink-2 sm:mt-6 sm:text-[1.25rem] sm:leading-[1.55]">
        Draw a character. Thirteen layers of arithmetic turn it into a guess<span className="hidden sm:inline">, and you can inspect every step</span>.
      </motion.p>

      <motion.div {...rise(0.28)} className="mt-8 hidden sm:block">
        <Specimen />
      </motion.div>
    </div>
  );
}
