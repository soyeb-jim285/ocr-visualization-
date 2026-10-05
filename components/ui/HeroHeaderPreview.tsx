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
        {modelLoaded ? "MODEL EMNIST-CNN · 13 LAYERS · 146 CLASSES" : "LOADING WEIGHTS"}
      </motion.p>

      <motion.h1
        {...rise(0.08)}
        className="mt-6 text-balance font-serif text-[clamp(2.4rem,11vw,3.25rem)] font-light leading-[0.98] tracking-[-0.02em] text-ink/90 sm:text-[clamp(2.6rem,6.2vw,5.25rem)]"
      >
        Watch a network <em className="italic text-ink">read</em> your handwriting.
      </motion.h1>

      <motion.p {...rise(0.18)} className="lead mt-6 max-w-[62ch] font-serif text-[1.125rem] leading-[1.55] text-ink-2 sm:text-[1.25rem]">
        Draw a character. Thirteen layers of arithmetic will turn it into a guess, and you can inspect every step.
      </motion.p>

      <motion.div {...rise(0.28)} className="mt-8 hidden sm:block">
        <Specimen />
      </motion.div>
    </div>
  );
}
