"use client";

import type { ReactNode } from "react";
import { motion, useReducedMotion } from "framer-motion";

interface SectionHeaderProps {
  title: string;
  subtitle: string;
  /** Plate number, shown as PLATE 04 */
  step?: number;
  /** Mono margin note, e.g. "Conv2 · 128 ch · 14×14" */
  tag?: string;
  /** Optional extra content rendered under the lead (inside the header block) */
  figure?: ReactNode;
  /** Defaults to the SectionWrapper's `mirror` */
  mirror?: boolean;
  /** Full-width title with a single-row margin note (Training, Model Lab) */
  wide?: boolean;
}

export function SectionHeader({
  title,
  subtitle,
  step,
  tag,
  figure,
  mirror,
  wide = false,
}: SectionHeaderProps) {
  const reduce = useReducedMotion();
  const isMirror = !wide && (mirror ?? false); // one consistent layout: meta left, title right
  const plateNo = step !== undefined ? `PLATE ${String(step).padStart(2, "0")}` : null;

  const reveal = (delay = 0) => ({
    initial: reduce ? false : ({ opacity: 0, y: 12 } as const),
    whileInView: { opacity: 1, y: 0 },
    viewport: { once: true, margin: "-12% 0px" },
    transition: reduce
      ? { duration: 0.15 }
      : { duration: 0.5, delay, ease: [0.16, 1, 0.3, 1] as const },
  });

  const note = (
    <>
      {plateNo && <p className="eyebrow">{plateNo}</p>}
      {plateNo && tag && <div className="my-3 h-px w-10 bg-sig/60" />}
      {tag && <p className="caption">{tag}</p>}
    </>
  );

  return (
    <header className="mb-12 grid grid-cols-12 gap-x-6 md:mb-16">
      {/* Mobile: one mono line above the title */}
      {(plateNo || tag) && (
        <motion.p {...reveal()} className="eyebrow col-span-12 mb-4 md:hidden">
          {plateNo}
          {plateNo && tag && <span className="text-ink-3"> · {tag}</span>}
        </motion.p>
      )}

      {/* Desktop margin column */}
      {!wide && (plateNo || tag) && (
        <motion.div
          {...reveal()}
          className={`hidden pt-[0.6em] md:block md:col-span-3 ${
            isMirror ? "order-2 col-start-10 text-right [&>div]:ml-auto" : "order-1"
          }`}
        >
          {note}
        </motion.div>
      )}

      <div
        className={`col-span-12 ${
          wide
            ? ""
            : isMirror
              ? "md:order-1 md:col-span-7 md:col-start-1"
              : "md:order-2 md:col-span-7 md:col-start-4"
        }`}
      >
        <motion.h2
          {...reveal(0.04)}
          className="text-balance font-serif text-[clamp(2rem,3.8vw,3.25rem)] font-normal leading-[1.05] tracking-[-0.015em] text-ink"
        >
          {title}
        </motion.h2>
        <motion.p {...reveal(0.12)} className="lead mt-6">
          {subtitle}
        </motion.p>
        {figure && <div className="mt-8">{figure}</div>}
      </div>

      {wide && (plateNo || tag) && (
        <motion.div
          {...reveal(0.16)}
          className="col-span-12 mt-6 hidden items-center gap-4 md:flex"
        >
          {plateNo && <span className="eyebrow">{plateNo}</span>}
          {plateNo && tag && <span className="h-px w-10 bg-sig/60" />}
          {tag && <span className="caption">{tag}</span>}
        </motion.div>
      )}
    </header>
  );
}
