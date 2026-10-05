"use client";

import { createContext, useContext, type CSSProperties } from "react";
import { motion, useReducedMotion } from "framer-motion";

const PlateContext = createContext<{ mirror: boolean }>({ mirror: false });
export const usePlate = () => useContext(PlateContext);

interface SectionWrapperProps {
  id: string;
  children: React.ReactNode;
  className?: string;
  /** Opt in to min-h-svh (default false: plates are as tall as their content) */
  fullHeight?: boolean;
  /** Plate signal color: a CSS value ("var(--sig-pool)") or a short token ("pool") */
  sig?: string;
  /** Mirror the header margin note to the right (inherited by SectionHeader) */
  mirror?: boolean;
}

export function SectionWrapper({
  id,
  children,
  className,
  fullHeight = false,
  sig,
  mirror = false,
}: SectionWrapperProps) {
  const reduce = useReducedMotion();
  const sigValue = sig && /^[a-z0-9]+$/i.test(sig) ? `var(--sig-${sig})` : sig;

  return (
    <section
      id={id}
      className={`plate py-12 md:py-[clamp(64px,12svh,144px)] ${fullHeight ? "min-h-svh" : ""} ${className ?? ""}`}
      style={sigValue ? ({ "--sig": sigValue } as CSSProperties) : undefined}
    >
      <PlateContext.Provider value={{ mirror }}>
        <div className="relative z-10 mx-auto max-w-[1200px] px-4 md:px-16 min-[1400px]:px-8">
          <motion.div
            className="plate-rule mb-8 md:mb-14"
            initial={reduce ? false : { scaleX: 0 }}
            whileInView={{ scaleX: 1 }}
            viewport={{ once: true, margin: "-8% 0px" }}
            transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
            aria-hidden
          />
          {children}
        </div>
      </PlateContext.Provider>
    </section>
  );
}
