"use client";

import { useRef } from "react";
import { motion, useInView } from "framer-motion";

interface SectionWrapperProps {
  id: string;
  children: React.ReactNode;
  className?: string;
  fullHeight?: boolean;
}

export function SectionWrapper({
  id,
  children,
  className,
  fullHeight = true,
}: SectionWrapperProps) {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { amount: 0.2, once: true });

  return (
    <section
      id={id}
      ref={ref}
      className={`relative flex flex-col items-center justify-center px-4 py-20 sm:px-6 md:px-8 ${
        fullHeight ? "min-h-screen" : ""
      } ${className ?? ""}`}
    >
      <motion.div
        initial={{ opacity: 0, y: 50 }}
        animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 50 }}
        transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
        className="w-full"
        style={{ maxWidth: "var(--content-max-width)" }}
      >
        {children}
      </motion.div>
    </section>
  );
}
