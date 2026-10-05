"use client";

import { useState, type ReactNode, type CSSProperties } from "react";
import {
  motion,
  AnimatePresence,
  useReducedMotion,
} from "framer-motion";
import { ActivationHeatmap } from "./ActivationHeatmap";

/** Scroll-reveal wrapper shared by the mid sections (once, transform/opacity only). */
export function Reveal({
  children,
  delay = 0,
  className,
}: {
  children: ReactNode;
  delay?: number;
  className?: string;
}) {
  const reduce = useReducedMotion();
  return (
    <motion.div
      className={className}
      initial={reduce ? false : { opacity: 0, y: 12 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-12% 0px" }}
      transition={{ duration: reduce ? 0.15 : 0.5, delay, ease: [0.16, 1, 0.3, 1] }}
    >
      {children}
    </motion.div>
  );
}

interface FeatureMapGridProps {
  /** Array of feature maps: [numFilters][height][width] */
  featureMaps: number[][][];
  layerName: string;
  columns?: number;
  columnsSm?: number;
  cellSize?: number;
}

export function FeatureMapGrid({
  featureMaps,
  layerName,
  columns = 8,
  columnsSm,
  cellSize = 72,
}: FeatureMapGridProps) {
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);
  const smCols = columnsSm ?? columns;

  if (!featureMaps || featureMaps.length === 0) {
    return (
      <div className="viz-empty-state h-40">
        <p className="font-serif italic">Draw something to light this up</p>
      </div>
    );
  }

  const expandedMap =
    expandedIdx !== null ? featureMaps[expandedIdx] : null;

  return (
    <div className="flex flex-col gap-4">
      {/* Specimen grid */}
      <div
        className="grid grid-cols-[repeat(var(--cs),minmax(0,1fr))] justify-items-center gap-x-2 gap-y-3 sm:grid-cols-[repeat(var(--c),minmax(0,1fr))]"
        style={{ "--c": columns, "--cs": smCols } as CSSProperties}
      >
        {featureMaps.map((fm, i) => (
          <ActivationHeatmap
            key={`${layerName}-${i}`}
            data={fm}
            size={cellSize}
            label={String(i + 1).padStart(3, "0")}
            onClick={() => setExpandedIdx(expandedIdx === i ? null : i)}
            selected={expandedIdx === i}
          />
        ))}
      </div>

      {/* Expanded view */}
      <AnimatePresence>
        {expandedMap && expandedIdx !== null && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
            className="overflow-hidden"
          >
            <div className="figure mx-auto flex max-w-sm flex-col items-center gap-3">
              <div className="well plate-marks p-2">
                <ActivationHeatmap data={expandedMap} size={220} />
              </div>
              <p className="figcap mt-0 text-center">
                <b>
                  {layerName.toUpperCase()} · {String(expandedIdx + 1).padStart(3, "0")}
                </b>{" "}
                {expandedMap.length}×{expandedMap[0]?.length ?? 0}
              </p>
              <button onClick={() => setExpandedIdx(null)} className="text-btn">
                Close
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
