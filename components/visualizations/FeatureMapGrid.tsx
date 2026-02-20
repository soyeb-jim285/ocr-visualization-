"use client";

import { useState, useId } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ActivationHeatmap } from "./ActivationHeatmap";

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
  const gridId = useId();
  const smCols = columnsSm ?? columns;

  if (!featureMaps || featureMaps.length === 0) {
    return (
      <div className="flex h-40 items-center justify-center rounded-xl border border-border bg-surface">
        <p className="text-foreground/30">
          Draw a character to see feature maps
        </p>
      </div>
    );
  }

  const expandedMap =
    expandedIdx !== null ? featureMaps[expandedIdx] : null;

  const cssId = `fmg${gridId.replace(/:/g, "")}`;

  return (
    <div className="flex flex-col gap-4">
      {smCols !== columns && (
        <style>{`
          .${cssId} { grid-template-columns: repeat(${smCols}, minmax(0, 1fr)); }
          @media (min-width: 640px) { .${cssId} { grid-template-columns: repeat(${columns}, minmax(0, 1fr)); } }
        `}</style>
      )}
      {/* Compact grid */}
      <div
        className={`grid gap-2 ${smCols !== columns ? cssId : ""}`}
        style={smCols === columns ? { gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` } : undefined}
      >
        {featureMaps.map((fm, i) => (
          <ActivationHeatmap
            key={`${layerName}-${i}`}
            data={fm}
            size={cellSize}
            label={`#${i + 1}`}
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
            className="overflow-hidden"
          >
            <div className="flex flex-col items-center gap-3 rounded-xl border border-accent-primary/30 bg-surface p-6">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium text-accent-primary">
                  Filter #{expandedIdx + 1}
                </span>
                <span className="text-xs text-foreground/40">
                  {expandedMap.length}x{expandedMap[0]?.length ?? 0}
                </span>
              </div>
              <ActivationHeatmap
                data={expandedMap}
                size={240}
              />
              <button
                onClick={() => setExpandedIdx(null)}
                className="text-xs text-foreground/40 hover:text-foreground/60"
              >
                Close
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
