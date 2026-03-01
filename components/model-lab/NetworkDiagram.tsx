"use client";

import { useMemo } from "react";
import { useModelLabStore } from "@/stores/modelLabStore";

// Dataset → class count (matches architectureValidator logic)
function getNumClasses(dataset: string): number {
  if (dataset === "digits") return 10;
  if (dataset === "bangla") return 84;
  if (dataset === "combined") return 146;
  return 62; // emnist
}

// Log-scale width so filter counts 16→512 don't span too wildly
function scaleWidth(units: number, min: number, max: number): number {
  const logMin = Math.log2(Math.max(1, min));
  const logMax = Math.log2(Math.max(1, max));
  const logVal = Math.log2(Math.max(1, units));
  const t = logMax === logMin ? 0.5 : (logVal - logMin) / (logMax - logMin);
  return 60 + t * 180; // 60px to 240px
}

interface BlockDef {
  label: string;
  sublabel: string;
  dims: string;
  width: number;
  color: string;
  glowColor: string;
  type: "input" | "conv" | "dense" | "output";
}

export function NetworkDiagram() {
  const architecture = useModelLabStore((s) => s.architecture);
  const validation = useModelLabStore((s) => s.validation);
  const datasetType = useModelLabStore((s) => s.datasetType);

  const blocks = useMemo(() => {
    const { convLayers, dense } = architecture;
    const { spatialDims } = validation;
    const numClasses = getNumClasses(datasetType);

    // Gather all unit counts to determine scale range
    const allUnits = [
      1, // input channels
      ...convLayers.map((l) => l.filters),
      dense.width,
      numClasses,
    ];
    const minUnits = Math.min(...allUnits);
    const maxUnits = Math.max(...allUnits);

    const result: BlockDef[] = [];

    // Input
    result.push({
      label: "Input",
      sublabel: "28 × 28 × 1",
      dims: "28×28×1",
      width: scaleWidth(1, minUnits, maxUnits),
      color: "#06b6d4",
      glowColor: "#06b6d4",
      type: "input",
    });

    // Conv layers
    convLayers.forEach((layer, i) => {
      const dim = spatialDims[i + 1]; // spatialDims[0] is input
      const h = dim?.height ?? "?";
      const w = dim?.width ?? "?";

      const parts: string[] = [layer.activation];
      if (layer.pooling !== "none")
        parts.push(layer.pooling === "max" ? "MaxPool" : "AvgPool");
      if (layer.batchNorm) parts.push("BN");

      result.push({
        label: `Conv2D · ${layer.filters} · ${layer.kernelSize}×${layer.kernelSize}`,
        sublabel: parts.join(" · "),
        dims: `${h}×${w}×${layer.filters}`,
        width: scaleWidth(layer.filters, minUnits, maxUnits),
        color: "#6366f1",
        glowColor: "#6366f1",
        type: "conv",
      });
    });

    // Dense
    const dropStr = dense.dropout > 0 ? ` · drop ${dense.dropout}` : "";
    result.push({
      label: `Dense · ${dense.width}`,
      sublabel: `${dense.activation}${dropStr}`,
      dims: `${dense.width}`,
      width: scaleWidth(dense.width, minUnits, maxUnits),
      color: "#8b5cf6",
      glowColor: "#8b5cf6",
      type: "dense",
    });

    // Output
    result.push({
      label: "Output",
      sublabel: `${numClasses} classes · softmax`,
      dims: `${numClasses}`,
      width: scaleWidth(numClasses, minUnits, maxUnits),
      color: "#06b6d4",
      glowColor: "#06b6d4",
      type: "output",
    });

    return result;
  }, [architecture, validation, datasetType]);

  const blockHeight = 44;
  const gap = 28;
  const totalHeight = blocks.length * blockHeight + (blocks.length - 1) * gap + 32;
  const svgWidth = 320;
  const centerX = svgWidth / 2;

  return (
    <div className="flex flex-col items-center rounded-xl border border-border/30 bg-black/20 p-4">
      <h3 className="mb-3 text-sm font-semibold text-foreground/70">
        Architecture
      </h3>
      <svg
        viewBox={`0 0 ${svgWidth} ${totalHeight}`}
        className="w-full max-w-[320px]"
        style={{ height: "auto" }}
      >
        <defs>
          {/* Glow filters for each color */}
          {["#06b6d4", "#6366f1", "#8b5cf6"].map((c, i) => (
            <filter key={i} id={`glow-${i}`} x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur in="SourceGraphic" stdDeviation="3" result="blur" />
              <feFlood floodColor={c} floodOpacity="0.3" />
              <feComposite in2="blur" operator="in" />
              <feMerge>
                <feMergeNode />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          ))}
        </defs>

        {blocks.map((block, i) => {
          const y = 16 + i * (blockHeight + gap);
          const halfW = block.width / 2;
          const rx = 6;
          const filterIdx =
            block.color === "#06b6d4" ? 0 : block.color === "#6366f1" ? 1 : 2;

          // Connector to next block
          let connector = null;
          if (i < blocks.length - 1) {
            const next = blocks[i + 1];
            const nextY = 16 + (i + 1) * (blockHeight + gap);
            const curHalfW = halfW;
            const nextHalfW = next.width / 2;
            const connY1 = y + blockHeight;
            const connY2 = nextY;
            // Trapezoid connector
            connector = (
              <polygon
                points={`
                  ${centerX - curHalfW * 0.5},${connY1}
                  ${centerX + curHalfW * 0.5},${connY1}
                  ${centerX + nextHalfW * 0.5},${connY2}
                  ${centerX - nextHalfW * 0.5},${connY2}
                `}
                fill="url(#connGrad)"
                opacity="0.15"
              />
            );
          }

          // Flatten marker between last conv and dense
          let flattenMarker = null;
          if (block.type === "dense" && i > 0 && blocks[i - 1].type === "conv") {
            const markerY = y - gap / 2;
            flattenMarker = (
              <text
                x={centerX}
                y={markerY}
                textAnchor="middle"
                dominantBaseline="central"
                fill="rgba(232,232,237,0.25)"
                fontSize="9"
                fontFamily="var(--font-mono), monospace"
              >
                flatten
              </text>
            );
          }

          return (
            <g key={i}>
              {connector}
              {flattenMarker}

              {/* Block rect */}
              <rect
                x={centerX - halfW}
                y={y}
                width={block.width}
                height={blockHeight}
                rx={rx}
                fill={`${block.color}10`}
                stroke={block.color}
                strokeOpacity={0.4}
                strokeWidth={1}
                filter={`url(#glow-${filterIdx})`}
              />

              {/* Label */}
              <text
                x={centerX}
                y={y + 16}
                textAnchor="middle"
                dominantBaseline="central"
                fill="rgba(232,232,237,0.7)"
                fontSize="10"
                fontFamily="var(--font-mono), monospace"
                fontWeight="500"
              >
                {block.label}
              </text>

              {/* Sublabel */}
              <text
                x={centerX}
                y={y + 32}
                textAnchor="middle"
                dominantBaseline="central"
                fill="rgba(232,232,237,0.3)"
                fontSize="9"
                fontFamily="var(--font-mono), monospace"
              >
                {block.sublabel}
              </text>

              {/* Dims badge on the right */}
              <text
                x={centerX + halfW + 8}
                y={y + blockHeight / 2}
                dominantBaseline="central"
                fill="rgba(232,232,237,0.2)"
                fontSize="9"
                fontFamily="var(--font-mono), monospace"
              >
                {block.dims}
              </text>
            </g>
          );
        })}

        {/* Gradient for connectors */}
        <defs>
          <linearGradient id="connGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgba(232,232,237,0.5)" />
            <stop offset="100%" stopColor="rgba(232,232,237,0.5)" />
          </linearGradient>
        </defs>
      </svg>

      {/* Param count */}
      <div className="mt-3 font-mono text-xs text-foreground/30">
        {validation.paramCount < 1e6
          ? `${(validation.paramCount / 1e3).toFixed(1)}K params`
          : `${(validation.paramCount / 1e6).toFixed(1)}M params`}
      </div>
    </div>
  );
}
