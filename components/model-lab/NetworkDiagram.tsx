"use client";

import { useMemo } from "react";
import { useModelLabStore } from "@/stores/modelLabStore";
import { SIG, INK, INK2, INK3 } from "@/lib/theme";

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
  const expandedLayerId = useModelLabStore((s) => s.expandedLayerId);

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
      color: SIG.input,
      glowColor: SIG.input,
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
        color: SIG.conv1,
        glowColor: SIG.conv1,
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
      color: SIG.dense,
      glowColor: SIG.dense,
      type: "dense",
    });

    // Output
    result.push({
      label: "Output",
      sublabel: `${numClasses} classes · softmax`,
      dims: `${numClasses}`,
      width: scaleWidth(numClasses, minUnits, maxUnits),
      color: SIG.out,
      glowColor: SIG.out,
      type: "output",
    });

    return result;
  }, [architecture, validation, datasetType]);

  // Map expandedLayerId to block index (blocks[0]=Input, conv starts at 1)
  const highlightBlockIdx = expandedLayerId
    ? architecture.convLayers.findIndex((l) => l.id === expandedLayerId) + 1
    : -1;

  const blockHeight = 44;
  const gap = 28;
  const totalHeight = blocks.length * blockHeight + (blocks.length - 1) * gap + 32;
  const svgWidth = 400;
  const centerX = 140; // shifted left so right-hand dim labels fit

  return (
    <div className="figure">
      <h3 className="mb-1 font-serif text-xl text-ink">Architecture preview</h3>
      <p className="caption mb-4">Block width is log-scaled by channel count.</p>
      <div className="well plate-marks flex flex-col items-center overflow-hidden p-4">
      <svg
        viewBox={`0 0 ${svgWidth} ${totalHeight}`}
        className="w-full max-w-[400px]"
        style={{ height: "auto" }}
      >

        {blocks.map((block, i) => {
          const y = 16 + i * (blockHeight + gap);
          const halfW = block.width / 2;
          const rx = 2;
          const isHighlighted = i === highlightBlockIdx;

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
                opacity="0.18"
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
                fill={INK3}
                fontSize="10.5"
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
                fill={isHighlighted ? `${block.color}33` : `${block.color}12`}
                stroke={block.color}
                strokeOpacity={isHighlighted ? 1 : 0.55}
                strokeWidth={isHighlighted ? 1.5 : 1}
                style={{ transition: "fill 0.2s, stroke-opacity 0.2s, stroke-width 0.2s" }}
              />

              {/* Label */}
              <text
                x={centerX}
                y={y + 16}
                textAnchor="middle"
                dominantBaseline="central"
                fill={isHighlighted ? INK : INK2}
                fontSize="10.5"
                fontFamily="var(--font-mono), monospace"
                fontWeight={isHighlighted ? "600" : "500"}
              >
                {block.label}
              </text>

              {/* Sublabel */}
              <text
                x={centerX}
                y={y + 32}
                textAnchor="middle"
                dominantBaseline="central"
                fill={isHighlighted ? INK2 : INK3}
                fontSize="10.5"
                fontFamily="var(--font-mono), monospace"
              >
                {block.sublabel}
              </text>

              {/* Dims badge on the right */}
              <text
                x={centerX + halfW + 8}
                y={y + blockHeight / 2}
                dominantBaseline="central"
                fill={INK3}
                fontSize="10.5"
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
            <stop offset="0%" stopColor="#8896a3" />
            <stop offset="100%" stopColor="#8896a3" />
          </linearGradient>
        </defs>
      </svg>

      {/* Param count */}
      <div className="readout mt-3">
        {validation.paramCount < 1e6
          ? `${(validation.paramCount / 1e3).toFixed(1)}K params`
          : `${(validation.paramCount / 1e6).toFixed(1)}M params`}
      </div>
      </div>
      <p className="figcap">
        <b>FIG. 10.0</b> Live from your configuration. Expand a conv layer to highlight it here.
      </p>
    </div>
  );
}
