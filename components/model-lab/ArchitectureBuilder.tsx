"use client";

import { useState } from "react";
import { ChevronUp, ChevronDown, Plus, Trash2 } from "lucide-react";
import { useModelLabStore } from "@/stores/modelLabStore";
import { LayerConfig } from "./LayerConfig";
import { MAX_CONV_LAYERS } from "@/lib/model-lab/architecture";
import type { Activation } from "@/lib/model-lab/architecture";

const ACTIVATION_OPTIONS: Activation[] = ["relu", "gelu", "silu", "leakyRelu", "tanh"];

function formatPooling(p: string) {
  if (p === "max") return "MaxPool";
  if (p === "avg") return "AvgPool";
  return "";
}

export function ArchitectureBuilder() {
  const {
    architecture,
    validation,
    addConvLayer,
    removeConvLayer,
    updateConvLayer,
    reorderConvLayers,
    setDenseConfig,
  } = useModelLabStore();

  const [expandedId, setExpandedId] = useState<string | null>(null);

  const { convLayers, dense } = architecture;
  const { spatialDims, paramCount, errors, warnings } = validation;

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold text-foreground/70">Architecture</h3>

      {/* Spatial dim flow */}
      <div className="flex flex-wrap items-center gap-1 text-[11px] text-foreground/40">
        <span className="rounded bg-white/5 px-1.5 py-0.5 font-mono">
          28×28×1
        </span>
        {spatialDims.slice(1).map((dim, i) => (
          <span key={i} className="flex items-center gap-1">
            <span className="text-foreground/20">→</span>
            <span className="rounded bg-white/5 px-1.5 py-0.5 font-mono">
              {dim.height}×{dim.width}×{convLayers[i]?.filters ?? "?"}
            </span>
          </span>
        ))}
      </div>

      {/* Conv layers stack */}
      <div className="space-y-2">
        {convLayers.map((layer, i) => {
          const isExpanded = expandedId === layer.id;
          const summary = `Conv ${layer.filters} ${layer.kernelSize}×${layer.kernelSize} ${layer.activation}${layer.pooling !== "none" ? ` ${formatPooling(layer.pooling)}` : ""}${layer.batchNorm ? " BN" : ""}`;

          return (
            <div
              key={layer.id}
              className="rounded-lg border border-border/50 bg-black/20"
            >
              {/* Header */}
              <div className="flex items-center gap-2 px-3 py-2">
                <span className="shrink-0 font-mono text-[10px] text-foreground/30">
                  {i + 1}
                </span>

                <button
                  onClick={() =>
                    setExpandedId(isExpanded ? null : layer.id)
                  }
                  className="flex-1 text-left text-xs font-medium text-foreground/65 hover:text-foreground/80"
                >
                  {summary}
                </button>

                {/* Reorder */}
                <button
                  onClick={() => i > 0 && reorderConvLayers(i, i - 1)}
                  disabled={i === 0}
                  className="p-0.5 text-foreground/25 hover:text-foreground/50 disabled:opacity-20"
                >
                  <ChevronUp size={14} />
                </button>
                <button
                  onClick={() =>
                    i < convLayers.length - 1 &&
                    reorderConvLayers(i, i + 1)
                  }
                  disabled={i === convLayers.length - 1}
                  className="p-0.5 text-foreground/25 hover:text-foreground/50 disabled:opacity-20"
                >
                  <ChevronDown size={14} />
                </button>

                {/* Remove */}
                <button
                  onClick={() => removeConvLayer(layer.id)}
                  disabled={convLayers.length <= 1}
                  className="p-0.5 text-foreground/25 hover:text-red-400 disabled:opacity-20"
                >
                  <Trash2 size={14} />
                </button>
              </div>

              {/* Expanded config */}
              {isExpanded && (
                <div className="border-t border-border/30 px-3 pb-3">
                  <LayerConfig
                    layer={layer}
                    onUpdate={(updates) =>
                      updateConvLayer(layer.id, updates)
                    }
                  />
                </div>
              )}
            </div>
          );
        })}

        {/* Add layer */}
        {convLayers.length < MAX_CONV_LAYERS && (
          <button
            onClick={addConvLayer}
            className="flex w-full items-center justify-center gap-1.5 rounded-lg border border-dashed border-border/40 py-2 text-xs text-foreground/35 transition hover:border-indigo-500/40 hover:text-indigo-400"
          >
            <Plus size={14} />
            Add Conv Layer
          </button>
        )}
      </div>

      {/* Dense config */}
      <div className="rounded-lg border border-border/50 bg-black/20 px-3 py-3">
        <h4 className="mb-2 text-xs font-semibold text-foreground/55">
          Dense Layer
        </h4>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="mb-1 block text-[10px] uppercase tracking-wider text-foreground/30">
              Width
            </label>
            <input
              type="number"
              min={64}
              max={1024}
              step={64}
              value={dense.width}
              onChange={(e) =>
                setDenseConfig({
                  width: Math.max(64, Math.min(1024, +e.target.value)),
                })
              }
              className="w-full rounded-md border border-border/40 bg-black/30 px-2 py-1 font-mono text-xs text-foreground/70"
            />
          </div>
          <div>
            <label className="mb-1 block text-[10px] uppercase tracking-wider text-foreground/30">
              Dropout
            </label>
            <input
              type="number"
              min={0}
              max={0.7}
              step={0.05}
              value={dense.dropout}
              onChange={(e) =>
                setDenseConfig({
                  dropout: Math.max(0, Math.min(0.7, +e.target.value)),
                })
              }
              className="w-full rounded-md border border-border/40 bg-black/30 px-2 py-1 font-mono text-xs text-foreground/70"
            />
          </div>
        </div>
        <div className="mt-2">
          <label className="mb-1 block text-[10px] uppercase tracking-wider text-foreground/30">
            Activation
          </label>
          <div className="flex flex-wrap gap-1">
            {ACTIVATION_OPTIONS.map((act) => (
              <button
                key={act}
                onClick={() => setDenseConfig({ activation: act })}
                className={`rounded-md px-2 py-1 text-xs font-medium transition ${
                  dense.activation === act
                    ? "bg-indigo-500/20 text-indigo-400"
                    : "bg-white/5 text-foreground/40 hover:bg-white/10"
                }`}
              >
                {act}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Param count */}
      <div className="text-center font-mono text-xs text-foreground/40">
        {paramCount < 1e6
          ? `${(paramCount / 1e3).toFixed(1)}K`
          : `${(paramCount / 1e6).toFixed(1)}M`}{" "}
        parameters
      </div>

      {/* Errors & warnings */}
      {errors.length > 0 && (
        <div className="space-y-1">
          {errors.map((err, i) => (
            <p key={i} className="text-xs text-red-400">
              {err}
            </p>
          ))}
        </div>
      )}
      {warnings.length > 0 && (
        <div className="space-y-1">
          {warnings.map((warn, i) => (
            <p key={i} className="text-xs text-yellow-400/70">
              {warn}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}
