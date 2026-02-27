"use client";

import type { ConvLayerConfig, Activation, PoolingType } from "@/lib/model-lab/architecture";
import {
  FILTER_OPTIONS,
  KERNEL_OPTIONS,
  ACTIVATION_OPTIONS,
  POOLING_OPTIONS,
} from "@/lib/model-lab/architecture";

interface LayerConfigProps {
  layer: ConvLayerConfig;
  onUpdate: (updates: Partial<ConvLayerConfig>) => void;
}

function ButtonGroup<T extends string | number>({
  options,
  value,
  onChange,
  format,
}: {
  options: readonly T[];
  value: T;
  onChange: (v: T) => void;
  format?: (v: T) => string;
}) {
  return (
    <div className="flex flex-wrap gap-1">
      {options.map((opt) => (
        <button
          key={String(opt)}
          onClick={() => onChange(opt)}
          className={`rounded-md px-2.5 py-1 text-xs font-medium transition ${
            value === opt
              ? "bg-indigo-500/20 text-indigo-400"
              : "bg-white/5 text-foreground/40 hover:bg-white/10 hover:text-foreground/60"
          }`}
        >
          {format ? format(opt) : String(opt)}
        </button>
      ))}
    </div>
  );
}

export function LayerConfig({ layer, onUpdate }: LayerConfigProps) {
  return (
    <div className="space-y-3 pt-3">
      {/* Filters */}
      <div>
        <label className="mb-1 block text-[11px] uppercase tracking-wider text-foreground/35">
          Filters
        </label>
        <ButtonGroup
          options={FILTER_OPTIONS}
          value={layer.filters}
          onChange={(v) => onUpdate({ filters: v })}
        />
      </div>

      {/* Kernel size */}
      <div>
        <label className="mb-1 block text-[11px] uppercase tracking-wider text-foreground/35">
          Kernel
        </label>
        <ButtonGroup
          options={KERNEL_OPTIONS}
          value={layer.kernelSize}
          onChange={(v) => onUpdate({ kernelSize: v as 3 | 5 | 7 })}
          format={(v) => `${v}×${v}`}
        />
      </div>

      {/* Activation */}
      <div>
        <label className="mb-1 block text-[11px] uppercase tracking-wider text-foreground/35">
          Activation
        </label>
        <ButtonGroup<Activation>
          options={ACTIVATION_OPTIONS}
          value={layer.activation}
          onChange={(v) => onUpdate({ activation: v })}
        />
      </div>

      {/* Pooling */}
      <div>
        <label className="mb-1 block text-[11px] uppercase tracking-wider text-foreground/35">
          Pooling
        </label>
        <ButtonGroup<PoolingType>
          options={POOLING_OPTIONS}
          value={layer.pooling}
          onChange={(v) => onUpdate({ pooling: v })}
        />
      </div>

      {/* BatchNorm toggle */}
      <div className="flex items-center gap-2">
        <button
          onClick={() => onUpdate({ batchNorm: !layer.batchNorm })}
          className={`h-5 w-9 rounded-full transition ${
            layer.batchNorm ? "bg-indigo-500" : "bg-white/10"
          }`}
        >
          <div
            className={`h-4 w-4 rounded-full bg-white transition-transform ${
              layer.batchNorm ? "translate-x-4" : "translate-x-0.5"
            }`}
          />
        </button>
        <span className="text-xs text-foreground/50">BatchNorm</span>
      </div>
    </div>
  );
}
