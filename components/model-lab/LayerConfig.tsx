"use client";

import type { ConvLayerConfig, Activation, PoolingType } from "@/lib/model-lab/architecture";
import {
  FILTER_OPTIONS,
  KERNEL_OPTIONS,
  ACTIVATION_OPTIONS,
  POOLING_OPTIONS,
} from "@/lib/model-lab/architecture";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Switch } from "@/components/ui/switch";

interface LayerConfigProps {
  layer: ConvLayerConfig;
  onUpdate: (updates: Partial<ConvLayerConfig>) => void;
}

export function LayerConfig({ layer, onUpdate }: LayerConfigProps) {
  return (
    <div className="space-y-3 pt-3">
      {/* Filters */}
      <div>
        <label className="mb-1 block text-[11px] uppercase tracking-wider text-foreground/35">
          Filters
        </label>
        <ToggleGroup
          type="single"
          value={String(layer.filters)}
          onValueChange={(v) => v && onUpdate({ filters: Number(v) as typeof layer.filters })}
          spacing={1}
          className="flex flex-wrap gap-1"
        >
          {FILTER_OPTIONS.map((opt) => (
            <ToggleGroupItem
              key={opt}
              value={String(opt)}
              className="h-auto min-w-0 shrink rounded-md px-2.5 py-1 text-xs font-medium bg-white/5 text-foreground/40 hover:bg-white/10 hover:text-foreground/60 data-[state=on]:bg-indigo-500/20 data-[state=on]:text-indigo-400"
            >
              {opt}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </div>

      {/* Kernel size */}
      <div>
        <label className="mb-1 block text-[11px] uppercase tracking-wider text-foreground/35">
          Kernel
        </label>
        <ToggleGroup
          type="single"
          value={String(layer.kernelSize)}
          onValueChange={(v) => v && onUpdate({ kernelSize: Number(v) as 3 | 5 | 7 })}
          spacing={1}
          className="flex flex-wrap gap-1"
        >
          {KERNEL_OPTIONS.map((opt) => (
            <ToggleGroupItem
              key={opt}
              value={String(opt)}
              className="h-auto min-w-0 shrink rounded-md px-2.5 py-1 text-xs font-medium bg-white/5 text-foreground/40 hover:bg-white/10 hover:text-foreground/60 data-[state=on]:bg-indigo-500/20 data-[state=on]:text-indigo-400"
            >
              {opt}×{opt}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </div>

      {/* Activation */}
      <div>
        <label className="mb-1 block text-[11px] uppercase tracking-wider text-foreground/35">
          Activation
        </label>
        <ToggleGroup
          type="single"
          value={layer.activation}
          onValueChange={(v) => v && onUpdate({ activation: v as Activation })}
          spacing={1}
          className="flex flex-wrap gap-1"
        >
          {ACTIVATION_OPTIONS.map((opt) => (
            <ToggleGroupItem
              key={opt}
              value={opt}
              className="h-auto min-w-0 shrink rounded-md px-2.5 py-1 text-xs font-medium bg-white/5 text-foreground/40 hover:bg-white/10 hover:text-foreground/60 data-[state=on]:bg-indigo-500/20 data-[state=on]:text-indigo-400"
            >
              {opt}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </div>

      {/* Pooling */}
      <div>
        <label className="mb-1 block text-[11px] uppercase tracking-wider text-foreground/35">
          Pooling
        </label>
        <ToggleGroup
          type="single"
          value={layer.pooling}
          onValueChange={(v) => v && onUpdate({ pooling: v as PoolingType })}
          spacing={1}
          className="flex flex-wrap gap-1"
        >
          {POOLING_OPTIONS.map((opt) => (
            <ToggleGroupItem
              key={opt}
              value={opt}
              className="h-auto min-w-0 shrink rounded-md px-2.5 py-1 text-xs font-medium bg-white/5 text-foreground/40 hover:bg-white/10 hover:text-foreground/60 data-[state=on]:bg-indigo-500/20 data-[state=on]:text-indigo-400"
            >
              {opt}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </div>

      {/* BatchNorm toggle */}
      <div className="flex items-center gap-2">
        <Switch
          size="sm"
          checked={layer.batchNorm}
          onCheckedChange={(checked) => onUpdate({ batchNorm: checked })}
          className="data-[state=checked]:bg-indigo-500"
        />
        <span className="text-xs text-foreground/50">BatchNorm</span>
      </div>
    </div>
  );
}
