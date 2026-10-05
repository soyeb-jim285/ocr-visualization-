"use client";

import type { ReactNode } from "react";
import type { ConvLayerConfig, Activation, PoolingType } from "@/lib/model-lab/architecture";
import {
  FILTER_OPTIONS,
  KERNEL_OPTIONS,
  ACTIVATION_OPTIONS,
  POOLING_OPTIONS,
} from "@/lib/model-lab/architecture";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Switch } from "@/components/ui/switch";

/** Shared chip styling for ToggleGroupItems (overrides base toggle look). */
export const CHIP =
  "chip h-auto min-h-8 min-w-0 pointer-coarse:min-w-11 shrink justify-center rounded-[2px] border border-rule bg-transparent px-2.5 py-1 font-mono text-[11px] font-normal text-ink-2 hover:bg-transparent hover:text-ink data-[state=on]:bg-[color-mix(in_oklab,var(--sig)_14%,transparent)] data-[state=on]:text-ink";

export function Field({
  label,
  value,
  children,
}: {
  label: string;
  /** Optional readout shown on the label row, right-aligned. */
  value?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between gap-2">
        <label className="eyebrow !text-ink-3">{label}</label>
        {value != null && <span className="readout text-ink">{value}</span>}
      </div>
      {children}
    </div>
  );
}

interface LayerConfigProps {
  layer: ConvLayerConfig;
  onUpdate: (updates: Partial<ConvLayerConfig>) => void;
}

export function LayerConfig({ layer, onUpdate }: LayerConfigProps) {
  return (
    <div className="space-y-4 pt-4">
      <Field label="FILTERS">
        <ToggleGroup
          type="single"
          value={String(layer.filters)}
          onValueChange={(v) => v && onUpdate({ filters: Number(v) as typeof layer.filters })}
          spacing={1}
          className="flex flex-wrap gap-1"
        >
          {FILTER_OPTIONS.map((opt) => (
            <ToggleGroupItem key={opt} value={String(opt)} className={CHIP}>
              {opt}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </Field>

      <Field label="KERNEL">
        <ToggleGroup
          type="single"
          value={String(layer.kernelSize)}
          onValueChange={(v) => v && onUpdate({ kernelSize: Number(v) as 3 | 5 | 7 })}
          spacing={1}
          className="flex flex-wrap gap-1"
        >
          {KERNEL_OPTIONS.map((opt) => (
            <ToggleGroupItem key={opt} value={String(opt)} className={CHIP}>
              {opt}×{opt}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </Field>

      <Field label="ACTIVATION">
        <ToggleGroup
          type="single"
          value={layer.activation}
          onValueChange={(v) => v && onUpdate({ activation: v as Activation })}
          spacing={1}
          className="flex flex-wrap gap-1"
        >
          {ACTIVATION_OPTIONS.map((opt) => (
            <ToggleGroupItem key={opt} value={opt} className={CHIP}>
              {opt}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </Field>

      <Field label="POOLING">
        <ToggleGroup
          type="single"
          value={layer.pooling}
          onValueChange={(v) => v && onUpdate({ pooling: v as PoolingType })}
          spacing={1}
          className="flex flex-wrap gap-1"
        >
          {POOLING_OPTIONS.map((opt) => (
            <ToggleGroupItem key={opt} value={opt} className={CHIP}>
              {opt}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </Field>

      <label className="flex cursor-pointer items-center gap-2.5">
        <Switch
          size="sm"
          checked={layer.batchNorm}
          onCheckedChange={(checked) => onUpdate({ batchNorm: checked })}
        />
        <span className="font-sans text-[13px] text-ink-2">Batch normalization</span>
      </label>
    </div>
  );
}
