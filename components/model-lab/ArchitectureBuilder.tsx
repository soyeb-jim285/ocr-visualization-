"use client";

import { useCallback } from "react";
import { GripVertical, ChevronRight, Plus, Trash2 } from "lucide-react";
import { Accordion as AccordionPrimitive } from "radix-ui";
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  type DragEndEvent,
} from "@dnd-kit/core";
import {
  SortableContext,
  verticalListSortingStrategy,
  useSortable,
  arrayMove,
} from "@dnd-kit/sortable";
import { restrictToVerticalAxis } from "@dnd-kit/modifiers";
import { CSS } from "@dnd-kit/utilities";
import { useModelLabStore } from "@/stores/modelLabStore";
import { LayerConfig, Field, CHIP } from "./LayerConfig";
import { MAX_CONV_LAYERS } from "@/lib/model-lab/architecture";
import type { Activation, ConvLayerConfig } from "@/lib/model-lab/architecture";
import { Accordion, AccordionItem, AccordionContent } from "@/components/ui/accordion";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";

const ACTIVATION_OPTIONS: Activation[] = ["relu", "gelu", "silu", "leakyRelu", "tanh"];

function formatPooling(p: string) {
  if (p === "max") return "MaxPool";
  if (p === "avg") return "AvgPool";
  return "";
}

/* ------------------------------------------------------------------ */
/*  Sortable conv-layer row                                           */
/* ------------------------------------------------------------------ */

function SortableConvLayer({
  layer,
  index,
  total,
  removeConvLayer,
  updateConvLayer,
}: {
  layer: ConvLayerConfig;
  index: number;
  total: number;
  removeConvLayer: (id: string) => void;
  updateConvLayer: (id: string, updates: Partial<ConvLayerConfig>) => void;
}) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: layer.id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    zIndex: isDragging ? 50 : undefined,
    opacity: isDragging ? 0.5 : 1,
  };

  const summary = `Conv ${layer.filters} ${layer.kernelSize}×${layer.kernelSize} ${layer.activation}${layer.pooling !== "none" ? ` ${formatPooling(layer.pooling)}` : ""}${layer.batchNorm ? " BN" : ""}`;

  return (
    <AccordionItem
      ref={setNodeRef}
      style={style}
      value={layer.id}
      className="rounded-[3px] border border-rule bg-bg-inset transition-colors data-[state=open]:border-rule-strong"
    >
      <div className="flex min-h-11 items-center gap-1 px-2">
        <button
          type="button"
          aria-label={`Reorder layer ${index + 1}`}
          className="flex size-7 pointer-coarse:size-11 shrink-0 cursor-grab touch-none items-center justify-center text-ink-4 transition-colors hover:text-ink-2 active:cursor-grabbing"
          {...attributes}
          {...listeners}
        >
          <GripVertical size={14} />
        </button>

        <span className="shrink-0 font-mono text-[11px] text-sig">
          {String(index + 1).padStart(2, "0")}
        </span>

        <AccordionPrimitive.Trigger className="flex min-h-9 pointer-coarse:min-h-11 flex-1 items-center gap-1.5 text-left font-mono text-[11px] text-ink-2 transition-colors hover:text-ink [&[data-state=open]>svg]:rotate-90">
          <ChevronRight
            size={12}
            className="shrink-0 text-ink-3 transition-transform duration-200"
          />
          {summary}
        </AccordionPrimitive.Trigger>

        <Button
          variant="ghost"
          size="icon-xs"
          aria-label={`Remove layer ${index + 1}`}
          onClick={() => removeConvLayer(layer.id)}
          disabled={total <= 1}
          className="text-ink-3 hover:bg-transparent hover:text-annotation pointer-coarse:size-11"
        >
          <Trash2 size={14} />
        </Button>
      </div>

      <AccordionContent className="border-t border-rule px-3 pb-4 pt-0">
        <LayerConfig
          layer={layer}
          onUpdate={(updates) => updateConvLayer(layer.id, updates)}
        />
      </AccordionContent>
    </AccordionItem>
  );
}

/* ------------------------------------------------------------------ */
/*  Main builder                                                       */
/* ------------------------------------------------------------------ */

export function ArchitectureBuilder() {
  const {
    architecture,
    validation,
    addConvLayer,
    removeConvLayer,
    updateConvLayer,
    setConvLayerOrder,
    setDenseConfig,
    setExpandedLayerId,
  } = useModelLabStore();

  const { convLayers, dense } = architecture;
  const { spatialDims, paramCount, errors, warnings } = validation;

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 5 } }),
    useSensor(KeyboardSensor),
  );

  const handleDragEnd = useCallback(
    (event: DragEndEvent) => {
      const { active, over } = event;
      if (!over || active.id === over.id) return;

      const oldIndex = convLayers.findIndex((l) => l.id === active.id);
      const newIndex = convLayers.findIndex((l) => l.id === over.id);
      if (oldIndex === -1 || newIndex === -1) return;

      setConvLayerOrder(arrayMove(convLayers, oldIndex, newIndex));
    },
    [convLayers, setConvLayerOrder],
  );

  const handleAccordionChange = useCallback(
    (value: string) => {
      setExpandedLayerId(value || null);
    },
    [setExpandedLayerId],
  );

  const params =
    paramCount < 1e6
      ? `${(paramCount / 1e3).toFixed(1)}K`
      : `${(paramCount / 1e6).toFixed(1)}M`;

  return (
    <div className="figure space-y-5">
      <div className="flex items-baseline justify-between gap-3">
        <h3 className="font-serif text-xl text-ink">Architecture</h3>
        <span className="readout">
          {params} <span className="text-ink-3">params</span>
        </span>
      </div>

      {/* Spatial dim flow */}
      <div className="flex flex-wrap items-center gap-1 font-mono text-[11px] text-ink-2">
        <span className="rounded-[2px] border border-rule px-1.5 py-0.5">28×28×1</span>
        {spatialDims.slice(1).map((dim, i) => (
          <span key={i} className="flex items-center gap-1">
            <span className="text-ink-4">&rarr;</span>
            <span className="rounded-[2px] border border-rule px-1.5 py-0.5">
              {dim.height}×{dim.width}×{convLayers[i]?.filters ?? "?"}
            </span>
          </span>
        ))}
      </div>

      <DndContext
        sensors={sensors}
        collisionDetection={closestCenter}
        modifiers={[restrictToVerticalAxis]}
        onDragEnd={handleDragEnd}
      >
        <SortableContext
          items={convLayers.map((l) => l.id)}
          strategy={verticalListSortingStrategy}
        >
          <Accordion
            type="single"
            collapsible
            className="space-y-2"
            onValueChange={handleAccordionChange}
          >
            {convLayers.map((layer, i) => (
              <SortableConvLayer
                key={layer.id}
                layer={layer}
                index={i}
                total={convLayers.length}
                removeConvLayer={removeConvLayer}
                updateConvLayer={updateConvLayer}
              />
            ))}

            {convLayers.length < MAX_CONV_LAYERS && (
              <button
                type="button"
                onClick={addConvLayer}
                className="flex min-h-11 w-full items-center justify-center gap-2 rounded-[3px] border border-dashed border-rule-strong font-mono text-[11px] text-ink-2 transition-colors hover:border-sig hover:text-ink"
              >
                <Plus size={14} />
                Add conv layer
              </button>
            )}
          </Accordion>
        </SortableContext>
      </DndContext>

      {/* Dense config */}
      <div className="rounded-[3px] border border-rule bg-bg-inset p-4">
        <h4 className="eyebrow mb-4">DENSE LAYER</h4>
        <div className="grid grid-cols-2 gap-5">
          <Field label="WIDTH" value={dense.width}>
            <Slider
              value={[dense.width]}
              onValueChange={([v]) => setDenseConfig({ width: v })}
              min={64}
              max={1024}
              step={64}
              className="my-2"
            />
          </Field>
          <Field label="DROPOUT" value={dense.dropout.toFixed(2)}>
            <Slider
              value={[dense.dropout]}
              onValueChange={([v]) => setDenseConfig({ dropout: v })}
              min={0}
              max={0.7}
              step={0.05}
              className="my-2"
            />
          </Field>
        </div>
        <div className="mt-4">
          <Field label="ACTIVATION">
            <ToggleGroup
              type="single"
              value={dense.activation}
              onValueChange={(v) => v && setDenseConfig({ activation: v as Activation })}
              spacing={1}
              className="flex flex-wrap gap-1.5"
            >
              {ACTIVATION_OPTIONS.map((act) => (
                <ToggleGroupItem key={act} value={act} className={CHIP}>
                  {act}
                </ToggleGroupItem>
              ))}
            </ToggleGroup>
          </Field>
        </div>
      </div>

      {errors.length > 0 && (
        <div className="space-y-1" role="alert">
          {errors.map((err, i) => (
            <p key={i} className="font-mono text-[11px] text-annotation">
              {err}
            </p>
          ))}
        </div>
      )}
      {warnings.length > 0 && (
        <div className="space-y-2">
          {warnings.map((warn, i) => (
            <p
              key={i}
              className="flex items-start gap-2 rounded-[3px] border border-[var(--accent-warning)]/40 bg-[color-mix(in_oklab,var(--accent-warning)_8%,transparent)] px-3 py-2 font-mono text-[12px] leading-snug text-accent-warning"
            >
              <span className="shrink-0 font-semibold tracking-[0.06em]">WARN</span>
              <span>{warn}</span>
            </p>
          ))}
        </div>
      )}
    </div>
  );
}
