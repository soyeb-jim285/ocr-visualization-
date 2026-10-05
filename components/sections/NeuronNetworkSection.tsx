"use client";

import {
  useRef,
  useState,
  useCallback,
  useMemo,
  useEffect,
  useLayoutEffect,
} from "react";
import { motion, useDragControls, useScroll, useTransform, type PanInfo } from "framer-motion";
import { DrawingCanvas } from "@/components/canvas/DrawingCanvas";
import { NeuronNetworkCanvas } from "@/components/canvas/NeuronNetworkCanvas";
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { ChartContainer } from "@/components/ui/chart";
import { HeroHeader } from "@/components/ui/HeroHeaderPreview";
import { Bar, BarChart, XAxis, YAxis, LabelList } from "recharts";
import { useInferenceStore } from "@/stores/inferenceStore";
import { useUIStore } from "@/stores/uiStore";
import { EMNIST_CLASSES, BYMERGE_MERGED_INDICES } from "@/lib/model/classes";
import {
  LAYERS,
  extractActivations,
  getOutputLabels,
  displayToActualIndex,
  viridis,
  clamp,
  type NeuronLayerDef,
  type HoveredNeuron,
} from "@/lib/network/networkConstants";
import { useSharedCanvas } from "@/hooks/useSharedCanvas";
import { PHOSPHOR, BG_INSET } from "@/lib/theme";

// ---------------------------------------------------------------------------
// NeuronHeatmapTooltipContent
// ---------------------------------------------------------------------------

function NeuronHeatmapTooltipContent({
  neuron, layerActivations, inputTensor, outputLabels, prediction,
}: {
  neuron: HoveredNeuron;
  layerActivations: Record<string, number[][][] | number[]>;
  inputTensor: number[][] | null;
  outputLabels: string[];
  prediction: number[] | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const layer = LAYERS[neuron.layerIdx];
  const actualIdx = displayToActualIndex(neuron.layerIdx, neuron.neuronIdx);

  const isConv3D = layer.type === "conv" || layer.type === "relu" || layer.type === "pool";
  const isDense = layer.type === "dense" || (layer.type === "relu" && layer.name === "relu4");
  const isInput = layer.type === "input";
  const isOutput = layer.type === "output";
  const canvasSize = isInput ? 112 : (isConv3D ? 112 : 80);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const w = canvas.width, h = canvas.height;
    ctx.fillStyle = BG_INSET;
    ctx.fillRect(0, 0, w, h);

    if (isInput && inputTensor) {
      const cellW = w / 28, cellH = h / 28;
      for (let r = 0; r < 28; r++) for (let c = 0; c < 28; c++) {
        const gray = Math.round(inputTensor[r][c] * 255);
        ctx.fillStyle = `rgb(${gray},${gray},${gray})`;
        ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5);
      }
      const patchCols = 5, patchRows = 4;
      const pc = neuron.neuronIdx % patchCols, pr = Math.floor(neuron.neuronIdx / patchCols);
      const c0 = Math.floor(pc * 28 / patchCols), c1 = Math.floor((pc + 1) * 28 / patchCols);
      const r0 = Math.floor(pr * 28 / patchRows), r1 = Math.floor((pr + 1) * 28 / patchRows);
      const px = c0 * cellW, py = r0 * cellH;
      const pw = (c1 - c0) * cellW, ph = (r1 - r0) * cellH;
      ctx.strokeStyle = PHOSPHOR; ctx.lineWidth = 2; ctx.strokeRect(px, py, pw, ph);
      ctx.fillStyle = "rgba(0,0,0,0.5)";
      ctx.fillRect(0, 0, w, py); ctx.fillRect(0, py + ph, w, h - py - ph);
      ctx.fillRect(0, py, px, ph); ctx.fillRect(px + pw, py, w - px - pw, ph);
    } else if (isConv3D && layer.name !== "relu4") {
      const acts = layerActivations[layer.name];
      if (acts && Array.isArray(acts[0]) && Array.isArray((acts[0] as number[][])[0])) {
        const acts3d = acts as number[][][];
        if (actualIdx < acts3d.length) {
          const ch = acts3d[actualIdx];
          const rows = ch.length, cols = ch[0].length;
          let minVal = Infinity, maxVal = -Infinity;
          for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) { if (ch[r][c] < minVal) minVal = ch[r][c]; if (ch[r][c] > maxVal) maxVal = ch[r][c]; }
          const range = maxVal - minVal;
          const cellW = w / cols, cellH = h / rows;
          for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
            const [cr, cg, cb] = viridis(range > 0 ? (ch[r][c] - minVal) / range : 0);
            ctx.fillStyle = `rgb(${cr},${cg},${cb})`;
            ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5);
          }
        }
      }
    } else if (isDense) {
      const acts = layerActivations[layer.name];
      if (acts && !Array.isArray(acts[0])) {
        const vals = acts as number[];
        if (actualIdx < vals.length) {
          let minVal = Infinity, maxVal = -Infinity;
          for (const val of vals) { if (val < minVal) minVal = val; if (val > maxVal) maxVal = val; }
          const range = Math.max(maxVal - minVal, 0.001);
          const norm = (vals[actualIdx] - minVal) / range;
          const [cr, cg, cb] = viridis(norm);
          ctx.fillStyle = `rgb(${cr},${cg},${cb})`;
          ctx.fillRect(4, h / 2 - 10, norm * (w - 8), 20);
          ctx.strokeStyle = "rgba(255,255,255,0.1)"; ctx.strokeRect(4, h / 2 - 10, w - 8, 20);
        }
      }
    } else if (isOutput && prediction) {
      const valid: { val: number; idx: number }[] = [];
      for (let i = 0; i < prediction.length; i++) if (!BYMERGE_MERGED_INDICES.has(i)) valid.push({ val: prediction[i], idx: i });
      valid.sort((a, b) => b.val - a.val);
      if (neuron.neuronIdx < valid.length) {
        const d = valid[neuron.neuronIdx];
        const [cr, cg, cb] = viridis(d.val);
        ctx.fillStyle = `rgb(${cr},${cg},${cb})`;
        ctx.fillRect(4, h / 2 - 10, d.val * (w - 8), 20);
        ctx.strokeStyle = "rgba(255,255,255,0.1)"; ctx.strokeRect(4, h / 2 - 10, w - 8, 20);
      }
    }
  }, [neuron, layerActivations, inputTensor, prediction, layer, actualIdx, isConv3D, isDense, isInput, isOutput, canvasSize]);

  let label = "";
  if (isInput) { const pc = neuron.neuronIdx % 5, pr = Math.floor(neuron.neuronIdx / 5); label = `Patch [${pr},${pc}]`; }
  else if (isConv3D && layer.name !== "relu4") label = `Channel ${actualIdx}`;
  else if (isDense) label = `Neuron ${actualIdx}`;
  else if (isOutput) label = outputLabels[neuron.neuronIdx] ? `Class "${outputLabels[neuron.neuronIdx]}"` : `Output ${neuron.neuronIdx}`;

  let valueText = "";
  if (isInput && inputTensor) {
    const patchCols = 5, patchRows = 4;
    const pc = neuron.neuronIdx % patchCols, pr = Math.floor(neuron.neuronIdx / patchCols);
    const r0 = Math.floor(pr * 28 / patchRows), r1 = Math.floor((pr + 1) * 28 / patchRows);
    const c0 = Math.floor(pc * 28 / patchCols), c1 = Math.floor((pc + 1) * 28 / patchCols);
    let sum = 0, count = 0;
    for (let r = r0; r < r1; r++) for (let c = c0; c < c1; c++) { sum += inputTensor[r]?.[c] ?? 0; count++; }
    valueText = `mean: ${(sum / count).toFixed(3)}`;
  } else if ((isConv3D && layer.name !== "relu4") || isDense) {
    const acts = layerActivations[layer.name];
    if (acts) {
      if (Array.isArray(acts[0])) {
        const acts3d = acts as number[][][];
        if (actualIdx < acts3d.length) {
          let sum = 0, count = 0;
          for (const row of acts3d[actualIdx]) for (const v of row) { sum += Math.abs(v); count++; }
          valueText = `mean |act|: ${(sum / count).toFixed(4)}`;
        }
      } else {
        const vals = acts as number[];
        if (actualIdx < vals.length) valueText = `value: ${vals[actualIdx].toFixed(4)}`;
      }
    }
  } else if (isOutput && prediction) {
    const valid: { val: number; idx: number }[] = [];
    for (let i = 0; i < prediction.length; i++) if (!BYMERGE_MERGED_INDICES.has(i)) valid.push({ val: prediction[i], idx: i });
    valid.sort((a, b) => b.val - a.val);
    if (neuron.neuronIdx < valid.length) valueText = `confidence: ${(valid[neuron.neuronIdx].val * 100).toFixed(2)}%`;
  }

  return (
    <div className="flex flex-col gap-1.5">
      <div className="font-mono text-[11px] font-medium" style={{ color: layer.color }}>{layer.displayName} · {label}</div>
      <canvas ref={canvasRef} width={canvasSize} height={isDense || isOutput ? 40 : canvasSize}
        className="rounded-[2px] border border-rule"
        style={{ width: canvasSize, height: isDense || isOutput ? 40 : canvasSize, imageRendering: (isInput || isConv3D) ? "pixelated" : "auto", display: "block" }}
      />
      {valueText && <div className="font-mono text-[11px] text-ink-2">{valueText}</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// LayerTooltip
// ---------------------------------------------------------------------------

function LayerTooltipContent({ layer, activationMap }: { layer: NeuronLayerDef; activationMap: Map<string, number[]> }) {
  const acts = activationMap.get(layer.name);
  const meanAct = acts ? acts.reduce((s, v) => s + v, 0) / acts.length : 0;
  return (
    <div className="flex items-center gap-3 whitespace-nowrap">
      <div className="size-2 rotate-45" style={{ background: layer.color }} />
      <div>
        <div className="font-serif text-[15px] text-ink">{layer.displayName}</div>
        <div className="font-mono text-[11px] text-ink-3">{layer.description}</div>
      </div>
      <div className="font-mono text-[11px] text-ink-3">
        {layer.totalNeurons.toLocaleString()} {(layer.type === "conv" || (layer.type === "relu" && layer.name !== "relu4") || layer.type === "pool") ? "ch" : (layer.type === "input" ? "px" : "n")}
      </div>
      {acts && (
        <div className="font-mono text-[11px]">
          <span className="text-ink-3">avg </span>
          <span style={{ color: layer.color }}>{(meanAct * 100).toFixed(1)}%</span>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// InspectorPanel (shadcn Dialog)
// ---------------------------------------------------------------------------

function InspectorPanel({
  layer, activations, inputTensor, prediction, topPrediction, initialChannel, open, onClose,
}: {
  layer: NeuronLayerDef | null; activations: number[][][] | number[] | null; inputTensor: number[][] | null;
  prediction: number[] | null; topPrediction: { classIndex: number; confidence: number } | null;
  initialChannel: number; open: boolean; onClose: () => void;
}) {
  const [selectedChannel, setSelectedChannel] = useState(initialChannel);
  const mainCanvasRef = useRef<HTMLCanvasElement>(null);

  // Snapshot data so content persists during close animation (derived during render, not a ref)
  type Snap = {
    layer: NeuronLayerDef; activations: typeof activations; prediction: typeof prediction;
    topPrediction: typeof topPrediction; channelCount: number;
  };
  const [snapshot, setSnapshot] = useState<Snap | null>(null);
  if (layer && (snapshot?.layer !== layer || snapshot.activations !== activations
    || snapshot.prediction !== prediction || snapshot.topPrediction !== topPrediction)) {
    setSnapshot({
      layer, activations, prediction, topPrediction,
      channelCount: activations && Array.isArray(activations[0]) ? (activations as number[][][]).length : 0,
    });
  }

  // Re-sync channel when (re)opened or retargeted
  const [syncKey, setSyncKey] = useState(`${open}:${initialChannel}`);
  if (syncKey !== `${open}:${initialChannel}`) {
    setSyncKey(`${open}:${initialChannel}`);
    if (open) setSelectedChannel(initialChannel);
  }

  useEffect(() => {
    if (!layer || !open) return;
    // Delay one frame so Radix Dialog Portal has mounted the canvas element
    const raf = requestAnimationFrame(() => {
      const canvas = mainCanvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const w = canvas.width, h = canvas.height;
      ctx.fillStyle = BG_INSET; ctx.fillRect(0, 0, w, h);

      if (layer.type === "input" && inputTensor) {
        const cellW = w / 28, cellH = h / 28;
        for (let r = 0; r < 28; r++) for (let c = 0; c < 28; c++) {
          const gray = Math.round(inputTensor[r][c] * 255);
          ctx.fillStyle = `rgb(${gray},${gray},${gray})`;
          ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5);
        }
      } else if (layer.type === "output" && prediction) {
        const sorted = prediction.map((v, i) => ({ v, i })).filter(d => !BYMERGE_MERGED_INDICES.has(d.i)).sort((a, b) => b.v - a.v);
        const barH = h / Math.min(sorted.length, 20);
        ctx.font = "11px ui-monospace,monospace";
        for (let j = 0; j < Math.min(sorted.length, 20); j++) {
          const d = sorted[j]; const barW = (d.v / Math.max(sorted[0].v, 0.001)) * (w - 60);
          const [cr, cg, cb] = viridis(d.v / Math.max(sorted[0].v, 0.001));
          ctx.fillStyle = `rgb(${cr},${cg},${cb})`; ctx.fillRect(40, j * barH + 2, barW, barH - 4);
          ctx.fillStyle = "#e9eef2"; ctx.textAlign = "right"; ctx.fillText(EMNIST_CLASSES[d.i], 35, j * barH + barH / 2 + 4);
          ctx.textAlign = "left"; ctx.fillText(`${(d.v * 100).toFixed(1)}%`, 40 + barW + 4, j * barH + barH / 2 + 4);
        }
      } else if (activations && Array.isArray(activations[0])) {
        const acts = activations as number[][][];
        if (selectedChannel < acts.length) {
          const ch = acts[selectedChannel]; const rows = ch.length, cols = ch[0].length;
          let minVal = Infinity, maxVal = -Infinity;
          for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) { if (ch[r][c] < minVal) minVal = ch[r][c]; if (ch[r][c] > maxVal) maxVal = ch[r][c]; }
          const range = maxVal - minVal;
          const cellW = w / cols, cellH = h / rows;
          for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
            const [cr, cg, cb] = viridis(range > 0 ? (ch[r][c] - minVal) / range : 0);
            ctx.fillStyle = `rgb(${cr},${cg},${cb})`; ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5);
          }
        }
      } else if (activations && !Array.isArray(activations[0])) {
        const vals = activations as number[]; const n = vals.length;
        const cols = Math.ceil(Math.sqrt(n)), rows = Math.ceil(n / cols);
        const cellW = w / cols, cellH = h / rows;
        let minVal = Infinity, maxVal = -Infinity; for (const v of vals) { if (v < minVal) minVal = v; if (v > maxVal) maxVal = v; }
        const range = Math.max(maxVal - minVal, 0.001);
        for (let i = 0; i < n; i++) {
          const [cr, cg, cb] = viridis((vals[i] - minVal) / range);
          ctx.fillStyle = `rgb(${cr},${cg},${cb})`;
          ctx.fillRect((i % cols) * cellW + 0.5, Math.floor(i / cols) * cellH + 0.5, cellW - 1, cellH - 1);
        }
      } else {
        ctx.fillStyle = "#8896a3"; ctx.font = "italic 15px Georgia,serif";
        ctx.textAlign = "center"; ctx.fillText("Draw a character to see activations", w / 2, h / 2);
      }
    });
    return () => cancelAnimationFrame(raf);
  }, [layer, activations, inputTensor, selectedChannel, prediction, open]);

  const stats = useMemo(() => {
    if (!snapshot?.activations) return null;
    if (Array.isArray(snapshot.activations[0])) {
      const acts = snapshot.activations as number[][][];
      let min = Infinity, max = -Infinity, sum = 0, count = 0, activeCount = 0;
      for (const ch of acts) for (const row of ch) for (const v of row) {
        if (v < min) min = v; if (v > max) max = v; sum += v; count++; if (v > 0) activeCount++;
      }
      return { min, max, mean: sum / count, activePercent: (activeCount / count) * 100 };
    } else {
      const vals = snapshot.activations as number[];
      let min = Infinity, max = -Infinity, sum = 0, activeCount = 0;
      for (const v of vals) { if (v < min) min = v; if (v > max) max = v; sum += v; if (v > 0) activeCount++; }
      return { min, max, mean: sum / vals.length, activePercent: (activeCount / vals.length) * 100 };
    }
  }, [snapshot]);

  const outputChartData = useMemo(() => {
    if (!snapshot?.prediction || snapshot.layer.type !== "output") return [];
    return snapshot.prediction
      .map((v, i) => ({ char: EMNIST_CLASSES[i], confidence: +(v * 100).toFixed(1), idx: i }))
      .filter(d => !BYMERGE_MERGED_INDICES.has(d.idx))
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 15);
  }, [snapshot]);

  if (!snapshot) return null;
  const dl = snapshot.layer;
  const unitLabel = (dl.type === "conv" || (dl.type === "relu" && dl.name !== "relu4") || dl.type === "pool") ? "channels" : (dl.type === "input" ? "pixels" : "neurons");

  return (
    <Dialog open={open} onOpenChange={(o) => { if (!o) onClose(); }}>
      <DialogContent
        className="max-h-[88svh] gap-4 overflow-y-auto p-0 max-sm:bottom-0 max-sm:inset-x-0 max-sm:w-auto sm:max-w-[900px]"
        style={{ borderColor: `${dl.color}66` }}
        onOpenAutoFocus={(e) => { e.preventDefault(); (e.currentTarget as HTMLElement).focus(); }}
      >
        <DialogHeader className="px-5 pt-6 pb-0 sm:px-6">
          <p className="font-mono text-[11px] tracking-[0.08em]" style={{ color: dl.color }}>LAYER {String(inspectedLayerNumber(dl.name)).padStart(2, "0")}</p>
          <DialogTitle className="flex flex-wrap items-baseline gap-x-3 font-serif text-2xl font-normal sm:text-3xl">
            {dl.displayName}
            <span className="font-mono text-xs font-normal text-ink-3">
              {dl.totalNeurons.toLocaleString()} {unitLabel}
            </span>
          </DialogTitle>
          <DialogDescription className={dl.description === `${dl.totalNeurons} neurons` ? "sr-only" : undefined}>{dl.description}</DialogDescription>
        </DialogHeader>

        {stats && (
          <div className="mx-5 grid grid-cols-2 gap-x-5 gap-y-1 border-y border-rule py-2.5 font-mono text-xs text-ink-3 sm:mx-6 sm:flex sm:flex-wrap">
            <span>MIN <span className="text-ink">{stats.min.toFixed(3)}</span></span>
            <span>MAX <span className="text-ink">{stats.max.toFixed(3)}</span></span>
            <span>MEAN <span className="text-ink">{stats.mean.toFixed(3)}</span></span>
            <span>ACTIVE <span className="text-accent-positive">{stats.activePercent.toFixed(1)}%</span></span>
          </div>
        )}

        {dl.type === "output" && snapshot.topPrediction && (
          <div
            className="mx-5 flex items-center gap-5 border-y py-3 sm:mx-6"
            style={{ borderColor: `${dl.color}55` }}
          >
            <span className="font-serif text-6xl leading-none" style={{ color: dl.color }}>
              {EMNIST_CLASSES[snapshot.topPrediction.classIndex]}
            </span>
            <div>
              <div className="text-sm text-ink-2">
                Predicted <strong className="font-medium text-ink">{EMNIST_CLASSES[snapshot.topPrediction.classIndex]}</strong>
              </div>
              <div className="font-mono text-xs text-ink-3">
                CONFIDENCE {(snapshot.topPrediction.confidence * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        )}

        {dl.type === "output" && outputChartData.length > 0 ? (
          <div className="px-5 pb-6 sm:px-6">
            <ChartContainer
              config={{ confidence: { label: "Confidence", color: dl.color } }}
              className="aspect-auto w-full [&_.recharts-cartesian-axis-tick_text]:fill-foreground [&_.recharts-label]:fill-muted-foreground"
              style={{ height: outputChartData.length * 28 + 16 }}
            >
              <BarChart data={outputChartData} layout="vertical" margin={{ left: -10, right: 50, top: 0, bottom: 0 }}>
                <YAxis
                  type="category"
                  dataKey="char"
                  width={30}
                  axisLine={false}
                  tickLine={false}
                  style={{ fontSize: 14, fontFamily: "var(--font-geist-mono)" }}
                />
                <XAxis type="number" hide domain={[0, 100]} />
                <Bar dataKey="confidence" fill="var(--color-confidence)" radius={[0, 2, 2, 0]}>
                  <LabelList
                    dataKey="confidence"
                    position="right"
                    formatter={(v: number) => `${v}%`}
                    style={{ fontSize: 12 }}
                  />
                </Bar>
              </BarChart>
            </ChartContainer>
          </div>
        ) : (
          <div className="flex flex-col gap-4 px-5 pb-6 sm:flex-row sm:px-6">
            <canvas
              ref={mainCanvasRef}
              width={snapshot.channelCount > 0 ? 350 : 500}
              height={snapshot.channelCount > 0 ? 350 : 300}
              className="mx-auto max-w-[240px] shrink-0 rounded-[2px] border border-rule sm:mx-0 sm:max-w-full"
              style={{
                width: snapshot.channelCount > 0 ? "min(350px, 100%)" : "100%",
                height: snapshot.channelCount > 0 ? "auto" : 300,
                aspectRatio: snapshot.channelCount > 0 ? "1 / 1" : undefined,
                imageRendering: dl.type === "input" ? "pixelated" : "auto",
              }}
            />
            {snapshot.channelCount > 0 && (
              <div className="min-w-0 flex-1">
                <p className="mb-2 font-mono text-[11px] text-ink-3">
                  <span className="sm:hidden">Tap</span><span className="max-sm:hidden">Click</span> a channel to inspect
                </p>
                <div className="grid grid-cols-6 gap-1.5 sm:flex sm:max-h-[340px] sm:flex-wrap sm:gap-1 sm:overflow-y-auto">
                  {Array.from({ length: snapshot.channelCount }, (_, i) => (
                    <ChannelThumb
                      key={i} chIdx={i}
                      activations={snapshot.activations as number[][][]}
                      selected={i === selectedChannel}
                      color={dl.color}
                      onClick={() => setSelectedChannel(i)}
                    />
                  ))}
                </div>
                <p className="mt-2 font-mono text-[11px] text-ink-3">
                  CHANNEL {selectedChannel}
                </p>
              </div>
            )}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}

function inspectedLayerNumber(name: string) {
  return LAYERS.findIndex((l) => l.name === name);
}

function ChannelThumb({ chIdx, activations, selected, color, onClick }: {
  chIdx: number; activations: number[][][]; selected: boolean; color: string; onClick: () => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const el = canvasRef.current;
    if (!el || chIdx >= activations.length) return;
    const ctx = el.getContext("2d");
    if (!ctx) return;
    const ch = activations[chIdx]; const rows = ch.length, cols = ch[0].length;
    let minVal = Infinity, maxVal = -Infinity;
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) { if (ch[r][c] < minVal) minVal = ch[r][c]; if (ch[r][c] > maxVal) maxVal = ch[r][c]; }
    const range = maxVal - minVal;
    const cellW = 40 / cols, cellH = 40 / rows;
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const [cr, cg, cb] = viridis(range > 0 ? (ch[r][c] - minVal) / range : 0);
      ctx.fillStyle = `rgb(${cr},${cg},${cb})`; ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5);
    }
  }, [chIdx, activations]);
  return (
    <canvas
      ref={canvasRef} width={40} height={40} onClick={onClick}
      className="aspect-square w-full cursor-pointer rounded-[2px] [image-rendering:pixelated] sm:size-10"
      style={{ outline: selected ? `1px solid ${color}` : "none", outlineOffset: 2 }}
    />
  );
}

// ---------------------------------------------------------------------------
// NeuronNetworkSection — main exported component
// ---------------------------------------------------------------------------

// Layout viewport. On phones, ignore height-only changes <120px (Safari toolbar collapse) so the
// network and floating card do not re-layout while scrolling. sb = safe-area-inset-bottom.
let sv = { w: 0, h: 0, sb: 0 };
function readViewport() {
  const w = document.documentElement.clientWidth, h = window.innerHeight;
  if (!(w < 640 && sv.w === w && Math.abs(h - sv.h) < 120)) {
    const probe = document.createElement("div");
    probe.style.cssText = "position:fixed;visibility:hidden;padding-bottom:env(safe-area-inset-bottom)";
    document.body.appendChild(probe);
    const sb = parseFloat(getComputedStyle(probe).paddingBottom) || 0;
    probe.remove();
    sv = { w, h, sb };
  }
  return sv;
}

export function NeuronNetworkSection() {
  const dragControls = useDragControls();
  const sharedPixels = useSharedCanvas();
  const inputTensor = useInferenceStore(s => s.inputTensor);
  const layerActivations = useInferenceStore(s => s.layerActivations);
  const prediction = useInferenceStore(s => s.prediction);
  const topPrediction = useInferenceStore(s => s.topPrediction);
  const inferenceTimeMs = useInferenceStore(s => s.inferenceTimeMs);
  const isInferring = useInferenceStore(s => s.isInferring);
  const heroStage = useUIStore(s => s.heroStage);
  const heroOffscreen = useUIStore(s => s.activeSection !== 0);
  const setHeroStage = useUIStore(s => s.setHeroStage);

  const [inspectedLayerIdx, setInspectedLayerIdx] = useState<number | null>(null);
  const [inspectedNeuronIdx, setInspectedNeuronIdx] = useState<number | null>(null);
  const [viewport, setViewport] = useState({ w: 0, h: 0, sb: 0 });
  const [customFloatingPos, setCustomFloatingPos] = useState<{ x: number; y: number } | null>(null);

  const isDrawingStage = heroStage === "drawing";
  const isShrinkingStage = heroStage === "shrinking";
  const isRevealedStage = heroStage === "revealed";
  const shouldUseFloatingLayout = !isDrawingStage;

  useEffect(() => {
    const updateViewport = () => {
      setViewport(readViewport());
    };

    updateViewport();
    window.addEventListener("resize", updateViewport);
    return () => window.removeEventListener("resize", updateViewport);
  }, []);

  // Hover state refs are used by canvas drawing loops.
  const hoveredLayerRef = useRef<number | null>(null);
  const hoveredNeuronRef = useRef<HoveredNeuron | null>(null);
  const [hoveredLayer, setHoveredLayer] = useState<number | null>(null);
  const [hoveredNeuron, setHoveredNeuron] = useState<HoveredNeuron | null>(null);
  const [hoveredNeuronTooltipPos, setHoveredNeuronTooltipPos] = useState({ left: 0, top: 0 });
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasContainerRef = useRef<HTMLDivElement>(null);
  const [containerSize, setContainerSize] = useState({ w: 1200, h: 500 });

  // Wave progress — ref only, no state, updated in RAF
  const waveRef = useRef(0);
  const waveTargetRef = useRef(0);
  const startWaveRef = useRef<() => void>(() => {});
  const hasData = Object.keys(layerActivations).length > 0;

  useEffect(() => {
    if (hasData && isDrawingStage) {
      setHeroStage("shrinking");
    }
  }, [hasData, isDrawingStage, setHeroStage]);

  const isMobile = viewport.w > 0 && viewport.w < 640;
  const stacked = viewport.w < 1024; // copy above the card instead of beside it

  const [copyBottom, setCopyBottom] = useState(0);
  const copyRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = copyRef.current;
    if (!el) return;
    const update = () => setCopyBottom(Math.round(el.getBoundingClientRect().bottom + window.scrollY));
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Phones: card spans the 16px gutters; canvas shrinks to keep the whole card (chrome ~90px) in the first screen
  const mobileChrome = 90;
  const expandedCanvasSize = useMemo(() => {
    if (!viewport.w) return 300;
    return viewport.w < 640
      ? clamp(Math.min(viewport.w - 34, viewport.h - copyBottom - 16 - mobileChrome - 24 - viewport.sb), 200, 340)
      : 328;
  }, [viewport.w, viewport.h, viewport.sb, copyBottom]);
  const floatingCanvasSize = isMobile ? 140 : 128;

  const heroPad = isMobile ? 0 : 16;
  const expandedCardWidth = isMobile ? viewport.w - 32 : expandedCanvasSize + heroPad * 2 + 2;
  const expandedCardHeight = expandedCanvasSize + heroPad * 2 + (isMobile ? mobileChrome : 112);
  const floatingCardWidth = floatingCanvasSize + 18;
  const floatingCardHeight = floatingCanvasSize + 104;
  const CHIP = isMobile ? 44 : 40;
  const bottomGap = Math.max(16, viewport.sb) + 8; // keeps the chip/sheet above Safari's toolbar zone

  // Chip: once past the hero (or on phones) the floating card collapses so it never covers content
  const [chipOpen, setChipOpen] = useState(false);
  const collapsed = isRevealedStage && !chipOpen && (heroOffscreen || isMobile);

  // Content box of SectionWrapper-style container (max 1200, px 16/64/32)
  const pad = viewport.w >= 1400 ? 32 : viewport.w >= 768 ? 64 : 16;
  const boxW = Math.min(viewport.w, 1200);
  const contentLeft = (viewport.w - boxW) / 2 + pad;
  const contentW = boxW - pad * 2;

  const { scrollY } = useScroll();
  const followScrollRef = useRef(false);
  useEffect(() => { followScrollRef.current = isDrawingStage && stacked; }, [isDrawingStage, stacked]);
  const heroY = useTransform(scrollY, (v) => (followScrollRef.current ? -v : 0));

  const stageHeight = viewport.h
    ? stacked
      ? Math.max(viewport.h, copyBottom + 28 + expandedCardHeight + 40)
      : Math.max(viewport.h, 720)
    : 760;

  const expandedX = !viewport.w
    ? 16
    : stacked
      ? contentLeft
      : contentLeft + contentW - expandedCardWidth;
  const expandedY = !viewport.h
    ? 120
    : stacked
      ? copyBottom + (isMobile ? 16 : 28)
      : Math.max(72, (viewport.h - expandedCardHeight) / 2);

  const maxFloatingX = Math.max(12, viewport.w - floatingCardWidth - 12);
  const maxFloatingY = Math.max(12, viewport.h - floatingCardHeight - 12);

  // Bottom-left on desktop (keeps the output column clear); bottom-right on phones
  const defaultFloatingPos = useMemo(
    () => ({
      x: isMobile ? Math.max(12, viewport.w - floatingCardWidth - 16) : Math.max(12, contentLeft),
      y: Math.max(64, viewport.h - floatingCardHeight - bottomGap),
    }),
    [viewport.w, viewport.h, floatingCardWidth, floatingCardHeight, isMobile, bottomGap, contentLeft]
  );
  const chipPos = { x: Math.max(12, viewport.w - CHIP - 16), y: Math.max(12, viewport.h - CHIP - bottomGap) };

  const floatingPos = customFloatingPos
    ? {
        x: clamp(customFloatingPos.x, 12, maxFloatingX),
        y: clamp(customFloatingPos.y, 12, maxFloatingY),
      }
    : defaultFloatingPos;

  // Delay the shrink so multi-stroke glyphs (A, 4, i) can be finished in place
  const shrinkTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const handleFirstDraw = useCallback(() => {
    if (!isDrawingStage) return;
    if (shrinkTimer.current) clearTimeout(shrinkTimer.current);
    shrinkTimer.current = setTimeout(() => setHeroStage("shrinking"), 700);
  }, [isDrawingStage, setHeroStage]);
  useEffect(() => () => { if (shrinkTimer.current) clearTimeout(shrinkTimer.current); }, []);

  const handleDragEnd = useCallback(
    (_event: MouseEvent | TouchEvent | PointerEvent, info: PanInfo) => {
      setCustomFloatingPos({
        x: clamp(floatingPos.x + info.offset.x, 12, maxFloatingX),
        y: clamp(floatingPos.y + info.offset.y, 12, maxFloatingY),
      });
    },
    [floatingPos.x, floatingPos.y, maxFloatingX, maxFloatingY]
  );

  const handleCanvasTransitionComplete = useCallback(() => {
    if (isShrinkingStage) {
      setHeroStage("revealed");
    }
  }, [isShrinkingStage, setHeroStage]);

  // Reset wave + hero stage on clear (hasData→false) or new inference
  const prevHasDataRef = useRef(false);
  const pendingWaveRef = useRef(false);
  useEffect(() => {
    if (!hasData) {
      waveRef.current = 0;
      waveTargetRef.current = 0;
      pendingWaveRef.current = false;
      if (prevHasDataRef.current) {
        setHeroStage("drawing");
      }
    } else if (isRevealedStage) {
      // Already revealed — start wave immediately
      waveRef.current = 0;
      waveTargetRef.current = LAYERS.length + 1;
      startWaveRef.current();
    } else {
      // Not revealed yet — defer until section appears
      pendingWaveRef.current = true;
    }
    prevHasDataRef.current = hasData;
  }, [hasData, layerActivations, isRevealedStage, setHeroStage]);

  // Start pending wave once revealed stage begins
  useEffect(() => {
    if (isRevealedStage && pendingWaveRef.current) {
      pendingWaveRef.current = false;
      waveRef.current = 0;
      waveTargetRef.current = LAYERS.length + 1;
      startWaveRef.current();
    }
  }, [isRevealedStage]);

  // Wave animation via RAF — no React state updates
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      raf = 0;
      const target = waveTargetRef.current;
      const current = waveRef.current;
      if (Math.abs(target - current) <= 0.01) return; // settled; startWave restarts
      waveRef.current = target > current ? current + (target - current) * 0.035 : 0;
      raf = requestAnimationFrame(tick);
    };
    startWaveRef.current = () => { if (!raf) raf = requestAnimationFrame(tick); };
    startWaveRef.current();
    return () => cancelAnimationFrame(raf);
  }, []);

  // Measure — outer fills remaining viewport, canvas fills its flex area
  useEffect(() => {
    const measure = () => {
      if (containerRef.current) {
        const top = containerRef.current.getBoundingClientRect().top + window.scrollY;
        const h = Math.max(400, readViewport().h - top - (readViewport().w < 640 ? 64 : 48));
        containerRef.current.style.height = `${h}px`;
      }
      if (canvasContainerRef.current) {
        const rect = canvasContainerRef.current.getBoundingClientRect();
        setContainerSize({ w: Math.round(rect.width), h: Math.round(rect.height) });
      }
    };
    measure();
    // Re-measure after a tick so flex layout has settled
    requestAnimationFrame(measure);
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, []);

  // Activation data as refs (canvas reads these directly)
  const activationMapRef = useRef<Map<string, number[]>>(new Map());
  const outputLabelsRef = useRef<string[]>([]);

  const activationMap = useMemo(() => extractActivations(layerActivations, inputTensor, prediction), [layerActivations, inputTensor, prediction]);
  useLayoutEffect(() => {
    activationMapRef.current = activationMap;
  }, [activationMap]);

  const outputLabels = useMemo(() => getOutputLabels(prediction), [prediction]);
  useLayoutEffect(() => {
    outputLabelsRef.current = outputLabels;
  }, [outputLabels]);

  const onHoverLayer = useCallback((li: number | null) => {
    if (hoveredLayerRef.current !== li) {
      hoveredLayerRef.current = li;
      setHoveredLayer(li);
    }
  }, []);

  const onHoverNeuron = useCallback((n: HoveredNeuron | null) => {
    const prev = hoveredNeuronRef.current;
    if (prev?.layerIdx !== n?.layerIdx || prev?.neuronIdx !== n?.neuronIdx) {
      hoveredNeuronRef.current = n;
      setHoveredNeuron(n);
      if (n) {
        const rect = canvasContainerRef.current?.getBoundingClientRect();
        setHoveredNeuronTooltipPos({
          left: n.screenX - (rect?.left ?? 0),
          top: n.screenY - (rect?.top ?? 0),
        });
      }
    }
  }, []);

  const onClickLayer = useCallback((li: number, ni: number | null) => {
    setInspectedLayerIdx(li);
    setInspectedNeuronIdx(ni);
    // touch: the emulated mouse-move leaves the tooltip over the sheet
    hoveredNeuronRef.current = null;
    setHoveredNeuron(null);
    hoveredLayerRef.current = null;
    setHoveredLayer(null);
  }, []);

  const getActivation = useCallback(
    (name: string) => name === "input" ? null : layerActivations[name] ?? null,
    [layerActivations],
  );

  const inspectedLayer = inspectedLayerIdx !== null ? LAYERS[inspectedLayerIdx] : null;

  const topChar = topPrediction ? EMNIST_CLASSES[topPrediction.classIndex] : null;

  return (
    <motion.section
      id="neuron-network"
      className="relative select-none overflow-hidden px-0 sm:px-3 md:px-5"
      initial={false}
      animate={{
        minHeight: isDrawingStage ? stageHeight : 0,
        paddingTop: isDrawingStage ? 56 : (isMobile ? 72 : 56),
        paddingBottom: isDrawingStage ? 36 : 24,
      }}
      transition={{ duration: 0.75, ease: [0.22, 1, 0.36, 1] }}
    >
      {/* Projector beam behind the drawing card (static, drawing stage only) */}
      <div
        aria-hidden
        className={`pointer-events-none absolute inset-0 hidden transition-opacity duration-700 lg:block ${isDrawingStage ? "opacity-100" : "opacity-0"}`}
        style={{ background: "radial-gradient(60% 50% at 70% 45%, rgba(143,227,255,0.10), transparent 70%)" }}
      />

      {/* Hero copy: fades up and out once the first stroke lands */}
      <motion.div
        initial={false}
        animate={{ opacity: isDrawingStage ? 1 : 0, y: isDrawingStage ? 0 : -12 }}
        transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
        aria-hidden={!isDrawingStage}
        className={`absolute inset-x-0 top-0 z-10 ${stacked ? "pt-[72px] sm:pt-[88px]" : "flex items-center"} ${isDrawingStage ? "" : "pointer-events-none"}`}
        style={stacked ? undefined : { height: stageHeight }}
      >
        <div className="mx-auto w-full max-w-[1200px] px-4 md:px-16 min-[1400px]:px-8">
          <div ref={copyRef} style={stacked ? undefined : { maxWidth: Math.max(320, contentW - expandedCardWidth - 48) }}>
            <HeroHeader />
          </div>
        </div>
      </motion.div>

      {/* Revealed-state plate title, to the right of the index rail */}
      <p
        className={`pointer-events-none absolute left-24 top-5 z-10 hidden font-mono text-[11px] tracking-[0.12em] text-ink-3 transition-opacity duration-700 md:block ${isRevealedStage ? "opacity-100" : "opacity-0"}`}
      >
        NEURAL NETWORK X-RAY <span className="ml-3 text-phosphor">PLATE 0</span>
      </p>

      <div
        className={`relative flex w-full items-stretch overflow-hidden ${
          isRevealedStage ? "pointer-events-auto" : "pointer-events-none"
        }`}
        ref={containerRef}
      >
        <div
          className={`relative min-w-0 flex-1 transition-opacity duration-700 motion-reduce:transition-none ${isDrawingStage ? "opacity-0" : "lg:pr-[168px]"}`}
          ref={canvasContainerRef}
        >
          <NeuronNetworkCanvas
            width={containerSize.w}
            height={containerSize.h}
            activationMapRef={activationMapRef}
            outputLabelsRef={outputLabelsRef}
            hoveredLayerRef={hoveredLayerRef}
            hoveredNeuronRef={hoveredNeuronRef}
            waveRef={waveRef}
            paused={!isRevealedStage}
            onHoverLayer={onHoverLayer}
            onHoverNeuron={onHoverNeuron}
            onClickLayer={onClickLayer}
          />

          {isRevealedStage && (
            <>
              {/* Neuron tooltip */}
              <Tooltip open={!!hoveredNeuron && hasData}>
                <TooltipTrigger asChild>
                  <div
                    className="pointer-events-none absolute h-px w-px"
                    style={{
                      left: hoveredNeuronTooltipPos.left,
                      top: hoveredNeuronTooltipPos.top,
                    }}
                  />
                </TooltipTrigger>
                <TooltipContent side="right" sideOffset={12} className="p-2.5 [&>svg]:hidden">
                  {hoveredNeuron && hasData && (
                    <NeuronHeatmapTooltipContent
                      neuron={hoveredNeuron}
                      layerActivations={layerActivations}
                      inputTensor={inputTensor}
                      outputLabels={outputLabels}
                      prediction={prediction}
                    />
                  )}
                </TooltipContent>
              </Tooltip>

              {/* Layer tooltip */}
              <Tooltip open={hoveredLayer !== null && !hoveredNeuron}>
                <TooltipTrigger asChild>
                  <div className="pointer-events-none absolute bottom-4 left-1/2 h-px w-px" />
                </TooltipTrigger>
                <TooltipContent side="top" sideOffset={8} className="p-2.5 [&>svg]:hidden">
                  {hoveredLayer !== null && !hoveredNeuron && (
                    <LayerTooltipContent layer={LAYERS[hoveredLayer]} activationMap={activationMap} />
                  )}
                </TooltipContent>
              </Tooltip>
            </>
          )}
        </div>

        <InspectorPanel
          layer={inspectedLayer}
          activations={inspectedLayer ? getActivation(inspectedLayer.name) : null}
          inputTensor={inputTensor}
          prediction={prediction}
          topPrediction={topPrediction}
          initialChannel={inspectedNeuronIdx !== null && inspectedLayerIdx !== null ? displayToActualIndex(inspectedLayerIdx, inspectedNeuronIdx) : 0}
          open={isRevealedStage && inspectedLayerIdx !== null}
          onClose={() => { setInspectedLayerIdx(null); setInspectedNeuronIdx(null); }}
        />
      </div>

      {/* FIG. 1 caption under the network */}
      <p
        className={`select-none px-4 pt-3 font-mono text-[11px] leading-[1.5] tracking-[0.04em] text-ink-3 transition-opacity duration-700 md:px-3 md:text-center ${isRevealedStage ? "opacity-100" : "opacity-0"}`}
      >
        <b className="font-medium text-phosphor">FIG. 1</b>
        {" — "}
        {topChar && !isMobile ? <>Activations for the specimen &ldquo;{topChar}&rdquo;, </> : isMobile ? "" : "Activations, "}
        {isMobile ? "Tap a neuron to inspect." : <>13 layers, 146-way output. Hover or tap a neuron to inspect.</>}
      </p>

      {/* Phone chip sheet: tap outside to collapse */}
      {isMobile && isRevealedStage && chipOpen && (
        <div className="fixed inset-0 z-40" onClick={() => setChipOpen(false)} aria-hidden />
      )}

      {/* Outer layer pins to the viewport; in the drawing stage on stacked layouts it scrolls with the hero copy */}
      <motion.div style={{ y: heroY }} className="fixed left-0 top-0 z-50">
      <motion.div
        drag={isRevealedStage && !collapsed && !isMobile}
        dragControls={dragControls}
        dragListener={false}
        dragElastic={0.08}
        dragMomentum={false}
        dragConstraints={{ left: 12, top: 12, right: maxFloatingX, bottom: maxFloatingY }}
        onDragEnd={handleDragEnd}
        onAnimationComplete={handleCanvasTransitionComplete}
        initial={false}
        animate={{
          ...(collapsed
            ? { x: chipPos.x, y: chipPos.y, width: CHIP }
            : shouldUseFloatingLayout
              ? { x: floatingPos.x, y: floatingPos.y, width: floatingCardWidth }
              : { x: expandedX, y: expandedY, width: expandedCardWidth }),
          // hide the big hero card once scrolled past the hero before first stroke
          opacity: isDrawingStage && heroOffscreen ? 0 : collapsed ? 0.6 : 1,
          pointerEvents: isDrawingStage && heroOffscreen ? "none" : "auto",
        }}
        transition={{ type: "spring", stiffness: 260, damping: 28, mass: 0.6 }}
        className="absolute left-0 top-0"
      >
        {collapsed && (
          <button
            type="button"
            onClick={() => setChipOpen(true)}
            className="relative flex size-10 select-none items-center justify-center rounded-[4px] border border-rule-strong bg-bg-raised max-sm:size-11"
            aria-label="Open drawing canvas"
          >
            <span className="font-serif text-xl leading-none text-annotation">{topChar ?? "?"}</span>
          </button>
        )}

        <div
          className={`plate-marks border ${collapsed ? "hidden" : ""} ${
            shouldUseFloatingLayout
              ? "rounded-[4px] border-rule-strong bg-bg-raised p-2"
              : "well rounded-[4px] border-rule-strong max-sm:border-0 max-sm:bg-transparent max-sm:p-0 max-sm:shadow-none sm:p-4"
          }`}
          style={{ "--pm-c": "var(--phosphor)" } as React.CSSProperties}
        >
          {shouldUseFloatingLayout && (
            <div className="mb-1.5 flex items-center justify-between">
              <button
                type="button"
                onPointerDown={(event) => dragControls.start(event)}
                className="flex cursor-grab touch-none items-center gap-2 py-1 active:cursor-grabbing"
                aria-label="Move floating canvas"
              >
                <span aria-hidden className="grid grid-cols-2 gap-[3px]">
                  {Array.from({ length: 6 }, (_, i) => (
                    <span key={i} className="size-[3px] rounded-full bg-ink-4" />
                  ))}
                </span>
                <span className="font-mono text-[11px] tracking-[0.08em] text-ink-2">SPECIMEN</span>
              </button>
              {(heroOffscreen || isMobile) && (
                <button type="button" onClick={() => setChipOpen(false)} className="text-btn -my-3 inline-flex size-11 items-center justify-center" aria-label="Collapse canvas">
                  –
                </button>
              )}
            </div>
          )}

          <DrawingCanvas
            variant={shouldUseFloatingLayout ? "floating" : "hero"}
            displaySize={shouldUseFloatingLayout ? floatingCanvasSize : expandedCanvasSize}
            onFirstDraw={handleFirstDraw}
            sharedPixels={sharedPixels}
          />

          {shouldUseFloatingLayout && (
            <p
              className="mt-1.5 flex items-center gap-1.5 font-mono text-[11px] text-ink-2"
              style={{ visibility: inferenceTimeMs !== null ? "visible" : "hidden" }}
            >
              <span
                aria-hidden
                className={`inline-block size-1.5 rounded-full bg-annotation transition-opacity duration-300 ${isInferring ? "opacity-100" : "opacity-40"}`}
              />
              INFER {inferenceTimeMs !== null ? (inferenceTimeMs < 1 ? "<1" : Math.round(inferenceTimeMs)) : "0"} ms
            </p>
          )}
        </div>
      </motion.div>
      </motion.div>
    </motion.section>
  );
}
