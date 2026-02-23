"use client";

import {
  useRef,
  useState,
  useCallback,
  useMemo,
  useEffect,
} from "react";
import { motion, useDragControls, type PanInfo } from "framer-motion";
import { DrawingCanvas } from "@/components/canvas/DrawingCanvas";
import { ImageUploader } from "@/components/canvas/ImageUploader";
import { NeuronNetworkCanvas } from "@/components/canvas/NeuronNetworkCanvas";
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { ChartContainer } from "@/components/ui/chart";
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
    ctx.fillStyle = "#111118";
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
      ctx.strokeStyle = "#6366f1"; ctx.lineWidth = 2; ctx.strokeRect(px, py, pw, ph);
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
          const range = Math.max(maxVal - minVal, 0.001);
          const cellW = w / cols, cellH = h / rows;
          for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
            const [cr, cg, cb] = viridis((ch[r][c] - minVal) / range);
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
      <div className="text-[11px] font-semibold" style={{ color: layer.color }}>{layer.displayName} — {label}</div>
      <canvas ref={canvasRef} width={canvasSize} height={isDense || isOutput ? 40 : canvasSize}
        className="rounded-md"
        style={{ width: canvasSize, height: isDense || isOutput ? 40 : canvasSize, imageRendering: (isInput || isConv3D) ? "pixelated" : "auto", display: "block" }}
      />
      {valueText && <div className="font-mono text-[10px] text-foreground/50">{valueText}</div>}
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
      <div className="h-2.5 w-2.5 rounded-full" style={{ background: layer.color }} />
      <div>
        <div className="text-[13px] font-semibold text-foreground">{layer.displayName}</div>
        <div className="text-[11px] text-foreground/40">{layer.description}</div>
      </div>
      <div className="font-mono text-[11px] text-foreground/30">
        {layer.totalNeurons.toLocaleString()} {(layer.type === "conv" || (layer.type === "relu" && layer.name !== "relu4") || layer.type === "pool") ? "ch" : (layer.type === "input" ? "px" : "n")}
      </div>
      {acts && (
        <div className="font-mono text-[11px]">
          <span className="text-foreground/30">avg: </span>
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

  // Snapshot data so content persists during close animation
  const snapRef = useRef<{
    layer: NeuronLayerDef; activations: typeof activations; prediction: typeof prediction;
    topPrediction: typeof topPrediction; channelCount: number;
  } | null>(null);
  if (layer) {
    snapRef.current = {
      layer, activations, prediction, topPrediction,
      channelCount: activations && Array.isArray(activations[0]) ? (activations as number[][][]).length : 0,
    };
  }
  const snap = snapRef.current;

  useEffect(() => { if (open) setSelectedChannel(initialChannel); }, [open, initialChannel]);

  useEffect(() => {
    if (!layer || !open) return;
    // Delay one frame so Radix Dialog Portal has mounted the canvas element
    const raf = requestAnimationFrame(() => {
      const canvas = mainCanvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const w = canvas.width, h = canvas.height;
      ctx.fillStyle = "#111118"; ctx.fillRect(0, 0, w, h);

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
        ctx.font = "11px system-ui,sans-serif";
        for (let j = 0; j < Math.min(sorted.length, 20); j++) {
          const d = sorted[j]; const barW = (d.v / Math.max(sorted[0].v, 0.001)) * (w - 60);
          const [cr, cg, cb] = viridis(d.v / Math.max(sorted[0].v, 0.001));
          ctx.fillStyle = `rgb(${cr},${cg},${cb})`; ctx.fillRect(40, j * barH + 2, barW, barH - 4);
          ctx.fillStyle = "#e8e8ed"; ctx.textAlign = "right"; ctx.fillText(EMNIST_CLASSES[d.i], 35, j * barH + barH / 2 + 4);
          ctx.textAlign = "left"; ctx.fillText(`${(d.v * 100).toFixed(1)}%`, 40 + barW + 4, j * barH + barH / 2 + 4);
        }
      } else if (activations && Array.isArray(activations[0])) {
        const acts = activations as number[][][];
        if (selectedChannel < acts.length) {
          const ch = acts[selectedChannel]; const rows = ch.length, cols = ch[0].length;
          let minVal = Infinity, maxVal = -Infinity;
          for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) { if (ch[r][c] < minVal) minVal = ch[r][c]; if (ch[r][c] > maxVal) maxVal = ch[r][c]; }
          const range = Math.max(maxVal - minVal, 0.001);
          const cellW = w / cols, cellH = h / rows;
          for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
            const [cr, cg, cb] = viridis((ch[r][c] - minVal) / range);
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
        ctx.fillStyle = "rgba(255,255,255,0.15)"; ctx.font = "14px system-ui,sans-serif";
        ctx.textAlign = "center"; ctx.fillText("Draw a character to see activations", w / 2, h / 2);
      }
    });
    return () => cancelAnimationFrame(raf);
  }, [layer, activations, inputTensor, selectedChannel, prediction, open]);

  const stats = useMemo(() => {
    if (!snap?.activations) return null;
    if (Array.isArray(snap.activations[0])) {
      const acts = snap.activations as number[][][];
      let min = Infinity, max = -Infinity, sum = 0, count = 0, activeCount = 0;
      for (const ch of acts) for (const row of ch) for (const v of row) {
        if (v < min) min = v; if (v > max) max = v; sum += v; count++; if (v > 0) activeCount++;
      }
      return { min, max, mean: sum / count, activePercent: (activeCount / count) * 100 };
    } else {
      const vals = snap.activations as number[];
      let min = Infinity, max = -Infinity, sum = 0, activeCount = 0;
      for (const v of vals) { if (v < min) min = v; if (v > max) max = v; sum += v; if (v > 0) activeCount++; }
      return { min, max, mean: sum / vals.length, activePercent: (activeCount / vals.length) * 100 };
    }
  }, [snap?.activations]);

  const outputChartData = useMemo(() => {
    if (!snap?.prediction || snap.layer.type !== "output") return [];
    return snap.prediction
      .map((v, i) => ({ char: EMNIST_CLASSES[i], confidence: +(v * 100).toFixed(1), idx: i }))
      .filter(d => !BYMERGE_MERGED_INDICES.has(d.idx))
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 15);
  }, [snap?.prediction, snap?.layer.type]);

  if (!snap) return null;
  const dl = snap.layer;
  const unitLabel = (dl.type === "conv" || (dl.type === "relu" && dl.name !== "relu4") || dl.type === "pool") ? "channels" : (dl.type === "input" ? "pixels" : "neurons");

  return (
    <Dialog open={open} onOpenChange={(o) => { if (!o) onClose(); }}>
      <DialogContent
        className="max-h-[90vh] gap-3 overflow-y-auto p-0 sm:max-w-[900px]"
        style={{ borderColor: `${dl.color}40` }}
      >
        <DialogHeader className="px-6 pt-6 pb-0">
          <DialogTitle className="flex items-center gap-2.5 text-xl">
            <span className="inline-block h-3 w-3 shrink-0 rounded-full" style={{ background: dl.color }} />
            {dl.displayName}
            <span className="font-mono text-sm font-normal text-foreground/35">
              {dl.totalNeurons.toLocaleString()} {unitLabel}
            </span>
          </DialogTitle>
          <DialogDescription>{dl.description}</DialogDescription>
        </DialogHeader>

        {stats && (
          <div className="mx-6 flex flex-wrap gap-x-5 gap-y-1 rounded-lg bg-foreground/[0.03] px-3 py-2 font-mono text-xs text-foreground/50">
            <span>Min: <span className="text-foreground">{stats.min.toFixed(3)}</span></span>
            <span>Max: <span className="text-foreground">{stats.max.toFixed(3)}</span></span>
            <span>Mean: <span className="text-foreground">{stats.mean.toFixed(3)}</span></span>
            <span>Active: <span className="text-green-500">{stats.activePercent.toFixed(1)}%</span></span>
          </div>
        )}

        {dl.type === "output" && snap.topPrediction && (
          <div
            className="mx-6 flex items-center gap-4 rounded-lg border px-4 py-3"
            style={{ background: `${dl.color}15`, borderColor: `${dl.color}30` }}
          >
            <span className="text-4xl font-bold" style={{ color: dl.color }}>
              {EMNIST_CLASSES[snap.topPrediction.classIndex]}
            </span>
            <div>
              <div className="text-sm text-foreground">
                Predicted: <strong>{EMNIST_CLASSES[snap.topPrediction.classIndex]}</strong>
              </div>
              <div className="text-[13px] text-foreground/50">
                Confidence: {(snap.topPrediction.confidence * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        )}

        {dl.type === "output" && outputChartData.length > 0 ? (
          <div className="px-6 pb-6">
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
                <Bar dataKey="confidence" fill="var(--color-confidence)" radius={[0, 4, 4, 0]}>
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
          <div className="flex gap-4 px-6 pb-6">
            <canvas
              ref={mainCanvasRef}
              width={snap.channelCount > 0 ? 350 : 500}
              height={snap.channelCount > 0 ? 350 : 300}
              className="shrink-0 rounded-lg"
              style={{
                width: snap.channelCount > 0 ? 350 : "100%",
                height: snap.channelCount > 0 ? 350 : 300,
                imageRendering: dl.type === "input" ? "pixelated" : "auto",
              }}
            />
            {snap.channelCount > 0 && (
              <div className="min-w-0 flex-1">
                <p className="mb-2 text-xs text-foreground/40">
                  {snap.channelCount} channels — click to inspect
                </p>
                <div className="flex max-h-[340px] flex-wrap gap-1 overflow-y-auto">
                  {Array.from({ length: snap.channelCount }, (_, i) => (
                    <ChannelThumb
                      key={i} chIdx={i}
                      activations={snap.activations as number[][][]}
                      selected={i === selectedChannel}
                      color={dl.color}
                      onClick={() => setSelectedChannel(i)}
                    />
                  ))}
                </div>
                <p className="mt-1.5 font-mono text-[11px] text-foreground/30">
                  Channel {selectedChannel}
                </p>
              </div>
            )}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
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
    const range = Math.max(maxVal - minVal, 0.001);
    const cellW = 40 / cols, cellH = 40 / rows;
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const [cr, cg, cb] = viridis((ch[r][c] - minVal) / range);
      ctx.fillStyle = `rgb(${cr},${cg},${cb})`; ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5);
    }
  }, [chIdx, activations]);
  return (
    <canvas
      ref={canvasRef} width={40} height={40} onClick={onClick}
      className="h-10 w-10 cursor-pointer rounded [image-rendering:pixelated]"
      style={{ border: selected ? `2px solid ${color}` : "2px solid transparent" }}
    />
  );
}

// ---------------------------------------------------------------------------
// NeuronNetworkSection — main exported component
// ---------------------------------------------------------------------------

export function NeuronNetworkSection() {
  const dragControls = useDragControls();
  const inputTensor = useInferenceStore(s => s.inputTensor);
  const layerActivations = useInferenceStore(s => s.layerActivations);
  const prediction = useInferenceStore(s => s.prediction);
  const topPrediction = useInferenceStore(s => s.topPrediction);
  const heroStage = useUIStore(s => s.heroStage);
  const setHeroStage = useUIStore(s => s.setHeroStage);

  const [inspectedLayerIdx, setInspectedLayerIdx] = useState<number | null>(null);
  const [inspectedNeuronIdx, setInspectedNeuronIdx] = useState<number | null>(null);
  const [viewport, setViewport] = useState({ w: 0, h: 0 });
  const [customFloatingPos, setCustomFloatingPos] = useState<{ x: number; y: number } | null>(null);

  const isDrawingStage = heroStage === "drawing";
  const isShrinkingStage = heroStage === "shrinking";
  const isRevealedStage = heroStage === "revealed";
  const shouldUseFloatingLayout = !isDrawingStage;

  useEffect(() => {
    const updateViewport = () => {
      setViewport({ w: window.innerWidth, h: window.innerHeight });
    };

    updateViewport();
    window.addEventListener("resize", updateViewport);
    return () => window.removeEventListener("resize", updateViewport);
  }, []);

  // Hover state as refs — avoids re-render on every mouse move
  const [, forceRender] = useState(0);
  const hoveredLayerRef = useRef<number | null>(null);
  const hoveredNeuronRef = useRef<HoveredNeuron | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasContainerRef = useRef<HTMLDivElement>(null);
  const [containerSize, setContainerSize] = useState({ w: 1200, h: 500 });

  // Wave progress — ref only, no state, updated in RAF
  const waveRef = useRef(0);
  const waveTargetRef = useRef(0);
  const hasData = Object.keys(layerActivations).length > 0;

  useEffect(() => {
    if (hasData && isDrawingStage) {
      setHeroStage("shrinking");
    }
  }, [hasData, isDrawingStage, setHeroStage]);

  const expandedCanvasSize = useMemo(() => {
    if (!viewport.w) return 320;
    return clamp(Math.round(viewport.w * 0.55), 200, 360);
  }, [viewport.w]);

  const floatingCanvasSize = useMemo(() => {
    if (!viewport.w) return 108;
    return clamp(Math.round(viewport.w * 0.23), 92, 126);
  }, [viewport.w]);

  const expandedCardWidth = expandedCanvasSize + (viewport.w < 640 ? 24 : 56);
  const floatingCardWidth = floatingCanvasSize + 10;
  const floatingCardHeight = floatingCanvasSize + 40;

  const expandedX = viewport.w ? (viewport.w - expandedCardWidth) / 2 : 16;
  const expandedY = viewport.h ? Math.max(viewport.w < 640 ? 80 : 100, viewport.h * (viewport.w < 640 ? 0.18 : 0.22)) : 120;

  const maxFloatingX = Math.max(12, viewport.w - floatingCardWidth - 12);
  const maxFloatingY = Math.max(12, viewport.h - floatingCardHeight - 12);

  const defaultFloatingPos = useMemo(
    () => ({
      x: 16,
      y: 16,
    }),
    [viewport.w, viewport.h, floatingCardWidth, maxFloatingX, maxFloatingY]
  );

  const floatingPos = customFloatingPos
    ? {
        x: clamp(customFloatingPos.x, 12, maxFloatingX),
        y: clamp(customFloatingPos.y, 12, maxFloatingY),
      }
    : defaultFloatingPos;

  const handleFirstDraw = useCallback(() => {
    if (isDrawingStage) {
      setHeroStage("shrinking");
    }
  }, [isDrawingStage, setHeroStage]);

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

  // Reset wave on clear (hasData→false) or new inference — only animate after revealed
  const prevHasDataRef = useRef(false);
  const pendingWaveRef = useRef(false);
  useEffect(() => {
    if (!hasData) {
      waveRef.current = 0;
      waveTargetRef.current = 0;
      pendingWaveRef.current = false;
    } else if (isRevealedStage) {
      // Already revealed — start wave immediately
      waveRef.current = 0;
      waveTargetRef.current = LAYERS.length + 1;
    } else {
      // Not revealed yet — defer until section appears
      pendingWaveRef.current = true;
    }
    prevHasDataRef.current = hasData;
  }, [hasData, layerActivations, isRevealedStage]);

  // Start pending wave once revealed stage begins
  useEffect(() => {
    if (isRevealedStage && pendingWaveRef.current) {
      pendingWaveRef.current = false;
      waveRef.current = 0;
      waveTargetRef.current = LAYERS.length + 1;
    }
  }, [isRevealedStage]);

  // Wave animation via RAF — no React state updates
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const target = waveTargetRef.current;
      const current = waveRef.current;
      if (Math.abs(target - current) > 0.01) {
        waveRef.current = target > current ? current + (target - current) * 0.035 : 0;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  // Measure — outer fills remaining viewport, canvas fills its flex area
  useEffect(() => {
    const measure = () => {
      if (containerRef.current) {
        const top = containerRef.current.getBoundingClientRect().top + window.scrollY;
        const h = Math.max(400, window.innerHeight - top);
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
  activationMapRef.current = activationMap;

  const outputLabels = useMemo(() => getOutputLabels(prediction), [prediction]);
  outputLabelsRef.current = outputLabels;

  const onHoverLayer = useCallback((li: number | null) => {
    if (hoveredLayerRef.current !== li) {
      hoveredLayerRef.current = li;
      // Only re-render for tooltip display, not for canvas (canvas reads ref)
      forceRender(n => n + 1);
    }
  }, []);

  const onHoverNeuron = useCallback((n: HoveredNeuron | null) => {
    const prev = hoveredNeuronRef.current;
    if (prev?.layerIdx !== n?.layerIdx || prev?.neuronIdx !== n?.neuronIdx) {
      hoveredNeuronRef.current = n;
      forceRender(nn => nn + 1);
    }
  }, []);

  const onClickLayer = useCallback((li: number, ni: number | null) => {
    setInspectedLayerIdx(li);
    setInspectedNeuronIdx(ni);
  }, []);

  const getActivation = useCallback(
    (name: string) => name === "input" ? null : layerActivations[name] ?? null,
    [layerActivations],
  );

  const inspectedLayer = inspectedLayerIdx !== null ? LAYERS[inspectedLayerIdx] : null;
  const hoveredLayer = hoveredLayerRef.current;
  const hoveredNeuron = hoveredNeuronRef.current;
  const stageHeight = viewport.h ? Math.max(viewport.h, 760) : 760;

  return (
    <motion.section
      id="neuron-network"
      className="relative overflow-hidden px-1 sm:px-3 md:px-5"
      initial={false}
      animate={{
        minHeight: isDrawingStage ? stageHeight : Math.max(400, Math.min(560, viewport.h * 0.75)),
        paddingTop: isDrawingStage ? 56 : (viewport.w < 640 ? 12 : 48),
        paddingBottom: isDrawingStage ? 36 : (viewport.w < 640 ? 8 : 20),
      }}
      transition={{ duration: 0.75, ease: [0.22, 1, 0.36, 1] }}
    >
      {/* Hero text */}
      <motion.div
        className="pointer-events-none absolute inset-x-0 top-0 z-10 flex flex-col items-center px-4"
        style={{ paddingTop: Math.max(24, expandedY * 0.22) }}
        initial={{ opacity: 0 }}
        animate={{ opacity: isDrawingStage ? 1 : 0 }}
        transition={{ duration: 0.6, ease: "easeOut" }}
      >
        <motion.p
          className="mb-3 font-mono text-[11px] uppercase tracking-[0.25em] text-foreground/30"
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: isDrawingStage ? 1 : 0, y: isDrawingStage ? 0 : -8 }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          Interactive CNN Visualization
        </motion.p>

        <motion.h1
          className="mb-3 text-center text-3xl font-semibold tracking-tight text-foreground sm:text-4xl lg:text-5xl"
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: isDrawingStage ? 1 : 0, y: isDrawingStage ? 0 : -10 }}
          transition={{ duration: 0.7, delay: 0.2 }}
        >
          Neural Network X-Ray
        </motion.h1>

        <motion.p
          className="max-w-sm text-center text-sm leading-relaxed text-foreground/40 sm:text-base"
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: isDrawingStage ? 1 : 0, y: isDrawingStage ? 0 : -6 }}
          transition={{ duration: 0.7, delay: 0.35 }}
        >
          Draw a character below to see every layer process it in real time
        </motion.p>
      </motion.div>

      {/* Supported characters hint */}
      <motion.p
        className="pointer-events-none absolute inset-x-0 z-10 text-center font-mono text-[11px] tracking-wider text-foreground/15"
        style={{ top: expandedY + expandedCanvasSize + 170 }}
        initial={{ opacity: 0 }}
        animate={{ opacity: isDrawingStage ? 1 : 0 }}
        transition={{ duration: 0.5, delay: 0.7 }}
      >
        A–Z &middot; a–z &middot; 0–9
      </motion.p>

      <motion.div
        initial={false}
        animate={
          isRevealedStage
            ? { opacity: 1, y: 0, scale: 1 }
            : { opacity: 0, y: 30, scale: 0.98 }
        }
        transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
        className={`relative flex w-full items-stretch overflow-hidden ${
          isRevealedStage ? "pointer-events-auto" : "pointer-events-none"
        }`}
        ref={containerRef}
      >
        <div className="relative min-w-0 flex-1" ref={canvasContainerRef}>
          {isRevealedStage ? (
            <>
              <NeuronNetworkCanvas
                width={containerSize.w}
                height={containerSize.h}
                activationMapRef={activationMapRef}
                outputLabelsRef={outputLabelsRef}
                hoveredLayerRef={hoveredLayerRef}
                hoveredNeuronRef={hoveredNeuronRef}
                waveRef={waveRef}
                onHoverLayer={onHoverLayer}
                onHoverNeuron={onHoverNeuron}
                onClickLayer={onClickLayer}
              />

              {/* Neuron tooltip */}
              <Tooltip open={!!hoveredNeuron && hasData}>
                <TooltipTrigger asChild>
                  <div
                    className="pointer-events-none absolute h-px w-px"
                    style={{
                      left: (hoveredNeuron?.screenX ?? 0) - (canvasContainerRef.current?.getBoundingClientRect().left ?? 0),
                      top: (hoveredNeuron?.screenY ?? 0) - (canvasContainerRef.current?.getBoundingClientRect().top ?? 0),
                    }}
                  />
                </TooltipTrigger>
                <TooltipContent side="right" sideOffset={12} className="border-border/50 bg-surface p-2.5 shadow-xl backdrop-blur-xl [&>svg]:hidden">
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
                <TooltipContent side="top" sideOffset={8} className="border-border/50 bg-surface p-2.5 shadow-xl backdrop-blur-xl [&>svg]:hidden">
                  {hoveredLayer !== null && !hoveredNeuron && (
                    <LayerTooltipContent layer={LAYERS[hoveredLayer]} activationMap={activationMap} />
                  )}
                </TooltipContent>
              </Tooltip>
            </>
          ) : (
            <div className="flex h-full items-center justify-center text-foreground/35">
              <p className="text-xs uppercase tracking-[0.24em]">
                Neural map appears after canvas shrinks
              </p>
            </div>
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
      </motion.div>

      <motion.div
        drag={isRevealedStage}
        dragControls={dragControls}
        dragListener={false}
        dragElastic={0.08}
        dragMomentum={false}
        dragConstraints={{ left: 12, top: 12, right: maxFloatingX, bottom: maxFloatingY }}
        onDragEnd={handleDragEnd}
        onAnimationComplete={handleCanvasTransitionComplete}
        initial={false}
        animate={
          shouldUseFloatingLayout
            ? { x: floatingPos.x, y: floatingPos.y, width: floatingCardWidth }
            : { x: expandedX, y: expandedY, width: expandedCardWidth }
        }
        transition={{ type: "spring", stiffness: 260, damping: 28, mass: 0.6 }}
        className="fixed left-0 top-0 z-50"
      >
        <div
          className={`border ${
            shouldUseFloatingLayout
              ? "rounded-md border-border/50 bg-surface/92 p-1 shadow-lg shadow-black/30 backdrop-blur-xl"
              : "rounded-xl border-border/60 bg-surface p-2 sm:p-4 md:p-5"
          }`}
        >
          {shouldUseFloatingLayout && (
            <button
              type="button"
              onPointerDown={(event) => dragControls.start(event)}
              className="mb-0.5 flex w-full cursor-grab justify-center py-0.5 active:cursor-grabbing"
              aria-label="Move floating canvas"
            >
              <span className="h-0.5 w-5 rounded-full bg-foreground/25" />
            </button>
          )}

          <DrawingCanvas
            variant={shouldUseFloatingLayout ? "floating" : "hero"}
            displaySize={shouldUseFloatingLayout ? floatingCanvasSize : expandedCanvasSize}
            onFirstDraw={handleFirstDraw}
          />

          {isDrawingStage && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.25, duration: 0.5 }}
              className="mt-3"
            >
              <ImageUploader />
            </motion.div>
          )}
        </div>
      </motion.div>
    </motion.section>
  );
}
