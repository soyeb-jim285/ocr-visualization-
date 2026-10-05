"use client";

import { useMemo } from "react";
import { useModelLabStore } from "@/stores/modelLabStore";
import type { TrainingMode } from "@/stores/modelLabStore";
import type { DatasetType } from "@/lib/model-lab/dataLoader";
import type { OptimizerType } from "@/lib/model-lab/trainModel";
import { Slider } from "@/components/ui/slider";
import { Progress } from "@/components/ui/progress";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Field, CHIP } from "./LayerConfig";

function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  const rem = Math.round(s % 60);
  return `${m}m ${rem}s`;
}

const DATASET_TABS: { value: DatasetType; label: string; desc: string }[] = [
  { value: "digits", label: "Digits", desc: "0-9 (10 classes, fastest)" },
  { value: "emnist", label: "EMNIST", desc: "A-Z, a-z, 0-9 (62 classes)" },
  { value: "bangla", label: "Bengali", desc: "84 Bengali character classes" },
  { value: "combined", label: "Combined", desc: "All 146 classes" },
];

const BATCH_OPTIONS = [16, 32, 64, 128] as const;
const OPTIMIZER_OPTIONS: { value: OptimizerType; label: string }[] = [
  { value: "adam", label: "Adam" },
  { value: "sgd", label: "SGD" },
  { value: "rmsprop", label: "RMSProp" },
];

const MODE_OPTIONS: { value: TrainingMode; label: string; desc: string }[] = [
  { value: "browser", label: "Browser", desc: "TF.js, no server" },
  { value: "hf", label: "HF CPU", desc: "HuggingFace Space" },
  { value: "gpu", label: "GPU", desc: "Modal T4" },
];

interface TrainingControlsProps {
  onTrain: () => void;
  onStop: () => void;
  onReset: () => void;
}

export function TrainingControls({
  onTrain,
  onStop,
  onReset,
}: TrainingControlsProps) {
  const {
    trainingMode,
    setTrainingMode,
    gpuStatus,
    maxSamples,
    setMaxSamples,
    datasetType,
    setDatasetType,
    learningRate,
    setLearningRate,
    epochs,
    setEpochs,
    batchSize,
    setBatchSize,
    optimizer,
    setOptimizer,
    phase,
    currentEpoch,
    currentBatch,
    totalBatches,
    trainingHistory,
    validation,
    errorMessage,
  } = useModelLabStore();

  const isTraining = phase === "training";

  const timingInfo = useMemo(() => {
    if (trainingHistory.length === 0) return null;
    const times = trainingHistory
      .map((m) => m.epochTimeMs)
      .filter((t): t is number => t != null);
    if (times.length === 0) return null;
    const avgMs = times.reduce((a, b) => a + b, 0) / times.length;
    const elapsed = times.reduce((a, b) => a + b, 0);
    const remaining = Math.max(0, (epochs - currentEpoch) * avgMs);
    return { avgMs, elapsed, remaining };
  }, [trainingHistory, epochs, currentEpoch]);
  const isBusy = phase === "loading-data" || phase === "building" || phase === "training";
  const hasErrors = validation.errors.length > 0;

  // Log-scale learning rate slider: map [0, 1] → [1e-4, 1e-2]
  const lrToSlider = (lr: number) =>
    (Math.log10(lr) - Math.log10(1e-4)) / (Math.log10(1e-2) - Math.log10(1e-4));
  const sliderToLr = (v: number) =>
    10 ** (v * (Math.log10(1e-2) - Math.log10(1e-4)) + Math.log10(1e-4));

  const chipBig =
    "chip !h-full min-h-[52px] w-full self-stretch min-w-0 flex-col items-start justify-center gap-0.5 rounded-[2px] border border-rule bg-transparent px-2.5 py-1.5 text-left font-normal text-ink-2 hover:bg-transparent hover:text-ink data-[state=on]:bg-[color-mix(in_oklab,var(--sig)_14%,transparent)] data-[state=on]:text-ink";

  const trainLabel =
    phase === "loading-data"
      ? trainingMode !== "browser"
        ? "Connecting…"
        : "Loading data…"
      : phase === "building"
        ? "Building model…"
        : trainingMode === "gpu"
          ? "Train on GPU"
          : trainingMode === "hf"
            ? "Train on HF"
            : "Train";

  return (
    <div className="figure space-y-5">
      <h3 className="font-serif text-xl text-ink">Training</h3>

      <Field label="COMPUTE">
        <ToggleGroup
          type="single"
          value={trainingMode}
          onValueChange={(v) => v && setTrainingMode(v as TrainingMode)}
          disabled={isBusy}
          spacing={1}
          className="grid w-full grid-cols-3 items-stretch gap-2"
        >
          {MODE_OPTIONS.map((mode) => (
            <ToggleGroupItem key={mode.value} value={mode.value} className={chipBig}>
              <span className="text-[12px] font-medium">{mode.label}</span>
              <span className="whitespace-normal text-[11px] leading-snug text-ink-3">{mode.desc}</span>
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </Field>

      <Field label="DATASET">
        <ToggleGroup
          type="single"
          value={datasetType}
          onValueChange={(v) => v && setDatasetType(v as DatasetType)}
          disabled={isBusy}
          spacing={1}
          className="grid w-full grid-cols-2 items-stretch gap-2"
        >
          {DATASET_TABS.map((tab) => (
            <ToggleGroupItem key={tab.value} value={tab.value} className={chipBig}>
              <span className="text-[12px] font-medium">{tab.label}</span>
              <span className="whitespace-normal text-[11px] leading-snug text-ink-3">{tab.desc}</span>
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </Field>

      <div className="grid grid-cols-2 gap-x-6 gap-y-5">
        <Field label="LEARNING RATE" value={learningRate.toExponential(1)}>
          <Slider
            value={[lrToSlider(learningRate)]}
            onValueChange={([v]) => setLearningRate(sliderToLr(v))}
            min={0}
            max={1}
            step={0.01}
            disabled={isBusy}
            className="my-2"
          />
        </Field>

        <Field label="EPOCHS" value={String(epochs).padStart(3, "0")}>
          <Slider
            value={[epochs]}
            onValueChange={([v]) => setEpochs(v)}
            min={1}
            max={50}
            step={1}
            disabled={isBusy}
            className="my-2"
          />
        </Field>

        <Field label="BATCH SIZE">
          <ToggleGroup
            type="single"
            value={String(batchSize)}
            onValueChange={(v) => v && setBatchSize(Number(v) as typeof batchSize)}
            disabled={isBusy}
            spacing={1}
            className="grid w-full grid-cols-4 gap-2"
          >
            {BATCH_OPTIONS.map((bs) => (
              <ToggleGroupItem key={bs} value={String(bs)} className={`${CHIP} px-1`}>
                {bs}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
        </Field>

        <Field label="OPTIMIZER">
          <ToggleGroup
            type="single"
            value={optimizer}
            onValueChange={(v) => v && setOptimizer(v as OptimizerType)}
            disabled={isBusy}
            spacing={1}
            className="grid w-full grid-cols-3 gap-2"
          >
            {OPTIMIZER_OPTIONS.map((opt) => (
              <ToggleGroupItem key={opt.value} value={opt.value} className={`${CHIP} !px-1 text-[10.5px]`}>
                {opt.label}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
        </Field>
      </div>

      {trainingMode !== "browser" && (
        <Field label="TRAINING SAMPLES" value={`${(maxSamples / 1000).toFixed(0)}K samples`}>
          <Slider
            value={[maxSamples]}
            onValueChange={([v]) => setMaxSamples(v)}
            min={5000}
            max={50000}
            step={5000}
            disabled={isBusy}
            className="my-2"
          />
        </Field>
      )}

      {/* Actions */}
      <div className="flex flex-wrap items-center gap-2 border-t border-rule pt-5 max-sm:sticky max-sm:bottom-[max(0.75rem,env(safe-area-inset-bottom))] max-sm:z-20 max-sm:bg-bg/90 max-sm:pb-2 max-sm:backdrop-blur">
        {!isTraining ? (
          <button type="button" onClick={onTrain} disabled={isBusy || hasErrors} className="btn-primary max-sm:flex-1">
            {isBusy && (
              <span className="size-1.5 rounded-full bg-annotation motion-safe:animate-pulse" aria-hidden />
            )}
            {trainLabel}
          </button>
        ) : (
          <button
            type="button"
            onClick={onStop}
            className="btn-ghost !border-annotation/60 !text-annotation hover:!border-annotation"
          >
            Stop
          </button>
        )}

        <button type="button" onClick={onReset} disabled={isBusy} className="btn-ghost">
          Reset
        </button>

        {gpuStatus && trainingMode !== "browser" && (
          <span className="font-mono text-[11px] text-sig">{gpuStatus}</span>
        )}
        {isTraining && !gpuStatus && (
          <span className="readout">
            EPOCH {String(currentEpoch).padStart(3, "0")}/{String(epochs).padStart(3, "0")}
            {trainingMode === "browser" && totalBatches > 0 && (
              <span className="text-ink-3">
                {" · "}BATCH {currentBatch}/{totalBatches}
              </span>
            )}
          </span>
        )}
      </div>

      {hasErrors && !isBusy && (
        <p className="font-mono text-[11px] text-ink-3">Fix the architecture errors above to enable training.</p>
      )}

      {timingInfo && (
        <div className="grid grid-cols-2 gap-px overflow-hidden rounded-[2px] border border-rule bg-rule sm:grid-cols-4">
          {[
            ["AVG / EPOCH", formatDuration(timingInfo.avgMs)],
            ["ELAPSED", formatDuration(timingInfo.elapsed)],
            isTraining
              ? ["REMAINING", timingInfo.remaining > 0 ? `~${formatDuration(timingInfo.remaining)}` : "—"]
              : ["TOTAL", formatDuration(timingInfo.elapsed)],
            ["EST. TOTAL", `~${formatDuration(timingInfo.avgMs * epochs)}`],
          ].map(([k, v]) => (
            <div key={k} className="bg-bg-inset px-3 py-2">
              <p className="font-mono text-[10.5px] tracking-[0.06em] text-ink-3">{k}</p>
              <p className="readout">{v}</p>
            </div>
          ))}
        </div>
      )}

      {isTraining && trainingMode === "browser" && totalBatches > 0 && (
        <Progress
          value={(currentBatch / totalBatches) * 100}
          className="h-0.5 bg-rule [&_[data-slot=progress-indicator]]:bg-phosphor"
        />
      )}
      {isTraining && trainingMode !== "browser" && epochs > 0 && (
        <Progress
          value={(currentEpoch / epochs) * 100}
          className="h-0.5 bg-rule [&_[data-slot=progress-indicator]]:bg-phosphor"
        />
      )}

      {phase === "loading-data" && trainingMode !== "browser" && (
        <p className="caption">Cold start may take up to ~1 minute while the server spins up.</p>
      )}

      {phase === "error" && errorMessage && (
        <p className="border-y border-annotation/40 py-2 font-mono text-[11px] text-annotation" role="alert">
          {errorMessage}
        </p>
      )}
    </div>
  );
}
