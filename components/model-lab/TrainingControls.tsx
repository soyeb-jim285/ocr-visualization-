"use client";

import { useModelLabStore } from "@/stores/modelLabStore";
import type { TrainingMode } from "@/stores/modelLabStore";
import type { DatasetType } from "@/lib/model-lab/dataLoader";
import type { OptimizerType } from "@/lib/model-lab/trainModel";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Progress } from "@/components/ui/progress";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { cn } from "@/lib/utils";

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
    validation,
    errorMessage,
  } = useModelLabStore();

  const isTraining = phase === "training";
  const isBusy = phase === "loading-data" || phase === "building" || phase === "training";
  const hasErrors = validation.errors.length > 0;

  // Log-scale learning rate slider: map [0, 1] → [1e-4, 1e-2]
  const lrToSlider = (lr: number) =>
    (Math.log10(lr) - Math.log10(1e-4)) / (Math.log10(1e-2) - Math.log10(1e-4));
  const sliderToLr = (v: number) =>
    10 ** (v * (Math.log10(1e-2) - Math.log10(1e-4)) + Math.log10(1e-4));

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold text-foreground/70">Training</h3>

      {/* Mode toggle: Browser / HF CPU / GPU */}
      <div>
        <label className="mb-1.5 block text-[11px] uppercase tracking-wider text-foreground/35">
          Compute
        </label>
        <ToggleGroup
          type="single"
          value={trainingMode}
          onValueChange={(v) => v && setTrainingMode(v as TrainingMode)}
          disabled={isBusy}
          spacing={1}
          className="flex w-full gap-1"
        >
          {MODE_OPTIONS.map((mode) => (
            <ToggleGroupItem
              key={mode.value}
              value={mode.value}
              className={cn(
                "h-auto min-w-0 shrink flex-1 flex-col items-start gap-0 overflow-hidden rounded-md px-2 py-1.5",
                "bg-white/5 text-foreground/40 hover:bg-white/10 hover:text-foreground/40",
                mode.value === "gpu" && "data-[state=on]:bg-emerald-500/15 data-[state=on]:text-emerald-400",
                mode.value === "hf" && "data-[state=on]:bg-purple-500/15 data-[state=on]:text-purple-400",
                mode.value === "browser" && "data-[state=on]:bg-indigo-500/15 data-[state=on]:text-indigo-400",
              )}
            >
              <span className="text-xs font-medium">{mode.label}</span>
              <span className="truncate text-[10px] text-foreground/30">{mode.desc}</span>
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </div>

      {/* Dataset selector */}
      <div>
        <label className="mb-1.5 block text-[11px] uppercase tracking-wider text-foreground/35">
          Dataset
        </label>
        <ToggleGroup
          type="single"
          value={datasetType}
          onValueChange={(v) => v && setDatasetType(v as DatasetType)}
          disabled={isBusy}
          spacing={1}
          className="grid w-full grid-cols-2 gap-1.5 sm:grid-cols-4"
        >
          {DATASET_TABS.map((tab) => (
            <ToggleGroupItem
              key={tab.value}
              value={tab.value}
              className="h-auto min-w-0 shrink flex-col items-start gap-0 overflow-hidden rounded-md px-2 py-1.5 bg-white/5 text-foreground/40 hover:bg-white/10 hover:text-foreground/40 data-[state=on]:bg-indigo-500/15 data-[state=on]:text-indigo-400"
            >
              <span className="text-xs font-medium">{tab.label}</span>
              <span className="truncate self-stretch text-[10px] text-foreground/30">{tab.desc}</span>
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </div>

      {/* Hyperparameters */}
      <div className="space-y-3">
        <div className="grid grid-cols-2 gap-3">
          {/* Learning rate */}
          <div>
            <label className="mb-1 block text-[10px] uppercase tracking-wider text-foreground/30">
              Learning Rate
            </label>
            <Slider
              value={[lrToSlider(learningRate)]}
              onValueChange={([v]) => setLearningRate(sliderToLr(v))}
              min={0}
              max={1}
              step={0.01}
              disabled={isBusy}
              className="my-2 [&_[data-slot=slider-range]]:bg-indigo-500 [&_[data-slot=slider-thumb]]:border-indigo-500 [&_[data-slot=slider-thumb]]:size-3"
            />
            <span className="block text-center font-mono text-[10px] text-foreground/40">
              {learningRate.toExponential(1)}
            </span>
          </div>

          {/* Epochs */}
          <div>
            <label className="mb-1 block text-[10px] uppercase tracking-wider text-foreground/30">
              Epochs
            </label>
            <Slider
              value={[epochs]}
              onValueChange={([v]) => setEpochs(v)}
              min={1}
              max={50}
              step={1}
              disabled={isBusy}
              className="my-2 [&_[data-slot=slider-range]]:bg-indigo-500 [&_[data-slot=slider-thumb]]:border-indigo-500 [&_[data-slot=slider-thumb]]:size-3"
            />
            <span className="block text-center font-mono text-[10px] text-foreground/40">
              {epochs}
            </span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3">
          {/* Batch size */}
          <div>
            <label className="mb-1 block text-[10px] uppercase tracking-wider text-foreground/30">
              Batch Size
            </label>
            <ToggleGroup
              type="single"
              value={String(batchSize)}
              onValueChange={(v) => v && setBatchSize(Number(v) as typeof batchSize)}
              disabled={isBusy}
              spacing={1}
              className="flex w-full gap-1"
            >
              {BATCH_OPTIONS.map((bs) => (
                <ToggleGroupItem
                  key={bs}
                  value={String(bs)}
                  className="h-auto min-w-0 shrink flex-1 rounded-md px-1 py-1 text-[11px] font-medium bg-white/5 text-foreground/40 hover:bg-white/10 hover:text-foreground/60 data-[state=on]:bg-indigo-500/20 data-[state=on]:text-indigo-400"
                >
                  {bs}
                </ToggleGroupItem>
              ))}
            </ToggleGroup>
          </div>

          {/* Optimizer */}
          <div>
            <label className="mb-1 block text-[10px] uppercase tracking-wider text-foreground/30">
              Optimizer
            </label>
            <ToggleGroup
              type="single"
              value={optimizer}
              onValueChange={(v) => v && setOptimizer(v as OptimizerType)}
              disabled={isBusy}
              spacing={1}
              className="flex w-full gap-1"
            >
              {OPTIMIZER_OPTIONS.map((opt) => (
                <ToggleGroupItem
                  key={opt.value}
                  value={opt.value}
                  className="h-auto min-w-0 shrink flex-1 rounded-md px-1 py-1 text-[11px] font-medium bg-white/5 text-foreground/40 hover:bg-white/10 hover:text-foreground/60 data-[state=on]:bg-indigo-500/20 data-[state=on]:text-indigo-400"
                >
                  {opt.label}
                </ToggleGroupItem>
              ))}
            </ToggleGroup>
          </div>
        </div>
      </div>

      {/* Max samples slider (server modes) */}
      {trainingMode !== "browser" && (
        <div>
          <label className="mb-1 block text-[10px] uppercase tracking-wider text-foreground/30">
            Training Samples
          </label>
          <Slider
            value={[maxSamples]}
            onValueChange={([v]) => setMaxSamples(v)}
            min={5000}
            max={50000}
            step={5000}
            disabled={isBusy}
            className="my-2 [&_[data-slot=slider-range]]:bg-purple-500 [&_[data-slot=slider-thumb]]:border-purple-500 [&_[data-slot=slider-thumb]]:size-3"
          />
          <span className="block text-center font-mono text-[10px] text-foreground/40">
            {(maxSamples / 1000).toFixed(0)}K samples
          </span>
        </div>
      )}

      {/* Action buttons */}
      <div className="flex items-center gap-2">
        {!isTraining ? (
          <Button
            onClick={onTrain}
            disabled={isBusy || hasErrors}
            size="sm"
            className={cn(
              "text-xs font-semibold text-white",
              trainingMode === "gpu"
                ? "bg-emerald-600 hover:bg-emerald-500"
                : trainingMode === "hf"
                  ? "bg-purple-600 hover:bg-purple-500"
                  : "bg-indigo-600 hover:bg-indigo-500",
            )}
          >
            {phase === "loading-data"
              ? trainingMode !== "browser"
                ? "Connecting..."
                : "Loading data..."
              : phase === "building"
                ? "Building model..."
                : trainingMode === "gpu"
                  ? "Train on GPU"
                  : trainingMode === "hf"
                    ? "Train on HF"
                    : "Train"}
          </Button>
        ) : (
          <Button
            onClick={onStop}
            variant="destructive"
            size="sm"
            className="text-xs font-semibold"
          >
            Stop
          </Button>
        )}

        <Button
          onClick={onReset}
          disabled={isBusy}
          variant="outline"
          size="sm"
          className="text-xs font-medium text-foreground/50"
        >
          Reset
        </Button>

        {/* Status */}
        {gpuStatus && trainingMode !== "browser" && (
          <span className={cn(
            "ml-2 font-mono text-xs",
            trainingMode === "gpu" ? "text-emerald-400/70" : "text-purple-400/70"
          )}>
            {gpuStatus}
          </span>
        )}
        {isTraining && !gpuStatus && (
          <span className="ml-2 font-mono text-xs text-foreground/40">
            Epoch {currentEpoch}/{epochs}
            {trainingMode === "browser" && totalBatches > 0 && (
              <span>
                {" "}
                — Batch {currentBatch}/{totalBatches}
              </span>
            )}
          </span>
        )}
      </div>

      {/* Progress bar */}
      {isTraining && trainingMode === "browser" && totalBatches > 0 && (
        <Progress
          value={(currentBatch / totalBatches) * 100}
          className="h-1 bg-white/5 [&_[data-slot=progress-indicator]]:bg-indigo-500"
        />
      )}
      {isTraining && trainingMode !== "browser" && epochs > 0 && (
        <Progress
          value={(currentEpoch / epochs) * 100}
          className={cn(
            "h-1 bg-white/5",
            trainingMode === "gpu"
              ? "[&_[data-slot=progress-indicator]]:bg-emerald-500"
              : "[&_[data-slot=progress-indicator]]:bg-purple-500"
          )}
        />
      )}

      {/* Error message */}
      {phase === "error" && errorMessage && (
        <p className="text-xs text-red-400">{errorMessage}</p>
      )}
    </div>
  );
}
