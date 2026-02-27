"use client";

import { useModelLabStore } from "@/stores/modelLabStore";
import type { TrainingMode } from "@/stores/modelLabStore";
import type { DatasetType } from "@/lib/model-lab/dataLoader";
import type { OptimizerType } from "@/lib/model-lab/trainModel";

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
        <div className="flex gap-1">
          {MODE_OPTIONS.map((mode) => (
            <button
              key={mode.value}
              onClick={() => setTrainingMode(mode.value)}
              disabled={isBusy}
              className={`flex-1 rounded-md px-2 py-1.5 text-left transition ${
                trainingMode === mode.value
                  ? mode.value === "gpu"
                    ? "bg-emerald-500/15 text-emerald-400"
                    : mode.value === "hf"
                      ? "bg-purple-500/15 text-purple-400"
                      : "bg-indigo-500/15 text-indigo-400"
                  : "bg-white/5 text-foreground/40 hover:bg-white/10"
              } disabled:opacity-40`}
            >
              <span className="block text-xs font-medium">{mode.label}</span>
              <span className="block text-[10px] text-foreground/30">
                {mode.desc}
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* Dataset selector */}
      <div>
        <label className="mb-1.5 block text-[11px] uppercase tracking-wider text-foreground/35">
          Dataset
        </label>
        <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-4">
          {DATASET_TABS.map((tab) => (
            <button
              key={tab.value}
              onClick={() => setDatasetType(tab.value)}
              disabled={isBusy}
              className={`rounded-md px-2 py-1.5 text-left transition ${
                datasetType === tab.value
                  ? "bg-indigo-500/15 text-indigo-400"
                  : "bg-white/5 text-foreground/40 hover:bg-white/10"
              } disabled:opacity-40`}
            >
              <span className="block text-xs font-medium">{tab.label}</span>
              <span className="block text-[10px] text-foreground/30">
                {tab.desc}
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* Hyperparameters */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {/* Learning rate */}
        <div>
          <label className="mb-1 block text-[10px] uppercase tracking-wider text-foreground/30">
            Learning Rate
          </label>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={lrToSlider(learningRate)}
            onChange={(e) => setLearningRate(sliderToLr(+e.target.value))}
            disabled={isBusy}
            className="w-full accent-indigo-500"
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
          <input
            type="number"
            min={1}
            max={50}
            value={epochs}
            onChange={(e) => setEpochs(Math.max(1, Math.min(50, +e.target.value)))}
            disabled={isBusy}
            className="w-full rounded-md border border-border/40 bg-black/30 px-2 py-1 font-mono text-xs text-foreground/70"
          />
        </div>

        {/* Batch size */}
        <div>
          <label className="mb-1 block text-[10px] uppercase tracking-wider text-foreground/30">
            Batch Size
          </label>
          <div className="flex gap-1">
            {BATCH_OPTIONS.map((bs) => (
              <button
                key={bs}
                onClick={() => setBatchSize(bs)}
                disabled={isBusy}
                className={`flex-1 rounded-md py-1 text-[11px] font-medium transition ${
                  batchSize === bs
                    ? "bg-indigo-500/20 text-indigo-400"
                    : "bg-white/5 text-foreground/40 hover:bg-white/10"
                } disabled:opacity-40`}
              >
                {bs}
              </button>
            ))}
          </div>
        </div>

        {/* Optimizer */}
        <div>
          <label className="mb-1 block text-[10px] uppercase tracking-wider text-foreground/30">
            Optimizer
          </label>
          <div className="flex gap-1">
            {OPTIMIZER_OPTIONS.map((opt) => (
              <button
                key={opt.value}
                onClick={() => setOptimizer(opt.value)}
                disabled={isBusy}
                className={`flex-1 rounded-md py-1 text-[11px] font-medium transition ${
                  optimizer === opt.value
                    ? "bg-indigo-500/20 text-indigo-400"
                    : "bg-white/5 text-foreground/40 hover:bg-white/10"
                } disabled:opacity-40`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Max samples slider (server modes) */}
      {trainingMode !== "browser" && (
        <div>
          <label className="mb-1 block text-[10px] uppercase tracking-wider text-foreground/30">
            Training Samples
          </label>
          <input
            type="range"
            min={5000}
            max={50000}
            step={5000}
            value={maxSamples}
            onChange={(e) => setMaxSamples(+e.target.value)}
            disabled={isBusy}
            className="w-full accent-purple-500"
          />
          <span className="block text-center font-mono text-[10px] text-foreground/40">
            {(maxSamples / 1000).toFixed(0)}K samples
          </span>
        </div>
      )}

      {/* Action buttons */}
      <div className="flex items-center gap-2">
        {!isTraining ? (
          <button
            onClick={onTrain}
            disabled={isBusy || hasErrors}
            className={`rounded-lg px-4 py-2 text-xs font-semibold text-white transition disabled:opacity-40 ${
              trainingMode === "gpu"
                ? "bg-emerald-600 hover:bg-emerald-500"
                : trainingMode === "hf"
                  ? "bg-purple-600 hover:bg-purple-500"
                  : "bg-indigo-600 hover:bg-indigo-500"
            }`}
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
          </button>
        ) : (
          <button
            onClick={onStop}
            className="rounded-lg bg-red-600/80 px-4 py-2 text-xs font-semibold text-white transition hover:bg-red-500"
          >
            Stop
          </button>
        )}

        <button
          onClick={onReset}
          disabled={isBusy}
          className="rounded-lg border border-border/40 px-4 py-2 text-xs font-medium text-foreground/50 transition hover:bg-white/5 disabled:opacity-30"
        >
          Reset
        </button>

        {/* Status */}
        {gpuStatus && trainingMode !== "browser" && (
          <span className={`ml-2 font-mono text-xs ${
            trainingMode === "gpu" ? "text-emerald-400/70" : "text-purple-400/70"
          }`}>
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
        <div className="h-1 w-full overflow-hidden rounded-full bg-white/5">
          <div
            className="h-full rounded-full bg-indigo-500 transition-all duration-150"
            style={{
              width: `${(currentBatch / totalBatches) * 100}%`,
            }}
          />
        </div>
      )}
      {isTraining && trainingMode !== "browser" && epochs > 0 && (
        <div className="h-1 w-full overflow-hidden rounded-full bg-white/5">
          <div
            className={`h-full rounded-full transition-all duration-300 ${
              trainingMode === "gpu" ? "bg-emerald-500" : "bg-purple-500"
            }`}
            style={{
              width: `${(currentEpoch / epochs) * 100}%`,
            }}
          />
        </div>
      )}

      {/* Error message */}
      {phase === "error" && errorMessage && (
        <p className="text-xs text-red-400">{errorMessage}</p>
      )}
    </div>
  );
}
