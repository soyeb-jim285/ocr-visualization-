import { create } from "zustand";
import type {
  ArchitectureConfig,
  ConvLayerConfig,
  DenseConfig,
} from "@/lib/model-lab/architecture";
import {
  DEFAULT_ARCHITECTURE,
  createDefaultConvLayer,
  MAX_CONV_LAYERS,
} from "@/lib/model-lab/architecture";
import {
  validateArchitecture,
  type ValidationResult,
} from "@/lib/model-lab/architectureValidator";
import type { DatasetType } from "@/lib/model-lab/dataLoader";
import type { OptimizerType, EpochMetrics } from "@/lib/model-lab/trainModel";
import type { LayerActivations } from "@/stores/inferenceStore";

export type TrainingMode = "browser" | "hf" | "gpu";

export type TrainingPhase =
  | "idle"
  | "loading-data"
  | "building"
  | "training"
  | "trained"
  | "error";

interface ModelLabState {
  // Architecture
  architecture: ArchitectureConfig;
  validation: ValidationResult;

  // Dataset
  datasetType: DatasetType;

  // Training mode
  trainingMode: TrainingMode;
  gpuStatus: string | null;
  maxSamples: number;

  // Training config
  learningRate: number;
  epochs: number;
  batchSize: number;
  optimizer: OptimizerType;

  // Training state
  phase: TrainingPhase;
  errorMessage: string | null;
  currentEpoch: number;
  currentBatch: number;
  totalBatches: number;
  trainingHistory: EpochMetrics[];

  // Trained model info (the actual tf.LayersModel is stored outside Zustand
  // to avoid serialization issues — managed by ModelLabSection)
  hasTrainedModel: boolean;
  intermediateLayerNames: string[];

  // Custom inference results
  customPrediction: number[] | null;
  customTopPrediction: { classIndex: number; confidence: number } | null;
  customActivations: LayerActivations;

  // Actions — Architecture
  addConvLayer: () => void;
  removeConvLayer: (id: string) => void;
  updateConvLayer: (id: string, updates: Partial<ConvLayerConfig>) => void;
  reorderConvLayers: (fromIndex: number, toIndex: number) => void;
  setDenseConfig: (config: Partial<DenseConfig>) => void;

  // Actions — Mode
  setTrainingMode: (mode: TrainingMode) => void;
  setGpuStatus: (status: string | null) => void;
  setMaxSamples: (n: number) => void;

  // Actions — Config
  setDatasetType: (type: DatasetType) => void;
  setLearningRate: (lr: number) => void;
  setEpochs: (n: number) => void;
  setBatchSize: (bs: number) => void;
  setOptimizer: (opt: OptimizerType) => void;

  // Actions — Training state
  setPhase: (phase: TrainingPhase) => void;
  setErrorMessage: (msg: string | null) => void;
  setCurrentEpoch: (epoch: number) => void;
  setBatchProgress: (batch: number, total: number) => void;
  addEpochMetrics: (metrics: EpochMetrics) => void;
  setTrainedModel: (layerNames: string[]) => void;

  // Actions — Inference
  setCustomPrediction: (prediction: number[] | null) => void;
  setCustomActivations: (activations: LayerActivations) => void;

  // Reset
  reset: () => void;
}

function revalidate(arch: ArchitectureConfig, datasetType: DatasetType): ValidationResult {
  const numClasses = datasetType === "digits" ? 10 : datasetType === "bangla" ? 84 : datasetType === "combined" ? 146 : 62;
  return validateArchitecture(arch, numClasses);
}

const initialArch = { ...DEFAULT_ARCHITECTURE, convLayers: [...DEFAULT_ARCHITECTURE.convLayers] };
const initialValidation = revalidate(initialArch, "digits");

export const useModelLabStore = create<ModelLabState>((set, get) => ({
  // Architecture
  architecture: initialArch,
  validation: initialValidation,

  // Dataset
  datasetType: "digits",

  // Training mode
  trainingMode: "browser",
  gpuStatus: null,
  maxSamples: 20000,

  // Training config
  learningRate: 0.001,
  epochs: 10,
  batchSize: 32,
  optimizer: "adam",

  // Training state
  phase: "idle",
  errorMessage: null,
  currentEpoch: 0,
  currentBatch: 0,
  totalBatches: 0,
  trainingHistory: [],

  // Model
  hasTrainedModel: false,
  intermediateLayerNames: [],

  // Inference
  customPrediction: null,
  customTopPrediction: null,
  customActivations: {},

  // Architecture actions
  addConvLayer: () =>
    set((s) => {
      if (s.architecture.convLayers.length >= MAX_CONV_LAYERS) return s;
      const newLayers = [...s.architecture.convLayers, createDefaultConvLayer()];
      const arch = { ...s.architecture, convLayers: newLayers };
      return { architecture: arch, validation: revalidate(arch, s.datasetType) };
    }),

  removeConvLayer: (id) =>
    set((s) => {
      const newLayers = s.architecture.convLayers.filter((l) => l.id !== id);
      const arch = { ...s.architecture, convLayers: newLayers };
      return { architecture: arch, validation: revalidate(arch, s.datasetType) };
    }),

  updateConvLayer: (id, updates) =>
    set((s) => {
      const newLayers = s.architecture.convLayers.map((l) =>
        l.id === id ? { ...l, ...updates } : l,
      );
      const arch = { ...s.architecture, convLayers: newLayers };
      return { architecture: arch, validation: revalidate(arch, s.datasetType) };
    }),

  reorderConvLayers: (fromIndex, toIndex) =>
    set((s) => {
      const layers = [...s.architecture.convLayers];
      const [moved] = layers.splice(fromIndex, 1);
      layers.splice(toIndex, 0, moved);
      const arch = { ...s.architecture, convLayers: layers };
      return { architecture: arch, validation: revalidate(arch, s.datasetType) };
    }),

  setDenseConfig: (config) =>
    set((s) => {
      const dense = { ...s.architecture.dense, ...config };
      const arch = { ...s.architecture, dense };
      return { architecture: arch, validation: revalidate(arch, s.datasetType) };
    }),

  // Mode actions
  setTrainingMode: (mode) => {
    set({ trainingMode: mode });
  },
  setGpuStatus: (status) => set({ gpuStatus: status }),
  setMaxSamples: (n) => set({ maxSamples: n }),

  // Config actions
  setDatasetType: (type) =>
    set((s) => ({
      datasetType: type,
      validation: revalidate(s.architecture, type),
    })),

  setLearningRate: (lr) => set({ learningRate: lr }),
  setEpochs: (n) => set({ epochs: n }),
  setBatchSize: (bs) => set({ batchSize: bs }),
  setOptimizer: (opt) => set({ optimizer: opt }),

  // Training state actions
  setPhase: (phase) => set({ phase }),
  setErrorMessage: (msg) => set({ errorMessage: msg }),
  setCurrentEpoch: (epoch) => set({ currentEpoch: epoch }),
  setBatchProgress: (batch, total) =>
    set({ currentBatch: batch, totalBatches: total }),
  addEpochMetrics: (metrics) =>
    set((s) => ({ trainingHistory: [...s.trainingHistory, metrics] })),
  setTrainedModel: (layerNames) =>
    set({ hasTrainedModel: true, intermediateLayerNames: layerNames }),

  // Inference actions
  setCustomPrediction: (prediction) => {
    if (!prediction) {
      set({ customPrediction: null, customTopPrediction: null });
      return;
    }
    const maxIdx = prediction.indexOf(Math.max(...prediction));
    set({
      customPrediction: prediction,
      customTopPrediction: { classIndex: maxIdx, confidence: prediction[maxIdx] },
    });
  },
  setCustomActivations: (activations) => set({ customActivations: activations }),

  // Reset
  reset: () =>
    set({
      architecture: { ...DEFAULT_ARCHITECTURE, convLayers: [...DEFAULT_ARCHITECTURE.convLayers] },
      validation: revalidate(DEFAULT_ARCHITECTURE, get().datasetType),
      gpuStatus: null,
      phase: "idle",
      errorMessage: null,
      currentEpoch: 0,
      currentBatch: 0,
      totalBatches: 0,
      trainingHistory: [],
      hasTrainedModel: false,
      intermediateLayerNames: [],
      customPrediction: null,
      customTopPrediction: null,
      customActivations: {},
    }),
}));
