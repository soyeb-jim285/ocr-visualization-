import { create } from "zustand";

export interface NeuronSelection {
  layerName: string;
  neuronIndex: number;
}

/** Activations can be 3D (conv layers: [channels][height][width]) or 1D (dense layers) */
export type LayerActivations = Record<string, number[][][] | number[]>;

interface InferenceState {
  // Input
  inputImageData: ImageData | null;
  inputTensor: number[][] | null; // 28x28 normalized pixel values

  // Per-layer activations (keyed by layer name)
  layerActivations: LayerActivations;

  // Prediction
  prediction: number[] | null; // 62-element probability array
  topPrediction: { classIndex: number; confidence: number } | null;

  // Neuron inspection
  selectedNeuron: NeuronSelection | null;

  // Loading state
  isInferring: boolean;

  // Actions
  setInputImageData: (data: ImageData | null) => void;
  setInputTensor: (tensor: number[][] | null) => void;
  setLayerActivations: (activations: LayerActivations) => void;
  setPrediction: (prediction: number[] | null) => void;
  setSelectedNeuron: (neuron: NeuronSelection | null) => void;
  setIsInferring: (val: boolean) => void;
  reset: () => void;
}

export const useInferenceStore = create<InferenceState>((set) => ({
  inputImageData: null,
  inputTensor: null,
  layerActivations: {},
  prediction: null,
  topPrediction: null,
  selectedNeuron: null,
  isInferring: false,

  setInputImageData: (data) => set({ inputImageData: data }),

  setInputTensor: (tensor) => set({ inputTensor: tensor }),

  setLayerActivations: (activations) => set({ layerActivations: activations }),

  setPrediction: (prediction) => {
    if (!prediction) {
      set({ prediction: null, topPrediction: null });
      return;
    }
    const maxIdx = prediction.indexOf(Math.max(...prediction));
    set({
      prediction,
      topPrediction: { classIndex: maxIdx, confidence: prediction[maxIdx] },
    });
  },

  setSelectedNeuron: (neuron) => set({ selectedNeuron: neuron }),

  setIsInferring: (val) => set({ isInferring: val }),

  reset: () =>
    set({
      inputImageData: null,
      inputTensor: null,
      layerActivations: {},
      prediction: null,
      topPrediction: null,
      selectedNeuron: null,
      isInferring: false,
    }),
}));
