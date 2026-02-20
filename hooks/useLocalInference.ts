"use client";

import { useState, useCallback, useRef } from "react";
import { preprocessCanvas } from "@/lib/model/preprocess";
import { runInference, type InferenceResult } from "@/lib/model/predict";
import { useUIStore } from "@/stores/uiStore";
import type { LayerActivations } from "@/stores/inferenceStore";

export interface LocalInferenceState {
  inputTensor: number[][] | null;
  layerActivations: LayerActivations;
  prediction: number[] | null;
  topPrediction: { classIndex: number; confidence: number } | null;
  isInferring: boolean;
}

/**
 * Like useInference, but stores results in local React state instead of global Zustand store.
 * Used by Layout 7 (Comparison Lab) for multi-input side-by-side comparison.
 */
export function useLocalInference() {
  const [state, setState] = useState<LocalInferenceState>({
    inputTensor: null,
    layerActivations: {},
    prediction: null,
    topPrediction: null,
    isInferring: false,
  });
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const modelLoaded = useUIStore((s) => s.modelLoaded);

  const infer = useCallback(
    (imageData: ImageData) => {
      if (!modelLoaded) return;
      if (debounceRef.current) clearTimeout(debounceRef.current);

      debounceRef.current = setTimeout(async () => {
        setState((s) => ({ ...s, isInferring: true }));
        try {
          const { tensor, pixelArray } = preprocessCanvas(imageData);
          const { prediction, layerActivations } = await runInference(tensor);
          const maxIdx = prediction.indexOf(Math.max(...prediction));
          setState({
            inputTensor: pixelArray,
            layerActivations,
            prediction,
            topPrediction: { classIndex: maxIdx, confidence: prediction[maxIdx] },
            isInferring: false,
          });
        } catch (error) {
          console.error("Local inference failed:", error);
          setState((s) => ({ ...s, isInferring: false }));
        }
      }, 150);
    },
    [modelLoaded]
  );

  const reset = useCallback(() => {
    setState({
      inputTensor: null,
      layerActivations: {},
      prediction: null,
      topPrediction: null,
      isInferring: false,
    });
  }, []);

  return { ...state, infer, reset };
}
