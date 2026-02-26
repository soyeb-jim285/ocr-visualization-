"use client";

import { useCallback, useRef } from "react";
import { preprocessCanvas } from "@/lib/model/preprocess";
import { runInference } from "@/lib/model/predict";
import { useInferenceStore } from "@/stores/inferenceStore";
import { useUIStore } from "@/stores/uiStore";

export function useInference() {
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const modelLoaded = useUIStore((s) => s.modelLoaded);

  const cancel = useCallback(() => {
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
      debounceRef.current = null;
    }
  }, []);

  const infer = useCallback(
    async (imageData: ImageData) => {
      if (!modelLoaded) return;

      // Debounce: wait 150ms after last stroke
      if (debounceRef.current) clearTimeout(debounceRef.current);

      debounceRef.current = setTimeout(async () => {
        // Snapshot generation before async work
        const store = useInferenceStore.getState();
        const gen = store.generation;
        store.setIsInferring(true);
        store.setInputImageData(imageData);

        try {
          const { tensor, pixelArray } = preprocessCanvas(imageData);
          // Abort if reset happened during preprocessing
          if (useInferenceStore.getState().generation !== gen) return;
          store.setInputTensor(pixelArray);

          const { prediction, layerActivations } = await runInference(tensor);
          // Abort if reset happened during inference
          if (useInferenceStore.getState().generation !== gen) return;
          store.setPrediction(prediction);
          store.setLayerActivations(layerActivations);
        } catch (error) {
          console.error("Inference failed:", error);
        } finally {
          // Only clear isInferring if this generation is still current
          const current = useInferenceStore.getState();
          if (current.generation === gen) {
            current.setIsInferring(false);
          }
        }
      }, 150);
    },
    [modelLoaded]
  );

  return { infer, cancel };
}
