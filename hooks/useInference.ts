"use client";

import { useCallback, useRef } from "react";
import { preprocessCanvas } from "@/lib/model/preprocess";
import { runInference } from "@/lib/model/predict";
import { useInferenceStore } from "@/stores/inferenceStore";
import { useUIStore } from "@/stores/uiStore";

export function useInference() {
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const modelLoaded = useUIStore((s) => s.modelLoaded);

  const infer = useCallback(
    async (imageData: ImageData) => {
      if (!modelLoaded) return;

      // Debounce: wait 150ms after last stroke
      if (debounceRef.current) clearTimeout(debounceRef.current);

      debounceRef.current = setTimeout(async () => {
        // Access actions via getState() to avoid subscribing to the entire store
        const store = useInferenceStore.getState();
        store.setIsInferring(true);
        store.setInputImageData(imageData);

        try {
          const { tensor, pixelArray } = preprocessCanvas(imageData);
          store.setInputTensor(pixelArray);

          const { prediction, layerActivations } = await runInference(tensor);
          store.setPrediction(prediction);
          store.setLayerActivations(layerActivations);
        } catch (error) {
          console.error("Inference failed:", error);
        } finally {
          useInferenceStore.getState().setIsInferring(false);
        }
      }, 150);
    },
    [modelLoaded]
  );

  return { infer };
}
