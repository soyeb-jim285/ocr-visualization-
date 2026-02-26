"use client";

import { useEffect, useRef } from "react";
import {
  prefetchAllEpochs,
  prefetchAllEpochInferences,
  clearInferenceCache,
  getCachedModelCount,
  PREFETCH_EPOCHS,
} from "@/lib/model/epochModels";
import { preprocessCanvas } from "@/lib/model/preprocess";
import { useInferenceStore } from "@/stores/inferenceStore";

/**
 * Invisible component that:
 * 1. Starts downloading key epoch checkpoint models on page load
 * 2. When the user draws something, pre-computes all epoch inferences in the background
 *
 * Mount at the top level (not inside LazySection) so it runs immediately.
 */
export function EpochPrefetcher() {
  const inputImageData = useInferenceStore((s) => s.inputImageData);
  const modelsReadyRef = useRef(false);

  // Start downloading all epoch models immediately
  useEffect(() => {
    prefetchAllEpochs((loaded, total) => {
      if (loaded >= total) modelsReadyRef.current = true;
    });
  }, []);

  // When input changes & models are available, pre-compute all epoch inferences
  useEffect(() => {
    if (!inputImageData) return;

    const { tensor } = preprocessCanvas(inputImageData);
    const inputId = clearInferenceCache();

    // Wait until at least some models are loaded before starting inference prefetch
    // The inference function will load models on-demand anyway, but let's give
    // the model download a head start
    const delay = getCachedModelCount() >= PREFETCH_EPOCHS.length ? 0 : 2000;

    const timer = setTimeout(() => {
      prefetchAllEpochInferences(tensor, inputId);
    }, delay);

    return () => clearTimeout(timer);
  }, [inputImageData]);

  return null;
}
