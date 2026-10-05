"use client";

import { useEffect } from "react";
import { clearInferenceCache } from "@/lib/model/epochModels";
import { useInferenceStore } from "@/stores/inferenceStore";

/**
 * Invisible component: invalidates the shared epoch inference cache when the
 * drawn input changes. Epoch downloads/inference are driven by
 * EpochNetworkVisualization (lazy, only when scrolled near).
 */
export function EpochPrefetcher() {
  const inputImageData = useInferenceStore((s) => s.inputImageData);

  useEffect(() => {
    clearInferenceCache();
  }, [inputImageData]);

  return null;
}
