"use client";

import { useCallback, useRef, useState } from "react";
import { preprocessImage } from "@/lib/model/preprocess";
import { runInference } from "@/lib/model/predict";
import { useInferenceStore } from "@/stores/inferenceStore";
import { useUIStore } from "@/stores/uiStore";

interface ImageUploaderProps {
  compact?: boolean;
}

export function ImageUploader({ compact = false }: ImageUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const modelLoaded = useUIStore((s) => s.modelLoaded);

  const handleFile = useCallback(
    async (file: File) => {
      if (!modelLoaded) return;
      if (!file.type.startsWith("image/")) return;

      // Access actions via getState() to avoid subscribing to the entire store
      const store = useInferenceStore.getState();
      const gen = store.generation;
      store.setIsInferring(true);
      try {
        const { tensor, pixelArray } = await preprocessImage(file);
        if (useInferenceStore.getState().generation !== gen) return;
        store.setInputTensor(pixelArray);

        const { prediction, layerActivations } = await runInference(tensor);
        if (useInferenceStore.getState().generation !== gen) return;
        store.setPrediction(prediction);
        store.setLayerActivations(layerActivations);
      } catch (error) {
        console.error("Image processing failed:", error);
      } finally {
        const current = useInferenceStore.getState();
        if (current.generation === gen) {
          current.setIsInferring(false);
        }
      }
    },
    [modelLoaded]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  if (compact) {
    return (
      <>
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className="relative inline-flex size-11 sm:size-6 items-center justify-center rounded-[2px] border border-rule bg-transparent text-ink-2 transition-colors duration-150 sm:after:absolute sm:after:-inset-2 hover:border-rule-strong hover:text-phosphor"
          aria-label="Upload image"
          title="Upload image"
        >
          <svg
            width="12"
            height="12"
            viewBox="0 0 16 16"
            fill="none"
            className="opacity-80"
          >
            <path
              d="M14 10v3a1 1 0 01-1 1H3a1 1 0 01-1-1v-3M11 5L8 2 5 5M8 2v9"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleFile(file);
          }}
        />
      </>
    );
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      className="inline-flex"
    >
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        data-dragging={isDragging}
        className="link-mono py-2 data-[dragging=true]:text-phosphor"
      >
        {isDragging ? "drop to read" : "or upload an image"}
      </button>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFile(file);
        }}
      />
    </div>
  );
}
