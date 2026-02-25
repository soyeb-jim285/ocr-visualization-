"use client";

import { useModel } from "@/hooks/useModel";

export function ModelProvider({ children }: { children: React.ReactNode }) {
  const { modelLoaded, modelLoadingProgress, error } = useModel();

  return (
    <>
      {/* Loading overlay */}
      {!modelLoaded && !error && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-background/95 backdrop-blur-sm">
          <div className="flex flex-col items-center gap-6">
            {/* Animated neural network icon */}
            <div className="relative h-16 w-16">
              <div className="absolute inset-0 animate-ping rounded-full bg-accent-primary/20" />
              <div className="absolute inset-2 animate-pulse rounded-full bg-accent-primary/40" />
              <div className="absolute inset-4 rounded-full bg-accent-primary" />
            </div>

            <div className="flex flex-col items-center gap-2">
              <p className="text-lg font-medium text-foreground">
                Loading Neural Network
              </p>
              <p className="text-sm text-foreground/50">
                {Math.round(modelLoadingProgress * 100)}% loaded
              </p>
            </div>

            {/* Progress bar */}
            <div className="h-1.5 w-64 overflow-hidden rounded-full bg-border">
              <div
                className="h-full rounded-full bg-accent-primary transition-all duration-300"
                style={{ width: `${modelLoadingProgress * 100}%` }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Error state */}
      {error && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-background/95">
          <div className="flex max-w-md flex-col items-center gap-4 text-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-accent-negative/10">
              <span className="text-2xl text-accent-negative">!</span>
            </div>
            <p className="text-lg font-medium text-foreground">
              Failed to load model
            </p>
            <p className="text-sm text-foreground/50">
              {error}. Make sure the model files exist in public/models/emnist-cnn/.
            </p>
          </div>
        </div>
      )}

      {children}
    </>
  );
}
