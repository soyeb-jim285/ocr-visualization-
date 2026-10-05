"use client";

import { useModel } from "@/hooks/useModel";

export function ModelProvider({ children }: { children: React.ReactNode }) {
  const { modelLoaded, error } = useModel();

  return (
    <>
      {!modelLoaded && !error && (
        <div
          role="status"
          className="fixed bottom-4 left-1/2 z-40 flex -translate-x-1/2 items-center gap-3 rounded-full border border-border bg-surface px-4 py-2"
        >
          <span className="text-xs text-foreground/60">Warming up model&hellip;</span>
          <div className="h-1 w-24 overflow-hidden rounded-full bg-border">
            <div className="h-full w-1/2 animate-pulse rounded-full bg-accent-primary motion-reduce:animate-none" />
          </div>
        </div>
      )}

      {error && (
        <div role="alert" className="mx-auto max-w-md p-4 text-center">
          <p className="text-lg font-medium text-foreground">Failed to load model</p>
          <p className="text-sm text-foreground/60">
            {error}. Make sure the model files exist in public/models/combined-cnn/.
          </p>
        </div>
      )}

      {children}
    </>
  );
}
