"use client";

import { useModel } from "@/hooks/useModel";

export function ModelProvider({ children }: { children: React.ReactNode }) {
  const { modelLoaded, error } = useModel();

  return (
    <>
      {!modelLoaded && !error && (
        <div
          role="status"
          className="fixed bottom-4 left-1/2 z-40 flex -translate-x-1/2 items-center gap-3 rounded-[3px] border border-rule-strong bg-bg-raised px-3 py-2"
        >
          <span className="font-mono text-[11px] tracking-[0.08em] text-ink-2">LOADING WEIGHTS</span>
          <div className="h-px w-20 overflow-hidden bg-rule">
            <div className="h-full w-1/2 animate-pulse bg-phosphor motion-reduce:animate-none" />
          </div>
        </div>
      )}

      {error && (
        <div role="alert" className="mx-auto max-w-md p-4 text-center">
          <p className="font-serif text-xl text-ink">Failed to load model</p>
          <p className="text-sm text-ink-3">
            {error}. Make sure the model files exist in public/models/emnist-cnn/.
          </p>
        </div>
      )}

      {children}
    </>
  );
}
