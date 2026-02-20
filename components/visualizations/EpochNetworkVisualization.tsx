"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import {
  runEpochInference,
  TOTAL_EPOCHS,
  getCachedModelCount,
  getInferenceCache,
} from "@/lib/model/epochModels";
import { preprocessCanvas } from "@/lib/model/preprocess";
import { useInferenceStore } from "@/stores/inferenceStore";
import { EMNIST_CLASSES } from "@/lib/model/classes";
import { NeuronNetworkCanvas } from "@/components/canvas/NeuronNetworkCanvas";
import {
  LAYERS,
  extractActivations,
  getOutputLabels,
  type HoveredNeuron,
} from "@/lib/network/networkConstants";
import type { InferenceResult } from "@/lib/model/predict";
import type { LayerActivations } from "@/stores/inferenceStore";

const ANIM_DURATION = 200; // ms for brightness transition between epochs

export function EpochNetworkVisualization() {
  const inputImageData = useInferenceStore((s) => s.inputImageData);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [epochPrediction, setEpochPrediction] = useState<number[] | null>(null);
  const [epochActivations, setEpochActivations] = useState<LayerActivations>({});
  const [isLoading, setIsLoading] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadedCount, setLoadedCount] = useState(() => getCachedModelCount());
  const [drawTrigger, setDrawTrigger] = useState(0);

  const tensorRef = useRef<Float32Array | null>(null);
  const inputTensor2DRef = useRef<number[][] | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingEpochRef = useRef<number>(0);
  const playIntervalRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Use the shared module-level inference cache (populated by EpochPrefetcher)
  const resultsCacheRef = useRef(getInferenceCache());

  // Canvas refs
  const activationMapRef = useRef<Map<string, number[]>>(new Map());
  const outputLabelsRef = useRef<string[]>([]);

  // Animation refs for smooth brightness transitions between epochs
  const prevMapRef = useRef<Map<string, number[]>>(new Map());
  const targetMapRef = useRef<Map<string, number[]>>(new Map());
  const animStartRef = useRef(0);
  const animRafRef = useRef(0);
  const hoveredLayerRef = useRef<number | null>(null);
  const hoveredNeuronRef = useRef<HoveredNeuron | null>(null);
  const waveRef = useRef(LAYERS.length + 1);

  // Container sizing
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerSize, setContainerSize] = useState({ w: 800, h: 400 });

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const measure = () => {
      const rect = el.getBoundingClientRect();
      setContainerSize({ w: Math.round(rect.width), h: Math.round(rect.height) });
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Poll model download progress (driven by EpochPrefetcher)
  useEffect(() => {
    const interval = setInterval(() => {
      const count = getCachedModelCount();
      setLoadedCount(count);
      if (count >= TOTAL_EPOCHS) clearInterval(interval);
    }, 500);
    return () => clearInterval(interval);
  }, []);

  // Preprocess input — sync with shared cache
  useEffect(() => {
    tensorRef.current = null;
    inputTensor2DRef.current = null;
    // Sync with the shared cache (EpochPrefetcher clears it on input change)
    resultsCacheRef.current = getInferenceCache();
    if (inputImageData) {
      const { tensor, pixelArray } = preprocessCanvas(inputImageData);
      tensorRef.current = tensor;
      inputTensor2DRef.current = pixelArray;
    }
  }, [inputImageData]);

  // Smooth transition: lerp activation values over ANIM_DURATION ms
  const startTransition = useCallback((newActivations: LayerActivations, newPrediction: number[] | null) => {
    const target = extractActivations(newActivations, inputTensor2DRef.current, newPrediction);
    const newLabels = getOutputLabels(newPrediction);

    // First data → snap immediately, no animation
    if (activationMapRef.current.size === 0) {
      activationMapRef.current = target;
      outputLabelsRef.current = newLabels;
      setDrawTrigger((n) => n + 1);
      return;
    }

    // Snapshot current displayed values as "from"
    const prev = new Map<string, number[]>();
    for (const [key, vals] of activationMapRef.current) {
      prev.set(key, vals.slice());
    }
    prevMapRef.current = prev;
    targetMapRef.current = target;
    outputLabelsRef.current = newLabels;

    cancelAnimationFrame(animRafRef.current);
    animStartRef.current = performance.now();

    const animate = () => {
      const elapsed = performance.now() - animStartRef.current;
      const t = Math.min(elapsed / ANIM_DURATION, 1);
      const eased = 1 - (1 - t) * (1 - t); // ease-out quad

      const interpolated = new Map<string, number[]>();
      for (const [key, targetVals] of targetMapRef.current) {
        const prevVals = prevMapRef.current.get(key);
        if (!prevVals) {
          interpolated.set(key, targetVals);
          continue;
        }
        const len = Math.max(prevVals.length, targetVals.length);
        const result = new Array<number>(len);
        for (let i = 0; i < len; i++) {
          const from = i < prevVals.length ? prevVals[i] : 0;
          const to = i < targetVals.length ? targetVals[i] : 0;
          result[i] = from + (to - from) * eased;
        }
        interpolated.set(key, result);
      }

      activationMapRef.current = interpolated;
      setDrawTrigger((n) => n + 1);

      if (t < 1) {
        animRafRef.current = requestAnimationFrame(animate);
      }
    };

    animRafRef.current = requestAnimationFrame(animate);
  }, []);

  // Apply a cached or fresh result to state
  const applyResult = useCallback((result: InferenceResult, epoch: number) => {
    if (pendingEpochRef.current !== epoch) return;
    setEpochPrediction(result.prediction);
    setEpochActivations(result.layerActivations);
    startTransition(result.layerActivations, result.prediction);
    setIsLoading(false);
  }, [startTransition]);

  // Run inference (cache-first) and pre-compute adjacent epochs
  const runAtEpoch = useCallback(async (epoch: number) => {
    if (!tensorRef.current) {
      setEpochPrediction(null);
      setEpochActivations({});
      return;
    }

    // Cache hit — instant
    const cached = resultsCacheRef.current.get(epoch);
    if (cached) {
      applyResult(cached, epoch);
      return;
    }

    setIsLoading(true);
    setError(null);
    try {
      const result = await runEpochInference(tensorRef.current, epoch);
      resultsCacheRef.current.set(epoch, result);
      applyResult(result, epoch);
    } catch {
      if (pendingEpochRef.current === epoch) {
        setError("Model checkpoint not available");
        setEpochPrediction(null);
        setEpochActivations({});
        setIsLoading(false);
      }
    }
  }, [applyResult]);

  // Debounced slider change
  const handleEpochChange = useCallback(
    (epoch: number) => {
      setCurrentEpoch(epoch);
      pendingEpochRef.current = epoch;

      // If cached, apply immediately — no debounce needed
      const cached = resultsCacheRef.current.get(epoch);
      if (cached) {
        if (debounceRef.current) clearTimeout(debounceRef.current);
        applyResult(cached, epoch);
        return;
      }

      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        runAtEpoch(epoch);
      }, 100);
    },
    [runAtEpoch, applyResult],
  );

  // Run on input change
  useEffect(() => {
    pendingEpochRef.current = currentEpoch;
    runAtEpoch(currentEpoch);
    // Only re-run when input changes, not on epoch change (handled by handleEpochChange)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inputImageData]);

  // Cleanup
  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
      if (playIntervalRef.current) clearTimeout(playIntervalRef.current);
      cancelAnimationFrame(animRafRef.current);
    };
  }, []);

  // Auto-play — waits for inference to finish before advancing
  useEffect(() => {
    if (!isPlaying) {
      if (playIntervalRef.current) clearTimeout(playIntervalRef.current);
      return;
    }

    let cancelled = false;
    const advance = async () => {
      if (cancelled) return;
      const next = ((pendingEpochRef.current >= 49 ? -1 : pendingEpochRef.current) + 1);
      setCurrentEpoch(next);
      pendingEpochRef.current = next;
      await runAtEpoch(next);
      if (!cancelled) {
        playIntervalRef.current = setTimeout(advance, 400);
      }
    };

    playIntervalRef.current = setTimeout(advance, 400);
    return () => {
      cancelled = true;
      if (playIntervalRef.current) clearTimeout(playIntervalRef.current);
    };
  }, [isPlaying, runAtEpoch]);

  // Hover callbacks
  const onHoverLayer = useCallback((li: number | null) => {
    hoveredLayerRef.current = li;
  }, []);
  const onHoverNeuron = useCallback((n: HoveredNeuron | null) => {
    hoveredNeuronRef.current = n;
  }, []);
  const onClickLayer = useCallback(() => {}, []);

  const topPrediction = epochPrediction
    ? (() => {
        const maxIdx = epochPrediction.indexOf(Math.max(...epochPrediction));
        return {
          label: EMNIST_CLASSES[maxIdx],
          confidence: epochPrediction[maxIdx],
        };
      })()
    : null;

  const hasInput = inputImageData !== null;
  const allLoaded = loadedCount >= TOTAL_EPOCHS;
  const hasActivations = Object.keys(epochActivations).length > 0;

  return (
    <div className="flex flex-col items-center gap-4">
      {!allLoaded && (
        <div className="flex w-full max-w-xl flex-col items-center gap-1">
          <div className="flex w-full items-center justify-between text-xs text-foreground/30">
            <span>Loading epoch models...</span>
            <span>{loadedCount}/{TOTAL_EPOCHS}</span>
          </div>
          <div className="h-1 w-full overflow-hidden rounded-full bg-surface-elevated">
            <div
              className="h-full rounded-full bg-accent-primary/50 transition-all duration-300"
              style={{ width: `${(loadedCount / TOTAL_EPOCHS) * 100}%` }}
            />
          </div>
        </div>
      )}

      <div className="relative w-full" style={{ height: 780 }}>
        <div ref={containerRef} className="absolute inset-0">
          {hasInput && hasActivations ? (
            <NeuronNetworkCanvas
              width={containerSize.w}
              height={containerSize.h}
              activationMapRef={activationMapRef}
              outputLabelsRef={outputLabelsRef}
              hoveredLayerRef={hoveredLayerRef}
              hoveredNeuronRef={hoveredNeuronRef}
              waveRef={waveRef}
              showSignals={false}
              drawTrigger={drawTrigger}
              onHoverLayer={onHoverLayer}
              onHoverNeuron={onHoverNeuron}
              onClickLayer={onClickLayer}
            />
          ) : (
            <div className="flex h-full items-center justify-center">
              <p className="text-sm text-foreground/20">
                {hasInput ? "Loading..." : "Draw a character to see network activations evolve"}
              </p>
            </div>
          )}
        </div>

        {hasInput && (
          <>
            <div className="pointer-events-none absolute left-4 top-4 z-10">
              {topPrediction && (
                <motion.div
                  key={topPrediction.label}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="flex items-baseline gap-2"
                >
                  <span className="text-4xl font-bold text-accent-primary drop-shadow-lg">
                    {topPrediction.label}
                  </span>
                  <span className="font-mono text-sm text-foreground/50">
                    {(topPrediction.confidence * 100).toFixed(1)}%
                  </span>
                </motion.div>
              )}
              {isLoading && (
                <div className="mt-1 h-4 w-4 animate-spin rounded-full border-2 border-accent-primary border-t-transparent" />
              )}
            </div>

            <div className="pointer-events-none absolute right-4 top-4 z-10">
              <span className="font-mono text-lg font-bold text-foreground/60">
                Epoch {currentEpoch}
              </span>
            </div>
          </>
        )}
      </div>

      <div className="flex w-full max-w-xl items-center gap-3">
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          disabled={!hasInput}
          className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-border/50 bg-surface text-foreground/60 transition-colors hover:bg-surface-elevated hover:text-foreground disabled:opacity-30"
          aria-label={isPlaying ? "Pause" : "Play"}
        >
          {isPlaying ? (
            <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
              <rect x="2" y="1" width="3.5" height="12" rx="1" />
              <rect x="8.5" y="1" width="3.5" height="12" rx="1" />
            </svg>
          ) : (
            <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
              <path d="M3 1.5v11l9-5.5z" />
            </svg>
          )}
        </button>

        <div className="flex min-w-0 flex-1 flex-col gap-1">
          <input
            type="range"
            min={0}
            max={49}
            value={currentEpoch}
            onChange={(e) => handleEpochChange(parseInt(e.target.value))}
            className="w-full accent-accent-primary"
            disabled={!hasInput}
          />
          <div className="flex w-full justify-between px-0.5">
            {[0, 10, 20, 30, 40, 49].map((e) => (
              <button
                key={e}
                onClick={() => { handleEpochChange(e); setIsPlaying(false); }}
                className={`rounded px-1.5 py-0.5 font-mono text-[10px] transition-colors ${
                  currentEpoch === e
                    ? "bg-accent-primary/20 text-accent-primary"
                    : "text-foreground/30 hover:text-foreground/50"
                }`}
              >
                {e}
              </button>
            ))}
          </div>
        </div>
      </div>

      {error && (
        <p className="text-sm text-accent-negative/60">{error}</p>
      )}

      {!hasInput && (
        <p className="text-sm text-foreground/30">
          Draw a character above to see how the model improves across training epochs
        </p>
      )}
    </div>
  );
}
