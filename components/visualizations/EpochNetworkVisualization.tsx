"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import {
  runEpochInference,
  TOTAL_EPOCHS,
  PREFETCH_EPOCHS,
  getPrefetchedCount,
  prefetchAllEpochs,
  getInferenceCache,
  clearInferenceCache,
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
  const [loadedCount, setLoadedCount] = useState(() => getPrefetchedCount());
  const [drawTrigger, setDrawTrigger] = useState(0);

  const tensorRef = useRef<Float32Array | null>(null);
  const inputTensor2DRef = useRef<number[][] | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingEpochRef = useRef<number>(0);
  const playIntervalRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Use the shared module-level inference cache (cleared by EpochPrefetcher on input change)
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

  // Start background checkpoint download on mount (idempotent, low concurrency)
  // and poll its progress
  useEffect(() => {
    prefetchAllEpochs();
    const interval = setInterval(() => {
      const count = getPrefetchedCount();
      setLoadedCount(count);
      if (count >= PREFETCH_EPOCHS.length) clearInterval(interval);
    }, 500);
    return () => clearInterval(interval);
  }, []);

  // Preprocess input — sync with shared cache
  useEffect(() => {
    tensorRef.current = null;
    inputTensor2DRef.current = null;
    // Sync with the shared cache (EpochPrefetcher clears it on input change)
    // clear here too: effect order vs EpochPrefetcher must not let a stale cache hit through
    clearInferenceCache();
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
      setError(null);
      applyResult(cached, epoch);
      return;
    }

    setIsLoading(true);
    setError(null);
    const tensor = tensorRef.current;
    const cache = resultsCacheRef.current;
    try {
      let result;
      try {
        result = await runEpochInference(tensor, epoch);
      } catch (e) {
        console.warn("Epoch inference failed, retrying once:", e);
        result = await runEpochInference(tensor, epoch);
      }
      // input changed while running: drop the stale result
      if (tensorRef.current !== tensor) return;
      cache.set(epoch, result);
      applyResult(result, epoch);
    } catch (err) {
      console.error(err);
      if (pendingEpochRef.current === epoch && tensorRef.current === tensor) {
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
        setError(null);
        if (debounceRef.current) clearTimeout(debounceRef.current);
        applyResult(cached, epoch);
        return;
      }

      if (debounceRef.current) clearTimeout(debounceRef.current);
      if (tensorRef.current) setIsLoading(true);
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
      const next = ((pendingEpochRef.current >= TOTAL_EPOCHS - 1 ? -1 : pendingEpochRef.current) + 1);
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

  // pending checkpoint: don't show the previous epoch's prediction next to the new epoch number
  const topPrediction = epochPrediction && !isLoading
    ? (() => {
        const maxIdx = epochPrediction.indexOf(Math.max(...epochPrediction));
        return {
          label: EMNIST_CLASSES[maxIdx],
          confidence: epochPrediction[maxIdx],
        };
      })()
    : null;

  const hasInput = inputImageData !== null;
  const allLoaded = loadedCount >= PREFETCH_EPOCHS.length;
  const hasActivations = Object.keys(epochActivations).length > 0;

  const pct = (currentEpoch / (TOTAL_EPOCHS - 1)) * 100;

  return (
    <div className="flex flex-col gap-4">
      {!allLoaded && (
        <div className="flex w-full flex-col gap-1.5">
          <div className="flex w-full items-center justify-between font-mono text-[11px] text-ink-3">
            <span>LOADING CHECKPOINTS</span>
            <span>{loadedCount}/{PREFETCH_EPOCHS.length}</span>
          </div>
          <div className="h-px w-full bg-rule">
            <div
              className="h-px origin-left bg-phosphor transition-transform duration-300"
              style={{ transform: `scaleX(${loadedCount / PREFETCH_EPOCHS.length})` }}
            />
          </div>
        </div>
      )}

      {hasInput && (
        <div className="flex items-end justify-between font-mono text-[11px] text-ink-3 sm:hidden">
          <div>
            <p className="tracking-[0.08em]">PREDICTION</p>
            <p className="flex items-baseline gap-2">
              <span className="font-serif text-4xl leading-none text-annotation">{topPrediction?.label ?? "–"}</span>
              {topPrediction && (
                <span className="text-xs tabular-nums text-ink">{(topPrediction.confidence * 100).toFixed(1)}%</span>
              )}
            </p>
          </div>
          <div className="text-right">
            <p className="tracking-[0.08em]">EPOCH</p>
            <p className="text-2xl leading-none tabular-nums text-ink">
              {String(currentEpoch).padStart(3, "0")}
              <span className="ml-1 text-[11px] text-ink-3">/ {TOTAL_EPOCHS - 1}</span>
            </p>
          </div>
        </div>
      )}

      <div
        ref={(el) => {
          // center the wide network on first show (phones)
          if (el && !el.dataset.c && el.scrollWidth > el.clientWidth) {
            el.dataset.c = "1";
            el.scrollLeft = (el.scrollWidth - el.clientWidth) / 2;
          }
        }}
        className={`well plate-marks relative overflow-x-auto overscroll-x-contain scrollbar-none transition-opacity ${isLoading && hasActivations ? "opacity-50" : ""}`}
      >
        {!(hasInput && hasActivations) && (
          <div className="flex h-[300px] items-center justify-center px-6 text-center lg:h-[360px]">
            <p className="font-serif text-xl italic text-ink-3 sm:text-2xl">
              {!hasInput ? "Draw something to light this up" : error ? (
                <>
                  {error}.{" "}
                  <button type="button" className="underline" onClick={() => runAtEpoch(currentEpoch)}>
                    Retry
                  </button>
                </>
              ) : "Loading…"}
            </p>
          </div>
        )}
        <div
          className={
            hasInput && hasActivations
              ? "relative h-[460px] min-w-[720px] sm:h-[680px] lg:h-[780px]"
              : "absolute inset-0"
          }
          aria-hidden={!(hasInput && hasActivations)}
        >
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
            ) : null}
          </div>

          {hasInput && (
            <>
              <div className="pointer-events-none absolute left-4 top-4 z-10 max-sm:hidden font-mono text-[11px] text-ink-3">
                <p className="tracking-[0.08em]">PREDICTION</p>
                {topPrediction && (
                  <motion.div
                    key={topPrediction.label}
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
                    className="flex items-baseline gap-3"
                  >
                    <span className="font-serif text-5xl leading-none text-annotation">
                      {topPrediction.label}
                    </span>
                    <span className="text-xs tabular-nums text-ink">
                      {(topPrediction.confidence * 100).toFixed(1)}%
                    </span>
                  </motion.div>
                )}
                {isLoading && (
                  <span className="mt-1 inline-block size-1.5 rounded-full bg-phosphor" />
                )}
              </div>

              <div className="pointer-events-none absolute right-4 top-4 z-10 max-sm:hidden text-right font-mono text-[11px] text-ink-3">
                <p className="tracking-[0.08em]">EPOCH</p>
                <p className="text-3xl leading-none tabular-nums text-ink">
                  {String(currentEpoch).padStart(3, "0")}
                  <span className="ml-1 text-[11px] text-ink-3">/ {TOTAL_EPOCHS - 1}</span>
                </p>
              </div>
            </>
          )}
        </div>
      </div>

      {hasInput && hasActivations && (
        <p className="caption sm:hidden">Swipe sideways to pan the network</p>
      )}

      {/* Transport */}
      <div className="flex w-full items-center gap-4 max-sm:sticky max-sm:bottom-[env(safe-area-inset-bottom)] max-sm:z-10 max-sm:bg-bg/90 max-sm:py-2 max-sm:backdrop-blur">
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          disabled={!hasInput}
          className="btn-ghost size-11 shrink-0 !px-0 text-phosphor sm:size-9"
          aria-label={isPlaying ? "Pause" : "Play"}
        >
          {isPlaying ? (
            <svg width="12" height="12" viewBox="0 0 14 14" fill="currentColor" aria-hidden>
              <rect x="2" y="1" width="3.5" height="12" />
              <rect x="8.5" y="1" width="3.5" height="12" />
            </svg>
          ) : (
            <svg width="12" height="12" viewBox="0 0 14 14" fill="currentColor" aria-hidden>
              <path d="M3 1.5v11l9-5.5z" />
            </svg>
          )}
        </button>

        <div className="flex min-w-0 flex-1 flex-col gap-1">
          <input
            type="range"
            min={0}
            max={TOTAL_EPOCHS - 1}
            value={currentEpoch}
            onChange={(e) => handleEpochChange(parseInt(e.target.value))}
            className="w-full"
            style={{ ["--fill" as string]: `${pct}%` }}
            disabled={!hasInput}
            aria-label="Training epoch"
          />
          <div className="flex w-full justify-between gap-1">
            {[0, 5, 10, 20, 30, TOTAL_EPOCHS - 1].map((e) => (
              <button
                key={e}
                onClick={() => { handleEpochChange(e); setIsPlaying(false); }}
                data-selected={currentEpoch === e}
                className="chip !h-6 !min-h-0 !px-1.5 !text-[10.5px] max-sm:!min-h-11 max-sm:!min-w-11 max-sm:justify-center max-sm:[&:nth-child(2)]:hidden"
              >
                {e}
              </button>
            ))}
          </div>
        </div>
      </div>

      {error && (
        <p className="font-mono text-[11px] text-annotation">{error}</p>
      )}

      {!hasInput && (
        <p className="font-mono text-[11px] text-ink-3">
          Draw a character above to see how the model improves across training epochs
        </p>
      )}
    </div>
  );
}
