"use client";

import { useCallback, useEffect, useRef } from "react";
import { SectionWrapper } from "@/components/ui/SectionWrapper";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { ArchitectureBuilder } from "@/components/model-lab/ArchitectureBuilder";
import { TrainingControls } from "@/components/model-lab/TrainingControls";
import { TrainingChart } from "@/components/model-lab/TrainingChart";
import { ModelLabInference } from "@/components/model-lab/ModelLabInference";
import { useModelLabStore } from "@/stores/modelLabStore";
import { disposeModel } from "@/lib/model-lab/memoryManager";
import {
  registerCustomInfer,
  unregisterCustomInfer,
} from "@/lib/model-lab/customInferBridge";
import type * as ort from "onnxruntime-web";

// TF.js types — the actual import is dynamic
import type * as TF from "@tensorflow/tfjs";

export function ModelLabSection() {
  const store = useModelLabStore;
  const phase = useModelLabStore((s) => s.phase);
  const hasTrainedModel = useModelLabStore((s) => s.hasTrainedModel);

  // Refs for TF.js module and model (outside React state to avoid serialization)
  const tfRef = useRef<typeof TF | null>(null);
  const modelRef = useRef<TF.LayersModel | null>(null);
  const controllerRef = useRef<{ stop: () => void } | null>(null);
  const intermediateNamesRef = useRef<string[]>([]);

  // Refs for GPU-trained ONNX model
  const onnxSessionRef = useRef<ort.InferenceSession | null>(null);
  const gpuLayerNamesRef = useRef<string[]>([]);
  const gpuControllerRef = useRef<{ cancel: () => void } | null>(null);

  const loadTf = useCallback(async () => {
    if (tfRef.current) return tfRef.current;
    const tf = await import("@tensorflow/tfjs");
    await tf.ready();
    tfRef.current = tf;
    return tf;
  }, []);

  /** Reset training state shared by both modes. */
  const resetTrainingState = useCallback(() => {
    useModelLabStore.setState({
      trainingHistory: [],
      currentEpoch: 0,
      currentBatch: 0,
      totalBatches: 0,
      hasTrainedModel: false,
      intermediateLayerNames: [],
      customPrediction: null,
      customTopPrediction: null,
      customActivations: {},
      errorMessage: null,
      gpuStatus: null,
    });
  }, []);

  const handleBrowserTrain = useCallback(async () => {
    const state = store.getState();
    const { architecture, datasetType, learningRate, epochs, batchSize, optimizer } = state;

    // Dispose previous model
    if (modelRef.current && tfRef.current) {
      modelRef.current = disposeModel(tfRef.current, modelRef.current);
    }
    controllerRef.current = null;
    intermediateNamesRef.current = [];
    resetTrainingState();

    try {
      store.getState().setPhase("loading-data");

      const tf = await loadTf();

      const { loadDataset } = await import("@/lib/model-lab/dataLoader");
      const dataset = await loadDataset(datasetType);

      store.getState().setPhase("building");

      const { buildModel } = await import("@/lib/model-lab/buildModel");
      const { model, intermediateLayerNames } = buildModel(
        tf,
        architecture,
        dataset.numClasses,
      );

      modelRef.current = model;
      intermediateNamesRef.current = intermediateLayerNames;

      store.getState().setPhase("training");

      const { createTrainingController } = await import(
        "@/lib/model-lab/trainModel"
      );

      const controller = createTrainingController(tf, model, dataset, {
        learningRate,
        epochs,
        batchSize,
        optimizer,
      }, {
        onEpochEnd: (metrics) => {
          store.getState().addEpochMetrics(metrics);
          store.getState().setCurrentEpoch(metrics.epoch);
        },
        onBatchEnd: (batch, total) => {
          store.getState().setBatchProgress(batch, total);
        },
        onTrainingEnd: () => {
          store.getState().setTrainedModel(intermediateLayerNames);
          store.getState().setPhase("trained");
        },
      });

      controllerRef.current = controller;
      await controller.start();
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Training failed";
      store.getState().setErrorMessage(msg);
      store.getState().setPhase("error");
    }
  }, [loadTf, store, resetTrainingState]);

  /** Shared callbacks for server-side training (HF or RunPod). */
  const serverCallbacks = useCallback(() => ({
    onStatusChange: (status: string) => {
      store.getState().setGpuStatus(status);
    },
    onEpochEnd: (metrics: { epoch: number; loss: number; acc: number; valLoss: number; valAcc: number }) => {
      store.getState().setGpuStatus(null);
      store.getState().setPhase("training");
      store.getState().addEpochMetrics(metrics);
      store.getState().setCurrentEpoch(metrics.epoch);
    },
    onComplete: (session: ort.InferenceSession, layerNames: string[], _numClasses: number) => {
      const validNames = layerNames.filter((n) => n && n !== "output");
      onnxSessionRef.current = session;
      gpuLayerNamesRef.current = validNames;
      store.getState().setTrainedModel(validNames);
      store.getState().setPhase("trained");
      store.getState().setGpuStatus(null);
    },
    onError: (message: string) => {
      store.getState().setErrorMessage(message);
      store.getState().setPhase("error");
      store.getState().setGpuStatus(null);
    },
  }), [store]);

  const prepareServerTrain = useCallback(() => {
    const state = store.getState();
    onnxSessionRef.current = null;
    gpuLayerNamesRef.current = [];
    gpuControllerRef.current = null;
    resetTrainingState();
    store.getState().setPhase("loading-data");
    return state;
  }, [store, resetTrainingState]);

  const handleHfTrain = useCallback(async () => {
    const { architecture, datasetType, learningRate, epochs, batchSize, optimizer, maxSamples } = prepareServerTrain();

    const { createGpuTrainingController } = await import(
      "@/lib/model-lab/gpuTraining"
    );

    const controller = createGpuTrainingController(
      {
        architecture,
        training: { dataset: datasetType, learningRate, epochs, batchSize, optimizer, maxSamples },
      },
      serverCallbacks(),
    );

    gpuControllerRef.current = controller;
    await controller.start();
  }, [prepareServerTrain, serverCallbacks]);

  const handleRunpodTrain = useCallback(async () => {
    const { architecture, datasetType, learningRate, epochs, batchSize, optimizer, maxSamples } = prepareServerTrain();

    const { createRunpodTrainingController } = await import(
      "@/lib/model-lab/runpodTraining"
    );

    const controller = createRunpodTrainingController(
      {
        architecture,
        training: { dataset: datasetType, learningRate, epochs, batchSize, optimizer, maxSamples },
      },
      serverCallbacks(),
    );

    gpuControllerRef.current = controller;
    await controller.start();
  }, [prepareServerTrain, serverCallbacks]);

  const handleTrain = useCallback(() => {
    const mode = store.getState().trainingMode;
    if (mode === "gpu") {
      handleRunpodTrain();
    } else if (mode === "hf") {
      handleHfTrain();
    } else {
      handleBrowserTrain();
    }
  }, [handleBrowserTrain, handleHfTrain, handleRunpodTrain, store]);

  const handleStop = useCallback(() => {
    controllerRef.current?.stop();
    gpuControllerRef.current?.cancel();
  }, []);

  const handleReset = useCallback(() => {
    if (modelRef.current && tfRef.current) {
      modelRef.current = disposeModel(tfRef.current, modelRef.current);
    }
    controllerRef.current = null;
    intermediateNamesRef.current = [];
    onnxSessionRef.current = null;
    gpuLayerNamesRef.current = [];
    gpuControllerRef.current = null;
    store.getState().reset();
  }, [store]);

  const handleCustomInfer = useCallback(
    async (inputData: Float32Array) => {
      try {
        const mode = store.getState().trainingMode;
        const hasOnnx = !!onnxSessionRef.current;
        const hasTfModel = !!modelRef.current;
        const hasTf = !!tfRef.current;
        console.log("[model-lab] handleCustomInfer called", {
          mode,
          hasOnnx,
          hasTfModel,
          hasTf,
          inputLen: inputData.length,
        });

        if ((mode === "gpu" || mode === "hf") && onnxSessionRef.current) {
          const { runGpuModelInference } = await import(
            "@/lib/model-lab/gpuInference"
          );
          const result = await runGpuModelInference(
            onnxSessionRef.current,
            gpuLayerNamesRef.current,
            inputData,
          );
          console.log("[model-lab] GPU inference result:", result.prediction?.length, "classes");
          store.getState().setCustomPrediction(result.prediction);
          store.getState().setCustomActivations(result.layerActivations);
        } else if (modelRef.current && tfRef.current) {
          const { runCustomInference } = await import(
            "@/lib/model-lab/customInference"
          );
          const result = runCustomInference(
            tfRef.current,
            modelRef.current,
            intermediateNamesRef.current,
            inputData,
          );
          console.log("[model-lab] Browser inference result:", result.prediction?.length, "classes");
          store.getState().setCustomPrediction(result.prediction);
          store.getState().setCustomActivations(result.layerActivations);
        } else {
          console.warn("[model-lab] No model available for inference — neither branch matched");
        }
      } catch (e) {
        console.error("Custom model inference failed:", e);
      }
    },
    [store],
  );

  // Register the custom inference callback so the main inference pipeline
  // (useInference) can trigger it directly after each stroke.
  const handleCustomInferRef = useRef(handleCustomInfer);
  handleCustomInferRef.current = handleCustomInfer;
  useEffect(() => {
    registerCustomInfer(
      (tensor) => {
        if (!store.getState().hasTrainedModel) return;
        console.log("[model-lab] bridge: triggering custom inference");
        handleCustomInferRef.current(tensor).catch((e) => {
          console.error("[model-lab] bridge: custom inference failed:", e);
        });
      },
      () => {
        console.log("[model-lab] bridge: clearing custom predictions");
        store.getState().setCustomPrediction(null);
        store.getState().setCustomActivations({});
      },
    );
    return () => {
      unregisterCustomInfer();
    };
  }, [store]);

  return (
    <SectionWrapper id="model-lab" fullHeight={false}>
      <SectionHeader
        step={10}
        title="Model Lab"
        subtitle="Design your own CNN architecture, choose a dataset, and train it live in the browser. After training, draw characters to compare your model's predictions with the pre-trained model."
      />

      <div className="flex flex-col gap-8 lg:flex-row lg:gap-10">
        {/* Left column: Architecture + Training controls */}
        <div className="w-full space-y-6 lg:w-[380px] lg:shrink-0">
          <ArchitectureBuilder />
          <TrainingControls
            onTrain={handleTrain}
            onStop={handleStop}
            onReset={handleReset}
          />
        </div>

        {/* Right column: Charts + Inference */}
        <div className="min-w-0 flex-1 space-y-6">
          <TrainingChart />

          {(phase === "trained" || hasTrainedModel) && (
            <ModelLabInference />
          )}

          {phase === "idle" && (
            <div className="flex h-48 items-center justify-center rounded-xl border border-dashed border-border/30">
              <p className="text-sm text-foreground/25">
                Configure your architecture and hit Train to get started
              </p>
            </div>
          )}
        </div>
      </div>
    </SectionWrapper>
  );
}
