"use client";

import { useCallback, useEffect, useRef } from "react";
import { useModelLabStore } from "@/stores/modelLabStore";
import { useInferenceStore } from "@/stores/inferenceStore";
import { disposeModel } from "@/lib/model-lab/memoryManager";
import {
  registerCustomInfer,
  unregisterCustomInfer,
} from "@/lib/model-lab/customInferBridge";
import type * as ort from "onnxruntime-web";
import type * as TF from "@tensorflow/tfjs";

/**
 * Shared hook that encapsulates all Model Lab training logic, refs, and callbacks.
 * Used by all Model Lab layout variants so they can focus purely on UI.
 */
export function useModelLab() {
  const store = useModelLabStore;
  const phase = useModelLabStore((s) => s.phase);
  const hasTrainedModel = useModelLabStore((s) => s.hasTrainedModel);
  const trainingMode = useModelLabStore((s) => s.trainingMode);

  // Refs for TF.js module and model (outside React state to avoid serialization)
  const tfRef = useRef<typeof TF | null>(null);
  const modelRef = useRef<TF.LayersModel | null>(null);
  const controllerRef = useRef<{ stop: () => void } | null>(null);
  const intermediateNamesRef = useRef<string[]>([]);

  // Refs for GPU-trained ONNX model
  const onnxSessionRef = useRef<ort.InferenceSession | null>(null);
  const gpuLayerNamesRef = useRef<string[]>([]);
  const gpuControllerRef = useRef<{ cancel: () => void } | null>(null);

  // Refs for export — preserve ONNX bytes for download
  const onnxBytesRef = useRef<Uint8Array | null>(null);
  const onnxModelUrlRef = useRef<string | null>(null);
  const chartContainerRef = useRef<HTMLDivElement>(null);

  const loadTf = useCallback(async () => {
    if (tfRef.current) return tfRef.current;
    const tf = await import("@tensorflow/tfjs");
    await tf.ready();
    tfRef.current = tf;
    return tf;
  }, []);

  const inferExistingDrawing = useCallback(() => {
    const imageData = useInferenceStore.getState().inputImageData;
    if (!imageData) return;
    import("@/lib/model/preprocess").then(({ preprocessCanvas }) => {
      const { tensor } = preprocessCanvas(imageData);
      handleCustomInferRef.current(tensor).catch(console.error);
    });
  }, []);

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
      const { model, intermediateLayerNames } = buildModel(tf, architecture, dataset.numClasses);

      modelRef.current = model;
      intermediateNamesRef.current = intermediateLayerNames;
      store.getState().setPhase("training");

      const { createTrainingController } = await import("@/lib/model-lab/trainModel");
      const controller = createTrainingController(tf, model, dataset, {
        learningRate, epochs, batchSize, optimizer,
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
          inferExistingDrawing();
        },
      });

      controllerRef.current = controller;
      await controller.start();
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Training failed";
      store.getState().setErrorMessage(msg);
      store.getState().setPhase("error");
    }
  }, [loadTf, store, resetTrainingState, inferExistingDrawing]);

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
    onComplete: (session: ort.InferenceSession, layerNames: string[], _numClasses: number, bytesOrUrl: Uint8Array | string) => {
      const validNames = layerNames.filter((n) => n && n !== "output");
      onnxSessionRef.current = session;
      gpuLayerNamesRef.current = validNames;

      if (bytesOrUrl instanceof Uint8Array) {
        onnxBytesRef.current = bytesOrUrl;
      } else {
        onnxModelUrlRef.current = bytesOrUrl;
        fetch(bytesOrUrl)
          .then((r) => r.arrayBuffer())
          .then((buf) => { onnxBytesRef.current = new Uint8Array(buf); })
          .catch(console.error);
      }

      store.getState().setTrainedModel(validNames);
      store.getState().setPhase("trained");
      store.getState().setGpuStatus(null);
      inferExistingDrawing();
    },
    onError: (message: string) => {
      store.getState().setErrorMessage(message);
      store.getState().setPhase("error");
      store.getState().setGpuStatus(null);
    },
  }), [store, inferExistingDrawing]);

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
    const { createGpuTrainingController } = await import("@/lib/model-lab/gpuTraining");
    const controller = createGpuTrainingController(
      { architecture, training: { dataset: datasetType, learningRate, epochs, batchSize, optimizer, maxSamples } },
      serverCallbacks(),
    );
    gpuControllerRef.current = controller;
    await controller.start();
  }, [prepareServerTrain, serverCallbacks]);

  const handleRunpodTrain = useCallback(async () => {
    const { architecture, datasetType, learningRate, epochs, batchSize, optimizer, maxSamples } = prepareServerTrain();
    const { createRunpodTrainingController } = await import("@/lib/model-lab/runpodTraining");
    const controller = createRunpodTrainingController(
      { architecture, training: { dataset: datasetType, learningRate, epochs, batchSize, optimizer, maxSamples } },
      serverCallbacks(),
    );
    gpuControllerRef.current = controller;
    await controller.start();
  }, [prepareServerTrain, serverCallbacks]);

  const handleTrain = useCallback(() => {
    const mode = store.getState().trainingMode;
    if (mode === "gpu") handleRunpodTrain();
    else if (mode === "hf") handleHfTrain();
    else handleBrowserTrain();
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
    onnxBytesRef.current = null;
    onnxModelUrlRef.current = null;
    store.getState().reset();
  }, [store]);

  const handleExportModel = useCallback(async () => {
    if (onnxBytesRef.current) {
      const { downloadModelWeights } = await import("@/lib/model-lab/exportUtils");
      downloadModelWeights(onnxBytesRef.current);
    } else if (modelRef.current) {
      await modelRef.current.save("downloads://model");
    }
  }, []);

  const handleExportReport = useCallback(async () => {
    const state = store.getState();
    const { downloadTrainingReport } = await import("@/lib/model-lab/exportUtils");
    downloadTrainingReport({
      architecture: state.architecture,
      dataset: state.datasetType,
      learningRate: state.learningRate,
      epochs: state.epochs,
      batchSize: state.batchSize,
      optimizer: state.optimizer,
      maxSamples: state.maxSamples,
      trainingMode: state.trainingMode,
      history: state.trainingHistory,
    });
  }, [store]);

  const handleExportChart = useCallback(async () => {
    if (!chartContainerRef.current) return;
    const { downloadChartAsPng } = await import("@/lib/model-lab/exportUtils");
    await downloadChartAsPng(chartContainerRef.current);
  }, []);

  const handleCustomInfer = useCallback(
    async (inputData: Float32Array) => {
      try {
        const mode = store.getState().trainingMode;
        const hasOnnx = !!onnxSessionRef.current;
        const hasTfModel = !!modelRef.current;
        const hasTf = !!tfRef.current;
        console.log("[model-lab] handleCustomInfer called", {
          mode, hasOnnx, hasTfModel, hasTf, inputLen: inputData.length,
        });

        if ((mode === "gpu" || mode === "hf") && onnxSessionRef.current) {
          const { runGpuModelInference } = await import("@/lib/model-lab/gpuInference");
          const result = await runGpuModelInference(
            onnxSessionRef.current, gpuLayerNamesRef.current, inputData,
          );
          store.getState().setCustomPrediction(result.prediction);
          store.getState().setCustomActivations(result.layerActivations);
        } else if (modelRef.current && tfRef.current) {
          const { runCustomInference } = await import("@/lib/model-lab/customInference");
          const result = runCustomInference(
            tfRef.current, modelRef.current, intermediateNamesRef.current, inputData,
          );
          store.getState().setCustomPrediction(result.prediction);
          store.getState().setCustomActivations(result.layerActivations);
        }
      } catch (e) {
        console.error("Custom model inference failed:", e);
      }
    },
    [store],
  );

  const handleCustomInferRef = useRef(handleCustomInfer);
  handleCustomInferRef.current = handleCustomInfer;

  useEffect(() => {
    registerCustomInfer(
      (tensor) => {
        if (!store.getState().hasTrainedModel) return;
        handleCustomInferRef.current(tensor).catch(console.error);
      },
      () => {
        store.getState().setCustomPrediction(null);
        store.getState().setCustomActivations({});
      },
    );
    return () => {
      unregisterCustomInfer();
    };
  }, [store]);

  return {
    // State
    phase,
    hasTrainedModel,
    trainingMode,
    // Actions
    handleTrain,
    handleStop,
    handleReset,
    handleExportModel,
    handleExportReport,
    handleExportChart,
    // Refs
    chartContainerRef,
  };
}
