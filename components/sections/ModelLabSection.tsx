"use client";

import { useCallback, useEffect, useRef } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { SectionWrapper } from "@/components/ui/SectionWrapper";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { ArchitectureBuilder } from "@/components/model-lab/ArchitectureBuilder";
import { TrainingControls } from "@/components/model-lab/TrainingControls";
import { TrainingChart } from "@/components/model-lab/TrainingChart";
import { ExportPanel } from "@/components/model-lab/ExportPanel";
import { ModelLabInference } from "@/components/model-lab/ModelLabInference";
import { useModelLabStore } from "@/stores/modelLabStore";
import { useInferenceStore } from "@/stores/inferenceStore";
import { NetworkDiagram } from "@/components/model-lab/NetworkDiagram";
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
  const reduce = useReducedMotion();
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

  // Stacked layout: results move above the form, so bring them into view when training starts
  useEffect(() => {
    if (phase === "idle" || !window.matchMedia("(max-width: 1023px)").matches) return;
    chartContainerRef.current?.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
  }, [phase === "idle"]); // eslint-disable-line react-hooks/exhaustive-deps

  const loadTf = useCallback(async () => {
    if (tfRef.current) return tfRef.current;
    const tf = await import("@tensorflow/tfjs");
    await tf.ready();
    tfRef.current = tf;
    return tf;
  }, []);

  /**
   * After training completes, auto-run custom inference if the user already
   * has a drawing on the canvas so predictions appear immediately.
   */
  const inferExistingDrawing = useCallback(() => {
    const imageData = useInferenceStore.getState().inputImageData;
    if (!imageData) return;
    import("@/lib/model/preprocess").then(({ preprocessCanvas }) => {
      const { tensor } = preprocessCanvas(imageData);
      handleCustomInferRef.current(tensor).catch(console.error);
    });
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
    onComplete: (session: ort.InferenceSession, layerNames: string[], _numClasses: number, bytesOrUrl: Uint8Array | string) => {
      const validNames = layerNames.filter((n) => n && n !== "output");
      onnxSessionRef.current = session;
      gpuLayerNamesRef.current = validNames;

      // Store bytes for export
      if (bytesOrUrl instanceof Uint8Array) {
        onnxBytesRef.current = bytesOrUrl;
      } else {
        // URL case (RunPod) — eagerly fetch and cache as bytes before URL expires
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
    onnxBytesRef.current = null;
    onnxModelUrlRef.current = null;
    store.getState().reset();
  }, [store]);

  const handleExportModel = useCallback(async () => {
    if (onnxBytesRef.current) {
      const { downloadModelWeights } = await import("@/lib/model-lab/exportUtils");
      downloadModelWeights(onnxBytesRef.current);
    } else if (modelRef.current) {
      // TF.js browser model — use built-in save to downloads
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

  const trained = phase === "trained" || hasTrainedModel;
  const status =
    phase === "idle"
      ? "IDLE"
      : phase === "trained"
        ? "TRAINED"
        : phase === "error"
          ? "ERROR"
          : "RUNNING";

  return (
    <SectionWrapper id="model-lab" fullHeight={false} sig="lab">
      <SectionHeader
        step={10}
        wide
        title="Model Lab"
        tag={`Workspace · ${status}`}
        subtitle="Design your own CNN architecture, choose a dataset, and train it live in the browser. After training, draw characters to compare your model's predictions with the pre-trained model."
      />

      <motion.div
        initial={reduce ? false : { opacity: 0, y: 16 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-8% 0px" }}
        transition={{ duration: reduce ? 0.15 : 0.5, ease: [0.16, 1, 0.3, 1] }}
        className="grid gap-8 sm:gap-10 lg:grid-cols-12 lg:gap-x-10"
      >
        {/* Left: builder + controls */}
        <div className="min-w-0 space-y-10 lg:col-span-5 xl:col-span-4">
          <p className="caption sm:hidden">Configure layers, then press Train.</p>
          <ArchitectureBuilder />
          <TrainingControls
            onTrain={handleTrain}
            onStop={handleStop}
            onReset={handleReset}
          />
        </div>

        {/* Right: results */}
        <div className={`min-w-0 space-y-10 self-start lg:col-span-7 xl:sticky xl:top-8 xl:col-span-8 ${phase !== "idle" ? "max-lg:order-first" : ""}`}>
          {trained && (
            <ExportPanel
              onExportModel={handleExportModel}
              onExportReport={handleExportReport}
              onExportChart={handleExportChart}
              modelFormat={trainingMode === "browser" ? "TF.js" : "ONNX"}
            />
          )}

          <TrainingChart ref={chartContainerRef} />

          {trained && <ModelLabInference />}

          {phase === "idle" && (
            <div className="overflow-x-auto scrollbar-none">
              <NetworkDiagram />
            </div>
          )}

          {phase === "idle" && !hasTrainedModel && (
            <div className="viz-empty-state min-h-[120px] max-sm:hidden">
              Press Train to watch loss and accuracy fall, epoch by epoch
            </div>
          )}
        </div>
      </motion.div>
    </SectionWrapper>
  );
}
