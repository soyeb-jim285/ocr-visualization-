/**
 * Export utilities for Model Lab — download trained model, training report, and chart PNG.
 */

import type { ArchitectureConfig } from "./architecture";
import type { EpochMetrics, OptimizerType } from "./trainModel";
import type { DatasetType } from "./dataLoader";

function triggerDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Download ONNX model weights as a .onnx file.
 */
export function downloadModelWeights(onnxBytes: Uint8Array) {
  const blob = new Blob([onnxBytes.buffer as ArrayBuffer], { type: "application/octet-stream" });
  triggerDownload(blob, "model.onnx");
}

/**
 * Download a JSON training report with architecture, hyperparams, and epoch history.
 */
export function downloadTrainingReport(data: {
  architecture: ArchitectureConfig;
  dataset: DatasetType;
  learningRate: number;
  epochs: number;
  batchSize: number;
  optimizer: OptimizerType;
  maxSamples: number;
  trainingMode: string;
  history: EpochMetrics[];
}) {
  const { history } = data;
  const lastEpoch = history[history.length - 1];

  const report = {
    architecture: {
      convLayers: data.architecture.convLayers.map(
        ({ id: _id, ...rest }) => rest,
      ),
      dense: data.architecture.dense,
    },
    training: {
      dataset: data.dataset,
      learningRate: data.learningRate,
      epochs: data.epochs,
      batchSize: data.batchSize,
      optimizer: data.optimizer,
      maxSamples: data.maxSamples,
      mode: data.trainingMode,
    },
    history,
    finalMetrics: lastEpoch
      ? {
          loss: lastEpoch.loss,
          accuracy: lastEpoch.acc,
          valLoss: lastEpoch.valLoss,
          valAccuracy: lastEpoch.valAcc,
        }
      : null,
    metadata: {
      exportedAt: new Date().toISOString(),
      source: "ocr-visualization-model-lab",
    },
  };

  const blob = new Blob([JSON.stringify(report, null, 2)], {
    type: "application/json",
  });
  triggerDownload(blob, "training-report.json");
}

/**
 * Render a DOM element to a PNG image and trigger download.
 * Uses html-to-image which handles SVG content (Recharts) correctly.
 */
export async function downloadChartAsPng(
  element: HTMLElement,
  filename = "training-chart.png",
) {
  const { toBlob } = await import("html-to-image");
  const blob = await toBlob(element, {
    pixelRatio: 2,
    backgroundColor: "#0a0a0f",
    skipFonts: true,
  });
  if (blob) triggerDownload(blob, filename);
}
