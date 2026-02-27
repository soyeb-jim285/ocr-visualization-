/**
 * GPU training client — communicates with the HuggingFace Space Gradio app.
 * Streams epoch metrics via SSE and returns an ONNX InferenceSession on completion.
 */

import type { Client as GradioClient } from "@gradio/client";
import * as ort from "onnxruntime-web";
import type { ArchitectureConfig } from "./architecture";
import type { DatasetType } from "./dataLoader";
import type { OptimizerType, EpochMetrics } from "./trainModel";

const HF_SPACE = "soyeb-jim285/ocr-model-lab";

export interface GpuTrainingConfig {
  architecture: ArchitectureConfig;
  training: {
    dataset: DatasetType;
    learningRate: number;
    epochs: number;
    batchSize: number;
    optimizer: OptimizerType;
    maxSamples: number;
  };
}

export interface GpuTrainingCallbacks {
  onEpochEnd: (metrics: EpochMetrics) => void;
  onComplete: (
    session: ort.InferenceSession,
    layerNames: string[],
    numClasses: number,
  ) => void;
  onError: (message: string) => void;
  onStatusChange: (status: string) => void;
}

export interface GpuTrainingController {
  start: () => Promise<void>;
  cancel: () => void;
}

export function createGpuTrainingController(
  config: GpuTrainingConfig,
  callbacks: GpuTrainingCallbacks,
): GpuTrainingController {
  let cancelled = false;

  const start = async () => {
    try {
      callbacks.onStatusChange("Connecting to GPU server...");

      const { Client } = await import("@gradio/client");
      const app: GradioClient = await Client.connect(HF_SPACE);

      // Serialize config — strip frontend-only 'id' fields from convLayers
      const payload = {
        architecture: {
          convLayers: config.architecture.convLayers.map(
            ({ id: _id, ...rest }) => rest,
          ),
          dense: config.architecture.dense,
        },
        training: config.training,
      };

      callbacks.onStatusChange("Queued for GPU...");

      const submission = app.submit("/train", {
        config_json: JSON.stringify(payload),
      });

      for await (const msg of submission) {
        if (cancelled) {
          submission.cancel();
          return;
        }

        if (msg.type === "status") {
          const status = msg as { stage?: string; position?: number };
          if (status.stage === "pending") {
            callbacks.onStatusChange(
              `In queue${status.position != null ? ` (position ${status.position})` : ""}...`,
            );
          } else if (status.stage === "generating") {
            callbacks.onStatusChange("Training on GPU...");
          }
          continue;
        }

        if (msg.type === "data") {
          // Gradio wraps generator outputs in an array
          const rawData = (msg as { data?: unknown[] }).data;
          const jsonStr = Array.isArray(rawData) ? rawData[0] : rawData;
          if (typeof jsonStr !== "string") continue;

          const parsed = JSON.parse(jsonStr);

          if (parsed.type === "epoch") {
            callbacks.onEpochEnd({
              epoch: parsed.epoch,
              loss: parsed.loss,
              acc: parsed.acc,
              valLoss: parsed.valLoss,
              valAcc: parsed.valAcc,
            });
          } else if (parsed.type === "complete") {
            // Decode base64 ONNX → ArrayBuffer → InferenceSession
            const binaryStr = atob(parsed.onnxBase64);
            const bytes = new Uint8Array(binaryStr.length);
            for (let i = 0; i < binaryStr.length; i++) {
              bytes[i] = binaryStr.charCodeAt(i);
            }

            const session = await ort.InferenceSession.create(bytes.buffer);
            callbacks.onComplete(
              session,
              parsed.layerNames,
              parsed.numClasses,
            );
          } else if (parsed.type === "error") {
            callbacks.onError(parsed.message);
          }
        }
      }
    } catch (err) {
      if (!cancelled) {
        callbacks.onError(
          err instanceof Error ? err.message : "GPU training failed",
        );
      }
    }
  };

  const cancel = () => {
    cancelled = true;
  };

  return { start, cancel };
}
