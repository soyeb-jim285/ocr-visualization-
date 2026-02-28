/**
 * GPU training client.
 * Sends config to /api/gpu-train (Next.js proxy), reads SSE stream.
 * Trained ONNX model is downloaded from HuggingFace after training completes.
 */

import * as ort from "onnxruntime-web";
import type { ArchitectureConfig } from "./architecture";
import type { DatasetType } from "./dataLoader";
import type { OptimizerType, EpochMetrics } from "./trainModel";

export interface RunpodTrainingConfig {
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

export interface RunpodTrainingCallbacks {
  onEpochEnd: (metrics: EpochMetrics) => void;
  onComplete: (
    session: ort.InferenceSession,
    layerNames: string[],
    numClasses: number,
    modelUrl: string,
  ) => void;
  onError: (message: string) => void;
  onStatusChange: (status: string) => void;
}

export interface RunpodTrainingController {
  start: () => Promise<void>;
  cancel: () => void;
}

export function createRunpodTrainingController(
  config: RunpodTrainingConfig,
  callbacks: RunpodTrainingCallbacks,
): RunpodTrainingController {
  let abortController: AbortController | null = null;

  const start = async () => {
    try {
      callbacks.onStatusChange("Submitting to GPU...");

      abortController = new AbortController();

      // Strip frontend-only 'id' from conv layers
      const payload = {
        architecture: {
          convLayers: config.architecture.convLayers.map(
            ({ id: _id, ...rest }) => rest,
          ),
          dense: config.architecture.dense,
        },
        training: config.training,
      };

      const res = await fetch("/api/gpu-train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: abortController.signal,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: res.statusText }));
        callbacks.onError(err.error || `HTTP ${res.status}`);
        return;
      }

      callbacks.onStatusChange("Training on GPU...");

      // Read SSE stream
      const reader = res.body?.getReader();
      if (!reader) {
        callbacks.onError("No response stream");
        return;
      }

      const decoder = new TextDecoder();
      let buffer = "";

      const processSSEChunks = async (chunks: string[]) => {
        for (const chunk of chunks) {
          const dataLine = chunk
            .split("\n")
            .find((l) => l.startsWith("data: "));
          if (!dataLine) continue;

          const jsonStr = dataLine.slice(6); // strip "data: "
          let parsed;
          try {
            parsed = JSON.parse(jsonStr);
          } catch {
            continue;
          }

          if (parsed.type === "status") {
            callbacks.onStatusChange(parsed.message);
          } else if (parsed.type === "epoch") {
            callbacks.onStatusChange("");
            callbacks.onEpochEnd({
              epoch: parsed.epoch,
              loss: parsed.loss,
              acc: parsed.acc,
              valLoss: parsed.valLoss,
              valAcc: parsed.valAcc,
            });
          } else if (parsed.type === "complete") {
            console.log(
              `[gpu] Complete. cached=${parsed.cached}, layers: ${parsed.layerNames}`,
            );
            try {
              callbacks.onStatusChange(
                parsed.cached ? "Loading cached model..." : "Downloading model...",
              );
              console.log(`[gpu] Fetching ONNX from ${parsed.modelUrl}`);
              const session = await ort.InferenceSession.create(parsed.modelUrl);
              console.log(`[gpu] ONNX session created`);
              callbacks.onComplete(
                session,
                parsed.layerNames,
                parsed.numClasses,
                parsed.modelUrl,
              );
            } catch (onnxErr) {
              console.error(`[gpu] Model load failed:`, onnxErr);
              callbacks.onError(
                `Model loading failed: ${onnxErr instanceof Error ? onnxErr.message : onnxErr}`,
              );
            }
          } else if (parsed.type === "error") {
            callbacks.onError(parsed.message);
          }
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Parse SSE messages (data: {...}\n\n)
        const lines = buffer.split("\n\n");
        buffer = lines.pop() ?? "";
        await processSSEChunks(lines);
      }

      // Process any remaining data left in the buffer after stream closes
      if (buffer.trim()) {
        console.log(`[gpu] Processing remaining buffer (${buffer.length} chars)`);
        await processSSEChunks(buffer.split("\n\n").filter(Boolean));
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      callbacks.onError(
        err instanceof Error ? err.message : "GPU training failed",
      );
    }
  };

  const cancel = () => {
    abortController?.abort();
  };

  return { start, cancel };
}
