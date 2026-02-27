import type * as tf from "@tensorflow/tfjs";
import type { LoadedDataset } from "./dataLoader";

export type OptimizerType = "adam" | "sgd" | "rmsprop";

export interface TrainingConfig {
  learningRate: number;
  epochs: number;
  batchSize: number;
  optimizer: OptimizerType;
}

export interface EpochMetrics {
  epoch: number;
  loss: number;
  acc: number;
  valLoss: number;
  valAcc: number;
}

export interface TrainingCallbacks {
  onEpochEnd: (metrics: EpochMetrics) => void;
  onBatchEnd: (batch: number, totalBatches: number) => void;
  onTrainingEnd: () => void;
}

export interface TrainingController {
  start: () => Promise<void>;
  stop: () => void;
  isRunning: () => boolean;
}

function createOptimizer(
  tf: typeof import("@tensorflow/tfjs"),
  config: TrainingConfig,
): tf.Optimizer {
  switch (config.optimizer) {
    case "adam":
      return tf.train.adam(config.learningRate);
    case "sgd":
      return tf.train.sgd(config.learningRate);
    case "rmsprop":
      return tf.train.rmsprop(config.learningRate);
  }
}

/** Create tensors from flat Float32Array + Uint8Array dataset. */
function createTensors(
  tf: typeof import("@tensorflow/tfjs"),
  images: Float32Array,
  labels: Uint8Array,
  numSamples: number,
  numClasses: number,
): { xs: tf.Tensor4D; ys: tf.Tensor2D } {
  // Reshape flat images into [N, 28, 28, 1] (NHWC)
  const xs = tf.tensor4d(images, [numSamples, 28, 28, 1]);
  const ys = tf.oneHot(tf.tensor1d(Array.from(labels), "int32"), numClasses).toFloat() as tf.Tensor2D;
  return { xs, ys };
}

export function createTrainingController(
  tf: typeof import("@tensorflow/tfjs"),
  model: tf.LayersModel,
  dataset: LoadedDataset,
  config: TrainingConfig,
  callbacks: TrainingCallbacks,
): TrainingController {
  let running = false;

  const stop = () => {
    model.stopTraining = true;
  };

  const start = async () => {
    running = true;

    const optimizer = createOptimizer(tf, config);
    model.compile({
      optimizer,
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    });

    // Create training tensors
    const { xs: trainXs, ys: trainYs } = createTensors(
      tf,
      dataset.trainImages,
      dataset.trainLabels,
      dataset.numTrain,
      dataset.numClasses,
    );

    // Create validation tensors
    const { xs: valXs, ys: valYs } = createTensors(
      tf,
      dataset.testImages,
      dataset.testLabels,
      dataset.numTest,
      dataset.numClasses,
    );

    const totalBatches = Math.ceil(dataset.numTrain / config.batchSize);

    try {
      await model.fit(trainXs, trainYs, {
        epochs: config.epochs,
        batchSize: config.batchSize,
        validationData: [valXs, valYs],
        shuffle: true,
        yieldEvery: "batch",
        callbacks: {
          onBatchEnd: (batch) => {
            callbacks.onBatchEnd(batch + 1, totalBatches);
          },
          onEpochEnd: (epoch, logs) => {
            callbacks.onEpochEnd({
              epoch: epoch + 1,
              loss: logs?.loss ?? 0,
              acc: logs?.acc ?? 0,
              valLoss: logs?.val_loss ?? 0,
              valAcc: logs?.val_acc ?? 0,
            });
          },
        },
      });
    } finally {
      // Dispose training tensors
      trainXs.dispose();
      trainYs.dispose();
      valXs.dispose();
      valYs.dispose();

      running = false;
      callbacks.onTrainingEnd();
    }
  };

  return {
    start,
    stop,
    isRunning: () => running,
  };
}
