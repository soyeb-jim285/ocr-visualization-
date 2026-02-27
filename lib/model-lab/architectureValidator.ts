import type { ArchitectureConfig } from "./architecture";

export interface ValidationResult {
  spatialDims: { width: number; height: number }[];
  paramCount: number;
  errors: string[];
  warnings: string[];
}

/** Track spatial dimensions and param count through the architecture. */
export function validateArchitecture(
  config: ArchitectureConfig,
  numClasses: number,
): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];
  const spatialDims: { width: number; height: number }[] = [];

  let h = 28;
  let w = 28;
  let channels = 1;
  let paramCount = 0;

  spatialDims.push({ width: w, height: h });

  if (config.convLayers.length === 0) {
    errors.push("At least one convolutional layer is required");
  }

  for (let i = 0; i < config.convLayers.length; i++) {
    const layer = config.convLayers[i];
    const k = layer.kernelSize;

    // Conv2D with 'same' padding preserves spatial dims
    const convParams = layer.filters * (channels * k * k + 1); // weights + bias
    paramCount += convParams;

    if (layer.batchNorm) {
      paramCount += layer.filters * 4; // gamma, beta, moving_mean, moving_var
    }

    channels = layer.filters;

    // Pooling halves spatial dims
    if (layer.pooling !== "none") {
      h = Math.floor(h / 2);
      w = Math.floor(w / 2);
    }

    spatialDims.push({ width: w, height: h });

    if (h < 1 || w < 1) {
      errors.push(
        `Layer ${i + 1}: spatial dimensions reduced to ${h}×${w} (too small)`,
      );
      break;
    }

    if (h < 2 && w < 2 && i < config.convLayers.length - 1) {
      // Check if any remaining layers have pooling
      const hasMorePooling = config.convLayers
        .slice(i + 1)
        .some((l) => l.pooling !== "none");
      if (hasMorePooling) {
        errors.push(
          `Layer ${i + 1}: spatial dims are ${h}×${w}, subsequent pooling will fail`,
        );
      }
    }
  }

  // Flatten → Dense
  const flattenSize = h * w * channels;
  const denseParams = flattenSize * config.dense.width + config.dense.width; // weights + bias
  paramCount += denseParams;

  // Output layer
  const outputParams = config.dense.width * numClasses + numClasses;
  paramCount += outputParams;

  if (paramCount > 5_000_000) {
    warnings.push(
      `Model has ${(paramCount / 1e6).toFixed(1)}M parameters — training may be slow in-browser`,
    );
  }

  if (paramCount > 20_000_000) {
    errors.push(
      `Model has ${(paramCount / 1e6).toFixed(1)}M parameters — too large for browser training`,
    );
  }

  return { spatialDims, paramCount, errors, warnings };
}
