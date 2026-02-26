interface Conv1Data {
  weights: number[];
  biases: number[];
  shape: [number, number, number, number]; // [64, 1, 3, 3]
}

interface Conv1Weights {
  kernels: number[][][];
  biases: number[];
}

let cached: Conv1Weights | null = null;

/**
 * Fetch conv1 weights and reshape to [N][3][3] kernels.
 *
 * Kernels are transposed during reshape to match display-space convolution.
 * The EMNIST pipeline transposes the input before feeding it to the model,
 * and nchwToChannels un-transposes the output. To make manual convolution
 * on the display-space inputTensor match the model's conv1 output, we need
 * the kernel in its transposed form: kernel_display[r][c] = kernel_model[c][r].
 */
export async function fetchConv1Weights(): Promise<Conv1Weights> {
  if (cached) return cached;

  const res = await fetch("/models/combined-cnn/conv1-weights.json");
  const data: Conv1Data = await res.json();

  const [numFilters, , kH, kW] = data.shape;
  const kernels: number[][][] = [];

  for (let f = 0; f < numFilters; f++) {
    const kernel: number[][] = [];
    for (let r = 0; r < kH; r++) {
      const row: number[] = [];
      for (let c = 0; c < kW; c++) {
        // Transpose: read [c][r] instead of [r][c] to match display-space
        row.push(data.weights[f * kH * kW + c * kW + r]);
      }
      kernel.push(row);
    }
    kernels.push(kernel);
  }

  cached = { kernels, biases: data.biases };
  return cached;
}
