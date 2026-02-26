import { BYMERGE_MERGED_INDICES } from "./classes";

/** Names of the intermediate outputs from the ONNX model (must match export) */
export const INTERMEDIATE_OUTPUTS = [
  "conv1",
  "relu1",
  "conv2",
  "relu2",
  "pool1",
  "conv3",
  "relu3",
  "pool2",
  "dense1",
  "relu4",
  "output",
] as const;

/** Layer output shapes (NCHW for conv, N*units for dense) */
export const LAYER_SHAPES: Record<
  string,
  { type: "conv"; c: number; h: number; w: number } | { type: "dense"; units: number }
> = {
  conv1: { type: "conv", c: 64, h: 28, w: 28 },
  relu1: { type: "conv", c: 64, h: 28, w: 28 },
  conv2: { type: "conv", c: 128, h: 28, w: 28 },
  relu2: { type: "conv", c: 128, h: 28, w: 28 },
  pool1: { type: "conv", c: 128, h: 14, w: 14 },
  conv3: { type: "conv", c: 256, h: 14, w: 14 },
  relu3: { type: "conv", c: 256, h: 14, w: 14 },
  pool2: { type: "conv", c: 256, h: 7, w: 7 },
  dense1: { type: "dense", units: 512 },
  relu4: { type: "dense", units: 512 },
  output: { type: "dense", units: 146 },
};

/**
 * Convert a flat Float32Array from ONNX (NCHW) to [C][H][W] for visualization.
 * Also transposes each channel (swap H/W) so the display orientation matches
 * how the user drew the character.
 */
export function nchwToChannels(
  data: Float32Array,
  c: number,
  h: number,
  w: number
): number[][][] {
  const channels: number[][][] = [];
  for (let ci = 0; ci < c; ci++) {
    const channel: number[][] = [];
    for (let hi = 0; hi < h; hi++) {
      const row: number[] = [];
      for (let wi = 0; wi < w; wi++) {
        // Transpose: read [wi][hi] instead of [hi][wi]
        row.push(data[ci * h * w + wi * w + hi]);
      }
      channel.push(row);
    }
    channels.push(channel);
  }
  return channels;
}

/**
 * Softmax over a Float32Array, masking out untrained ByMerge indices.
 * Single-pass implementation.
 */
export function softmax(arr: Float32Array): number[] {
  const n = arr.length;
  let max = -Infinity;
  for (let i = 0; i < n; i++) {
    if (!BYMERGE_MERGED_INDICES.has(i) && arr[i] > max) max = arr[i];
  }
  const result = new Array<number>(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    if (BYMERGE_MERGED_INDICES.has(i)) {
      result[i] = 0;
    } else {
      const e = Math.exp(arr[i] - max);
      result[i] = e;
      sum += e;
    }
  }
  if (sum > 0) {
    for (let i = 0; i < n; i++) result[i] /= sum;
  }
  return result;
}
