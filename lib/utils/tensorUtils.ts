/**
 * Transpose a 3D array from [height][width][channels] to [channels][height][width].
 * TF.js outputs conv layers as [h, w, c] but we want [c, h, w] for visualization
 * (each channel is a separate feature map to display as a heatmap).
 */
export function transposeHWCtoCHW(
  data: number[][][],
  height: number,
  width: number,
  channels: number
): number[][][] {
  const result: number[][][] = [];
  for (let c = 0; c < channels; c++) {
    const channel: number[][] = [];
    for (let h = 0; h < height; h++) {
      const row: number[] = [];
      for (let w = 0; w < width; w++) {
        row.push(data[h][w][c]);
      }
      channel.push(row);
    }
    result.push(channel);
  }
  return result;
}

/** Get the min and max values from a 2D array */
export function getMinMax2D(data: number[][]): { min: number; max: number } {
  let min = Infinity;
  let max = -Infinity;
  for (const row of data) {
    for (const val of row) {
      if (val < min) min = val;
      if (val > max) max = val;
    }
  }
  return { min, max };
}

/** Get the min and max values from a 1D array */
export function getMinMax1D(data: number[]): { min: number; max: number } {
  let min = Infinity;
  let max = -Infinity;
  for (const val of data) {
    if (val < min) min = val;
    if (val > max) max = val;
  }
  return { min, max };
}

/** Get top-K indices from an array (sorted by value descending) */
export function topK(arr: number[], k: number): { index: number; value: number }[] {
  return arr
    .map((value, index) => ({ index, value }))
    .sort((a, b) => b.value - a.value)
    .slice(0, k);
}
