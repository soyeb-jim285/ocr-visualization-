/** Apply ReLU: max(0, x) */
export function relu(x: number): number {
  return Math.max(0, x);
}

/** Apply ReLU to a 2D array */
export function relu2D(data: number[][]): number[][] {
  return data.map((row) => row.map((val) => Math.max(0, val)));
}

/** Apply softmax to a 1D array */
export function softmax(logits: number[]): number[] {
  const maxLogit = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - maxLogit));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sumExps);
}

/**
 * Compute a single convolution step (dot product of kernel and input patch).
 * Used for the step-through animation.
 */
export function convolveStep(
  input: number[][],
  kernel: number[][],
  row: number,
  col: number
): number {
  const kSize = kernel.length;
  let sum = 0;
  for (let kr = 0; kr < kSize; kr++) {
    for (let kc = 0; kc < kSize; kc++) {
      const ir = row + kr;
      const ic = col + kc;
      if (ir >= 0 && ir < input.length && ic >= 0 && ic < input[0].length) {
        sum += input[ir][ic] * kernel[kr][kc];
      }
    }
  }
  return sum;
}

/**
 * Apply 2x2 max pooling to a 2D array.
 * Returns a new array with half the dimensions.
 */
export function maxPool2x2(data: number[][]): number[][] {
  const rows = data.length;
  const cols = data[0].length;
  const outRows = Math.floor(rows / 2);
  const outCols = Math.floor(cols / 2);
  const result: number[][] = [];

  for (let r = 0; r < outRows; r++) {
    const row: number[] = [];
    for (let c = 0; c < outCols; c++) {
      const r2 = r * 2;
      const c2 = c * 2;
      row.push(
        Math.max(data[r2][c2], data[r2][c2 + 1], data[r2 + 1][c2], data[r2 + 1][c2 + 1])
      );
    }
    result.push(row);
  }

  return result;
}

/**
 * For max pooling visualization: return which cell in each 2x2 block was the max.
 * Returns an array of {row, col} positions (in the original grid) that survived pooling.
 */
export function maxPool2x2Indices(
  data: number[][]
): { row: number; col: number }[] {
  const rows = data.length;
  const cols = data[0].length;
  const indices: { row: number; col: number }[] = [];

  for (let r = 0; r < rows - 1; r += 2) {
    for (let c = 0; c < cols - 1; c += 2) {
      let maxVal = -Infinity;
      let maxR = r;
      let maxC = c;
      for (let dr = 0; dr < 2; dr++) {
        for (let dc = 0; dc < 2; dc++) {
          if (data[r + dr][c + dc] > maxVal) {
            maxVal = data[r + dr][c + dc];
            maxR = r + dr;
            maxC = c + dc;
          }
        }
      }
      indices.push({ row: maxR, col: maxC });
    }
  }

  return indices;
}
