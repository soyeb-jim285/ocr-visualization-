import { viridisRGB } from "./colorScales";

/**
 * Render a 2D number array as a heatmap on a Canvas2D context.
 * Uses ImageData pixel manipulation for performance.
 */
export function drawHeatmap(
  ctx: CanvasRenderingContext2D,
  data: number[][],
  max: number,
  cellSize: number = 1
): void {
  const rows = data.length;
  const cols = data[0].length;
  const width = cols * cellSize;
  const height = rows * cellSize;
  const imageData = ctx.createImageData(width, height);
  const pixels = imageData.data;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const [red, green, blue] = viridisRGB(data[r][c], max);

      // Fill the cellSize x cellSize block
      for (let dy = 0; dy < cellSize; dy++) {
        for (let dx = 0; dx < cellSize; dx++) {
          const px = c * cellSize + dx;
          const py = r * cellSize + dy;
          const idx = (py * width + px) * 4;
          pixels[idx] = red;
          pixels[idx + 1] = green;
          pixels[idx + 2] = blue;
          pixels[idx + 3] = 255;
        }
      }
    }
  }

  ctx.putImageData(imageData, 0, 0);
}

/**
 * Draw a kernel overlay box on a canvas.
 */
export function drawKernelOverlay(
  ctx: CanvasRenderingContext2D,
  row: number,
  col: number,
  kernelSize: number,
  cellSize: number,
  color: string = "#06b6d4"
): void {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.strokeRect(
    col * cellSize,
    row * cellSize,
    kernelSize * cellSize,
    kernelSize * cellSize
  );
}

/**
 * Draw a grayscale image from a 2D array onto a canvas.
 */
export function drawGrayscaleImage(
  ctx: CanvasRenderingContext2D,
  data: number[][],
  cellSize: number = 1
): void {
  const rows = data.length;
  const cols = data[0].length;
  const width = cols * cellSize;
  const height = rows * cellSize;
  const imageData = ctx.createImageData(width, height);
  const pixels = imageData.data;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const val = Math.round(data[r][c] * 255);

      for (let dy = 0; dy < cellSize; dy++) {
        for (let dx = 0; dx < cellSize; dx++) {
          const px = c * cellSize + dx;
          const py = r * cellSize + dy;
          const idx = (py * width + px) * 4;
          pixels[idx] = val;
          pixels[idx + 1] = val;
          pixels[idx + 2] = val;
          pixels[idx + 3] = 255;
        }
      }
    }
  }

  ctx.putImageData(imageData, 0, 0);
}
