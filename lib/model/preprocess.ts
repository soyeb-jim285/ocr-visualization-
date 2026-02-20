/**
 * Preprocess a drawing canvas ImageData for EMNIST inference.
 *
 * The canvas should be white strokes on black background.
 * Returns:
 *  - tensor: Float32Array in NCHW format [1, 1, 28, 28] for ONNX Runtime
 *  - pixelArray: 2D number array [28][28] for visualization
 */
export function preprocessCanvas(imageData: ImageData): {
  tensor: Float32Array;
  pixelArray: number[][]; // display-oriented (matches what user drew)
} {
  // Step 1: Convert ImageData to grayscale 2D array at original resolution
  const { width, height, data } = imageData;
  const gray2D: number[][] = [];
  for (let y = 0; y < height; y++) {
    const row: number[] = [];
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      // Use luminance formula or just red channel (canvas is grayscale)
      const val = data[idx]; // R channel (white strokes on black = 255 on 0)
      row.push(val / 255);
    }
    gray2D.push(row);
  }

  // Step 2: Resize to 28x28 using bilinear interpolation
  const resized = resize2D(gray2D, height, width, 28, 28);

  // Step 3: Build NCHW tensor with transpose for the model.
  // EMNIST images are stored transposed, so we transpose our input to match.
  const tensor = new Float32Array(1 * 1 * 28 * 28);
  for (let r = 0; r < 28; r++) {
    for (let c = 0; c < 28; c++) {
      // Transpose: model expects resized[c][r] at position [r][c]
      tensor[r * 28 + c] = resized[c][r];
    }
  }

  // pixelArray stays in the original (non-transposed) orientation for display
  return { tensor, pixelArray: resized };
}

/** Area-average (box filter) resize — properly averages all source pixels per target pixel */
function resize2D(
  src: number[][],
  srcH: number,
  srcW: number,
  dstH: number,
  dstW: number
): number[][] {
  const dst: number[][] = [];
  for (let y = 0; y < dstH; y++) {
    const row: number[] = [];
    const srcY0 = (y / dstH) * srcH;
    const srcY1 = ((y + 1) / dstH) * srcH;
    const iy0 = Math.floor(srcY0);
    const iy1 = Math.min(Math.ceil(srcY1), srcH);

    for (let x = 0; x < dstW; x++) {
      const srcX0 = (x / dstW) * srcW;
      const srcX1 = ((x + 1) / dstW) * srcW;
      const ix0 = Math.floor(srcX0);
      const ix1 = Math.min(Math.ceil(srcX1), srcW);

      let sum = 0;
      let area = 0;

      for (let sy = iy0; sy < iy1; sy++) {
        // vertical overlap of this source row with the target pixel
        const wy = Math.min(sy + 1, srcY1) - Math.max(sy, srcY0);
        for (let sx = ix0; sx < ix1; sx++) {
          // horizontal overlap of this source column with the target pixel
          const wx = Math.min(sx + 1, srcX1) - Math.max(sx, srcX0);
          const w = wx * wy;
          sum += src[sy][sx] * w;
          area += w;
        }
      }

      row.push(area > 0 ? sum / area : 0);
    }
    dst.push(row);
  }
  return dst;
}

/**
 * Preprocess an uploaded image file.
 */
export async function preprocessImage(file: File): Promise<{
  tensor: Float32Array;
  pixelArray: number[][]; // display-oriented
}> {
  const img = new Image();
  const url = URL.createObjectURL(file);

  return new Promise((resolve, reject) => {
    img.onload = () => {
      URL.revokeObjectURL(url);
      const canvas = document.createElement("canvas");
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(img, 0, 0);
      const imageData = ctx.getImageData(0, 0, img.width, img.height);
      resolve(preprocessCanvas(imageData));
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Failed to load image"));
    };
    img.src = url;
  });
}
