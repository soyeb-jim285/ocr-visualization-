/**
 * Encode/decode 28×28 canvas pixel data for URL sharing.
 *
 * Flow: pixelArray[28][28] (0–1 floats) → Uint8Array(784) → deflate → base64url → hash fragment
 */

const HASH_KEY = "d";
const SIZE = 28;

// -- Base64url helpers (no padding) ------------------------------------------

function uint8ToBase64url(bytes: Uint8Array): string {
  let binary = "";
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function base64urlToUint8(str: string): Uint8Array {
  const padded = str.replace(/-/g, "+").replace(/_/g, "/") + "==".slice(0, (4 - (str.length % 4)) % 4);
  const binary = atob(padded);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

// -- Compression via native CompressionStream --------------------------------

async function deflate(data: Uint8Array): Promise<Uint8Array> {
  const cs = new CompressionStream("deflate");
  const writer = cs.writable.getWriter();
  writer.write(data as unknown as BufferSource);
  writer.close();
  const reader = cs.readable.getReader();
  const chunks: Uint8Array[] = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
  }
  const total = chunks.reduce((s, c) => s + c.length, 0);
  const result = new Uint8Array(total);
  let offset = 0;
  for (const c of chunks) { result.set(c, offset); offset += c.length; }
  return result;
}

async function inflate(data: Uint8Array): Promise<Uint8Array> {
  const ds = new DecompressionStream("deflate");
  const writer = ds.writable.getWriter();
  writer.write(data as unknown as BufferSource);
  writer.close();
  const reader = ds.readable.getReader();
  const chunks: Uint8Array[] = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
  }
  const total = chunks.reduce((s, c) => s + c.length, 0);
  const result = new Uint8Array(total);
  let offset = 0;
  for (const c of chunks) { result.set(c, offset); offset += c.length; }
  return result;
}

// -- Public API --------------------------------------------------------------

/** Encode a 28×28 pixel array into a URL hash string (without the #). */
export async function encodePixelsToHash(pixelArray: number[][]): Promise<string> {
  const raw = new Uint8Array(SIZE * SIZE);
  for (let r = 0; r < SIZE; r++) {
    for (let c = 0; c < SIZE; c++) {
      raw[r * SIZE + c] = Math.round(pixelArray[r][c] * 255);
    }
  }
  const compressed = await deflate(raw);
  return `${HASH_KEY}=${uint8ToBase64url(compressed)}`;
}

/** Decode a URL hash string back to a 28×28 pixel array. Returns null if invalid. */
export async function decodeHashToPixels(hash: string): Promise<number[][] | null> {
  const stripped = hash.startsWith("#") ? hash.slice(1) : hash;
  const prefix = `${HASH_KEY}=`;
  if (!stripped.startsWith(prefix)) return null;
  const encoded = stripped.slice(prefix.length);
  if (!encoded) return null;

  try {
    const compressed = base64urlToUint8(encoded);
    const raw = await inflate(compressed);
    if (raw.length !== SIZE * SIZE) return null;

    const pixelArray: number[][] = [];
    for (let r = 0; r < SIZE; r++) {
      const row: number[] = [];
      for (let c = 0; c < SIZE; c++) {
        row.push(raw[r * SIZE + c] / 255);
      }
      pixelArray.push(row);
    }
    return pixelArray;
  } catch {
    return null;
  }
}

/** Reconstruct the NCHW Float32Array tensor from a 28×28 pixel array (applies EMNIST transpose). */
export function pixelsToTensor(pixelArray: number[][]): Float32Array {
  const tensor = new Float32Array(SIZE * SIZE);
  for (let r = 0; r < SIZE; r++) {
    for (let c = 0; c < SIZE; c++) {
      // Transpose for EMNIST format (same as preprocess.ts)
      tensor[r * SIZE + c] = pixelArray[c][r];
    }
  }
  return tensor;
}

/** Upscale 28×28 pixel array to an ImageData at the given resolution (nearest-neighbor). */
export function pixelsToImageData(pixelArray: number[][], width: number, height: number): ImageData {
  const imageData = new ImageData(width, height);
  const scaleX = SIZE / width;
  const scaleY = SIZE / height;
  for (let y = 0; y < height; y++) {
    const srcR = Math.min(Math.floor(y * scaleY), SIZE - 1);
    for (let x = 0; x < width; x++) {
      const srcC = Math.min(Math.floor(x * scaleX), SIZE - 1);
      const gray = Math.round(pixelArray[srcR][srcC] * 255);
      const idx = (y * width + x) * 4;
      imageData.data[idx] = gray;
      imageData.data[idx + 1] = gray;
      imageData.data[idx + 2] = gray;
      imageData.data[idx + 3] = 255;
    }
  }
  return imageData;
}
