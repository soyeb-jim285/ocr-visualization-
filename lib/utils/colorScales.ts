/**
 * Fast color scale functions using inline lookup tables.
 * Replaces d3 scaleSequential + interpolateViridis/Inferno/RdBu
 * to eliminate per-pixel string allocation and regex parsing.
 */

// Viridis colormap - 32 control points [r, g, b] from t=0 to t=1
// prettier-ignore
const VIRIDIS = new Uint8Array([
  68,  1, 84,   71, 13, 96,   72, 24,106,   69, 37,116,
  64, 49,124,   57, 61,131,   49, 73,137,   42, 84,140,
  35, 95,142,   29,106,142,   24,116,140,   21,127,138,
  21,137,132,   27,147,124,   40,156,113,   59,165,100,
  80,173, 85,  104,181, 67,  131,188, 47,  159,193, 33,
 186,197, 29,  208,199, 35,  225,204, 43,  237,210, 49,
 246,216, 51,  251,222, 50,  253,228, 45,  253,232, 38,
 251,236, 35,  248,239, 33,  243,241, 38,  253,231, 37,
]);

// Inferno colormap - 32 control points [r, g, b]
// prettier-ignore
const INFERNO = new Uint8Array([
   0,  0,  4,    6,  2, 23,   19,  6, 50,   38,  9, 74,
  58,  9, 90,   78, 10, 97,   98, 14, 96,  117, 22, 89,
 135, 33, 79,  151, 45, 68,  165, 58, 57,  178, 71, 47,
 190, 85, 37,  200, 99, 28,  209,114, 21,  217,129, 14,
 224,144,  8,  229,159,  6,  233,175, 11,  235,190, 24,
 234,205, 43,  230,220, 64,  224,233, 88,  218,244,113,
 214,251,136,  215,254,158,  220,255,179,  227,255,199,
 237,254,218,  247,253,235,  252,255,247,  252,255,164,
]);

const VIRIDIS_N = 31; // VIRIDIS.length/3 - 1
const INFERNO_N = 31;

/** Map value in [0, max] to [r, g, b] using viridis colormap */
export function viridisRGB(
  value: number,
  max: number,
): [number, number, number] {
  const t = max > 0 ? Math.max(0, Math.min(1, value / max)) : 0;
  const idx = t * VIRIDIS_N;
  const lo = (idx | 0) * 3; // floor + multiply
  const hi = Math.min(lo + 3, VIRIDIS_N * 3);
  const f = idx - (idx | 0);
  return [
    (VIRIDIS[lo] + (VIRIDIS[hi] - VIRIDIS[lo]) * f) | 0,
    (VIRIDIS[lo + 1] + (VIRIDIS[hi + 1] - VIRIDIS[lo + 1]) * f) | 0,
    (VIRIDIS[lo + 2] + (VIRIDIS[hi + 2] - VIRIDIS[lo + 2]) * f) | 0,
  ];
}

/** Map value in [0, max] to [r, g, b] using inferno colormap */
export function infernoRGB(
  value: number,
  max: number,
): [number, number, number] {
  const t = max > 0 ? Math.max(0, Math.min(1, value / max)) : 0;
  const idx = t * INFERNO_N;
  const lo = (idx | 0) * 3;
  const hi = Math.min(lo + 3, INFERNO_N * 3);
  const f = idx - (idx | 0);
  return [
    (INFERNO[lo] + (INFERNO[hi] - INFERNO[lo]) * f) | 0,
    (INFERNO[lo + 1] + (INFERNO[hi + 1] - INFERNO[lo + 1]) * f) | 0,
    (INFERNO[lo + 2] + (INFERNO[hi + 2] - INFERNO[lo + 2]) * f) | 0,
  ];
}

/** Map value in [-max, max] to [r, g, b] using diverging blue-white-red */
export function divergingRGB(
  value: number,
  maxAbs: number,
): [number, number, number] {
  const bound = Math.max(maxAbs, 0.001);
  const t = Math.max(-1, Math.min(1, value / bound)); // -1..1
  if (t < 0) {
    // Blue to white
    const s = -t; // 0..1
    return [
      (255 * (1 - s * 0.7)) | 0,
      (255 * (1 - s * 0.6)) | 0,
      255,
    ];
  } else {
    // White to red
    const s = t;
    return [
      255,
      (255 * (1 - s * 0.6)) | 0,
      (255 * (1 - s * 0.7)) | 0,
    ];
  }
}
