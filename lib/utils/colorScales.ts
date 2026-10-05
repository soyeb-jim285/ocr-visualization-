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

const VIRIDIS_N = 31; // VIRIDIS.length/3 - 1

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

/** Map value in [-max, max] to [r, g, b]: indigo (neg) -> ink (zero) -> coral (pos), tuned for the dark film-base palette */
export function divergingRGB(
  value: number,
  maxAbs: number,
): [number, number, number] {
  const bound = Math.max(maxAbs, 0.001);
  const t = Math.max(-1, Math.min(1, value / bound)); // -1..1
  const s = Math.abs(t);
  // zero = ink #e9eef2, neg end = #7f8cff, pos end = #ff6b4a
  const [r, g, b] = t < 0 ? [127, 140, 255] : [255, 107, 74];
  return [
    (233 + (r - 233) * s) | 0,
    (238 + (g - 238) * s) | 0,
    (242 + (b - 242) * s) | 0,
  ];
}
