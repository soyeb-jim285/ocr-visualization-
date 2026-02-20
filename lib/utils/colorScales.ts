import {
  scaleSequential,
  interpolateViridis,
  interpolateInferno,
  interpolateRdBu,
} from "d3";

/** Activation values [0, max] -> viridis color (perceptually uniform, colorblind safe) */
export function activationColorScale(maxVal: number) {
  return scaleSequential(interpolateViridis).domain([0, Math.max(maxVal, 0.001)]);
}

/** Weight values [-max, max] -> diverging red-blue */
export function weightColorScale(maxAbsVal: number) {
  const bound = Math.max(maxAbsVal, 0.001);
  return scaleSequential(interpolateRdBu).domain([-bound, bound]);
}

/** Generic heatmap [min, max] -> inferno */
export function heatmapColorScale(min: number, max: number) {
  return scaleSequential(interpolateInferno).domain([min, Math.max(max, min + 0.001)]);
}

/** Parse a CSS color string to [r, g, b] */
export function parseColor(color: string): [number, number, number] {
  // Handle rgb() format from d3
  const match = color.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
  if (match) {
    return [parseInt(match[1]), parseInt(match[2]), parseInt(match[3])];
  }

  // Handle hex format
  if (color.startsWith("#")) {
    const hex = color.slice(1);
    return [
      parseInt(hex.slice(0, 2), 16),
      parseInt(hex.slice(2, 4), 16),
      parseInt(hex.slice(4, 6), 16),
    ];
  }

  return [128, 128, 128]; // fallback gray
}
