// EMNIST ByMerge: 62 output neurons, but only 47 classes are trained.
// ByMerge merges 15 lowercase letters that look like their uppercase
// counterparts into the uppercase class. Those indices are untrained.
//
// Label mapping: 0-9 digits, 10-35 uppercase A-Z, 36-61 lowercase a-z
// Merged (untrained) indices: c(38), i(44), j(45), k(46), l(47), m(48),
//   o(50), p(51), s(54), u(56), v(57), w(58), x(59), y(60), z(61)
export const EMNIST_CLASSES: string[] = [
  "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
  "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
  "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
  "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
  "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
];

// Indices merged in ByMerge (lowercase → uppercase). These output neurons
// are untrained and must be masked out before softmax.
export const BYMERGE_MERGED_INDICES = new Set([
  38, // c → C (12)
  44, // i → I (18)
  45, // j → J (19)
  46, // k → K (20)
  47, // l → L (21)
  48, // m → M (22)
  50, // o → O (24)
  51, // p → P (25)
  54, // s → S (28)
  56, // u → U (30)
  57, // v → V (31)
  58, // w → W (58)
  59, // x → X (33)
  60, // y → Y (34)
  61, // z → Z (35)
]);

export const CLASS_GROUPS = {
  digits: { start: 0, end: 9, label: "Digits" },
  uppercase: { start: 10, end: 35, label: "Uppercase" },
  lowercase: { start: 36, end: 61, label: "Lowercase" },
} as const;

export const NUM_CLASSES = 62;
