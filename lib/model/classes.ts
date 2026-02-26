// Combined model: 146 output neurons
// - Indices 0–61: EMNIST ByMerge (0-9, A-Z, a-z)
// - Indices 62–145: BanglaLekha-Isolated (84 classes)
//
// BanglaLekha folder order:
//   Folders 1-11  → indices 62-72:  Vowels (স্বরবর্ণ)
//   Folders 12-50 → indices 73-111: Consonants + signs (ব্যঞ্জনবর্ণ)
//   Folders 51-60 → indices 112-121: Digits (সংখ্যা)
//   Folders 61-84 → indices 122-145: Compounds (যুক্তবর্ণ)
//
// EMNIST ByMerge merges 15 lowercase letters that look like their uppercase
// counterparts into the uppercase class. Those indices are untrained.
export const EMNIST_CLASSES: string[] = [
  // EMNIST (0-61)
  "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
  "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
  "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
  "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
  "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
  // BanglaLekha vowels — folders 1-11 (indices 62-72)
  "অ", "আ", "ই", "ঈ", "উ", "ঊ", "ঋ", "এ", "ঐ", "ও", "ঔ",
  // BanglaLekha consonants — folders 12-50 (indices 73-111)
  "ক", "খ", "গ", "ঘ", "ঙ", "চ", "ছ", "জ", "ঝ", "ঞ",
  "ট", "ঠ", "ড", "ঢ", "ণ", "ত", "থ", "দ", "ধ", "ন",
  "প", "ফ", "ব", "ভ", "ম", "য", "র", "ল", "শ", "ষ",
  "স", "হ", "ড়", "ঢ়", "য়",
  "ৎ", "ং", "ঃ", "ঁ",
  // BanglaLekha digits — folders 51-60 (indices 112-121)
  "০", "১", "২", "৩", "৪", "৫", "৬", "৭", "৮", "৯",
  // BanglaLekha compounds — folders 61-84 (indices 122-145)
  "ক্ষ", "জ্ঞ", "ঞ্চ", "ঞ্ছ", "ঞ্জ", "ত্ত", "ত্র", "দ্ধ", "দ্ব", "ন্ত",
  "ন্দ", "ন্ধ", "ম্প", "ল্ক", "ষ্ট", "স্ত", "ক্ত", "ক্র", "ক্ম", "গ্ন",
  "ঙ্ক", "ঙ্গ", "ণ্ড", "হ্ম",
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
  bnVowels: { start: 62, end: 72, label: "স্বরবর্ণ" },
  bnConsonants: { start: 73, end: 111, label: "ব্যঞ্জনবর্ণ" },
  bnDigits: { start: 112, end: 121, label: "বাংলা সংখ্যা" },
  bnCompounds: { start: 122, end: 145, label: "যুক্তবর্ণ" },
} as const;

export const NUM_CLASSES = 146;
