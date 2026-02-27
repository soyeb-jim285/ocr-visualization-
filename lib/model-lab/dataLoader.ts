/** Fetch & parse compact binary training data for in-browser training. */

export interface LoadedDataset {
  trainImages: Float32Array; // flat, each image 784 floats
  trainLabels: Uint8Array;
  testImages: Float32Array;
  testLabels: Uint8Array;
  numClasses: number;
  numTrain: number;
  numTest: number;
  imageSize: number;
  labelMap: string[];
}

export type DatasetType = "emnist" | "bangla" | "combined" | "digits";

// EMNIST label map (62 classes: 0-9, A-Z, a-z)
const EMNIST_LABELS: string[] = [
  ..."0123456789".split(""),
  ..."ABCDEFGHIJKLMNOPQRSTUVWXYZ".split(""),
  ..."abcdefghijklmnopqrstuvwxyz".split(""),
];

// Bangla label map (84 classes)
const BANGLA_LABELS: string[] = [
  // Vowels (11)
  "অ", "আ", "ই", "ঈ", "উ", "ঊ", "ঋ", "এ", "ঐ", "ও", "ঔ",
  // Consonants (39)
  "ক", "খ", "গ", "ঘ", "ঙ", "চ", "ছ", "জ", "ঝ", "ঞ",
  "ট", "ঠ", "ড", "ঢ", "ণ", "ত", "থ", "দ", "ধ", "ন",
  "প", "ফ", "ব", "ভ", "ম", "য", "র", "ল", "শ", "ষ",
  "স", "হ", "ড়", "ঢ়", "য়", "ৎ", "ং", "ঃ", "ঁ",
  // Digits (10)
  "০", "১", "২", "৩", "৪", "৫", "৬", "৭", "৮", "৯",
  // Compounds (24)
  "ক্ষ", "জ্ঞ", "ঞ্চ", "ঞ্ছ", "ঞ্জ", "ত্ত", "ত্র", "দ্ধ", "দ্ব", "ন্ত",
  "ন্দ", "ন্ধ", "ম্প", "ল্ক", "ষ্ট", "স্ত", "ক্ত", "ক্র", "ক্ম", "গ্ন",
  "ঙ্ক", "ঙ্গ", "ণ্ড", "হ্ম",
];

const DIGIT_LABELS = "0123456789".split("");

// Module-level cache
const cache = new Map<string, LoadedDataset>();

interface RawBinary {
  totalSamples: number;
  trainSamples: number;
  imageSize: number;
  numClasses: number;
  images: Float32Array; // normalized 0-1
  labels: Uint8Array;
}

async function fetchAndParse(url: string): Promise<RawBinary> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status}`);
  const buffer = await response.arrayBuffer();
  const view = new DataView(buffer);

  // Header: uint32 totalSamples, uint32 trainSamples, uint16 imageSize, uint16 numClasses
  const totalSamples = view.getUint32(0, true);
  const trainSamples = view.getUint32(4, true);
  const imageSize = view.getUint16(8, true);
  const numClasses = view.getUint16(10, true);

  const pixelsPerImage = imageSize * imageSize;
  const bytesPerSample = pixelsPerImage + 1; // pixels + label
  const bodyOffset = 12;

  const images = new Float32Array(totalSamples * pixelsPerImage);
  const labels = new Uint8Array(totalSamples);

  for (let i = 0; i < totalSamples; i++) {
    const offset = bodyOffset + i * bytesPerSample;
    for (let p = 0; p < pixelsPerImage; p++) {
      images[i * pixelsPerImage + p] = view.getUint8(offset + p) / 255;
    }
    labels[i] = view.getUint8(offset + pixelsPerImage);
  }

  return { totalSamples, trainSamples, imageSize, numClasses, images, labels };
}

function splitData(
  raw: RawBinary,
  labelMap: string[],
): LoadedDataset {
  const ppi = raw.imageSize * raw.imageSize;
  const numTrain = raw.trainSamples;
  const numTest = raw.totalSamples - numTrain;

  return {
    trainImages: raw.images.slice(0, numTrain * ppi),
    trainLabels: raw.labels.slice(0, numTrain),
    testImages: raw.images.slice(numTrain * ppi),
    testLabels: raw.labels.slice(numTrain),
    numClasses: raw.numClasses,
    numTrain,
    numTest,
    imageSize: raw.imageSize,
    labelMap,
  };
}

export async function loadDataset(type: DatasetType): Promise<LoadedDataset> {
  const cached = cache.get(type);
  if (cached) return cached;

  let result: LoadedDataset;

  if (type === "emnist") {
    const raw = await fetchAndParse("/data/emnist-subset.bin");
    result = splitData(raw, EMNIST_LABELS);
  } else if (type === "bangla") {
    const raw = await fetchAndParse("/data/bangla-subset.bin");
    result = splitData(raw, BANGLA_LABELS);
  } else if (type === "digits") {
    // Load EMNIST and filter to digits only (labels 0-9)
    const raw = await fetchAndParse("/data/emnist-subset.bin");
    const full = splitData(raw, EMNIST_LABELS);

    // Filter train
    const trainIndices: number[] = [];
    for (let i = 0; i < full.numTrain; i++) {
      if (full.trainLabels[i] < 10) trainIndices.push(i);
    }

    // Filter test
    const testIndices: number[] = [];
    for (let i = 0; i < full.numTest; i++) {
      if (full.testLabels[i] < 10) testIndices.push(i);
    }

    const ppi = 784;
    const trainImages = new Float32Array(trainIndices.length * ppi);
    const trainLabels = new Uint8Array(trainIndices.length);
    trainIndices.forEach((idx, i) => {
      trainImages.set(full.trainImages.subarray(idx * ppi, (idx + 1) * ppi), i * ppi);
      trainLabels[i] = full.trainLabels[idx];
    });

    const testImages = new Float32Array(testIndices.length * ppi);
    const testLabels = new Uint8Array(testIndices.length);
    testIndices.forEach((idx, i) => {
      testImages.set(full.testImages.subarray(idx * ppi, (idx + 1) * ppi), i * ppi);
      testLabels[i] = full.testLabels[idx];
    });

    result = {
      trainImages,
      trainLabels,
      testImages,
      testLabels,
      numClasses: 10,
      numTrain: trainIndices.length,
      numTest: testIndices.length,
      imageSize: 28,
      labelMap: DIGIT_LABELS,
    };
  } else {
    // combined: load both, concatenate, remap Bengali labels
    const [emnistRaw, banglaRaw] = await Promise.all([
      fetchAndParse("/data/emnist-subset.bin"),
      fetchAndParse("/data/bangla-subset.bin"),
    ]);

    const emnist = splitData(emnistRaw, EMNIST_LABELS);
    const bangla = splitData(banglaRaw, BANGLA_LABELS);

    const emnistClasses = emnist.numClasses;
    const totalClasses = emnistClasses + bangla.numClasses;

    // Remap Bengali labels: offset by emnistClasses
    const remapLabels = (labels: Uint8Array): Uint8Array => {
      const remapped = new Uint8Array(labels.length);
      for (let i = 0; i < labels.length; i++) {
        remapped[i] = labels[i] + emnistClasses;
      }
      return remapped;
    };

    const totalTrain = emnist.numTrain + bangla.numTrain;
    const totalTest = emnist.numTest + bangla.numTest;
    const ppi = 784;

    const trainImages = new Float32Array(totalTrain * ppi);
    trainImages.set(emnist.trainImages, 0);
    trainImages.set(bangla.trainImages, emnist.numTrain * ppi);

    const trainLabels = new Uint8Array(totalTrain);
    trainLabels.set(emnist.trainLabels, 0);
    trainLabels.set(remapLabels(bangla.trainLabels), emnist.numTrain);

    const testImages = new Float32Array(totalTest * ppi);
    testImages.set(emnist.testImages, 0);
    testImages.set(bangla.testImages, emnist.numTest * ppi);

    const testLabels = new Uint8Array(totalTest);
    testLabels.set(emnist.testLabels, 0);
    testLabels.set(remapLabels(bangla.testLabels), emnist.numTest);

    result = {
      trainImages,
      trainLabels,
      testImages,
      testLabels,
      numClasses: totalClasses,
      numTrain: totalTrain,
      numTest: totalTest,
      imageSize: 28,
      labelMap: [...EMNIST_LABELS, ...BANGLA_LABELS],
    };
  }

  cache.set(type, result);
  return result;
}
