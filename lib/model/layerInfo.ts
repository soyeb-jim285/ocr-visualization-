export type LayerType =
  | "input"
  | "conv2d"
  | "batchnorm"
  | "relu"
  | "maxpool"
  | "flatten"
  | "dense"
  | "dropout"
  | "softmax";

export interface LayerMeta {
  name: string;
  type: LayerType;
  displayName: string;
  outputShape: number[];
  description: string;
  /** Whether this layer has a dedicated scroll section visualization */
  visualizable: boolean;
  /** Which scroll section index this layer belongs to (0-9) */
  sectionIndex: number;
}

export const LAYER_CONFIG: LayerMeta[] = [
  {
    name: "conv1",
    type: "conv2d",
    displayName: "Convolution 1",
    outputShape: [28, 28, 32],
    description:
      "32 filters scan the image looking for basic features like edges, corners, and curves.",
    visualizable: true,
    sectionIndex: 2,
  },
  {
    name: "bn1",
    type: "batchnorm",
    displayName: "Batch Norm 1",
    outputShape: [28, 28, 32],
    description:
      "Normalizes activations so training is more stable and faster.",
    visualizable: false,
    sectionIndex: 2,
  },
  {
    name: "relu1",
    type: "relu",
    displayName: "ReLU 1",
    outputShape: [28, 28, 32],
    description:
      "Sets all negative values to zero. This non-linearity lets the network learn complex patterns.",
    visualizable: true,
    sectionIndex: 3,
  },
  {
    name: "conv2",
    type: "conv2d",
    displayName: "Convolution 2",
    outputShape: [28, 28, 64],
    description:
      "64 filters combine basic features into more complex patterns like loops and intersections.",
    visualizable: true,
    sectionIndex: 5,
  },
  {
    name: "bn2",
    type: "batchnorm",
    displayName: "Batch Norm 2",
    outputShape: [28, 28, 64],
    description: "Normalizes the second convolution layer's output.",
    visualizable: false,
    sectionIndex: 5,
  },
  {
    name: "relu2",
    type: "relu",
    displayName: "ReLU 2",
    outputShape: [28, 28, 64],
    description: "Applies non-linearity to the deeper feature maps.",
    visualizable: true,
    sectionIndex: 5,
  },
  {
    name: "pool1",
    type: "maxpool",
    displayName: "Max Pooling 1",
    outputShape: [14, 14, 64],
    description:
      "Shrinks each feature map by half, keeping only the strongest activations in each 2x2 region.",
    visualizable: true,
    sectionIndex: 4,
  },
  {
    name: "conv3",
    type: "conv2d",
    displayName: "Convolution 3",
    outputShape: [14, 14, 128],
    description:
      "128 filters detect high-level features like character parts and structural patterns.",
    visualizable: true,
    sectionIndex: 5,
  },
  {
    name: "bn3",
    type: "batchnorm",
    displayName: "Batch Norm 3",
    outputShape: [14, 14, 128],
    description: "Normalizes the third convolution layer's output.",
    visualizable: false,
    sectionIndex: 5,
  },
  {
    name: "relu3",
    type: "relu",
    displayName: "ReLU 3",
    outputShape: [14, 14, 128],
    description: "Applies non-linearity to the deepest convolutional features.",
    visualizable: true,
    sectionIndex: 5,
  },
  {
    name: "pool2",
    type: "maxpool",
    displayName: "Max Pooling 2",
    outputShape: [7, 7, 128],
    description:
      "Further compresses spatial information, producing compact 7x7 feature maps.",
    visualizable: true,
    sectionIndex: 5,
  },
  {
    name: "flatten",
    type: "flatten",
    displayName: "Flatten",
    outputShape: [6272],
    description:
      "Reshapes the 7x7x128 volume into a single vector of 6,272 values.",
    visualizable: false,
    sectionIndex: 6,
  },
  {
    name: "dense1",
    type: "dense",
    displayName: "Dense 1",
    outputShape: [256],
    description:
      "256 neurons combine all spatial features to form a high-level representation of the character.",
    visualizable: true,
    sectionIndex: 6,
  },
  {
    name: "relu4",
    type: "relu",
    displayName: "ReLU 4",
    outputShape: [256],
    description: "Non-linearity on the dense layer output.",
    visualizable: true,
    sectionIndex: 6,
  },
  {
    name: "dropout",
    type: "dropout",
    displayName: "Dropout",
    outputShape: [256],
    description:
      "Randomly turns off 50% of neurons during training to prevent overfitting. Inactive during inference.",
    visualizable: false,
    sectionIndex: 6,
  },
  {
    name: "output",
    type: "softmax",
    displayName: "Output (Softmax)",
    outputShape: [62],
    description:
      "Converts raw scores into probabilities across all 62 character classes.",
    visualizable: true,
    sectionIndex: 7,
  },
];

/** Get only visualizable layers */
export function getVisualizableLayers(): LayerMeta[] {
  return LAYER_CONFIG.filter((l) => l.visualizable);
}

/** Get layers for a specific section */
export function getLayersForSection(sectionIndex: number): LayerMeta[] {
  return LAYER_CONFIG.filter((l) => l.sectionIndex === sectionIndex);
}
