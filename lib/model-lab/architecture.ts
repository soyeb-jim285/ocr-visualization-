/** Layer & architecture types for the Model Lab CNN builder. */

export type Activation = "relu" | "gelu" | "silu" | "leakyRelu" | "tanh";
export type PoolingType = "max" | "avg" | "none";

export interface ConvLayerConfig {
  id: string;
  filters: number; // 16–512
  kernelSize: 3 | 5 | 7;
  activation: Activation;
  pooling: PoolingType;
  batchNorm: boolean;
}

export interface DenseConfig {
  width: number; // 64–1024
  dropout: number; // 0–0.7
  activation: Activation;
}

export interface ArchitectureConfig {
  convLayers: ConvLayerConfig[];
  dense: DenseConfig;
}

let nextId = 1;
export function generateLayerId(): string {
  return `layer-${nextId++}`;
}

export function createDefaultConvLayer(
  overrides?: Partial<ConvLayerConfig>,
): ConvLayerConfig {
  return {
    id: generateLayerId(),
    filters: 32,
    kernelSize: 3,
    activation: "relu",
    pooling: "none",
    batchNorm: false,
    ...overrides,
  };
}

export const DEFAULT_ARCHITECTURE: ArchitectureConfig = {
  convLayers: [
    {
      id: "layer-default-1",
      filters: 64,
      kernelSize: 3,
      activation: "relu",
      pooling: "none",
      batchNorm: true,
    },
    {
      id: "layer-default-2",
      filters: 128,
      kernelSize: 3,
      activation: "relu",
      pooling: "max",
      batchNorm: true,
    },
    {
      id: "layer-default-3",
      filters: 256,
      kernelSize: 3,
      activation: "relu",
      pooling: "max",
      batchNorm: true,
    },
  ],
  dense: {
    width: 512,
    dropout: 0.5,
    activation: "relu",
  },
};

export const FILTER_OPTIONS = [16, 32, 64, 128, 256, 512] as const;
export const KERNEL_OPTIONS = [3, 5, 7] as const;
export const ACTIVATION_OPTIONS: Activation[] = [
  "relu",
  "gelu",
  "silu",
  "leakyRelu",
  "tanh",
];
export const POOLING_OPTIONS: PoolingType[] = ["none", "max", "avg"];
export const MAX_CONV_LAYERS = 6;
