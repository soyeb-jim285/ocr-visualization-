export interface TrainingHistory {
  loss: number[];
  accuracy: number[];
  val_loss: number[];
  val_accuracy: number[];
}

export interface WeightSnapshot {
  [layerName: string]:
    | number[]
    | { mean: number; std: number; min: number; max: number; shape: number[] };
}

export type WeightSnapshots = Record<string, WeightSnapshot>;

let cachedHistory: TrainingHistory | null = null;
let cachedSnapshots: WeightSnapshots | null = null;

/** Load pre-computed training history */
export async function loadTrainingHistory(): Promise<TrainingHistory> {
  if (cachedHistory) return cachedHistory;

  const res = await fetch("/training/history.json");
  cachedHistory = await res.json();
  return cachedHistory!;
}

/** Load pre-computed weight snapshots */
export async function loadWeightSnapshots(): Promise<WeightSnapshots> {
  if (cachedSnapshots) return cachedSnapshots;

  const res = await fetch("/training/weight-snapshots.json");
  cachedSnapshots = await res.json();
  return cachedSnapshots!;
}
