// Task-aware presentation of the per-epoch `*_acc` metric.
// The backend reuses the train_acc / val_acc keys for every task:
// accuracy in [0, 1] for classification, R² (≤ 1, unbounded below) for regression.
import type { Task } from "../types/config";

export function metricName(task: Task | null): string {
  return task === "regression" ? "R²" : "acc";
}

export function formatMetric(value: number, task: Task | null): string {
  return task === "regression" ? value.toFixed(3) : `${(value * 100).toFixed(1)}%`;
}

/** Chart-scale value: percent for accuracy, raw for R². */
export function chartMetric(value: number, task: Task | null): number {
  return task === "regression" ? +value.toFixed(4) : +(value * 100).toFixed(2);
}
