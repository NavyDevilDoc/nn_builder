// Single source of truth for data shapes shared between frontend and backend.
// Keep in sync with backend/models.py.
// Layer card `id` fields are frontend-only UUID routing keys; strip before POSTing.

export type Activation = "ReLU" | "Tanh" | "Sigmoid" | "LeakyReLU" | "Linear";
export type LayerType  = "Linear" | "Conv2d" | "MaxPool2d" | "Flatten" | "LSTM" | "GRU";

export interface LayerConfig {
  id: string;
  type: LayerType;
  // Linear
  units?: number;
  activation?: Activation;
  batch_norm?: boolean;
  dropout?: boolean;
  dropout_rate?: number;
  // Conv2d
  out_channels?: number;
  kernel_size?: number;    // shared with MaxPool2d
  // LSTM / GRU
  hidden_size?: number;
  num_layers?: number;
  bidirectional?: boolean;
  rnn_dropout?: number;
}

export interface BuilderConfig {
  dataset: string;
  hidden_layers: Omit<LayerConfig, "id">[];  // ids stripped before POST
  optimizer: "Adam" | "SGD" | "Momentum";
  lr: number;
  epochs: number;
  batch_size: number | "full";
  weight_decay: number;
  seed: number;
  early_stopping: boolean;
  es_patience: number;
  es_min_delta: number;
  device: "cpu" | "cuda";
  // Sequential datasets only: window-length override. null => dataset default.
  seq_len: number | null;
}

export type Task = "binary" | "multiclass" | "regression";

export interface DatasetInfo {
  task: Task;
  input_dim: number;        // features per timestep for sequential datasets
  n_classes: number;        // 0 for regression
  description: string;
  input_shape: [number, number, number] | null;
  seq_len: number | null;   // default window length; non-null => sequential
}

/** Window length actually used for a dataset: user override, else dataset default. */
export function effectiveSeqLen(meta: DatasetInfo, override: number | null | undefined): number | null {
  if (meta.seq_len === null) return null;
  return override ?? meta.seq_len;
}

export interface DatasetsResponse {
  names: string[];
  meta: Record<string, DatasetInfo>;
  suggested: Record<string, Omit<LayerConfig, "id">[]>;
}

// Training stream event types
export interface EpochEvent {
  type: "epoch";
  epoch: number;
  total: number;
  train_loss: number;
  val_loss: number;
  train_acc: number;   // accuracy for classification, R² for regression
  val_acc: number;
}

export interface DoneEvent {
  type: "done";
  session_id: string;
  epochs_trained: number;
}

export interface ErrorEvent {
  type: "error";
  message: string;
}

export type TrainingEvent = EpochEvent | DoneEvent | ErrorEvent;

export interface ValidationResult {
  valid: boolean;
  error: string | null;
  shapes: (number[] | null)[];
}

export interface VisualizeResponse {
  image: string;  // "data:image/png;base64,..."
}

export interface CodeResponse {
  code: string;
}
