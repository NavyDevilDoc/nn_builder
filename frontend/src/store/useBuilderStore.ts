import { create } from "zustand";
import type {
  BuilderConfig,
  EpochEvent,
  LayerConfig,
  LayerType,
} from "../types/config";

// ── Layer defaults per type ───────────────────────────────────────────────────

export const LAYER_DEFAULTS: Record<LayerType, Omit<LayerConfig, "id" | "type">> = {
  Linear:   { units: 64, activation: "ReLU", batch_norm: false, dropout: false, dropout_rate: 0.0 },
  Conv2d:   { out_channels: 32, kernel_size: 3, activation: "ReLU", batch_norm: false },
  MaxPool2d:{ kernel_size: 2 },
  Flatten:  {},
  LSTM:     { hidden_size: 64, num_layers: 1, bidirectional: false, rnn_dropout: 0.0 },
  GRU:      { hidden_size: 64, num_layers: 1, bidirectional: false, rnn_dropout: 0.0 },
};

// ── Default config (everything except layers) ────────────────────────────────

const DEFAULT_HYPERPARAMS: Omit<BuilderConfig, "hidden_layers"> = {
  dataset:        "moons",
  optimizer:      "Adam",
  lr:             0.01,
  epochs:         100,
  batch_size:     32,
  weight_decay:   0.0,
  seed:           42,
  early_stopping: false,
  es_patience:    20,
  es_min_delta:   0.0,
  device:         "cpu",
};

// ── Store shape ───────────────────────────────────────────────────────────────

interface TrainingState {
  active:     boolean;
  sessionId:  string | null;
  epochsDone: number;
  history: {
    train_loss: number[];
    val_loss:   number[];
    train_acc:  number[];
    val_acc:    number[];
  };
}

interface ResultsState {
  confusionMatrix:   string | null;
  decisionBoundary:  string | null;
}

interface BuilderStore {
  layers:      LayerConfig[];                       // frontend layers — carry id
  hyperparams: Omit<BuilderConfig, "hidden_layers">;
  training:    TrainingState;
  results:     ResultsState;

  // Build a backend-ready BuilderConfig (strips layer ids)
  getConfig: () => BuilderConfig;

  // Layer actions
  addLayer:      (type: LayerType) => void;
  removeLayer:   (id: string) => void;
  updateLayer:   (id: string, patch: Partial<LayerConfig>) => void;
  setLayerType:  (id: string, type: LayerType) => void;
  reorderLayers: (layers: LayerConfig[]) => void;

  // Hyperparameter actions
  updateHyperparams: (patch: Partial<Omit<BuilderConfig, "hidden_layers">>) => void;

  // Training actions
  startTraining:  (sessionId: string) => void;
  setSessionId:   (sessionId: string) => void;
  pushEpoch:      (event: EpochEvent) => void;
  finishTraining: () => void;

  // Result actions
  setConfusionMatrix:  (image: string) => void;
  setDecisionBoundary: (image: string) => void;
}

// ── Store implementation ──────────────────────────────────────────────────────

export const useBuilderStore = create<BuilderStore>((set, get) => ({
  layers:      [],
  hyperparams: DEFAULT_HYPERPARAMS,
  training: {
    active:     false,
    sessionId:  null,
    epochsDone: 0,
    history:    { train_loss: [], val_loss: [], train_acc: [], val_acc: [] },
  },
  results: {
    confusionMatrix:  null,
    decisionBoundary: null,
  },

  getConfig: () => {
    const { layers, hyperparams } = get();
    return {
      ...hyperparams,
      hidden_layers: layers.map(({ id: _id, ...rest }) => rest),
    };
  },

  // ── Layer actions ───────────────────────────────────────────────────────────

  addLayer: (type) =>
    set((s) => ({
      layers: [
        ...s.layers,
        { id: crypto.randomUUID(), type, ...LAYER_DEFAULTS[type] } as LayerConfig,
      ],
    })),

  removeLayer: (id) =>
    set((s) => ({ layers: s.layers.filter((l) => l.id !== id) })),

  updateLayer: (id, patch) =>
    set((s) => ({
      layers: s.layers.map((l) => (l.id === id ? { ...l, ...patch } : l)),
    })),

  // Swap to new type, preserving the id, resetting all other fields to defaults
  setLayerType: (id, type) =>
    set((s) => ({
      layers: s.layers.map((l) =>
        l.id === id
          ? ({ id, type, ...LAYER_DEFAULTS[type] } as LayerConfig)
          : l
      ),
    })),

  reorderLayers: (layers) => set({ layers }),

  // ── Hyperparameter actions ──────────────────────────────────────────────────

  updateHyperparams: (patch) =>
    set((s) => ({ hyperparams: { ...s.hyperparams, ...patch } })),

  // ── Training actions ────────────────────────────────────────────────────────

  startTraining: (sessionId) =>
    set({
      training: {
        active:     true,
        sessionId,
        epochsDone: 0,
        history:    { train_loss: [], val_loss: [], train_acc: [], val_acc: [] },
      },
      results: { confusionMatrix: null, decisionBoundary: null },
    }),

  pushEpoch: (event) =>
    set((s) => ({
      training: {
        ...s.training,
        epochsDone: event.epoch + 1,
        history: {
          train_loss: [...s.training.history.train_loss, event.train_loss],
          val_loss:   [...s.training.history.val_loss,   event.val_loss],
          train_acc:  [...s.training.history.train_acc,  event.train_acc],
          val_acc:    [...s.training.history.val_acc,    event.val_acc],
        },
      },
    })),

  setSessionId: (sessionId) =>
    set((s) => ({ training: { ...s.training, sessionId } })),

  finishTraining: () =>
    set((s) => ({ training: { ...s.training, active: false } })),

  // ── Result actions ──────────────────────────────────────────────────────────

  setConfusionMatrix:  (image) => set((s) => ({ results: { ...s.results, confusionMatrix: image } })),
  setDecisionBoundary: (image) => set((s) => ({ results: { ...s.results, decisionBoundary: image } })),
}));
