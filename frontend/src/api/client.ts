/**
 * Typed wrappers for all backend API routes.
 *
 * The Vite dev proxy forwards /api/* → http://localhost:8000, so no CORS
 * handling is needed here.  In production, FastAPI serves the built frontend
 * from frontend/dist/ so the same origin applies.
 *
 * Training uses fetch + ReadableStream (not EventSource) because the
 * train endpoint is POST — EventSource only supports GET.
 */
import type {
  BuilderConfig,
  CodeResponse,
  DatasetsResponse,
  TrainingEvent,
  ValidationResult,
  VisualizeResponse,
} from "../types/config";

const API = "/api";

// ── Datasets ──────────────────────────────────────────────────────────────────

export async function fetchDatasets(): Promise<DatasetsResponse> {
  const res = await fetch(`${API}/datasets`);
  if (!res.ok) throw new Error(`fetchDatasets failed: ${res.statusText}`);
  return res.json();
}

// ── Validation ────────────────────────────────────────────────────────────────

export async function validateConfig(
  inputShape: number[] | null,
  inputDim: number,
  task: string,
  nClasses: number,
  hiddenLayers: BuilderConfig["hidden_layers"]
): Promise<ValidationResult> {
  const res = await fetch(`${API}/validate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      input_shape: inputShape,
      input_dim: inputDim,
      task,
      n_classes: nClasses,
      hidden_layers: hiddenLayers,
    }),
  });
  if (!res.ok) throw new Error(`validateConfig failed: ${res.statusText}`);
  return res.json();
}

// ── Training (SSE via fetch streaming) ────────────────────────────────────────

export async function* streamTraining(
  config: BuilderConfig,
  signal?: AbortSignal
): AsyncGenerator<TrainingEvent> {
  const res = await fetch(`${API}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
    signal,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Training request failed: ${res.statusText} — ${text}`);
  }
  if (!res.body) throw new Error("No response body from /api/train");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (line.startsWith("data: ")) {
          yield JSON.parse(line.slice(6)) as TrainingEvent;
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

export async function cancelTraining(sessionId: string): Promise<void> {
  await fetch(`${API}/train/${sessionId}`, { method: "DELETE" });
}

// ── Visualization ─────────────────────────────────────────────────────────────

export async function fetchConfusionMatrix(
  sessionId: string,
  config: BuilderConfig
): Promise<VisualizeResponse> {
  const res = await fetch(`${API}/visualize/confusion_matrix`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, config }),
  });
  if (!res.ok) throw new Error(`fetchConfusionMatrix failed: ${res.statusText}`);
  return res.json();
}

export async function fetchDecisionBoundary(
  sessionId: string,
  config: BuilderConfig
): Promise<VisualizeResponse> {
  const res = await fetch(`${API}/visualize/decision_boundary`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, config }),
  });
  if (!res.ok) throw new Error(`fetchDecisionBoundary failed: ${res.statusText}`);
  return res.json();
}

// ── Code generation ───────────────────────────────────────────────────────────

export async function fetchCode(config: BuilderConfig): Promise<CodeResponse> {
  const params = new URLSearchParams({ config: JSON.stringify(config) });
  const res = await fetch(`${API}/code?${params}`);
  if (!res.ok) throw new Error(`fetchCode failed: ${res.statusText}`);
  return res.json();
}
