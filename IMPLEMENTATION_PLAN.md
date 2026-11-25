# Neural Network Builder — React Rebuild Implementation Plan

*Created: 2026-11-23 (Streamlit prototype)*
*Revised: 2026-11-25 (React + FastAPI target, RNN layer planning)*

---

## 1. Project Direction

The Streamlit prototype proved the concept end-to-end: configurable MLP/CNN architecture, live training progress, decision boundary, confusion matrix, generated PyTorch code. Its limitations are all structural — Streamlit's page-rerun model makes drag-and-drop impossible, real-time streaming requires polling hacks, and CSS injection is fragile.

The React rebuild keeps every piece of Python logic and replaces only the presentation layer:

- **Keep:** `trainer.py`, `datasets/`, `components/shapes.py`, `components/code_gen.py`, `components/visualizer.py` — all wrapped behind a FastAPI backend
- **Replace:** `app.py` + `components/sidebar.py` + `components/diagram.py` — with a proper React frontend

**Primary wins over Streamlit:**

| Pain point in Streamlit | React solution |
|---|---|
| ↑/↓ buttons to reorder layers | Native drag-and-drop (`@dnd-kit`) |
| Polling loop + `time.sleep` for live training | Server-Sent Events — push from server |
| CSS selector hacks for button colors | Direct component styling |
| Page rerun on every widget change | Local React state, no reruns |
| Matplotlib figures served as PNGs | Interactive Recharts + SVG diagrams |
| No persistent state across sessions | localStorage config save/load |

---

## 2. Technology Stack

### Backend
| Component | Choice | Rationale |
|---|---|---|
| API framework | **FastAPI** | Async, SSE support, automatic OpenAPI docs |
| Training stream | **Server-Sent Events** (SSE) | One-way push from server; simpler than WebSocket for this use case |
| ML | **PyTorch** (existing) | No change |
| Visualization | **Matplotlib → base64 PNG** | Reuse existing visualizer.py; PNG over API |
| Environment | **uv** | Consistent with existing project |

### Frontend
| Component | Choice | Rationale |
|---|---|---|
| Framework | **React 18 + TypeScript** | Type safety catches config schema drift |
| Bundler | **Vite** | Fast HMR, TypeScript-first |
| Styling | **Tailwind CSS + shadcn/ui** | Dark mode built in; composable, no style conflicts |
| State | **Zustand** | Minimal boilerplate; perfect scale for this app |
| Drag-and-drop | **@dnd-kit/core** | Accessible, used by shadcn; no DOM mutations |
| Charts | **Recharts** | React-native SVG charts; zoomable, responsive |
| Diagram | **React Flow** | Draggable nodes, animated edges, free zoom/pan |
| Code highlighting | **Shiki** | Server-rendered; same tokenizer as VS Code |
| HTTP client | **TanStack Query + fetch** | Cache, loading states, error handling |

---

## 3. Project Structure

```
nn_builder_react/
├── backend/
│   ├── main.py                  # FastAPI app, CORS, lifespan
│   ├── routes/
│   │   ├── datasets.py          # GET /api/datasets, /api/datasets/{name}/sample
│   │   ├── validate.py          # POST /api/validate
│   │   ├── train.py             # POST /api/train  (SSE stream), DELETE /api/train/{id}
│   │   ├── visualize.py         # POST /api/visualize/...
│   │   └── code.py              # GET /api/code
│   ├── core/
│   │   ├── trainer.py           # LSTMWrapper, GRUWrapper, extended build_model()
│   │   └── shapes.py            # extended compute_shapes() with RNN support
│   ├── models.py                # Pydantic request/response schemas
│   └── session_store.py         # In-memory store: model objects + cancellation events
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   ├── api/
│   │   │   └── client.ts        # Typed wrappers for all backend routes
│   │   ├── store/
│   │   │   └── useBuilderStore.ts   # Zustand: config + training state
│   │   ├── components/
│   │   │   ├── layout/
│   │   │   │   ├── Sidebar.tsx
│   │   │   │   └── MainPanel.tsx
│   │   │   ├── layers/
│   │   │   │   ├── LayerCard.tsx       # Individual draggable card
│   │   │   │   ├── LayerBuilder.tsx    # @dnd-kit sortable list
│   │   │   │   └── LayerTypeSelect.tsx # Inline type picker
│   │   │   ├── diagram/
│   │   │   │   └── ArchDiagram.tsx     # React Flow network graph
│   │   │   ├── training/
│   │   │   │   ├── TrainingPanel.tsx
│   │   │   │   ├── LossChart.tsx       # Recharts, live update
│   │   │   │   └── TrainingControls.tsx # Train / Stop / Early stopping config
│   │   │   └── results/
│   │   │       ├── ConfusionMatrix.tsx  # img tag, base64 PNG from backend
│   │   │       ├── DecisionBoundary.tsx
│   │   │       └── CodeViewer.tsx      # Shiki syntax highlighting
│   │   └── types/
│   │       └── config.ts           # TypeScript interfaces mirroring Python schema
│   ├── index.html
│   ├── package.json
│   ├── tailwind.config.ts
│   └── vite.config.ts
├── pyproject.toml               # backend deps (fastapi, uvicorn, nn-builder path dep)
├── CLAUDE.md
├── IMPLEMENTATION_PLAN.md
├── PROGRESS.md
└── SESSION_HANDOFF.md
```

---

## 4. API Design

### Dataset endpoints
```
GET  /api/datasets
     → { names: string[], meta: { [name]: DatasetInfo }, suggested: { [name]: LayerConfig[] } }

GET  /api/datasets/{name}/sample
     → { image: string }   # base64 PNG thumbnail (image datasets only); 404 for tabular
```

### Architecture validation
```
POST /api/validate
     body: { input_shape: [C,H,W] | null, input_dim: int, hidden_layers: Layer[] }
     → { valid: bool, error: string | null, shapes: (number[] | null)[] }
```

### Training (SSE stream)
```
POST /api/train
     body: BuilderConfig
     Content-Type: text/event-stream

     Emits one SSE event per epoch:
       data: { type: "epoch", epoch: int, total: int, train_loss: f, val_loss: f,
               train_acc: f, val_acc: f }

     On completion:
       data: { type: "done", session_id: string, epochs_trained: int }

     On error:
       data: { type: "error", message: string }

DELETE /api/train/{session_id}
     → 204  (sets the cancellation Event; next epoch break)
```

**Client note:** `EventSource` only supports GET. The React client uses `fetch` + `ReadableStream` to consume this POST SSE stream. The SSE wire format (`data: {...}\n\n`) is identical.

### Visualization
```
POST /api/visualize/confusion_matrix
     body: { session_id: string, config: BuilderConfig }
     → { image: string }   # base64 PNG

POST /api/visualize/decision_boundary
     body: { session_id: string, config: BuilderConfig }
     → { image: string }

GET  /api/code?config=<json-encoded BuilderConfig>
     → { code: string }
```

### TypeScript config schema
```typescript
// types/config.ts
export type Activation = "ReLU" | "Tanh" | "Sigmoid" | "LeakyReLU" | "Linear";
export type LayerType  = "Linear" | "Conv2d" | "MaxPool2d" | "Flatten" | "LSTM" | "GRU";

export interface LayerConfig {
  id: string;             // client-side UUID, stripped before sending to backend
  type: LayerType;
  // Linear
  units?: number;
  activation?: Activation;
  batch_norm?: boolean;
  dropout?: boolean;
  dropout_rate?: number;
  // Conv2d
  out_channels?: number;
  kernel_size?: number;   // shared with MaxPool2d
  // LSTM / GRU
  hidden_size?: number;
  num_layers?: number;
  bidirectional?: boolean;
  rnn_dropout?: number;
}

export interface BuilderConfig {
  dataset: string;
  hidden_layers: Omit<LayerConfig, "id">[];  // IDs stripped before POST
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
}
```

---

## 5. Backend SSE Pattern

```python
# routes/train.py
import asyncio, json, threading, uuid
from fastapi import APIRouter
from fastapi.responses import StreamingResponse, Response
from backend.core.trainer import build_model
from components.trainer import train_model
from backend.session_store import sessions
import datasets.registry as registry

router = APIRouter()

@router.post("/api/train")
async def train(config: BuilderConfig):
    session_id  = str(uuid.uuid4())
    stop_event  = threading.Event()
    queue: asyncio.Queue = asyncio.Queue()
    loop        = asyncio.get_running_loop()

    def on_epoch(epoch, metrics):
        loop.call_soon_threadsafe(
            queue.put_nowait,
            {"type": "epoch", "epoch": epoch, "total": config.epochs, **metrics}
        )

    def _thread():
        try:
            cfg = config.model_dump()
            info = registry.meta(cfg["dataset"])
            cfg.update(task=info.task, n_classes=info.n_classes,
                       input_dim=info.input_dim,
                       input_shape=list(info.input_shape) if info.input_shape else None)
            X_train, y_train, X_test, y_test = registry.load(cfg["dataset"], seed=cfg["seed"])
            model = build_model(cfg)
            trained, history = train_model(model, cfg, X_train, y_train, X_test, y_test,
                                           on_epoch=on_epoch, stop_event=stop_event)
            sessions[session_id].update(model=trained, history=history,
                                        X_test=X_test, y_test=y_test, config=cfg)
            loop.call_soon_threadsafe(queue.put_nowait, {
                "type": "done", "session_id": session_id,
                "epochs_trained": len(history["train_loss"]),
            })
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": str(exc)})

    sessions[session_id] = {"stop_event": stop_event}
    threading.Thread(target=_thread, daemon=True).start()

    async def event_stream():
        while True:
            event = await queue.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event["type"] in ("done", "error"):
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.delete("/api/train/{session_id}")
async def cancel(session_id: str):
    if session_id in sessions:
        sessions[session_id]["stop_event"].set()
    return Response(status_code=204)
```

---

## 6. Frontend State (Zustand)

```typescript
// store/useBuilderStore.ts
interface BuilderState {
  config: BuilderConfig;
  training: {
    active: boolean;
    sessionId: string | null;
    history: { train_loss: number[]; val_loss: number[]; train_acc: number[]; val_acc: number[] };
    epochsDone: number;
  };
  results: {
    confusionMatrix: string | null;   // base64 PNG
    decisionBoundary: string | null;
  };
  // actions
  updateConfig: (patch: Partial<BuilderConfig>) => void;
  setLayers: (layers: LayerConfig[]) => void;
  startTraining: (sessionId: string) => void;
  pushEpoch: (metrics: EpochMetrics) => void;
  finishTraining: () => void;
}
```

---

## 7. RNN Architecture

### LSTMWrapper / GRUWrapper (backend/core/trainer.py)

`nn.LSTM` and `nn.GRU` return `(output, hidden_state)` tuples. `nn.Sequential` passes the return value directly to the next layer, so a bare `nn.LSTM` breaks the chain. The wrappers fix this:

```python
class LSTMWrapper(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.out_size = hidden_size * (2 if bidirectional else 1)

    def forward(self, x):          # x: (N, seq_len, input_size)
        out, _ = self.lstm(x)
        return out[:, -1, :]       # last timestep → (N, out_size)
```

An `_Unsqueeze` module is auto-inserted before each RNN layer to handle tabular input `(N, features)` → `(N, 1, features)`. This is a 1-step sequence, which is architecturally valid but pedagogically degenerate. Meaningful RNN use requires sequential datasets.

### Shape tracking (backend/core/shapes.py)

Extended `compute_shapes()` adds:
- `LSTM`/`GRU` layer: input `(features,)` → output `(hidden_size,)` or `(2*hidden_size,)` for bidirectional
- RNN + image input returns `None` (not supported)
- CNN layers mixed with RNN layers return `None`

### Sequential datasets (future phase)

RNN layers on current 9 datasets use `seq_len=1` — they compile and train but don't demonstrate sequence modeling. When a sequential dataset phase is added, the following will be needed:
- New dataset loaders returning `(N, seq_len, features)` arrays
- Updated `train_model()` to skip the `(N, D)` → `(N, 1, D)` unsqueeze (or make it opt-in)
- A `seq_len` field in `BuilderConfig`

---

## 8. Phased Build Order

### Phase 1 — Infrastructure (backend + frontend scaffold)

| Step | Deliverable | Done when |
|---|---|---|
| 1 | `pyproject.toml` with FastAPI, uvicorn; `backend/main.py` with CORS | `uvicorn backend.main:app` returns `{"status":"ok"}` |
| 2 | `GET /api/datasets` using existing `datasets/registry.py` | curl returns JSON list |
| 3 | Pydantic models in `models.py` for all request/response shapes | Schemas visible in `/docs` |
| 4 | Vite + React + TypeScript + Tailwind + shadcn/ui scaffold | `npm run dev` renders a blank page with correct fonts |
| 5 | `api/client.ts` with `fetchDatasets()` and type-safe response | Dataset names render in a `<ul>` on screen |

### Phase 2 — Layer Builder

| Step | Deliverable | Done when |
|---|---|---|
| 6 | `LayerCard.tsx` for each layer type (Linear, Conv2d, MaxPool2d, Flatten, LSTM, GRU) | All types render with correct controls |
| 7 | `LayerBuilder.tsx` using `@dnd-kit/sortable` | Drag reorders layers; list updates immediately |
| 8 | Inline type picker: changing type swaps card defaults in Zustand | Switching Conv2d → MaxPool2d replaces the card's fields |
| 9 | `POST /api/validate` + live shape annotations on each card | Shape badge updates as user edits; invalid shapes shown in red |

### Phase 3 — Training Pipeline

| Step | Deliverable | Done when |
|---|---|---|
| 10 | `POST /api/train` SSE endpoint + cancellation via stop_event | `curl` streams epoch events to terminal |
| 11 | `TrainingControls.tsx`: Train / Stop buttons + hyperparameter fields | Buttons fire correct API calls |
| 12 | `LossChart.tsx` consuming SSE stream via `fetch` + `ReadableStream` | Live chart updates every epoch in browser |
| 13 | Cancel: `DELETE /api/train/{id}` wired to Stop button | Training stops within one epoch of clicking Stop |

### Phase 4 — Visualization & Code

| Step | Deliverable | Done when |
|---|---|---|
| 14 | `ArchDiagram.tsx` using React Flow | Static diagram renders matching Python version; layer types color-coded |
| 15 | `POST /api/visualize/confusion_matrix` + `ConfusionMatrix.tsx` | Matrix PNG appears after training completes |
| 16 | `POST /api/visualize/decision_boundary` + `DecisionBoundary.tsx` | Boundary plot appears for 2D datasets |
| 17 | `GET /api/code` + `CodeViewer.tsx` with Shiki | Syntax-highlighted code updates live as config changes |

### Phase 5 — Polish

| Step | Deliverable | Done when |
|---|---|---|
| 18 | Save/load config to localStorage via Zustand `persist` middleware | Reload page → dataset, layers, and hyperparams restore exactly |
| 19 | Dataset image thumbnails (MNIST/CIFAR sample grid) | Thumbnail shown below dataset selector for image datasets |
| 20 | Dark/light mode toggle | Theme switches without layout shift |
| 21 | Comparison mode: overlay two training runs on the loss chart | Select run A and B from history → both curves visible |
| 22 | Block-level mismatch indicators on `LayerCard` | Card with null incoming shape gets red left border + warning icon next to shape badge; clearer than the banner-only approach |
| 23 | JSON config export / import | "Export" downloads `nn_builder_config.json`; "Import" restores layers + hyperparams from file; pairs with custom dataset upload |

### Phase 6 — RNN Support

| Step | Deliverable | Done when |
|---|---|---|
| 24 | ✅ `LayerCard` variants for LSTM/GRU (hidden_size, num_layers, bidirectional, rnn_dropout) | LSTM/GRU cards render with correct controls |
| 25 | Sequential datasets — at minimum: sine wave (regression warmup), multi-frequency superposition, and one real-world series (e.g., airline passengers). Loaders return `(N, seq_len, features)` arrays. `seq_len` added to `BuilderConfig`. | Sequential data loads, shapes validate, and an LSTM trains meaningfully |
| 26 | `compute_shapes()` extended for multi-step sequential shapes | Shape annotations show `(seq_len, hidden_size)` for LSTM/GRU hidden states |
| 27 | `code_gen.py` extended to emit `LSTMWrapper` / `GRUWrapper` | Generated code is complete and runnable for RNN configs |

### Phase 7 — Custom Datasets

| Step | Deliverable | Done when |
|---|---|---|
| 28 | CSV tabular upload — backend `POST /api/datasets/upload`; auto-detects binary vs. multiclass from label column; validates shape and class balance | Uploaded CSV appears in dataset selector and trains correctly |
| 29 | Frontend upload UI — file picker + preview (row count, feature count, class distribution); error messages for malformed files | User can upload, inspect, and select a custom dataset without leaving the page |
| 30 | Custom dataset persistence — uploaded datasets stored server-side per session (same session store pattern); cleared on backend restart | Custom dataset survives a Train → Visualize → Code round-trip within a session |
| 31 | Sequence data upload — `.npz` file with `X: (N, seq_len, features)` and `y: (N,)` arrays; registered alongside built-in sequential datasets | Custom time-series data trains through LSTM/GRU |

---

## 9. Key Decisions

| Decision | Rationale |
|---|---|
| SSE not WebSocket | Training is one-way (server→client only). SSE is simpler: native `EventSource`-compatible format, auto-reconnect, works through proxies. |
| `fetch` + `ReadableStream` for SSE | `EventSource` only supports GET. The train endpoint is POST (needs body). Same wire format, different client. |
| `asyncio.Queue` bridge | The training loop is sync/threaded; the SSE generator is async. The queue safely bridges the two worlds without blocking the event loop. |
| `asyncio.get_running_loop()` | Preferred over deprecated `get_event_loop()` when inside an async function. |
| Zustand not Redux | Redux boilerplate is unnecessary at this scale. Zustand gives the same store-action pattern with 90% less code. |
| shadcn/ui not MUI | shadcn copies components into your project — no version lock, fully customizable, Tailwind-native. |
| React Flow for diagram | Free zoom/pan, animated edges, and draggable nodes come for free. The alternative (custom SVG) adds 500 lines of code for less capability. |
| Images as base64 PNGs | Reuses `visualizer.py` exactly. The alternative (re-implement charts in Recharts/D3) is weeks of work for marginal gain on these specific plots. |
| Backend holds model objects | The trained `nn.Module` can't cross the HTTP boundary as JSON. `session_store.py` keeps an in-memory dict keyed by session ID; visualization endpoints look up the model there. |
| IDs stripped before backend POST | Layer card `id` fields are frontend-only UUID routing keys. The backend sees clean dicts matching the existing Python config schema. |
| Flat optional fields in `LayerConfig` | A discriminated union (one Pydantic/TS type per layer kind) would require structural changes everywhere when adding a new layer type. The flat optional pattern means adding LSTM/GRU was a single `elif` branch per function. |
| `LSTMWrapper` / `GRUWrapper` | `nn.LSTM` and `nn.GRU` return `(output, hidden_state)` tuples; `nn.Sequential` can't pipe a tuple to the next layer. Wrapper modules extract only the last timestep's output, restoring compatibility. |
| `_Unsqueeze` before each RNN layer | Tabular input is `(N, features)`; RNN expects `(N, seq_len, features)`. Auto-inserting `unsqueeze(1)` avoids a mandatory Reshape layer in every RNN config. |
| Strip `input_shape` for non-CNN networks | `train_model()` reshapes data to `(N, C, H, W)` whenever `input_shape` is set, even for pure-Linear networks. `validate.py` and `train.py` both strip `input_shape` from the config when no CNN layer types are present, so Linear networks on image datasets use `input_dim` (the flat size) rather than getting the channel count as `in_features`. |

---

## 10. Ports and Dev Setup

```
backend:  uvicorn backend.main:app --reload --port 8000
frontend: npm run dev  (Vite default: port 5173)
proxy:    Vite proxies /api/* → http://localhost:8000 in dev
CORS:     backend allows origin http://localhost:5173 in dev
```

For production, Vite builds to `frontend/dist/` and FastAPI serves it as a static mount — single process, no separate servers.

---

## 11. What the Streamlit Version Already Solved

The following design decisions are settled and carry forward unchanged:

- Config schema (all fields, types, defaults)
- Dataset registry and loaders (all 9 datasets including CIFAR-10, Fashion-MNIST)
- Layer type set: Linear, Conv2d, MaxPool2d, Flatten (+ LSTM, GRU added in React rebuild)
- `compute_shapes()` for real-time shape validation
- `train_model()` with `stop_event` + `on_epoch` callback + early stopping
- `build_model()` dispatch logic (extended in `backend/core/trainer.py` for RNN)
- `generate_code()` output
- `plot_loss_curve`, `plot_confusion_matrix`, `plot_decision_boundary` figures
- Output layer locked by task type (prevents loss/activation mismatch)
- 80/20 train/val split seeded by config
