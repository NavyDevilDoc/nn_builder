# CLAUDE.md — nn_builder

An interactive neural network builder: configure, train, and visualize PyTorch models in the browser.
Streamlit prototype is complete. React + FastAPI rebuild is in progress in the same directory.
Full spec: `IMPLEMENTATION_PLAN.md`. History: `PROGRESS.md`.

## Audience

- **User:** Jeremy — MS in applied mathematics, early career on neural networks, recent years in NLP. Comfortable with linear algebra, calculus, and PyTorch. Interested in clean architecture and learning-by-building, not in production deployment.
- **Tone:** Direct and technical. No hand-holding on ML fundamentals; do explain non-obvious engineering choices (async patterns, React state model, CSS scoping quirks).

## Project state

Both the Streamlit prototype and the React + FastAPI rebuild live in the same `nn_builder/` directory.
`components/` and `datasets/` are shared — the FastAPI backend imports them directly with no path setup.

```
# Streamlit prototype (complete)
# Run from the repo root (path differs between Jeremy's desktop and laptop)
uv run streamlit run app.py          # http://localhost:8501

# FastAPI backend (React rebuild)
uv run uvicorn backend.main:app --reload --port 8000   # http://localhost:8000/docs

# React frontend (React rebuild)
cd frontend
npm run dev   # http://localhost:5173
```

## Session handoff protocol

`SESSION_HANDOFF.md` is a **rolling baton-pass document** — not a history log.
Rewrite it completely at the end of every session. Never append to it. Three parts only:

1. **Where we left off** — final state: what was completed, the command that confirms it, any gotchas.
2. **Where we're starting** — the concrete first task for the next session, specific enough to begin without re-reading the conversation.
3. **Applicable supporting documentation** — paths to specs or reference files directly relevant to the next session.

Anything worth preserving beyond one session goes in `CLAUDE.md`, `IMPLEMENTATION_PLAN.md`, or `PROGRESS.md` — not in the handoff.

## Working rules

### General
- Keep dependencies minimal. Ask before adding a package to either `pyproject.toml` or `package.json`.
- Use `pathlib` for all Python paths. No hardcoded Windows path strings in source code.
- Platform is Windows (desktop with NVIDIA RTX 4060 Ti). Backend runs on `uv`; frontend on Node/npm.
- The backend imports `components/` and `datasets/` directly — they are siblings of `backend/` within the same `nn_builder/` project. No path setup is needed. The Streamlit prototype's modules are the source of truth for all ML logic.

### Python backend
- The existing `components/` and `datasets/` modules are stable. Do not modify them without a clear reason.
- FastAPI routes are thin wrappers. Business logic stays in `components/` or `backend/core/`; routes handle HTTP concerns only (request parsing, response serialization, SSE framing).
- `train_model()` signature is the contract: `(model, config, X_train, y_train, X_val, y_val, on_epoch=None, stop_event=None)`. Do not change it.
- Use `asyncio.Queue` to bridge the synchronous training thread and the async SSE generator. Never block the event loop.
- Session store (`session_store.py`) holds in-memory model objects keyed by UUID. Keep it simple — no persistence, no TTL in Phase 1.
- Use `asyncio.get_running_loop()` (not the deprecated `get_event_loop()`) when inside an async context.

### TypeScript / React frontend
- Config schema lives in `types/config.ts`. It is the single source of truth for the shape of data sent to the backend. Keep it in sync with `backend/models.py`.
- Layer card `id` fields are frontend-only UUID routing keys. Strip them before POSTing to the backend.
- Zustand store owns all mutable app state. No `useState` for data that is shared across components.
- `@dnd-kit/sortable` for drag-and-drop. Do not implement custom drag logic.
- Recharts for all charts. Do not add a second charting library.
- shadcn/ui for all UI primitives. Extend with Tailwind utilities; do not add component libraries that conflict with Tailwind.
- TanStack Query (`@tanstack/react-query`) for all server state. `streamTraining()` in `api/client.ts` uses `fetch` + `ReadableStream` (not `EventSource`) because the train endpoint is POST.

### CSS / styling
- Tailwind utility classes only. No inline `style={{}}` except for truly dynamic values (e.g., progress bar width).
- Dark mode is a first-class requirement. Use Tailwind's `dark:` variants throughout; never hardcode light colors.
- `darkMode: 'class'` in `tailwind.config.ts`; `<html class="dark">` in `index.html`.

### RNN layer support
- `backend/core/rnn_plan.py` is the single source of truth for RNN configs. `compute_shapes()`, `build_model()`, and RNN code generation all read `plan_rnn(config)`; never re-derive sequence logic elsewhere.
- Sequence rule: the time dimension is kept until the **last** LSTM/GRU. An RNN returns its full sequence iff another RNN follows (inferred, not user-set); the last one returns its final timestep. Linear layers that see a sequence act per timestep; their batch norm is `SeqBatchNorm1d`. `_Unsqueeze` is inserted only when an RNN receives flat input (non-sequential data).
- `LSTMWrapper` / `GRUWrapper` (`backend/core/trainer.py`) take `return_sequences`. Bidirectional final state = forward at `t=-1` concatenated with backward at `t=0` (`_last_step`).
- RNN code generation (`backend/core/code_gen.py`) emits the helper classes via `inspect.getsource`, so they must stay self-contained (torch / torch.nn only).
- RNN layers cannot be mixed with CNN layers (Conv2d, MaxPool2d, Flatten).
- Input layout (image vs flat, sequence vs flattened window) is decided only in `backend/core/inputs.py`. Train, validate, and code routes must all go through it — never re-derive it in a route.

### Task types
- `binary` | `multiclass` | `regression`. Regression: `Linear(→1)`, `MSELoss`, y float32 `(N, 1)`. History keys stay `train_acc`/`val_acc` for every task; for regression they hold R². Keep `components/trainer.py` and `components/code_gen.py` in sync when touching task handling.
- Sequential splits are chronological (never shuffled), normalized with training-span statistics only.

## Key decisions (do not re-litigate)

| Decision | Rationale |
|---|---|
| SSE not WebSocket | Training is one-way push. SSE is simpler: native `EventSource`-compatible format, auto-reconnect, works through proxies. |
| `fetch` + `ReadableStream` for SSE | `EventSource` only supports GET; the train endpoint is POST. The SSE wire format is identical. |
| `asyncio.Queue` bridge | The training loop is sync/threaded; the SSE generator is async. The queue safely bridges the two worlds without blocking the event loop. |
| Backend holds model objects | `nn.Module` can't cross HTTP as JSON. Session store is the bridge between training and visualization endpoints. |
| Images as base64 PNGs from backend | Reuses `visualizer.py` exactly. Re-implementing in D3/canvas is weeks of work for marginal gain. |
| Output layer locked by task type | Prevents the classic beginner mistake of mismatching loss and final activation. |
| `nn.Sequential` only | Covers all MLP, standard CNN, and (with wrappers) RNN architectures in scope. No skip connections. |
| 80/20 train/val split seeded by config | Reproducibility across Train clicks with the same config. |
| Flat optional fields in `LayerConfig` | A discriminated union (one Pydantic/TS type per layer kind) would require structural changes to add a new layer type. The flat optional pattern means adding LSTM/GRU was one `elif` branch. |
| `LSTMWrapper` / `GRUWrapper` | `nn.LSTM` and `nn.GRU` return `(output, hidden_state)` tuples; `nn.Sequential` can't pipe a tuple to the next layer. The wrappers extract only the last timestep's output, restoring compatibility. |
| `_Unsqueeze` only for flat RNN input | Tabular input is `(N, features)`; RNN expects `(N, seq_len, features)`. Auto-inserting `unsqueeze(1)` avoids a mandatory Reshape layer. Sequential datasets never need it. |
| Inferred `return_sequences` | An RNN returns its full sequence iff another RNN follows. Removes a checkbox users could set inconsistently; the layer card states what was inferred. |
| Generated code uses `inspect.getsource` on the real wrappers | Generated script and trained model cannot drift; verified by loading `build_model` weights into the generated `Network` and asserting identical outputs. |
| Regression reuses `*_acc` history keys (R²) | Keeps the SSE wire format, store, and charts task-agnostic; the frontend relabels by task. |
| Windows flattened for non-RNN nets on sequential data | Gives an MLP baseline on identical windows instead of an error — directly comparable to the LSTM. |
| Vite proxy `/api` → `localhost:8000` | Eliminates CORS during development. The backend still carries `CORSMiddleware` for production serving (single process, FastAPI serves the built `frontend/dist/`). |
