# nn_builder

An interactive neural network builder: configure a PyTorch architecture layer by layer, train it on a built-in dataset, and watch the results update in real time.

**Primary goal:** make the connection between architecture choices and training behaviour immediately visible — a tool for learning and experimentation, not production deployment.

---

## Overview

The project has two working interfaces that share all Python ML logic:

| Interface | Stack | Status |
|---|---|---|
| **React app** | FastAPI + React 18 + TypeScript + Tailwind | Active development |
| **Streamlit prototype** | Streamlit + Matplotlib | Complete (reference) |

---

## Quick start — React app

```bash
# Terminal 1 — FastAPI backend
uv run uvicorn backend.main:app --reload --port 8000

# Terminal 2 — Vite dev server
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Quick start — Streamlit prototype

```bash
uv run streamlit run app.py
```

Open `http://localhost:8501`.

---

## React app features

**Left panel — configuration**
- Dataset selector (9 built-in datasets)
- Layer builder: drag-and-drop reordering, all layer types (Linear, Conv2d, MaxPool2d, Flatten, LSTM, GRU)
- Live shape validation: each card shows the tensor shape entering it; invalid shapes highlighted immediately
- Hyperparameter panel: optimizer, learning rate, epochs, batch size, weight decay, seed, early stopping

**Right panel — outputs**
- Architecture diagram (React Flow): type-colored layer nodes, shape-annotated edges
- Training: SSE-streamed live progress bar + loss/accuracy stat pills
- Results: interactive Recharts loss and accuracy curves, confusion matrix, decision boundary
- Generated code: collapsible Python viewer with syntax highlighting and clipboard copy

---

## Datasets

| Name | Task | Input | Classes |
|---|---|---|---|
| XOR | Binary | 2 features | 2 |
| Moons | Binary | 2 features | 2 |
| Spirals | Binary | 2 features | 2 |
| Iris | Multiclass | 4 features | 3 |
| Breast Cancer | Binary | 30 features | 2 |
| Wine | Multiclass | 13 features | 3 |
| MNIST | Multiclass | 784 (28×28×1) | 10 |
| Fashion-MNIST | Multiclass | 784 (28×28×1) | 10 |
| CIFAR-10 | Multiclass | 3072 (32×32×3) | 10 |

Scikit-learn datasets are StandardScaler-normalised. Image pixels divided by 255. All splits are stratified 80/20, seeded by the Seed control.

---

## Project structure

```
nn_builder/
├── app.py                      # Streamlit entry point (prototype)
├── backend/                    # FastAPI application
│   ├── main.py                 # App, CORS, router registration
│   ├── models.py               # Pydantic request/response schemas
│   ├── session_store.py        # In-memory model store (keyed by session UUID)
│   ├── core/
│   │   ├── trainer.py          # LSTMWrapper, GRUWrapper, extended build_model()
│   │   └── shapes.py           # Extended compute_shapes() with RNN support
│   └── routes/
│       ├── datasets.py         # GET /api/datasets, /api/datasets/{name}/sample
│       ├── validate.py         # POST /api/validate
│       ├── train.py            # POST /api/train (SSE), DELETE /api/train/{id}
│       ├── visualize.py        # POST /api/visualize/confusion_matrix & /decision_boundary
│       └── code.py             # GET /api/code
├── components/                 # Shared Python ML logic (Streamlit + FastAPI)
│   ├── trainer.py              # build_model / train_model
│   ├── shapes.py               # compute_shapes()
│   ├── code_gen.py             # generate_code()
│   ├── visualizer.py           # plot_* figures
│   └── sidebar.py              # Streamlit sidebar (prototype only)
├── datasets/
│   ├── registry.py             # meta() / load() / list()
│   ├── synthetic.py            # XOR, moons, spirals generators
│   └── builtin.py              # sklearn + torchvision wrappers
├── frontend/                   # Vite + React 18 + TypeScript + Tailwind v4
│   ├── src/
│   │   ├── App.tsx             # Two-column split layout
│   │   ├── api/client.ts       # Typed API wrappers; streamTraining() via ReadableStream
│   │   ├── store/useBuilderStore.ts  # Zustand: layers, hyperparams, training, results
│   │   ├── types/config.ts     # TypeScript interfaces (single source of truth)
│   │   └── components/
│   │       ├── layers/         # LayerCard, LayerBuilder, LayerTypeSelect
│   │       ├── HyperparamPanel.tsx
│   │       ├── TrainingPanel.tsx
│   │       ├── ResultsPanel.tsx
│   │       ├── CodeViewer.tsx
│   │       └── ArchDiagram.tsx
│   ├── vite.config.ts          # Proxy /api → localhost:8000
│   └── package.json
├── data/                       # torchvision download cache (gitignored)
├── pyproject.toml
├── IMPLEMENTATION_PLAN.md      # Phased build plan with acceptance criteria
├── PROGRESS.md                 # Narrative history and design rationale
└── SESSION_HANDOFF.md          # Rolling baton-pass for session continuity
```

---

## Key design decisions

| Decision | Rationale |
|---|---|
| SSE not WebSocket | Training is one-way push. SSE is simpler and works through proxies. |
| `fetch` + `ReadableStream` for SSE | `EventSource` only supports GET; the train endpoint is POST. |
| `asyncio.Queue` bridge | Training loop is sync/threaded; SSE generator is async. The queue crosses the boundary safely. |
| `LSTMWrapper` / `GRUWrapper` | `nn.LSTM`/`nn.GRU` return tuples; `nn.Sequential` can't pipe a tuple. Wrappers extract last-timestep output. |
| Flat optional fields in `LayerConfig` | Adding a new layer type requires one `elif` branch, not a schema restructure. |
| Backend holds model objects | `nn.Module` can't cross HTTP as JSON. Session store bridges training and visualization endpoints. |

---

## Roadmap

See `IMPLEMENTATION_PLAN.md` for the full phased plan. Upcoming:

- **Phase 5 (Polish):** localStorage config persistence, block-level shape mismatch indicators, JSON export/import, dataset thumbnails
- **Phase 6 (RNN):** sequential datasets (sine wave, real-world series), multi-step shape tracking, RNN code generation
- **Phase 7 (Custom datasets):** CSV tabular upload, `.npz` sequence upload
