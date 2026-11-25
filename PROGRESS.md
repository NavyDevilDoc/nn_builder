# Neural Network Builder — Project Progress

*Last updated: 2026-11-25*

---

## Background

This project grew out of `nn-refresher`, a 9-module notebook course that rebuilt neural network fundamentals from scratch in NumPy. That course culminated in a full MLP implementation (`nnlib/`) and a PyTorch parity test. With the fundamentals solid, the next question was: could we build an interactive tool that makes the connection between architecture choices and training behavior immediately visible?

The answer was the Neural Network Builder — first as a Streamlit prototype to prove the concept quickly, now evolving toward a React + FastAPI application with richer interactivity.

---

## Chapter 1: The Streamlit Prototype

### Phase 1 — MLP Builder

The first version launched as a pure-Python Streamlit app. The core idea: a sidebar where users configure a network layer by layer, a live architecture diagram, generated PyTorch code, and a training panel that runs the model and shows results.

The initial feature set covered:

- **7 datasets**: XOR, Moons, Spirals, Iris, Breast Cancer, Wine, and a 10,000-sample MNIST subset
- **Layer builder**: add/remove hidden Linear layers with configurable units and activation (ReLU, Tanh, Sigmoid, LeakyReLU)
- **Architecture diagram**: a Matplotlib-rendered vertical flow of boxes (color-coded by type) with shape annotations and downward arrows
- **Generated PyTorch code**: a complete, runnable script that mirrors exactly what the trainer executes — no gap between what's shown and what runs
- **Training**: Adam/SGD/Momentum optimizers, full-batch or mini-batch DataLoader, train/val split seeded for reproducibility
- **Results**: a two-panel loss/accuracy curve and either a decision boundary plot (for 2D datasets) or a confusion matrix

The session state design was careful: `last_config` tracked the config at the time of the last Train click, so the UI could warn the user when the displayed results no longer matched the current settings. A `_config_changed()` function used `json.dumps` with `default=str` to handle non-JSON-serializable types (like tuples for image shapes).

### Phase 1.5 — Linear Activation and GPU Support

Two smaller additions followed:

**Linear activation**: Added `nn.Identity` as a "Linear" activation option. This sounds trivial but matters: stacking Linear layers without nonlinearities is a useful demonstration that depth alone doesn't add representational power.

**GPU toggle**: The desktop has an NVIDIA RTX 4060 Ti. We reinstalled PyTorch with CUDA 12.6 wheels (`torch==2.14.1+cu126`, `torchvision==0.29.1+cu126`), added a device toggle in the sidebar, and plumbed `model.to(device)` and tensor `.to(device)` through the training loop. The fix that unlocked this was adding `[[tool.uv.index]]` for the PyTorch cu126 index in `pyproject.toml`.

### Phase 2 — CNN Extension

The second phase was the largest feature addition: support for convolutional layers and image datasets.

**New datasets**: Fashion-MNIST (1×28×28, 10 classes) and CIFAR-10 (3×32×32, 10 classes), each loaded as 10,000-sample subsets. CIFAR-10's `.data` array arrives in HWC order `(N, H, W, C)` from torchvision — this required a transpose to CHW `(N, C, H, W)` before flattening to `(N, 3072)`.

**New layer types**: Conv2d, MaxPool2d, and Flatten joined the existing Linear. Each has its own UI card and serialization format. A key design constraint: the sidebar enforces correct ordering (Conv/Pool before Flatten, Linear after Flatten) with a non-blocking warning rather than an error — the user can still try to train, which produces a clear error from `build_model()` if the architecture is invalid.

**Shape tracking**: `components/shapes.py` introduced `compute_shapes(config)`, a pure function that walks the layer list and tracks the spatial dimensions `(C, H, W)` through each operation. It propagates `None` for invalid transitions (Conv2d on flat input, spatial collapse to zero). This function is used by the diagram, the sidebar's shape annotations, and the builder's validation.

**Diagram update**: Layer boxes became color-coded by type: blue for input, yellow for Conv2d, red for MaxPool2d, cyan for Flatten, white for Linear, green for output. Each CNN box shows the output shape annotation.

**Code generation**: `code_gen.py` learned to emit Conv2d, BatchNorm2d, MaxPool2d, and Flatten layers, plus the reshape block (`X_tr.view(-1, C, H, W)`) for image inputs.

**Live training progress**: Replaced the blocking `st.spinner` with a background `threading.Thread`. An `on_epoch` closure writes metrics to a shared dict; the main Streamlit thread polls every 400ms with `time.sleep` + `st.rerun()`, updating a progress bar, status line, and live loss chart. This was necessary because Streamlit's execution model blocks the UI during synchronous training.

**Stop button**: The threading architecture made a true Stop button possible. Clicking it sets a `threading.Event`; `train_model()` checks the event at the top of each epoch loop and breaks cleanly, returning partial history. The success banner notes how many epochs actually ran when training was cut short.

**Early stopping**: Added patience-based early stopping directly to `train_model()`. After each epoch, if validation loss hasn't improved by more than `es_min_delta` for `es_patience` consecutive epochs, training breaks. Configured via a toggle + two sliders in the sidebar's Training section.

### Phase 2.5 — UX Refinements

Three UX improvements followed user feedback:

**Layer cards with borders**: Each layer's controls are wrapped in `st.container(border=True)`, giving clear visual separation between layers. Previously they ran together without obvious boundaries.

**Reorderable layers with highlight tracking**: Up/Down arrow buttons (↑ ↓) let users reorder layers without deleting and re-adding them. The last moved or added layer gets a blue bar across the top of its card (`_selected_layer` in session state), making it easy to track where something landed after moving.

**Inline type selector**: The static "Layer 1 — Conv2d" label became a dropdown. Changing the type swaps the layer dict for fresh defaults while preserving the internal ID, so widget keys don't collide.

**Button colors**: The Train button is styled green (`button[kind="primary"]`), the Stop button red. The red styling uses a `.red-btn-marker` sentinel div inside the Stop column with CSS `:has()` to scope the rule. The critical lesson: the selector must target `[data-testid="stColumn"]`, not `[data-testid="stVerticalBlock"]` — the vertical block selector matches ancestor containers that span the entire page, bleeding red onto all buttons. The column selector is scoped to siblings.

---

## Chapter 2: What Streamlit Taught Us

The prototype was valuable precisely because it forced every architectural decision out into the open:

- The config schema is stable and well-tested across all components
- `compute_shapes()` is correct and handles all edge cases (None propagation, Flatten no-op on flat input)
- The dataset registry pattern (DatasetInfo dataclass + loaders + suggested architectures) is clean and extensible
- `train_model()` with `stop_event` + `on_epoch` + early stopping is the right interface for the training loop
- Generated code that exactly mirrors what actually runs is a non-negotiable feature

The parts that felt wrong were all structural: polling hacks for live updates, CSS injection for styling, arrow buttons instead of drag-and-drop, `st.rerun()` on every interaction.

---

## Chapter 3: The React Rebuild

*Started: 2026-11-25*

The React rebuild lives in the same `nn_builder/` directory as the Streamlit prototype, with the backend importing `components/` and `datasets/` directly — no path dependency gymnastics needed. The full plan is in `IMPLEMENTATION_PLAN.md`.

### Phase 1 — Infrastructure

The backend is a FastAPI application in `backend/` with five route modules: `datasets`, `validate`, `train`, `visualize`, and `code`. The frontend is a Vite + React 18 + TypeScript + Tailwind v4 app in `frontend/`.

Key decisions made during Phase 1:

- **SSE over WebSocket** for training progress: training is strictly server→client, so SSE's simpler model wins. The train endpoint is POST, so `fetch` + `ReadableStream` replaces `EventSource` (which only supports GET).
- **`asyncio.Queue` bridge**: the training loop runs in a daemon thread; the SSE generator is async. `loop.call_soon_threadsafe(queue.put_nowait, ...)` bridges the two worlds safely.
- **RNN layer support built in from day one**: `LSTMWrapper` and `GRUWrapper` in `backend/core/trainer.py` make `nn.LSTM`/`nn.GRU` `nn.Sequential`-compatible by extracting only the last timestep's output `out[:, -1, :]`. An `_Unsqueeze` module is auto-inserted before each RNN layer to handle tabular input `(N, features)` → `(N, 1, features)`. Both `LayerType` (TypeScript) and Pydantic schemas include `"LSTM"` and `"GRU"` from the start.
- **Flat optional fields in `LayerConfig`**: a discriminated union would require structural changes everywhere a new layer type is added. The flat optional pattern means adding LSTM/GRU was one `elif` branch per function.

Both servers confirmed working end of Phase 1:
```
uv run uvicorn backend.main:app --reload --port 8000  → {"status":"ok"}
npm run dev  → dataset list renders at http://localhost:5173
```

### Phase 2 — Full UI (complete)

Phase 2 built the entire interactive frontend in one session. The component tree:

**State layer** (`store/useBuilderStore.ts`): Zustand store separating `layers` (with frontend-only UUIDs), `hyperparams`, `training` state, and `results`. `getConfig()` merges them and strips IDs for the backend. `LAYER_DEFAULTS` per type ensures sensible initial values when a layer is added or its type is switched.

**Layer builder** (`components/layers/`):
- `LayerTypeSelect.tsx` — type-colored `<select>` wired to `setLayerType`; swaps the card to fresh defaults while preserving the layer's UUID
- `LayerCard.tsx` — drag handle (dnd-kit `useSortable`), type selector, delete button, and type-specific controls for all 6 layer types. LSTM/GRU's `rnn_dropout` row only appears when `num_layers > 1`.
- `LayerBuilder.tsx` — `@dnd-kit` sortable container; shape validation fires via `useEffect` on every layer or dataset change, calling `POST /api/validate`; validation error banner appears inline; "Add" buttons for all types

**Config panels**:
- `HyperparamPanel.tsx` — all fields in `BuilderConfig` except `hidden_layers`: optimizer, lr, epochs, batch size (with "Full batch" toggle), weight decay, seed, early stopping (patience/min-delta rows appear only when enabled), device
- Both panels read/write Zustand `hyperparams` exclusively — no local state

**Training panel** (`TrainingPanel.tsx`): Train/Stop buttons, SSE stream consumed via async generator, progress bar (`epochsDone / totalEpochs`), live stat pills (train/val loss and accuracy). A subtle design detail: `startTraining("pending")` resets history immediately on click; the real session ID arrives in the `done` event and is written via `setSessionId()` — a separate action that doesn't reset history. `Promise.allSettled` is not needed here because cancellation uses an `AbortController` on the client side plus `DELETE /api/train/{id}` on the server.

**Results panel** (`ResultsPanel.tsx`): Hidden until after the first training run. Two Recharts `LineChart`s — loss (indigo vs amber) and accuracy (green vs cyan). A "Generate visualizations" button calls both visualization endpoints with `Promise.allSettled` so a decision-boundary failure (expected for non-2D datasets) doesn't block the confusion matrix.

**Code viewer** (`CodeViewer.tsx`): Collapsible toggle, disabled when no layers exist. Fetches `GET /api/code` via TanStack Query with the full config as the query key, so code re-generates automatically as the config changes. `react-syntax-highlighter` with `atomOneDark` for Python syntax highlighting, line numbers, and a clipboard copy button.

**Architecture diagram** (`ArchDiagram.tsx`): React Flow canvas with input/output sentinel nodes and one node per hidden layer. Nodes are color-coded by type (matching the layer cards). Edge labels show the tensor shape entering each layer, pulled from the same `/api/validate` endpoint used by the layer cards. The diagram grows vertically with the layer count; non-interactive (you configure layers in the left column, not by dragging diagram nodes).

**Layout**: Two-column split at `xl` breakpoint — left column scrolls the config (dataset selector, layer builder, hyperparams), right column scrolls the outputs (diagram, training, results, code). Both columns scroll independently.

### Key bug fixed: image dataset + Linear layers

`train_model()` reshapes data to `(N, C, H, W)` whenever `input_shape` is set in the config — even for a pure-Linear network on MNIST. `build_model()` then uses `shapes[i][0]` for the first Linear layer's `in_features`, which pulls the channel count (1) rather than the flattened dimension (784), causing a shape mismatch at runtime.

**Fix**: both `backend/routes/train.py` and `backend/routes/validate.py` now check whether any CNN layer types (`Conv2d`, `MaxPool2d`, `Flatten`) are present. If not, `input_shape` is stripped from the config before it reaches `build_model()` and `train_model()`, so both treat the data as flat tabular with `input_dim=784`. CNN pipelines are unaffected.

### What's next

- **localStorage persistence**: config survives page reload
- **Dataset image thumbnails**: sample grid for MNIST/CIFAR/Fashion-MNIST shown below the dataset selector
- **Sequential datasets**: sine wave, ECG, or similar — required to demonstrate LSTM/GRU meaningfully (seq_len=1 is architecturally valid but pedagogically degenerate)
- **Training run comparison**: overlay two runs on the loss chart
