# Session Handoff
*Last updated: 2026-11-25*

---

## 1. Where we left off

**Phase 6, Step 25 complete: sequential datasets + a `regression` task type, verified end-to-end in the browser.**

```
# Backend — run from nn_builder/
uv run uvicorn backend.main:app --reload --port 8000
# Frontend — run from nn_builder/frontend/
npm run dev
# Confirm: select "sine" → "+ LSTM" → Train → stats read "train R² / val R²" ≈ 0.99;
#          "Generate visualizations" → forecast-vs-actual plot.
```

**What was built:**
- `datasets/sequential.py` (sine, multi_sine, airline loaders + `make_windows`), `datasets/airline_passengers.csv`
- `registry.load(name, seed, seq_len)` — windowing, chronological split, train-span normalization
- `backend/core/inputs.py` — single source of truth for input layout, used by train / validate / code routes
- Regression in `components/trainer.py`, `shapes.py`, `code_gen.py`, `visualizer.py` (`plot_predictions`); `POST /api/visualize/predictions`
- Frontend: seq_len input in the dataset selector (resets on dataset change), task-aware R² labels (`src/lib/metrics.ts`), forecast plot, persist `merge` that deep-merges `hyperparams`
- Fixed a pre-existing `tsc` error in `CodeViewer.tsx` (`npm run build` now passes)

**Gotchas:**
- **Laptop only:** `uv sync` needs `--native-tls` (TLS interception). Python/Node clients intermittently get `ECONNRESET` on loopback responses ≳70 KB — reproduced with a bare `http.server`, so it's environmental, not the app. It can make "Generate visualizations" fail through the Vite proxy; retrying works. curl is unaffected.
- PowerShell 5.1 strips embedded `"` from arguments to native exes — put JSON for curl in a file and use `--data-binary @file`.
- On sequential datasets the first layer must be LSTM/GRU (Linear-first shows as invalid shape).

---

## 2. Where we're starting

**Phase 6, Step 26: multi-step sequential shapes.**

1. Allow a per-timestep `Linear` *before* the first RNN on sequential data (`nn.Linear` acts on the last dim: `(N, T, F)` → `(N, T, units)`; `BatchNorm1d` needs a transpose or must be disallowed there).
2. Extend `backend/core/shapes.py` to report `(seq_len, features)` / `(seq_len, hidden)` shapes along the sequence path; the layer card and diagram should show them.
3. Decide whether stacked RNN layers should pass full sequences (return_sequences-style wrapper option) instead of the current `_Unsqueeze` degenerate path.

Confirm: on `sine`, `Linear(16) → LSTM(32) → Linear(8)` validates with sensible shapes on every card and trains.

---

## 3. Applicable supporting documentation

| Document | Location | Purpose |
|---|---|---|
| **IMPLEMENTATION_PLAN.md** | `nn_builder/IMPLEMENTATION_PLAN.md` | Steps 26–27 acceptance criteria |
| **PROGRESS.md** | `nn_builder/PROGRESS.md` | Step 25 design notes and airline LSTM-vs-MLP results |
| **backend/core/inputs.py** | `nn_builder/backend/core/inputs.py` | Input-layout rules Step 26 must extend |
| **backend/core/shapes.py** / **trainer.py** | `nn_builder/backend/core/` | RNN shape tracking and `build_model` sequential path |
