# Session Handoff
*Last updated: 2026-11-25*

---

## 1. Where we left off

**Phase 6 complete (Steps 25–27) on branch `phase6-sequential` — not yet merged to `master` or pushed.**

```
# Backend — run from nn_builder/
uv run uvicorn backend.main:app --reload --port 8000
# Frontend — run from nn_builder/frontend/
npm run dev
# Confirm: sine → + Linear, + LSTM, + LSTM, + Linear → cards read
#   [20×1] → [20×16] → [20×32] → [16]; first LSTM says "Returns [20 × 32] — full sequence";
#   Train → val R² ≈ 0.99; open Generated Code → LSTMWrapper(..., return_sequences=True/False).
```

**What was built (Steps 26–27):**
- `backend/core/rnn_plan.py` — `plan_rnn()`: per-layer shapes, inferred `return_sequences`, `_Unsqueeze` placement, error messages. Read by `compute_shapes`, `build_model`, and code gen.
- `backend/core/trainer.py` — wrappers take `return_sequences`; `SeqBatchNorm1d`; bidirectional `_last_step` fix.
- `backend/core/code_gen.py` — RNN code gen via `inspect.getsource`; `components/code_gen.py` gained `body`/`preamble` hooks and a sequential data section.
- `LayerCard.tsx` — RNN cards state what was inferred; validation shows specific RNN errors.
- `ArchDiagram.tsx` — panel grows with layer count (288→576 px cap); viewport computed directly and re-fit on add/remove; top-aligns past the 0.5 zoom floor.

**Gotchas:**
- **Laptop only:** `uv sync --native-tls`; Python/Node loopback responses ≳70 KB intermittently reset (environmental — retry).
- PowerShell 5.1: `git commit -F <file>` for multi-line messages; JSON for curl goes in a file (`--data-binary @file`).

---

## 2. Where we're starting

**Decide: merge `phase6-sequential` → `master` (and push), then Phase 7, Step 28: CSV tabular upload.**

(The arch-diagram sizing fix is done — see PROGRESS.md.)

Step 28: `POST /api/datasets/upload` — parse CSV, auto-detect binary vs multiclass from the label column (and consider regression for a float label now that the task exists), validate shape/class balance, register alongside built-ins (session-scoped, Step 30).

---

## 3. Applicable supporting documentation

| Document | Location | Purpose |
|---|---|---|
| **IMPLEMENTATION_PLAN.md** | `nn_builder/IMPLEMENTATION_PLAN.md` | Phase 7 Steps 28–31 acceptance criteria |
| **PROGRESS.md** | `nn_builder/PROGRESS.md` | Phase 6 design notes and verification |
| **datasets/registry.py** | `nn_builder/datasets/registry.py` | Where uploaded datasets must register (`meta`, `load`) |
| **backend/core/inputs.py** | `nn_builder/backend/core/inputs.py` | Input-layout rules an uploaded `.npz` (Step 31) must satisfy |
