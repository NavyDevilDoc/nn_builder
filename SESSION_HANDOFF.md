# Session Handoff
*Last updated: 2026-11-26*

---

## 1. Where we left off

**Phase 6 complete (Steps 25–27) and merged to `master` (`2a9f2c3`, pushed). Phase 7 (custom datasets) deliberately deferred — Phase 8 goes next.**

```
# Backend — run from nn_builder/
uv run uvicorn backend.main:app --reload --port 8000
# Frontend — run from nn_builder/frontend/
npm run dev
# Confirm: sine → + Linear, + LSTM, + LSTM, + Linear → cards read
#   [20×1] → [20×16] → [20×32] → [16]; Train → val R² ≈ 0.99;
#   Generated Code shows LSTMWrapper(..., return_sequences=True/False).
```

Phase 7 → Phase 8 has no dependency: graph mode reads datasets through the existing registry (Step 37's acceptance target, CIFAR-10, is built in), and `train_model()` accepts any `nn.Module` — it only reads `config` to reshape inputs. Uploaded datasets will appear in graph mode for free when Phase 7 lands.

**Gotchas:**
- **Laptop only:** `uv sync --native-tls`; Python/Node loopback responses ≳70 KB intermittently reset (environmental — retry).
- PowerShell 5.1: `git commit -F <file>` for multi-line messages; JSON for curl goes in a file (`--data-binary @file`).

---

## 2. Where we're starting

**Phase 8, Step 32: graph types + `useGraphStore.ts` — co-designed with Step 36's `GraphConfig` Pydantic schema.**

The plan orders frontend (32–35) before backend (36–38), but CLAUDE.md requires `types/config.ts` and `backend/models.py` to stay in sync, so define the wire schema once on both sides first:

- `GraphNode`: `id`, `kind` (`input` | `output` | `layer` | `add` | `concat`), `layer?: LayerConfig` (reuse the flat optional layer fields), `position` (frontend-only, like layer-card `id` — strip before POST? decide).
- `GraphEdge`: `source`, `target`, and for merge nodes a `targetHandle` (input slot index — `Concat` order matters).
- `useGraphStore.ts`: separate Zustand store; the sequential `useBuilderStore` stays untouched. Decide whether it persists (sequential store does).

Scope questions to settle at the start:
- LSTM/GRU nodes in graph mode — recommend **excluded in v1** (the `rnn_plan` sequence rules assume a chain).
- Where input-layout resolution happens for graphs — must still go through `backend/core/inputs.py`.
- Merge semantics: `Add` requires identical shapes; `Concat` joins on channel dim (images) / feature dim (flat). Shape errors should name the offending edge.

When Phase 8 lands, update CLAUDE.md's "`nn.Sequential` only — no skip connections" key decision.

---

## 3. Applicable supporting documentation

| Document | Location | Purpose |
|---|---|---|
| **IMPLEMENTATION_PLAN.md §8, Phase 8** | `nn_builder/IMPLEMENTATION_PLAN.md` | Steps 32–38 deliverables and acceptance criteria (Path A design) |
| **types/config.ts** | `nn_builder/frontend/src/types/config.ts` | `LayerConfig` to reuse inside graph nodes |
| **backend/models.py** | `nn_builder/backend/models.py` | Where `GraphConfig` goes; must mirror the TS types |
| **backend/core/inputs.py** | `nn_builder/backend/core/inputs.py` | Input-layout rules the graph builder must also go through |
| **ArchDiagram.tsx** | `nn_builder/frontend/src/components/ArchDiagram.tsx` | Existing React Flow v11 usage + measurement/fitView quirks (see PROGRESS.md "Arch diagram sizing fix") |
