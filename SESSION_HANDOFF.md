# Session Handoff
*Last updated: 2026-11-25*

---

## 1. Where we left off

**Phases 1–5 complete. Phase 8 planned. Both servers confirmed working.**

```
# Backend — run from nn_builder/
uv run uvicorn backend.main:app --reload --port 8000
# Confirm: curl http://localhost:8000/api/datasets  → JSON with 9 datasets

# Frontend — run from nn_builder/frontend/
npm run dev
# Confirm: http://localhost:5173  → split-panel UI, both columns scroll independently
```

**What was built this session:**

- `useBuilderStore.ts` — `insertLayerBefore`, `loadConfig`, `weightDecayEnabled`, `fixedSeed`, `setWeightDecayEnabled`, `setFixedSeed`; `getConfig()` zeroes weight_decay when disabled; `persist` middleware now also saves `weightDecayEnabled` and `fixedSeed`
- `LayerBuilder.tsx` — `InsertDivider` between every card (closed: thin line + `+` icon; open: inline type picker); uses `Fragment` so dnd-kit only sees `LayerCard` nodes as sortable
- `LayerCard.tsx` — red left border + `AlertTriangle` icon when `shape === null`; number inputs left-aligned
- `HyperparamPanel.tsx` — weight decay and seed rows now have `Enabled`/`Fixed` toggles matching the early stopping pattern; number inputs left-aligned
- `TrainingPanel.tsx` — randomizes seed at train time when `fixedSeed` is off
- `ArchDiagram.tsx` — fixed height `h-72`; `zoomOnScroll={false}` so wheel scrolls the page
- `App.tsx` — `h-screen overflow-hidden` for independent column scrolling; dark/light toggle (`Sun`/`Moon`); export (`Download`) and import (`Upload`) in header; dataset thumbnail via TanStack Query (`enabled: !!meta?.input_shape`, `staleTime: Infinity`)
- `client.ts` — `fetchDatasetSample(name)` added
- `IMPLEMENTATION_PLAN.md` — Phase 5 steps marked complete, Step 21 dropped, Phase 8 (Steps 32–38) added
- `PROGRESS.md` — Phase 5 additions documented; Phase 8 rationale written; "What's next" updated

**Gotchas from this session:**
- Zustand selector returning an object literal (`(s) => ({ a: s.a, b: s.b })`) causes infinite re-renders. Use separate scalar selectors or `useBuilderStore.getState()` inside event handlers.
- Tailwind v4 uses `@custom-variant dark` in CSS — no `tailwind.config.ts`. The variant is already wired in `src/index.css`.

---

## 2. Where we're starting

**Phase 6, Step 25: Sequential datasets.**

The LSTM/GRU layer types exist and train, but on tabular data with `seq_len=1` — architecturally valid, pedagogically useless. Meaningful RNN demos require real sequence data.

Minimum deliverable:
1. **Three dataset loaders** in `datasets/` returning `(N, seq_len, features)` arrays:
   - Sine wave (single frequency, regression warmup)
   - Multi-frequency superposition (harder regression)
   - Airline passengers or similar real-world series (classic LSTM benchmark)
2. **`seq_len` added to `BuilderConfig`** — both Pydantic (`backend/models.py`) and TypeScript (`types/config.ts`)
3. **Backend `train_model()` update**: skip the `_Unsqueeze` auto-insert when `seq_len > 1` (the data already has the sequence dimension)
4. **`datasets/registry.py`** — register the new loaders, add them to `meta()` and `list()`

Confirm: select a sequential dataset, add an LSTM layer, click Train → loss decreases meaningfully.

---

## 3. Applicable supporting documentation

| Document | Location | Purpose |
|---|---|---|
| **IMPLEMENTATION_PLAN.md** | `nn_builder/IMPLEMENTATION_PLAN.md` | Phase 6 steps 25–27 with acceptance criteria; Phase 8 graph mode sketch |
| **CLAUDE.md** | `nn_builder/CLAUDE.md` | RNN layer design; `_Unsqueeze` rationale; `train_model()` signature contract |
| **datasets/builtin.py** | `nn_builder/datasets/builtin.py` | Existing loader pattern to follow for new sequential loaders |
| **datasets/registry.py** | `nn_builder/datasets/registry.py` | Where to register new datasets |
| **backend/core/trainer.py** | `nn_builder/backend/core/trainer.py` | `LSTMWrapper`, `GRUWrapper`, `_Unsqueeze` — the code to conditionally bypass |
