"""Extended shape tracker supporting LSTM and GRU layers.

For configs without RNN layers, delegates to the original compute_shapes in
components.shapes — that implementation is stable and handles all CNN/Linear cases.

For RNN-containing configs, shapes come from backend/core/rnn_plan.py (the
same plan build_model and code generation use):
    (seq_len, features) — sequence: dataset input, per-timestep Linear output,
                          or an RNN that returns its full sequence
    (features,)         — flat vector: tabular input, or after the last RNN
    None                — invalid configuration

Re-exports compute_shapes so callers can always import from this module.
"""
from __future__ import annotations

from backend.core.rnn_plan import RNN_TYPES, plan_rnn


def compute_shapes(config: dict) -> list[tuple | None]:
    """Return per-layer shape list (input → each hidden layer → output)."""
    layers = config.get("hidden_layers", [])
    has_rnn = any(layer.get("type") in RNN_TYPES for layer in layers)

    if not has_rnn:
        from components.shapes import compute_shapes as _orig
        return _orig(config)

    steps = plan_rnn(config)
    shapes: list[tuple | None] = [steps[0].in_shape if steps else None]
    shapes += [s.out_shape for s in steps]

    final = shapes[-1]
    if final is None:
        shapes.append(None)
    elif config["task"] in ("binary", "regression"):
        shapes.append((1,))
    else:
        shapes.append((config["n_classes"],))
    return shapes
