"""Extended shape tracker supporting LSTM and GRU layers.

For configs without RNN layers, delegates to the original compute_shapes in
components.shapes — that implementation is stable and handles all CNN/Linear cases.

For RNN-containing configs:
  Shape representations used here:
    (seq_len, features) — sequence window (sequential-dataset input only)
    (features,)         — flat vector (tabular input, Linear output, RNN output)
    (C, H, W)           — spatial feature map (CNN layers, not mixed with RNN)
    None                — invalid configuration

  RNN layer output: (hidden_size * dirs,) where dirs = 2 if bidirectional else 1.
  The _Unsqueeze that build_model inserts before each RNN is transparent to shapes
  (it doesn't change the logical feature count, only adds a seq dimension).

Re-exports compute_shapes so callers can always import from this module.
"""
from __future__ import annotations

_RNN_TYPES = frozenset({"LSTM", "GRU"})
_CNN_TYPES = frozenset({"Conv2d", "MaxPool2d", "Flatten"})


def compute_shapes(config: dict) -> list[tuple | None]:
    """Return per-layer shape list (input → each hidden layer → output)."""
    layers = config.get("hidden_layers", [])
    has_rnn = any(layer.get("type") in _RNN_TYPES for layer in layers)

    if not has_rnn:
        from components.shapes import compute_shapes as _orig
        return _orig(config)

    # ── RNN path ──────────────────────────────────────────────────────────────

    if config.get("input_shape") is not None:
        # RNN + image not supported — propagate None for all positions
        return [None] * (len(layers) + 2)

    # Sequential datasets: input is (seq_len, features); otherwise flat (features,)
    seq_len = config.get("seq_len")
    current: tuple | None = (
        (seq_len, config["input_dim"]) if seq_len is not None else (config["input_dim"],)
    )
    shapes: list[tuple | None] = [current]

    for layer in layers:
        ltype = layer.get("type", "Linear")

        if current is None:
            shapes.append(None)
            continue

        if ltype in _RNN_TYPES:
            if len(current) not in (1, 2):
                current = None  # Can't apply RNN to spatial (C, H, W) input
            else:
                hidden_size   = layer.get("hidden_size") or 64
                bidirectional = layer.get("bidirectional", False)
                out_size      = hidden_size * (2 if bidirectional else 1)
                current       = (out_size,)

        elif ltype in _CNN_TYPES:
            # CNN mixed with RNN — invalid in current architecture
            current = None

        else:  # Linear
            # Linear on an un-consumed (seq_len, F) window is not supported (yet);
            # build_model requires an RNN first on sequential data.
            current = (layer["units"],) if len(current) == 1 else None

        shapes.append(current)

    # Output layer
    if config["task"] in ("binary", "regression"):
        shapes.append((1,) if current is not None else None)
    else:
        shapes.append((config["n_classes"],) if current is not None else None)

    return shapes
