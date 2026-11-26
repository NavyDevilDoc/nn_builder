"""Extended model builder supporting Linear, Conv2d, MaxPool2d, Flatten, LSTM, and GRU.

For configs with no RNN layers, delegates directly to the original
components.trainer.build_model — that code is stable and well-tested.

For RNN-containing configs:
  - LSTMWrapper / GRUWrapper make nn.LSTM / nn.GRU compatible with nn.Sequential
    by discarding the hidden-state return value and yielding only the last
    timestep's output: (N, seq_len, in) → (N, hidden_size).
  - Sequential datasets (config["seq_len"] set) deliver (N, seq_len, features)
    directly; the first layer must then be the RNN that consumes it.
  - _Unsqueeze is auto-inserted before every other RNN layer to handle flat
    input: (N, features) → (N, 1, features).  This covers tabular datasets and
    RNNs stacked after the first one (whose input is the previous RNN's last
    hidden state).  A 1-step sequence is architecturally valid but
    pedagogically degenerate; use num_layers for real stacking.
  - CNN layers (Conv2d, MaxPool2d) cannot be mixed with RNN layers; build_model
    raises ValueError if both are present.

train_model is re-exported unchanged from components.trainer.
"""
from __future__ import annotations

import torch.nn as nn
from components.trainer import train_model, _ACT  # noqa: F401  (re-export train_model)

_RNN_TYPES = frozenset({"LSTM", "GRU"})
_CNN_TYPES = frozenset({"Conv2d", "MaxPool2d", "Flatten"})


# ── Sequential-compatible RNN wrappers ────────────────────────────────────────

class _Unsqueeze(nn.Module):
    """Promote (N, features) → (N, 1, features) for tabular-to-RNN transition."""

    def forward(self, x):
        return x.unsqueeze(1)


class LSTMWrapper(nn.Module):
    """nn.LSTM wrapped for nn.Sequential: returns last-timestep output only.

    Input:  (N, seq_len, input_size)
    Output: (N, out_size)  where out_size = hidden_size * (2 if bidirectional else 1)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_size = hidden_size * (2 if bidirectional else 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out[:, -1, :]


class GRUWrapper(nn.Module):
    """nn.GRU wrapped for nn.Sequential: returns last-timestep output only.

    Input:  (N, seq_len, input_size)
    Output: (N, out_size)  where out_size = hidden_size * (2 if bidirectional else 1)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_size = hidden_size * (2 if bidirectional else 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return out[:, -1, :]


# ── Extended build_model ──────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """Return an nn.Sequential from config, supporting all layer types.

    For configs without RNN layers, delegates to the original build_model in
    components.trainer (unchanged, stable).  For RNN-containing configs,
    builds the Sequential here with LSTMWrapper / GRUWrapper.
    """
    layers = config.get("hidden_layers", [])
    has_rnn = any(layer.get("type") in _RNN_TYPES for layer in layers)

    if not has_rnn:
        from components.trainer import build_model as _orig_build
        return _orig_build(config)

    # ── RNN path ──────────────────────────────────────────────────────────────

    if not layers:
        raise ValueError("The network must have at least one hidden layer.")

    if config.get("input_shape") is not None:
        raise ValueError("RNN layers are not supported with image input shapes.")

    has_cnn = any(layer.get("type") in _CNN_TYPES for layer in layers)
    if has_cnn:
        raise ValueError(
            "Conv2d, MaxPool2d, and Flatten layers cannot be mixed with LSTM/GRU "
            "in the current architecture."
        )

    sequential_input = config.get("seq_len") is not None
    if sequential_input and layers[0].get("type") not in _RNN_TYPES:
        raise ValueError(
            "On sequential datasets the first layer must be LSTM or GRU "
            "(it consumes the (seq_len, features) window). Linear layers can follow it."
        )

    pytorch_layers: list[nn.Module] = []
    prev = config["input_dim"]

    for idx, layer in enumerate(layers):
        ltype = layer.get("type", "Linear")

        if ltype in _RNN_TYPES:
            hidden_size  = layer.get("hidden_size") or 64
            num_layers   = layer.get("num_layers") or 1
            bidirectional = layer.get("bidirectional", False)
            rnn_dropout  = layer.get("rnn_dropout", 0.0)

            # Data already (N, seq_len, F) for the first RNN on sequential data;
            # otherwise promote flat input: (N, prev) → (N, 1, prev)
            if not (sequential_input and idx == 0):
                pytorch_layers.append(_Unsqueeze())

            if ltype == "LSTM":
                wrapper = LSTMWrapper(prev, hidden_size, num_layers, bidirectional, rnn_dropout)
            else:
                wrapper = GRUWrapper(prev, hidden_size, num_layers, bidirectional, rnn_dropout)

            pytorch_layers.append(wrapper)
            prev = wrapper.out_size

        else:  # Linear
            pytorch_layers.append(nn.Linear(prev, layer["units"]))
            if layer.get("batch_norm", False):
                pytorch_layers.append(nn.BatchNorm1d(layer["units"]))
            pytorch_layers.append(_ACT[layer["activation"]]())
            if layer.get("dropout", False) and layer.get("dropout_rate", 0.0) > 0:
                pytorch_layers.append(nn.Dropout(layer["dropout_rate"]))
            prev = layer["units"]

    # Output layer
    if config["task"] == "binary":
        pytorch_layers.append(nn.Linear(prev, 1))
        pytorch_layers.append(nn.Sigmoid())
    elif config["task"] == "regression":
        pytorch_layers.append(nn.Linear(prev, 1))
    else:
        pytorch_layers.append(nn.Linear(prev, config["n_classes"]))

    return nn.Sequential(*pytorch_layers)
