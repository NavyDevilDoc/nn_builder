"""Extended model builder supporting Linear, Conv2d, MaxPool2d, Flatten, LSTM, and GRU.

For configs with no RNN layers, delegates directly to the original
components.trainer.build_model — that code is stable and well-tested.

For RNN-containing configs the layer-by-layer layout comes from
backend/core/rnn_plan.py (shared with compute_shapes and code generation):
  - LSTMWrapper / GRUWrapper make nn.LSTM / nn.GRU compatible with
    nn.Sequential by discarding the hidden-state tuple.  With
    return_sequences=True they emit every timestep (N, T, H); otherwise only
    the final state (N, H).  The plan sets return_sequences on every RNN
    except the last one.
  - Linear layers that see a sequence act per timestep; their batch norm is
    SeqBatchNorm1d.
  - _Unsqueeze promotes flat input (N, F) → (N, 1, F) before the first RNN on
    non-sequential data.
  - CNN layers cannot be mixed with RNN layers.

The helper classes below are also emitted verbatim (inspect.getsource) into
generated code by backend/core/code_gen.py — keep them self-contained
(depend only on torch / torch.nn).

train_model is re-exported unchanged from components.trainer.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from components.trainer import train_model, _ACT  # noqa: F401  (re-export train_model)

from backend.core.rnn_plan import RNN_TYPES, first_error, plan_rnn


# ── Sequential-compatible RNN helpers ─────────────────────────────────────────

class _Unsqueeze(nn.Module):
    """Promote (N, features) → (N, 1, features): a flat vector as a 1-step sequence."""

    def forward(self, x):
        return x.unsqueeze(1)


class SeqBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d for (N, T, C) sequences: normalises each channel over batch and time.

    nn.BatchNorm1d expects channels in dim 1, i.e. (N, C, T), so transpose around it.
    """

    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


def _last_step(out, hidden_size: int, bidirectional: bool):
    """Final state of each direction from a (N, T, dirs*H) RNN output.

    The forward direction finishes at t = T-1; the backward direction finishes
    at t = 0 (it reads the sequence in reverse), so out[:, -1] alone would give
    a backward state that has seen only one timestep.
    """
    if not bidirectional:
        return out[:, -1, :]
    return torch.cat([out[:, -1, :hidden_size], out[:, 0, hidden_size:]], dim=1)


class LSTMWrapper(nn.Module):
    """nn.LSTM wrapped for nn.Sequential.

    Input:  (N, T, input_size)
    Output: (N, T, out_size) if return_sequences else (N, out_size),
            where out_size = hidden_size * (2 if bidirectional else 1)
    """

    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False,
                 dropout=0.0, return_sequences=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.return_sequences = return_sequences
        self.out_size = hidden_size * (2 if bidirectional else 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        if self.return_sequences:
            return out
        return _last_step(out, self.hidden_size, self.bidirectional)


class GRUWrapper(nn.Module):
    """nn.GRU wrapped for nn.Sequential.

    Input:  (N, T, input_size)
    Output: (N, T, out_size) if return_sequences else (N, out_size),
            where out_size = hidden_size * (2 if bidirectional else 1)
    """

    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False,
                 dropout=0.0, return_sequences=False):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.return_sequences = return_sequences
        self.out_size = hidden_size * (2 if bidirectional else 1)

    def forward(self, x):
        out, _ = self.gru(x)
        if self.return_sequences:
            return out
        return _last_step(out, self.hidden_size, self.bidirectional)


# ── Extended build_model ──────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """Return an nn.Sequential from config, supporting all layer types.

    For configs without RNN layers, delegates to the original build_model in
    components.trainer (unchanged, stable).  For RNN-containing configs,
    follows the plan from backend/core/rnn_plan.py.
    """
    layers = config.get("hidden_layers", [])
    has_rnn = any(layer.get("type") in RNN_TYPES for layer in layers)

    if not has_rnn:
        from components.trainer import build_model as _orig_build
        return _orig_build(config)

    # ── RNN path ──────────────────────────────────────────────────────────────

    steps = plan_rnn(config)
    if any(s.out_shape is None for s in steps):
        raise ValueError(first_error(steps) or "One or more layers produce an invalid shape.")

    pytorch_layers: list[nn.Module] = []

    for step, layer in zip(steps, layers):
        if step.ltype in RNN_TYPES:
            if step.unsqueeze:
                pytorch_layers.append(_Unsqueeze())
            cls = LSTMWrapper if step.ltype == "LSTM" else GRUWrapper
            pytorch_layers.append(cls(
                step.in_features,
                layer.get("hidden_size") or 64,
                layer.get("num_layers") or 1,
                layer.get("bidirectional", False),
                layer.get("rnn_dropout", 0.0),
                return_sequences=step.return_sequences,
            ))

        else:  # Linear (per timestep when step.is_sequence_input)
            units = layer["units"]
            pytorch_layers.append(nn.Linear(step.in_features, units))
            if layer.get("batch_norm", False):
                bn = SeqBatchNorm1d if step.is_sequence_input else nn.BatchNorm1d
                pytorch_layers.append(bn(units))
            pytorch_layers.append(_ACT[layer["activation"]]())
            if layer.get("dropout", False) and layer.get("dropout_rate", 0.0) > 0:
                pytorch_layers.append(nn.Dropout(layer["dropout_rate"]))

    # Output layer — the last RNN always collapses time, so this input is flat
    prev = steps[-1].out_shape[0]
    if config["task"] == "binary":
        pytorch_layers.append(nn.Linear(prev, 1))
        pytorch_layers.append(nn.Sigmoid())
    elif config["task"] == "regression":
        pytorch_layers.append(nn.Linear(prev, 1))
    else:
        pytorch_layers.append(nn.Linear(prev, config["n_classes"]))

    return nn.Sequential(*pytorch_layers)
