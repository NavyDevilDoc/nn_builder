"""Effective-input resolution shared by the train, validate, and code routes.

The same dataset can be fed to a network in different layouts depending on
which layer types the network contains:

  Image dataset   + CNN layers     -> (N, C, H, W)      input_shape kept
  Image dataset   + no CNN layers  -> (N, C*H*W)        input_shape cleared (flat MLP)
  Sequential data + RNN layers     -> (N, seq_len, F)   seq_len kept; first RNN
                                                        consumes it directly
  Sequential data + no RNN layers  -> (N, seq_len * F)  windows flattened (MLP baseline)
  Tabular data                     -> (N, D)

Keeping this in one place guarantees that shape validation, training, and
code generation all agree on what the first layer actually receives.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CNN_TYPES = frozenset({"Conv2d", "MaxPool2d", "Flatten"})
RNN_TYPES = frozenset({"LSTM", "GRU"})


@dataclass(frozen=True)
class EffectiveInput:
    input_dim: int                # width of a flat input, or features per timestep
    input_shape: list[int] | None  # [C, H, W] when CNN layers consume images
    seq_len: int | None           # window length when RNN layers consume sequences
    flatten_sequence: bool        # True => reshape X (N, T, F) -> (N, T*F)


def resolve_input(
    input_dim: int,
    input_shape: list[int] | tuple | None,
    seq_len: int | None,
    layer_types: set[str],
) -> EffectiveInput:
    """Decide the input layout from dataset metadata and the network's layer types.

    seq_len is the *effective* window length (user override or dataset default),
    or None for non-sequential datasets.
    """
    shape = list(input_shape) if (input_shape and layer_types & CNN_TYPES) else None

    if seq_len is None:
        return EffectiveInput(input_dim, shape, None, False)
    if layer_types & RNN_TYPES:
        return EffectiveInput(input_dim, None, seq_len, False)
    return EffectiveInput(input_dim * seq_len, None, None, True)


def apply_to_config(cfg: dict, info) -> dict:
    """Merge registry DatasetInfo + effective input layout into a config dict (in place).

    cfg must carry "dataset", "hidden_layers", and optionally "seq_len"
    (a user override; falls back to the dataset default).
    """
    layer_types = {l.get("type", "Linear") for l in cfg["hidden_layers"]}
    raw_seq_len = (cfg.get("seq_len") or info.seq_len) if info.seq_len is not None else None
    eff = resolve_input(info.input_dim, info.input_shape, raw_seq_len, layer_types)

    cfg["task"]             = info.task
    cfg["n_classes"]        = info.n_classes
    cfg["input_dim"]        = eff.input_dim
    cfg["input_shape"]      = eff.input_shape
    cfg["seq_len"]          = eff.seq_len
    cfg["data_seq_len"]     = raw_seq_len       # window length to load, regardless of layout
    cfg["flatten_sequence"] = eff.flatten_sequence
    return cfg


def reshape_for_config(X: np.ndarray, cfg: dict) -> np.ndarray:
    """Apply the flatten_sequence decision to a loaded X array."""
    if cfg.get("flatten_sequence"):
        return X.reshape(len(X), -1)
    return X
