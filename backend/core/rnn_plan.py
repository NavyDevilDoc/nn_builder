"""Layer-by-layer plan for networks that contain LSTM / GRU layers.

One function, three consumers — compute_shapes(), build_model(), and the RNN
code generator all read this plan, so shape annotations, the trained model,
and the generated script cannot disagree.

Sequence rules
--------------
  * Input is (seq_len, F) for sequential datasets, (F,) otherwise.
  * The sequence dimension is kept until the LAST recurrent layer:
      - an RNN returns its full output sequence (T, H) iff another RNN
        appears later in the stack (inferred "return_sequences");
      - the last RNN returns only its final timestep (H,).
  * A Linear layer that sees a sequence (T, F) is applied per timestep
    (nn.Linear acts on the last dim): (T, F) -> (T, units).  Its batch norm
    becomes SeqBatchNorm1d, which normalises channels over batch and time.
  * An RNN that sees a flat vector (F,) gets an _Unsqueeze first: (F,) -> (1, F).
    This only happens on non-sequential data, before the first RNN.
  * Conv2d / MaxPool2d / Flatten cannot be mixed with RNN layers.

Shapes exclude the batch dimension.  Bidirectional RNNs output 2 * hidden_size.
"""
from __future__ import annotations

from dataclasses import dataclass

RNN_TYPES = frozenset({"LSTM", "GRU"})
CNN_TYPES = frozenset({"Conv2d", "MaxPool2d", "Flatten"})


@dataclass(frozen=True)
class LayerStep:
    ltype: str
    in_shape: tuple[int, ...] | None
    out_shape: tuple[int, ...] | None
    unsqueeze: bool = False          # insert _Unsqueeze before this RNN
    return_sequences: bool = False   # RNN emits every timestep, not just the last
    error: str | None = None         # first-cause message when out_shape is None

    @property
    def in_features(self) -> int:
        assert self.in_shape is not None
        return self.in_shape[-1]

    @property
    def is_sequence_input(self) -> bool:
        return self.in_shape is not None and len(self.in_shape) == 2


def rnn_out_size(layer: dict) -> int:
    hidden = layer.get("hidden_size") or 64
    return hidden * (2 if layer.get("bidirectional", False) else 1)


def plan_rnn(config: dict) -> list[LayerStep]:
    """Return one LayerStep per hidden layer.  Invalid layers have out_shape=None."""
    layers  = config.get("hidden_layers", [])
    rnn_idx = [i for i, l in enumerate(layers) if l.get("type") in RNN_TYPES]
    last_rnn = rnn_idx[-1] if rnn_idx else -1

    if config.get("input_shape") is not None:
        msg = "LSTM/GRU layers are not supported on image input; use a tabular or sequential dataset."
        return [LayerStep(l.get("type", "Linear"), None, None, error=msg) for l in layers]

    seq_len = config.get("seq_len")
    current: tuple[int, ...] | None = (
        (seq_len, config["input_dim"]) if seq_len is not None else (config["input_dim"],)
    )

    steps: list[LayerStep] = []
    for i, layer in enumerate(layers):
        ltype = layer.get("type", "Linear")

        if current is None:                      # propagate an earlier failure
            steps.append(LayerStep(ltype, None, None))
            continue

        if ltype in RNN_TYPES:
            unsqueeze = len(current) == 1
            timesteps = 1 if unsqueeze else current[0]
            out_size  = rnn_out_size(layer)
            ret_seq   = i < last_rnn
            nxt = (timesteps, out_size) if ret_seq else (out_size,)
            steps.append(LayerStep(ltype, current, nxt, unsqueeze=unsqueeze, return_sequences=ret_seq))

        elif ltype in CNN_TYPES:
            steps.append(LayerStep(
                ltype, current, None,
                error=f"Layer {i + 1} ({ltype}): Conv2d, MaxPool2d, and Flatten cannot be mixed with LSTM/GRU.",
            ))
            nxt = None

        else:  # Linear — per timestep when the input is still a sequence
            nxt = (*current[:-1], layer["units"])
            steps.append(LayerStep(ltype, current, nxt))

        current = nxt

    return steps


def first_error(steps: list[LayerStep]) -> str | None:
    return next((s.error for s in steps if s.error), None)
