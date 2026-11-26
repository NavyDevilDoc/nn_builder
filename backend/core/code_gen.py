"""Code generation with LSTM / GRU support.

For configs without RNN layers, delegates to components.code_gen.generate_code
unchanged.  For RNN configs, builds the nn.Sequential body from the same plan
build_model uses (backend/core/rnn_plan.py) and emits the helper classes via
inspect.getsource — the generated script contains the exact classes that
trained, so the two cannot drift apart.
"""
from __future__ import annotations

import inspect

from components.code_gen import _ACT, generate_code as _generate_code
from backend.core import trainer as _trainer
from backend.core.rnn_plan import RNN_TYPES, first_error, plan_rnn

_INDENT = " " * 12


def _rnn_body(config: dict) -> tuple[str, set[str]]:
    """nn.Sequential lines + the helper names they reference."""
    steps  = plan_rnn(config)
    if any(s.out_shape is None for s in steps):
        raise ValueError(first_error(steps) or "One or more layers produce an invalid shape.")

    lines: list[str] = []
    needed: set[str] = set()

    for step, layer in zip(steps, config["hidden_layers"]):
        shape_note = f"  # {tuple(step.in_shape)} -> {tuple(step.out_shape)}"

        if step.ltype in RNN_TYPES:
            if step.unsqueeze:
                lines.append(f"{_INDENT}_Unsqueeze(),  # flat (F,) -> 1-step sequence (1, F)")
                needed.add("_Unsqueeze")
            cls = "LSTMWrapper" if step.ltype == "LSTM" else "GRUWrapper"
            needed.add(cls)
            needed.add("_last_step")
            lines.append(
                f"{_INDENT}{cls}({step.in_features}, {layer.get('hidden_size') or 64}, "
                f"num_layers={layer.get('num_layers') or 1}, "
                f"bidirectional={layer.get('bidirectional', False)}, "
                f"dropout={layer.get('rnn_dropout', 0.0)}, "
                f"return_sequences={step.return_sequences}),{shape_note}"
            )

        else:  # Linear — per timestep when the input is a sequence
            units = layer["units"]
            per_t = "  (per timestep)" if step.is_sequence_input else ""
            lines.append(f"{_INDENT}nn.Linear({step.in_features}, {units}),{shape_note}{per_t}")
            if layer.get("batch_norm"):
                if step.is_sequence_input:
                    lines.append(f"{_INDENT}SeqBatchNorm1d({units}),")
                    needed.add("SeqBatchNorm1d")
                else:
                    lines.append(f"{_INDENT}nn.BatchNorm1d({units}),")
            lines.append(f"{_INDENT}{_ACT[layer['activation']]},")
            if layer.get("dropout") and layer.get("dropout_rate", 0.0) > 0:
                lines.append(f"{_INDENT}nn.Dropout({layer['dropout_rate']}),")

    prev = steps[-1].out_shape[0]
    if config["task"] == "binary":
        lines += [f"{_INDENT}nn.Linear({prev}, 1),", f"{_INDENT}nn.Sigmoid(),"]
    elif config["task"] == "regression":
        lines += [f"{_INDENT}nn.Linear({prev}, 1),", f"{_INDENT}# Regression: unbounded output, no activation"]
    else:
        lines += [
            f"{_INDENT}nn.Linear({prev}, {config['n_classes']}),",
            f"{_INDENT}# CrossEntropyLoss applies log-softmax internally - no activation here",
        ]
    return "\n".join(lines), needed


# Emission order matters: _last_step must precede the wrappers that call it.
_HELPER_ORDER = ["_Unsqueeze", "SeqBatchNorm1d", "_last_step", "LSTMWrapper", "GRUWrapper"]


def generate_code(config: dict) -> str:
    """Return a complete, runnable PyTorch script for any supported config."""
    if not any(l.get("type") in RNN_TYPES for l in config.get("hidden_layers", [])):
        return _generate_code(config)

    body, needed = _rnn_body(config)
    preamble = "\n\n\n".join(
        inspect.getsource(getattr(_trainer, name)).rstrip()
        for name in _HELPER_ORDER if name in needed
    )
    return _generate_code(config, body=body, preamble=preamble)
