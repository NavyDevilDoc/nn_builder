"""Feature-map dimension tracking: compute_shapes(config) -> list[tuple | None].

Returns a list of shape tuples, one per network row in top-to-bottom order:
  index 0          : input shape
  index 1 .. N-1   : output shape after each hidden layer
  index N          : output shape (n_classes,) or (1,) for binary

Shape representations:
  (C, H, W)   -- spatial feature map (after Conv2d / MaxPool2d)
  (units,)    -- flat vector (after Flatten, Linear, or tabular input)
  None        -- invalid: spatial dimension collapsed to zero, or
                 conv/pool applied to already-flat input

No streamlit imports. No side effects. Safe to call multiple times per rerun.
"""
from __future__ import annotations


def compute_shapes(config: dict) -> list[tuple | None]:
    """Return per-layer shape list for the given config."""
    input_shape = config.get("input_shape")  # (C, H, W) or None for tabular

    if input_shape is not None:
        current: tuple | None = tuple(input_shape)   # (C, H, W)
    else:
        current = (config["input_dim"],)             # flat 1-tuple

    shapes: list[tuple | None] = [current]

    for layer in config["hidden_layers"]:
        ltype = layer.get("type", "Linear")

        if current is None:
            # Propagate invalid marker once an error is detected
            shapes.append(None)
            continue

        if ltype == "Conv2d":
            if len(current) != 3:
                # Conv2d applied to a flat vector — invalid
                current = None
            else:
                C_out = layer["out_channels"]
                k     = layer["kernel_size"]
                H_out = current[1] - (k - 1)
                W_out = current[2] - (k - 1)
                if H_out <= 0 or W_out <= 0:
                    current = None
                else:
                    current = (C_out, H_out, W_out)

        elif ltype == "MaxPool2d":
            if len(current) != 3:
                current = None
            else:
                k     = layer["kernel_size"]
                H_out = current[1] // k
                W_out = current[2] // k
                if H_out <= 0 or W_out <= 0:
                    current = None
                else:
                    current = (current[0], H_out, W_out)

        elif ltype == "Flatten":
            if len(current) == 1:
                # Already flat — treat as no-op
                pass
            else:
                flat  = current[0] * current[1] * current[2]
                current = (flat,)

        else:  # Linear
            current = (layer["units"],)

        shapes.append(current)

    # Output layer shape
    if config["task"] in ("binary", "regression"):
        shapes.append((1,) if current is not None else None)
    else:
        shapes.append((config["n_classes"],) if current is not None else None)

    return shapes
