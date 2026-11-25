"""Architecture validation endpoint.

POST /api/validate
     Runs compute_shapes() on the submitted layer list and returns per-layer
     shape annotations plus a validity flag.
"""
from __future__ import annotations

from fastapi import APIRouter

from backend.core.shapes import compute_shapes
from backend.models import ValidationRequest, ValidationResponse

router = APIRouter(prefix="/api", tags=["validate"])


_CNN_TYPES = frozenset({"Conv2d", "MaxPool2d", "Flatten"})


@router.post("/validate", response_model=ValidationResponse)
async def validate(req: ValidationRequest) -> ValidationResponse:
    layers = [layer.model_dump(exclude_none=False) for layer in req.hidden_layers]
    layer_types = {l.get("type", "Linear") for l in layers}

    # Mirror train.py: a pure-Linear/RNN network on an image dataset is treated
    # as flat tabular — input_shape only matters when CNN layers are present.
    effective_input_shape = (
        req.input_shape if (req.input_shape and layer_types & _CNN_TYPES) else None
    )

    config = {
        "input_shape": effective_input_shape,
        "input_dim":   req.input_dim,
        "task":        req.task,
        "n_classes":   req.n_classes,
        "hidden_layers": layers,
    }

    try:
        raw_shapes = compute_shapes(config)
    except Exception as exc:
        return ValidationResponse(
            valid=False,
            error=str(exc),
            shapes=[None] * (len(req.hidden_layers) + 2),
        )

    # Convert tuples → lists (JSON-serialisable); keep None as-is
    shapes = [list(s) if s is not None else None for s in raw_shapes]
    has_invalid = any(s is None for s in shapes)

    return ValidationResponse(
        valid=not has_invalid,
        error=None if not has_invalid else "One or more layers produce an invalid shape.",
        shapes=shapes,
    )
