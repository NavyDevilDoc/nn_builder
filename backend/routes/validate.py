"""Architecture validation endpoint.

POST /api/validate
     Runs compute_shapes() on the submitted layer list and returns per-layer
     shape annotations plus a validity flag.
"""
from __future__ import annotations

from fastapi import APIRouter

from backend.core.inputs import resolve_input
from backend.core.rnn_plan import RNN_TYPES, first_error, plan_rnn
from backend.core.shapes import compute_shapes
from backend.models import ValidationRequest, ValidationResponse

router = APIRouter(prefix="/api", tags=["validate"])


@router.post("/validate", response_model=ValidationResponse)
async def validate(req: ValidationRequest) -> ValidationResponse:
    layers = [layer.model_dump(exclude_none=False) for layer in req.hidden_layers]
    layer_types = {l.get("type", "Linear") for l in layers}

    # Same input-layout decision as train.py (image vs flat, sequence vs
    # flattened window) so shape annotations match what training will build.
    eff = resolve_input(req.input_dim, req.input_shape, req.seq_len, layer_types)

    config = {
        "input_shape": eff.input_shape,
        "input_dim":   eff.input_dim,
        "seq_len":     eff.seq_len,
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

    error = None
    if has_invalid:
        # RNN configs know *why* a layer is invalid; others get the generic message
        specific = first_error(plan_rnn(config)) if layer_types & RNN_TYPES else None
        error = specific or "One or more layers produce an invalid shape."

    return ValidationResponse(valid=not has_invalid, error=error, shapes=shapes)
