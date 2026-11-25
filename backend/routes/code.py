"""Code generation endpoint.

GET /api/code?config=<json-encoded BuilderConfig>

Returns a complete, runnable PyTorch script string matching the current config.
Dataset metadata (task, input_dim, etc.) is merged in server-side so the client
only needs to send the BuilderConfig fields.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Query

import datasets.registry as registry
from backend.models import CodeResponse
from components.code_gen import generate_code

router = APIRouter(prefix="/api", tags=["code"])


@router.get("/code", response_model=CodeResponse)
async def get_code(config: str = Query(..., description="JSON-encoded BuilderConfig")) -> CodeResponse:
    try:
        cfg = json.loads(config)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in config: {exc}")

    try:
        info = registry.meta(cfg["dataset"])
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    cfg["task"]        = info.task
    cfg["n_classes"]   = info.n_classes
    cfg["input_dim"]   = info.input_dim
    cfg["input_shape"] = list(info.input_shape) if info.input_shape else None

    # hidden_layers may arrive as list of dicts (from JSON) — that's what generate_code expects
    try:
        return CodeResponse(code=generate_code(cfg))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
