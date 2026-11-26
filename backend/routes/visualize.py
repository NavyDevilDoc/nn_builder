"""Visualization endpoints.

POST /api/visualize/confusion_matrix
POST /api/visualize/decision_boundary
POST /api/visualize/predictions        (regression only)

All look up the trained model from session_store and return a base64 PNG.
"""
from __future__ import annotations

import base64
import io

from fastapi import APIRouter, HTTPException
from matplotlib.figure import Figure

from backend.models import VisualizeRequest, VisualizeResponse
from backend.session_store import sessions

router = APIRouter(prefix="/api/visualize", tags=["visualize"])


def _fig_to_base64(fig: Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def _get_session(session_id: str) -> dict:
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Train a model first.")
    sess = sessions[session_id]
    if "model" not in sess:
        raise HTTPException(status_code=400, detail="Training has not completed for this session.")
    return sess


@router.post("/confusion_matrix", response_model=VisualizeResponse)
async def confusion_matrix(req: VisualizeRequest) -> VisualizeResponse:
    sess = _get_session(req.session_id)
    cfg  = sess["config"]

    if cfg["task"] == "regression":
        raise HTTPException(
            status_code=400,
            detail="Confusion matrix is only available for classification datasets.",
        )

    from components.visualizer import plot_confusion_matrix
    fig = plot_confusion_matrix(
        sess["model"],
        sess["X_test"],
        sess["y_test"],
        cfg,
        input_shape=tuple(cfg["input_shape"]) if cfg.get("input_shape") else None,
    )
    return VisualizeResponse(image=_fig_to_base64(fig))


@router.post("/decision_boundary", response_model=VisualizeResponse)
async def decision_boundary(req: VisualizeRequest) -> VisualizeResponse:
    sess = _get_session(req.session_id)
    cfg  = sess["config"]

    if cfg.get("input_dim") != 2:
        raise HTTPException(
            status_code=400,
            detail="Decision boundary is only available for 2D datasets (xor, moons, spirals).",
        )

    from components.visualizer import plot_decision_boundary
    fig = plot_decision_boundary(
        sess["model"],
        sess["X_test"],
        sess["y_test"],
        cfg,
    )
    return VisualizeResponse(image=_fig_to_base64(fig))


@router.post("/predictions", response_model=VisualizeResponse)
async def predictions(req: VisualizeRequest) -> VisualizeResponse:
    sess = _get_session(req.session_id)
    cfg  = sess["config"]

    if cfg["task"] != "regression":
        raise HTTPException(
            status_code=400,
            detail="Prediction plot is only available for regression datasets.",
        )

    from components.visualizer import plot_predictions
    fig = plot_predictions(sess["model"], sess["X_test"], sess["y_test"])
    return VisualizeResponse(image=_fig_to_base64(fig))
