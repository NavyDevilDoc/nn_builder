"""Training endpoints.

POST   /api/train           Start training; streams SSE epoch events.
DELETE /api/train/{id}      Cancel training (sets stop_event).

The SSE wire format is:
  data: {"type": "epoch", "epoch": N, "total": N, "train_loss": f, ...}\n\n
  data: {"type": "done",  "session_id": "...", "epochs_trained": N}\n\n
  data: {"type": "error", "message": "..."}\n\n

The React client consumes this via fetch + ReadableStream (not EventSource,
which only supports GET).
"""
from __future__ import annotations

import asyncio
import json
import threading
import uuid

from fastapi import APIRouter
from fastapi.responses import Response, StreamingResponse

import datasets.registry as registry
from backend.core.inputs import apply_to_config, reshape_for_config
from backend.core.trainer import build_model
from backend.models import BuilderConfig
from backend.session_store import sessions
from components.trainer import train_model

router = APIRouter(prefix="/api/train", tags=["train"])


@router.post("")
async def start_training(config: BuilderConfig) -> StreamingResponse:
    session_id = str(uuid.uuid4())
    stop_event = threading.Event()
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_epoch(epoch: int, metrics: dict) -> None:
        loop.call_soon_threadsafe(
            queue.put_nowait,
            {"type": "epoch", "epoch": epoch, "total": config.epochs, **metrics},
        )

    def _thread() -> None:
        try:
            cfg = config.model_dump()

            # Merge dataset metadata and decide the input layout (image vs flat,
            # sequence vs flattened window) — see backend/core/inputs.py.
            apply_to_config(cfg, registry.meta(cfg["dataset"]))

            X_train, y_train, X_test, y_test = registry.load(
                cfg["dataset"], seed=cfg["seed"], seq_len=cfg["data_seq_len"]
            )
            X_train = reshape_for_config(X_train, cfg)
            X_test  = reshape_for_config(X_test, cfg)

            model = build_model(cfg)
            trained_model, history = train_model(
                model, cfg,
                X_train, y_train,
                X_test, y_test,
                on_epoch=on_epoch,
                stop_event=stop_event,
            )

            sessions[session_id].update(
                model=trained_model,
                history=history,
                X_test=X_test,
                y_test=y_test,
                config=cfg,
            )

            loop.call_soon_threadsafe(
                queue.put_nowait,
                {
                    "type": "done",
                    "session_id": session_id,
                    "epochs_trained": len(history["train_loss"]),
                },
            )

        except Exception as exc:
            loop.call_soon_threadsafe(
                queue.put_nowait, {"type": "error", "message": str(exc)}
            )

    sessions[session_id] = {"stop_event": stop_event}
    threading.Thread(target=_thread, daemon=True).start()

    async def event_stream():
        while True:
            event = await queue.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event["type"] in ("done", "error"):
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete("/{session_id}")
async def cancel_training(session_id: str) -> Response:
    if session_id in sessions:
        sessions[session_id]["stop_event"].set()
    return Response(status_code=204)
