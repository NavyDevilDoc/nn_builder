"""FastAPI application entry point.

Dev:  uvicorn backend.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import datasets, validate, train, visualize, code

app = FastAPI(title="nn_builder API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router)
app.include_router(validate.router)
app.include_router(train.router)
app.include_router(visualize.router)
app.include_router(code.router)


@app.get("/")
async def health() -> dict:
    return {"status": "ok"}
