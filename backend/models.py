"""Pydantic request / response schemas.

Keep in sync with frontend/src/types/config.ts.
Layer card `id` fields are frontend-only; they are never sent to the backend.
"""
from __future__ import annotations

from typing import Literal, Union
from pydantic import BaseModel, Field


# ── Layer types ───────────────────────────────────────────────────────────────

class LayerConfig(BaseModel):
    type: Literal["Linear", "Conv2d", "MaxPool2d", "Flatten", "LSTM", "GRU"] = "Linear"
    # Linear
    units: int | None = None
    activation: Literal["ReLU", "Tanh", "Sigmoid", "LeakyReLU", "Linear"] | None = None
    batch_norm: bool = False
    dropout: bool = False
    dropout_rate: float = 0.0
    # Conv2d
    out_channels: int | None = None
    kernel_size: int | None = None          # shared with MaxPool2d
    # LSTM / GRU
    hidden_size: int | None = None
    num_layers: int | None = None
    bidirectional: bool = False
    rnn_dropout: float = 0.0


# ── Training config ───────────────────────────────────────────────────────────

class BuilderConfig(BaseModel):
    dataset: str
    hidden_layers: list[LayerConfig] = Field(default_factory=list)
    optimizer: Literal["Adam", "SGD", "Momentum"] = "Adam"
    lr: float = 0.01
    epochs: int = 100
    batch_size: Union[int, Literal["full"]] = 32
    weight_decay: float = 0.0
    seed: int = 42
    early_stopping: bool = False
    es_patience: int = 20
    es_min_delta: float = 0.0
    device: Literal["cpu", "cuda"] = "cpu"


# ── Dataset endpoints ─────────────────────────────────────────────────────────

class DatasetInfo(BaseModel):
    task: Literal["binary", "multiclass"]
    input_dim: int
    n_classes: int
    description: str
    input_shape: list[int] | None = None    # [C, H, W] for image datasets


class DatasetsResponse(BaseModel):
    names: list[str]
    meta: dict[str, DatasetInfo]
    suggested: dict[str, list[LayerConfig]]


class SampleResponse(BaseModel):
    image: str                              # "data:image/png;base64,..."


# ── Validation endpoint ───────────────────────────────────────────────────────

class ValidationRequest(BaseModel):
    input_shape: list[int] | None = None    # [C, H, W] or None for tabular
    input_dim: int
    task: str
    n_classes: int
    hidden_layers: list[LayerConfig] = Field(default_factory=list)


class ValidationResponse(BaseModel):
    valid: bool
    error: str | None = None
    shapes: list[list[int] | None]


# ── Visualization endpoints ───────────────────────────────────────────────────

class VisualizeRequest(BaseModel):
    session_id: str
    config: BuilderConfig


class VisualizeResponse(BaseModel):
    image: str                              # "data:image/png;base64,..."


# ── Code endpoint ─────────────────────────────────────────────────────────────

class CodeResponse(BaseModel):
    code: str
