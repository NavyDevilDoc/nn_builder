"""Dataset endpoints.

GET  /api/datasets
     Returns all dataset names, metadata, and suggested architectures.

GET  /api/datasets/{name}/sample
     Returns a base64 PNG thumbnail for image datasets (MNIST, Fashion-MNIST,
     CIFAR-10).  Returns 404 for tabular datasets.
"""
from __future__ import annotations

import base64
import io

from fastapi import APIRouter, HTTPException

import datasets.registry as registry
from backend.models import DatasetInfo, DatasetsResponse, LayerConfig, SampleResponse

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


def _info_to_model(name: str) -> DatasetInfo:
    info = registry.meta(name)
    return DatasetInfo(
        task=info.task,
        input_dim=info.input_dim,
        n_classes=info.n_classes,
        description=info.description,
        input_shape=list(info.input_shape) if info.input_shape else None,
        seq_len=info.seq_len,
    )


def _layer_dict_to_model(layer: dict) -> LayerConfig:
    return LayerConfig(**layer)


@router.get("", response_model=DatasetsResponse)
async def get_datasets() -> DatasetsResponse:
    names = registry.list()
    meta = {n: _info_to_model(n) for n in names}
    suggested = {
        n: [_layer_dict_to_model(l) for l in registry.suggested(n)]
        for n in names
    }
    return DatasetsResponse(names=names, meta=meta, suggested=suggested)


@router.get("/{name}/sample", response_model=SampleResponse)
async def get_sample(name: str) -> SampleResponse:
    try:
        info = registry.meta(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown dataset: {name!r}")

    if info.input_shape is None:
        raise HTTPException(
            status_code=404,
            detail="Sample thumbnails are only available for image datasets.",
        )

    return SampleResponse(image=_render_image_thumbnail(name, info.input_shape))


def _render_image_thumbnail(name: str, input_shape: tuple) -> str:
    """Load 16 cached samples from torchvision and return a 4×4 grid as base64 PNG."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    import datasets.builtin as _builtin

    data_dir = Path(_builtin.__file__).parent.parent / "data"
    C, H, W = input_shape

    if name == "mnist":
        from torchvision.datasets import MNIST
        ds = MNIST(root=data_dir, train=True, download=False)
        images = ds.data[:16].numpy()          # (16, H, W) uint8
        cmap = "gray"
    elif name == "fashion_mnist":
        from torchvision.datasets import FashionMNIST
        ds = FashionMNIST(root=data_dir, train=True, download=False)
        images = ds.data[:16].numpy()
        cmap = "gray"
    elif name == "cifar10":
        from torchvision.datasets import CIFAR10
        ds = CIFAR10(root=data_dir, train=True, download=False)
        images = ds.data[:16]                  # (16, H, W, C) uint8 HWC
        cmap = None
    else:
        raise HTTPException(status_code=404, detail="No thumbnail for this dataset.")

    fig, axes = plt.subplots(4, 4, figsize=(4, 4))
    fig.patch.set_facecolor("#18181b")
    for ax, img in zip(axes.flat, images):
        ax.imshow(img, cmap=cmap)
        ax.axis("off")
    fig.tight_layout(pad=0.2)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{encoded}"
