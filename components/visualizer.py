"""Result visualizations.  No streamlit imports — all functions return a Figure.

plot_loss_curve(history)                          -> Figure  (always)
plot_decision_boundary(model, X, y, config)       -> Figure  (2D datasets)
plot_confusion_matrix(model, X_test, y_test, config) -> Figure  (all datasets)
"""
from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import confusion_matrix


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _tab10_colors(n: int) -> list:
    """Return n RGBA colours from the tab10 palette."""
    cmap = plt.get_cmap("tab10")
    return [cmap(i * 0.1) for i in range(min(n, 10))]


def _predict(
    model: torch.nn.Module,
    X: np.ndarray,
    task: str,
    input_shape: tuple | None = None,
) -> np.ndarray:
    """Return integer class predictions (N,) for X.

    input_shape: (C, H, W) for CNN models — reshapes flat X before inference.
    """
    model.eval()
    with torch.no_grad():
        t = torch.tensor(X, dtype=torch.float32)
        if input_shape is not None:
            C, H, W = input_shape
            t = t.view(-1, C, H, W)
        out = model(t)
        if task == "binary":
            return (out > 0.5).long().squeeze(1).numpy()
        return out.argmax(1).numpy()


# ── Loss curve ─────────────────────────────────────────────────────────────────

def plot_loss_curve(history: dict) -> Figure:
    """Two-panel figure: loss (left) and accuracy (right) vs epoch."""
    epochs = range(1, len(history["train_loss"]) + 1)
    blue, red = "#2E86C1", "#E74C3C"

    fig = Figure(figsize=(7.0, 3.2))
    ax_l = fig.add_subplot(121)
    ax_a = fig.add_subplot(122)

    # Loss
    ax_l.plot(epochs, history["train_loss"], color=blue, lw=1.6, label="Train")
    ax_l.plot(epochs, history["val_loss"],   color=red,  lw=1.6, ls="--", label="Val")
    ax_l.set_xlabel("Epoch")
    ax_l.set_ylabel("Loss")
    ax_l.set_title("Loss")
    ax_l.legend(fontsize=8)
    ax_l.grid(True, ls=":", lw=0.5, alpha=0.5)

    # Accuracy
    ax_a.plot(epochs, history["train_acc"], color=blue, lw=1.6, label="Train")
    ax_a.plot(epochs, history["val_acc"],   color=red,  lw=1.6, ls="--", label="Val")
    ax_a.set_xlabel("Epoch")
    ax_a.set_ylabel("Accuracy")
    ax_a.set_title("Accuracy")
    ax_a.set_ylim(-0.05, 1.05)
    ax_a.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_a.legend(fontsize=8)
    ax_a.grid(True, ls=":", lw=0.5, alpha=0.5)

    fig.tight_layout(pad=0.6)
    return fig


# ── Decision boundary ──────────────────────────────────────────────────────────

def plot_decision_boundary(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    config: dict,
) -> Figure:
    """Filled decision-boundary plot for 2-D datasets.

    X: (N, 2) float32   y: (N,) int64
    """
    n_classes = config["n_classes"]
    colors    = _tab10_colors(n_classes)
    boundary_cmap = ListedColormap(colors)

    # Build a dense grid over the data extent
    margin = 0.4
    x0_min, x0_max = float(X[:, 0].min()) - margin, float(X[:, 0].max()) + margin
    x1_min, x1_max = float(X[:, 1].min()) - margin, float(X[:, 1].max()) + margin
    h = max(x0_max - x0_min, x1_max - x1_min) / 200.0

    xx, yy = np.meshgrid(
        np.arange(x0_min, x0_max, h),
        np.arange(x1_min, x1_max, h),
    )
    grid_X  = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    Z = _predict(model, grid_X, config["task"]).reshape(xx.shape)

    fig = Figure(figsize=(5.0, 4.5))
    ax  = fig.add_subplot(111)

    ax.pcolormesh(xx, yy, Z, cmap=boundary_cmap, alpha=0.25,
                  shading="auto", vmin=0, vmax=n_classes - 1)

    for cls in range(n_classes):
        mask = y == cls
        ax.scatter(
            X[mask, 0], X[mask, 1],
            color=colors[cls],
            edgecolors="white", linewidths=0.5,
            s=28, alpha=0.9, zorder=3,
            label=f"Class {cls}",
        )

    ax.set_xlim(x0_min, x0_max)
    ax.set_ylim(x1_min, x1_max)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Decision Boundary")
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout(pad=0.5)
    return fig


# ── Confusion matrix ───────────────────────────────────────────────────────────

def plot_confusion_matrix(
    model: torch.nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict,
    input_shape: tuple | None = None,
) -> Figure:
    """Annotated confusion matrix on the held-out test set.

    X_test: (N, D) float32   y_test: (N,) int64
    input_shape: (C, H, W) for CNN models.
    """
    y_pred = _predict(model, X_test, config["task"], input_shape=input_shape)
    cm     = confusion_matrix(y_test, y_pred)
    n      = cm.shape[0]

    # Scale figure height with number of classes
    fig_size = max(3.5, n * 0.55 + 1.5)
    fig = Figure(figsize=(fig_size + 0.5, fig_size))
    ax  = fig.add_subplot(111)

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ticks = list(range(n))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(i) for i in ticks], fontsize=8)
    ax.set_yticklabels([str(i) for i in ticks], fontsize=8)

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=8,
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix (test set)")

    fig.tight_layout(pad=0.5)
    return fig
