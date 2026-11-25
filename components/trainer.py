"""Model building and training.

Contracts (must stay in sync with components/code_gen.py):
  - Binary      : Sigmoid output, BCELoss, y cast to float32 shape (N, 1)
  - Multiclass  : raw logit output, CrossEntropyLoss, y cast to long shape (N,)
  - Momentum    : SGD with momentum=0.9
  - Mini-batch  : DataLoader over TensorDataset; epoch metrics on full tensors
  - Image input : flat (N, D) arrays reshaped to (N, C, H, W) via input_shape
  - on_epoch    : optional callback(epoch_idx, metrics_dict) called each epoch
  - stop_event  : optional threading.Event; training breaks at next epoch start
  - history keys: train_loss, val_loss, train_acc, val_acc (one value per epoch)
"""
from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

_ACT: dict[str, type] = {
    "ReLU":      nn.ReLU,
    "Tanh":      nn.Tanh,
    "Sigmoid":   nn.Sigmoid,
    "LeakyReLU": lambda: nn.LeakyReLU(0.01),
    "Linear":    nn.Identity,
}


def build_model(config: dict) -> nn.Module:
    """Return an nn.Sequential from config.

    Supports Linear, Conv2d, MaxPool2d, and Flatten layer types.
    Raises ValueError if hidden_layers is empty or if a CNN layer produces
    an invalid (zero or negative) spatial dimension.
    """
    if not config["hidden_layers"]:
        raise ValueError(
            "The network must have at least one hidden layer. "
            "Use '+ Add Layer' in the sidebar."
        )

    is_image = config.get("input_shape") is not None
    pytorch_layers: list[nn.Module] = []

    if is_image:
        from components.shapes import compute_shapes
        shapes = compute_shapes(config)
        # Validate: any None in hidden-layer outputs means an invalid architecture
        for i in range(1, len(config["hidden_layers"]) + 1):
            if shapes[i] is None:
                ltype = config["hidden_layers"][i - 1].get("type", "Linear")
                raise ValueError(
                    f"Layer {i} ({ltype}) produces an invalid output shape "
                    f"(spatial dimension collapsed to zero or Conv/Pool applied to flat input). "
                    f"Check kernel sizes and layer ordering."
                )
        prev = None  # tabular width tracking not used for image configs
    else:
        shapes = None
        prev   = config["input_dim"]

    for i, layer in enumerate(config["hidden_layers"]):
        ltype = layer.get("type", "Linear")

        if ltype == "Conv2d":
            in_c = shapes[i][0]
            pytorch_layers.append(nn.Conv2d(in_c, layer["out_channels"], layer["kernel_size"]))
            if layer.get("batch_norm", False):
                pytorch_layers.append(nn.BatchNorm2d(layer["out_channels"]))
            pytorch_layers.append(_ACT[layer["activation"]]())

        elif ltype == "MaxPool2d":
            pytorch_layers.append(nn.MaxPool2d(layer["kernel_size"]))

        elif ltype == "Flatten":
            pytorch_layers.append(nn.Flatten())

        else:  # Linear
            in_units = shapes[i][0] if is_image else prev
            pytorch_layers.append(nn.Linear(in_units, layer["units"]))
            if layer.get("batch_norm", False):
                pytorch_layers.append(nn.BatchNorm1d(layer["units"]))
            pytorch_layers.append(_ACT[layer["activation"]]())
            if layer.get("dropout", False) and layer.get("dropout_rate", 0.0) > 0:
                pytorch_layers.append(nn.Dropout(layer["dropout_rate"]))
            if not is_image:
                prev = layer["units"]

    # Output layer — input dim is last hidden layer's output width
    final_units = shapes[len(config["hidden_layers"])][0] if is_image else prev
    if config["task"] == "binary":
        pytorch_layers.append(nn.Linear(final_units, 1))
        pytorch_layers.append(nn.Sigmoid())
    else:
        pytorch_layers.append(nn.Linear(final_units, config["n_classes"]))

    return nn.Sequential(*pytorch_layers)


def train_model(
    model: nn.Module,
    config: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    on_epoch: Callable[[int, dict], None] | None = None,
    stop_event=None,
) -> tuple[nn.Module, dict]:
    """Train model; return (model, history).

    X arrays: float32 (N, D).  y arrays: int64 (N,).
    history values are lists of length ≤ config["epochs"] (may be shorter if
    training is stopped early by stop_event or early-stopping patience).
    on_epoch(epoch_idx, metrics) is called at the end of each epoch if provided.
    stop_event: any object with .is_set() — training breaks at the top of the
    next epoch when set (e.g. threading.Event).
    Model is moved back to CPU before returning.
    """
    torch.manual_seed(config["seed"])

    device = torch.device(config.get("device", "cpu"))
    model  = model.to(device)

    # Criterion
    if config["task"] == "binary":
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer — weight_decay wires L2 regularisation
    opt, lr = config["optimizer"], config["lr"]
    wd      = config.get("weight_decay", 0.0)
    if opt == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt == "Momentum":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

    # Tensors — dtype and shape must match code_gen.py
    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_v  = torch.tensor(X_val,   dtype=torch.float32).to(device)
    if config["task"] == "binary":
        y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
        y_v  = torch.tensor(y_val,   dtype=torch.float32).unsqueeze(1).to(device)
    else:
        y_tr = torch.tensor(y_train, dtype=torch.long).to(device)
        y_v  = torch.tensor(y_val,   dtype=torch.long).to(device)

    # Reshape flat arrays to (N, C, H, W) for CNN input
    if config.get("input_shape") is not None:
        C, H, W = config["input_shape"]
        X_tr = X_tr.view(-1, C, H, W)
        X_v  = X_v.view(-1, C, H, W)

    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
    }

    # DataLoader (mini-batch only); global seed above controls shuffle order
    bs     = config["batch_size"]
    loader = None
    if bs != "full":
        loader = DataLoader(
            TensorDataset(X_tr, y_tr), batch_size=bs, shuffle=True,
        )

    # Early stopping state
    es_enabled   = config.get("early_stopping", False)
    es_patience  = int(config.get("es_patience", 20))
    es_min_delta = float(config.get("es_min_delta", 0.0))
    es_best_val  = float("inf")
    es_wait      = 0

    for epoch in range(config["epochs"]):
        # Honour external stop request (e.g. user clicked Stop)
        if stop_event is not None and stop_event.is_set():
            break

        model.train()
        if loader is None:
            optimizer.zero_grad()
            out  = model(X_tr)
            loss = criterion(out, y_tr)
            loss.backward()
            optimizer.step()
        else:
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                out  = model(X_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                optimizer.step()

        # Epoch-level metrics — always computed on full tensors
        model.eval()
        with torch.no_grad():
            train_out  = model(X_tr)
            train_loss = criterion(train_out, y_tr).item()
            val_out    = model(X_v)
            val_loss   = criterion(val_out, y_v).item()

            if config["task"] == "binary":
                train_acc = ((train_out > 0.5).float() == y_tr).float().mean().item()
                val_acc   = ((val_out   > 0.5).float() == y_v ).float().mean().item()
            else:
                train_acc = (train_out.argmax(1) == y_tr).float().mean().item()
                val_acc   = (val_out.argmax(1)   == y_v ).float().mean().item()

        if math.isnan(train_loss) or math.isnan(val_loss):
            raise RuntimeError(
                "NaN loss detected. "
                "Try a lower learning rate or a different weight initialisation "
                "(PyTorch uses Kaiming uniform by default for nn.Linear)."
            )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if on_epoch is not None:
            on_epoch(epoch, {
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "train_acc":  train_acc,
                "val_acc":    val_acc,
            })

        # Early stopping — check after recording metrics so the final epoch is included
        if es_enabled:
            if val_loss < es_best_val - es_min_delta:
                es_best_val = val_loss
                es_wait     = 0
            else:
                es_wait += 1
                if es_wait >= es_patience:
                    break

    return model.to("cpu"), history
