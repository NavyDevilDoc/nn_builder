"""Dataset registry.

Public API
----------
list()          -> list of available dataset names
meta(name)      -> DatasetInfo (task, input_dim, n_classes, description)
load(name, seed=42) -> (X_train, y_train, X_test, y_test) as numpy arrays

Split convention
----------------
XOR   : all 4 points returned as both train and test (pedagogical only).
MNIST : pre-split by torchvision (train=10k, test=10k).
Sequential (sine, multi_sine, airline):
        windowed into (N, seq_len, 1) one-step-ahead pairs; chronological
        80/20 split on target index (no shuffling - avoids look-ahead leakage);
        StandardScaler-style normalisation fit on the training span only.
        airline is log-transformed first (multiplicative seasonality -> additive).
Others: stratified 80/20 split; sklearn datasets also get StandardScaler
        fit on train, applied to both.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from datasets.synthetic import xor, moons, spirals
from datasets.builtin import iris, breast_cancer, wine, mnist, fashion_mnist, cifar10
from datasets.sequential import sine, multi_sine, airline, make_windows


@dataclass(frozen=True)
class DatasetInfo:
    task: str           # "binary" | "multiclass" | "regression"
    input_dim: int      # features per sample (per timestep for sequential datasets)
    n_classes: int      # 0 for regression
    description: str
    input_shape: tuple | None = None  # (C, H, W) for image datasets; None for tabular
    seq_len: int | None = None        # default window length; non-None => sequential


_META: dict[str, DatasetInfo] = {
    "xor": DatasetInfo(
        task="binary", input_dim=2, n_classes=2,
        description="4-point XOR problem",
    ),
    "moons": DatasetInfo(
        task="binary", input_dim=2, n_classes=2,
        description="Noisy half-circles (n=400)",
    ),
    "spirals": DatasetInfo(
        task="binary", input_dim=2, n_classes=2,
        description="Interleaved Archimedean spirals (n=400)",
    ),
    "iris": DatasetInfo(
        task="multiclass", input_dim=4, n_classes=3,
        description="Iris flower dataset (n=150)",
    ),
    "breast_cancer": DatasetInfo(
        task="binary", input_dim=30, n_classes=2,
        description="Breast Cancer Wisconsin (n=569)",
    ),
    "wine": DatasetInfo(
        task="multiclass", input_dim=13, n_classes=3,
        description="Wine recognition dataset (n=178)",
    ),
    "mnist": DatasetInfo(
        task="multiclass", input_dim=784, n_classes=10,
        description="MNIST flattened, pixel/255 (train=10k, test=10k)",
        input_shape=(1, 28, 28),
    ),
    "fashion_mnist": DatasetInfo(
        task="multiclass", input_dim=784, n_classes=10,
        description="Fashion-MNIST 28x28 grayscale (train=10k, test=10k)",
        input_shape=(1, 28, 28),
    ),
    "cifar10": DatasetInfo(
        task="multiclass", input_dim=3072, n_classes=10,
        description="CIFAR-10 32x32 RGB (train=10k, test=10k) - CPU training is slow",
        input_shape=(3, 32, 32),
    ),
    "sine": DatasetInfo(
        task="regression", input_dim=1, n_classes=0,
        description="Noisy sine wave, one-step-ahead forecast (T=1000, period ~63)",
        seq_len=20,
    ),
    "multi_sine": DatasetInfo(
        task="regression", input_dim=1, n_classes=0,
        description="Sum of three incommensurate sines + noise (T=1500)",
        seq_len=40,
    ),
    "airline": DatasetInfo(
        task="regression", input_dim=1, n_classes=0,
        description="Monthly airline passengers 1949-1960, log-scaled (T=144)",
        seq_len=12,
    ),
}

# name -> series loader taking the run seed (only synthetic series use it for noise)
_SERIES = {
    "sine":       lambda seed: sine(seed=seed),
    "multi_sine": lambda seed: multi_sine(seed=seed),
    "airline":    lambda seed: airline(),
}
_LOG_TRANSFORM = {"airline"}
_MIN_WINDOWS = 20   # reject seq_len values that leave too few (X, y) pairs

# Datasets whose features span different scales â€” apply StandardScaler
_NEEDS_SCALING = {"iris", "breast_cancer", "wine"}


def list() -> list[str]:  # noqa: A001  (shadows built-in intentionally for API clarity)
    """Return sorted list of available dataset names."""
    return sorted(_META.keys())


def meta(name: str) -> DatasetInfo:
    """Return DatasetInfo for `name`."""
    if name not in _META:
        raise KeyError(f"Unknown dataset {name!r}. Available: {list()}")
    return _META[name]


# â”€â”€ Suggested architectures â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_SUGGESTED: dict[str, list[dict]] = {
    "xor": [
        {"units": 4,   "activation": "ReLU"},
    ],
    "moons": [
        {"units": 16,  "activation": "ReLU"},
        {"units": 8,   "activation": "ReLU"},
    ],
    "spirals": [
        {"units": 32,  "activation": "ReLU"},
        {"units": 16,  "activation": "ReLU"},
        {"units": 8,   "activation": "ReLU"},
    ],
    "iris": [
        {"units": 16,  "activation": "ReLU"},
        {"units": 8,   "activation": "ReLU"},
    ],
    "breast_cancer": [
        {"units": 64,  "activation": "ReLU"},
        {"units": 32,  "activation": "ReLU"},
    ],
    "wine": [
        {"units": 32,  "activation": "ReLU"},
        {"units": 16,  "activation": "ReLU"},
    ],
    "mnist": [
        {"units": 256, "activation": "ReLU"},
        {"units": 128, "activation": "ReLU"},
        {"units": 64,  "activation": "ReLU"},
    ],
    "fashion_mnist": [
        {"type": "Conv2d",    "out_channels": 16, "kernel_size": 3, "activation": "ReLU", "batch_norm": False},
        {"type": "MaxPool2d", "kernel_size": 2},
        {"type": "Conv2d",    "out_channels": 32, "kernel_size": 3, "activation": "ReLU", "batch_norm": False},
        {"type": "MaxPool2d", "kernel_size": 2},
        {"type": "Flatten"},
        {"type": "Linear",    "units": 128, "activation": "ReLU", "batch_norm": False, "dropout": False, "dropout_rate": 0.0},
    ],
    "cifar10": [
        {"type": "Conv2d",    "out_channels": 32, "kernel_size": 3, "activation": "ReLU", "batch_norm": False},
        {"type": "MaxPool2d", "kernel_size": 2},
        {"type": "Conv2d",    "out_channels": 64, "kernel_size": 3, "activation": "ReLU", "batch_norm": False},
        {"type": "MaxPool2d", "kernel_size": 2},
        {"type": "Flatten"},
        {"type": "Linear",    "units": 256, "activation": "ReLU", "batch_norm": False, "dropout": False, "dropout_rate": 0.0},
        {"type": "Linear",    "units": 128, "activation": "ReLU", "batch_norm": False, "dropout": False, "dropout_rate": 0.0},
    ],
    "sine": [
        {"type": "LSTM", "hidden_size": 32, "num_layers": 1, "bidirectional": False, "rnn_dropout": 0.0},
    ],
    "multi_sine": [
        {"type": "LSTM", "hidden_size": 64, "num_layers": 1, "bidirectional": False, "rnn_dropout": 0.0},
    ],
    "airline": [
        {"type": "LSTM", "hidden_size": 32, "num_layers": 1, "bidirectional": False, "rnn_dropout": 0.0},
    ],
}


def suggested(name: str) -> list[dict]:
    """Return a copy of the suggested hidden-layer list for `name`."""
    if name not in _SUGGESTED:
        raise KeyError(f"No suggested architecture for {name!r}.")
    return [dict(layer) for layer in _SUGGESTED[name]]


def _load_sequential(
    name: str, seed: int, seq_len: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    series = _SERIES[name](seed)
    if name in _LOG_TRANSFORM:
        series = np.log(series)

    n_windows = len(series) - seq_len
    if seq_len < 2 or n_windows < _MIN_WINDOWS:
        raise ValueError(
            f"seq_len={seq_len} is out of range for {name!r} (T={len(series)}); "
            f"use 2 <= seq_len <= {len(series) - _MIN_WINDOWS}."
        )
    n_train = int(0.8 * n_windows)

    # Normalise with statistics of every point a training window can see
    # (inputs and targets up to index seq_len + n_train - 1) - nothing later.
    train_span = series[: seq_len + n_train]
    mu, sigma = float(train_span.mean()), float(train_span.std())
    series = ((series - mu) / sigma).astype(np.float32)

    X, y = make_windows(series, seq_len)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def load(
    name: str, seed: int = 42, seq_len: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_train, y_train, X_test, y_test).

    X arrays: float32 - (N, D) tabular/image, (N, seq_len, 1) sequential.
    y arrays: int64 for classification, float32 (N,) for regression.
    seq_len is used only by sequential datasets (defaults to meta(name).seq_len).
    """
    if name not in _META:
        raise KeyError(f"Unknown dataset {name!r}. Available: {list()}")

    if name in _SERIES:
        return _load_sequential(name, seed, seq_len or _META[name].seq_len)

    # --- special cases with their own splits ---
    if name == "xor":
        X, y = xor()
        return X, y, X, y

    if name == "mnist":
        return mnist()

    if name == "fashion_mnist":
        return fashion_mnist()

    if name == "cifar10":
        return cifar10()

    # --- generate / load raw (X, y) ---
    if name == "moons":
        X, y = moons(seed=seed)
    elif name == "spirals":
        X, y = spirals(seed=seed)
    elif name == "iris":
        X, y = iris()
    elif name == "breast_cancer":
        X, y = breast_cancer()
    elif name == "wine":
        X, y = wine()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    if name in _NEEDS_SCALING:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)

    return X_train, y_train, X_test, y_test
