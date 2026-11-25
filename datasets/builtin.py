"""Wrappers for sklearn tabular datasets and MNIST/FashionMNIST/CIFAR-10.
All return float32 X, int64 y.  Image datasets are the only pre-split loaders.
"""
from pathlib import Path
import numpy as np

_DATA_DIR = Path(__file__).parent.parent / "data"


def iris() -> tuple[np.ndarray, np.ndarray]:
    """Iris flowers. X: (150, 4) float32, y: (150,) int64, 3 classes."""
    from sklearn.datasets import load_iris
    d = load_iris()
    return d.data.astype(np.float32), d.target.astype(np.int64)


def breast_cancer() -> tuple[np.ndarray, np.ndarray]:
    """Breast Cancer Wisconsin. X: (569, 30) float32, y: (569,) int64, 2 classes."""
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    return d.data.astype(np.float32), d.target.astype(np.int64)


def wine() -> tuple[np.ndarray, np.ndarray]:
    """Wine recognition. X: (178, 13) float32, y: (178,) int64, 3 classes."""
    from sklearn.datasets import load_wine
    d = load_wine()
    return d.data.astype(np.float32), d.target.astype(np.int64)


def mnist(n_train: int = 10_000) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """MNIST flattened to 784 features, pixel values in [0, 1].

    Returns pre-split (X_train, y_train, X_test, y_test).
    X_train: (n_train, 784) float32, y_train: (n_train,) int64
    X_test:  (10000, 784)   float32, y_test:  (10000,)  int64
    """
    from torchvision.datasets import MNIST

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_ds = MNIST(root=_DATA_DIR, train=True, download=True)
    test_ds = MNIST(root=_DATA_DIR, train=False, download=True)

    X_train = train_ds.data[:n_train].numpy().reshape(n_train, -1).astype(np.float32) / 255.0
    y_train = train_ds.targets[:n_train].numpy().astype(np.int64)

    n_test = len(test_ds)
    X_test = test_ds.data.numpy().reshape(n_test, -1).astype(np.float32) / 255.0
    y_test = test_ds.targets.numpy().astype(np.int64)

    return X_train, y_train, X_test, y_test


def fashion_mnist(n_train: int = 10_000) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fashion-MNIST flattened to 784 features, pixel values in [0, 1].

    Returns pre-split (X_train, y_train, X_test, y_test).
    X_train: (n_train, 784) float32, y_train: (n_train,) int64
    X_test:  (10000, 784)   float32, y_test:  (10000,)  int64
    """
    from torchvision.datasets import FashionMNIST

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_ds = FashionMNIST(root=_DATA_DIR, train=True, download=True)
    test_ds  = FashionMNIST(root=_DATA_DIR, train=False, download=True)

    X_train = train_ds.data[:n_train].numpy().reshape(n_train, -1).astype(np.float32) / 255.0
    y_train = train_ds.targets[:n_train].numpy().astype(np.int64)

    n_test  = len(test_ds)
    X_test  = test_ds.data.numpy().reshape(n_test, -1).astype(np.float32) / 255.0
    y_test  = test_ds.targets.numpy().astype(np.int64)

    return X_train, y_train, X_test, y_test


def cifar10(n_train: int = 10_000) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """CIFAR-10 flattened to 3072 features (32x32 RGB), pixel values in [0, 1].

    Returns pre-split (X_train, y_train, X_test, y_test).
    X_train: (n_train, 3072) float32, y_train: (n_train,) int64
    X_test:  (10000, 3072)   float32, y_test:  (10000,)  int64

    Note: .data is (N, 32, 32, 3) uint8 HWC numpy array; .targets is a Python list.
    """
    from torchvision.datasets import CIFAR10

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_ds = CIFAR10(root=_DATA_DIR, train=True, download=True)
    test_ds  = CIFAR10(root=_DATA_DIR, train=False, download=True)

    # .data: (N, H, W, C) uint8 numpy — transpose to (N, C, H, W) then flatten
    X_train = (
        train_ds.data[:n_train]              # (n_train, 32, 32, 3)
        .transpose(0, 3, 1, 2)              # (n_train, 3, 32, 32) CHW
        .reshape(n_train, -1)               # (n_train, 3072)
        .astype(np.float32) / 255.0
    )
    y_train = np.array(train_ds.targets[:n_train], dtype=np.int64)

    n_test  = len(test_ds)
    X_test  = (
        test_ds.data                         # (n_test, 32, 32, 3)
        .transpose(0, 3, 1, 2)
        .reshape(n_test, -1)
        .astype(np.float32) / 255.0
    )
    y_test  = np.array(test_ds.targets, dtype=np.int64)

    return X_train, y_train, X_test, y_test
