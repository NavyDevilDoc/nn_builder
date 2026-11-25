"""Synthetic 2D datasets generated from scratch (XOR) or via sklearn (moons).
Spirals are parametric. All return float32 X, int64 y.
"""
import numpy as np


def xor() -> tuple[np.ndarray, np.ndarray]:
    """The 4 XOR points. X: (4, 2) float32, y: (4,) int64."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.int64)
    return X, y


def moons(n: int = 400, noise: float = 0.1, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaved half-circles. X: (n, 2) float32, y: (n,) int64."""
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    return X.astype(np.float32), y.astype(np.int64)


def spirals(n: int = 400, noise: float = 0.05, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaved Archimedean spirals. X: (n, 2) float32, y: (n,) int64.

    Each class traces theta in [0, 4pi]; class 1 is rotated by pi.
    """
    rng = np.random.default_rng(seed)
    n_per = n // 2
    # Uniform spacing in angle (sqrt gives uniform arc-length distribution)
    theta = np.sqrt(rng.uniform(0, 1, n_per)) * 4 * np.pi

    x0 = theta * np.cos(theta) / 4 + rng.normal(0, noise, n_per)
    y0 = theta * np.sin(theta) / 4 + rng.normal(0, noise, n_per)

    x1 = theta * np.cos(theta + np.pi) / 4 + rng.normal(0, noise, n_per)
    y1 = theta * np.sin(theta + np.pi) / 4 + rng.normal(0, noise, n_per)

    X = np.vstack([
        np.column_stack([x0, y0]),
        np.column_stack([x1, y1]),
    ]).astype(np.float32)
    y = np.array([0] * n_per + [1] * n_per, dtype=np.int64)
    return X, y
