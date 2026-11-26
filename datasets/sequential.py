"""Sequential (time-series) datasets for LSTM / GRU demonstrations.

Series loaders return a raw 1-D float32 array of length T.  Windowing,
splitting, and normalisation live in registry.load() so that seq_len can be
chosen per training run.

  sine()        : single-frequency sine + Gaussian noise   (regression warmup)
  multi_sine()  : three incommensurate sines + noise       (harder: long period)
  airline()     : monthly airline passengers 1949-1960     (classic benchmark)

make_windows(series, seq_len) turns a series into supervised one-step-ahead
pairs:  X[i] = series[i : i+seq_len],  y[i] = series[i+seq_len].
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_AIRLINE_CSV = Path(__file__).parent / "airline_passengers.csv"


def sine(n: int = 1000, dt: float = 0.1, noise: float = 0.05, seed: int = 42) -> np.ndarray:
    """sin(t) sampled every dt, plus N(0, noise^2).  Period ~= 63 samples."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    return (np.sin(t) + rng.normal(0.0, noise, n)).astype(np.float32)


def multi_sine(n: int = 1500, dt: float = 0.1, noise: float = 0.05, seed: int = 42) -> np.ndarray:
    """Superposition of three sines with incommensurate frequencies, plus noise.

    The slowest component (omega = 0.5, period ~= 126 samples) is longer than any
    reasonable window, so the model must infer phase from the faster components.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    y = (
        1.0 * np.sin(0.5 * t)
        + 0.6 * np.sin(1.3 * t + 1.0)
        + 0.3 * np.sin(3.1 * t + 2.0)
    )
    return (y + rng.normal(0.0, noise, n)).astype(np.float32)


def airline() -> np.ndarray:
    """Box & Jenkins monthly international airline passengers (thousands), T=144."""
    data = np.loadtxt(_AIRLINE_CSV, delimiter=",", skiprows=1, usecols=1)
    return data.astype(np.float32)


def make_windows(series: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Return X: (N, seq_len, 1) float32 and y: (N,) float32, N = T - seq_len."""
    n = len(series) - seq_len
    idx = np.arange(seq_len)[None, :] + np.arange(n)[:, None]   # (N, seq_len)
    X = series[idx][..., None].astype(np.float32)               # (N, seq_len, 1)
    y = series[seq_len:].astype(np.float32)                     # (N,)
    return X, y
