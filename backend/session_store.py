"""In-memory session store.

Keyed by UUID string.  Each entry is a plain dict that grows over the
session lifetime:

  Initial (at training start):
    { "stop_event": threading.Event }

  After training completes:
    { "stop_event": ..., "model": nn.Module, "history": dict,
      "X_test": np.ndarray, "y_test": np.ndarray, "config": dict }
"""
from __future__ import annotations

sessions: dict[str, dict] = {}
