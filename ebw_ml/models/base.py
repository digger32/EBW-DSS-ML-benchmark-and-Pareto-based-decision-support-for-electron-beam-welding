"""
Base abstractions for the multi-output regression benchmark.

All regressors expose three methods:

* ``fit(X, Y)``    -- train on inputs X (n x 4) and outputs Y (n x 2).
* ``predict(X)``  -- return point predictions (n x 2).
* ``predict_dist(X)`` -- return (mean, std) per output, where supported;
  models without native variance estimates raise ``NotImplementedError``.

Device resolution (patch v5)
----------------------------
``FitContext.resolve_device`` now resolves the effective torch device in the
following priority order:

1. ``EBW_DEVICE`` environment variable, if set (e.g. ``EBW_DEVICE=cpu``).
   This is a *global operator override* and wins over everything. It exists so
   that a whole campaign -- including the HPO inner loop, which constructs its
   own default ``FitContext`` and would otherwise ignore ``--device`` -- can be
   pinned to CPU from the launch script with a single environment variable.
2. An explicit ``device`` other than ``"auto"`` passed into the context.
3. ``"auto"``: a *one-time, cached* empirical probe. ``torch.cuda.is_available()``
   can return ``True`` on a host whose CUDA runtime cannot actually run a kernel
   (driver/runtime mismatch). The probe runs a tiny CUDA matmul; if it raises,
   the device falls back to ``"cpu"`` automatically. The result is cached per
   process so the probe runs at most once.

Note: when ``EBW_DEVICE=cpu`` is set, the auto-probe never runs, so CPU worker
processes never create a CUDA context -- important when many shard processes
share a single GPU host.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

INPUT_COLS = ["IW", "IF", "VW", "FP"]
OUTPUT_COLS = ["Depth", "Width"]

# Cached result of the one-time auto device probe (None = not yet probed).
_AUTO_DEVICE: str | None = None


def _probe_auto_device() -> str:
    """Empirically decide 'cuda' vs 'cpu' for device='auto', cached per process.

    Returns 'cuda' only if a tiny CUDA matmul actually succeeds; otherwise 'cpu'.
    """
    global _AUTO_DEVICE
    if _AUTO_DEVICE is not None:
        return _AUTO_DEVICE
    dev = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            try:
                a = torch.randn(16, 16, device="cuda")
                b = torch.randn(16, 16, device="cuda")
                _ = a @ b
                torch.cuda.synchronize()
                dev = "cuda"
            except Exception:
                # CUDA is "available" but a kernel cannot run -> fall back.
                dev = "cpu"
    except ImportError:
        dev = "cpu"
    _AUTO_DEVICE = dev
    return dev


@dataclass
class FitContext:
    """Optional context passed through ``fit`` to support stateful logging."""
    seed: int = 42
    feature_names: Sequence[str] = field(default_factory=lambda: tuple(INPUT_COLS))
    target_names: Sequence[str] = field(default_factory=lambda: tuple(OUTPUT_COLS))
    device: str = "auto"  # "auto", "cpu", "cuda", "cuda:0", etc.

    def resolve_device(self) -> str:
        """Resolve the effective torch device (see module docstring for order)."""
        forced = os.environ.get("EBW_DEVICE", "").strip()
        if forced:
            return forced
        if self.device != "auto":
            return self.device
        return _probe_auto_device()


class BaseRegressor(ABC):
    """Abstract base class for all multi-output regressors.

    Subclasses must override ``_fit`` and ``_predict`` and may override
    ``_predict_dist`` to expose epistemic or aleatoric uncertainty.
    """

    #: short identifier used in tables and figures (e.g. "rf", "xgb")
    name: str = "abstract"

    #: human-readable family ("GLM", "Kernel", "Trees", "Boosting", "Other", "NN/DL")
    family: str = "abstract"

    #: hyperparameter search space; overridden in each subclass
    search_space: dict[str, Any] = {}

    def __init__(self, **hp: Any) -> None:
        self.hp = dict(hp)
        self._fitted: bool = False
        self._n_features: int | None = None
        self._n_targets: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray | pd.DataFrame, Y: np.ndarray | pd.DataFrame,
            ctx: FitContext | None = None) -> "BaseRegressor":
        Xa = self._as_array(X)
        Ya = self._as_array(Y)
        if Ya.ndim == 1:
            Ya = Ya.reshape(-1, 1)
        self._n_features = Xa.shape[1]
        self._n_targets = Ya.shape[1]
        self._fit(Xa, Ya, ctx or FitContext())
        self._fitted = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        assert self._fitted, f"{self.name} not fitted"
        Xa = self._as_array(X)
        Yhat = self._predict(Xa)
        if Yhat.ndim == 1:
            Yhat = Yhat.reshape(-1, 1)
        return Yhat

    def predict_dist(self, X: np.ndarray | pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) arrays. Default: zero std for point-only models."""
        assert self._fitted, f"{self.name} not fitted"
        Xa = self._as_array(X)
        return self._predict_dist(Xa)

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------
    @abstractmethod
    def _fit(self, X: np.ndarray, Y: np.ndarray, ctx: FitContext) -> None: ...

    @abstractmethod
    def _predict(self, X: np.ndarray) -> np.ndarray: ...

    def _predict_dist(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = self._predict(X)
        return mu, np.zeros_like(mu)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def _as_array(arr: np.ndarray | pd.DataFrame) -> np.ndarray:
        if isinstance(arr, pd.DataFrame):
            return arr.values.astype(np.float64)
        return np.asarray(arr, dtype=np.float64)
