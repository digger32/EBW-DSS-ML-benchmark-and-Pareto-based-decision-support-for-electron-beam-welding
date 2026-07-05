"""
Uniform interface for the 12 hyperparameter optimisers compared in
Section 3.6 of the manuscript.

Every optimiser implements a single public method, ``search``, which receives:

* ``model_cls`` -- a subclass of ``ebw_ml.models.BaseRegressor``;
* ``X``, ``Y`` -- arrays of shape (n, 4) and (n, 2);
* ``cv`` -- a cross-validation splitter (default 5-fold);
* ``n_trials`` -- the budget;
* ``seed`` -- the random seed.

The method returns a ``SearchResult`` with the best hyperparameters, the
corresponding cross-validated RMSE, the full trial history and timing.

The 12 optimisers are deliberately driven by the same evaluation function
(``_make_objective``) so that comparison across optimisers measures only the
contribution of the search strategy.
"""
from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from sklearn.model_selection import KFold


@dataclass
class SearchResult:
    name: str
    best_hp: dict
    best_score: float            # mean RMSE across CV folds (minimisation)
    n_evaluations: int
    elapsed_s: float
    history: list[dict] = field(default_factory=list)


@dataclass
class SearchSpace:
    """Numeric and categorical hyperparameter specification.

    Each entry of ``space`` is:
      * (lo, hi, "uniform")       -- continuous uniform
      * (lo, hi, "log-uniform")   -- continuous log-uniform
      * [c1, c2, ...]             -- categorical / discrete-integer
    Constants (single-value lists or scalars) are passed through unchanged.
    """
    raw: dict[str, Any]

    def keys(self) -> list[str]:
        return list(self.raw.keys())

    def is_continuous(self, k: str) -> bool:
        v = self.raw[k]
        return isinstance(v, tuple) and len(v) == 3 and v[2] in ("uniform", "log-uniform")

    def is_log(self, k: str) -> bool:
        v = self.raw[k]
        return isinstance(v, tuple) and v[2] == "log-uniform"

    def bounds(self, k: str) -> tuple[float, float]:
        v = self.raw[k]
        return float(v[0]), float(v[1])

    def categories(self, k: str) -> list:
        v = self.raw[k]
        if isinstance(v, list):
            return v
        return [v]


def _make_cv(seed: int, n_splits: int = 5) -> KFold:
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def evaluate_hp(model_cls, hp: dict, X: np.ndarray, Y: np.ndarray,
                cv: KFold | None = None, seed: int = 42) -> float:
    """Mean RMSE across CV folds, averaged across both targets.

    All optimisers minimise this scalar. Failures return a large finite value
    so that optimisers see a usable signal rather than a NaN.
    """
    cv = cv or _make_cv(seed)
    fold_rmses = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr_idx, va_idx in cv.split(X):
            try:
                m = model_cls(**hp)
                m.fit(X[tr_idx], Y[tr_idx])
                yp = m.predict(X[va_idx])
                rmse_d = float(np.sqrt(np.mean((yp[:, 0] - Y[va_idx, 0]) ** 2)))
                rmse_w = float(np.sqrt(np.mean((yp[:, 1] - Y[va_idx, 1]) ** 2)))
                fold_rmses.append(0.5 * (rmse_d + rmse_w))
            except Exception:
                fold_rmses.append(10.0)  # large but finite penalty
    return float(np.mean(fold_rmses))


class BaseOptimiser(ABC):
    name: str = "abstract"
    family: str = "abstract"

    def __init__(self, n_trials: int = 30, seed: int = 42) -> None:
        self.n_trials = int(n_trials)
        self.seed = int(seed)

    @abstractmethod
    def search(self, model_cls, X: np.ndarray, Y: np.ndarray,
               cv: KFold | None = None) -> SearchResult: ...

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_space(model_cls) -> SearchSpace:
        return SearchSpace(dict(model_cls.search_space))

    def _objective_factory(self, model_cls, X, Y, cv) -> Callable[[dict], float]:
        history: list[dict] = []

        def obj(hp: dict) -> float:
            score = evaluate_hp(model_cls, hp, X, Y, cv=cv, seed=self.seed)
            history.append({"hp": dict(hp), "score": float(score)})
            return score

        obj.history = history  # type: ignore[attr-defined]
        return obj
