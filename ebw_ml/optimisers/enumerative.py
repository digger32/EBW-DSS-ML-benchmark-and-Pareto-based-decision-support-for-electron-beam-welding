"""
Enumerative search baselines: Grid Search and Random Search.
"""
from __future__ import annotations

import time

import numpy as np

from ebw_ml.optimisers.base import (
    BaseOptimiser,
    SearchResult,
    evaluate_hp,
)


def _grid_from_space(space: dict, n_grid: int = 5) -> list[dict]:
    """Build a Cartesian-product grid from a search space.

    Continuous ranges are discretised with ``n_grid`` points; log-uniform
    ranges use logarithmic spacing. Categorical lists are used verbatim.
    """
    axes: dict[str, list] = {}
    for k, v in space.items():
        if isinstance(v, tuple) and len(v) == 3 and v[2] in ("uniform", "log-uniform"):
            lo, hi = float(v[0]), float(v[1])
            if v[2] == "log-uniform":
                axes[k] = list(np.logspace(np.log10(lo), np.log10(hi), n_grid))
            else:
                axes[k] = list(np.linspace(lo, hi, n_grid))
        elif isinstance(v, list):
            axes[k] = list(v)
        else:
            axes[k] = [v]
    # Cartesian product
    keys = list(axes.keys())
    out: list[dict] = [{}]
    for k in keys:
        new = []
        for prev in out:
            for val in axes[k]:
                d = dict(prev); d[k] = val
                new.append(d)
        out = new
    return out


def _random_sample(space: dict, rng: np.random.Generator) -> dict:
    out: dict = {}
    for k, v in space.items():
        if isinstance(v, tuple) and len(v) == 3 and v[2] in ("uniform", "log-uniform"):
            lo, hi = float(v[0]), float(v[1])
            if v[2] == "log-uniform":
                out[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            else:
                out[k] = float(rng.uniform(lo, hi))
        elif isinstance(v, list):
            out[k] = v[int(rng.integers(0, len(v)))]
        else:
            out[k] = v
    return out


class GridSearchOpt(BaseOptimiser):
    name = "grid"; family = "Enumerative"

    def search(self, model_cls, X, Y, cv=None):
        t0 = time.time()
        # Choose grid resolution so that the grid does not exceed n_trials.
        space = dict(model_cls.search_space)
        # Estimate grid size for n_grid in {3, 4, 5}
        chosen = 3
        for ng in (5, 4, 3):
            n = len(_grid_from_space(space, n_grid=ng))
            if n <= self.n_trials:
                chosen = ng; break
        grid = _grid_from_space(space, n_grid=chosen)[: self.n_trials]
        obj = self._objective_factory(model_cls, X, Y, cv)
        best_score, best_hp = float("inf"), None
        for hp in grid:
            s = obj(hp)
            if s < best_score:
                best_score, best_hp = s, dict(hp)
        return SearchResult(self.name, best_hp or {}, best_score,
                            len(grid), time.time() - t0, obj.history)


class RandomSearchOpt(BaseOptimiser):
    name = "random"; family = "Enumerative"

    def search(self, model_cls, X, Y, cv=None):
        t0 = time.time()
        rng = np.random.default_rng(self.seed)
        space = dict(model_cls.search_space)
        obj = self._objective_factory(model_cls, X, Y, cv)
        best_score, best_hp = float("inf"), None
        for _ in range(self.n_trials):
            hp = _random_sample(space, rng)
            s = obj(hp)
            if s < best_score:
                best_score, best_hp = s, dict(hp)
        return SearchResult(self.name, best_hp or {}, best_score,
                            self.n_trials, time.time() - t0, obj.history)
