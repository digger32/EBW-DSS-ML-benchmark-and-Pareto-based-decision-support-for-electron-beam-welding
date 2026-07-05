"""
Stacking and voting meta-ensembles (Family 7, 2 models).

* StackingRegressor: trains 4 diverse base learners (Ridge, RF, XGBoost,
  GPR) and a Ridge meta-learner with 5-fold internal cross-validation
  to produce out-of-fold predictions.
* VotingRegressor: averages the predictions of the same 4 base learners
  with equal weights.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, VotingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

from ebw_ml.models.base import BaseRegressor, FitContext


def _base_learners(seed: int) -> list[tuple[str, object]]:
    """Return 4 diverse base learners for stacking / voting."""
    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.1,
                            random_state=seed, verbosity=0, n_jobs=1,
                            tree_method="hist")
    except ImportError:
        xgb = RandomForestRegressor(n_estimators=200, random_state=seed + 1)
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
    return [
        ("ridge", Ridge(alpha=1.0, random_state=seed)),
        ("rf", RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1)),
        ("xgb", xgb),
        ("gpr", GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                          random_state=seed, alpha=1e-8)),
    ]


class StackingRegressorWrap(BaseRegressor):
    name = "stack"; family = "Ensemble"
    search_space = {"final_alpha": (1e-3, 1.0, "log-uniform"),
                    "cv": [3, 5]}

    def _fit(self, X, Y, ctx: FitContext):
        final = Ridge(alpha=float(self.hp.get("final_alpha", 0.1)),
                       random_state=ctx.seed)
        cv = int(self.hp.get("cv", 5))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = StackingRegressor(estimators=_base_learners(ctx.seed),
                                       final_estimator=final, cv=cv, n_jobs=1)
            if Y.shape[1] > 1:
                self._est = MultiOutputRegressor(base).fit(X, Y)
            else:
                self._est = base.fit(X, Y.ravel())

    def _predict(self, X):
        y = self._est.predict(X)
        return y.reshape(-1, 1) if y.ndim == 1 else y


class VotingRegressorWrap(BaseRegressor):
    name = "vote"; family = "Ensemble"
    search_space = {}  # plain equal-weight average; no tuneable hyperparameters

    def _fit(self, X, Y, ctx: FitContext):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = VotingRegressor(estimators=_base_learners(ctx.seed), n_jobs=1)
            if Y.shape[1] > 1:
                self._est = MultiOutputRegressor(base).fit(X, Y)
            else:
                self._est = base.fit(X, Y.ravel())

    def _predict(self, X):
        y = self._est.predict(X)
        return y.reshape(-1, 1) if y.ndim == 1 else y


ENSEMBLE_MODELS = [StackingRegressorWrap, VotingRegressorWrap]
