"""
Generalised linear models (Family 1, 9 regressors).

Maps to Table 3 of the manuscript: LR, Ridge, Lasso, ElasticNet, BayesianRidge,
ARDR, HuberRegressor, QuantileRegressor, TheilSenRegressor.
"""
from __future__ import annotations

import numpy as np
from sklearn import linear_model as sklm
from sklearn.multioutput import MultiOutputRegressor

from ebw_ml.models.base import BaseRegressor, FitContext


class _SklearnMultiOutputWrap(BaseRegressor):
    """Helper: wrap any sklearn estimator with MultiOutputRegressor."""

    _sklearn_cls: type
    _sklearn_kw: dict = {}

    def _fit(self, X: np.ndarray, Y: np.ndarray, ctx: FitContext) -> None:
        base = self._sklearn_cls(**self._sklearn_kw, **self.hp)
        if Y.shape[1] == 1:
            self._est = base.fit(X, Y.ravel())
            self._single = True
        else:
            self._est = MultiOutputRegressor(base).fit(X, Y)
            self._single = False

    def _predict(self, X: np.ndarray) -> np.ndarray:
        y = self._est.predict(X)
        return y.reshape(-1, 1) if y.ndim == 1 else y


class LRRegressor(_SklearnMultiOutputWrap):
    name = "lr"; family = "GLM"
    search_space = {"fit_intercept": [True, False]}
    _sklearn_cls = sklm.LinearRegression


class RidgeRegressor(_SklearnMultiOutputWrap):
    name = "ridge"; family = "GLM"
    search_space = {"alpha": (1e-4, 1e2, "log-uniform")}
    _sklearn_cls = sklm.Ridge
    _sklearn_kw = {"random_state": 42}


class LassoRegressor(_SklearnMultiOutputWrap):
    name = "lasso"; family = "GLM"
    search_space = {"alpha": (1e-4, 1e1, "log-uniform")}
    _sklearn_cls = sklm.Lasso
    _sklearn_kw = {"random_state": 42, "max_iter": 5000}


class ElasticNetRegressor(_SklearnMultiOutputWrap):
    name = "elasticnet"; family = "GLM"
    search_space = {"alpha": (1e-4, 1e1, "log-uniform"),
                    "l1_ratio": (0.0, 1.0, "uniform")}
    _sklearn_cls = sklm.ElasticNet
    _sklearn_kw = {"random_state": 42, "max_iter": 5000}


class BayesianRidgeRegressor(_SklearnMultiOutputWrap):
    name = "bridge"; family = "GLM"
    search_space = {"alpha_1": (1e-7, 1e-3, "log-uniform"),
                    "lambda_1": (1e-7, 1e-3, "log-uniform")}
    _sklearn_cls = sklm.BayesianRidge

    def _predict_dist(self, X: np.ndarray):
        # BayesianRidge gives per-target std for single-output sub-estimators.
        if self._single:
            mu, std = self._est.predict(X, return_std=True)
            return mu.reshape(-1, 1), std.reshape(-1, 1)
        mus, stds = [], []
        for est in self._est.estimators_:
            mu, std = est.predict(X, return_std=True)
            mus.append(mu); stds.append(std)
        return np.column_stack(mus), np.column_stack(stds)


class ARDRRegressor(_SklearnMultiOutputWrap):
    name = "ardr"; family = "GLM"
    search_space = {"alpha_1": (1e-7, 1e-3, "log-uniform"),
                    "threshold_lambda": (1e2, 1e5, "log-uniform")}
    _sklearn_cls = sklm.ARDRegression


class HuberRegressorWrap(_SklearnMultiOutputWrap):
    name = "huber"; family = "GLM"
    search_space = {"epsilon": (1.0, 3.0, "uniform"),
                    "alpha": (1e-5, 1e-1, "log-uniform")}
    _sklearn_cls = sklm.HuberRegressor
    _sklearn_kw = {"max_iter": 500}


class QuantileRegressorWrap(_SklearnMultiOutputWrap):
    name = "quantile"; family = "GLM"
    search_space = {"alpha": (1e-3, 1e1, "log-uniform")}
    _sklearn_cls = sklm.QuantileRegressor
    _sklearn_kw = {"quantile": 0.5, "solver": "highs"}


class TheilSenRegressorWrap(_SklearnMultiOutputWrap):
    name = "theilsen"; family = "GLM"
    search_space = {"max_subpopulation": [500, 1000, 5000]}
    _sklearn_cls = sklm.TheilSenRegressor
    _sklearn_kw = {"random_state": 42, "n_jobs": 1}


GLM_MODELS = [
    LRRegressor, RidgeRegressor, LassoRegressor, ElasticNetRegressor,
    BayesianRidgeRegressor, ARDRRegressor, HuberRegressorWrap,
    QuantileRegressorWrap, TheilSenRegressorWrap,
]
