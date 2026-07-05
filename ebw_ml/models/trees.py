"""
Tree, ensemble and gradient-boosting regressors (Families 3 and 4, 11 models).

Trees / ensembles (4):  DTR, RFR, ETR, Bagging.
Gradient boosting (7):  GBR, HGBR, AdaBoost, XGBoost, LightGBM, CatBoost, NGBoost.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor

from ebw_ml.models.base import BaseRegressor, FitContext

# Silence noisy library banners
os.environ.setdefault("LIGHTGBM_VERBOSITY", "-1")


class _SklearnEnsemble(BaseRegressor):
    _cls: type
    _kw: dict = {"random_state": 42}
    _multi_native: bool = True

    def _fit(self, X, Y, ctx: FitContext):
        kw = dict(self._kw)
        if "random_state" in kw:
            kw["random_state"] = ctx.seed
        base = self._cls(**kw, **self.hp)
        if Y.shape[1] > 1 and not self._multi_native:
            self._est = MultiOutputRegressor(base).fit(X, Y)
        else:
            self._est = base.fit(X, Y if Y.shape[1] > 1 else Y.ravel())

    def _predict(self, X):
        y = self._est.predict(X)
        return y.reshape(-1, 1) if y.ndim == 1 else y


class DTRegressor(_SklearnEnsemble):
    name = "dtr"; family = "Trees"
    search_space = {"max_depth": [3, 5, 8, 12, None],
                    "min_samples_split": [2, 5, 10]}
    _cls = DecisionTreeRegressor


class RFRegressor(_SklearnEnsemble):
    name = "rfr"; family = "Trees"
    search_space = {"n_estimators": [100, 300, 500],
                    "max_depth": [5, 10, 20, None],
                    "min_samples_split": [2, 5, 10]}
    _cls = RandomForestRegressor
    _kw = {"random_state": 42, "n_jobs": 1}


class ETRegressor(_SklearnEnsemble):
    name = "etr"; family = "Trees"
    search_space = {"n_estimators": [100, 300, 500],
                    "max_depth": [5, 10, 20, None],
                    "min_samples_split": [2, 5, 10]}
    _cls = ExtraTreesRegressor
    _kw = {"random_state": 42, "n_jobs": 1}


class BaggingRegr(_SklearnEnsemble):
    name = "bag"; family = "Trees"
    search_space = {"n_estimators": [50, 100, 200],
                    "max_samples": (0.5, 1.0, "uniform")}
    _cls = BaggingRegressor
    _kw = {"random_state": 42, "n_jobs": 1}
    _multi_native = False


class GBRegressor(_SklearnEnsemble):
    name = "gbr"; family = "Boosting"
    search_space = {"n_estimators": [100, 300, 500],
                    "max_depth": [2, 3, 5],
                    "learning_rate": (1e-3, 0.3, "log-uniform")}
    _cls = GradientBoostingRegressor
    _multi_native = False


class HGBRegressor(_SklearnEnsemble):
    name = "hgbr"; family = "Boosting"
    search_space = {"max_iter": [100, 300, 500],
                    "max_depth": [3, 5, 8, None],
                    "learning_rate": (1e-3, 0.3, "log-uniform"),
                    "l2_regularization": (1e-6, 1.0, "log-uniform")}
    _cls = HistGradientBoostingRegressor
    _multi_native = False


class AdaBoostRegressorWrap(_SklearnEnsemble):
    name = "ada"; family = "Boosting"
    search_space = {"n_estimators": [50, 100, 200],
                    "learning_rate": (1e-3, 1.0, "log-uniform")}
    _cls = AdaBoostRegressor
    _multi_native = False


class XGBRegressorWrap(BaseRegressor):
    name = "xgb"; family = "Boosting"
    search_space = {"n_estimators": [100, 300, 500],
                    "max_depth": [3, 5, 8],
                    "learning_rate": (1e-3, 0.3, "log-uniform"),
                    "subsample": (0.5, 1.0, "uniform"),
                    "colsample_bytree": (0.5, 1.0, "uniform")}

    def _fit(self, X, Y, ctx: FitContext):
        import xgboost as xgb
        kw = dict(self.hp); kw.setdefault("verbosity", 0)
        kw.setdefault("tree_method", "hist")
        kw.setdefault("n_jobs", 1)
        kw["random_state"] = ctx.seed
        # GPU acceleration if available
        dev = ctx.resolve_device()
        if dev.startswith("cuda"):
            kw["device"] = "cuda"
        base = xgb.XGBRegressor(**kw)
        if Y.shape[1] > 1:
            self._est = MultiOutputRegressor(base).fit(X, Y)
        else:
            self._est = base.fit(X, Y.ravel())

    def _predict(self, X):
        y = self._est.predict(X)
        return y.reshape(-1, 1) if y.ndim == 1 else y


class LightGBMRegressor(BaseRegressor):
    name = "lgbm"; family = "Boosting"
    search_space = {"n_estimators": [100, 300, 500],
                    "num_leaves": [15, 31, 63],
                    "learning_rate": (1e-3, 0.3, "log-uniform"),
                    "min_child_samples": [3, 5, 10],
                    "subsample": (0.5, 1.0, "uniform")}

    def _fit(self, X, Y, ctx: FitContext):
        import lightgbm as lgb
        kw = dict(self.hp); kw.setdefault("verbose", -1); kw.setdefault("n_jobs", 1)
        kw["random_state"] = ctx.seed
        # LightGBM GPU is opt-in: the PyPI wheel is built without CUDA support;
        # users must rebuild with -DUSE_CUDA=1 and then set EBW_USE_GPU_BOOSTING=1.
        dev = ctx.resolve_device()
        if dev.startswith("cuda") and os.environ.get("EBW_USE_GPU_BOOSTING") == "1":
            kw["device_type"] = "cuda"
        base = lgb.LGBMRegressor(**kw)
        if Y.shape[1] > 1:
            self._est = MultiOutputRegressor(base).fit(X, Y)
        else:
            self._est = base.fit(X, Y.ravel())

    def _predict(self, X):
        y = self._est.predict(X)
        return y.reshape(-1, 1) if y.ndim == 1 else y


class CatBoostRegressorWrap(BaseRegressor):
    name = "cat"; family = "Boosting"
    search_space = {"iterations": [200, 500, 1000],
                    "depth": [4, 6, 8],
                    "learning_rate": (1e-3, 0.3, "log-uniform"),
                    "l2_leaf_reg": (1.0, 10.0, "log-uniform")}

    def _fit(self, X, Y, ctx: FitContext):
        from catboost import CatBoostRegressor as _CB
        kw = dict(self.hp); kw.setdefault("verbose", False); kw.setdefault("thread_count", 1)
        kw["random_seed"] = ctx.seed
        # CatBoost GPU is opt-in: it requires a CatBoost build matched to the
        # host CUDA driver version. On many systems (e.g. driver 535 with the
        # PyPI wheel) it raises a CUDA error at training time. Default to CPU
        # and let users enable GPU via EBW_USE_GPU_BOOSTING=1 if their stack
        # matches.
        dev = ctx.resolve_device()
        if dev.startswith("cuda") and os.environ.get("EBW_USE_GPU_BOOSTING") == "1":
            kw["task_type"] = "GPU"; kw["devices"] = "0"
        if Y.shape[1] > 1:
            kw["loss_function"] = "MultiRMSE"
            self._est = _CB(**kw).fit(X, Y)
            self._multi = True
        else:
            self._est = _CB(**kw).fit(X, Y.ravel())
            self._multi = False

    def _predict(self, X):
        y = self._est.predict(X)
        return y.reshape(-1, 1) if y.ndim == 1 else y


class NGBoostRegressor(BaseRegressor):
    name = "ngb"; family = "Boosting"
    search_space = {"n_estimators": [100, 300, 500],
                    "learning_rate": (1e-3, 0.3, "log-uniform")}

    def _fit(self, X, Y, ctx: FitContext):
        from ngboost import NGBRegressor
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = NGBRegressor(verbose=False, random_state=ctx.seed, **self.hp)
            if Y.shape[1] > 1:
                self._est = MultiOutputRegressor(base).fit(X, Y)
            else:
                self._est = base.fit(X, Y.ravel())

    def _predict(self, X):
        y = self._est.predict(X)
        return y.reshape(-1, 1) if y.ndim == 1 else y

    def _predict_dist(self, X):
        # Pull mean+std from each NGBoost sub-model.
        mus, stds = [], []
        ests = self._est.estimators_ if hasattr(self._est, "estimators_") else [self._est]
        for est in ests:
            dist = est.pred_dist(X)
            mus.append(dist.loc); stds.append(dist.scale)
        return np.column_stack(mus), np.column_stack(stds)


TREE_MODELS = [DTRegressor, RFRegressor, ETRegressor, BaggingRegr]
BOOSTING_MODELS = [
    GBRegressor, HGBRegressor, AdaBoostRegressorWrap, XGBRegressorWrap,
    LightGBMRegressor, CatBoostRegressorWrap, NGBoostRegressor,
]
