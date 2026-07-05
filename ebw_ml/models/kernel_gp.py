"""
Kernel-based regressors and Gaussian processes (Family 2, 4 regressors):
SVR (RBF), NuSVR, Kernel Ridge, Gaussian Process Regression.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import NuSVR, SVR

from ebw_ml.models.base import BaseRegressor, FitContext


class SVRRegressor(BaseRegressor):
    name = "svr_rbf"; family = "Kernel"
    search_space = {"C": (1e-2, 1e3, "log-uniform"),
                    "gamma": (1e-4, 1e1, "log-uniform"),
                    "epsilon": (1e-3, 1.0, "log-uniform")}

    def _fit(self, X, Y, ctx: FitContext):
        base = SVR(kernel="rbf", **self.hp)
        self._est = MultiOutputRegressor(base).fit(X, Y)

    def _predict(self, X):
        return self._est.predict(X)


class NuSVRRegressor(BaseRegressor):
    name = "nusvr"; family = "Kernel"
    search_space = {"C": (1e-2, 1e3, "log-uniform"),
                    "gamma": (1e-4, 1e1, "log-uniform"),
                    "nu": (0.1, 0.9, "uniform")}

    def _fit(self, X, Y, ctx: FitContext):
        base = NuSVR(kernel="rbf", **self.hp)
        self._est = MultiOutputRegressor(base).fit(X, Y)

    def _predict(self, X):
        return self._est.predict(X)


class KernelRidgeRegressor(BaseRegressor):
    name = "kridge"; family = "Kernel"
    search_space = {"alpha": (1e-4, 1e1, "log-uniform"),
                    "gamma": (1e-4, 1e1, "log-uniform")}

    def _fit(self, X, Y, ctx: FitContext):
        self._est = KernelRidge(kernel="rbf", **self.hp).fit(X, Y)

    def _predict(self, X):
        return self._est.predict(X)


class GPRRegressor(BaseRegressor):
    name = "gpr"; family = "Kernel"
    search_space = {"length_scale": (0.1, 10.0, "log-uniform"),
                    "noise_level": (1e-5, 1.0, "log-uniform")}

    def _fit(self, X, Y, ctx: FitContext):
        ls = self.hp.get("length_scale", 1.0)
        nl = self.hp.get("noise_level", 1e-3)
        kernel = ConstantKernel(1.0) * RBF(length_scale=ls) + WhiteKernel(noise_level=nl)
        base = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                         random_state=ctx.seed, alpha=1e-8,
                                         n_restarts_optimizer=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._est = MultiOutputRegressor(base).fit(X, Y)

    def _predict(self, X):
        return self._est.predict(X)

    def _predict_dist(self, X):
        mus, stds = [], []
        for est in self._est.estimators_:
            mu, std = est.predict(X, return_std=True)
            mus.append(mu); stds.append(std)
        return np.column_stack(mus), np.column_stack(stds)


KERNEL_MODELS = [SVRRegressor, NuSVRRegressor, KernelRidgeRegressor, GPRRegressor]
