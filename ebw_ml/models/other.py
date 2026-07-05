"""
Other classical regressors (Family 5, 4 models): KNN, PLS, MARS, Symbolic.

The MARS and Symbolic implementations are lightweight in-house variants:

* ``MARSLite`` -- a forward-selection hinge regression in the spirit of
  Friedman (1991). It greedily adds piecewise-linear basis functions of the
  form ``max(0, x - c)`` and ``max(0, c - x)`` per feature, with a small
  pool of candidate knots taken from the quantiles of each input.
* ``SymbolicPoly`` -- a stand-in for the genetic-programming symbolic
  regression of Koza (1994), using a degree-3 polynomial basis with Lasso
  selection. The conceptual contribution (sparse non-linear function
  expansion) is preserved; the optimisation differs in being convex.

These in-house variants are used because the original Py-Earth and gplearn
packages are not currently maintained for recent Python releases. The
production version of the manuscript code-base will substitute the canonical
libraries when available.
"""
from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures

from ebw_ml.models.base import BaseRegressor, FitContext


class KNNRegressor(BaseRegressor):
    name = "knn"; family = "Other"
    search_space = {"n_neighbors": [3, 5, 7, 10, 15],
                    "weights": ["uniform", "distance"],
                    "p": [1, 2]}

    def _fit(self, X, Y, ctx: FitContext):
        self._est = KNeighborsRegressor(**self.hp, n_jobs=1).fit(X, Y)

    def _predict(self, X):
        return self._est.predict(X)


class PLSRegressorWrap(BaseRegressor):
    name = "pls"; family = "Other"
    search_space = {"n_components": [1, 2, 3, 4]}

    def _fit(self, X, Y, ctx: FitContext):
        n = min(self.hp.get("n_components", 2), X.shape[1])
        self._est = PLSRegression(n_components=n).fit(X, Y)

    def _predict(self, X):
        return self._est.predict(X)


class MARSLite(BaseRegressor):
    """Forward-selection hinge regression (Friedman, 1991, simplified).

    For each feature, candidate knots are placed at the 0.25, 0.5, 0.75
    quantiles. At each forward step the basis function whose addition most
    reduces residual sum of squares is added. Basis functions are
    ``max(0, x_j - c)`` and ``max(0, c - x_j)``. A fixed budget of
    ``max_terms`` basis functions is used; backward pruning is omitted in
    this lightweight variant.
    """

    name = "mars"; family = "Other"
    search_space = {"max_terms": [6, 10, 15, 20]}

    def _fit(self, X, Y, ctx: FitContext):
        max_terms = int(self.hp.get("max_terms", 10))
        n, d = X.shape
        knots = []
        for j in range(d):
            q = np.quantile(X[:, j], [0.25, 0.5, 0.75])
            knots.append(q)
        # Build candidate basis: intercept + all (x - c)+ and (c - x)+
        cand_funcs = []
        cand_funcs.append(("const", None, None))
        for j in range(d):
            for c in knots[j]:
                cand_funcs.append(("pos", j, c))
                cand_funcs.append(("neg", j, c))

        chosen: list[tuple] = [cand_funcs[0]]  # always include intercept
        remaining = list(cand_funcs[1:])

        def design(F, X):
            cols = []
            for tag, j, c in F:
                if tag == "const":
                    cols.append(np.ones(X.shape[0]))
                elif tag == "pos":
                    cols.append(np.maximum(0.0, X[:, j] - c))
                else:
                    cols.append(np.maximum(0.0, c - X[:, j]))
            return np.column_stack(cols)

        for _ in range(max_terms - 1):
            best = None
            best_rss = np.inf
            best_idx = -1
            for k, f in enumerate(remaining):
                trial = chosen + [f]
                D = design(trial, X)
                try:
                    coef, *_ = np.linalg.lstsq(D, Y, rcond=None)
                    pred = D @ coef
                    rss = float(np.sum((Y - pred) ** 2))
                    if rss < best_rss:
                        best_rss = rss; best = f; best_idx = k
                except np.linalg.LinAlgError:
                    continue
            if best is None:
                break
            chosen.append(best); remaining.pop(best_idx)

        self._basis = chosen
        D = design(chosen, X)
        self._coef, *_ = np.linalg.lstsq(D, Y, rcond=None)

    def _predict(self, X):
        def design(F, X):
            cols = []
            for tag, j, c in F:
                if tag == "const":
                    cols.append(np.ones(X.shape[0]))
                elif tag == "pos":
                    cols.append(np.maximum(0.0, X[:, j] - c))
                else:
                    cols.append(np.maximum(0.0, c - X[:, j]))
            return np.column_stack(cols)
        return design(self._basis, X) @ self._coef


class SymbolicPoly(BaseRegressor):
    """Polynomial expansion with Lasso (stand-in for genetic-programming SR).

    Constructs a degree-d polynomial basis and selects sparse coefficients
    via Lasso. The result is a closed-form polynomial of low effective
    dimension, retaining the symbolic-regression spirit of producing a
    sparse, interpretable closed-form expression.
    """

    name = "sym"; family = "Other"
    search_space = {"degree": [2, 3],
                    "alpha": (1e-4, 1.0, "log-uniform")}

    def _fit(self, X, Y, ctx: FitContext):
        d = int(self.hp.get("degree", 2))
        a = float(self.hp.get("alpha", 1e-2))
        self._poly = PolynomialFeatures(degree=d, include_bias=False)
        Z = self._poly.fit_transform(X)
        base = Lasso(alpha=a, max_iter=5000, random_state=ctx.seed)
        if Y.shape[1] > 1:
            self._est = MultiOutputRegressor(base).fit(Z, Y)
            self._multi = True
        else:
            self._est = base.fit(Z, Y.ravel()); self._multi = False

    def _predict(self, X):
        Z = self._poly.transform(X)
        y = self._est.predict(Z)
        return y.reshape(-1, 1) if y.ndim == 1 else y


OTHER_MODELS = [KNNRegressor, PLSRegressorWrap, MARSLite, SymbolicPoly]
