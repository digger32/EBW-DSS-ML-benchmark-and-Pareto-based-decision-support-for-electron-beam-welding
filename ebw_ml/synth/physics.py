"""
Physics-informed synthetic-data generator based on Rosenthal's moving
point-source thermal solution for the EBW process.

The generator follows Eq. (3) of the manuscript:

    D \\propto (eta * P) / (rho * cp * Vw * (Tm - T0)) * g_D(IF, FP)
    W \\propto (eta * P) / (rho * cp * Vw * (Tm - T0)) * g_W(IF, FP)

where P = U_acc * IW is the beam power. We assume constant material properties
for thin-walled titanium and absorb all dimensional pre-factors into
calibration constants. The shape functions g_D and g_W depend on focusing
current IF and focal position FP and are fitted from the real data set
through ordinary least squares in log--log space.

Inputs (IW, IF, VW, FP) are drawn from a Latin-hypercube design over the
operational envelope, then outputs D and W are produced via the fitted
relations plus heteroscedastic Gaussian residual noise estimated from the
empirical residuals.
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import qmc

EBW_COLUMNS = ["IW", "IF", "VW", "FP", "Depth", "Width"]
INPUT_COLS = ["IW", "IF", "VW", "FP"]


class PhysicsRosenthalGenerator:
    """Latin-hypercube + log-linear Rosenthal-style generator.

    The fitted form is

        log(Y) = a0 + a_IW * log(IW) + a_IF * log(IF) + a_VW * log(VW)
                  + a_FP * log(FP) + epsilon,    Y in {Depth, Width},

    which subsumes the canonical inverse-VW scaling of the moving-source
    solution as a special case (a_VW = -1) while letting the data correct
    departures from the idealised closed form. Residuals are drawn from a
    Gaussian whose standard deviation is the empirical residual standard
    deviation of the fit.
    """

    name = "physics"

    def __init__(self, seed: int = 42, ridge: float = 1e-8) -> None:
        self.seed = seed
        self.ridge = ridge  # small ridge for numerical stability
        self._coef_D: Optional[np.ndarray] = None
        self._coef_W: Optional[np.ndarray] = None
        self._sigma_D: Optional[float] = None
        self._sigma_W: Optional[float] = None
        self._bounds: Optional[dict[str, tuple[float, float]]] = None

    def _design_matrix(self, X: pd.DataFrame) -> np.ndarray:
        """Log inputs with an intercept column."""
        Xl = np.log(X[INPUT_COLS].values)
        return np.hstack([np.ones((Xl.shape[0], 1)), Xl])

    def _fit_log_linear(self, X_design: np.ndarray, y_log: np.ndarray) -> tuple[np.ndarray, float]:
        # Ridge-regularised normal equations.
        XtX = X_design.T @ X_design + self.ridge * np.eye(X_design.shape[1])
        Xty = X_design.T @ y_log
        beta = np.linalg.solve(XtX, Xty)
        residuals = y_log - X_design @ beta
        sigma = float(np.std(residuals, ddof=X_design.shape[1]))
        return beta, sigma

    def fit(self, df: pd.DataFrame) -> "PhysicsRosenthalGenerator":
        d = df[EBW_COLUMNS].copy()
        # Operational envelope from the real data.
        self._bounds = {c: (float(d[c].min()), float(d[c].max())) for c in INPUT_COLS}

        Xd = self._design_matrix(d)
        y_logD = np.log(d["Depth"].values)
        y_logW = np.log(d["Width"].values)

        self._coef_D, self._sigma_D = self._fit_log_linear(Xd, y_logD)
        self._coef_W, self._sigma_W = self._fit_log_linear(Xd, y_logW)
        return self

    def _lhs(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Latin-hypercube design over the operational envelope (4 dims)."""
        sampler = qmc.LatinHypercube(d=4, seed=rng.integers(0, 2**31 - 1))
        u = sampler.random(n)
        lo = np.array([self._bounds[c][0] for c in INPUT_COLS])
        hi = np.array([self._bounds[c][1] for c in INPUT_COLS])
        return lo + u * (hi - lo)

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        assert self._coef_D is not None and self._coef_W is not None, "fit first"
        rng = np.random.default_rng(self.seed if seed is None else seed)

        X = self._lhs(n, rng)
        X_df = pd.DataFrame(X, columns=INPUT_COLS)
        Xd = self._design_matrix(X_df)

        log_D = Xd @ self._coef_D + rng.normal(0.0, self._sigma_D, size=n)
        log_W = Xd @ self._coef_W + rng.normal(0.0, self._sigma_W, size=n)

        df = X_df.copy()
        df["Depth"] = np.exp(log_D)
        df["Width"] = np.exp(log_W)
        # Clip to the empirical output range (avoids unphysical extrapolation
        # beyond the experimental envelope).
        df["Depth"] = df["Depth"].clip(lower=0.5, upper=2.5)
        df["Width"] = df["Width"].clip(lower=1.0, upper=3.5)
        return df[EBW_COLUMNS]

    def save(self, path: Path) -> None:
        state = {
            "coef_D": self._coef_D, "coef_W": self._coef_W,
            "sigma_D": self._sigma_D, "sigma_W": self._sigma_W,
            "bounds": self._bounds, "seed": self.seed, "ridge": self.ridge,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: Path) -> "PhysicsRosenthalGenerator":
        obj = cls()
        with open(path, "rb") as f:
            st = pickle.load(f)
        obj._coef_D = st["coef_D"]; obj._coef_W = st["coef_W"]
        obj._sigma_D = st["sigma_D"]; obj._sigma_W = st["sigma_W"]
        obj._bounds = st["bounds"]; obj.seed = st["seed"]; obj.ridge = st["ridge"]
        return obj

    # Expose fitted parameters for the manuscript.
    @property
    def coefficients(self) -> pd.DataFrame:
        terms = ["intercept", "log_IW", "log_IF", "log_VW", "log_FP"]
        return pd.DataFrame({"Depth": self._coef_D, "Width": self._coef_W}, index=terms)

    @property
    def residual_sigma(self) -> dict[str, float]:
        return {"Depth": self._sigma_D, "Width": self._sigma_W}
