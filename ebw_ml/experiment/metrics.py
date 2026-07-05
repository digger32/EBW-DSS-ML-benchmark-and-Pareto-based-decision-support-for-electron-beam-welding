"""
Fifteen evaluation metrics for the EBW regression benchmark.

Classical regression scores: RMSE, MAE, MAPE, sMAPE, MASE, R^2, adj. R^2,
explained variance. Hydrological agreement indices: NSE, KGE, Willmott index
of agreement (IoA), Lin's CCC. Distributional and probabilistic scores:
Pearson r, Spearman rho, and the bootstrap 95% confidence interval of RMSE.
"""
from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
from scipy.stats import pearsonr, spearmanr


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0)


def _smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(denom, eps)) * 100.0)


def _mase(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """Mean absolute scaled error with the naive mean predictor as baseline."""
    naive = np.mean(np.abs(y_true - np.mean(y_true)))
    return float(np.mean(np.abs(y_true - y_pred)) / max(naive, eps))


def _r2(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / max(ss_tot, eps))


def _r2_adj(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    n = len(y_true)
    if n - n_features - 1 <= 0:
        return float("nan")
    r2 = _r2(y_true, y_pred)
    return float(1.0 - (1.0 - r2) * (n - 1) / (n - n_features - 1))


def _explained_variance(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    return float(1.0 - np.var(y_true - y_pred) / max(np.var(y_true), eps))


def _nse(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """Nash--Sutcliffe efficiency."""
    num = np.sum((y_true - y_pred) ** 2)
    den = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - num / max(den, eps))


def _kge(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """Kling--Gupta efficiency."""
    mu_o, mu_s = np.mean(y_true), np.mean(y_pred)
    sd_o, sd_s = np.std(y_true), np.std(y_pred)
    if sd_o < eps or sd_s < eps:
        return float("nan")
    try:
        r = pearsonr(y_true, y_pred)[0]
    except Exception:
        r = 0.0
    alpha = sd_s / sd_o
    beta = mu_s / max(abs(mu_o), eps)
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def _ioa(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """Willmott's index of agreement."""
    num = np.sum((y_true - y_pred) ** 2)
    mu = np.mean(y_true)
    den = np.sum((np.abs(y_pred - mu) + np.abs(y_true - mu)) ** 2)
    return float(1.0 - num / max(den, eps))


def _ccc(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """Lin's concordance correlation coefficient."""
    mu_t, mu_p = np.mean(y_true), np.mean(y_pred)
    var_t, var_p = np.var(y_true), np.var(y_pred)
    cov = np.mean((y_true - mu_t) * (y_pred - mu_p))
    den = var_t + var_p + (mu_t - mu_p) ** 2
    return float(2 * cov / max(den, eps))


def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r, _ = pearsonr(y_true, y_pred)
        return float(r)
    except Exception:
        return float("nan")


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r, _ = spearmanr(y_true, y_pred)
        return float(r)
    except Exception:
        return float("nan")


def _rmse_bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                       n_boot: int = 1000, alpha: float = 0.05,
                       seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b] = np.sqrt(np.mean((y_true[idx] - y_pred[idx]) ** 2))
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return lo, hi


def compute_all(y_true: np.ndarray, y_pred: np.ndarray,
                n_features: int = 4, seed: int = 42,
                bootstrap: bool = False) -> dict[str, float]:
    """Compute 15 metrics for a single (y_true, y_pred) vector pair."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    out = {
        "rmse": _rmse(y_true, y_pred),
        "mae": _mae(y_true, y_pred),
        "mape": _mape(y_true, y_pred),
        "smape": _smape(y_true, y_pred),
        "mase": _mase(y_true, y_pred),
        "r2": _r2(y_true, y_pred),
        "r2_adj": _r2_adj(y_true, y_pred, n_features),
        "exp_var": _explained_variance(y_true, y_pred),
        "nse": _nse(y_true, y_pred),
        "kge": _kge(y_true, y_pred),
        "ioa": _ioa(y_true, y_pred),
        "ccc": _ccc(y_true, y_pred),
        "pearson_r": _pearson(y_true, y_pred),
        "spearman_r": _spearman(y_true, y_pred),
    }
    if bootstrap:
        lo, hi = _rmse_bootstrap_ci(y_true, y_pred, seed=seed)
        out["rmse_ci_lo"] = lo
        out["rmse_ci_hi"] = hi
    else:
        out["rmse_ci_lo"] = float("nan")
        out["rmse_ci_hi"] = float("nan")
    return out


def compute_multi(y_true: np.ndarray, y_pred: np.ndarray,
                  target_names: Iterable[str] = ("Depth", "Width"),
                  n_features: int = 4, seed: int = 42,
                  bootstrap: bool = False) -> dict[str, float]:
    """Compute all metrics per target and a macro-average."""
    target_names = list(target_names)
    flat: dict[str, float] = {}
    per_target: dict[str, dict[str, float]] = {}
    for j, name in enumerate(target_names):
        per_target[name] = compute_all(y_true[:, j], y_pred[:, j],
                                        n_features=n_features, seed=seed,
                                        bootstrap=bootstrap)
        for k, v in per_target[name].items():
            flat[f"{name}_{k}"] = v
    # Macro-average across targets
    metric_keys = list(per_target[target_names[0]].keys())
    for k in metric_keys:
        vals = [per_target[t][k] for t in target_names if not np.isnan(per_target[t][k])]
        flat[f"macro_{k}"] = float(np.mean(vals)) if vals else float("nan")
    return flat
