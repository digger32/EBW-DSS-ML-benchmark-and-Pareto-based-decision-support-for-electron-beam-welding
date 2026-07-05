"""
Distributional validation of synthetic data vs real data.

Implements the four-layer validation protocol from Section 3.3 of the
manuscript:

1. Univariate goodness-of-fit:
   - Two-sample Kolmogorov-Smirnov (Kolmogorov 1933)
   - Anderson-Darling k-sample (Anderson and Darling 1952)
   per column, with Holm-Bonferroni correction.

2. Multivariate equivalence:
   - Maximum mean discrepancy with RBF kernel (Gretton et al. 2012)
   - Energy distance (Szekely and Rizzo)
   - 2-Wasserstein distance via the sliced approximation (Villani 2009)
   on standardised features.

3. Population stability index per column.

4. (Computed externally in the downstream-utility step.)
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler

EBW_COLUMNS = ["IW", "IF", "VW", "FP", "Depth", "Width"]


# ---------------------------------------------------------------------------
# Univariate goodness-of-fit
# ---------------------------------------------------------------------------
def ks_per_column(real: pd.DataFrame, synth: pd.DataFrame,
                  columns: Iterable[str] = EBW_COLUMNS) -> pd.DataFrame:
    """Two-sample Kolmogorov-Smirnov test per column."""
    rows = []
    for c in columns:
        stat, p = sps.ks_2samp(real[c].values, synth[c].values)
        rows.append({"column": c, "ks_statistic": float(stat), "ks_pvalue": float(p)})
    return pd.DataFrame(rows).set_index("column")


def ad_per_column(real: pd.DataFrame, synth: pd.DataFrame,
                  columns: Iterable[str] = EBW_COLUMNS) -> pd.DataFrame:
    """k-sample Anderson-Darling test per column."""
    rows = []
    for c in columns:
        try:
            res = sps.anderson_ksamp([real[c].values, synth[c].values])
            stat = float(res.statistic)
            # significance_level is returned in percent (0.001..0.25); convert to fraction
            p = float(res.significance_level) / 100.0
        except Exception as e:
            stat, p = np.nan, np.nan
        rows.append({"column": c, "ad_statistic": stat, "ad_pvalue": p})
    return pd.DataFrame(rows).set_index("column")


def holm_bonferroni(pvals: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
    """Holm-Bonferroni step-down correction over a vector of p-values."""
    n = len(pvals)
    order = pvals.argsort().values
    sorted_p = pvals.values[order]
    rejected = np.zeros(n, dtype=bool)
    adj_p = np.zeros(n)
    running_max = 0.0
    for k in range(n):
        adj = sorted_p[k] * (n - k)
        running_max = max(running_max, adj)
        adj_p[k] = min(running_max, 1.0)
        rejected[k] = adj_p[k] < alpha
    # restore original order
    out = pd.DataFrame(index=pvals.index)
    out["p_raw"] = pvals.values
    out["p_holm"] = np.nan
    out["rejected@0.05"] = False
    for k, idx in enumerate(order):
        out.iloc[idx, out.columns.get_loc("p_holm")] = adj_p[k]
        out.iloc[idx, out.columns.get_loc("rejected@0.05")] = rejected[k]
    return out


# ---------------------------------------------------------------------------
# Multivariate equivalence
# ---------------------------------------------------------------------------
def _subsample(X: np.ndarray, n_max: int, seed: int) -> np.ndarray:
    """Random sub-sample without replacement; identity if already small enough."""
    if X.shape[0] <= n_max:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=n_max, replace=False)
    return X[idx]


def mmd_rbf(real: pd.DataFrame, synth: pd.DataFrame,
            gamma: float | None = None,
            columns: Iterable[str] = EBW_COLUMNS,
            n_max_synth: int = 1000,
            seed: int = 42) -> float:
    """Squared maximum mean discrepancy with RBF kernel (biased estimate).

    Gretton et al. (2012) JMLR. ``gamma`` defaults to the median heuristic.
    Features are standardised before kernel computation. Synthetic data are
    randomly sub-sampled to ``n_max_synth`` rows to keep the kernel matrix
    in memory.
    """
    sc = StandardScaler().fit(real[list(columns)].values)
    Xr = sc.transform(real[list(columns)].values)
    Xs = sc.transform(synth[list(columns)].values)
    Xs = _subsample(Xs, n_max_synth, seed)
    if gamma is None:
        d2 = cdist(np.vstack([Xr, Xs]), np.vstack([Xr, Xs]), metric="sqeuclidean")
        med = np.median(d2[d2 > 0])
        gamma = 1.0 / med if med > 0 else 1.0
    Krr = np.exp(-gamma * cdist(Xr, Xr, metric="sqeuclidean"))
    Kss = np.exp(-gamma * cdist(Xs, Xs, metric="sqeuclidean"))
    Krs = np.exp(-gamma * cdist(Xr, Xs, metric="sqeuclidean"))
    mmd2 = Krr.mean() + Kss.mean() - 2.0 * Krs.mean()
    return float(max(mmd2, 0.0))


def energy_distance(real: pd.DataFrame, synth: pd.DataFrame,
                    columns: Iterable[str] = EBW_COLUMNS,
                    n_max_synth: int = 1000,
                    seed: int = 42) -> float:
    """Energy distance between two empirical distributions (standardised)."""
    sc = StandardScaler().fit(real[list(columns)].values)
    Xr = sc.transform(real[list(columns)].values)
    Xs = sc.transform(synth[list(columns)].values)
    Xs = _subsample(Xs, n_max_synth, seed)
    d_xy = cdist(Xr, Xs).mean()
    d_xx = cdist(Xr, Xr).mean()
    d_yy = cdist(Xs, Xs).mean()
    return float(2.0 * d_xy - d_xx - d_yy)


def wasserstein_per_column(real: pd.DataFrame, synth: pd.DataFrame,
                            columns: Iterable[str] = EBW_COLUMNS) -> pd.DataFrame:
    """1-Wasserstein distance per column (on the original scale)."""
    rows = []
    for c in columns:
        w = wasserstein_distance(real[c].values, synth[c].values)
        rows.append({"column": c, "wasserstein": float(w)})
    return pd.DataFrame(rows).set_index("column")


def sliced_wasserstein2(real: pd.DataFrame, synth: pd.DataFrame,
                         n_projections: int = 200, seed: int = 42,
                         columns: Iterable[str] = EBW_COLUMNS) -> float:
    """Sliced 2-Wasserstein distance approximation in standardised space."""
    rng = np.random.default_rng(seed)
    sc = StandardScaler().fit(real[list(columns)].values)
    Xr = sc.transform(real[list(columns)].values)
    Xs = sc.transform(synth[list(columns)].values)
    d = Xr.shape[1]
    accum = 0.0
    for _ in range(n_projections):
        u = rng.normal(size=d)
        u /= np.linalg.norm(u) + 1e-12
        pr = np.sort(Xr @ u)
        ps = np.sort(Xs @ u)
        # match lengths via quantile interpolation
        q = np.linspace(0, 1, max(len(pr), len(ps)))
        prq = np.interp(q, np.linspace(0, 1, len(pr)), pr)
        psq = np.interp(q, np.linspace(0, 1, len(ps)), ps)
        accum += np.mean((prq - psq) ** 2)
    return float(np.sqrt(accum / n_projections))


# ---------------------------------------------------------------------------
# Population stability index
# ---------------------------------------------------------------------------
def psi_per_column(real: pd.DataFrame, synth: pd.DataFrame,
                   columns: Iterable[str] = EBW_COLUMNS, bins: int = 10,
                   eps: float = 1e-6) -> pd.DataFrame:
    """Population stability index per column.

    PSI = sum_i (p_real_i - p_synth_i) * ln(p_real_i / p_synth_i)
    Conventional thresholds: <= 0.10 (no shift), <= 0.25 (acceptable),
    > 0.25 (significant shift).

    Equal-width bins on the union range are used because the EBW real data
    exhibit a discrete factorial structure (only a handful of distinct levels
    per input), which makes quantile binning numerically unstable.
    """
    rows = []
    for c in columns:
        all_vals = np.concatenate([real[c].values, synth[c].values])
        lo, hi = float(all_vals.min()), float(all_vals.max())
        if hi - lo < 1e-12:
            rows.append({"column": c, "psi": 0.0})
            continue
        edges = np.linspace(lo - 1e-9, hi + 1e-9, bins + 1)
        p_r, _ = np.histogram(real[c].values, bins=edges)
        p_s, _ = np.histogram(synth[c].values, bins=edges)
        p_r = p_r / max(p_r.sum(), 1)
        p_s = p_s / max(p_s.sum(), 1)
        p_r = np.clip(p_r, eps, None)
        p_s = np.clip(p_s, eps, None)
        psi = float(np.sum((p_r - p_s) * np.log(p_r / p_s)))
        rows.append({"column": c, "psi": psi})
    return pd.DataFrame(rows).set_index("column")


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------
def validate(real: pd.DataFrame, synth: pd.DataFrame,
             columns: Iterable[str] = EBW_COLUMNS,
             seed: int = 42) -> dict:
    """Return a dict with all distributional metrics for one (real, synth) pair."""
    cols = list(columns)
    ks = ks_per_column(real, synth, cols)
    ad = ad_per_column(real, synth, cols)
    ks_holm = holm_bonferroni(ks["ks_pvalue"])
    wd = wasserstein_per_column(real, synth, cols)
    psi = psi_per_column(real, synth, cols)
    return {
        "ks_per_column": ks,
        "ad_per_column": ad,
        "ks_holm": ks_holm,
        "wasserstein_per_column": wd,
        "psi_per_column": psi,
        "ks_p_mean": float(ks["ks_pvalue"].mean()),
        "ks_p_min": float(ks["ks_pvalue"].min()),
        "ad_p_mean": float(ad["ad_pvalue"].mean()),
        "wasserstein_mean": float(wd["wasserstein"].mean()),
        "psi_mean": float(psi["psi"].mean()),
        "psi_max": float(psi["psi"].max()),
        "mmd_rbf": mmd_rbf(real, synth, columns=cols),
        "energy_distance": energy_distance(real, synth, columns=cols),
        "sliced_wasserstein2": sliced_wasserstein2(real, synth, seed=seed, columns=cols),
    }
