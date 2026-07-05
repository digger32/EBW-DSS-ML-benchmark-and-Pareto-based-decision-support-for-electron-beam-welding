"""
Stage S-4 pipeline: generate 4 x 5 = 20 synthetic data sets, validate
distributional fidelity, and emit publication tables and convergence figures.

Outputs:
    data/synth/{generator}_n{N}.parquet      -- raw synthetic data
    data/synth/{generator}_n{N}_validation.json
    data/synth_validation_summary.csv        -- 20 rows x metrics
    data/physics_coefficients.csv            -- fitted Rosenthal coefficients
    figures/synth/fig7_convergence.pdf       -- convergence curves
    figures/synth/fig8_distvalid_canonical.pdf  -- bar chart at n=10000
    figures/synth/fig9_marginals_canonical.pdf  -- KDE overlays at n=10000
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ebw_ml.synth import (
    CTGANGenerator,
    CopulaGenerator,
    PhysicsRosenthalGenerator,
    TVAEGenerator,
)
from ebw_ml.validation import validate

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "data" / "ebw_real_72.csv"
SYNTH_DIR = ROOT / "data" / "synth"
FIG_DIR = ROOT / "figures" / "synth"
SYNTH_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZES = [1_000, 5_000, 10_000, 25_000, 50_000]
EBW_COLUMNS = ["IW", "IF", "VW", "FP", "Depth", "Width"]
SEED = 42

# Generator configurations. CTGAN and TVAE use 600 epochs to balance
# convergence and runtime; physics-informed and copula fit in milliseconds.
GENERATORS = {
    "ctgan":   lambda: CTGANGenerator(epochs=600, batch_size=50, seed=SEED),
    "tvae":    lambda: TVAEGenerator(epochs=600, batch_size=50, seed=SEED),
    "copula":  lambda: CopulaGenerator(seed=SEED),
    "physics": lambda: PhysicsRosenthalGenerator(seed=SEED),
}

DISPLAY_NAME = {
    "ctgan": "CTGAN",
    "tvae": "TVAE",
    "copula": "Gaussian copula",
    "physics": "Physics-informed",
}

# ---------------------------------------------------------------------------
# Plot styling (consistent with eda.py)
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "lines.linewidth": 1.2,
})

COLORS = {
    "ctgan":   "#0072B2",
    "tvae":    "#E69F00",
    "copula":  "#009E73",
    "physics": "#D55E00",
}


def save_figure(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Load real
# ---------------------------------------------------------------------------
def load_real() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    df.columns = [c.strip() for c in df.columns]
    for c in EBW_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=EBW_COLUMNS).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stage 1: fit + sample + validate
# ---------------------------------------------------------------------------
def generate_and_validate(real: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fitted = {}
    for gname, factory in GENERATORS.items():
        print(f"\n[{gname}] fitting on N=72 real observations...", flush=True)
        t0 = time.time()
        gen = factory().fit(real)
        fit_time = time.time() - t0
        fitted[gname] = gen
        print(f"[{gname}] fit time: {fit_time:.1f} s", flush=True)

        for n in SAMPLE_SIZES:
            t0 = time.time()
            synth = gen.sample(n=n, seed=SEED + n)
            sample_time = time.time() - t0
            out = SYNTH_DIR / f"{gname}_n{n}.csv"
            synth.to_csv(out, index=False)

            t0 = time.time()
            v = validate(real, synth, seed=SEED)
            val_time = time.time() - t0

            row = {
                "generator": gname,
                "n_synth": n,
                "fit_time_s": round(fit_time, 2),
                "sample_time_s": round(sample_time, 3),
                "validation_time_s": round(val_time, 3),
                "ks_p_mean": v["ks_p_mean"],
                "ks_p_min": v["ks_p_min"],
                "ad_p_mean": v["ad_p_mean"],
                "mmd_rbf": v["mmd_rbf"],
                "energy_distance": v["energy_distance"],
                "sliced_w2": v["sliced_wasserstein2"],
                "wasserstein_mean": v["wasserstein_mean"],
                "psi_mean": v["psi_mean"],
                "psi_max": v["psi_max"],
            }
            rows.append(row)

            # Persist per-column details as JSON for the supplementary material
            per_col = {
                "ks_per_column": v["ks_per_column"].reset_index().to_dict("records"),
                "psi_per_column": v["psi_per_column"].reset_index().to_dict("records"),
                "wasserstein_per_column": v["wasserstein_per_column"].reset_index().to_dict("records"),
            }
            with open(SYNTH_DIR / f"{gname}_n{n}_validation.json", "w") as f:
                json.dump({"summary": row, "per_column": per_col}, f, indent=2, default=str)

            print(f"  n={n:>5}: KS p_mean={v['ks_p_mean']:.3f}  "
                  f"MMD={v['mmd_rbf']:.4f}  W2={v['sliced_wasserstein2']:.3f}  "
                  f"PSI_max={v['psi_max']:.3f}", flush=True)

    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(ROOT / "data" / "synth_validation_summary.csv", index=False)

    # Persist fitted physics coefficients
    if "physics" in fitted:
        fitted["physics"].coefficients.to_csv(ROOT / "data" / "physics_coefficients.csv")

    return df_summary, fitted


# ---------------------------------------------------------------------------
# Stage 2: figures
# ---------------------------------------------------------------------------
def figure_convergence(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    metrics = [
        ("ks_p_mean",      "Mean KS $p$-value (higher is better)",     "linear"),
        ("mmd_rbf",        "MMD with RBF kernel (lower is better)",    "log"),
        ("sliced_w2",      "Sliced 2-Wasserstein (lower is better)",   "linear"),
        ("psi_max",        "Max per-column PSI (lower is better)",     "linear"),
    ]
    for ax, (metric, title, yscale) in zip(axes.flat, metrics):
        for gname in GENERATORS:
            sub = summary[summary["generator"] == gname].sort_values("n_synth")
            ax.plot(sub["n_synth"], sub[metric],
                    "o-", color=COLORS[gname], label=DISPLAY_NAME[gname],
                    markersize=5, linewidth=1.4)
        ax.set_xscale("log")
        if yscale == "log":
            ax.set_yscale("log")
        ax.set_xlabel("Number of synthetic samples $n$")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(title)
        ax.grid(True, which="both", linestyle=":", alpha=0.5, linewidth=0.5)
    axes[0, 0].axhline(0.05, color="grey", linestyle="--", linewidth=0.6, alpha=0.6,
                        label=r"$\alpha = 0.05$")
    axes[1, 1].axhline(0.10, color="grey", linestyle="--", linewidth=0.6, alpha=0.6,
                        label=r"PSI = 0.10")
    axes[1, 1].axhline(0.25, color="grey", linestyle=":",  linewidth=0.6, alpha=0.6,
                        label=r"PSI = 0.25")
    # Shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02),
               frameon=False)
    fig.suptitle("Distributional convergence of the four synthetic-data generators",
                 fontsize=11, y=1.0)
    fig.tight_layout(rect=[0, 0.04, 1, 0.99])
    save_figure(fig, "fig7_convergence")


def figure_bar_canonical(summary: pd.DataFrame) -> None:
    sub = summary[summary["n_synth"] == 10_000].copy()
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4))
    metrics = [
        ("ks_p_mean",  "Mean KS $p$-value", True),
        ("mmd_rbf",    "MMD (RBF)",         False),
        ("sliced_w2",  "Sliced $W_2$",      False),
        ("psi_max",    "Max PSI",           False),
    ]
    for ax, (m, title, higher_better) in zip(axes, metrics):
        order = sub.sort_values(m, ascending=not higher_better)
        ax.bar(range(len(order)), order[m].values,
               color=[COLORS[g] for g in order["generator"]],
               edgecolor="white")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([DISPLAY_NAME[g] for g in order["generator"]],
                           rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(title)
        ax.set_title(title)
        if m == "ks_p_mean":
            ax.axhline(0.05, color="grey", linestyle="--", linewidth=0.6, alpha=0.6)
        if m == "psi_max":
            ax.axhline(0.10, color="grey", linestyle="--", linewidth=0.6, alpha=0.6)
            ax.axhline(0.25, color="grey", linestyle=":",  linewidth=0.6, alpha=0.6)
    fig.suptitle("Distributional fidelity at the canonical synthetic-sample size $n = 10\\,000$",
                 fontsize=10, y=1.04)
    fig.tight_layout()
    save_figure(fig, "fig8_distvalid_canonical")


def figure_marginals(real: pd.DataFrame) -> None:
    """KDE overlays of real vs synthetic distribution per generator x column at n=10k."""
    import seaborn as sns
    fig, axes = plt.subplots(4, 6, figsize=(13, 9))
    for i, gname in enumerate(GENERATORS):
        synth = pd.read_csv(SYNTH_DIR / f"{gname}_n10000.csv")
        for j, col in enumerate(EBW_COLUMNS):
            ax = axes[i, j]
            sns.kdeplot(real[col], ax=ax, color="black", linewidth=1.0, label="Real")
            sns.kdeplot(synth[col], ax=ax, color=COLORS[gname], linewidth=1.0,
                        label="Synthetic", linestyle="--")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel(DISPLAY_NAME[gname], fontsize=9)
            if i == 3:
                ax.set_xlabel(col, fontsize=9)
            if i == 0:
                ax.set_title(col, fontsize=9)
    # Single legend at the top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02),
               frameon=False)
    fig.suptitle("Real vs synthetic marginal distributions (KDE, $n = 10\\,000$)",
                 fontsize=10, y=1.05)
    fig.tight_layout()
    save_figure(fig, "fig9_marginals_canonical")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("Stage S-4: synthetic-data generation and distributional validation")
    print("=" * 70)
    real = load_real()
    print(f"Loaded real data: shape={real.shape}")

    print("\n>>> Generating and validating 4 generators x 5 sample sizes = 20 data sets")
    summary, fitted = generate_and_validate(real)

    print("\n>>> Generating publication figures")
    figure_convergence(summary)
    print("  fig7_convergence.pdf saved")
    figure_bar_canonical(summary)
    print("  fig8_distvalid_canonical.pdf saved")
    figure_marginals(real)
    print("  fig9_marginals_canonical.pdf saved")

    print("\n>>> Summary table (n=10,000):")
    print(summary[summary["n_synth"] == 10_000][
        ["generator", "ks_p_mean", "mmd_rbf", "sliced_w2", "psi_mean", "psi_max"]
    ].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
