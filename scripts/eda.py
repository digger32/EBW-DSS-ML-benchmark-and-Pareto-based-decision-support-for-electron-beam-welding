"""
Exploratory data analysis for the EBW real data set (72 observations).

Generates publication-quality figures and descriptive statistics for the
manuscript "Multi-output machine-learning benchmark and Pareto-based decision
support for electron-beam welding of thin-walled titanium structures".

Outputs are written to:
    paper_v1/figures/eda/   (PDF + PNG, 600 dpi)
    paper_v1/data/eda_summary.csv

Reproducibility: fixed random_state = 42 throughout.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from scipy import stats as sps
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "data" / "ebw_real_72.csv"
FIG_DIR = ROOT / "figures" / "eda"
FIG_DIR.mkdir(parents=True, exist_ok=True)

INPUT_COLS = ["IW", "IF", "VW", "FP"]
OUTPUT_COLS = ["Depth", "Width"]
INPUT_UNITS = {
    "IW": r"$I_{\mathrm{W}}$ (mA)",
    "IF": r"$I_{\mathrm{F}}$ (mA)",
    "VW": r"$V_{\mathrm{W}}$ (mm s$^{-1}$)",
    "FP": r"$F_{\mathrm{P}}$ (mm)",
}
OUTPUT_UNITS = {
    "Depth": r"$D$ (mm)",
    "Width": r"$W$ (mm)",
}
ALL_LABELS = {**INPUT_UNITS, **OUTPUT_UNITS}

SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Matplotlib styling: publication-grade, colour-blind safe (Okabe--Ito)
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
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": 9,
    "legend.frameon": False,
    "lines.linewidth": 1.2,
    "lines.markersize": 4.5,
})

# Okabe--Ito colour-blind-safe palette
COLORS = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "lblue":  "#56B4E9",
    "black":  "#000000",
}


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save a figure in both PDF (vector) and PNG (600 dpi)."""
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    df.columns = [c.strip() for c in df.columns]
    # types
    for c in INPUT_COLS + OUTPUT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=INPUT_COLS + OUTPUT_COLS).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Figure 1: descriptive statistics table + boxplots
# ---------------------------------------------------------------------------
def figure_descriptive(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 6, figsize=(11, 2.6))
    for ax, col in zip(axes, INPUT_COLS + OUTPUT_COLS):
        ax.boxplot(
            df[col],
            showfliers=True,
            flierprops=dict(marker="o", markersize=3, markerfacecolor=COLORS["red"],
                            markeredgecolor=COLORS["red"], alpha=0.6),
            boxprops=dict(color=COLORS["blue"], linewidth=1.0),
            whiskerprops=dict(color=COLORS["blue"], linewidth=0.8),
            medianprops=dict(color=COLORS["orange"], linewidth=1.2),
            capprops=dict(color=COLORS["blue"], linewidth=0.8),
            widths=0.5,
        )
        ax.set_title(ALL_LABELS[col])
        ax.set_xticks([])
    fig.suptitle("Distribution of EBW process parameters and bead-geometry responses (N = 72)",
                 fontsize=10, y=1.02)
    save_figure(fig, "fig1_descriptive_boxplots")


# ---------------------------------------------------------------------------
# Figure 2: pair plot (inputs + outputs) with diagonal histograms
# ---------------------------------------------------------------------------
def figure_pairplot(df: pd.DataFrame) -> None:
    sub = df[INPUT_COLS + OUTPUT_COLS].copy()
    sub.columns = [
        r"$I_{\mathrm{W}}$", r"$I_{\mathrm{F}}$", r"$V_{\mathrm{W}}$",
        r"$F_{\mathrm{P}}$", r"$D$", r"$W$"
    ]
    g = sns.PairGrid(sub, diag_sharey=False, height=1.4)
    g.map_diag(sns.histplot, kde=True, color=COLORS["blue"], edgecolor="white", bins=12)
    g.map_offdiag(sns.scatterplot, s=12, color=COLORS["blue"], alpha=0.7, edgecolor="none")
    for ax in g.axes.flat:
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_size(9)
        ax.yaxis.label.set_size(9)
    g.fig.suptitle("Pair plot of EBW inputs and weld-bead outputs (N = 72)",
                   fontsize=10, y=1.02)
    g.fig.set_size_inches(8.4, 8.4)
    save_figure(g.fig, "fig2_pairplot")


# ---------------------------------------------------------------------------
# Figure 3: correlation heatmaps (Pearson + Spearman)
# ---------------------------------------------------------------------------
def figure_correlation(df: pd.DataFrame) -> None:
    cols = INPUT_COLS + OUTPUT_COLS
    pretty = [r"$I_{\mathrm{W}}$", r"$I_{\mathrm{F}}$", r"$V_{\mathrm{W}}$",
              r"$F_{\mathrm{P}}$", r"$D$", r"$W$"]
    pearson = df[cols].corr(method="pearson")
    spearman = df[cols].corr(method="spearman")
    pearson.index = pearson.columns = pretty
    spearman.index = spearman.columns = pretty

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    for ax, mat, title in zip(axes, [pearson, spearman], ["Pearson", "Spearman"]):
        sns.heatmap(mat, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
                    square=True, linewidths=0.5, linecolor="white",
                    cbar_kws={"shrink": 0.7}, ax=ax, annot_kws={"size": 8})
        ax.set_title(f"{title} correlation")
        ax.tick_params(labelsize=8)
    fig.suptitle("Correlation structure of the EBW data set (N = 72)",
                 fontsize=10, y=1.02)
    save_figure(fig, "fig3_correlation")
    return pearson, spearman


# ---------------------------------------------------------------------------
# Figure 4: PCA projection
# ---------------------------------------------------------------------------
def figure_pca(df: pd.DataFrame) -> None:
    X = df[INPUT_COLS].values
    X_std = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=SEED)
    Z = pca.fit_transform(X_std)
    ev = pca.explained_variance_ratio_ * 100

    # Jitter to expose overlapping replicates (factorial design)
    rng = np.random.default_rng(SEED)
    jitter = rng.normal(0, 0.06, size=Z.shape)
    Zj = Z + jitter

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))

    sc1 = axes[0].scatter(Zj[:, 0], Zj[:, 1], c=df["Depth"], cmap="viridis",
                          s=34, alpha=0.85, edgecolor="white", linewidth=0.4)
    cb1 = fig.colorbar(sc1, ax=axes[0], shrink=0.85, pad=0.02)
    cb1.set_label(r"$D$ (mm)", fontsize=9)
    axes[0].set_xlabel(f"PC1 ({ev[0]:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    axes[0].set_title("PCA coloured by penetration depth $D$")
    axes[0].axhline(0, color="grey", linewidth=0.5, alpha=0.5)
    axes[0].axvline(0, color="grey", linewidth=0.5, alpha=0.5)

    sc2 = axes[1].scatter(Zj[:, 0], Zj[:, 1], c=df["Width"], cmap="plasma",
                          s=34, alpha=0.85, edgecolor="white", linewidth=0.4)
    cb2 = fig.colorbar(sc2, ax=axes[1], shrink=0.85, pad=0.02)
    cb2.set_label(r"$W$ (mm)", fontsize=9)
    axes[1].set_xlabel(f"PC1 ({ev[0]:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    axes[1].set_title("PCA coloured by bead width $W$")
    axes[1].axhline(0, color="grey", linewidth=0.5, alpha=0.5)
    axes[1].axvline(0, color="grey", linewidth=0.5, alpha=0.5)

    fig.suptitle("Principal-component projection of the EBW input space (jitter $\\sigma = 0.06$ for readability)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig4_pca")

    loadings = pd.DataFrame(
        pca.components_.T,
        index=INPUT_COLS,
        columns=["PC1", "PC2"]
    )
    print("PCA explained variance (%):", ev.round(2))
    print("PCA loadings:\n", loadings.round(3))
    return ev, loadings


# ---------------------------------------------------------------------------
# Figure 5: UMAP projection (4D inputs --> 2D)
# ---------------------------------------------------------------------------
def figure_umap(df: pd.DataFrame) -> None:
    X = StandardScaler().fit_transform(df[INPUT_COLS].values)
    reducer = umap.UMAP(
        n_components=2, n_neighbors=10, min_dist=0.3,
        random_state=SEED, metric="euclidean"
    )
    Z = reducer.fit_transform(X)
    rng = np.random.default_rng(SEED)
    jitter = rng.normal(0, 0.10, size=Z.shape)
    Zj = Z + jitter

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    sc1 = axes[0].scatter(Zj[:, 0], Zj[:, 1], c=df["Depth"], cmap="viridis",
                          s=34, alpha=0.85, edgecolor="white", linewidth=0.4)
    cb1 = fig.colorbar(sc1, ax=axes[0], shrink=0.85, pad=0.02)
    cb1.set_label(r"$D$ (mm)", fontsize=9)
    axes[0].set_xlabel("UMAP-1")
    axes[0].set_ylabel("UMAP-2")
    axes[0].set_title("UMAP coloured by penetration depth $D$")

    sc2 = axes[1].scatter(Zj[:, 0], Zj[:, 1], c=df["Width"], cmap="plasma",
                          s=34, alpha=0.85, edgecolor="white", linewidth=0.4)
    cb2 = fig.colorbar(sc2, ax=axes[1], shrink=0.85, pad=0.02)
    cb2.set_label(r"$W$ (mm)", fontsize=9)
    axes[1].set_xlabel("UMAP-1")
    axes[1].set_ylabel("UMAP-2")
    axes[1].set_title("UMAP coloured by bead width $W$")

    fig.suptitle("Uniform-manifold projection of the EBW input space (jitter $\\sigma = 0.10$ for readability)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig5_umap")


# ---------------------------------------------------------------------------
# Figure 6: input--output marginal scatter with LOWESS
# ---------------------------------------------------------------------------
def figure_marginal_scatter(df: pd.DataFrame) -> None:
    from statsmodels.nonparametric.smoothers_lowess import lowess

    fig, axes = plt.subplots(2, 4, figsize=(11, 5.2))
    for j, x_col in enumerate(INPUT_COLS):
        for i, y_col in enumerate(OUTPUT_COLS):
            ax = axes[i, j]
            x = df[x_col].values
            y = df[y_col].values
            ax.scatter(x, y, s=18, color=COLORS["blue"], alpha=0.7, edgecolor="none")
            # LOWESS overlay
            try:
                sm = lowess(y, x, frac=0.6, return_sorted=True)
                ax.plot(sm[:, 0], sm[:, 1], color=COLORS["red"], linewidth=1.4, alpha=0.85)
            except Exception:
                pass
            # Pearson r
            r, _ = sps.pearsonr(x, y)
            ax.text(0.04, 0.92, f"$r = {r:.2f}$", transform=ax.transAxes,
                    fontsize=8, color="black")
            if i == 1:
                ax.set_xlabel(INPUT_UNITS[x_col])
            if j == 0:
                ax.set_ylabel(OUTPUT_UNITS[y_col])
            ax.tick_params(labelsize=8)
    fig.suptitle("Marginal input--output relationships with LOWESS smoother (N = 72)",
                 fontsize=10, y=1.00)
    fig.tight_layout()
    save_figure(fig, "fig6_marginal_scatter")


# ---------------------------------------------------------------------------
# Descriptive statistics table
# ---------------------------------------------------------------------------
def descriptive_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = INPUT_COLS + OUTPUT_COLS
    stats = pd.DataFrame({
        "Mean":   df[cols].mean(),
        "SD":     df[cols].std(),
        "Min":    df[cols].min(),
        "Q1":     df[cols].quantile(0.25),
        "Median": df[cols].median(),
        "Q3":     df[cols].quantile(0.75),
        "Max":    df[cols].max(),
        "Skew":   df[cols].apply(sps.skew),
        "Kurt":   df[cols].apply(sps.kurtosis),
    }).round(3)
    stats.index = cols
    return stats


def bootstrap_median_ci(x: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    samples = rng.choice(x, size=(n_boot, len(x)), replace=True)
    meds = np.median(samples, axis=1)
    lo = float(np.percentile(meds, 100 * alpha / 2))
    hi = float(np.percentile(meds, 100 * (1 - alpha / 2)))
    return lo, hi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("EDA for the EBW real data set (Tynchenko et al. 2021, 72 obs.)")
    print("=" * 70)

    df = load_data()
    print(f"Loaded {len(df)} observations, columns: {list(df.columns)}")

    print("\n--- Descriptive statistics ---")
    stats = descriptive_table(df)
    print(stats.to_string())
    stats.to_csv(ROOT / "data" / "eda_descriptive.csv")

    print("\n--- Bootstrap 95% CI for medians (10,000 resamples) ---")
    boot_ci = {}
    for col in INPUT_COLS + OUTPUT_COLS:
        lo, hi = bootstrap_median_ci(df[col].values)
        boot_ci[col] = (lo, hi)
        print(f"  {col:6s}: median = {df[col].median():.3f}  CI95 = [{lo:.3f}, {hi:.3f}]")
    pd.DataFrame(boot_ci, index=["CI95_lo", "CI95_hi"]).T.to_csv(ROOT / "data" / "eda_bootstrap_median_ci.csv")

    print("\n--- Generating figures ---")
    figure_descriptive(df)
    print("  fig1_descriptive_boxplots.pdf/png saved")
    figure_pairplot(df)
    print("  fig2_pairplot.pdf/png saved")
    pearson, spearman = figure_correlation(df)
    print("  fig3_correlation.pdf/png saved")
    pearson.to_csv(ROOT / "data" / "eda_pearson.csv")
    spearman.to_csv(ROOT / "data" / "eda_spearman.csv")
    ev, loadings = figure_pca(df)
    print("  fig4_pca.pdf/png saved")
    loadings.to_csv(ROOT / "data" / "eda_pca_loadings.csv")
    pd.Series(ev, index=["PC1", "PC2"]).to_csv(ROOT / "data" / "eda_pca_explained_variance.csv")
    figure_umap(df)
    print("  fig5_umap.pdf/png saved")
    figure_marginal_scatter(df)
    print("  fig6_marginal_scatter.pdf/png saved")

    print("\nDone. All artefacts in:")
    print(f"  Figures: {FIG_DIR}")
    print(f"  Tables:  {ROOT / 'data'}")


if __name__ == "__main__":
    main()
