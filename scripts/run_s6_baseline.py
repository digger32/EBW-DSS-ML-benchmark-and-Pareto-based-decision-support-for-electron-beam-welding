"""
S-6 baseline HPO comparison: run all 12 optimisers on 3 representative
models (Ridge, SVR-RBF, XGBoost) for n_trials=30, and produce a baseline
figure for the manuscript's Section 4.5 placeholder.

This is a small pilot run that consumes the full HPO-grid only at S-7. The
purpose here is to show that the optimisers compare sensibly with each
other on a few model classes within a few-minutes time budget.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ebw_ml.models import MODEL_REGISTRY
from ebw_ml.optimisers import ALL_OPTIMISERS, OPTIMISER_FAMILY

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "ebw_real_72.csv"
FIG_DIR = ROOT / "figures" / "hpo"
FIG_DIR.mkdir(parents=True, exist_ok=True)

INPUT_COLS = ["IW", "IF", "VW", "FP"]
OUTPUT_COLS = ["Depth", "Width"]

MODELS_TO_TEST = ["ridge", "svr_rbf", "xgb"]
N_TRIALS = 30
SEED = 42

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 10, "axes.labelsize": 10, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8, "legend.frameon": False,
})

FAMILY_COLOR = {
    "Enumerative": "#999999",
    "Bayesian":    "#0072B2",
    "Evolutionary": "#009E73",
    "Swarm":       "#E69F00",
}


def main() -> None:
    df = pd.read_csv(DATA); df.columns = [c.strip() for c in df.columns]
    X = df[INPUT_COLS].values.astype(np.float64)
    Y = df[OUTPUT_COLS].values.astype(np.float64)

    rows = []
    for model_name in MODELS_TO_TEST:
        model_cls = MODEL_REGISTRY[model_name]
        print(f"\n=== Model: {model_name} ===")
        for opt_cls in ALL_OPTIMISERS:
            t0 = time.time()
            try:
                opt = opt_cls(n_trials=N_TRIALS, seed=SEED)
                res = opt.search(model_cls, X, Y)
                rows.append({
                    "model": model_name, "optimiser": res.name,
                    "family": opt.family,
                    "best_score": res.best_score, "n_evals": res.n_evaluations,
                    "elapsed_s": time.time() - t0,
                })
                print(f"  {res.name:14s}  score={res.best_score:.5f}  evals={res.n_evaluations:>3}  "
                      f"elapsed={time.time()-t0:6.2f}s", flush=True)
            except Exception as e:
                print(f"  {opt_cls.name:14s}  FAIL: {e}", flush=True)
                rows.append({"model": model_name, "optimiser": opt_cls.name,
                             "family": opt_cls.family,
                             "best_score": float("nan"), "n_evals": 0,
                             "elapsed_s": time.time() - t0})

    df_res = pd.DataFrame(rows)
    df_res.to_csv(ROOT / "data" / "hpo_baseline.csv", index=False)

    # Figure: per-model bar chart with optimiser families colour-coded.
    fig, axes = plt.subplots(1, len(MODELS_TO_TEST), figsize=(13, 4.0))
    if len(MODELS_TO_TEST) == 1:
        axes = [axes]
    for ax, model_name in zip(axes, MODELS_TO_TEST):
        sub = df_res[df_res["model"] == model_name].sort_values("best_score")
        colors = [FAMILY_COLOR.get(f, "grey") for f in sub["family"]]
        bars = ax.bar(range(len(sub)), sub["best_score"], color=colors, edgecolor="white")
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub["optimiser"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("CV RMSE (mean of $D$, $W$)")
        ax.set_title(f"Model: {model_name}")
        ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    # Shared legend by family
    handles = [mpl.patches.Patch(color=c, label=fam) for fam, c in FAMILY_COLOR.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"HPO baseline: {len(ALL_OPTIMISERS)} optimisers, {N_TRIALS} trials, 5-fold CV",
                 fontsize=11, y=1.01)
    fig.tight_layout(rect=[0, 0.03, 1, 0.99])
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig10_hpo_baseline.{ext}")
    plt.close(fig)
    print(f"\nWrote {FIG_DIR / 'fig10_hpo_baseline.pdf'}")


if __name__ == "__main__":
    main()
