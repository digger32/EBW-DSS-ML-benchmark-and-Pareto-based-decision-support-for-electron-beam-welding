#!/usr/bin/env python3
"""
run_s11_interpret.py -- S-11 interpretability for the EBW DSS.

Global and local interpretability of the best forward surrogates on the REAL
ablation (the clean 3-fold data), for the three decision-support candidates:
  ngb  -- NGBoost   (rank 1, native predictive distribution)
  mdn  -- Mixture density network (Pareto-optimal)
  vote -- Voting ensemble (fast practical reference)

Each model is tuned by a short random search and refit on all 72 real points
(identical protocol to run_s9_inverse.py), then analysed with four
model-agnostic techniques, computed independently for each output (Depth, Width):

  1. Permutation importance  (mean R^2 drop over n_repeats shuffles).
  2. Partial-dependence (PDP) -- 1-D, grid over the observed range of each input.
  3. Accumulated local effects (ALE) -- 1-D (Apley & Zhang 2020), robust to the
     strong input collinearity of this data set (|r(IF,FP)|~0.98), where PDP can
     mislead.
  4. SHAP values via the model-agnostic KernelExplainer (background = the 72
     real points; small enough to be exact-ish at this scale).

Outputs (under --out-dir, default runs/s11_interpret):
  s11_perm_importance.csv      model,target,feature,importance,std
  s11_pdp.csv                  model,target,feature,x,pdp
  s11_ale.csv                  model,target,feature,x,ale
  s11_shap_values.csv          model,target,<one column per feature>  (per-point)
  s11_shap_importance.csv      model,target,feature,mean_abs_shap
  figures/fig18_importance.pdf, fig19_pdp.pdf, fig20_ale.pdf, fig21_shap.pdf

Run on the server (shap + ngboost/torch available there):
    .venv/bin/python run_s11_interpret.py --models ngb,mdn,vote
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ebw_ml.models import MODEL_REGISTRY, FitContext          # noqa: E402
from ebw_ml.optimisers import OPTIMISER_REGISTRY              # noqa: E402

IN_COLS = ["IW", "IF", "VW", "FP"]
OUT_COLS = ["Depth", "Width"]


# --------------------------------------------------------------------------
def fit_surrogate(model_name, X, Y, optimiser="random", n_trials=30, seed=42):
    """Tune (short HPO on the full real set) and refit on all data."""
    from sklearn.model_selection import KFold
    model_cls = MODEL_REGISTRY[model_name]
    opt = OPTIMISER_REGISTRY[optimiser](n_trials=n_trials, seed=seed)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = opt.search(model_cls, X, Y, cv=kf)
        m = model_cls(**res.best_hp)
        m.fit(X, Y, ctx=FitContext(seed=seed, device="cpu"))
    return m


def predict_target(model, X, t):
    """Point prediction for output index t (0=Depth, 1=Width)."""
    return np.asarray(model.predict(np.atleast_2d(X)))[:, t]


# --------------------------------------------------------------------------
def permutation_importance(model, X, Y, t, n_repeats=20, seed=42):
    from sklearn.metrics import r2_score
    rng = np.random.default_rng(seed)
    base = r2_score(Y[:, t], predict_target(model, X, t))
    imp = np.zeros((X.shape[1], n_repeats))
    for j in range(X.shape[1]):
        for r in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            imp[j, r] = base - r2_score(Y[:, t], predict_target(model, Xp, t))
    return imp.mean(1), imp.std(1)


def pdp_1d(model, X, t, j, n_grid=30):
    grid = np.linspace(X[:, j].min(), X[:, j].max(), n_grid)
    out = np.empty(n_grid)
    for i, v in enumerate(grid):
        Xp = X.copy(); Xp[:, j] = v
        out[i] = predict_target(model, Xp, t).mean()
    return grid, out


def ale_1d(model, X, t, j, n_bins=10):
    """1-D ALE (Apley & Zhang 2020): local differences accumulated over bins."""
    x = X[:, j]
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return np.array([x.min(), x.max()]), np.zeros(2)
    idx = np.clip(np.searchsorted(edges, x, side="left") - 1, 0, len(edges) - 2)
    local = np.zeros(len(edges) - 1)
    counts = np.zeros(len(edges) - 1)
    for b in range(len(edges) - 1):
        mask = idx == b
        if not mask.any():
            continue
        Xlo = X[mask].copy(); Xlo[:, j] = edges[b]
        Xhi = X[mask].copy(); Xhi[:, j] = edges[b + 1]
        diff = predict_target(model, Xhi, t) - predict_target(model, Xlo, t)
        local[b] = diff.mean(); counts[b] = mask.sum()
    acc = np.concatenate([[0.0], np.cumsum(local)])
    centres = 0.5 * (edges[:-1] + edges[1:])
    # centre the ALE to mean zero over the data distribution
    acc_at_x = np.interp(x, edges, acc)
    acc -= np.average(acc_at_x)
    return edges, acc


def shap_values(model, X, t, seed=42):
    import shap
    f = lambda A: predict_target(model, A, t)          # noqa: E731
    expl = shap.KernelExplainer(f, X, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sv = expl.shap_values(X, nsamples="auto", silent=True)
    return np.asarray(sv)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="S-11 interpretability.")
    ap.add_argument("--real-csv", type=Path, default=ROOT / "data" / "ebw_real_72.csv")
    ap.add_argument("--models", type=str, default="ngb,mdn,vote")
    ap.add_argument("--hpo-optimiser", type=str, default="random")
    ap.add_argument("--hpo-trials", type=int, default=30)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "s11_interpret")
    ap.add_argument("--no-shap", action="store_true", help="skip SHAP (slowest part)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.real_csv)
    df.columns = [c.strip() for c in df.columns]
    X = df[IN_COLS].values.astype(float)
    Y = df[OUT_COLS].values.astype(float)
    print(f"[s11] real data: {X.shape[0]} points, features {IN_COLS}, targets {OUT_COLS}")

    perm_rows, pdp_rows, ale_rows, shap_imp_rows, shap_val_rows = [], [], [], [], []
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    for mname in models:
        print(f"\n[s11] === surrogate: {mname} ===")
        model = fit_surrogate(mname, X, Y, args.hpo_optimiser, args.hpo_trials, args.seed)
        for t, tgt in enumerate(OUT_COLS):
            imp, istd = permutation_importance(model, X, Y, t, seed=args.seed)
            for j, feat in enumerate(IN_COLS):
                perm_rows.append({"model": mname, "target": tgt, "feature": feat,
                                  "importance": round(float(imp[j]), 5),
                                  "std": round(float(istd[j]), 5)})
                gx, gy = pdp_1d(model, X, t, j)
                for xv, yv in zip(gx, gy):
                    pdp_rows.append({"model": mname, "target": tgt, "feature": feat,
                                     "x": round(float(xv), 4), "pdp": round(float(yv), 5)})
                ex, ey = ale_1d(model, X, t, j)
                for xv, yv in zip(ex, ey):
                    ale_rows.append({"model": mname, "target": tgt, "feature": feat,
                                     "x": round(float(xv), 4), "ale": round(float(yv), 5)})
            order = np.argsort(-imp)
            print(f"   {tgt}: perm-importance "
                  + ", ".join(f"{IN_COLS[j]}={imp[j]:.3f}" for j in order))

            if not args.no_shap:
                try:
                    sv = shap_values(model, X, t, seed=args.seed)
                    mabs = np.abs(sv).mean(0)
                    for j, feat in enumerate(IN_COLS):
                        shap_imp_rows.append({"model": mname, "target": tgt,
                                              "feature": feat,
                                              "mean_abs_shap": round(float(mabs[j]), 5)})
                    for i in range(sv.shape[0]):
                        row = {"model": mname, "target": tgt, "point": i}
                        row.update({f"shap_{IN_COLS[j]}": round(float(sv[i, j]), 5)
                                    for j in range(len(IN_COLS))})
                        shap_val_rows.append(row)
                except Exception as e:                          # noqa: BLE001
                    print(f"   [shap skipped for {mname}/{tgt}: {type(e).__name__}: {e}]")

    pd.DataFrame(perm_rows).to_csv(args.out_dir / "s11_perm_importance.csv", index=False)
    pd.DataFrame(pdp_rows).to_csv(args.out_dir / "s11_pdp.csv", index=False)
    pd.DataFrame(ale_rows).to_csv(args.out_dir / "s11_ale.csv", index=False)
    if shap_imp_rows:
        pd.DataFrame(shap_imp_rows).to_csv(args.out_dir / "s11_shap_importance.csv", index=False)
        pd.DataFrame(shap_val_rows).to_csv(args.out_dir / "s11_shap_values.csv", index=False)
    print(f"\n[s11] wrote CSVs to {args.out_dir}")

    # ---- figures -----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        perm = pd.DataFrame(perm_rows)

        # fig18: permutation importance (grouped bars, per target, per model)
        fig, axes = plt.subplots(1, len(OUT_COLS), figsize=(9, 4), sharey=True)
        for ax, tgt in zip(np.atleast_1d(axes), OUT_COLS):
            sub = perm[perm.target == tgt]
            piv = sub.pivot(index="feature", columns="model", values="importance").loc[IN_COLS]
            piv.plot.bar(ax=ax, legend=(tgt == OUT_COLS[0]))
            ax.set_title(f"Permutation importance: {tgt}")
            ax.set_ylabel("mean $R^2$ drop"); ax.set_xlabel("")
            ax.axhline(0, color="0.6", lw=0.6)
        fig.tight_layout(); fig.savefig(args.out_dir / "figures" / "fig18_importance.pdf")

        # fig19: PDP grid (rows=target, cols=feature)
        pdp = pd.DataFrame(pdp_rows)
        fig, axes = plt.subplots(len(OUT_COLS), len(IN_COLS),
                                 figsize=(12, 5.5), sharex="col")
        for ti, tgt in enumerate(OUT_COLS):
            for ji, feat in enumerate(IN_COLS):
                ax = axes[ti, ji]
                for mname in models:
                    s = pdp[(pdp.model == mname) & (pdp.target == tgt) & (pdp.feature == feat)]
                    ax.plot(s["x"], s["pdp"], label=mname)
                if ti == 0:
                    ax.set_title(feat)
                if ji == 0:
                    ax.set_ylabel(f"PDP {tgt} [mm]")
                if ti == len(OUT_COLS) - 1:
                    ax.set_xlabel(feat)
        axes[0, -1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(args.out_dir / "figures" / "fig19_pdp.pdf")

        # fig20: ALE grid
        ale = pd.DataFrame(ale_rows)
        fig, axes = plt.subplots(len(OUT_COLS), len(IN_COLS),
                                 figsize=(12, 5.5), sharex="col")
        for ti, tgt in enumerate(OUT_COLS):
            for ji, feat in enumerate(IN_COLS):
                ax = axes[ti, ji]
                for mname in models:
                    s = ale[(ale.model == mname) & (ale.target == tgt) & (ale.feature == feat)]
                    ax.plot(s["x"], s["ale"], label=mname)
                ax.axhline(0, color="0.6", lw=0.5)
                if ti == 0:
                    ax.set_title(feat)
                if ji == 0:
                    ax.set_ylabel(f"ALE {tgt} [mm]")
                if ti == len(OUT_COLS) - 1:
                    ax.set_xlabel(feat)
        axes[0, -1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(args.out_dir / "figures" / "fig20_ale.pdf")

        # fig21: mean |SHAP| (if computed)
        if shap_imp_rows:
            si = pd.DataFrame(shap_imp_rows)
            fig, axes = plt.subplots(1, len(OUT_COLS), figsize=(9, 4), sharey=True)
            for ax, tgt in zip(np.atleast_1d(axes), OUT_COLS):
                piv = (si[si.target == tgt]
                       .pivot(index="feature", columns="model", values="mean_abs_shap")
                       .loc[IN_COLS])
                piv.plot.bar(ax=ax, legend=(tgt == OUT_COLS[0]))
                ax.set_title(f"mean |SHAP|: {tgt}"); ax.set_xlabel("")
            fig.tight_layout(); fig.savefig(args.out_dir / "figures" / "fig21_shap.pdf")
        print(f"[s11] wrote figures to {args.out_dir/'figures'}")
    except Exception as e:                                       # noqa: BLE001
        print(f"[s11] (figures skipped: {type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
