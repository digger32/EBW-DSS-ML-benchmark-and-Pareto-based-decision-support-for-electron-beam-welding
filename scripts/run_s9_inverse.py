#!/usr/bin/env python3
"""
run_s9_inverse.py -- S-9 inverse design for the EBW DSS.

Given a trained forward surrogate Y = f(X) where
  X = (IW, IF, VW, FP)  and  Y = (Depth, Width),
this script answers the decision-support question: "which process parameters
produce a desired weld geometry?" and "what Depth/Width trade-offs are even
achievable?".

Two products:
  1. TARGET MATCHING (primary). For each target (Depth*, Width*), NSGA-II
     searches the 4-D process space (bounded to the observed envelope, so the
     72-point surrogate is never extrapolated) minimising the two absolute
     errors |Depth-Depth*| and |Width-Width*|. The knee of the resulting front
     (min std-normalised combined error) is reported as the recommended setting,
     with the surrogate's predicted geometry and uncertainty.
  2. CAPABILITY FRONTIER (secondary). NSGA-II over the same space with
     objectives (maximise Depth, minimise Width) maps the achievable
     deep-vs-narrow trade-off curve and the parameters that realise it.

Forward surrogate: the best models on the real ablation -- 'ngb' (R^2=0.936,
native predictive distribution) by default, optionally 'mdn'. Each is tuned on
all 72 real points with a short HPO and refit before optimisation.

Run on the server (pymoo + ngboost/torch available there):
    .venv/bin/python run_s9_inverse.py --models ngb,mdn --n-gen 120 --pop 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ebw_ml.models import MODEL_REGISTRY, FitContext          # noqa: E402
from ebw_ml.optimisers import OPTIMISER_REGISTRY              # noqa: E402

IN_COLS = ["IW", "IF", "VW", "FP"]
OUT_COLS = ["Depth", "Width"]


# ---------------------------------------------------------------------------
# Forward surrogate
# ---------------------------------------------------------------------------
def fit_surrogate(model_name: str, X: np.ndarray, Y: np.ndarray,
                  optimiser: str = "random", n_trials: int = 30, seed: int = 42):
    """Tune (short HPO on the full real set) and fit a forward model on all data."""
    from sklearn.model_selection import KFold
    model_cls = MODEL_REGISTRY[model_name]
    opt = OPTIMISER_REGISTRY[optimiser](n_trials=n_trials, seed=seed)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = opt.search(model_cls, X, Y, cv=kf)
        m = model_cls(**res.best_hp)
        m.fit(X, Y, ctx=FitContext(seed=seed, device="cpu"))
    return m, res.best_hp


def predict_one(model, x: np.ndarray):
    """Return (mu(2,), std(2,)) for a single input row x(4,)."""
    xr = np.asarray(x, dtype=float).reshape(1, -1)
    mu = np.asarray(model.predict(xr)).reshape(-1)[:2]
    try:
        m, s = model.predict_dist(xr)
        std = np.asarray(s).reshape(-1)[:2]
    except (NotImplementedError, Exception):  # noqa: BLE001
        std = np.zeros(2)
    return mu, std


# ---------------------------------------------------------------------------
# pymoo problems
# ---------------------------------------------------------------------------
def _make_problems():
    # Vectorised (non-elementwise) problems: _evaluate receives the whole
    # population X of shape (pop, 4) and predicts it in ONE model.predict call.
    # The objective uses point predictions only; predict_dist (heavy for ngb)
    # is computed once afterwards, for the recommended point.
    from pymoo.core.problem import Problem

    class TargetMatch(Problem):
        def __init__(self, model, target, xl, xu):
            super().__init__(n_var=4, n_obj=2, xl=xl, xu=xu)
            self.model = model
            self.target = np.asarray(target, float)

        def _evaluate(self, X, out, *a, **k):
            P = np.asarray(self.model.predict(X))[:, :2]   # (pop, 2)
            out["F"] = np.column_stack([np.abs(P[:, 0] - self.target[0]),
                                        np.abs(P[:, 1] - self.target[1])])

    class Capability(Problem):
        # maximise Depth (=> minimise -Depth), minimise Width
        def __init__(self, model, xl, xu):
            super().__init__(n_var=4, n_obj=2, xl=xl, xu=xu)
            self.model = model

        def _evaluate(self, X, out, *a, **k):
            P = np.asarray(self.model.predict(X))[:, :2]
            out["F"] = np.column_stack([-P[:, 0], P[:, 1]])

    return TargetMatch, Capability


def run_nsga2(problem, pop_size: int, n_gen: int, seed: int = 42):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    algo = NSGA2(pop_size=pop_size)
    res = minimize(problem, algo, ("n_gen", n_gen), seed=seed, verbose=False)
    X = np.atleast_2d(res.X)
    F = np.atleast_2d(res.F)
    return X, F


# ---------------------------------------------------------------------------
def default_targets(Y: np.ndarray) -> list[tuple[float, float]]:
    d, w = Y[:, 0], Y[:, 1]
    qd = np.percentile(d, [25, 50, 75])
    qw = np.percentile(w, [25, 50, 75])
    # median, and the two practically interesting corners
    return [
        (round(qd[1], 3), round(qw[1], 3)),   # median geometry
        (round(qd[2], 3), round(qw[0], 3)),   # deep + narrow (high quality)
        (round(qd[0], 3), round(qw[2], 3)),   # shallow + wide
        (round(qd[2], 3), round(qw[2], 3)),   # deep + wide
        (round(qd[0], 3), round(qw[0], 3)),   # shallow + narrow
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="S-9 NSGA-II inverse design.")
    ap.add_argument("--real-csv", type=Path, default=ROOT / "data" / "ebw_real_72.csv")
    ap.add_argument("--models", type=str, default="ngb,mdn")
    ap.add_argument("--hpo-optimiser", type=str, default="random")
    ap.add_argument("--hpo-trials", type=int, default=30)
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--n-gen", type=int, default=120)
    ap.add_argument("--targets", type=str, default="",
                    help="d1:w1,d2:w2,...  (default: derived from data quantiles)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "s9_inverse")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.real_csv)
    df.columns = [c.strip() for c in df.columns]
    X = df[IN_COLS].values.astype(float)
    Y = df[OUT_COLS].values.astype(float)
    xl, xu = X.min(0), X.max(0)
    ystd = Y.std(0) + 1e-9
    print(f"[s9] data: {X.shape[0]} points | input bounds:")
    for c, lo, hi in zip(IN_COLS, xl, xu):
        print(f"      {c}: [{lo:.3f}, {hi:.3f}]")

    targets = ([tuple(map(float, t.split(":"))) for t in args.targets.split(",")]
               if args.targets else default_targets(Y))
    print(f"[s9] targets (Depth, Width): {targets}")

    TargetMatch, Capability = _make_problems()
    rec_rows, cap_rows = [], []

    for mname in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n[s9] === surrogate: {mname} ===")
        model, best_hp = fit_surrogate(mname, X, Y, args.hpo_optimiser,
                                       args.hpo_trials, args.seed)
        # in-sample fit quality (sanity)
        Yhat = model.predict(X)
        from sklearn.metrics import r2_score
        print(f"[s9] {mname} in-sample R2 Depth={r2_score(Y[:,0],Yhat[:,0]):.3f} "
              f"Width={r2_score(Y[:,1],Yhat[:,1]):.3f} | best_hp={best_hp}")

        # ---- target matching ----
        for (dt, wt) in targets:
            Xp, Fp = run_nsga2(TargetMatch(model, (dt, wt), xl, xu),
                               args.pop, args.n_gen, args.seed)
            # knee = min std-normalised combined error
            comb = (Fp[:, 0] / ystd[0]) ** 2 + (Fp[:, 1] / ystd[1]) ** 2
            j = int(np.argmin(comb))
            xbest = Xp[j]
            mu, std = predict_one(model, xbest)
            rec_rows.append({
                "model": mname, "target_Depth": dt, "target_Width": wt,
                **{c: round(float(v), 4) for c, v in zip(IN_COLS, xbest)},
                "pred_Depth": round(float(mu[0]), 4), "pred_Width": round(float(mu[1]), 4),
                "pred_Depth_std": round(float(std[0]), 4),
                "pred_Width_std": round(float(std[1]), 4),
                "abs_err_Depth": round(float(abs(mu[0] - dt)), 4),
                "abs_err_Width": round(float(abs(mu[1] - wt)), 4),
            })
            print(f"   target D={dt} W={wt} -> IW/IF/VW/FP="
                  f"{np.round(xbest,3).tolist()}  pred D={mu[0]:.3f} W={mu[1]:.3f}")

        # ---- capability frontier ----
        Xc, Fc = run_nsga2(Capability(model, xl, xu), args.pop, args.n_gen, args.seed)
        order = np.argsort(-Fc[:, 0])  # by depth desc (Fc[:,0] = -Depth)
        for i in order:
            cap_rows.append({
                "model": mname,
                "Depth": round(float(-Fc[i, 0]), 4), "Width": round(float(Fc[i, 1]), 4),
                **{c: round(float(v), 4) for c, v in zip(IN_COLS, Xc[i])},
            })

    rec = pd.DataFrame(rec_rows)
    cap = pd.DataFrame(cap_rows)
    rec.to_csv(args.out_dir / "s9_inverse_recommendations.csv", index=False)
    cap.to_csv(args.out_dir / "s9_capability_frontier.csv", index=False)
    print(f"\n[s9] wrote {args.out_dir/'s9_inverse_recommendations.csv'} ({len(rec)} rows)")
    print(f"[s9] wrote {args.out_dir/'s9_capability_frontier.csv'} ({len(cap)} rows)")

    # optional figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(Y[:, 0], Y[:, 1], s=18, c="0.6", label="real data (72)")
        for mname, sub in cap.groupby("model"):
            ax.plot(sub["Depth"], sub["Width"], "-o", ms=3, label=f"{mname} capability")
        for _, r in rec.iterrows():
            ax.scatter(r["target_Depth"], r["target_Width"], marker="x", c="k")
            ax.scatter(r["pred_Depth"], r["pred_Width"], marker="*", s=80)
        ax.set_xlabel("Depth [mm]"); ax.set_ylabel("Width [mm]")
        ax.legend(fontsize=7); ax.set_title("S-9 inverse design: capability & targets")
        fig.tight_layout()
        fig.savefig(args.out_dir / "s9_inverse.png", dpi=150)
        print(f"[s9] wrote {args.out_dir/'s9_inverse.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"[s9] (figure skipped: {type(e).__name__})")


if __name__ == "__main__":
    main()
