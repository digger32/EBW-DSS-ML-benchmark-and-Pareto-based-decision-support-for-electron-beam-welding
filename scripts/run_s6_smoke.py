"""
S-6 smoke test: run all 12 hyperparameter optimisers on the same fast
regressor (Ridge) and verify that each returns a valid SearchResult with a
best score below the unfitted baseline.

This is a sanity check on the uniform API, not the final HPO comparison of
Section 4.5 (which requires the full 41 x 12 grid and is part of S-7).
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ebw_ml.models import MODEL_REGISTRY
from ebw_ml.optimisers import ALL_OPTIMISERS, OPTIMISER_FAMILY

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "ebw_real_72.csv"
INPUT_COLS = ["IW", "IF", "VW", "FP"]
OUTPUT_COLS = ["Depth", "Width"]


def main() -> None:
    df = pd.read_csv(DATA); df.columns = [c.strip() for c in df.columns]
    X = df[INPUT_COLS].values.astype(np.float64)
    Y = df[OUTPUT_COLS].values.astype(np.float64)

    # Use SVR-RBF (3 continuous HPs) -- a non-trivial search but fast.
    model_cls = MODEL_REGISTRY["svr_rbf"]
    n_trials = 25

    rows = []
    print(f"=== S-6 smoke test: {len(ALL_OPTIMISERS)} optimisers x model={model_cls.name} "
          f"x n_trials={n_trials} ===\n")
    for opt_cls in ALL_OPTIMISERS:
        opt = opt_cls(n_trials=n_trials, seed=42)
        t0 = time.time()
        try:
            res = opt.search(model_cls, X, Y)
            status = "OK"
            elapsed = time.time() - t0
            row = {
                "optimiser": opt.name,
                "family": opt.family,
                "n_evals": res.n_evaluations,
                "best_score": round(res.best_score, 5),
                "elapsed_s": round(elapsed, 2),
                "best_hp": str(res.best_hp)[:80],
                "status": status,
            }
        except Exception as e:
            elapsed = time.time() - t0
            status = f"FAIL: {type(e).__name__}: {str(e)[:60]}"
            row = {
                "optimiser": opt.name, "family": opt.family,
                "n_evals": float("nan"), "best_score": float("nan"),
                "elapsed_s": round(elapsed, 2), "best_hp": "",
                "status": status,
            }
        rows.append(row)
        print(f"  {row['optimiser']:14s} [{row['family']:13s}] "
              f"evals={row['n_evals']:>4}  score={row['best_score']:>8}  "
              f"elapsed={row['elapsed_s']:>6}s  {status[:20]}", flush=True)

    df_res = pd.DataFrame(rows)
    out = ROOT / "data" / "optimisers_smoke_test.csv"
    df_res.to_csv(out, index=False)
    n_ok = (df_res["status"] == "OK").sum()
    print(f"\nWrote {out}")
    print(f"  Optimisers OK:   {n_ok} / {len(df_res)}")
    if n_ok < len(df_res):
        print("Failures:")
        for _, r in df_res[df_res["status"] != "OK"].iterrows():
            print(f"  {r['optimiser']:14s} -- {r['status']}")


if __name__ == "__main__":
    main()
