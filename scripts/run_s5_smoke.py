"""
S-5 smoke test: fit + predict on N=72 real data for all 28 models.

Records for each model: family, fit time, prediction time, train RMSE on
(Depth, Width), and a basic in-sample R^2 sanity check. This is NOT the final
benchmark of Section 4.4 -- that requires the full 10-fold CV across all
ablation modes (S-7). The purpose here is only to verify the uniform model
API and capture order-of-magnitude reference numbers.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ebw_ml.models import ALL_MODELS, FAMILY_OF

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "ebw_real_72.csv"
INPUT_COLS = ["IW", "IF", "VW", "FP"]
OUTPUT_COLS = ["Depth", "Width"]


def main() -> None:
    df = pd.read_csv(DATA)
    df.columns = [c.strip() for c in df.columns]
    X = df[INPUT_COLS].values.astype(np.float64)
    Y = df[OUTPUT_COLS].values.astype(np.float64)

    rows = []
    print(f"=== S-5 smoke test on N={X.shape[0]}, {len(ALL_MODELS)} models ===\n")
    for cls in ALL_MODELS:
        m = cls()
        t0 = time.time()
        try:
            m.fit(X, Y)
            t_fit = time.time() - t0
            t0 = time.time()
            Yhat = m.predict(X)
            t_pred = time.time() - t0
            rmse_d = float(np.sqrt(np.mean((Yhat[:, 0] - Y[:, 0]) ** 2)))
            rmse_w = float(np.sqrt(np.mean((Yhat[:, 1] - Y[:, 1]) ** 2)))
            # in-sample R^2 (not a generalisation metric; only a sanity check)
            ss_res_d = float(np.sum((Yhat[:, 0] - Y[:, 0]) ** 2))
            ss_tot_d = float(np.sum((Y[:, 0] - Y[:, 0].mean()) ** 2))
            r2_d = 1.0 - ss_res_d / ss_tot_d
            status = "OK"
        except Exception as e:
            t_fit = t_pred = rmse_d = rmse_w = r2_d = float("nan")
            status = f"FAIL: {type(e).__name__}: {str(e)[:60]}"
        rows.append({
            "name": cls.name, "family": cls.family,
            "fit_s": round(t_fit, 3) if isinstance(t_fit, float) else t_fit,
            "pred_s": round(t_pred, 4) if isinstance(t_pred, float) else t_pred,
            "rmse_D": round(rmse_d, 4) if not np.isnan(rmse_d) else rmse_d,
            "rmse_W": round(rmse_w, 4) if not np.isnan(rmse_w) else rmse_w,
            "r2_D_insample": round(r2_d, 3) if not np.isnan(r2_d) else r2_d,
            "status": status,
        })
        line_status = "OK" if status == "OK" else "FAIL"
        print(f"  {cls.name:12s} [{cls.family:10s}] fit={t_fit:6.2f}s "
              f"rmse_D={rmse_d:.4f} rmse_W={rmse_w:.4f}  {line_status}", flush=True)

    df_res = pd.DataFrame(rows)
    out = ROOT / "data" / "models_smoke_test.csv"
    df_res.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print(f"  Models OK:   {(df_res['status'] == 'OK').sum()} / {len(df_res)}")
    print(f"  Models FAIL: {(df_res['status'] != 'OK').sum()}")
    if (df_res['status'] != 'OK').any():
        print("\nFailures:")
        for _, r in df_res[df_res['status'] != 'OK'].iterrows():
            print(f"  {r['name']:12s} -- {r['status']}")


if __name__ == "__main__":
    main()
