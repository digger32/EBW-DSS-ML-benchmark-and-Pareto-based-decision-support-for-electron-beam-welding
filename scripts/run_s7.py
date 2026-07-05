#!/usr/bin/env python3
"""
S-7 full experimental campaign runner.

Usage examples
--------------
# Full grid with all 41 models, 12 optimisers, 4 generators, 5 sample sizes,
# 4 ablation modes, 5-fold CV, 30 HPO trials per (model, optimiser):
python run_s7.py --out-dir runs/s7_full

# Quick smoke test (mini-grid):
python run_s7.py --out-dir runs/s7_smoke \
    --models ridge,svr_rbf,rfr \
    --optimisers random,tpe \
    --generators copula \
    --sample-sizes 1000 \
    --ablation-modes real,synth \
    --n-hpo-trials 8 \
    --n-folds 3

# Force CPU (recommended for tiny-n NN models; see GPU_NOTES.md):
EBW_DEVICE=cpu python run_s7.py --out-dir runs/s7_cpu --device cpu

# Resume an interrupted run (relies on done.csv):
python run_s7.py --out-dir runs/s7_full --resume

# Shard the grid across processes (round-robin grid[shard::num_shards]):
python run_s7.py --out-dir runs/s7_sh0 --num-shards 12 --shard 0 &
python run_s7.py --out-dir runs/s7_sh1 --num-shards 12 --shard 1 &
# ... then: python merge_runs.py --inputs runs/s7_sh* --output runs/s7_merged
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Resolve project root so this script works from any cwd
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ebw_ml.experiment import S7Config, run_experiment  # noqa: E402
from ebw_ml.models import MODEL_REGISTRY  # noqa: E402
from ebw_ml.optimisers import OPTIMISER_REGISTRY  # noqa: E402

ABLATION_CHOICES = ["real", "synth", "real_plus_synth", "tstr"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="S-7 full experimental campaign for the EBW DSS paper.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # IO
    p.add_argument("--real-csv", type=Path,
                   default=ROOT / "data" / "ebw_real_72.csv",
                   help="Path to real data CSV (72 rows).")
    p.add_argument("--synth-dir", type=Path,
                   default=ROOT / "data" / "synth",
                   help="Directory containing {gen}_n{N}.csv synthetic files.")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output directory for results.csv, results_aggregated.csv, done.csv.")
    # Grid
    p.add_argument("--models", type=str, default="",
                   help=f"Comma-separated subset; default = all 41. "
                        f"Available: {','.join(sorted(MODEL_REGISTRY))}")
    p.add_argument("--optimisers", type=str, default="",
                   help=f"Comma-separated subset; default = all 12. "
                        f"Available: {','.join(sorted(OPTIMISER_REGISTRY))}")
    p.add_argument("--generators", type=str, default="ctgan,tvae,copula,physics",
                   help="Comma-separated subset of synthetic generators.")
    p.add_argument("--sample-sizes", type=str, default="1000,5000,10000,25000,50000",
                   help="Comma-separated synthetic sample sizes.")
    p.add_argument("--ablation-modes", type=str,
                   default=",".join(ABLATION_CHOICES),
                   help=f"Comma-separated subset of {ABLATION_CHOICES}")
    # Compute
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-hpo-trials", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda", "cuda:0", "cuda:1"],
                   help="Device for torch and GPU-aware models. NOTE: the "
                        "EBW_DEVICE environment variable, if set, overrides this "
                        "everywhere (including the HPO inner loop).")
    p.add_argument("--max-train-size", type=int, default=None,
                   help="Cap the train-set size passed to any model. Useful "
                        "when synthetic n is large (10000+) and slow "
                        "regressors (GPR, kernel ridge, TheilSen, stacking) "
                        "would dominate the runtime. Recommended: 1000-2000.")
    p.add_argument("--per-config-timeout", type=float, default=None,
                   help="HARD per-configuration wall-clock budget in seconds "
                        "(SIGALRM). When exceeded, partial results are kept, the "
                        "row is flagged status=timeout, and the campaign moves "
                        "on. Recommended: 900-1200.")
    # Sharding (multi-process parallelism)
    p.add_argument("--num-shards", type=int, default=1,
                   help="Total number of shards. The grid is sliced "
                        "round-robin as grid[shard::num_shards].")
    p.add_argument("--shard", type=int, default=0,
                   help="This process's shard index in [0, num_shards).")
    # MLflow
    p.add_argument("--mlflow", action="store_true",
                   help="Enable MLflow tracking.")
    p.add_argument("--mlflow-uri", type=str, default=None,
                   help="MLflow tracking URI (e.g. file:./mlruns or http://host:5000).")
    p.add_argument("--mlflow-experiment", type=str, default="ebw_s7",
                   help="MLflow experiment name.")
    # Resume / dry-run
    p.add_argument("--resume", action="store_true",
                   help="Skip configurations already listed in done.csv (default behaviour).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the grid size and exit without running.")
    p.add_argument("--progress-every", type=int, default=5,
                   help="Print progress every N configurations.")
    return p.parse_args()


def _split_list(s: str, conv=str) -> list:
    return [conv(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    if not (0 <= args.shard < max(1, args.num_shards)):
        raise SystemExit(f"--shard {args.shard} out of range for "
                         f"--num-shards {args.num_shards}")
    cfg = S7Config(
        real_csv=args.real_csv,
        synth_dir=args.synth_dir,
        out_dir=args.out_dir,
        models=_split_list(args.models) if args.models else [],
        optimisers=_split_list(args.optimisers) if args.optimisers else [],
        generators=_split_list(args.generators),
        sample_sizes=_split_list(args.sample_sizes, int),
        ablation_modes=_split_list(args.ablation_modes),
        n_folds=args.n_folds,
        n_hpo_trials=args.n_hpo_trials,
        seed=args.seed,
        device=args.device,
        mlflow_enabled=args.mlflow,
        mlflow_tracking_uri=args.mlflow_uri,
        mlflow_experiment=args.mlflow_experiment,
        max_train_size=args.max_train_size,
        per_config_timeout_s=args.per_config_timeout,
        shard=args.shard,
        num_shards=args.num_shards,
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    # Persist configuration for reproducibility
    with open(cfg.out_dir / "config.json", "w") as f:
        cfg_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()}
        json.dump(cfg_dict, f, indent=2)

    if args.dry_run:
        from ebw_ml.experiment import expand_grid
        grid = expand_grid(cfg)
        sliced = grid[cfg.shard::cfg.num_shards] if cfg.num_shards > 1 else grid
        print(f"[dry-run] full grid = {len(grid)} configurations")
        print(f"[dry-run] this shard ({cfg.shard}/{cfg.num_shards}) = {len(sliced)}")
        if sliced:
            print(f"[dry-run] first config: {sliced[0]}")
            print(f"[dry-run] last config:  {sliced[-1]}")
        return

    print(f"[run_s7] cfg.out_dir = {cfg.out_dir}")
    print(f"[run_s7] device = {cfg.device}  (EBW_DEVICE override respected)")
    print(f"[run_s7] models = {cfg.models or 'ALL 41'}")
    print(f"[run_s7] optimisers = {cfg.optimisers or 'ALL 12'}")
    print(f"[run_s7] generators = {cfg.generators}")
    print(f"[run_s7] sample_sizes = {cfg.sample_sizes}")
    print(f"[run_s7] ablation_modes = {cfg.ablation_modes}")
    print(f"[run_s7] n_folds = {cfg.n_folds}, n_hpo_trials = {cfg.n_hpo_trials}")
    print(f"[run_s7] max_train_size = {cfg.max_train_size}, "
          f"per_config_timeout_s = {cfg.per_config_timeout_s} (hard)")
    print(f"[run_s7] shard = {cfg.shard}/{cfg.num_shards}")

    df = run_experiment(cfg, progress_every=args.progress_every)
    print(f"\nFinal results: {cfg.out_dir / 'results_aggregated.csv'}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
