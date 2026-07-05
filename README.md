# EBW-DSS — ML benchmark and Pareto-based decision support for electron-beam welding

Reproducibility repository for the manuscript *“Multi-output machine-learning
benchmark and Pareto-based decision support for electron-beam welding of
thin-walled titanium structures with comparative synthetic-data augmentation”*
(submitted to *The International Journal of Advanced Manufacturing Technology*).

The study predicts weld-bead geometry — penetration depth `Depth` and bead
width `Width` — from four EBW process parameters (`IW`, `IF`, `VW`, `FP`) on a
72-row real data set, and adds: a four-generator synthetic-data comparison
(CTGAN, TVAE, Gaussian copula, physics-informed Rosenthal), a 41-model × 12-
optimiser × 4-ablation benchmark, Friedman/Wilcoxon statistics, an NSGA-II
inverse design, and SHAP / PDP / ALE interpretability.

## Repository layout

```
ebw-dss/
├── ebw_ml/                 # core package (uniform Base{Regressor,Optimiser} APIs)
│   ├── synth/              # CTGAN, TVAE, copula, physics-informed generators
│   ├── validation/         # distributional validation (KS, AD, MMD, W2, PSI)
│   ├── models/             # 41 regressors across 7 families
│   ├── optimisers/         # 12 HPO methods across 4 families
│   └── experiment/         # runner, 15 metrics, resume-safe storage, MLflow
├── scripts/
│   ├── run_s4.py           # synthetic-data generation + distributional validation
│   ├── run_s7.py           # the main benchmark grid (sharded, resume-safe)
│   ├── bucket_D_prime.sh   # 3-fold re-run launcher for compute-heavy models
│   ├── merge_runs.py       # consolidate sharded run dirs (robust CSV reader)
│   ├── run_s9_inverse.py   # NSGA-II inverse design + capability frontier
│   ├── run_s11_interpret.py# permutation importance, PDP, ALE, SHAP
│   ├── eda.py              # exploratory data analysis (Section 4.1 figures)
│   └── check_gpu.py        # device / throughput diagnostic
├── data/
│   ├── ebw_real_72.csv     # the 72 real observations (also on Kaggle)
│   └── synth/              # synthetic sets per generator × size + validation JSON
└── docs/                   # S8_FINDINGS.md, S9_ANALYSIS.md, GPU/patch notes
```

## Environment

Python 3.12, single pinned environment (`requirements.txt`). A CUDA build of
PyTorch is listed but **not required**: the models and data are small and all
reported runs execute on CPU (`EBW_DEVICE=cpu`). See `scripts/check_gpu.py`.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Reproduction pipeline

```bash
# 1. Synthetic data + distributional validation (writes data/synth/*)
.venv/bin/python scripts/run_s4.py

# 2. Main benchmark grid. Sharded for parallelism; resume-safe.
#    Light models, default 3-fold for the non-TSTR ablations:
.venv/bin/python scripts/run_s7.py --n-folds 3 --num-shards 24 --shard 0
#    ... one process per shard, or use the launcher for compute-heavy models:
MODELS=ngb,mdn,vote,tabnet,ftt bash scripts/bucket_D_prime.sh

# 3. Merge all run directories into one consolidated set
.venv/bin/python scripts/merge_runs.py \
    --inputs runs/s7_A runs/s7_B runs/s7_C runs/s7_D runs/s7_merged_Dp3 runs/s7_Dp3_ftt_sh* \
    --output runs/s7_merged_final

# 4. Inverse design (forward surrogates: ngb, mdn) + capability frontier
.venv/bin/python scripts/run_s9_inverse.py --models ngb,mdn --n-gen 120 --pop 100

# 5. Interpretability for the top-3 surrogates on the real ablation
.venv/bin/python scripts/run_s11_interpret.py --models ngb,mdn,vote
```

Long-running jobs should be wrapped in `tmux`/`nohup` so they survive SSH
disconnect (the launcher already uses `nohup` per shard).

## Notes

- **Cross-validation depth.** 3-fold for the non-TSTR ablations (real,
  synth-only, real+synth) for feasibility of the full grid; TSTR is a single
  train-on-all-synthetic / test-on-all-real pass.
- **Synthetic-train cap.** Synthetic training sets are sub-sampled to
  `max_train_size = 2000` (verified to change macro-R² by < 0.02 vs 10 000).
- **Large artefacts.** `runs/` and `mlruns/` are git-ignored; the full
  per-configuration results and synthetic data are intended for a Zenodo
  deposit referenced from the manuscript.

## To confirm before public release

- `LICENSE` holder/year and the license choice itself (MIT is a placeholder).
- `CITATION.cff` author list and release metadata.
- The manuscript sources are not tracked in this repository; the paper
  figures are produced by the analysis scripts (`eda.py`, `run_s7.py` +
  post-processing, `run_s9_inverse.py`, `run_s11_interpret.py`).
