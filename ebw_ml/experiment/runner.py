"""
Main ExperimentRunner: orchestrates the full S-7 grid.

Axes:
* generators      -- {ctgan, tvae, copula, physics}
* sample_sizes    -- {1000, 5000, 10000, 25000, 50000}
* ablation modes  -- {real, synth, real_plus_synth, tstr}
* models          -- 41 models from ebw_ml.models
* optimisers      -- 12 from ebw_ml.optimisers

For each configuration the runner:
1. Builds the train and test sets from real + synthetic data.
2. Runs HPO with the given optimiser (5-fold CV inside the train set).
3. Refits the model with the best hyperparameters on the full train set.
4. Computes 15 evaluation metrics on the held-out fold or TSTR-test set.
5. Writes per-fold rows to ``results.csv`` and an aggregated row to
   ``results_aggregated.csv``; marks the configuration as done.

Patch v5 changes
----------------
* ``per_config_timeout_s`` is now a HARD budget, enforced with SIGALRM around
  the fold loop (the previous soft check only fired *between* folds, so a slow
  HPO search inside a single fold could not be interrupted -- this was the main
  reason bucket D blew far past its 600 s budget). When the budget is hit the
  partial results computed so far are kept, the row is marked with
  ``status="timeout"`` and the campaign moves on. The timer is disarmed before
  any CSV write, so storage is never interrupted mid-write.
* Grid sharding: ``shard`` / ``num_shards`` slice the grid round-robin so the
  campaign can be split across N single-threaded processes (one per core-group)
  without contending on done.csv. Each shard writes its own out_dir; use
  merge_runs.py to consolidate.
"""
from __future__ import annotations

import signal
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ebw_ml.experiment.metrics import compute_multi
from ebw_ml.experiment.mlflow_logger import MLflowLogger
from ebw_ml.experiment.storage import ResultStore
from ebw_ml.models import FAMILY_OF, MODEL_REGISTRY, FitContext
from ebw_ml.optimisers import OPTIMISER_FAMILY, OPTIMISER_REGISTRY

INPUT_COLS = ["IW", "IF", "VW", "FP"]
OUTPUT_COLS = ["Depth", "Width"]

ABLATION_MODES = ("real", "synth", "real_plus_synth", "tstr")


# ---------------------------------------------------------------------------
# Hard per-configuration timeout (SIGALRM-based)
# ---------------------------------------------------------------------------
class _ConfigTimeout(BaseException):
    """Raised by the SIGALRM handler when a configuration exceeds its budget.

    Inherits from BaseException (not Exception) on purpose: the fold body wraps
    HPO and refit in ``except Exception`` blocks, and we must NOT let those
    swallow the timeout -- it has to propagate up to the fold-loop handler.
    """


class _hard_timeout:
    """Context manager that raises ``_ConfigTimeout`` after ``seconds``.

    No-op if ``seconds`` is falsy, if SIGALRM is unavailable (non-POSIX), or if
    called from a non-main thread. Uses ``setitimer`` for sub-second precision.
    Note: a single uninterruptible C call (e.g. one long native fit) is only
    interrupted once control returns to the Python interpreter; for this
    campaign time is spread over many short calls, so the cap is effective.
    """

    def __init__(self, seconds: float | None) -> None:
        self.seconds = seconds
        self._enabled = False
        self._old_handler = None

    def __enter__(self) -> "_hard_timeout":
        if not self.seconds or self.seconds <= 0:
            return self
        if not hasattr(signal, "SIGALRM"):
            return self
        try:
            self._old_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, self._handler)
            signal.setitimer(signal.ITIMER_REAL, float(self.seconds))
            self._enabled = True
        except (ValueError, OSError):
            # ValueError: signal only works in the main thread.
            self._enabled = False
        return self

    @staticmethod
    def _handler(signum, frame):  # noqa: ANN001
        raise _ConfigTimeout()

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._enabled:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self._old_handler)
        return False  # never suppress


@dataclass
class S7Config:
    """Top-level configuration for the S-7 campaign."""
    real_csv: Path
    synth_dir: Path
    out_dir: Path
    models: list[str] = field(default_factory=list)
    optimisers: list[str] = field(default_factory=list)
    generators: list[str] = field(default_factory=lambda: ["ctgan", "tvae", "copula", "physics"])
    sample_sizes: list[int] = field(default_factory=lambda: [1000, 5000, 10000, 25000, 50000])
    ablation_modes: list[str] = field(default_factory=lambda: list(ABLATION_MODES))
    n_folds: int = 5
    n_hpo_trials: int = 30
    seed: int = 42
    device: str = "auto"
    mlflow_enabled: bool = False
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "ebw_s7"
    # Cap the size of any train set passed to a model. Synthetic data sets of
    # n=10000+ make O(n^2) and O(n^3) regressors (GPR, kernel ridge, TheilSen,
    # stacking-with-GPR) prohibitively slow inside HPO loops. Sub-sampling
    # the train set to `max_train_size` keeps the scientific contribution
    # (the comparison across generators, models and optimisers) while making
    # the campaign feasible on a single server. ``None`` disables the cap.
    max_train_size: int | None = None
    # HARD wall-clock budget per configuration (HPO + refit, all folds),
    # enforced with SIGALRM. ``None`` disables the budget.
    per_config_timeout_s: float | None = None
    # Grid sharding for multi-process parallelism. The grid is sliced
    # round-robin as grid[shard::num_shards]. Each shard should use its own
    # out_dir; consolidate with merge_runs.py.
    shard: int = 0
    num_shards: int = 1


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_real(real_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(real_csv)
    df.columns = [c.strip() for c in df.columns]
    X = df[INPUT_COLS].values.astype(np.float64)
    Y = df[OUTPUT_COLS].values.astype(np.float64)
    return X, Y


def load_synth(synth_dir: Path, generator: str, n_synth: int) -> tuple[np.ndarray, np.ndarray]:
    fp = Path(synth_dir) / f"{generator}_n{n_synth}.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Synthetic data not found: {fp}")
    df = pd.read_csv(fp)
    df.columns = [c.strip() for c in df.columns]
    X = df[INPUT_COLS].values.astype(np.float64)
    Y = df[OUTPUT_COLS].values.astype(np.float64)
    return X, Y


# ---------------------------------------------------------------------------
# Train/test split builders for each ablation mode
# ---------------------------------------------------------------------------
def _split_real_kfold(X_real: np.ndarray, Y_real: np.ndarray,
                      n_folds: int, seed: int):
    """5-fold splits over the real data set."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_real)):
        yield fold_idx, tr_idx, va_idx


def build_train_test(mode: str, X_real, Y_real, X_synth, Y_synth,
                      tr_idx, va_idx,
                      max_train_size: int | None = None,
                      seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct (X_tr, Y_tr, X_te, Y_te) for one fold of a given mode.

    If ``max_train_size`` is set and the resulting train set exceeds it,
    a deterministic random sub-sample (seeded) is drawn. Test sets are
    never sub-sampled.
    """
    if mode == "real":
        X_tr, Y_tr = X_real[tr_idx], Y_real[tr_idx]
        X_te, Y_te = X_real[va_idx], Y_real[va_idx]
    elif mode == "synth":
        X_tr, Y_tr = X_synth, Y_synth
        X_te, Y_te = X_real[va_idx], Y_real[va_idx]
    elif mode == "real_plus_synth":
        X_tr = np.vstack([X_real[tr_idx], X_synth])
        Y_tr = np.vstack([Y_real[tr_idx], Y_synth])
        X_te, Y_te = X_real[va_idx], Y_real[va_idx]
    elif mode == "tstr":
        # train-on-synth, test-on-all-real (single split, ignore va_idx)
        X_tr, Y_tr = X_synth, Y_synth
        X_te, Y_te = X_real, Y_real
    else:
        raise ValueError(f"Unknown ablation mode: {mode}")

    if max_train_size is not None and len(X_tr) > max_train_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_tr), size=max_train_size, replace=False)
        X_tr, Y_tr = X_tr[idx], Y_tr[idx]
    return X_tr, Y_tr, X_te, Y_te


# ---------------------------------------------------------------------------
# Core: run one (gen, n, mode, model, optimiser) configuration
# ---------------------------------------------------------------------------
def run_one_configuration(cfg: S7Config, store: ResultStore, mlf: MLflowLogger,
                           generator: str, n_synth: int, ablation: str,
                           model: str, optimiser: str) -> dict:
    mode = ablation
    model_name = model
    opt_name = optimiser
    key = {
        "generator": generator, "n_synth": n_synth, "ablation": mode,
        "model": model_name, "optimiser": opt_name,
    }
    if store.is_done(key):
        return {"status": "skipped"}

    t_start = time.time()
    model_cls = MODEL_REGISTRY[model_name]
    opt_cls = OPTIMISER_REGISTRY[opt_name]

    # Load data
    X_real, Y_real = load_real(cfg.real_csv)
    try:
        X_synth, Y_synth = load_synth(cfg.synth_dir, generator, n_synth)
    except FileNotFoundError:
        return {"status": "no_synth", **key}

    if mode == "tstr":
        folds = [(0, np.arange(len(X_real)), np.arange(len(X_real)))]
    else:
        folds = list(_split_real_kfold(X_real, Y_real, cfg.n_folds, cfg.seed))

    fold_rows: list[dict] = []
    fold_metrics: list[dict] = []

    run_name = f"{generator}|n{n_synth}|{mode}|{model_name}|{opt_name}"
    timed_out = False
    with mlf.run(run_name, tags={"model": model_name, "optimiser": opt_name,
                                  "generator": generator, "ablation": mode,
                                  "n_synth": str(n_synth)}) as run:
        # HARD timeout wraps the entire fold loop (HPO + refit). It is disarmed
        # on exit, before any CSV write below.
        try:
            with _hard_timeout(cfg.per_config_timeout_s):
                for fold_idx, tr_idx, va_idx in folds:
                    X_tr, Y_tr, X_te, Y_te = build_train_test(
                        mode, X_real, Y_real, X_synth, Y_synth, tr_idx, va_idx,
                        max_train_size=cfg.max_train_size, seed=cfg.seed + fold_idx,
                    )
                    # HPO on the training set (internal CV)
                    t_hpo_start = time.time()
                    try:
                        opt = opt_cls(n_trials=cfg.n_hpo_trials, seed=cfg.seed)
                        hpo_kf = KFold(n_splits=min(5, max(2, len(X_tr) // 5)),
                                        shuffle=True, random_state=cfg.seed)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            res = opt.search(model_cls, X_tr, Y_tr, cv=hpo_kf)
                        best_hp = res.best_hp
                        hpo_score = res.best_score
                        hpo_evals = res.n_evaluations
                        hpo_error = None
                    except Exception as e:
                        best_hp = {}
                        hpo_score = float("nan")
                        hpo_evals = 0
                        hpo_error = f"{type(e).__name__}: {str(e)[:120]}"
                    hpo_elapsed = time.time() - t_hpo_start

                    # Refit with best_hp and evaluate on test
                    t_fit_start = time.time()
                    ctx = FitContext(seed=cfg.seed, device=cfg.device)
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            m = model_cls(**best_hp)
                            m.fit(X_tr, Y_tr, ctx=ctx)
                            Y_hat = m.predict(X_te)
                        metrics = compute_multi(Y_te, Y_hat, n_features=X_tr.shape[1],
                                                  seed=cfg.seed, bootstrap=False)
                        fit_error = None
                    except Exception as e:
                        metrics = {}
                        fit_error = f"{type(e).__name__}: {str(e)[:120]}"
                    fit_elapsed = time.time() - t_fit_start

                    fold_row = {
                        **key, "fold": fold_idx,
                        "hpo_score": hpo_score, "hpo_evals": hpo_evals,
                        "hpo_elapsed_s": round(hpo_elapsed, 3),
                        "fit_elapsed_s": round(fit_elapsed, 3),
                        "best_hp": str(best_hp)[:300],
                        "hpo_error": hpo_error or "",
                        "fit_error": fit_error or "",
                        **metrics,
                    }
                    fold_rows.append(fold_row)
                    if metrics:
                        fold_metrics.append(metrics)
        except _ConfigTimeout:
            timed_out = True

        # Aggregate across folds: mean and std of each metric
        agg: dict[str, float] = {}
        if fold_metrics:
            keys_m = list(fold_metrics[0].keys())
            for k in keys_m:
                vals = [m[k] for m in fold_metrics if not np.isnan(m.get(k, np.nan))]
                if vals:
                    agg[f"{k}_mean"] = float(np.mean(vals))
                    agg[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

        total_elapsed = time.time() - t_start
        agg_row = {
            **key, "model_family": FAMILY_OF.get(model_name, "?"),
            "optimiser_family": OPTIMISER_FAMILY.get(opt_name, "?"),
            "n_folds_used": len(folds),          # planned folds (unchanged meaning)
            "n_folds_ok": len(fold_metrics),     # folds that produced valid metrics
            "status": "timeout" if timed_out else "ok",
            "total_elapsed_s": round(total_elapsed, 2),
            **agg,
        }
        run.log_params({**key})
        run.log_metrics({k: v for k, v in agg.items() if isinstance(v, (int, float))})

    store.append_fold_rows(fold_rows)
    store.append_aggregated_row(agg_row)
    store.mark_done(key)
    status = "timeout" if timed_out else "ok"
    return {"status": status, **key, "elapsed_s": total_elapsed}


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------
def expand_grid(cfg: S7Config) -> list[dict]:
    """Generate every (generator, n_synth, ablation, model, optimiser) tuple."""
    grid = []
    models = cfg.models or list(MODEL_REGISTRY.keys())
    opts = cfg.optimisers or list(OPTIMISER_REGISTRY.keys())
    for g in cfg.generators:
        for n in cfg.sample_sizes:
            for mode in cfg.ablation_modes:
                # 'real' mode does not use synthetic data, so it is enough to
                # evaluate it once per (model, optimiser); we still emit it
                # against (g, n) for bookkeeping but only once.
                if mode == "real" and (g, n) != (cfg.generators[0], cfg.sample_sizes[0]):
                    continue
                for m in models:
                    for o in opts:
                        grid.append({
                            "generator": g, "n_synth": n, "ablation": mode,
                            "model": m, "optimiser": o,
                        })
    return grid


def run_experiment(cfg: S7Config, progress_every: int = 5) -> pd.DataFrame:
    store = ResultStore(cfg.out_dir)
    mlf = MLflowLogger(cfg.mlflow_tracking_uri, cfg.mlflow_experiment, cfg.mlflow_enabled)
    grid = expand_grid(cfg)
    full_n = len(grid)
    if cfg.num_shards and cfg.num_shards > 1:
        grid = grid[cfg.shard::cfg.num_shards]
        print(f"[runner] shard {cfg.shard}/{cfg.num_shards}: "
              f"{len(grid)} of {full_n} configurations")
    else:
        print(f"[runner] Total configurations: {full_n}")
    done_before = len(store._done)
    print(f"[runner] Already completed: {done_before}")

    t0 = time.time()
    n_timeout = 0
    for i, conf in enumerate(grid):
        res = run_one_configuration(cfg, store, mlf, **conf)
        if res.get("status") == "timeout":
            n_timeout += 1
        if (i + 1) % progress_every == 0 or i == len(grid) - 1:
            elapsed = time.time() - t0
            done = len(store._done) - done_before
            processed = i + 1
            rate = processed / max(elapsed, 1e-6)
            remaining = (len(grid) - processed) / max(rate, 1e-6)
            print(f"[runner] {i+1}/{len(grid)}  done={done}  timeouts={n_timeout}  "
                  f"elapsed={elapsed:.0f}s  eta={remaining:.0f}s  "
                  f"last={res.get('status','?')}", flush=True)

    df = store.aggregated_df()
    print(f"[runner] FINISHED. Wrote {cfg.out_dir/'results_aggregated.csv'} "
          f"({len(df)} rows; {n_timeout} timed out this session)")
    return df
