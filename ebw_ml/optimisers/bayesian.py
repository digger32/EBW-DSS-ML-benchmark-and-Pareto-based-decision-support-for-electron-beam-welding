"""
Bayesian / surrogate-based optimisers:

* BO-GP (scikit-optimize, Gaussian-process surrogate)
* Optuna-TPE
* Optuna-CMA-ES
* Hyperopt-TPE
* Hyperband (Optuna with HyperbandPruner)
"""
from __future__ import annotations

import time
import warnings

import numpy as np

from ebw_ml.optimisers.base import (
    BaseOptimiser,
    SearchResult,
    evaluate_hp,
)


def _skopt_dimensions(space: dict):
    from skopt.space import Categorical, Integer, Real
    dims = []
    names = []
    for k, v in space.items():
        names.append(k)
        if isinstance(v, tuple) and len(v) == 3:
            lo, hi, kind = v
            if kind == "log-uniform":
                dims.append(Real(float(lo), float(hi), prior="log-uniform", name=k))
            else:
                dims.append(Real(float(lo), float(hi), prior="uniform", name=k))
        elif isinstance(v, list):
            dims.append(Categorical(v, name=k))
        else:
            dims.append(Categorical([v], name=k))
    return dims, names


class SkoptBOGP(BaseOptimiser):
    name = "bo_gp"; family = "Bayesian"

    def search(self, model_cls, X, Y, cv=None):
        from skopt import gp_minimize
        t0 = time.time()
        space = dict(model_cls.search_space)
        if not space:
            score = evaluate_hp(model_cls, {}, X, Y, cv=cv, seed=self.seed)
            return SearchResult(self.name, {}, score, 1, time.time() - t0, [])
        dims, names = _skopt_dimensions(space)
        history: list[dict] = []

        def f(x):
            hp = {n: v for n, v in zip(names, x)}
            s = evaluate_hp(model_cls, hp, X, Y, cv=cv, seed=self.seed)
            history.append({"hp": dict(hp), "score": float(s)})
            return s

        n_init = min(max(5, self.n_trials // 4), self.n_trials)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = gp_minimize(f, dims, n_calls=self.n_trials,
                              n_initial_points=n_init,
                              random_state=self.seed, verbose=False)
        best_hp = {n: v for n, v in zip(names, res.x)}
        return SearchResult(self.name, best_hp, float(res.fun),
                            self.n_trials, time.time() - t0, history)


def _optuna_suggest(trial, space: dict) -> dict:
    out: dict = {}
    for k, v in space.items():
        if isinstance(v, tuple) and len(v) == 3:
            lo, hi, kind = v
            if kind == "log-uniform":
                out[k] = trial.suggest_float(k, float(lo), float(hi), log=True)
            else:
                out[k] = trial.suggest_float(k, float(lo), float(hi))
        elif isinstance(v, list):
            out[k] = trial.suggest_categorical(k, v)
        else:
            out[k] = v
    return out


class OptunaTPE(BaseOptimiser):
    name = "tpe"; family = "Bayesian"

    def search(self, model_cls, X, Y, cv=None):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        t0 = time.time()
        space = dict(model_cls.search_space)
        history: list[dict] = []

        def objective(trial):
            hp = _optuna_suggest(trial, space)
            s = evaluate_hp(model_cls, hp, X, Y, cv=cv, seed=self.seed)
            history.append({"hp": dict(hp), "score": float(s)})
            return s

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        return SearchResult(self.name, dict(study.best_params), float(study.best_value),
                            self.n_trials, time.time() - t0, history)


class OptunaCMAES(BaseOptimiser):
    name = "cmaes"; family = "Bayesian"

    def search(self, model_cls, X, Y, cv=None):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        t0 = time.time()
        space = dict(model_cls.search_space)
        # CMA-ES needs at least one continuous dimension; if the space is
        # purely categorical, fall back to TPE.
        has_continuous = any(
            isinstance(v, tuple) and len(v) == 3 and v[2] in ("uniform", "log-uniform")
            for v in space.values()
        )
        sampler = (optuna.samplers.CmaEsSampler(seed=self.seed,
                                                  warn_independent_sampling=False)
                   if has_continuous else optuna.samplers.TPESampler(seed=self.seed))
        history: list[dict] = []

        def objective(trial):
            hp = _optuna_suggest(trial, space)
            s = evaluate_hp(model_cls, hp, X, Y, cv=cv, seed=self.seed)
            history.append({"hp": dict(hp), "score": float(s)})
            return s

        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        return SearchResult(self.name, dict(study.best_params), float(study.best_value),
                            self.n_trials, time.time() - t0, history)


class HyperoptTPE(BaseOptimiser):
    name = "hyperopt_tpe"; family = "Bayesian"

    def search(self, model_cls, X, Y, cv=None):
        from hyperopt import fmin, hp as hopt, tpe, Trials, STATUS_OK
        t0 = time.time()
        space_raw = dict(model_cls.search_space)
        if not space_raw:
            score = evaluate_hp(model_cls, {}, X, Y, cv=cv, seed=self.seed)
            return SearchResult(self.name, {}, score, 1, time.time() - t0, [])
        hp_space: dict = {}
        for k, v in space_raw.items():
            if isinstance(v, tuple) and len(v) == 3:
                lo, hi, kind = v
                if kind == "log-uniform":
                    hp_space[k] = hopt.loguniform(k, np.log(float(lo)), np.log(float(hi)))
                else:
                    hp_space[k] = hopt.uniform(k, float(lo), float(hi))
            elif isinstance(v, list):
                hp_space[k] = hopt.choice(k, v)
            else:
                hp_space[k] = v
        history: list[dict] = []

        def fn(hp):
            # hyperopt returns indices for categoricals via hopt.choice when
            # passed through fmin's "best", but inside fn we get values.
            score = evaluate_hp(model_cls, hp, X, Y, cv=cv, seed=self.seed)
            history.append({"hp": dict(hp), "score": float(score)})
            return {"loss": score, "status": STATUS_OK}

        trials = Trials()
        rng = np.random.default_rng(self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best = fmin(fn=fn, space=hp_space, algo=tpe.suggest,
                        max_evals=self.n_trials, trials=trials,
                        rstate=np.random.default_rng(self.seed),
                        show_progressbar=False)
        # hyperopt returns indices for choice -- convert back
        best_hp: dict = {}
        for k, v in space_raw.items():
            if isinstance(v, list):
                best_hp[k] = v[best[k]] if k in best else v[0]
            elif k in best:
                best_hp[k] = best[k]
        best_score = min(history, key=lambda r: r["score"])["score"]
        return SearchResult(self.name, best_hp, float(best_score),
                            self.n_trials, time.time() - t0, history)


class HyperbandOpt(BaseOptimiser):
    """Hyperband via Optuna's HyperbandPruner.

    Treats ``n_trials`` as the total number of configurations to evaluate.
    For models that support iterative training, the pruner cuts off poor
    configurations early; for the present benchmark we proxy the "resource"
    by a partial CV fold count (1 -> 2 -> 5 folds).
    """
    name = "hyperband"; family = "Bayesian"

    def search(self, model_cls, X, Y, cv=None):
        import optuna
        from sklearn.model_selection import KFold
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        t0 = time.time()
        space = dict(model_cls.search_space)
        history: list[dict] = []

        rungs = [2, 3, 5]  # CV folds at each rung (KFold requires >=2)

        def objective(trial):
            hp = _optuna_suggest(trial, space)
            # Progressive evaluation
            scores: list[float] = []
            for step, n_folds in enumerate(rungs):
                kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.seed + step)
                X_arr, Y_arr = np.asarray(X), np.asarray(Y)
                fold_rmses = []
                for tr, va in kf.split(X_arr):
                    try:
                        m = model_cls(**hp)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            m.fit(X_arr[tr], Y_arr[tr])
                            yp = m.predict(X_arr[va])
                        rmse = float(np.sqrt(np.mean((yp - Y_arr[va]) ** 2)))
                        fold_rmses.append(rmse)
                    except Exception:
                        fold_rmses.append(10.0)
                rung_score = float(np.mean(fold_rmses))
                scores.append(rung_score)
                trial.report(rung_score, step)
                if trial.should_prune():
                    history.append({"hp": dict(hp), "score": rung_score, "pruned_at": step})
                    raise optuna.TrialPruned()
            history.append({"hp": dict(hp), "score": scores[-1]})
            return scores[-1]

        pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=len(rungs))
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        return SearchResult(self.name, dict(study.best_params), float(study.best_value),
                            self.n_trials, time.time() - t0, history)
