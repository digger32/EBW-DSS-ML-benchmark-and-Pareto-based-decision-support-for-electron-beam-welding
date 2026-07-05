"""
Evolutionary optimisers:

* GA-DEAP        -- custom DEAP GA with elitism and adaptive mutation.
* GA-SK          -- sklearn-genetic-opt-style GA (DEAP-based, tournament,
                    uniform crossover).
* NSGA-II        -- multi-objective optimisation on (RMSE_Depth, RMSE_Width)
                    via pymoo, the hyperparameter optimiser version. The
                    inverse design at evaluation time uses a separate
                    pymoo NSGA-II run.
"""
from __future__ import annotations

import random
import time
import warnings

import numpy as np
from sklearn.model_selection import KFold

from ebw_ml.optimisers.base import BaseOptimiser, SearchResult, evaluate_hp


# ---------------------------------------------------------------------------
# Helpers: encoding/decoding HP vectors for evolutionary search.
# ---------------------------------------------------------------------------
def _encode_space(space: dict) -> list[tuple]:
    """Each entry: (key, kind, lo, hi or categories)."""
    enc = []
    for k, v in space.items():
        if isinstance(v, tuple) and len(v) == 3:
            lo, hi, kind = v
            enc.append((k, kind, float(lo), float(hi), None))
        elif isinstance(v, list):
            enc.append((k, "categorical", None, None, list(v)))
        else:
            enc.append((k, "constant", v, None, None))
    return enc


def _decode_genome(enc, genome: list[float]) -> dict:
    """Genome is a list of floats in [0, 1]; decode to actual HP."""
    hp = {}
    for (k, kind, lo, hi, cats), u in zip(enc, genome):
        if kind == "uniform":
            hp[k] = float(lo + u * (hi - lo))
        elif kind == "log-uniform":
            hp[k] = float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))
        elif kind == "categorical":
            idx = min(int(u * len(cats)), len(cats) - 1)
            hp[k] = cats[idx]
        elif kind == "constant":
            hp[k] = lo
    return hp


# ---------------------------------------------------------------------------
# 1. DEAP custom GA with elitism
# ---------------------------------------------------------------------------
class DEAPGAOpt(BaseOptimiser):
    name = "ga_deap"; family = "Evolutionary"

    def __init__(self, n_trials: int = 30, seed: int = 42,
                 pop_size: int = 12, cxpb: float = 0.7,
                 mutpb_start: float = 0.4, mutpb_end: float = 0.1) -> None:
        super().__init__(n_trials=n_trials, seed=seed)
        self.pop_size = int(pop_size)
        self.cxpb = float(cxpb)
        self.mutpb_start = float(mutpb_start)
        self.mutpb_end = float(mutpb_end)

    def search(self, model_cls, X, Y, cv=None):
        from deap import base, creator, tools
        t0 = time.time()
        space = dict(model_cls.search_space)
        enc = _encode_space(space)
        n_dim = len(enc)
        if n_dim == 0:
            score = evaluate_hp(model_cls, {}, X, Y, cv=cv, seed=self.seed)
            return SearchResult(self.name, {}, score, 1, time.time() - t0, [])

        # DEAP setup (guarded against duplicate registration)
        if not hasattr(creator, "FitMinEBW"):
            creator.create("FitMinEBW", base.Fitness, weights=(-1.0,))
            creator.create("IndivEBW", list, fitness=creator.FitMinEBW)

        random.seed(self.seed)
        np.random.seed(self.seed)
        history: list[dict] = []

        toolbox = base.Toolbox()
        toolbox.register("gene", random.random)
        toolbox.register("individual", tools.initRepeat, creator.IndivEBW,
                         toolbox.gene, n=n_dim)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxBlend, alpha=0.3)
        toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.15, indpb=0.5)
        toolbox.register("select", tools.selTournament, tournsize=3)

        def evaluate(ind):
            hp = _decode_genome(enc, ind)
            s = evaluate_hp(model_cls, hp, X, Y, cv=cv, seed=self.seed)
            history.append({"hp": dict(hp), "score": float(s)})
            return (s,)

        toolbox.register("evaluate", evaluate)

        pop = toolbox.population(n=self.pop_size)
        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind)
        n_evals = len(pop)
        n_gens = max(1, (self.n_trials - n_evals) // self.pop_size)

        for gen in range(n_gens):
            mutpb = self.mutpb_start + (self.mutpb_end - self.mutpb_start) * (gen / max(1, n_gens - 1))
            # Elitism: preserve top 2
            elites = tools.selBest(pop, 2)
            offspring = list(map(toolbox.clone, toolbox.select(pop, self.pop_size - 2)))
            # crossover
            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cxpb:
                    toolbox.mate(c1, c2)
                    del c1.fitness.values; del c2.fitness.values
            # mutation
            for m in offspring:
                if random.random() < mutpb:
                    toolbox.mutate(m)
                    m[:] = [max(0.0, min(1.0, g)) for g in m]
                    del m.fitness.values
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid:
                ind.fitness.values = toolbox.evaluate(ind)
                n_evals += 1
                if n_evals >= self.n_trials:
                    break
            pop[:] = elites + offspring
            if n_evals >= self.n_trials:
                break

        best = tools.selBest(pop, 1)[0]
        best_hp = _decode_genome(enc, best)
        return SearchResult(self.name, best_hp, float(best.fitness.values[0]),
                            n_evals, time.time() - t0, history)


# ---------------------------------------------------------------------------
# 2. sklearn-genetic-opt-style GA: tournament + uniform crossover, no elitism
# ---------------------------------------------------------------------------
class SkLearnGAOpt(BaseOptimiser):
    name = "ga_sk"; family = "Evolutionary"

    def __init__(self, n_trials: int = 30, seed: int = 42, pop_size: int = 10) -> None:
        super().__init__(n_trials=n_trials, seed=seed)
        self.pop_size = int(pop_size)

    def search(self, model_cls, X, Y, cv=None):
        t0 = time.time()
        space = dict(model_cls.search_space)
        enc = _encode_space(space)
        n_dim = len(enc)
        if n_dim == 0:
            score = evaluate_hp(model_cls, {}, X, Y, cv=cv, seed=self.seed)
            return SearchResult(self.name, {}, score, 1, time.time() - t0, [])

        rng = np.random.default_rng(self.seed)
        history: list[dict] = []

        def eval_genome(g):
            hp = _decode_genome(enc, g)
            s = evaluate_hp(model_cls, hp, X, Y, cv=cv, seed=self.seed)
            history.append({"hp": dict(hp), "score": float(s)})
            return s

        pop = [rng.uniform(0, 1, n_dim).tolist() for _ in range(self.pop_size)]
        fits = [eval_genome(g) for g in pop]
        n_evals = self.pop_size

        while n_evals < self.n_trials:
            # tournament size 2
            def tournament():
                a, b = int(rng.integers(0, self.pop_size)), int(rng.integers(0, self.pop_size))
                return pop[a] if fits[a] < fits[b] else pop[b]

            new_pop, new_fits = [], []
            while len(new_pop) < self.pop_size and n_evals < self.n_trials:
                p1, p2 = tournament(), tournament()
                # uniform crossover
                child = [p1[i] if rng.random() < 0.5 else p2[i] for i in range(n_dim)]
                # Gaussian mutation, indpb=0.2
                for i in range(n_dim):
                    if rng.random() < 0.2:
                        child[i] = float(np.clip(child[i] + rng.normal(0, 0.15), 0, 1))
                new_pop.append(child)
                new_fits.append(eval_genome(child))
                n_evals += 1
            pop, fits = new_pop, new_fits

        best_idx = int(np.argmin(fits))
        best_hp = _decode_genome(enc, pop[best_idx])
        return SearchResult(self.name, best_hp, float(fits[best_idx]),
                            n_evals, time.time() - t0, history)


# ---------------------------------------------------------------------------
# 3. NSGA-II (multi-objective HPO on RMSE_Depth and RMSE_Width)
# ---------------------------------------------------------------------------
def _evaluate_hp_multi(model_cls, hp: dict, X, Y, cv: KFold, seed: int) -> tuple[float, float]:
    """Mean RMSE per target across CV folds."""
    rmse_d, rmse_w = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, va in cv.split(X):
            try:
                m = model_cls(**hp); m.fit(X[tr], Y[tr])
                yp = m.predict(X[va])
                rmse_d.append(float(np.sqrt(np.mean((yp[:, 0] - Y[va, 0]) ** 2))))
                rmse_w.append(float(np.sqrt(np.mean((yp[:, 1] - Y[va, 1]) ** 2))))
            except Exception:
                rmse_d.append(10.0); rmse_w.append(10.0)
    return float(np.mean(rmse_d)), float(np.mean(rmse_w))


class NSGA2Opt(BaseOptimiser):
    name = "nsga2"; family = "Evolutionary"

    def __init__(self, n_trials: int = 30, seed: int = 42, pop_size: int = 12) -> None:
        super().__init__(n_trials=n_trials, seed=seed)
        self.pop_size = int(pop_size)

    def search(self, model_cls, X, Y, cv=None):
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.optimize import minimize
        from pymoo.termination.max_eval import MaximumFunctionCallTermination

        t0 = time.time()
        space = dict(model_cls.search_space)
        enc = _encode_space(space)
        n_dim = len(enc)
        if cv is None:
            cv = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        if n_dim == 0:
            score = evaluate_hp(model_cls, {}, X, Y, cv=cv, seed=self.seed)
            return SearchResult(self.name, {}, score, 1, time.time() - t0, [])

        history: list[dict] = []

        class HPProblem(ElementwiseProblem):
            def __init__(p_self):
                super().__init__(n_var=n_dim, n_obj=2,
                                 xl=np.zeros(n_dim), xu=np.ones(n_dim))

            def _evaluate(p_self, x, out, *args, **kwargs):
                hp = _decode_genome(enc, list(x))
                f1, f2 = _evaluate_hp_multi(model_cls, hp, X, Y, cv, self.seed)
                history.append({"hp": dict(hp), "score": 0.5 * (f1 + f2),
                                 "rmse_D": f1, "rmse_W": f2})
                out["F"] = [f1, f2]

        problem = HPProblem()
        algo = NSGA2(pop_size=self.pop_size)
        termination = MaximumFunctionCallTermination(self.n_trials)
        res = minimize(problem, algo, termination, seed=self.seed, verbose=False)
        # Pick the knee point: minimum of sum
        F = np.atleast_2d(res.F)
        sums = F.sum(axis=1)
        idx = int(np.argmin(sums))
        best_x = np.atleast_2d(res.X)[idx]
        best_hp = _decode_genome(enc, list(best_x))
        best_score = float(0.5 * F[idx].sum())
        return SearchResult(self.name, best_hp, best_score,
                            len(history), time.time() - t0, history)
