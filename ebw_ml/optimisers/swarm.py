"""
Swarm / population-based single-objective optimisers via pymoo:

* PSO -- particle swarm optimisation
* DE  -- differential evolution
"""
from __future__ import annotations

import time

import numpy as np

from ebw_ml.optimisers.base import BaseOptimiser, SearchResult, evaluate_hp
from ebw_ml.optimisers.evolutionary import _decode_genome, _encode_space


def _pymoo_so_search(algo, model_cls, X, Y, cv, seed: int,
                      n_trials: int, name: str) -> SearchResult:
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize
    from pymoo.termination.max_eval import MaximumFunctionCallTermination

    t0 = time.time()
    space = dict(model_cls.search_space)
    enc = _encode_space(space)
    n_dim = len(enc)
    if n_dim == 0:
        score = evaluate_hp(model_cls, {}, X, Y, cv=cv, seed=seed)
        return SearchResult(name, {}, score, 1, time.time() - t0, [])

    history: list[dict] = []

    class _Problem(ElementwiseProblem):
        def __init__(p_self):
            super().__init__(n_var=n_dim, n_obj=1,
                             xl=np.zeros(n_dim), xu=np.ones(n_dim))

        def _evaluate(p_self, x, out, *args, **kwargs):
            hp = _decode_genome(enc, list(x))
            s = evaluate_hp(model_cls, hp, X, Y, cv=cv, seed=seed)
            history.append({"hp": dict(hp), "score": float(s)})
            out["F"] = s

    problem = _Problem()
    termination = MaximumFunctionCallTermination(n_trials)
    res = minimize(problem, algo, termination, seed=seed, verbose=False)
    best_x = np.atleast_2d(res.X)[0]
    best_hp = _decode_genome(enc, list(best_x))
    best_score = float(np.atleast_1d(res.F)[0])
    return SearchResult(name, best_hp, best_score,
                        len(history), time.time() - t0, history)


class PSOOpt(BaseOptimiser):
    name = "pso"; family = "Swarm"

    def search(self, model_cls, X, Y, cv=None):
        from pymoo.algorithms.soo.nonconvex.pso import PSO
        algo = PSO(pop_size=10, w=0.7, c1=1.5, c2=1.5)
        return _pymoo_so_search(algo, model_cls, X, Y, cv,
                                self.seed, self.n_trials, self.name)


class DEOpt(BaseOptimiser):
    name = "de"; family = "Swarm"

    def search(self, model_cls, X, Y, cv=None):
        from pymoo.algorithms.soo.nonconvex.de import DE
        algo = DE(pop_size=12, variant="DE/rand/1/bin", CR=0.7, F=0.5)
        return _pymoo_so_search(algo, model_cls, X, Y, cv,
                                self.seed, self.n_trials, self.name)
