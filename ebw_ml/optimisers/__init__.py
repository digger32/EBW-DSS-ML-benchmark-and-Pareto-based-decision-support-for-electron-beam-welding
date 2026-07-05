"""Registry of 12 hyperparameter optimisers (Section 3.6 of the manuscript)."""
from ebw_ml.optimisers.base import (
    BaseOptimiser,
    SearchResult,
    SearchSpace,
    evaluate_hp,
)
from ebw_ml.optimisers.bayesian import (
    HyperbandOpt,
    HyperoptTPE,
    OptunaCMAES,
    OptunaTPE,
    SkoptBOGP,
)
from ebw_ml.optimisers.enumerative import GridSearchOpt, RandomSearchOpt
from ebw_ml.optimisers.evolutionary import (
    DEAPGAOpt,
    NSGA2Opt,
    SkLearnGAOpt,
)
from ebw_ml.optimisers.swarm import DEOpt, PSOOpt

ALL_OPTIMISERS = [
    GridSearchOpt, RandomSearchOpt,           # enumerative (2)
    SkoptBOGP, OptunaTPE, OptunaCMAES, HyperoptTPE, HyperbandOpt,  # Bayesian (5)
    SkLearnGAOpt, DEAPGAOpt, NSGA2Opt,        # evolutionary (3)
    PSOOpt, DEOpt,                             # swarm (2)
]

OPTIMISER_REGISTRY = {cls.name: cls for cls in ALL_OPTIMISERS}
OPTIMISER_FAMILY = {cls.name: cls.family for cls in ALL_OPTIMISERS}

__all__ = [
    "BaseOptimiser", "SearchResult", "SearchSpace", "evaluate_hp",
    "ALL_OPTIMISERS", "OPTIMISER_REGISTRY", "OPTIMISER_FAMILY",
    "GridSearchOpt", "RandomSearchOpt",
    "SkoptBOGP", "OptunaTPE", "OptunaCMAES", "HyperoptTPE", "HyperbandOpt",
    "SkLearnGAOpt", "DEAPGAOpt", "NSGA2Opt",
    "PSOOpt", "DEOpt",
]