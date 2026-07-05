"""Registry of 41 multi-output regression models for the EBW benchmark.

Seven families (9 + 4 + 4 + 7 + 4 + 11 + 2 = 41 models). Maps to Table 3 of
the manuscript.
"""
from ebw_ml.models.base import BaseRegressor, FitContext
from ebw_ml.models.ensemble import ENSEMBLE_MODELS
from ebw_ml.models.glm import GLM_MODELS
from ebw_ml.models.kernel_gp import KERNEL_MODELS
from ebw_ml.models.nn import NN_MODELS
from ebw_ml.models.other import OTHER_MODELS
from ebw_ml.models.trees import BOOSTING_MODELS, TREE_MODELS

ALL_MODELS = (
    GLM_MODELS
    + KERNEL_MODELS
    + TREE_MODELS
    + BOOSTING_MODELS
    + OTHER_MODELS
    + NN_MODELS
    + ENSEMBLE_MODELS
)

MODEL_REGISTRY = {cls.name: cls for cls in ALL_MODELS}

FAMILY_OF = {cls.name: cls.family for cls in ALL_MODELS}

__all__ = [
    "BaseRegressor", "FitContext",
    "ALL_MODELS", "MODEL_REGISTRY", "FAMILY_OF",
    "GLM_MODELS", "KERNEL_MODELS", "TREE_MODELS",
    "BOOSTING_MODELS", "OTHER_MODELS", "NN_MODELS", "ENSEMBLE_MODELS",
]