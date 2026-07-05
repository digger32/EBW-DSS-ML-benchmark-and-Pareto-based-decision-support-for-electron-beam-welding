"""Synthetic-data generators for the EBW data set."""
from ebw_ml.synth.base import SyntheticGenerator
from ebw_ml.synth.physics import PhysicsRosenthalGenerator
from ebw_ml.synth.sdv_generators import (
    CTGANGenerator,
    CopulaGenerator,
    TVAEGenerator,
)

GENERATOR_REGISTRY = {
    "ctgan": CTGANGenerator,
    "tvae": TVAEGenerator,
    "copula": CopulaGenerator,
    "physics": PhysicsRosenthalGenerator,
}

__all__ = [
    "SyntheticGenerator",
    "CTGANGenerator",
    "TVAEGenerator",
    "CopulaGenerator",
    "PhysicsRosenthalGenerator",
    "GENERATOR_REGISTRY",
]