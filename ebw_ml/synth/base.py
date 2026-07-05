"""
Base protocol for synthetic-data generators in the EBW-ML package.

All four families (CTGAN, TVAE, Gaussian copula, physics-informed Rosenthal)
follow this interface to enable a uniform pipeline in Section S-4 of the
manuscript.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class SyntheticGenerator(Protocol):
    """Uniform interface for tabular synthetic-data generators.

    Notes
    -----
    The protocol is structural (runtime_checkable) so that subclasses do not
    need to inherit from a concrete base class to be recognised. Concrete
    classes are expected to fix all random seeds during ``fit`` and to
    propagate the same seed to ``sample`` whenever an external seed is given.
    """

    name: str  # short identifier, e.g. "ctgan", "tvae", "copula", "physics"

    def fit(self, df: pd.DataFrame) -> "SyntheticGenerator":
        """Fit the generator on a real data frame.

        Parameters
        ----------
        df : pd.DataFrame
            Real data with columns IW, IF, VW, FP, Depth, Width.

        Returns
        -------
        self
        """
        ...

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        """Draw ``n`` synthetic observations.

        Parameters
        ----------
        n : int
            Number of synthetic rows.
        seed : int or None
            Random seed; ``None`` reuses the seed fixed during ``fit``.
        """
        ...

    def save(self, path: Path) -> None:  # pragma: no cover
        """Serialise the fitted generator to ``path``."""
        ...

    @classmethod
    def load(cls, path: Path) -> "SyntheticGenerator":  # pragma: no cover
        """Deserialise a fitted generator from ``path``."""
        ...
