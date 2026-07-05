"""
Adapters around SDV's CTGAN, TVAE and Gaussian-copula synthesisers.

These three families share an SDV ``SingleTableMetadata`` description of the
EBW schema and produce data frames with the same six columns as the real
data set (Tynchenko et al. 2021, 72 observations).
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import (
    CTGANSynthesizer,
    GaussianCopulaSynthesizer,
    TVAESynthesizer,
)

EBW_COLUMNS = ["IW", "IF", "VW", "FP", "Depth", "Width"]


def _ebw_metadata(df: pd.DataFrame) -> SingleTableMetadata:
    meta = SingleTableMetadata()
    meta.detect_from_dataframe(df)
    # Force numerical types -- all six columns are continuous-valued physical
    # quantities, even if they appear discrete because of the factorial
    # design.
    for col in EBW_COLUMNS:
        meta.update_column(column_name=col, sdtype="numerical",
                           computer_representation="Float")
    return meta


class CTGANGenerator:
    """Conditional tabular GAN (Xu et al. 2019) via SDV.

    The default architecture follows Xu et al. but with a reduced number of
    epochs and embedding dimension because the real sample size is only
    N = 72.
    """

    name = "ctgan"

    def __init__(
        self,
        epochs: int = 600,
        batch_size: int = 50,
        embedding_dim: int = 64,
        generator_dim: tuple[int, ...] = (128, 128),
        discriminator_dim: tuple[int, ...] = (128, 128),
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.seed = seed
        self.verbose = verbose
        self._synth: Optional[CTGANSynthesizer] = None

    def fit(self, df: pd.DataFrame) -> "CTGANGenerator":
        np.random.seed(self.seed)
        meta = _ebw_metadata(df[EBW_COLUMNS])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._synth = CTGANSynthesizer(
                metadata=meta,
                epochs=self.epochs,
                batch_size=self.batch_size,
                embedding_dim=self.embedding_dim,
                generator_dim=self.generator_dim,
                discriminator_dim=self.discriminator_dim,
                verbose=self.verbose,
                enforce_min_max_values=True,
                enforce_rounding=False,
            )
            self._synth.fit(df[EBW_COLUMNS])
        return self

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        assert self._synth is not None, "fit() must be called before sample()"
        if seed is not None:
            np.random.seed(seed)
        s = self._synth.sample(num_rows=n)[EBW_COLUMNS].copy()
        return s

    def save(self, path: Path) -> None:
        assert self._synth is not None
        self._synth.save(filepath=str(path))

    @classmethod
    def load(cls, path: Path) -> "CTGANGenerator":
        obj = cls()
        obj._synth = CTGANSynthesizer.load(filepath=str(path))
        return obj


class TVAEGenerator:
    """Tabular variational autoencoder (Patki et al. 2016 / SDV).

    Default depth is reduced from SDV defaults given the very small training
    set.
    """

    name = "tvae"

    def __init__(
        self,
        epochs: int = 600,
        batch_size: int = 50,
        embedding_dim: int = 64,
        compress_dims: tuple[int, ...] = (128, 128),
        decompress_dims: tuple[int, ...] = (128, 128),
        seed: int = 42,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims
        self.seed = seed
        self._synth: Optional[TVAESynthesizer] = None

    def fit(self, df: pd.DataFrame) -> "TVAEGenerator":
        np.random.seed(self.seed)
        meta = _ebw_metadata(df[EBW_COLUMNS])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._synth = TVAESynthesizer(
                metadata=meta,
                epochs=self.epochs,
                batch_size=self.batch_size,
                embedding_dim=self.embedding_dim,
                compress_dims=self.compress_dims,
                decompress_dims=self.decompress_dims,
                enforce_min_max_values=True,
                enforce_rounding=False,
            )
            self._synth.fit(df[EBW_COLUMNS])
        return self

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        assert self._synth is not None
        if seed is not None:
            np.random.seed(seed)
        s = self._synth.sample(num_rows=n)[EBW_COLUMNS].copy()
        return s

    def save(self, path: Path) -> None:
        assert self._synth is not None
        self._synth.save(filepath=str(path))

    @classmethod
    def load(cls, path: Path) -> "TVAEGenerator":
        obj = cls()
        obj._synth = TVAESynthesizer.load(filepath=str(path))
        return obj


class CopulaGenerator:
    """Gaussian copula model (Patki et al. 2016 / SDV).

    Marginal distributions are selected from a parametric family with default
    Gaussian KDE; the dependence structure is captured by a multivariate
    Gaussian copula in rank space (Sklar 1959).
    """

    name = "copula"

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._synth: Optional[GaussianCopulaSynthesizer] = None

    def fit(self, df: pd.DataFrame) -> "CopulaGenerator":
        np.random.seed(self.seed)
        meta = _ebw_metadata(df[EBW_COLUMNS])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._synth = GaussianCopulaSynthesizer(
                metadata=meta,
                enforce_min_max_values=True,
                enforce_rounding=False,
                default_distribution="gaussian_kde",
            )
            self._synth.fit(df[EBW_COLUMNS])
        return self

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        assert self._synth is not None
        if seed is not None:
            np.random.seed(seed)
        s = self._synth.sample(num_rows=n)[EBW_COLUMNS].copy()
        return s

    def save(self, path: Path) -> None:
        assert self._synth is not None
        self._synth.save(filepath=str(path))

    @classmethod
    def load(cls, path: Path) -> "CopulaGenerator":
        obj = cls()
        obj._synth = GaussianCopulaSynthesizer.load(filepath=str(path))
        return obj
