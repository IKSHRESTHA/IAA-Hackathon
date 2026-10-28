"""Tweedie GLM benchmark (log link, exposure offset) on a classical rating structure.

Continuous factors are banded on training quantiles (the usual tariff-style
treatment, robust to outliers and non-linearity) and missing values get their own
level. The result is fully transparent: each factor has a table of
multiplicative relativities that an actuary can review and smooth.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from glum import GeneralizedLinearRegressor, TweedieDistribution

log = logging.getLogger(__name__)

MISSING = "__missing__"


@dataclass
class Bander:
    """Quantile bands for numeric columns, fixed on training data."""

    n_bins: int = 10
    edges_: dict[str, np.ndarray] = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, numeric: list[str]) -> "Bander":
        self.edges_ = {}
        for c in numeric:
            v = X[c].astype(float)
            v = v[np.isfinite(v)]
            if v.nunique() <= self.n_bins:
                edges = np.unique(v)            # treat as discrete levels
                self.edges_[c] = np.concatenate([[-np.inf], (edges[:-1] + edges[1:]) / 2, [np.inf]])
            else:
                q = np.unique(np.quantile(v, np.linspace(0, 1, self.n_bins + 1)))
                q[0], q[-1] = -np.inf, np.inf
                self.edges_[c] = q
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c, e in self.edges_.items():
            idx = np.searchsorted(e, X[c].astype(float).to_numpy(), side="right") - 1
            idx = np.clip(idx, 0, len(e) - 2)
            lab = np.array([f"b{i:02d}" for i in range(len(e) - 1)], dtype=object)[idx]
            lab = np.where(np.isfinite(X[c].astype(float).to_numpy()), lab, MISSING)
            out[c] = pd.Categorical(lab, categories=[f"b{i:02d}" for i in range(len(e) - 1)] + [MISSING])
        return pd.DataFrame(out, index=X.index)


@dataclass
class TweedieGLM:
    numeric: list[str]
    categorical: list[str]
    power: float = 1.5
    alpha: float = 5e-4
    n_bins: int = 10
    family: str | None = None      # None = Tweedie(power); or "poisson" / "gamma"
    model_: GeneralizedLinearRegressor | None = field(default=None, repr=False)
    bander_: Bander | None = field(default=None, repr=False)

    def _design(self, X: pd.DataFrame) -> pd.DataFrame:
        banded = self.bander_.transform(X)
        cats = X[self.categorical].copy()
        for c in self.categorical:
            if not isinstance(cats[c].dtype, pd.CategoricalDtype):
                cats[c] = cats[c].astype("category")
        return pd.concat([banded, cats], axis=1)

    def fit(self, X: pd.DataFrame, y: np.ndarray, offset: np.ndarray) -> "TweedieGLM":
        self.bander_ = Bander(self.n_bins).fit(X, self.numeric)
        D = self._design(X)
        fam = TweedieDistribution(self.power) if self.family is None else self.family
        self.model_ = GeneralizedLinearRegressor(
            family=fam, link="log", alpha=self.alpha, l1_ratio=0.0,
            fit_intercept=True, drop_first=False, max_iter=200)
        self.model_.fit(D, np.asarray(y, dtype=float), offset=np.asarray(offset, dtype=float))
        log.info("GLM fitted on %d rows, %d coefficients", len(D), len(self.model_.coef_))
        return self

    def predict(self, X: pd.DataFrame, offset: np.ndarray) -> np.ndarray:
        return self.model_.predict(self._design(X), offset=np.asarray(offset, dtype=float))

    def relativities(self) -> pd.DataFrame:
        """Multiplicative relativity per factor level, rebased to the factor's mean-coef level."""
        names = list(self.model_.feature_names_)
        coef = pd.Series(self.model_.coef_, index=names)
        rows = []
        for n, b in coef.items():
            factor, _, level = n.partition("[")
            rows.append({"factor": factor, "level": level.rstrip("]"), "coef": b})
        df = pd.DataFrame(rows)
        df["relativity"] = np.exp(df["coef"] - df.groupby("factor")["coef"].transform("mean"))
        return df
