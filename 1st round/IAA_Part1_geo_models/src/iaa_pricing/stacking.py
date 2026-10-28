"""Stacking meta-learner over the experts (out-of-time, per line).

The meta-learner is a *log-linear pool* fitted with the competition loss (Tweedie, p=1.5):

    log mu_i = log c_i + b0 + sum_k b_k log T_k,i + g * log F_i(a)

* ``T_k`` are expert predictions of the ultimate cost (total GBM, GLM, peril experts,
  large-loss expert combinations);
* ``F_i(a) = (a + N_i) / (a + Lambda_i)`` is the Bayesian (Poisson-gamma) credibility
  factor of the policy: ``N_i`` claims observed in the 3-year history window and
  ``Lambda_i`` the claims expected there by the a-priori frequency expert. ``a`` is the
  gamma shape (credibility constant), chosen on the training years of the meta-learner;
* ``c_i`` converts ultimate to *reported* cost for back-test years (row-specific IBNR
  completeness from the peril experts' mix); it is 1 for 2025 rows.

Out-of-time protocol: expert predictions for 2023 come from models trained on 2021-22 and
for 2024 from models trained on 2021-23. The meta-learner is validated across years (fit
on 2023 -> score 2024 and vice versa) before the final fit on both years is applied to
the 2025 predictions of experts trained on 2021-24.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from glum import GeneralizedLinearRegressor, TweedieDistribution

from . import metrics as mt

EPS = 1e-6


def row_completeness(df: pd.DataFrame) -> np.ndarray:
    """Reported share of ultimate for each scored row (1 for the prediction year)."""
    if "peril_rep" in df and df["peril"].gt(0).all():
        c = (df["peril_rep"] / df["peril"]).to_numpy()
    else:
        c = df["comp_line"].fillna(1.0).to_numpy()
    c = np.where(np.isfinite(c), c, 1.0)
    return np.where(df["set"].eq("final").to_numpy(), 1.0, np.clip(c, 1e-4, 1.0))


def set_level(df: pd.DataFrame, ref: str = "total") -> np.ndarray:
    """Portfolio average rate of the reference expert within each scored set (per row).

    The pool works on *rates relative to this level*, which makes it scale-equivariant
    (if every expert is 6% higher in 2025 because of trend, so is the pooled prediction)
    and proportional to exposure (no distortion of part-year rows).
    """
    lvl = df.groupby("set")[ref].transform("sum") / df.groupby("set")["exposure"].transform("sum")
    return lvl.to_numpy(dtype=float)


def credibility_factor(df: pd.DataFrame, a: float) -> np.ndarray:
    """Poisson-gamma posterior mean frailty ``(a + N) / (a + Lambda)``.

    Uses the policy's *entire* known history when available (``hist_all_*``: every year
    before the row's year, claims reported before 1 January of that year), otherwise the
    3-year window. The posterior mean weighs longer histories more (2025 rows have four
    years of history against two or three in the back-tests) without any re-fitting.
    """
    if "hist_all_n" in df and "hist_all_expo" in df:
        n = df["hist_all_n"].fillna(0).to_numpy(dtype=float)
        expo = df["hist_all_expo"].fillna(0).to_numpy(dtype=float)
    else:
        n = df["hist_pol_n_3y"].fillna(0).to_numpy(dtype=float)
        expo = df["hist_pol_expo_3y"].fillna(0).to_numpy(dtype=float)
    lam = df["freq_rate"].to_numpy(dtype=float) * expo
    return (a + n) / (a + lam)


def all_history(scored: pd.DataFrame, claims: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """Claims and exposure of each scored row's policy over *all* earlier years (as-at).

    ``claims``: prepared claims (policy_id, loss_date, report_date, is_repudiated);
    ``keys``: training policy rows (policy_id, period_start, exposure). For a row in year
    Y only claims with loss and report date before 1 January Y count (no look-ahead).
    """
    c = claims.loc[~claims["is_repudiated"].astype(bool), ["policy_id", "loss_date", "report_date"]]
    k = keys.assign(kyear=keys["period_start"].dt.year)[["policy_id", "kyear", "exposure"]]
    out = pd.DataFrame(index=scored.index, data={"hist_all_n": 0.0, "hist_all_expo": 0.0})
    for y in sorted(scored["year"].astype(int).unique()):
        cut = pd.Timestamp(year=y, month=1, day=1)
        idx = scored.index[scored["year"].astype(int) == y]
        pol = scored.loc[idx, "policy_id"]
        n = c[(c["loss_date"] < cut) & (c["report_date"] < cut)].groupby("policy_id").size()
        e = k[k["kyear"] < y].groupby("policy_id")["exposure"].sum()
        out.loc[idx, "hist_all_n"] = pol.map(n).fillna(0).to_numpy(dtype=float)
        out.loc[idx, "hist_all_expo"] = pol.map(e).fillna(0).to_numpy(dtype=float)
    return out


@dataclass
class LogLinearPool:
    experts: list[str]
    a: float | None = 2.0                  # credibility constant (None = no credibility term)
    power: float = 1.5
    alpha: float = 1e-5
    exposure_term: bool = True             # mu ∝ exposure^(1 + b): lets the data correct
                                           # non-proportional claim rates of part-year rows
    nonneg: bool = False                   # constrain expert weights >= 0 (robust pooling)
    extra_cats: tuple[str, ...] = ()       # segment columns with their own level (e.g. products held)
    model_: GeneralizedLinearRegressor | None = field(default=None, repr=False)

    def design(self, df: pd.DataFrame) -> pd.DataFrame:
        """``log`` of each expert; an entry ``"a/b"`` is the log-ratio (tilt) of two experts,
        e.g. ``total_plus_large/total`` = how much the large-loss expert departs from the
        uniform load for this risk (avoids feeding two almost collinear experts)."""
        cols = {}
        e = df["exposure"].to_numpy(dtype=float)
        lvl = set_level(df, self.ref)
        for k in self.experts:
            if "/" in k:
                a, b = k.split("/")
                cols[f"tilt_{a}"] = np.log(np.maximum(df[a].to_numpy(dtype=float), EPS)
                                           / np.maximum(df[b].to_numpy(dtype=float), EPS))
            else:  # log of the expert's rate relative to the set's average rate
                cols[f"log_{k}"] = np.log(np.maximum(df[k].to_numpy(dtype=float), EPS) / e / lvl)
        X = pd.DataFrame(cols, index=df.index)
        if self.exposure_term:
            X["log_exposure"] = np.log(e)
        for col in self.extra_cats:      # one indicator per non-reference level
            v = df[col].astype(str).to_numpy()
            for lv in sorted(set(v))[1:] if not getattr(self, "_levels", None) else self._levels[col][1:]:
                X[f"{col}={lv}"] = (v == lv).astype(float)
        if self.a is not None:
            X["log_cred"] = np.log(credibility_factor(df, self.a))
        return X

    def fit(self, df: pd.DataFrame) -> "LogLinearPool":
        self._levels = {c: sorted(set(df[c].astype(str))) for c in self.extra_cats}
        X = self.design(df)
        lb = None
        if self.nonneg:  # experts (log_*) non-negative; tilts, exposure, credibility, segments free
            lb = np.array([0.0 if c.startswith("log_") and c not in ("log_exposure", "log_cred")
                           else -np.inf for c in X.columns])
        self.model_ = GeneralizedLinearRegressor(family=TweedieDistribution(self.power), link="log",
                                                 alpha=self.alpha, l1_ratio=0.0, fit_intercept=True,
                                                 lower_bounds=lb)
        self.model_.fit(X, df["y"].to_numpy(dtype=float), offset=self.offset(df, True))
        return self

    @property
    def ref(self) -> str:
        """Reference expert for the portfolio level (first non-tilt expert)."""
        return next(k for k in self.experts if "/" not in k)

    def offset(self, df: pd.DataFrame, reported: bool) -> np.ndarray:
        """log(exposure x set level [x completeness]): the pool predicts a relative rate."""
        off = np.log(df["exposure"].to_numpy(dtype=float) * set_level(df, self.ref))
        return off + (np.log(row_completeness(df)) if reported else 0.0)

    def predict(self, df: pd.DataFrame, reported: bool = True) -> np.ndarray:
        """Expected cost; ``reported=True`` multiplies by completeness (back-test scoring)."""
        return self.model_.predict(self.design(df), offset=self.offset(df, reported))

    def coefficients(self) -> dict[str, float]:
        return {"intercept": float(self.model_.intercept_),
                **dict(zip(self.model_.feature_names_, map(float, self.model_.coef_)))}


def choose_credibility(df: pd.DataFrame, experts: list[str], grid=(2, 4, 8, 16, 32, 64)) -> float:
    """Credibility constant minimising in-sample deviance of the pool (1-D, robust)."""
    best, best_dev = None, np.inf
    for a in grid:
        m = LogLinearPool(experts, a=a).fit(df)
        dev = mt.mean_tweedie_deviance(df["y"], m.predict(df), 1.5)
        if dev < best_dev:
            best, best_dev = a, dev
    return float(best)


def expert_scores(df: pd.DataFrame, cols: list[str], power: float = 1.5) -> pd.DataFrame:
    """D^2 / Gini / A-E of each expert column on a back-test year (reported basis)."""
    c = row_completeness(df)
    y, e = df["y"].to_numpy(dtype=float), df["exposure"].to_numpy(dtype=float)
    rows = []
    for k in cols:
        mu = np.maximum(df[k].to_numpy(dtype=float), EPS) * (c if not k.startswith("meta") else 1.0)
        rows.append({"model": k, "d2": mt.d2_tweedie(y, mu, power), "gini": mt.normalized_gini(y, mu, e),
                     "ae": float(y.sum() / mu.sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Gated pool: expert weights that vary by cluster (mixture-of-experts gating)
# ---------------------------------------------------------------------------
TAIL_PERILS: dict[str, tuple[str, ...]] = {
    "motor": ("bodily_injury",),
    "buildings": ("flood", "fire", "subsidence"),
    "contents": ("flood", "theft"),
}


@dataclass
class GatedPool:
    """Log-linear pool whose weights depend on a risk cluster (a simple gating network).

    Clusters are defined only from expert *predictions* (no outcome information), so they
    exist for the 2025 rows too: the band of predicted rate (risk level) crossed with the
    band of the predicted share of heavy-tailed perils (bodily injury; flood/fire/
    subsidence; flood/theft). Cluster-specific weights are ridge-penalised deviations from
    the global weights, so sparse clusters fall back to the global pool.
    """

    experts: list[str]
    line: str
    a: float | None = 2.0
    n_rate_bands: int = 4
    n_tail_bands: int = 2
    alpha: float = 1e-3
    power: float = 1.5
    rate_edges_: np.ndarray | None = None
    tail_edges_: np.ndarray | None = None
    levels_: list[str] = field(default_factory=list)
    model_: GeneralizedLinearRegressor | None = field(default=None, repr=False)

    def _tail_share(self, df: pd.DataFrame) -> np.ndarray:
        cols = [f"peril_{p}" for p in TAIL_PERILS.get(self.line, ()) if f"peril_{p}" in df]
        if not cols or "peril" not in df:
            return np.zeros(len(df))
        return (df[cols].sum(axis=1) / df["peril"].clip(lower=EPS)).to_numpy()

    def clusters(self, df: pd.DataFrame) -> np.ndarray:
        ref = self._pool().ref
        rate = np.log(np.maximum(df[ref].to_numpy(), EPS) / df["exposure"].to_numpy())
        tail = self._tail_share(df)
        if self.rate_edges_ is None:
            self.rate_edges_ = np.quantile(rate, np.linspace(0, 1, self.n_rate_bands + 1)[1:-1])
            self.tail_edges_ = np.quantile(tail, np.linspace(0, 1, self.n_tail_bands + 1)[1:-1])
        r = np.searchsorted(self.rate_edges_, rate)
        t = np.searchsorted(self.tail_edges_, tail)
        return np.array([f"r{i}t{j}" for i, j in zip(r, t)], dtype=object)

    def design(self, df: pd.DataFrame) -> pd.DataFrame:
        base = LogLinearPool(self.experts, a=self.a).design(df)
        cl = self.clusters(df)
        if not self.levels_:
            self.levels_ = sorted(set(cl))
        X = base.copy()
        for lv in self.levels_:
            ind = (cl == lv).astype(float)
            X[f"c_{lv}"] = ind
            for col in base.columns:
                X[f"{col}:{lv}"] = base[col].to_numpy() * ind
        return X

    def fit(self, df: pd.DataFrame) -> "GatedPool":
        self.rate_edges_, self.tail_edges_, self.levels_ = None, None, []
        X = self.design(df)
        self.model_ = GeneralizedLinearRegressor(family=TweedieDistribution(self.power), link="log",
                                                 alpha=self.alpha, l1_ratio=0.0, fit_intercept=True)
        self.model_.fit(X, df["y"].to_numpy(dtype=float), offset=self._pool().offset(df, True))
        return self

    def _pool(self) -> "LogLinearPool":
        return LogLinearPool(self.experts, a=self.a)

    def predict(self, df: pd.DataFrame, reported: bool = True) -> np.ndarray:
        return self.model_.predict(self.design(df), offset=self._pool().offset(df, reported))
