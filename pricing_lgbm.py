"""LightGBM pricing tools shared by the model steps of both rounds.

The model predicts the expected ultimate claim cost of a policy row. The method follows the
emerging-market pricing repository (iaa-pricing-part2), restricted to LightGBM:

* Exact log offset. LightGBM gets ``init_score = log(exposure x completeness) + base``, so the
  trees learn only the risk relativity, never the time on cover or the share of claims not yet
  reported. ``base`` is the log of the average cost per unit of offset. LightGBM skips its own
  starting value when an ``init_score`` is given, and without ``base`` the first trees would be
  spent fixing the level.
* On-levelled targets. Each peril's claim cost is projected to the cost level of the year being
  priced (frequency and severity trend), so older years compare with recent ones.
* Large losses. Training targets are capped at a high quantile, and the capped amount comes back
  as one factor per line. A handful of huge claims then cannot shape the trees.
* Early stopping on policies, then refit. The number of trees is chosen on a 20% sample of
  policies (all rows of a policy on the same side). The model is then refitted on every row with
  1.1 times that many trees, because the refit sees 25% more data.
* Balance. A Tweedie model with a log link is not calibrated in total, so its fitted training
  total is rescaled to the actual training total.

The competition score is the Tweedie deviance D^2 with power 1.5, all lines pooled.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

TWEEDIE_POWER = 1.5
# Columns that identify a row or its period. They are never model features.
KEYS = ("row_id", "policy_id", "customer_id", "address_id", "year", "period_start", "period_end")
MIN_COMPLETENESS = 1e-6        # floor for the reported share (a row starting on 31 Dec has almost 0)


# =================================================================================================
# Metric
# =================================================================================================
def tweedie_deviance(y: np.ndarray, mu: np.ndarray, p: float = TWEEDIE_POWER) -> np.ndarray:
    """Unit Tweedie deviance for 1 < p < 2 (compound Poisson-gamma), one value per row."""
    y = np.asarray(y, dtype=float)
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-12)           # guard against division by 0
    return 2 * (np.power(y, 2 - p) / ((1 - p) * (2 - p))
                - y * np.power(mu, 1 - p) / (1 - p)
                + np.power(mu, 2 - p) / (2 - p))


def d2_tweedie(y: np.ndarray, mu: np.ndarray, p: float = TWEEDIE_POWER) -> float:
    """Share of the deviance of a constant prediction (the mean) that the model removes.

    This is the competition metric: 0 is no better than one price for everyone, 1 is perfect.
    """
    y = np.asarray(y, dtype=float)
    null = tweedie_deviance(y, np.full_like(y, y.mean()), p).sum()    # deviance of the mean
    return float(1.0 - tweedie_deviance(y, mu, p).sum() / null)


def score(y: np.ndarray, mu: np.ndarray) -> dict[str, float]:
    """D^2 as predicted and with the level known (prediction rescaled to the actual total).

    The second one separates ranking from level: the level of a new year is the hardest part to
    get right, and the level-known D^2 shows how well the model orders the risks.
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    return {"rows": int(len(y)), "d2": d2_tweedie(y, mu),
            "d2_level_known": d2_tweedie(y, mu * y.sum() / mu.sum()),
            "actual": float(y.sum()), "predicted": float(mu.sum()), "a_over_e": float(y.sum() / mu.sum())}


# =================================================================================================
# Targets: trend, large losses, sums insured
# =================================================================================================
def trend_rates(trend: dict[str, Any], line: str, peril: str) -> tuple[float, float]:
    """Annual (frequency, severity) trend of one peril; severity is set per peril or per line."""
    t = trend[line]
    sev = t["severity"]
    return float(t["frequency"]), float(sev[peril] if isinstance(sev, dict) else sev)


def onlevel_cost(df: pd.DataFrame, line: str, perils: tuple[str, ...], trend: dict[str, Any],
                 target_year: int, level_shift: dict[str, Any] | None = None) -> np.ndarray:
    """Claim cost of each row restated at the cost level of ``target_year``.

    Each peril's cost is multiplied by ((1 + frequency) x (1 + severity)) ** (target_year - year).
    For an expected cost a frequency trend and a severity trend act the same way, so both go on
    the cost. ``level_shift`` restates a one-off change of regime: in the 1st round, bodily injury
    claims of 2021-22 were settled at a higher level than from 2023 on (factor 0.82).
    """
    years_ahead = target_year - df["year"].to_numpy(dtype=float)      # 0 for the target year itself
    total = np.zeros(len(df))
    for p in perils:
        f, s = trend_rates(trend, line, p)
        factor = np.power((1 + f) * (1 + s), years_ahead)
        if level_shift and level_shift["line"] == line and level_shift["peril"] == p:
            old = df["year"].isin(level_shift["years"]).to_numpy()   # rows of the old regime
            factor = np.where(old, factor * level_shift["factor"], factor)
        total += df[f"cost_{p}"].to_numpy(dtype=float) * factor
    return total


def cap_and_load(y: np.ndarray, quantile: float) -> tuple[np.ndarray, float, float]:
    """Cap row costs at a quantile of the rows with a claim, and the load that puts it back.

    Returns (capped costs, cap, load). ``load`` = total cost / total capped cost, so
    prediction x load restores the line's full level, large losses included.
    """
    y = np.asarray(y, dtype=float)
    cap = float(np.quantile(y[y > 0], quantile))                      # quantile of claiming rows only
    capped = np.minimum(y, cap)
    return capped, cap, float(y.sum() / capped.sum())


def index_sum_insured(df: pd.DataFrame, target_year: int, annual_rate: float) -> pd.DataFrame:
    """Sums insured restated in ``target_year`` money.

    Sums insured grow every year (about +3% buildings, +2% contents on renewals). Claims are
    on-levelled to the target year, so sums insured must be too, or the model would read
    inflation as a risk effect.
    """
    if "sum_insured" not in df or not annual_rate:
        return df
    out = df.copy()
    f = np.power(1.0 + annual_rate, float(target_year) - out["year"].to_numpy(dtype=float))
    out["sum_insured"] = (out["sum_insured"].to_numpy(dtype=float) * f).astype("float32")
    if "si_per_bedroom" in out:                                       # derived from sum_insured
        out["si_per_bedroom"] = (out["si_per_bedroom"].to_numpy(dtype=float) * f).astype("float32")
    return out


def completeness(df: pd.DataFrame, column: str = "comp_cost") -> np.ndarray:
    """Share of each training row's claims already reported at the valuation date (floored)."""
    return np.clip(df[column].to_numpy(dtype=float), MIN_COMPLETENESS, 1.0)


# =================================================================================================
# Model inputs
# =================================================================================================
def is_year_weather(feature: str) -> bool:
    """Weather of the row's own year (wx_<x>), as opposed to the address climate (wx_<x>_mean)."""
    return feature.startswith("wx_") and not feature.endswith("_mean")


def policy_sample(policy_id: pd.Series, share: float, salt: str = "es") -> np.ndarray:
    """True for a fixed ``share`` of the policies (hash of the id, so the same on every run).

    All rows of a policy fall on the same side: the sample cannot learn a policy from its other
    years and then score it again.
    """
    ids = policy_id.astype(str)
    table = {v: int(hashlib.md5(f"{salt}:{v}".encode()).hexdigest()[:8], 16) % 10_000
             for v in pd.unique(ids)}                                 # one hash per distinct policy
    return ids.map(table).to_numpy() < share * 10_000


def feature_columns(df: pd.DataFrame, drop: set[str] | None = None) -> list[str]:
    """Every column of a 2025 table except the keys, the exposure and ``drop``."""
    drop = set(KEYS) | {"exposure"} | set(drop or ())
    return [c for c in df.columns if c not in drop]


def prepare(df: pd.DataFrame, features: list[str], levels: dict[str, list] | None = None
            ) -> tuple[pd.DataFrame, dict[str, list]]:
    """Model matrix: text columns become categories, True/False/missing becomes 1/0/NaN.

    ``levels`` fixes the category levels. Pass the training levels when preparing the rows to
    predict, so the same code means the same level. Returns (matrix, levels).
    """
    X = df[features].copy()
    levels = dict(levels or {})
    for c in features:
        s = X[c]
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            X[c] = s.astype("float32")                                # booleans -> 1/0/NaN
        else:                                                         # text: a category
            if c not in levels:                                       # training call: learn the levels
                levels[c] = sorted(s.dropna().astype(str).unique())
            X[c] = pd.Categorical(s.astype("string"), categories=levels[c])   # unseen level -> NaN
    return X, levels


# =================================================================================================
# The model
# =================================================================================================
@dataclass
class TweedieLGBM:
    """LightGBM Tweedie regression with an exact log offset.

    Prediction = exp(trees + base + offset), where the offset is log(exposure x completeness),
    plus in the 2nd round the log of the developed-market prior.
    """

    params: dict[str, Any]
    features: list[str]
    monotone: list[int] | None = None        # +1 / -1 / 0 per feature: LightGBM monotone constraints
    booster: lgb.Booster | None = field(default=None, repr=False)
    base: float = 0.0
    n_trees: int = 0

    @property
    def identity(self) -> bool:
        """L2 regression models the value itself (no log link): used for the log of claim size."""
        return str(self.params.get("objective", "tweedie")) in ("regression", "regression_l2", "l2")

    def _params(self, seed: int) -> dict[str, Any]:
        p = {"objective": "tweedie", "tweedie_variance_power": TWEEDIE_POWER,
             "verbose": -1, "seed": seed, "deterministic": True, "force_col_wise": True}
        p.update(self.params)                                         # the round settings win
        p.setdefault("metric", "l2" if self.identity else p["objective"])   # stop on the model's own loss
        if self.monotone is not None and any(self.monotone):
            p["monotone_constraints"] = list(self.monotone)
            p.setdefault("monotone_constraints_method", "advanced")
        return p

    def fit(self, X: pd.DataFrame, y: np.ndarray, offset: np.ndarray, n_trees: int | None = None,
            X_es: pd.DataFrame | None = None, y_es: np.ndarray | None = None,
            offset_es: np.ndarray | None = None, max_trees: int = 2000, patience: int = 200,
            seed: int = 0) -> "TweedieLGBM":
        """Fit with a fixed number of trees, or early-stop on (X_es, y_es, offset_es)."""
        y = np.asarray(y, dtype=float)
        offset = np.asarray(offset, dtype=float)
        if (not self.identity and np.any(y < 0)) or not np.all(np.isfinite(offset)):
            raise ValueError("targets must be >= 0 and offsets finite")
        if self.identity:                    # starting point: the mean of what is left to explain
            self.base = float(np.mean(y - offset))
        else:                                # average target per unit of offset, on the log scale
            self.base = float(np.log(y.sum() / np.exp(offset).sum()))
        train = lgb.Dataset(X[self.features], label=y, init_score=offset + self.base,
                            categorical_feature="auto", free_raw_data=False)
        if X_es is None:                                              # fixed number of trees
            self.booster = lgb.train(self._params(seed), train, num_boost_round=int(n_trees))
            self.n_trees = int(n_trees)
            return self
        valid = lgb.Dataset(X_es[self.features], label=np.asarray(y_es, dtype=float),
                            init_score=np.asarray(offset_es, dtype=float) + self.base, reference=train)
        self.booster = lgb.train(self._params(seed), train, num_boost_round=max_trees,
                                 valid_sets=[valid], valid_names=["early_stop"],
                                 callbacks=[lgb.early_stopping(patience, verbose=False)])
        self.n_trees = max(int(self.booster.best_iteration or self.booster.current_iteration()), 1)
        return self

    def predict(self, X: pd.DataFrame, offset: np.ndarray) -> np.ndarray:
        """Expected value of each row: exp(trees + base + offset), or without exp() for L2."""
        raw = self.booster.predict(X[self.features], raw_score=True, num_iteration=self.n_trees)
        eta = raw + self.base + np.asarray(offset, dtype=float)
        return eta if self.identity else np.exp(eta)

    def importance(self) -> pd.DataFrame:
        """Share of the total split gain of each feature: how much each one drives the trees."""
        gain = self.booster.feature_importance("gain", iteration=self.n_trees)
        imp = pd.DataFrame({"feature": self.features, "gain": gain})
        imp["gain_share"] = imp["gain"] / max(imp["gain"].sum(), 1e-12)
        return imp.sort_values("gain", ascending=False, ignore_index=True)

    def save(self, folder: Path) -> None:
        """Booster as text, plus the offset intercept and tree count needed to predict again."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(folder / "booster.txt"), num_iteration=self.n_trees)
        meta = {"features": self.features, "params": self.params, "monotone": self.monotone,
                "base": self.base, "n_trees": self.n_trees}
        (folder / "meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")


def fit_early_stopped(params: dict[str, Any], X: pd.DataFrame, y: np.ndarray, offset: np.ndarray,
                      policy_id: pd.Series, refit: bool, seed: int, es_share: float = 0.2,
                      refit_factor: float = 1.1, min_trees: int = 20,
                      monotone: list[int] | None = None, patience: int = 200) -> TweedieLGBM:
    """Choose the number of trees on a policy sample, then (optionally) refit on every row."""
    features = list(X.columns)
    y, offset = np.asarray(y, dtype=float), np.asarray(offset, dtype=float)
    es = policy_sample(policy_id, es_share, salt=f"es{seed}")          # rows used only to stop
    m = TweedieLGBM(params, features, monotone).fit(X.loc[~es], y[~es], offset[~es], X_es=X.loc[es],
                                                     y_es=y[es], offset_es=offset[es], seed=seed,
                                                     patience=patience)
    if not refit:
        return m
    n = max(int(round(m.n_trees * refit_factor)), min_trees)           # more data: a few more trees
    return TweedieLGBM(params, features, monotone).fit(X, y, offset, n_trees=n, seed=seed)


def balance_factor(y: np.ndarray, pred: np.ndarray) -> float:
    """Factor that makes the fitted total equal the actual total on the training rows."""
    return float(np.sum(y) / np.sum(pred))


@dataclass
class LineFit:
    """A fitted line: the model plus everything needed to turn its output into a price."""

    model: TweedieLGBM
    levels: dict[str, list]      # category levels seen in training
    balance: float               # fitted total -> actual total (capped)
    load: float                  # capped total -> full total (large losses back in)
    cap: float                   # the large-loss cap on training row costs

    def predict(self, df: pd.DataFrame, offset: np.ndarray) -> np.ndarray:
        """Expected cost of each row of ``df``, large losses included."""
        X, _ = prepare(df, self.model.features, self.levels)
        return self.model.predict(X, offset) * self.balance * self.load


def fit_line(train: pd.DataFrame, features: list[str], y: np.ndarray, offset: np.ndarray,
             params: dict[str, Any], refit: bool, seed: int, cap_quantile: float,
             es_share: float = 0.2, refit_factor: float = 1.1) -> LineFit:
    """Cap the target, choose the trees, fit, balance: one line of business."""
    y_capped, cap, load = cap_and_load(y, cap_quantile)
    offset = np.asarray(offset, dtype=float)
    X, levels = prepare(train, features)
    m = fit_early_stopped(params, X, y_capped, offset, train["policy_id"], refit=refit, seed=seed,
                          es_share=es_share, refit_factor=refit_factor)
    b = balance_factor(y_capped, m.predict(X, offset))
    return LineFit(m, levels, b, load, cap)


# =================================================================================================
# Submission
# =================================================================================================
def check_submission(sub: pd.DataFrame, expected_ids: pd.Series) -> dict[str, bool]:
    """Hard checks before a file is written: exact columns, every row once, finite and positive."""
    ids = sub["row_id"].astype(str)
    pred = sub["prediction"].to_numpy(dtype=float)
    checks = {
        "columns are row_id, prediction": list(sub.columns) == ["row_id", "prediction"],
        "one row per expected row_id": len(sub) == len(expected_ids) and not ids.duplicated().any()
                                       and set(ids) == set(expected_ids.astype(str)),
        "predictions finite": bool(np.isfinite(pred).all()),
        "predictions strictly positive": bool((pred > 0).all()),
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"submission checks failed: {failed}")
    return checks


def save_by_line(new: pd.DataFrame, path: Path, lines: list[str], order: tuple[str, ...]) -> pd.DataFrame:
    """Write a per-line report, keeping the rows of the lines that were not refitted this run."""
    path = Path(path)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old.loc[~old["line"].isin(list(lines))], new], ignore_index=True)
    rank = {name: i for i, name in enumerate(order)}
    new = new.sort_values("line", key=lambda s: s.map(rank).fillna(len(rank)), kind="stable")
    new.to_csv(path, index=False)
    return new
