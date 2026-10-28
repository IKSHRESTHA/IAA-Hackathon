"""Model training, temporal back-testing and final fitting for one line of business.

Back-test protocol (mirrors the live exercise "price next year from history"):
  * train on years < H at the cost level of H; early-stop on a 10% policy-grouped
    sample of those years (never on H);
  * predict year H and compare with what has been *reported* for H: predictions are
    multiplied by each row's IBNR completeness, so immature years are scored fairly
    without inflating the observed claims;
  * score with Tweedie D^2 (p = 1.5), the competition metric, plus Gini and A/E.
Final fit: all training years at the 2025 cost level, same recipe, early stopping on
a policy-grouped sample, then the 2025 rows are predicted with offset log(exposure).
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import metrics as mt
from .config import Config
from .features import CategoryEncoder, FeatureSpec, credibility_encode, feature_spec, index_sum_insured
from .models.gbm import TweedieGBM
from .models.glm import TweedieGLM
from .targets import DevelopmentInputs, build_targets

log = logging.getLogger(__name__)

TARGET_PREFIXES = ("y", "n_", "comp_", "compn_")
TARGET_EXTRA = ("n_claims", "freq_trend", "log_offset")


@dataclass
class LineData:
    line: str
    train: pd.DataFrame
    test: pd.DataFrame
    spec: FeatureSpec
    encoder: CategoryEncoder
    target_summary: dict[str, Any] = field(default_factory=dict)


def policy_sample_mask(policy_id: pd.Series, share: float, salt: str = "es") -> np.ndarray:
    """Deterministic policy-grouped sample (hash of policy_id), stable across runs and lines."""
    h = policy_id.astype(str).map(lambda s: int(hashlib.md5(f"{salt}:{s}".encode()).hexdigest()[:8], 16))
    return (h.to_numpy() % 10_000) < share * 10_000


def load_line_data(
    line: str,
    cfg: Config,
    dev_inputs: DevelopmentInputs,
    target_year: int,
    cap_quantile: float | None,
    review_features: list[str] | None = None,
    max_train_year: int | None = None,
    fit_years: list[int] | None = None,
    target_encode: list[str] | None = None,
    encode_fit_max_year: int | None = None,
) -> LineData:
    """Features (from ``artifacts/features_<line>.parquet``) joined with targets.

    ``fit_years`` restricts the years used to set the large-loss cap and load (pass the
    training years in a back-test so the holdout year cannot influence them).
    """
    feats = pd.read_parquet(cfg.artifacts_dir / f"features_{line}.parquet")
    is_test = feats["is_test"].astype(bool)
    train = feats.loc[~is_test].reset_index(drop=True)
    test = feats.loc[is_test].reset_index(drop=True)
    del feats
    if max_train_year is not None:
        train = train.loc[train["year"] <= max_train_year].reset_index(drop=True)

    targets, summary = build_targets(line, train, dev_inputs, cfg, target_year, cap_quantile,
                                     fit_years=fit_years)
    tcols = [c for c in targets.columns
             if (c.startswith(TARGET_PREFIXES) or c in TARGET_EXTRA) and c not in train.columns]
    train = train.merge(targets[["row_id", *tcols]], on="row_id", how="left", validate="one_to_one")

    # Sums insured in target-year money (claims are on-levelled to target_year).
    si_rate = float((cfg.section("features").get("si_index", {}) or {}).get(line, 0.0))
    train = index_sum_insured(train, target_year, si_rate)
    test = index_sum_insured(test, target_year, si_rate)

    spec = feature_spec(line, cfg, available=train.columns)
    if review_features:
        spec = spec.with_review(review_features)
    # High-cardinality factors -> out-of-fold credibility relativities (numeric).
    for col in target_encode or []:
        k = float(cfg.section("features").get("credibility_k", 100.0))
        fit_mask = (train["year"] <= encode_fit_max_year).to_numpy() if encode_fit_max_year else None
        if fit_mask is None or fit_mask.all():
            enc_tr, (enc_te,) = credibility_encode(train, [test], col, k=k)
            train[f"{col}_te"], test[f"{col}_te"] = enc_tr, enc_te
        else:  # back-test: encode the holdout year from training years only
            fit = train.loc[fit_mask]
            enc_fit, (enc_rest, enc_te) = credibility_encode(fit, [train.loc[~fit_mask], test], col, k=k)
            train[f"{col}_te"] = 0.0
            train.loc[fit_mask, f"{col}_te"] = enc_fit
            train.loc[~fit_mask, f"{col}_te"] = enc_rest
            test[f"{col}_te"] = enc_te
        spec = FeatureSpec(line=spec.line, numeric=spec.numeric + (f"{col}_te",),
                           categorical=tuple(c for c in spec.categorical if c != col),
                           monotone=spec.monotone, review_numeric=spec.review_numeric,
                           review_categorical=spec.review_categorical, excluded=spec.excluded)
    min_expo = float(cfg.section("features").get("rare_level_min_exposure", 200))
    enc = CategoryEncoder(min_exposure=min_expo).fit(train, list(spec.categorical))
    train, test = enc.transform(train), enc.transform(test)
    return LineData(line, train, test, spec, enc, summary)


def gbm_from_config(spec: FeatureSpec, cfg: Config, overrides: dict[str, Any] | None = None,
                    use_monotone: bool = True) -> TweedieGBM:
    params = dict(cfg.section("model").get("gbm", {}))
    params.update(overrides or {})
    rounds = int(params.pop("num_boost_round", 3000))
    es = int(params.pop("early_stopping_rounds", 150))
    feats = spec.features
    return TweedieGBM(features=feats, categorical=list(spec.categorical), params=params,
                      monotone=spec.monotone_vector(feats) if use_monotone else None,
                      num_boost_round=rounds, early_stopping_rounds=es)


def glm_from_config(spec: FeatureSpec, cfg: Config, power: float = 1.5) -> TweedieGLM:
    g = cfg.section("model").get("glm", {})
    return TweedieGLM(numeric=list(spec.numeric), categorical=list(spec.categorical), power=power,
                      alpha=float(g.get("alpha", 5e-4)), n_bins=int(g.get("n_bins", 10)))


def score(y: np.ndarray, pred: np.ndarray, exposure: np.ndarray, clusters: pd.Series | None = None,
          power: float = 1.5, n_boot: int = 0, seed: int = 0) -> dict[str, float]:
    out = {
        "d2": mt.d2_tweedie(y, pred, power),
        "d2_rate_baseline": mt.d2_tweedie(y, pred, power, baseline=mt.constant_rate_baseline(y, exposure)),
        "gini": mt.normalized_gini(y, pred, exposure),
        "ae": float(np.sum(y) / np.sum(pred)),
        "mean_dev": mt.mean_tweedie_deviance(y, pred, power),
    }
    if n_boot and clusters is not None:
        b = mt.bootstrap_d2(y, pred, clusters, power, n_boot=n_boot, seed=seed)
        out.update({"d2_ci_low": b["ci_low"], "d2_ci_high": b["ci_high"], "d2_se": b["boot_se"]})
    return out


@dataclass
class BacktestResult:
    line: str
    holdout_year: int
    scores: dict[str, dict[str, float]]
    predictions: pd.DataFrame                # row-level holdout predictions (reported basis)
    info: dict[str, Any]
    curve: pd.DataFrame | None = None        # GBM holdout D^2 by number of trees
    models: dict[str, Any] = field(default_factory=dict, repr=False)


def balance_factor(y: np.ndarray, pred: np.ndarray) -> float:
    """Multiplicative factor that makes predictions sum to the actual total on training data.

    A Tweedie model with p > 1 and log link is calibrated in a mu^(1-p)-weighted
    sense, not in total, so the portfolio total drifts as trees are added
    (observed 3-25%). Re-balancing on the training data is standard practice.
    """
    return float(np.sum(y) / np.sum(pred))


def _fit_gbm(spec, cfg, tr, ycol, overrides, use_monotone, n_trees, es_share, seed, salt="es"):
    gbm = gbm_from_config(spec, cfg, overrides, use_monotone)
    if n_trees:
        gbm.fit(tr, tr[ycol].to_numpy(), tr["log_offset"].to_numpy(), num_boost_round=int(n_trees),
                seed=seed)
    else:
        es = policy_sample_mask(tr["policy_id"], es_share, salt=salt)
        gbm.fit(tr.loc[~es], tr.loc[~es, ycol].to_numpy(), tr.loc[~es, "log_offset"].to_numpy(),
                tr.loc[es], tr.loc[es, ycol].to_numpy(), tr.loc[es, "log_offset"].to_numpy(), seed=seed)
    return gbm


def run_backtest(
    line: str,
    cfg: Config,
    dev_inputs: DevelopmentInputs,
    holdout_year: int,
    cap_quantile: float | None = 0.995,
    gbm_overrides: dict[str, Any] | None = None,
    review_features: list[str] | None = None,
    drop_features: list[str] | None = None,
    fit_glm: bool = True,
    use_monotone: bool = True,
    n_trees: int | None = None,
    checkpoints: tuple[int, ...] = (),
    es_share: float = 0.1,
    n_boot: int = 0,
    data: LineData | None = None,
    target_encode: list[str] | None = None,
) -> BacktestResult:
    """Temporal back-test for one line: train on years < H, score on H (reported basis).

    With ``n_trees`` the GBM is trained for a fixed number of rounds (and scored at
    ``checkpoints`` to give a learning curve); otherwise it early-stops on a 10%
    policy-grouped sample of the training years.
    """
    t0 = time.time()
    if data is None:
        data = load_line_data(line, cfg, dev_inputs, target_year=holdout_year,
                              cap_quantile=cap_quantile, review_features=review_features,
                              max_train_year=holdout_year,
                              fit_years=[y for y in cfg.train_years if y < holdout_year],
                              target_encode=target_encode, encode_fit_max_year=holdout_year - 1)
    tr = data.train.loc[data.train["year"] < holdout_year]
    ho = data.train.loc[data.train["year"] == holdout_year]
    if tr.empty or ho.empty:
        raise ValueError(f"{line}: empty train or holdout for {holdout_year}")
    spec = data.spec
    if drop_features:
        spec = spec.restrict_to([c for c in spec.features if c not in set(drop_features)])
    load = float(data.target_summary["large_loss_load"])
    ycol = "y_capped" if cap_quantile is not None else "y"
    ytr = tr[ycol].to_numpy()

    gbm = _fit_gbm(spec, cfg, tr, ycol, gbm_overrides, use_monotone, n_trees, es_share, cfg.seed)
    bal_gbm = balance_factor(ytr, gbm.predict(tr, tr["log_offset"].to_numpy()))

    # Holdout: ultimate prediction at H's cost level (freq_trend = 1 in year H) ...
    off_ult = np.log(ho["exposure"].to_numpy() * ho["freq_trend"].to_numpy())
    comp = ho["comp_line"].to_numpy()
    y = ho["y"].to_numpy()
    e = ho["exposure"].to_numpy()
    preds = {"gbm": gbm.predict(ho, off_ult) * load * bal_gbm}
    models: dict[str, Any] = {"gbm": gbm}
    balances = {"gbm": bal_gbm}
    if fit_glm:
        glm = glm_from_config(spec, cfg).fit(tr, ytr, tr["log_offset"].to_numpy())
        balances["glm"] = balance_factor(ytr, glm.predict(tr, tr["log_offset"].to_numpy()))
        preds["glm"] = glm.predict(ho, off_ult) * load * balances["glm"]
        models["glm"] = glm
    # ... scored against reported claims: multiply by completeness.
    scores = {k: score(y, p * comp, e, ho["policy_id"], cfg.tweedie_power, n_boot, cfg.seed)
              for k, p in preds.items()}

    curve = None
    if n_trees and checkpoints:
        rows = []
        for k in sorted(c for c in checkpoints if c <= n_trees):
            b_k = balance_factor(ytr, gbm.predict(tr, tr["log_offset"].to_numpy(), num_iteration=k))
            p_k = gbm.predict(ho, off_ult, num_iteration=k) * load * b_k
            rows.append({"n_trees": k, "balance": b_k, **score(y, p_k * comp, e, power=cfg.tweedie_power)})
        curve = pd.DataFrame(rows)

    pred_df = pd.DataFrame({"row_id": ho["row_id"].to_numpy(), "policy_id": ho["policy_id"].to_numpy(),
                            "line": line, "year": holdout_year, "exposure": e, "comp": comp, "y": y,
                            **{f"pred_{k}": p for k, p in preds.items()}})
    info = {"n_train": int(len(tr)), "n_holdout": int(len(ho)), "n_trees": gbm.best_iteration_,
            "cap_quantile": cap_quantile, "large_loss_load": load, "balance": balances,
            "seconds": round(time.time() - t0, 1), "n_features": len(spec.features)}
    log.info("%s H=%d: %s | trees=%s", line, holdout_year,
             {k: round(v["d2"], 4) for k, v in scores.items()}, gbm.best_iteration_)
    return BacktestResult(line, holdout_year, scores, pred_df, info, curve, models)


def fit_final(
    line: str,
    cfg: Config,
    dev_inputs: DevelopmentInputs,
    cap_quantile: float | None = 0.995,
    gbm_overrides: dict[str, Any] | None = None,
    review_features: list[str] | None = None,
    drop_features: list[str] | None = None,
    fit_glm: bool = True,
    use_monotone: bool = True,
    n_trees: int | None = None,
    es_share: float = 0.1,
    data: LineData | None = None,
    n_seeds: int = 1,
    target_encode: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Fit on all training years at the prediction-year cost level; predict the test rows.

    Several seeds (different bagging samples) are averaged when ``n_seeds > 1``; each
    model is re-balanced on the full training data before averaging.
    """
    T = cfg.prediction_year
    if data is None:
        data = load_line_data(line, cfg, dev_inputs, target_year=T, cap_quantile=cap_quantile,
                              review_features=review_features, target_encode=target_encode)
    tr, te = data.train, data.test
    spec = data.spec
    if drop_features:
        spec = spec.restrict_to([c for c in spec.features if c not in set(drop_features)])
    load = float(data.target_summary["large_loss_load"])
    ycol = "y_capped" if cap_quantile is not None else "y"
    ytr = tr[ycol].to_numpy()
    off_test = np.log(te["exposure"].to_numpy())

    preds: dict[str, np.ndarray] = {}
    models: dict[str, Any] = {}
    balances: dict[str, float] = {}
    gbm_preds = []
    for k in range(n_seeds):
        gbm = _fit_gbm(spec, cfg, tr, ycol, gbm_overrides, use_monotone, n_trees, es_share,
                       cfg.seed + k, salt=f"final{k}")
        b = balance_factor(ytr, gbm.predict(tr, tr["log_offset"].to_numpy()))
        gbm_preds.append(gbm.predict(te, off_test) * load * b)
        models[f"gbm_{k}"], balances[f"gbm_{k}"] = gbm, b
    preds["gbm"] = np.mean(gbm_preds, axis=0)
    if fit_glm:
        glm = glm_from_config(spec, cfg).fit(tr, ytr, tr["log_offset"].to_numpy())
        balances["glm"] = balance_factor(ytr, glm.predict(tr, tr["log_offset"].to_numpy()))
        preds["glm"] = glm.predict(te, off_test) * load * balances["glm"]
        models["glm"] = glm
    out = pd.DataFrame({"row_id": te["row_id"].to_numpy(), "line": line,
                        "exposure": te["exposure"].to_numpy(),
                        "written_premium": te["written_premium"].to_numpy(),
                        **{f"pred_{k}": p for k, p in preds.items()}})
    info = {"n_train": int(len(tr)), "n_test": int(len(te)), "large_loss_load": load,
            "balance": balances,
            "n_trees": [m.best_iteration_ for k, m in models.items() if k.startswith("gbm")],
            "target_summary": data.target_summary}
    return out, info, models
