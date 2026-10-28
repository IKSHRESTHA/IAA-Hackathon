"""Mixture-of-experts components for heavy-tailed, zero-inflated claim costs.

Why experts: the 2024 back-test deviance decomposition showed that (i) more than half of
the achievable gain comes from ranking claim *frequency* (zero rows), (ii) peril-specific
drivers (escape of water, storm, bodily injury) were poorly captured by one total-cost
model, and (iii) a uniform large-loss load cannot tell which risks produce large claims.
Each expert therefore targets one part of the loss process, and a meta-learner
(:mod:`iaa_pricing.stacking`) combines them on out-of-time predictions.

Experts (all LightGBM with an exact log-offset, re-balanced on training data):
* ``total``   Tweedie on capped total cost (existing champion) x uniform large-loss load;
* ``glm``     banded Tweedie GLM (benchmark, very different inductive bias);
* ``peril``   one Tweedie model per peril on capped peril cost, offset with the peril's
              own IBNR completeness, x peril large-loss load, summed over perils;
* ``freq``    Poisson model of claim counts *without* claims-history features: the
              a-priori frequency used for Bayesian credibility (experience rating);
* ``large``   Poisson model of large-claim counts (claims above the cap) x mean excess:
              a risk-specific large-loss expert ("whale detection") that can replace the
              uniform load.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import PERILS, Config
from .features import FeatureSpec
from .modeling import (LineData, balance_factor, gbm_from_config, glm_from_config,
                       policy_sample_mask)
from .models.gbm import TweedieGBM

log = logging.getLogger(__name__)

HISTORY_PREFIX = "hist_"


@dataclass
class FitSet:
    """One training/scoring split: a back-test year (score = holdout year) or the final fit."""

    name: str                 # "ho2023", "ho2024" or "final"
    train: pd.DataFrame
    score: pd.DataFrame       # rows to predict (holdout rows or 2025 test rows)
    is_final: bool


def _fit_es(spec: FeatureSpec, cfg: Config, tr: pd.DataFrame, y: np.ndarray, offset: np.ndarray,
            overrides: dict[str, Any], refit: bool, seed: int, es_share: float = 0.2,
            max_rounds: int = 2000, patience: int = 200, use_monotone: bool = True) -> TweedieGBM:
    """Early-stop on a policy-grouped sample; optionally refit on all rows with 1.1x trees."""
    m = gbm_from_config(spec, cfg, overrides, use_monotone)
    m.num_boost_round, m.early_stopping_rounds = max_rounds, patience
    es = policy_sample_mask(tr["policy_id"], es_share, salt=f"es{seed}")
    m.fit(tr.loc[~es], y[~es], offset[~es], tr.loc[es], y[es], offset[es], seed=seed)
    if refit:
        n = max(int(round(m.best_iteration_ * 1.1)), 20)
        m2 = gbm_from_config(spec, cfg, overrides, use_monotone)
        m2.fit(tr, y, offset, num_boost_round=n, seed=seed)
        return m2
    return m


def _no_history(spec: FeatureSpec) -> FeatureSpec:
    keep = [f for f in spec.features if not f.startswith(HISTORY_PREFIX)]
    return spec.restrict_to(keep)


@dataclass
class ExpertResult:
    predictions: pd.DataFrame
    info: dict[str, Any] = field(default_factory=dict)


def run_experts(data: LineData, fs: FitSet, cfg: Config, total_trees: int | None,
                overrides: dict[str, Any] | None = None, fit_glm: bool = True) -> ExpertResult:
    """Fit every expert on ``fs.train`` and predict ``fs.score`` (ultimate basis, cost level T).

    Returned columns: identifiers, exposure, completeness (holdout rows), the actual ``y``
    (holdout rows) and one column per expert. ``total_cap`` is the capped total-cost
    prediction before any large-loss load, so it can be combined with ``large``.
    """
    t0 = time.time()
    line, spec, tr, sc = data.line, data.spec, fs.train, fs.score
    ov = dict(overrides or {})
    perils = list(PERILS[line])
    s = data.target_summary
    load = float(s["large_loss_load"])
    seed = cfg.seed
    off_tr = tr["log_offset"].to_numpy()
    # Scoring offset: ultimate basis at the cost level of the scored year (freq_trend = 1 there).
    trend_sc = sc["freq_trend"].to_numpy() if "freq_trend" in sc else np.ones(len(sc))
    off_sc = np.log(sc["exposure"].to_numpy() * trend_sc)
    out = pd.DataFrame({"row_id": sc["row_id"].to_numpy(), "policy_id": sc["policy_id"].to_numpy(),
                        "customer_id": sc["customer_id"].to_numpy(), "line": line,
                        "year": sc["year"].to_numpy(), "set": fs.name,
                        "exposure": sc["exposure"].to_numpy()})
    for c in ("y", "y_capped", "n_claims", "comp_line", "n_large", "y_excess"):
        out[c] = sc[c].to_numpy() if c in sc else np.nan
    for p in perils:
        out[f"comp_{p}"] = sc[f"comp_{p}"].to_numpy() if f"comp_{p}" in sc else 1.0
    for c in ("hist_pol_n_1y", "hist_pol_n_3y", "hist_pol_expo_3y", "hist_cust_n_3y",
              "hist_cust_n_other_3y", "hist_pol_cost_3y"):
        if c in sc:
            out[c] = sc[c].to_numpy()
    info: dict[str, Any] = {"line": line, "set": fs.name, "n_train": len(tr), "n_score": len(sc)}

    # 1) Total-cost Tweedie (champion) -------------------------------------------------
    y = tr["y_capped"].to_numpy()
    if total_trees:
        m = gbm_from_config(spec, cfg, ov)
        m.fit(tr, y, off_tr, num_boost_round=int(total_trees), seed=seed)
    else:
        m = _fit_es(spec, cfg, tr, y, off_tr, ov, refit=fs.is_final, seed=seed)
    b = balance_factor(y, m.predict(tr, off_tr))
    out["total_cap"] = m.predict(sc, off_sc) * b
    out["total"] = out["total_cap"] * load
    info["total"] = {"trees": m.best_iteration_, "balance": b, "load": load}

    # 2) GLM benchmark -----------------------------------------------------------------
    if fit_glm:
        g = glm_from_config(spec, cfg).fit(tr, y, off_tr)
        bg = balance_factor(y, g.predict(tr, off_tr))
        out["glm"] = g.predict(sc, off_sc) * bg * load
        info["glm"] = {"balance": bg}

    # 3) Peril experts (capped peril cost, peril-specific IBNR completeness) -------------
    peril_loads = s.get("peril_loads", {})
    out["peril_cap"] = 0.0
    out["peril"] = 0.0
    out["peril_rep"] = 0.0
    info["peril"] = {}
    for p in perils:
        yp = tr[f"yc_{p}"].to_numpy()
        if yp.sum() <= 0:
            continue
        off_p = np.log(tr["exposure"].to_numpy() * tr[f"comp_{p}"].to_numpy() * tr["freq_trend"].to_numpy())
        mp = _fit_es(spec, cfg, tr, yp, off_p, ov, refit=fs.is_final, seed=seed)
        bp = balance_factor(yp, mp.predict(tr, off_p))
        pred_p = mp.predict(sc, off_sc) * bp
        lp = float(peril_loads.get(p, 1.0))
        out[f"peril_{p}"] = pred_p * lp
        out["peril_cap"] += pred_p
        out["peril"] += pred_p * lp
        out["peril_rep"] += pred_p * lp * out[f"comp_{p}"].to_numpy()
        info["peril"][p] = {"trees": mp.best_iteration_, "balance": bp, "load": lp}

    # 4) A-priori frequency (no claims-history features) for credibility ---------------
    spec_nh = _no_history(spec)
    compn = np.column_stack([tr[f"compn_{p}"].to_numpy() if f"compn_{p}" in tr else np.ones(len(tr))
                             for p in perils])
    nshare = np.array([max(tr[f"n_{p}"].sum(), 1.0) for p in perils])
    compn_line = compn @ (nshare / nshare.sum())
    off_f = np.log(tr["exposure"].to_numpy() * compn_line * tr["freq_trend"].to_numpy())
    yf = tr["n_claims"].to_numpy().astype(float)
    mf = _fit_es(spec_nh, cfg, tr, yf, off_f, {**ov, "objective": "poisson", "metric": "poisson"},
                 refit=fs.is_final, seed=seed)
    bf = balance_factor(yf, mf.predict(tr, off_f))
    out["freq_rate"] = mf.predict(sc, np.zeros(len(sc))) * bf    # claims per exposure-year
    info["freq"] = {"trees": mf.best_iteration_, "balance": bf}

    # 5) Large-loss expert: count of claims above the cap x mean excess -----------------
    yl = tr["n_large"].to_numpy().astype(float)
    mean_excess = float(s.get("mean_excess", 0.0))
    if yl.sum() >= 30 and mean_excess > 0:
        ov_l = {**ov, "objective": "poisson", "metric": "poisson", "num_leaves": 7,
                "min_data_in_leaf": 5000, "learning_rate": 0.02, "lambda_l2": 50.0}
        ml = _fit_es(spec_nh, cfg, tr, yl, off_tr, ov_l, refit=fs.is_final, seed=seed, patience=300)
        bl = balance_factor(yl, ml.predict(tr, off_tr))
        out["large"] = ml.predict(sc, off_sc) * bl * mean_excess
        info["large"] = {"trees": ml.best_iteration_, "balance": bl, "mean_excess": mean_excess,
                         "n_large": int(yl.sum())}
    else:
        out["large"] = out["total_cap"] * (load - 1.0)
        info["large"] = {"fallback": "uniform load", "n_large": int(yl.sum())}
    out["total_plus_large"] = out["total_cap"] + out["large"]
    out["peril_plus_large"] = out["peril_cap"] + out["large"]
    info["seconds"] = round(time.time() - t0, 1)
    log.info("%s %s experts done in %.0fs", line, fs.name, info["seconds"])
    return ExpertResult(out, info)
