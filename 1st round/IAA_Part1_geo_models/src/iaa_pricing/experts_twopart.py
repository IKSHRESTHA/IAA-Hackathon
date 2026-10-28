"""Two-part (frequency x severity) experts, including a heteroscedastic lognormal severity.

* ``fs_gamma``:   Poisson frequency (all features incl. claims history) x Gamma severity
                  GBM on capped claim cost, x line large-loss load;
* ``fs_lognorm``: same frequency x lognormal severity with *risk-specific tail*: one GBM
                  for the mean of log cost, m(x), and one for the residual variance, s2(x),
                  so E[cost|x] = exp(m(x) + s2(x)/2). Uncapped: risks with heavier tails get
                  a higher mean. This is the zero-inflated-lognormal idea from customer
                  lifetime-value modelling (Wang et al., 2019) applied to claims.
Severity models are fitted on claim records (row features joined), so they learn the size
of claims without being swamped by the ~90% of rows that have no claim.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from .config import PERILS, Config
from .experts import FitSet, _fit_es
from .modeling import LineData, balance_factor, gbm_from_config, policy_sample_mask
from .targets import DevelopmentInputs, claim_level_frame

log = logging.getLogger(__name__)


def _count_offset(df: pd.DataFrame, line: str) -> np.ndarray:
    perils = list(PERILS[line])
    compn = np.column_stack([df[f"compn_{p}"].to_numpy() if f"compn_{p}" in df else np.ones(len(df))
                             for p in perils])
    w = np.array([max(df[f"n_{p}"].sum(), 1.0) if f"n_{p}" in df else 1.0 for p in perils])
    return np.log(df["exposure"].to_numpy() * (compn @ (w / w.sum())) * df["freq_trend"].to_numpy())


def run_twopart(data: LineData, fs: FitSet, cfg: Config, dev_inputs: DevelopmentInputs,
                target_year: int, overrides: dict[str, Any] | None = None) -> tuple[pd.DataFrame, dict]:
    t0 = time.time()
    line, spec, tr, sc = data.line, data.spec, fs.train, fs.score
    ov = dict(overrides or {})
    seed = cfg.seed
    s = data.target_summary
    cap = float(s["cap"])
    trend_sc = sc["freq_trend"].to_numpy() if "freq_trend" in sc else np.ones(len(sc))
    off_sc = np.log(sc["exposure"].to_numpy() * trend_sc)
    out = pd.DataFrame({"row_id": sc["row_id"].to_numpy()})
    info: dict[str, Any] = {"line": line, "set": fs.name}

    # Frequency with claims history ---------------------------------------------------
    off_f = _count_offset(tr, line)
    yf = tr["n_claims"].to_numpy(dtype=float)
    mf = _fit_es(spec, cfg, tr, yf, off_f, {**ov, "objective": "poisson", "metric": "poisson"},
                 refit=fs.is_final, seed=seed)
    bf = balance_factor(yf, mf.predict(tr, off_f))
    freq = mf.predict(sc, off_sc) * bf
    out["freq_hist"] = freq
    info["freq_hist"] = {"trees": mf.best_iteration_, "balance": bf}

    # Claim-level severity data (training rows only) ---------------------------------
    cl = claim_level_frame(line, tr[["row_id", "policy_id", "period_start", "period_end", "exposure"]],
                           dev_inputs, cfg, target_year)
    # Claims of the valuation year are immature: late-reported (bodily injury) claims are both
    # missing and larger, so reported severity is biased low. Severity is therefore learnt
    # from mature loss years only (this only affects the final fit; back-test training
    # years are all before the valuation year).
    cl = cl.loc[cl["loss_year"] < cfg.valuation_date.year]
    info["severity_claims_years"] = sorted(int(y) for y in cl["loss_year"].unique())
    feats = spec.features
    X = cl[["row_id", "cost_T"]].merge(tr[["row_id", "policy_id", *feats]], on="row_id", how="left")
    y_cap = np.minimum(X["cost_T"].to_numpy(), cap)
    zero_off = np.zeros(len(X))
    sev_ov = {**ov, "min_data_in_leaf": 300, "num_leaves": 15, "learning_rate": 0.03}

    # Gamma severity on capped cost
    mg = _fit_es(spec, cfg, X, y_cap, zero_off, {**sev_ov, "objective": "gamma", "metric": "gamma"},
                 refit=fs.is_final, seed=seed, use_monotone=False)
    bg = balance_factor(y_cap, mg.predict(X, zero_off))
    sev_gamma = mg.predict(sc, np.zeros(len(sc))) * bg
    load = float(s["large_loss_load"])
    out["sev_gamma"] = sev_gamma
    out["fs_gamma"] = freq * sev_gamma * load
    info["sev_gamma"] = {"trees": mg.best_iteration_, "balance": bg, "n_claims": int(len(X))}

    # Heteroscedastic lognormal severity on uncapped cost (mean and variance of log cost)
    ly = np.log(X["cost_T"].to_numpy())
    mm = _fit_es(spec, cfg, X, ly, zero_off, {**sev_ov, "objective": "regression", "metric": "l2"},
                 refit=fs.is_final, seed=seed, use_monotone=False)
    # TweedieGBM.predict applies exp(); for this identity-link L2 model use raw score + base.
    mu_log_tr = mm.booster_.predict(X[mm.features], raw_score=True, num_iteration=mm.best_iteration_) + mm.base_
    resid2 = (ly - mu_log_tr) ** 2
    mv = _fit_es(spec, cfg, X, resid2, zero_off,
                 {**sev_ov, "objective": "gamma", "metric": "gamma", "num_leaves": 7, "min_data_in_leaf": 1000},
                 refit=fs.is_final, seed=seed, use_monotone=False)
    bv = balance_factor(resid2, mv.predict(X, zero_off))
    mu_log_sc = mm.booster_.predict(sc[mm.features], raw_score=True, num_iteration=mm.best_iteration_) + mm.base_
    var_sc = mv.predict(sc, np.zeros(len(sc))) * bv
    sev_ln = np.exp(mu_log_sc + var_sc / 2.0)
    # Calibrate the lognormal severity to the observed mean claim cost on training claims.
    sev_ln_tr = np.exp(mu_log_tr + mv.predict(X, zero_off) * bv / 2.0)
    bln = float(X["cost_T"].sum() / sev_ln_tr.sum())
    out["sev_lognorm"] = sev_ln * bln
    out["sev_sigma2"] = var_sc
    out["fs_lognorm"] = freq * sev_ln * bln
    info["sev_lognorm"] = {"trees_mean": mm.best_iteration_, "trees_var": mv.best_iteration_,
                           "calibration": bln, "mean_var": float(var_sc.mean())}
    info["seconds"] = round(time.time() - t0, 1)
    log.info("%s %s two-part experts done in %.0fs", line, fs.name, info["seconds"])
    return out, info


def run_twopart_peril(data: LineData, fs: FitSet, cfg: Config, dev_inputs: DevelopmentInputs,
                      target_year: int, overrides: dict[str, Any] | None = None) -> tuple[pd.DataFrame, dict]:
    """Per-peril frequency x severity ("fs_peril"): one Poisson count model and one Gamma
    severity model per cause of loss, each with its own IBNR completeness, summed.

    Rationale: the line-level two-part model was the strongest single expert; perils have
    distinct frequency drivers (e.g. flood zone vs freezing days) and very different claim
    sizes (bodily injury vs glass), so modelling each peril separately sharpens both parts.
    """
    t0 = time.time()
    line, spec, tr, sc = data.line, data.spec, fs.train, fs.score
    ov = dict(overrides or {})
    seed = cfg.seed
    s = data.target_summary
    cap = float(s["cap"])
    peril_loads = s.get("peril_loads", {})
    trend_sc = sc["freq_trend"].to_numpy() if "freq_trend" in sc else np.ones(len(sc))
    off_sc = np.log(sc["exposure"].to_numpy() * trend_sc)
    out = pd.DataFrame({"row_id": sc["row_id"].to_numpy()})
    out["fs_peril"] = 0.0
    out["fs_peril_rep"] = 0.0
    info: dict[str, Any] = {"line": line, "set": fs.name, "perils": {}}

    cl = claim_level_frame(line, tr[["row_id", "policy_id", "period_start", "period_end", "exposure"]],
                           dev_inputs, cfg, target_year)
    cl = cl.loc[cl["loss_year"] < cfg.valuation_date.year]          # mature severities only
    feats = spec.features
    for p in PERILS[line]:
        n_p = tr[f"n_{p}"].to_numpy(dtype=float)
        if n_p.sum() < 50:
            continue
        compn = tr[f"compn_{p}"].to_numpy() if f"compn_{p}" in tr else np.ones(len(tr))
        off_f = np.log(tr["exposure"].to_numpy() * compn * tr["freq_trend"].to_numpy())
        mf = _fit_es(spec, cfg, tr, n_p, off_f, {**ov, "objective": "poisson", "metric": "poisson"},
                     refit=fs.is_final, seed=seed)
        bf = balance_factor(n_p, mf.predict(tr, off_f))
        freq = mf.predict(sc, off_sc) * bf

        cp = cl.loc[cl["peril"] == p]
        X = cp[["row_id", "cost_T"]].merge(tr[["row_id", "policy_id", *feats]], on="row_id", how="left")
        y_cap = np.minimum(X["cost_T"].to_numpy(), cap)
        if len(X) >= 300:
            sev_ov = {**ov, "objective": "gamma", "metric": "gamma", "num_leaves": 7,
                      "min_data_in_leaf": max(100, min(1000, len(X) // 30)), "learning_rate": 0.03}
            ms = _fit_es(spec, cfg, X, y_cap, np.zeros(len(X)), sev_ov, refit=fs.is_final, seed=seed,
                         use_monotone=False)
            bs = balance_factor(y_cap, ms.predict(X, np.zeros(len(X))))
            sev = ms.predict(sc, np.zeros(len(sc))) * bs
            sev_trees = ms.best_iteration_
        else:  # too few claims for a severity model: peril mean
            sev, sev_trees = np.full(len(sc), float(y_cap.mean()) if len(y_cap) else 0.0), 0
        lp = float(peril_loads.get(p, 1.0))
        cost = freq * sev * lp
        out[f"fsp_{p}"] = cost
        out["fs_peril"] += cost
        comp_p = sc[f"comp_{p}"].to_numpy() if f"comp_{p}" in sc else np.ones(len(sc))
        out["fs_peril_rep"] += cost * comp_p
        info["perils"][p] = {"freq_trees": mf.best_iteration_, "sev_trees": sev_trees,
                             "n_claims_sev": int(len(X)), "load": lp}
    info["seconds"] = round(time.time() - t0, 1)
    log.info("%s %s per-peril two-part done in %.0fs", line, fs.name, info["seconds"])
    return out, info


def run_twopart_glm(data: LineData, fs: FitSet, cfg: Config, dev_inputs: DevelopmentInputs,
                    target_year: int) -> tuple[pd.DataFrame, dict]:
    """Classical actuarial two-part GLM ("fs_glm"): Poisson frequency GLM x Gamma severity
    GLM on banded rating factors. Transparent, and a different inductive bias from the
    tree experts, which is what an equal-weight ensemble needs."""
    from .models.glm import TweedieGLM
    t0 = time.time()
    line, spec, tr, sc = data.line, data.spec, fs.train, fs.score
    s = data.target_summary
    cap, load = float(s["cap"]), float(s["large_loss_load"])
    gcfg = cfg.section("model").get("glm", {})
    alpha, n_bins = float(gcfg.get("alpha", 5e-4)), int(gcfg.get("n_bins", 10))
    trend_sc = sc["freq_trend"].to_numpy() if "freq_trend" in sc else np.ones(len(sc))
    off_sc = np.log(sc["exposure"].to_numpy() * trend_sc)
    num, cat = list(spec.numeric), list(spec.categorical)

    off_f = _count_offset(tr, line)
    yf = tr["n_claims"].to_numpy(dtype=float)
    gf = TweedieGLM(num, cat, alpha=alpha, n_bins=n_bins, family="poisson").fit(tr, yf, off_f)
    bf = balance_factor(yf, gf.predict(tr, off_f))
    freq = gf.predict(sc, off_sc) * bf

    cl = claim_level_frame(line, tr[["row_id", "policy_id", "period_start", "period_end", "exposure"]],
                           dev_inputs, cfg, target_year)
    cl = cl.loc[cl["loss_year"] < cfg.valuation_date.year]
    X = cl[["row_id", "cost_T"]].merge(tr[["row_id", *num, *cat]], on="row_id", how="left")
    y_cap = np.minimum(X["cost_T"].to_numpy(), cap)
    gs = TweedieGLM(num, cat, alpha=alpha * 10, n_bins=n_bins, family="gamma").fit(X, y_cap, np.zeros(len(X)))
    bs = balance_factor(y_cap, gs.predict(X, np.zeros(len(X))))
    sev = gs.predict(sc, np.zeros(len(sc))) * bs
    out = pd.DataFrame({"row_id": sc["row_id"].to_numpy(), "fs_glm": freq * sev * load})
    info = {"line": line, "set": fs.name, "balance_freq": bf, "balance_sev": bs,
            "n_claims_sev": int(len(X)), "seconds": round(time.time() - t0, 1)}
    log.info("%s %s GLM two-part done in %.0fs", line, fs.name, info["seconds"])
    return out, info
