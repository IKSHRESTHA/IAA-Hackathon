"""Claim-size mixture of experts and policy-segment sub-models.

1. ``run_severity_moe`` ("cluster the claims, then model each cluster"):
   * the claims of a line are clustered by size with a Gaussian mixture on log(cost)
     (components ordered small -> large: attritional, medium, large);
   * a gating network (LightGBM multiclass) learns pi_k(x) = P(claim falls in size cluster k
     | policy features) - the "whale detector" for large claims;
   * one severity expert per cluster (Gamma GBM on that cluster's claims) learns E[cost | k, x];
   * severity(x) = sum_k pi_k(x) E[cost | k, x]  (Fung, Badescu & Lin's mixture-of-experts
     severity, LRMoE; and the expert-routing idea of Li et al. 2023 for lifetime value).
   Multiplied by the claim-frequency model it gives the ``fs_moe`` expert.

2. ``run_segment_models`` ("cluster the policies, then model each cluster"):
   policies are clustered with k-means on standardised key rating factors; an independent
   Tweedie GBM is trained per segment and each row is predicted by its segment's model.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture

from .config import Config
from .experts import FitSet, _fit_es
from .modeling import LineData, balance_factor, gbm_from_config, policy_sample_mask
from .targets import DevelopmentInputs, claim_level_frame

log = logging.getLogger(__name__)


def size_clusters(log_cost: np.ndarray, k: int, seed: int) -> tuple[GaussianMixture, np.ndarray]:
    """Gaussian mixture on log claim cost; labels re-ordered so 0 = smallest component."""
    gm = GaussianMixture(n_components=k, random_state=seed, n_init=3).fit(log_cost.reshape(-1, 1))
    order = np.argsort(gm.means_.ravel())
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(k)
    return gm, remap[gm.predict(log_cost.reshape(-1, 1))]


def run_severity_moe(data: LineData, fs: FitSet, cfg: Config, dev_inputs: DevelopmentInputs,
                     target_year: int, k: int = 3, cap_quantile: float = 0.999) -> tuple[pd.DataFrame, dict]:
    t0 = time.time()
    line, spec, tr, sc = data.line, data.spec, fs.train, fs.score
    seed = cfg.seed
    cl = claim_level_frame(line, tr[["row_id", "policy_id", "period_start", "period_end", "exposure"]],
                           dev_inputs, cfg, target_year)
    cl = cl.loc[cl["loss_year"] < cfg.valuation_date.year]          # mature severities only
    feats = spec.features
    X = cl[["row_id", "cost_T"]].merge(tr[["row_id", "policy_id", *feats]], on="row_id", how="left")
    cap = float(np.quantile(X["cost_T"], cap_quantile))
    y = np.minimum(X["cost_T"].to_numpy(), cap)
    tail_load = float(X["cost_T"].sum() / y.sum())                   # excess above the cap, reloaded

    gm, lab = size_clusters(np.log(y), k, seed)
    info: dict[str, Any] = {"line": line, "set": fs.name, "k": k, "cap": cap, "tail_load": tail_load,
                            "cluster_share": np.bincount(lab, minlength=k).tolist(),
                            "cluster_mean": [float(y[lab == j].mean()) for j in range(k)]}

    # Gating network: P(size cluster | policy features)
    es = policy_sample_mask(X["policy_id"], 0.2, salt="gate")
    p_gate = {"objective": "multiclass", "num_class": k, "metric": "multi_logloss", "learning_rate": 0.05,
              "num_leaves": 15, "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8,
              "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1, "num_threads": 2, "seed": seed,
              "max_cat_to_onehot": 32, "cat_smooth": 50}
    cats = list(spec.categorical)
    dtr = lgb.Dataset(X.loc[~es, feats], label=lab[~es], categorical_feature=cats)
    des = lgb.Dataset(X.loc[es, feats], label=lab[es], categorical_feature=cats, reference=dtr)
    gate = lgb.train(p_gate, dtr, num_boost_round=2000, valid_sets=[des],
                     callbacks=[lgb.early_stopping(100, verbose=False)])
    n_gate = int(gate.best_iteration or 100)
    if fs.is_final:
        gate = lgb.train(p_gate, lgb.Dataset(X[feats], label=lab, categorical_feature=cats),
                         num_boost_round=max(int(n_gate * 1.1), 20))
    pi = gate.predict(sc[feats], num_iteration=None if fs.is_final else n_gate)   # (n, k)
    info["gate_trees"] = n_gate
    info["gate_base_rates"] = np.bincount(lab, minlength=k) / len(lab)
    info["gate_mean_pred"] = pi.mean(axis=0)

    # One severity expert per size cluster
    sev_k = np.zeros((len(sc), k))
    sev_k_tr = np.zeros((len(X), k))
    info["experts"] = {}
    for j in range(k):
        m = lab == j
        yj = y[m]
        if m.sum() >= 500:
            ov = {"objective": "gamma", "metric": "gamma", "num_leaves": 7,
                  "min_data_in_leaf": max(100, int(m.sum()) // 50), "learning_rate": 0.03}
            ej = _fit_es(spec, cfg, X.loc[m].reset_index(drop=True), yj, np.zeros(int(m.sum())), ov,
                         refit=fs.is_final, seed=seed, use_monotone=False)
            bj = balance_factor(yj, ej.predict(X.loc[m], np.zeros(int(m.sum()))))
            sev_k[:, j] = ej.predict(sc, np.zeros(len(sc))) * bj
            sev_k_tr[:, j] = ej.predict(X, np.zeros(len(X))) * bj
            info["experts"][j] = {"n": int(m.sum()), "trees": ej.best_iteration_, "balance": bj}
        else:
            sev_k[:, j] = sev_k_tr[:, j] = yj.mean()
            info["experts"][j] = {"n": int(m.sum()), "constant": float(yj.mean())}
    pi_tr = gate.predict(X[feats], num_iteration=None if fs.is_final else n_gate)
    calib = float(y.sum() / (pi_tr * sev_k_tr).sum())     # mixture calibrated to the mean claim size
    sev = (pi * sev_k).sum(axis=1) * calib
    info["calibration"] = calib
    out = pd.DataFrame({"row_id": sc["row_id"].to_numpy(), "sev_moe": sev * tail_load})
    for j in range(k):
        out[f"pi_size{j}"] = pi[:, j]
    info["seconds"] = round(time.time() - t0, 1)
    log.info("%s %s severity MoE done in %.0fs", line, fs.name, info["seconds"])
    return out, info


def segment_features(df: pd.DataFrame, line: str) -> list[str]:
    """Key rating factors used to cluster policies (numeric, interpretable)."""
    common = ["exposure", "prior_claims_5y", "credit_band_ord", "hist_pol_n_3y", "total_excess"]
    by_line = {
        "motor": ["driver_age", "years_licensed", "ncb_years", "vehicle_value_group", "vehicle_age",
                  "annual_mileage", "make_model_te"],
        "buildings": ["sum_insured", "flood_zone", "elevation_m", "distance_to_river_km", "property_age",
                      "bedrooms", "policyholder_age"],
        "contents": ["sum_insured", "flood_zone", "elevation_m", "policyholder_age", "high_value_items",
                     "security_level_ord"],
    }
    return [c for c in common + by_line[line] if c in df.columns]


def run_segment_models(data: LineData, fs: FitSet, cfg: Config, n_segments: int = 4) -> tuple[pd.DataFrame, dict]:
    """K-means policy segments, one Tweedie GBM per segment, hard routing (``seg_moe``)."""
    t0 = time.time()
    line, spec, tr, sc = data.line, data.spec, fs.train, fs.score
    cols = segment_features(tr, line)
    Z = tr[cols].astype(float)
    med, sd = Z.median(), Z.std().replace(0, 1.0)
    def zs(df):
        return ((df[cols].astype(float) - med) / sd).fillna(0.0).clip(-5, 5).to_numpy()
    km = MiniBatchKMeans(n_clusters=n_segments, random_state=cfg.seed, n_init=5, batch_size=4096).fit(zs(tr))
    seg_tr, seg_sc = km.predict(zs(tr)), km.predict(zs(sc))
    load = float(data.target_summary["large_loss_load"])
    trend_sc = sc["freq_trend"].to_numpy() if "freq_trend" in sc else np.ones(len(sc))
    off_sc = np.log(sc["exposure"].to_numpy() * trend_sc)
    pred = np.zeros(len(sc))
    info: dict[str, Any] = {"line": line, "set": fs.name, "features": cols, "segments": {}}
    for s in range(n_segments):
        m_tr, m_sc = seg_tr == s, seg_sc == s
        trs = tr.loc[m_tr].reset_index(drop=True)
        y, off = trs["y_capped"].to_numpy(), trs["log_offset"].to_numpy()
        model = _fit_es(spec, cfg, trs, y, off, {}, refit=fs.is_final, seed=cfg.seed)
        b = balance_factor(y, model.predict(trs, off))
        if m_sc.any():
            pred[m_sc] = model.predict(sc.loc[m_sc], off_sc[m_sc]) * b * load
        info["segments"][s] = {"n_train": int(m_tr.sum()), "n_score": int(m_sc.sum()),
                               "trees": model.best_iteration_, "balance": b,
                               "pp_train": float(y.sum() / trs["exposure"].sum())}
    out = pd.DataFrame({"row_id": sc["row_id"].to_numpy(), "seg_moe": pred, "segment": seg_sc})
    info["seconds"] = round(time.time() - t0, 1)
    log.info("%s %s segment models done in %.0fs", line, fs.name, info["seconds"])
    return out, info
