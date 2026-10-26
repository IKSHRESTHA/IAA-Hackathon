"""Step 6 - developed-market priors for every component of the ensemble's experts.

Step 4 fitted one prior: the expected total cost. The experts of the 1st round's winning
ensemble (``geo_experts.py``) also model claim frequency, claim size, and each peril on its own,
so each component gets a prior of its own here, fitted the same way as step 4: on every
1st-round row, with the factors that mean the same thing in both markets (step 4's choice,
ages as bands), at the 2025 cost level, with a fixed number of trees.

    freq          claims per year on cover (Poisson)
    cost_<peril>  capped cost of the peril per year on cover (Tweedie)
    freq_<peril>  claims of the peril per year on cover (Poisson)
    sev           size of a claim, capped (Gamma, claims of 2021-2023)
    sev_<peril>   size of a claim of the peril (Gamma)

The 1st-round targets are built as in its ensemble: open claims developed, claims on-levelled by
loss year, capped per claim. Output: ``data/prior/geo_priors_<line>.parquet`` (row_id and the log
prior of each component for every training and 2025 row of this round) and
``reports/06_geo/priors_<line>.json``.

Usage: python 06_geo_priors.py [--lines motor ...]
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys

import numpy as np
import pandas as pd

import config as C
import helpers as H

sys.path.insert(0, str(C.ROOT.parent))            # the shared modules sit in the folder above
import geo_experts as G  # noqa: E402
import pricing_lgbm as P  # noqa: E402

S4 = importlib.import_module("04_developed_prior")    # step 4: harmonised frames and shared features
C1 = S4.C1                                            # the 1st round's settings
OUT = C.REPORTS / "06_geo"
PRIOR_DIR = C.DATA / "prior"
DEV = C.DEVELOPED / "data"


def developed_claims() -> tuple[pd.DataFrame, dict]:
    """1st-round claims developed to ultimate, and the claim-count reporting pattern."""
    claims = pd.read_parquet(DEV / "clean" / "claims.parquet")
    dev = G.development_params(claims, C1.DEV_MATURE_YEARS, C1.DEV_IMMATURE_YEARS, C1.REPUDIATION_MIN_Z)
    nonrep = claims.loc[claims["status"] != "repudiated"]
    pattern = H.reporting_pattern(nonrep, C1.VALUATION_DATE, f"{min(C1.TRAIN_YEARS)}-01-01")
    return G.claims_to_ultimate(claims, dev), pattern


def developed_rows(line: str, claims_ult: pd.DataFrame, pattern: dict) -> tuple[pd.DataFrame, G.Targets]:
    """Harmonised 1st-round rows with their reported shares and targets at the 2025 cost level."""
    dev = S4.harmonise(*S4.read_market(C.DEVELOPED, line, ("policies",)), line, developed=True)
    shares = [f"comp_{p}" for p in C1.PERILS[line]]
    feats = pd.read_parquet(DEV / "features" / f"policies_{line}.parquet",
                            columns=["row_id", "comp_cost", "comp_count", *shares])     # step 3 shares
    dev = dev.merge(feats, on="row_id", how="inner", validate="one_to_one")
    for p in C1.PERILS[line]:                                   # claim-count share per peril
        dev[f"compn_{p}"] = H.row_completeness(dev, pattern[(line, p)], C1.VALUATION_DATE)
    tg = G.build_targets(dev, claims_ult, line, C1.PERILS[line], C1.TREND, C1.LEVEL_SHIFT, C.TEST_YEAR,
                         C1.CAP_QUANTILE, list(C1.TRAIN_YEARS))
    return dev.merge(tg.rows, on="row_id", how="left", validate="one_to_one"), tg


def fit_components(line: str, dev: pd.DataFrame, tg: G.Targets, features: list[str],
                   em: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict]:
    """One LightGBM per component on the developed rows or claims; its log prior for the ``em`` rows."""
    X, levels = P.prepare(dev, features)
    Xe, _ = P.prepare(em, features, levels)                     # this round's rows, same levels
    ft = G.frequency_trend(dev["year"], line, C1.TREND, C.TEST_YEAR)
    e = dev["exposure"].to_numpy(dtype=float)
    priors, info = {}, {}

    def fit(key: str, y, offset: np.ndarray, X_fit: pd.DataFrame, params: dict) -> None:
        y = np.asarray(y, dtype=float)
        m = P.TweedieLGBM({**C1.LGBM_PARAMS, **params}, features).fit(X_fit, y, offset, n_trees=C.PRIOR_TREES,
                                                                       seed=C.SEED)
        priors[key] = np.log(np.maximum(m.predict(Xe, np.zeros(len(Xe))), 1e-12))   # per year on cover
        info[key] = {"rows": int(len(y)), "target_total": float(y.sum())}

    def share_offset(share: str) -> np.ndarray:                 # years on cover x reported share / trend
        return np.log(e * P.completeness(dev, share) / ft)

    fit("freq", dev["n_claims"], share_offset("comp_count"), X, {"objective": "poisson"})
    for p in C1.PERILS[line]:
        fit(f"cost_{p}", dev[f"yc_{p}"], share_offset(f"comp_{p}"), X, {})              # Tweedie
        fit(f"freq_{p}", dev[f"n_{p}"], share_offset(f"compn_{p}"), X, {"objective": "poisson"})

    # Claim sizes, from the claims of mature loss years (2024 claims are still young).
    pos = pd.Series(np.arange(len(dev)), index=dev["row_id"].astype(str).to_numpy())
    cl = tg.claims.loc[(tg.claims["loss_year"] < pd.Timestamp(C1.VALUATION_DATE).year)
                       & tg.claims["row_id"].astype(str).isin(pos.index)].reset_index(drop=True)
    Xc = X.iloc[cl["row_id"].astype(str).map(pos).to_numpy()].reset_index(drop=True)
    size = np.minimum(cl["cost_T"].to_numpy(dtype=float), tg.cap)
    fit("sev", size, np.zeros(len(cl)), Xc, {**C1.GEO_SEVERITY, "objective": "gamma"})
    for p in C1.PERILS[line]:
        sel = (cl["peril"] == p).to_numpy()
        fit(f"sev_{p}", size[sel], np.zeros(sel.sum()), Xc.loc[sel].reset_index(drop=True),
            {"objective": "gamma", "num_leaves": 7, "learning_rate": 0.03,
             "min_data_in_leaf": max(100, min(1000, int(sel.sum()) // 30))})
    return priors, info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lines", nargs="*", default=list(C.LINES))
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    PRIOR_DIR.mkdir(parents=True, exist_ok=True)
    with H.Timer("1st-round claims to ultimate and claim-count reporting pattern"):
        claims_ult, pattern = developed_claims()
    for line in a.lines:
        with H.Timer(f"{line}: both markets measured the same way"):
            dev, tg = developed_rows(line, claims_ult, pattern)
            em = S4.harmonise(*S4.read_market(C.ROOT, line, ("policies", "test")), line, developed=False)
            features, _ = S4.prior_features(line, dev, em)
            print(f"   1st round {len(dev):,} rows, this round {len(em):,} rows, {len(features)} shared features")
        n = 2 + 3 * len(C1.PERILS[line])
        with H.Timer(f"{line}: {n} component priors ({C.PRIOR_TREES} trees each)"):
            priors, info = fit_components(line, dev, tg, features, em)
            del dev
            frame = pd.DataFrame({"row_id": em["row_id"].astype(str).to_numpy(), **priors})
            frame.to_parquet(PRIOR_DIR / f"geo_priors_{line}.parquet", index=False)
            (OUT / f"priors_{line}.json").write_text(json.dumps(
                {"features": features, "cap": tg.cap, "load": tg.load, "components": info}, indent=1),
                encoding="utf-8")
            print(f"   {len(priors)} priors for {len(frame):,} rows -> {PRIOR_DIR / f'geo_priors_{line}.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
