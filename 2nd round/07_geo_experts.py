"""Step 7: the experts of the 1st round's winning ensemble, each on top of its developed prior.

The six experts of ``geo_experts.py`` (total, glm, peril, fs_gamma, fs_lognorm, fs_peril) are
fitted on this round's data with the matching developed-market prior in their offsets (step 4
for the total cost, step 6 for every other component), so each learns only how this market
departs from the developed one. Where a peril has too few claims here, its prior alone,
re-levelled to this market, takes the place of a model.

Three fit sets per line:

* ``ho2024``  fit on 2023 (cost level 2024), price 2024: the one forward back-test, kept as a check;
* ``cv``      5 folds of customers over 2023-2024 (cost level 2025): every training row is priced by
              models that never saw its customer. Step 8 learns the ensemble's slope from these;
* ``final``   fit on 2023-2024, price 2025.

Tuning, per line, on the ``cv`` folds (pooled out-of-fold D^2 decides): the LightGBM settings of
the frequency and cost experts (``config.TUNE_GRID``, tried on ``total``) and the GLM's ridge
(``config.GLM_ALPHA_GRID``).

Output: ``data/experts/<line>_<set>.parquet`` and ``reports/07_geo/`` (tuning.csv, development.json,
experts_<line>_<set>.json).

Usage: python 07_geo_experts.py [--lines motor ...] [--sets ho2024 cv final] [--no-tuning]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys

import numpy as np
import pandas as pd

import config as C
import helpers as H

sys.path.insert(0, str(C.ROOT.parent))            # the shared modules sit in the folder above
import geo_experts as G  # noqa: E402
import pricing_lgbm as P  # noqa: E402

OUT = C.REPORTS / "07_geo"
EXPERTS_DIR = C.DATA / "experts"
PRIOR_DIR = C.DATA / "prior"
VALUATION_YEAR = pd.Timestamp(C.VALUATION_DATE).year
STEP3_TARGETS = ("n_", "cost_", "claim_cost", "n_open")   # step 3 targets, rebuilt here on the ultimate basis


def model_features(test: pd.DataFrame) -> list[str]:
    """The features kept by step 3, without the weather of the row's own year."""
    features = P.feature_columns(test)
    return [f for f in features if not P.is_year_weather(f)] if C.DROP_YEAR_WEATHER else features


def load_line(line: str, count_pattern: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Step-3 rows with claim-count shares, the developed priors, new business and gender."""
    train = H.read_table(f"policies_{line}", "features")
    train = train.drop(columns=[c for c in train.columns if c.startswith(STEP3_TARGETS)])
    test = H.read_table(f"test_{line}", "features")
    features = model_features(test)
    for p in C.PERILS[line]:                                   # share of each peril's claim count reported
        train[f"compn_{p}"] = H.row_completeness(train, count_pattern[(line, p)], C.VALUATION_DATE)
    priors = (pd.read_parquet(PRIOR_DIR / f"prior_{line}.parquet").rename(columns={"log_prior": "cost"})
              .merge(pd.read_parquet(PRIOR_DIR / f"geo_priors_{line}.parquet"), on="row_id", validate="one_to_one")
              .set_index("row_id"))
    nb = pd.concat([H.read_table(f"{k}_{line}", "clean", columns=["row_id", "new_business"])
                    for k in ("policies", "test")]).set_index("row_id")["new_business"]
    gender = H.read_table("customers", "clean", columns=["customer_id", "gender"]).set_index("customer_id")["gender"]
    for df in (train, test):
        ids = df["row_id"].astype(str)
        for k in priors.columns:                                # lp_<component>: log prior per year on cover
            df[f"lp_{k}"] = ids.map(priors[k]).to_numpy(dtype=float)
        df["_new_business"] = ids.map(nb).astype("boolean").fillna(False).to_numpy(dtype=bool)
        df["_gender"] = df["customer_id"].astype(str).map(gender).astype("string")
        if df[[f"lp_{k}" for k in priors.columns]].isna().any().any():
            raise RuntimeError(f"{line}: rows without a prior; run 04_developed_prior.py and 06_geo_priors.py")
    return train, test, features, list(priors.columns)


def make_fit(line: str, train: pd.DataFrame, score: pd.DataFrame, features: list[str], keys: list[str],
             tg: G.Targets, year: int, params: dict, final: bool, cv: bool = False) -> G.FitData:
    """One fit set: model matrices, claims for the claim sizes, centred priors and offsets."""
    si = C.SI_INDEX.get(line, 0.0)
    train, score = P.index_sum_insured(train, year, si), P.index_sum_insured(score, year, si)
    X, levels = P.prepare(train, features)
    Xs, _ = P.prepare(score, features, levels)
    pos = pd.Series(np.arange(len(train)), index=train["row_id"].astype(str).to_numpy())
    cl = tg.claims.loc[tg.claims["row_id"].astype(str).isin(pos.index)].copy()
    if year > VALUATION_YEAR:              # pricing 2025: claim sizes from mature loss years only
        cl = cl.loc[cl["loss_year"] < VALUATION_YEAR]
    cl["pos"] = cl["row_id"].astype(str).map(pos).to_numpy()
    cl["policy_id"] = train["policy_id"].to_numpy()[cl["pos"].to_numpy()]
    expo = train["exposure"].to_numpy(dtype=float)
    prior, prior_score = {}, {}
    for k in keys:                         # centred: only the prior's relativities are used, not its level
        c = float(np.average(train[f"lp_{k}"].to_numpy(dtype=float), weights=expo))
        prior[k] = train[f"lp_{k}"].to_numpy(dtype=float) - c
        prior_score[k] = score[f"lp_{k}"].to_numpy(dtype=float) - c
    trend = G.frequency_trend(train["year"], line, C.TREND, year)
    score_trend = G.frequency_trend(score["year"], line, C.TREND, year) if cv else None   # earlier-year rows
    mono = [int(C.GEO_MONOTONE.get(line, {}).get(f, 0)) for f in features]
    return G.FitData(line, C.PERILS[line], train, score, X, Xs, cl.reset_index(drop=True), tg, trend, mono, params,
                     refit=final, seed=C.SEED, score_trend=score_trend, prior=prior, prior_score=prior_score,
                     min_freq_claims=C.MIN_FREQ_CLAIMS, min_size_claims=C.MIN_SIZE_CLAIMS)


def tune(line: str, rows: pd.DataFrame, fold: np.ndarray, fits: list[G.FitData]) -> tuple[dict, float, list[dict]]:
    """Pick the LightGBM settings (on ``total``) and the GLM ridge with the best out-of-fold D^2."""
    y = rows["y"].to_numpy(dtype=float)
    share = P.completeness(rows, "comp_cost")                  # predictions compared on the reported basis
    recs = []

    def oof(run) -> float:
        mu = np.empty(len(rows))
        for k, d in enumerate(fits):
            mu[fold == k] = run(d)
        return P.d2_tweedie(y, mu * share)

    for i, cand in enumerate(C.TUNE_GRID):
        d2 = oof(lambda d: G.expert_total(dataclasses.replace(d, params=cand))[0]["total"])
        recs.append({"line": line, "expert": "total", "candidate": i, **{k: cand[k] for k in
                     ("num_leaves", "min_data_in_leaf", "lambda_l2", "learning_rate")}, "cv_d2": d2})
    for alpha in C.GLM_ALPHA_GRID:
        d2 = oof(lambda d: G.expert_glm(d, {**C.GEO_GLM, "alpha": alpha})[0]["glm"])
        recs.append({"line": line, "expert": "glm", "alpha": alpha, "cv_d2": d2})
    gbm = max((r for r in recs if r["expert"] == "total"), key=lambda r: r["cv_d2"])
    glm = max((r for r in recs if r["expert"] == "glm"), key=lambda r: r["cv_d2"])
    return C.TUNE_GRID[gbm["candidate"]], float(glm["alpha"]), recs


def frame(line: str, name: str, score: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """Expert predictions with what step 8 needs: claims, reported shares, new business, gender."""
    out = pd.DataFrame({"row_id": score["row_id"].astype(str).to_numpy(), "line": line, "set": name,
                        "year": score["year"].to_numpy(), "exposure": score["exposure"].to_numpy(dtype=float),
                        "customer_id": score["customer_id"].astype(str).to_numpy(),
                        "_new_business": score["_new_business"].to_numpy(), "_gender": score["_gender"].to_numpy()})
    for c in ("y", "y_capped", "n_claims", "comp_cost", "comp_count"):
        out[c] = score[c].to_numpy(dtype=float) if c in score else (1.0 if c.startswith("comp") else np.nan)
    return pd.concat([out, preds.drop(columns="row_id").reset_index(drop=True)], axis=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lines", nargs="*", default=list(C.LINES))
    ap.add_argument("--sets", nargs="*", default=list(C.GEO_SETS), choices=list(C.GEO_SETS))
    ap.add_argument("--no-tuning", action="store_true", help="step 5's LightGBM settings and a GLM ridge of 0.01")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    EXPERTS_DIR.mkdir(parents=True, exist_ok=True)
    steps = tuple(C.GEO_EXPERT_STEPS)

    with H.Timer("claims to ultimate and claim-count reporting patterns"):
        claims = H.read_table("claims", "clean")
        dev = G.development_params(claims, C.DEV_MATURE_YEARS, C.DEV_IMMATURE_YEARS, C.REPUDIATION_MIN_Z)
        claims_ult = G.claims_to_ultimate(claims, dev)
        is_open = claims_ult["status"] == "open"
        print(f"   open claims: {is_open.sum():,}, incurred {claims_ult.loc[is_open, 'net_incurred'].sum():,.0f} "
              f"-> ultimate {claims_ult.loc[is_open, 'net_ultimate'].sum():,.0f}")
        (OUT / "development.json").write_text(json.dumps({
            "recovery_ratio": {f"{l}/{p}": v for (l, p), v in dev["recovery_ratio"].items()},
            "repudiation": {f"{l}/{p}": v for (l, p), v in dev["repudiation"].items()}}, indent=1), encoding="utf-8")
        nonrep = claims.loc[claims["status"] != "repudiated"]
        count_pattern = H.reporting_pattern(nonrep, C.VALUATION_DATE, f"{min(C.TRAIN_YEARS)}-01-01")

    tuning = []
    for line in a.lines:
        with H.Timer(f"{line}: data, priors, targets at the 2025 cost level"):
            train_all, test, features, keys = load_line(line, count_pattern)
            tg25 = G.build_targets(train_all, claims_ult, line, C.PERILS[line], C.TREND, C.LEVEL_SHIFT,
                                   C.TEST_YEAR, C.CAP_QUANTILE, list(C.TRAIN_YEARS))
            rows = train_all.merge(tg25.rows, on="row_id", how="left", validate="one_to_one")
            fold = H.stable_hash(rows["customer_id"], salt="cv") % C.CV_FOLDS   # all rows of a customer together
            base = {**C.LGBM_PARAMS, **C.LGBM_PER_LINE.get(line, {})}
            fits = [make_fit(line, rows.loc[fold != k].reset_index(drop=True), rows.loc[fold == k].reset_index(drop=True),
                             features, keys, tg25, C.TEST_YEAR, base, final=False, cv=True) for k in range(C.CV_FOLDS)]
            print(f"   {len(rows):,} training rows, {len(test):,} to price, {len(features)} features, "
                  f"{len(keys)} priors; cap {tg25.cap:,.0f}, load {tg25.load:.3f}")
        if a.no_tuning:
            params, alpha = base, 1e-2
        else:
            with H.Timer(f"{line}: tuning by {C.CV_FOLDS}-fold cross-validation"):
                params, alpha, recs = tune(line, rows, fold, fits)
                tuning += recs
                for r in recs:
                    what = (f"leaves {r['num_leaves']:>2}, min leaf {r['min_data_in_leaf']:>4}, l2 {r['lambda_l2']:>5}"
                            if r["expert"] == "total" else f"alpha {r['alpha']:g}")
                    print(f"   {r['expert']:<5} {what:<36} out-of-fold D2 {r['cv_d2']:.4f}")
        glm_settings = {**C.GEO_GLM, "alpha": alpha}

        for name in a.sets:
            with H.Timer(f"{line} / {name}: experts"):
                if name == "ho2024":
                    tg = G.build_targets(train_all, claims_ult, line, C.PERILS[line], C.TREND, C.LEVEL_SHIFT,
                                         2024, C.CAP_QUANTILE, [2023])
                    r24 = train_all.merge(tg.rows, on="row_id", how="left", validate="one_to_one")
                    d = make_fit(line, r24.loc[r24["year"] < 2024].reset_index(drop=True),
                                 r24.loc[r24["year"] == 2024].reset_index(drop=True), features, keys, tg, 2024,
                                 params, final=False)
                    preds, info = G.run_experts(d, steps, glm_settings, C.GEO_SEVERITY)
                    out = frame(line, name, d.score, preds)
                elif name == "cv":
                    parts, info = [], {}
                    for k, d in enumerate(fits):                # every fold, with the tuned settings
                        d = dataclasses.replace(d, params=params)
                        preds, info[f"fold{k}"] = G.run_experts(d, steps, glm_settings, C.GEO_SEVERITY)
                        parts.append(frame(line, name, d.score, preds))
                    out = pd.concat(parts, ignore_index=True)
                else:
                    d = make_fit(line, rows, test, features, keys, tg25, C.TEST_YEAR, params, final=True)
                    preds, info = G.run_experts(d, steps, glm_settings, C.GEO_SEVERITY)
                    out = frame(line, name, d.score, preds)
                out.to_parquet(EXPERTS_DIR / f"{line}_{name}.parquet", index=False)
                (OUT / f"experts_{line}_{name}.json").write_text(json.dumps(
                    {"params": params, "glm_alpha": alpha, "experts": info}, indent=1, default=float), encoding="utf-8")
                if name != "final":
                    share, y = G.reported_share(out), out["y"].to_numpy(dtype=float)
                    print("   D2: " + ", ".join(f"{m} {P.d2_tweedie(y, out[m].to_numpy(dtype=float) * share):.4f}"
                                                for m in C.GEO_MEMBERS if m in out))
    if tuning:
        P.save_by_line(pd.DataFrame(tuning), OUT / "tuning.csv", a.lines, C.LINES)
    print(f"\nExpert predictions -> {EXPERTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
