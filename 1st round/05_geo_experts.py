"""Step 5 - the experts of the 1st round's winning ensemble, fitted on three fit sets.

The method of the best 1st-round private score (IAA_Part1_geo_models, 0.13830), built on this
folder's cleaned data and step-3 features with the shared code of ``geo_experts.py``:

* Claims to ultimate: open claims lose the recovery ratio seen on closed claims, and a share of
  them is declined later if recent years show significantly too few repudiations.
* Targets per fit set: claim costs on-levelled to the year priced (severity trend by loss year,
  bodily injury 2021-22 x 0.82), capped per claim at the 99.5% quantile of the fit years.
* Five experts (``total``, ``glm``, ``peril``, ``fs`` = ``fs_gamma`` + ``fs_lognorm``, ``fs_peril``),
  each fitted on 2021-22 to price 2023, on 2021-23 to price 2024, and on 2021-24 to price 2025.

Output: ``data/experts/<line>_<set>.parquet`` (one prediction column per expert, plus exposure,
claims and reported shares of the holdout rows) and ``reports/05_geo/`` (development parameters,
targets and trees per fit set). Step 6 combines them.

Usage: python 05_geo_experts.py [--lines motor ...] [--sets ho2023 ho2024 final]
                                [--experts total glm peril fs fs_peril] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

import config as C
import helpers as H

sys.path.insert(0, str(C.ROOT.parent))            # the shared modules sit in the folder above
import geo_experts as G  # noqa: E402
import pricing_lgbm as P  # noqa: E402

OUT = C.REPORTS / "05_geo"
EXPERTS_DIR = C.DATA / "experts"
VALUATION_YEAR = pd.Timestamp(C.VALUATION_DATE).year
STEP3_TARGETS = ("n_", "cost_", "claim_cost", "n_open")   # step 3 targets, rebuilt here on the ultimate basis


def model_features(test: pd.DataFrame) -> list[str]:
    """The features kept by step 3: every column of the 2025 table except keys and exposure."""
    features = P.feature_columns(test)
    if C.DROP_YEAR_WEATHER:
        features = [f for f in features if not P.is_year_weather(f)]
    return features


def count_shares(rows: pd.DataFrame, line: str, pattern: dict) -> dict[str, np.ndarray]:
    """Share of each peril's claim *count* reported at the valuation date (frequency experts)."""
    return {f"compn_{p}": H.row_completeness(rows, pattern[(line, p)], C.VALUATION_DATE) for p in C.PERILS[line]}


def without_step3_targets(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in df.columns if c.startswith(STEP3_TARGETS) and c != "comp_count"])


def add_codes(df: pd.DataFrame, line: str, kind: str) -> pd.DataFrame:
    """The raw many-level columns the ensemble encodes itself (make_model), from the clean table."""
    cols = list(C.CREDIBILITY_ENCODE.get(line, ()))
    if not cols:
        return df
    codes = H.read_table(f"{kind}_{line}", "clean", columns=["row_id", *cols])
    return df.merge(codes, on="row_id", how="left", validate="one_to_one")


def fit_set(line: str, name: str, train_all: pd.DataFrame, test: pd.DataFrame, features: list[str],
            claims_ult: pd.DataFrame, smoke: bool) -> tuple[G.FitData, dict]:
    """Everything one fit set needs: targets, model matrices, claims and offsets."""
    year = C.GEO_SETS[name]
    final = name == "final"
    rows = train_all if final else train_all.loc[train_all["year"] <= year].reset_index(drop=True)
    fit_years = [y for y in C.TRAIN_YEARS if y < year]          # the cap and loads never see the holdout
    tg = G.build_targets(rows, claims_ult, line, C.PERILS[line], C.TREND, C.LEVEL_SHIFT, year,
                         C.CAP_QUANTILE, fit_years)
    rows = rows.merge(tg.rows, on="row_id", how="left", validate="one_to_one")
    train = rows if final else rows.loc[rows["year"] < year].reset_index(drop=True)
    score = test.copy() if final else rows.loc[rows["year"] == year].reset_index(drop=True)
    if smoke:                                                   # a quick test of the code path
        train = train.sample(min(60_000, len(train)), random_state=1).reset_index(drop=True)
        score = score.sample(min(20_000, len(score)), random_state=2).reset_index(drop=True)
    si = C.SI_INDEX.get(line, 0.0)
    train, score = P.index_sum_insured(train, year, si), P.index_sum_insured(score, year, si)
    freq_trend = G.frequency_trend(train["year"], line, C.TREND, year)

    feats = list(features)
    for col in C.CREDIBILITY_ENCODE.get(line, ()):            # make_model -> credibility relativity
        off = np.log(train["exposure"].to_numpy(dtype=float) * P.completeness(train) / freq_trend)
        enc_tr, (enc_sc,) = G.credibility_encode(train, [score], col, train["y_capped"].to_numpy(dtype=float), off,
                                                 train["n_claims"].to_numpy(dtype=float), k=C.CREDIBILITY_K)
        train[f"{col}_te"], score[f"{col}_te"] = enc_tr, enc_sc
        feats.append(f"{col}_te")
    X, levels = P.prepare(train, feats)
    Xs, _ = P.prepare(score, feats, levels)

    # Claims for the claim-size models: those of the training rows; in the final fit only the
    # mature loss years (2024 claims are still young: late, larger claims are missing).
    pos = pd.Series(np.arange(len(train)), index=train["row_id"].astype(str).to_numpy())
    cl = tg.claims.loc[tg.claims["row_id"].astype(str).isin(pos.index)].copy()
    if final:
        cl = cl.loc[cl["loss_year"] < VALUATION_YEAR]
    cl["pos"] = cl["row_id"].astype(str).map(pos).to_numpy()
    cl["policy_id"] = train["policy_id"].to_numpy()[cl["pos"].to_numpy()]
    cl = cl.reset_index(drop=True)

    mono = [int(C.GEO_MONOTONE.get(line, {}).get(f, 0)) for f in feats]
    params = {**C.LGBM_PARAMS, **C.LGBM_PER_LINE.get(line, {})}
    d = G.FitData(line, C.PERILS[line], train, score, X, Xs, cl, tg, freq_trend, mono, params,
                  refit=final, seed=C.SEED)
    summary = {"target_year": year, "train_rows": int(len(train)), "score_rows": int(len(score)),
               "features": feats, "cap": tg.cap, "load": tg.load, "peril_loads": tg.peril_loads,
               "load_years": tg.load_years, "severity_claims": int(len(cl))}
    return d, summary


def quick_scores(preds: pd.DataFrame) -> str:
    """D^2 of each expert on the holdout year (reported basis), to follow the run."""
    share = G.reported_share(preds)
    y = preds["y"].to_numpy(dtype=float)
    cols = [c for c in ("total", "glm", "peril", "fs_gamma", "fs_lognorm", "fs_peril") if c in preds]
    return ", ".join(f"{c} {P.d2_tweedie(y, preds[c].to_numpy(dtype=float) * share):.4f}" for c in cols)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lines", nargs="*", default=list(C.LINES))
    ap.add_argument("--sets", nargs="*", default=list(C.GEO_SETS), choices=list(C.GEO_SETS))
    ap.add_argument("--experts", nargs="*", default=list(C.GEO_EXPERT_STEPS), choices=list(G.EXPERT_STEPS))
    ap.add_argument("--smoke", action="store_true", help="a small sample of every fit set, to test the code")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    EXPERTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if a.smoke else ""

    with H.Timer("claims to ultimate and claim-count reporting patterns"):
        claims = H.read_table("claims", "clean")
        dev = G.development_params(claims, C.DEV_MATURE_YEARS, C.DEV_IMMATURE_YEARS, C.REPUDIATION_MIN_Z)
        claims_ult = G.claims_to_ultimate(claims, dev)
        is_open = claims_ult["status"] == "open"
        print(f"   open claims: {is_open.sum():,}, incurred {claims_ult.loc[is_open, 'net_incurred'].sum():,.0f} "
              f"-> ultimate {claims_ult.loc[is_open, 'net_ultimate'].sum():,.0f}")
        adopted = {f"{l}/{p}": round(v["p"], 4) for (l, p), v in dev["repudiation"].items() if v["p"] > 0}
        print(f"   later repudiation adopted for: {adopted or 'none'}")
        (OUT / "development.json").write_text(json.dumps({
            "recovery_ratio": {f"{l}/{p}": v for (l, p), v in dev["recovery_ratio"].items()},
            "repudiation": {f"{l}/{p}": v for (l, p), v in dev["repudiation"].items()}}, indent=1), encoding="utf-8")
        nonrep = claims.loc[claims["status"] != "repudiated"]
        count_pattern = H.reporting_pattern(nonrep, C.VALUATION_DATE, f"{min(C.TRAIN_YEARS)}-01-01")

    for line in a.lines:
        with H.Timer(f"{line}: reading the model-ready tables"):
            train_all = without_step3_targets(H.read_table(f"policies_{line}", "features"))
            test = H.read_table(f"test_{line}", "features")
            features = model_features(test)
            train_all = train_all.assign(**count_shares(train_all, line, count_pattern))
            train_all, test = add_codes(train_all, line, "policies"), add_codes(test, line, "test")
            print(f"   {len(train_all):,} training rows, {len(test):,} rows to price, {len(features)} step-3 features")
        for name in a.sets:
            with H.Timer(f"{line} / {name}: experts {', '.join(a.experts)}"):
                d, summary = fit_set(line, name, train_all, test, features, claims_ult, a.smoke)
                preds, info = G.run_experts(d, tuple(a.experts), C.GEO_GLM, C.GEO_SEVERITY)
                score = d.score
                preds.insert(1, "line", line)
                preds.insert(2, "set", name)
                preds.insert(3, "year", summary["target_year"])
                preds.insert(4, "exposure", score["exposure"].to_numpy(dtype=float))
                preds.insert(5, "y", score["y"].to_numpy(dtype=float) if "y" in score else np.nan)
                preds.insert(6, "comp_line", score["comp_cost"].to_numpy(dtype=float) if "comp_cost" in score else 1.0)
                preds.to_parquet(EXPERTS_DIR / f"{line}_{name}{tag}.parquet", index=False)
                (OUT / f"experts_{line}_{name}{tag}.json").write_text(
                    json.dumps({"summary": summary, "experts": info}, indent=1, default=float), encoding="utf-8")
                print(f"   cap {summary['cap']:,.0f}, large-loss load {summary['load']:.3f}, "
                      f"{summary['severity_claims']:,} claims for claim sizes")
                if name != "final" and "peril" in preds:
                    print(f"   holdout D2: {quick_scores(preds)}")
    print(f"\nExpert predictions -> {EXPERTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
