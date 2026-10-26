"""Step 4 - LightGBM model of the 1st round (developed market) and its 2025 submission.

One LightGBM Tweedie model per line, fitted on the model-ready tables of step 3
(``data/features/policies_<line>`` and ``test_<line>``) with the shared tools of
``pricing_lgbm.py`` in the folder above:

1. Out-of-time check. Fit on 2021-2023 with claims on-levelled to 2024, then score the 2024 rows.
   The 2024 claims are not all reported yet, so each prediction is multiplied by the row's
   reported share before it is compared with the claims. D^2 is given per line and for all lines
   pooled (the competition score), as predicted and with the level known.
2. Final fit. All four years, claims on-levelled to 2025, the number of trees chosen on 20% of
   the policies, then a refit on every row. The 2025 rows need no completeness: their target is
   the ultimate cost.

Outputs: the next version in ``submissions/`` (``v<N>_lightgbm.csv``, in the order of the sample
submission), ``reports/04_model/`` (holdout.csv, summary.json, importance_<line>.csv) and
``models/lightgbm_<line>/`` (the fitted boosters).

Usage: python 04_lightgbm_model.py [--lines motor ...] [--skip-holdout]
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

import config as C
import helpers as H

sys.path.insert(0, str(C.ROOT.parent))            # pricing_lgbm.py sits in the folder above
import pricing_lgbm as P  # noqa: E402
import submissions as S  # noqa: E402

OUT = C.REPORTS / "04_model"
MODELS = C.ROOT / "models"


def model_features(test: pd.DataFrame) -> list[str]:
    """The features kept by step 3: every column of the 2025 table except keys and exposure."""
    features = P.feature_columns(test)
    if C.DROP_YEAR_WEATHER:                       # optional: no weather of the row's own year
        features = [f for f in features if not P.is_year_weather(f)]
    return features


def line_params(line: str) -> dict:
    """LightGBM settings of the round, with the line's own overrides on top."""
    return {**C.LGBM_PARAMS, **C.LGBM_PER_LINE.get(line, {})}


def target(df: pd.DataFrame, line: str, target_year: int) -> np.ndarray:
    """Claim cost of each row at the ``target_year`` cost level (trend and level shift)."""
    return P.onlevel_cost(df, line, C.PERILS[line], C.TREND, target_year, C.LEVEL_SHIFT)


def training_offset(df: pd.DataFrame) -> np.ndarray:
    """log(years on cover x share of the claims reported at the valuation date)."""
    return np.log(df["exposure"].to_numpy(dtype=float) * P.completeness(df))


def fit(train: pd.DataFrame, line: str, features: list[str], target_year: int, refit: bool) -> P.LineFit:
    """Fit one line on ``train``, with claims and sums insured in ``target_year`` money."""
    train = P.index_sum_insured(train, target_year, C.SI_INDEX.get(line, 0.0))
    y = target(train, line, target_year)
    return P.fit_line(train, features, y, training_offset(train), line_params(line), refit=refit,
                      seed=C.SEED, cap_quantile=C.CAP_QUANTILE, es_share=C.ES_SHARE,
                      refit_factor=C.REFIT_FACTOR)


def holdout(line: str, train: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, P.LineFit]:
    """Fit on the years before HOLDOUT_YEAR and predict the claims reported for HOLDOUT_YEAR."""
    year = C.HOLDOUT_YEAR
    past = train.loc[train["year"] < year].reset_index(drop=True)
    ho = train.loc[train["year"] == year].reset_index(drop=True)
    lf = fit(past, line, features, year, refit=False)          # the back-test keeps the early-stopped model
    ho_x = P.index_sum_insured(ho, year, C.SI_INDEX.get(line, 0.0))   # factor 1: already that year
    mu = lf.predict(ho_x, training_offset(ho))                 # reported basis, like the claims
    return ho["claim_cost"].to_numpy(dtype=float), mu, lf


def final(line: str, train: pd.DataFrame, test: pd.DataFrame,
          features: list[str]) -> tuple[np.ndarray, P.LineFit, float]:
    """Fit on every training year at the 2025 cost level and price the 2025 rows."""
    year = C.TEST_YEAR
    lf = fit(train, line, features, year, refit=True)
    te = P.index_sum_insured(test, year, C.SI_INDEX.get(line, 0.0))  # factor 1 for 2025 rows
    pred = lf.predict(te, np.log(test["exposure"].to_numpy(dtype=float)))   # ultimate: no completeness
    # Training cost per year on cover at the 2025 level, to compare with the 2025 prediction.
    tr = P.index_sum_insured(train, year, C.SI_INDEX.get(line, 0.0))
    seen = (tr["exposure"].to_numpy(dtype=float) * P.completeness(tr)).sum()
    return pred, lf, float(target(tr, line, year).sum() / seen)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lines", nargs="*", default=list(C.LINES))
    ap.add_argument("--skip-holdout", action="store_true", help="only the final fit and the submission")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    holdout_rows, pooled_y, pooled_mu, preds, summary = [], [], [], [], {}

    for line in a.lines:
        with H.Timer(f"{line}: reading the model-ready tables"):
            train = H.read_table(f"policies_{line}", "features")
            test = H.read_table(f"test_{line}", "features")
            features = model_features(test)
            print(f"   {len(train):,} training rows, {len(test):,} rows to price, {len(features)} features")

        if not a.skip_holdout:
            with H.Timer(f"{line}: out-of-time check (fit before {C.HOLDOUT_YEAR}, score {C.HOLDOUT_YEAR})"):
                y, mu, lf = holdout(line, train, features)
                s = P.score(y, mu)
                holdout_rows.append({"line": line, "trees": lf.model.n_trees, **s})
                pooled_y.append(y)
                pooled_mu.append(mu)
                print(f"   D2 {s['d2']:.4f} (level known {s['d2_level_known']:.4f}), "
                      f"actual / expected {s['a_over_e']:.3f}, {lf.model.n_trees} trees")

        with H.Timer(f"{line}: final fit on {min(C.TRAIN_YEARS)}-{max(C.TRAIN_YEARS)}, price {C.TEST_YEAR}"):
            pred, lf, train_avg = final(line, train, test, features)
            lf.model.save(MODELS / f"lightgbm_{line}")
            lf.model.importance().to_csv(OUT / f"importance_{line}.csv", index=False)
            preds.append(pd.DataFrame({"row_id": test["row_id"].astype(str).to_numpy(), "prediction": pred}))
            per_year = float(pred.sum() / test["exposure"].sum())
            summary[line] = {"features": len(features), "trees": lf.model.n_trees, "base": lf.model.base,
                             "balance": lf.balance, "large_loss_cap": lf.cap, "large_loss_load": lf.load,
                             "rows_2025": int(len(test)), "predicted_total_2025": float(pred.sum()),
                             "cost_per_year_on_cover_2025": per_year,
                             "training_cost_per_year_on_cover_at_2025_level": train_avg}
            print(f"   {lf.model.n_trees} trees, large-loss load {lf.load:.3f}; 2025 cost per year on cover "
                  f"{per_year:,.1f} (training average at the 2025 level {train_avg:,.1f})")

    if holdout_rows:
        all_y, all_mu = np.concatenate(pooled_y), np.concatenate(pooled_mu)
        holdout_rows.append({"line": "all lines", "trees": None, **P.score(all_y, all_mu)})
        table = pd.DataFrame(holdout_rows)
        table.to_csv(OUT / "holdout.csv", index=False)
        print(f"\nOut-of-time check, {C.HOLDOUT_YEAR} (reported claims):")
        print(table[["line", "rows", "trees", "d2", "d2_level_known", "a_over_e"]].round(4).to_string(index=False))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    if set(a.lines) != set(C.LINES):
        print("\nSubmission not written: only some lines were fitted.")
        return 0
    expected = H.read_table("sample_submission", "clean", columns=["row_id"])["row_id"].astype(str)
    sub = pd.concat(preds, ignore_index=True).set_index("row_id").loc[expected.to_numpy()].reset_index()
    for name, ok in P.check_submission(sub, expected).items():   # raises if any check fails
        print(f"   [{'PASS' if ok else 'FAIL'}] {name}")
    evidence = f"2024 holdout D2 {holdout_rows[-1]['d2']:.4f} (all lines)" if holdout_rows else ""
    S.save(sub, C.ROOT, "lightgbm", "04_lightgbm_model.py",
           "one LightGBM Tweedie model per line on the step-3 features", evidence)
    return 0


if __name__ == "__main__":
    sys.exit(main())
