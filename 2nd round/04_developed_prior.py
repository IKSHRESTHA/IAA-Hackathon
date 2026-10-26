"""Step 4 - the developed prior: a LightGBM fitted on the 1st round and applied to this round.

Two years of this market (20-50 thousand rows a line, 1-7 thousand claims) cannot pin down
dozens of rating factors. The 1st round has four years and about a million rows a line, with the
same products and the same category levels (all 20 occupations, all body types, and so on). The
relative effect of an excess, a flood zone or a vehicle group should largely carry over. The
level of costs, the climate and the regions do not.

So this step fits one LightGBM per line on the 1st round, using only the factors that mean the
same thing in both markets:

* kept: the features selected by the 1st round's step 3 that also exist here. Ages are measured
  the same way in both markets: exact 1st-round ages are mapped to this round's age bands.
* left out and learned locally: region, weather, claims history and the other lines held
  (``config.PRIOR_NOT_TRANSFERRED``), and factors that exist in one market only (credit band,
  mileage, telematics, gender).

Its log rate becomes part of the offset of step 5, so the 2nd-round model only has to learn how
this market departs from the developed one. In actuarial terms this is credibility: local
experience where there is enough of it, the larger market elsewhere.

Output: ``data/prior/prior_<line>.parquet`` (row_id, log_prior) for every training and 2025 row,
``reports/04_prior/`` (prior_check.csv, features.csv, importance_<line>.csv) and
``models/prior_<line>/``.

Usage: python 04_developed_prior.py [--lines motor ...]
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config as C
import helpers as H

sys.path.insert(0, str(C.ROOT.parent))            # pricing_lgbm.py sits in the folder above
import pricing_lgbm as P  # noqa: E402

FE = importlib.import_module("03_feature_engineering")   # step 3: the same derived columns

OUT = C.REPORTS / "04_prior"
PRIOR_DIR = C.DATA / "prior"
MODELS = C.ROOT / "models"
DEV_DATA = C.DEVELOPED / "data"
AGE_EDGES = np.array([25.0, 35.0, 50.0, 65.0])                       # 17-24 | 25-34 | 35-49 | 50-64 | 65+
AGE_MIDS = np.array(list(C.AGE_BANDS["driver_age_band"].values()))   # 21, 30, 42, 57, 72
EXACT_AGE = {"motor": "driver_age", "buildings": "policyholder_age", "contents": "policyholder_age"}


def load_developed_config():
    """The 1st round's config.py, under another module name (both rounds call theirs config)."""
    spec = importlib.util.spec_from_file_location("config_1st_round", C.DEVELOPED / "config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C1 = load_developed_config()                      # trend, settings and perils of the 1st round


def band_mid(age: pd.Series) -> np.ndarray:
    """Exact age -> the mid-point of this market's age band (missing stays missing)."""
    a = age.to_numpy(dtype=float)
    mid = AGE_MIDS[np.searchsorted(AGE_EDGES, np.nan_to_num(a, nan=0.0), side="right")]
    return np.where(np.isfinite(a), mid, np.nan).astype("float32")


def harmonise(policies: pd.DataFrame, customers: pd.DataFrame, addresses: pd.DataFrame,
              line: str, developed: bool) -> pd.DataFrame:
    """Policy rows joined with their customer and address, measured the same way in both markets."""
    df = policies.copy()
    df["year"] = df["period_start"].dt.year.astype("int16")
    df = FE.join_reference(df, customers, addresses)
    # Season on cover, property age, sum insured per bedroom, ordinal codes, and this round's age
    # bands turned into their mid-point ages: exactly as step 3 builds them.
    df = FE.extract_features(df, line)
    age = EXACT_AGE[line]
    if developed:
        df["age_band_mid"] = band_mid(df[age])     # exact age -> band: the coarser measurement
    else:
        df["age_band_mid"] = df[f"{age}_band"].astype("float32")   # already a mid-point (step 3 code)
    if line == "motor":                             # age at first licence, from the band as well
        df["age_first_licensed"] = (df["age_band_mid"] - df["years_licensed"]).astype("float32")
    df["flood_zone_missing"] = df["flood_zone"].isna().astype("int8")   # the informative-missing flag
    # Sums insured in 2025 money: the claims of the prior are on-levelled to 2025.
    return P.index_sum_insured(df, C.TEST_YEAR, C1.SI_INDEX.get(line, 0.0))


def read_market(folder, line: str, kinds: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean policy rows of ``kinds`` (policies, test) with the market's customers and addresses."""
    clean = folder / "data" / "clean"
    rows = pd.concat([pd.read_parquet(clean / f"{k}_{line}.parquet") for k in kinds], ignore_index=True)
    return rows, pd.read_parquet(clean / "customers.parquet"), pd.read_parquet(clean / "addresses.parquet")


def prior_features(line: str, developed: pd.DataFrame, emerging: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """The 1st round's selected features that exist, and mean the same, in both markets."""
    schema = pq.read_schema(DEV_DATA / "features" / f"test_{line}.parquet")
    selected = [c for c in schema.names if c not in P.KEYS and c != "exposure"]   # 1st-round step 3
    added = [c for c in C.PRIOR_ADD if c not in selected]                         # always offered
    selected += added
    prefixes = tuple(C.PRIOR_NOT_TRANSFERRED_PREFIXES)
    features, register = [], []
    for col in selected:
        c = "age_band_mid" if col == EXACT_AGE[line] else col   # exact age -> age band
        if c in C.PRIOR_NOT_TRANSFERRED:
            decision, why = "drop", C.PRIOR_NOT_TRANSFERRED[c]
        elif c.startswith(prefixes):
            decision, why = "drop", C.PRIOR_NOT_TRANSFERRED_PREFIXES[next(p for p in prefixes if c.startswith(p))]
        elif c not in emerging.columns:
            decision, why = "drop", "exists in the 1st round only"
        elif c not in developed.columns:
            decision, why = "drop", "not rebuilt from the clean tables"
        elif c in added:
            decision, why = "keep", C.PRIOR_ADD[c]
            features.append(c)
        else:
            decision, why = "keep", "same meaning in both markets" + (" (as age band)" if c != col else "")
            features.append(c)
        register.append({"line": line, "column": col, "used_as": c, "decision": decision, "reason": why})
    return features, pd.DataFrame(register)


def fit_prior(line: str, dev: pd.DataFrame, features: list[str]) -> tuple[P.TweedieLGBM, dict]:
    """LightGBM on every 1st-round row, claims at the 2025 cost level, a fixed number of trees."""
    y = P.onlevel_cost(dev, line, C1.PERILS[line], C1.TREND, C.TEST_YEAR, C1.LEVEL_SHIFT)
    y_capped, _, _ = P.cap_and_load(y, C1.CAP_QUANTILE)        # large losses would only add noise
    X, levels = P.prepare(dev, features)
    offset = np.log(dev["exposure"].to_numpy(dtype=float) * P.completeness(dev))
    model = P.TweedieLGBM(C1.LGBM_PARAMS, features).fit(X, y_capped, offset, n_trees=C.PRIOR_TREES,
                                                         seed=C.SEED)
    return model, levels


def log_prior(model: P.TweedieLGBM, levels: dict, rows: pd.DataFrame) -> np.ndarray:
    """Log expected cost per year on cover, at the developed market's level, for ``rows``."""
    X, _ = P.prepare(rows, model.features, levels)          # the 1st round's category levels
    return np.log(model.predict(X, np.zeros(len(X))))       # offset 0: one year on cover


def prior_check(line: str, train: pd.DataFrame, lp: pd.Series) -> list[dict]:
    """How well the prior alone ranks this round's claims, year by year, with the level fixed.

    The 1st round never saw these rows, so every year here is out of sample for the prior.
    "exposure only" is the reference: every row priced at the same rate per year on cover.
    """
    recs = []
    for year, g in train.groupby("year"):
        y = g["claim_cost"].to_numpy(dtype=float)
        seen = g["exposure"].to_numpy(dtype=float) * P.completeness(g)    # reported basis
        prior = seen * np.exp(g["row_id"].astype(str).map(lp).to_numpy(dtype=float))
        for name, mu in (("exposure only", seen), ("developed prior", prior)):
            mu = mu * y.sum() / mu.sum()                                   # level known
            recs.append({"line": line, "year": int(year), "model": name, "rows": int(len(g)),
                         "d2_level_known": round(P.d2_tweedie(y, mu), 5)})
    return recs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lines", nargs="*", default=list(C.LINES))
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    PRIOR_DIR.mkdir(parents=True, exist_ok=True)
    registers, checks = [], []

    for line in a.lines:
        with H.Timer(f"{line}: both markets measured the same way"):
            dev = harmonise(*read_market(C.DEVELOPED, line, ("policies",)), line, developed=True)
            perils = [f"cost_{p}" for p in C1.PERILS[line]]
            targets = pd.read_parquet(DEV_DATA / "features" / f"policies_{line}.parquet",
                                      columns=["row_id", "comp_cost", *perils])   # step 3 targets
            dev = dev.merge(targets, on="row_id", how="inner", validate="one_to_one")
            em = harmonise(*read_market(C.ROOT, line, ("policies", "test")), line, developed=False)
            features, register = prior_features(line, dev, em)
            registers.append(register)
            print(f"   1st round {len(dev):,} rows, this round {len(em):,} rows (training and 2025), "
                  f"{len(features)} shared features")

        with H.Timer(f"{line}: fitting the prior ({C.PRIOR_TREES} trees on every 1st-round row)"):
            model, levels = fit_prior(line, dev, features)
            del dev                                              # about a million rows: free them
            model.save(MODELS / f"prior_{line}")
            model.importance().to_csv(OUT / f"importance_{line}.csv", index=False)
            lp = pd.Series(log_prior(model, levels, em), index=em["row_id"].astype(str).to_numpy())
            pd.DataFrame({"row_id": lp.index, "log_prior": lp.to_numpy()}).to_parquet(
                PRIOR_DIR / f"prior_{line}.parquet", index=False)
            print(f"   prior written for {len(lp):,} rows -> {PRIOR_DIR / f'prior_{line}.parquet'}")

        with H.Timer(f"{line}: the prior alone on this round's claims"):
            train = H.read_table(f"policies_{line}", "features",
                                 columns=["row_id", "year", "exposure", "comp_cost", "claim_cost"])
            recs = prior_check(line, train, lp)
            checks += recs
            for r in recs:
                print(f"   {r['year']} {r['model']:<16} D2 (level known) {r['d2_level_known']:.4f}")

    P.save_by_line(pd.concat(registers, ignore_index=True), OUT / "features.csv", a.lines, C.LINES)
    P.save_by_line(pd.DataFrame(checks), OUT / "prior_check.csv", a.lines, C.LINES)
    print(f"\nReports -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
