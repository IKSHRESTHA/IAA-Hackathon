"""Step 5 - the 2nd round's LightGBM model on top of the developed prior, and the 2025 submission.

This is the transfer model of the pricing repository (iaa-pricing-part2), as a pure LightGBM:

    log E[cost] = log(exposure x completeness) + (log prior - its average) + trees(features)

The developed prior of step 4 carries the risk structure of the 1st round, so the trees only learn
how this market departs from it. In that repository this model alone, with no GLM and no blend,
gave the best public score (0.16529). Two later findings of the repository are applied too: the
weather of the row's own year is dropped (2025 lies far outside the two training years), and the
prior is fitted on every developed row (step 4).

1. Out-of-time check: fit on 2023 (claims on-levelled to 2024) and score 2024. Three models side
   by side: the prior alone (re-levelled), a LightGBM without the prior, and the transfer model.
   The business adjustments estimated from 2023 alone are then replayed on 2024.
2. Final fit on 2023-2024 at the 2025 cost level, then the business adjustments:
   * New business is 76% of the 2025 book against 12-15% of the training rows, so its relativity
     drives the 2025 level. The effect the model misses is measured on out-of-fold predictions,
     year by year, for claim frequency (and for motor, claim size), pooled over the two years and
     shrunk towards no effect.
   * Gender (motor, contents): the developed prior has no gender, and early stopping shrinks what
     the trees learn of it. It is measured and shrunk the same way, and it moves price between men
     and women without changing the line's total.

Outputs: the next version in ``submissions/`` (``v<N>_transfer_lightgbm.csv``),
``outputs/predictions_lightgbm.parquet``,
``reports/05_model/`` (holdout.csv, adjustments.json, summary.json, importance_<line>.csv) and
``models/transfer_<line>/``.

Usage: python 05_lightgbm_model.py [--lines motor ...] [--skip-holdout]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config as C
import helpers as H

sys.path.insert(0, str(C.ROOT.parent))            # pricing_lgbm.py sits in the folder above
import pricing_lgbm as P  # noqa: E402
import submissions as S  # noqa: E402

OUT = C.REPORTS / "05_model"
MODELS = C.ROOT / "models"
PREDICTIONS = C.ROOT / "outputs" / "predictions_lightgbm.parquet"


# =================================================================================================
# Data
# =================================================================================================
def load_line(line: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Model-ready rows of step 3, with the prior of step 4, the new-business flag and gender.

    The helper columns start with an underscore: they are never model features.
    """
    train = H.read_table(f"policies_{line}", "features")
    test = H.read_table(f"test_{line}", "features")
    features = P.feature_columns(test)
    if C.DROP_YEAR_WEATHER:                       # keep the address climate, not the year's weather
        features = [f for f in features if not P.is_year_weather(f)]
    prior = pd.read_parquet(C.DATA / "prior" / f"prior_{line}.parquet").set_index("row_id")["log_prior"]
    nb = pd.concat([H.read_table(f"{k}_{line}", "clean", columns=["row_id", "new_business"])
                    for k in ("policies", "test")]).set_index("row_id")["new_business"]
    gender = H.read_table("customers", "clean", columns=["customer_id", "gender"]).set_index("customer_id")["gender"]
    for df in (train, test):
        ids = df["row_id"].astype(str)
        df["_log_prior"] = ids.map(prior).to_numpy(dtype=float)
        df["_new_business"] = ids.map(nb).astype("boolean").fillna(False).to_numpy(dtype=bool)
        df["_gender"] = df["customer_id"].astype(str).map(gender).astype("string")
        if np.isnan(df["_log_prior"]).any():
            raise RuntimeError(f"{line}: rows without a prior; run 04_developed_prior.py first")
    return train, test, features


# =================================================================================================
# The model
# =================================================================================================
def line_params(line: str) -> dict:
    """LightGBM settings of the round, with the line's own overrides on top."""
    return {**C.LGBM_PARAMS, **C.LGBM_PER_LINE.get(line, {})}


def target(df: pd.DataFrame, line: str, year: int) -> np.ndarray:
    """Claim cost of each row at the ``year`` cost level."""
    return P.onlevel_cost(df, line, C.PERILS[line], C.TREND, year, C.LEVEL_SHIFT)


def prior_centre(df: pd.DataFrame) -> float:
    """Exposure-weighted average log prior: only the prior's relativities are used, not its level."""
    return float(np.average(df["_log_prior"].to_numpy(dtype=float), weights=df["exposure"].to_numpy(dtype=float)))


def offset(df: pd.DataFrame, centre: float, use_prior: bool, reported: bool) -> np.ndarray:
    """log(exposure x completeness), plus the centred log prior when ``use_prior``.

    ``reported``: multiply by the share of the claims reported at the valuation date (training rows
    and holdout rows). 2025 rows are priced at their ultimate cost, so their completeness is 1.
    """
    o = np.log(df["exposure"].to_numpy(dtype=float) * (P.completeness(df) if reported else 1.0))
    if use_prior:
        o = o + df["_log_prior"].to_numpy(dtype=float) - centre
    return o


@dataclass
class Fitted:
    """A fitted line plus how its offset was built, so new rows get the same offset."""

    lf: P.LineFit
    centre: float
    use_prior: bool

    def predict(self, rows: pd.DataFrame, reported: bool) -> np.ndarray:
        # Scored rows are always in their own year (2024 holdout, 2025 test), so their sums
        # insured are already in that year's money.
        return self.lf.predict(rows, offset(rows, self.centre, self.use_prior, reported))


def fit_model(rows: pd.DataFrame, line: str, features: list[str], year: int, use_prior: bool,
              refit: bool) -> Fitted:
    """Fit one line on ``rows`` with claims and sums insured in ``year`` money."""
    rows = P.index_sum_insured(rows, year, C.SI_INDEX.get(line, 0.0))
    centre = prior_centre(rows) if use_prior else 0.0
    lf = P.fit_line(rows, features, target(rows, line, year), offset(rows, centre, use_prior, True),
                    line_params(line), refit=refit, seed=C.SEED, cap_quantile=C.CAP_QUANTILE,
                    es_share=C.ES_SHARE, refit_factor=C.REFIT_FACTOR)
    return Fitted(lf, centre, use_prior)


def prior_alone(rows: pd.DataFrame, score: pd.DataFrame, line: str, year: int) -> np.ndarray:
    """The developed prior re-levelled to this market: no trees at all (holdout comparison)."""
    rows = P.index_sum_insured(rows, year, C.SI_INDEX.get(line, 0.0))
    centre = prior_centre(rows)
    y_capped, _, load = P.cap_and_load(target(rows, line, year), C.CAP_QUANTILE)
    level = y_capped.sum() / np.exp(offset(rows, centre, True, True)).sum()   # this market's level
    return np.exp(offset(score, centre, True, True)) * level * load


def out_of_fold(rows: pd.DataFrame, line: str, features: list[str], year: int,
                n_trees: int) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold expected cost (capped, reported basis) of every row, and the capped claims.

    Each fold of policies is predicted by a model that never saw it, with the settings and number
    of trees of the model being adjusted. The adjustments are measured against these predictions,
    so they only pick up what the model misses.
    """
    rows = P.index_sum_insured(rows, year, C.SI_INDEX.get(line, 0.0))
    y_capped, _, _ = P.cap_and_load(target(rows, line, year), C.CAP_QUANTILE)
    off = offset(rows, prior_centre(rows), True, True)
    X, _ = P.prepare(rows, features)
    fold = H.stable_hash(rows["policy_id"], salt="oof") % C.OOF_FOLDS    # all rows of a policy together
    mu = np.empty(len(rows))
    for k in range(C.OOF_FOLDS):
        out = fold == k
        m = P.TweedieLGBM(line_params(line), features).fit(X.loc[~out], y_capped[~out], off[~out],
                                                            n_trees=n_trees, seed=C.SEED)
        b = P.balance_factor(y_capped[~out], m.predict(X.loc[~out], off[~out]))
        mu[out] = m.predict(X.loc[out], off[out]) * b
    return mu, y_capped


# =================================================================================================
# Business adjustments: new business and gender
# =================================================================================================
def group_level(y: np.ndarray, m: np.ndarray, mask: np.ndarray, p: float,
                codes: np.ndarray, W: np.ndarray) -> tuple[float, np.ndarray]:
    """Log level of a group against its expected values ``m``, and its bootstrap replicates.

    For log E[y] = log m + a + b x (x a 0/1 flag), the score equations of a Tweedie GLM with power
    p separate by group, with the solution K = sum(y m^(1-p)) / sum(m^(2-p)). p = 1 is the Poisson
    case (sum y / sum m). No iterative fit is needed.
    """
    num = np.where(mask, y * np.power(m, 1 - p), 0.0)
    den = np.where(mask, np.power(m, 2 - p), 0.0)
    reps = H.ClusterBootstrap.replicate_sums(codes, W, num) / H.ClusterBootstrap.replicate_sums(codes, W, den)
    return float(np.log(num.sum() / den.sum())), np.log(reps)


def year_effects(rows: pd.DataFrame, mu: np.ndarray, y_capped: np.ndarray, flag: np.ndarray,
                 boot: H.ClusterBootstrap) -> dict[str, tuple[float, float]]:
    """Log effects of ``flag`` (vs not) on claim frequency, cost and size, with standard errors.

    The bootstrap resamples customers with the same weights for every quantity, so the size
    effect (cost minus frequency) gets a paired standard error.
    """
    codes, W = boot.weights(rows["customer_id"])
    n = rows["n_claims"].to_numpy(dtype=float)
    m_count = mu * P.completeness(rows, "comp_count") / P.completeness(rows)   # count completeness
    out, reps = {}, {}
    for name, y, m, p in (("freq", n, m_count, 1.0), ("cost", y_capped, mu, P.TWEEDIE_POWER)):
        a1, r1 = group_level(y, m, flag, p, codes, W)
        a0, r0 = group_level(y, m, ~flag, p, codes, W)
        out[name], reps[name] = a1 - a0, r1 - r0
    out["size"], reps["size"] = out["cost"] - out["freq"], reps["cost"] - reps["freq"]
    return {k: (float(out[k]), float(np.std(reps[k]))) for k in out}


def pooled(est: list[tuple[float, float]], tau0: float) -> dict[str, float]:
    """Random-effects mean over years, shrunk towards 0 with a normal prior of sd ``tau0``.

    The 2025 effect is taken as the mean effect plus year-to-year noise. The year-to-year variance
    is estimated from the years (floored at 0). The shrinkage keeps the effect near 0 unless the
    data are clear: credibility z = tau0^2 / (tau0^2 + variance of the mean).
    """
    b = np.array([x for x, _ in est])
    s2 = np.array([se ** 2 for _, se in est])
    tau_y2 = max(0.0, float(np.var(b, ddof=1)) - float(s2.mean())) if len(b) > 1 else 0.0
    w = 1.0 / (s2 + tau_y2)
    mean, var = float(np.sum(w * b) / w.sum()), float(1.0 / w.sum())
    z = tau0 ** 2 / (tau0 ** 2 + var)
    return {"data_mean": mean, "se": float(np.sqrt(var)), "tau_year": float(np.sqrt(tau_y2)),
            "credibility_z": float(z), "posterior": float(z * mean)}


def is_male(rows: pd.DataFrame) -> np.ndarray:
    return rows["_gender"].eq("m").fillna(False).to_numpy(dtype=bool)


def estimate_adjustments(line: str, rows: pd.DataFrame, mu: np.ndarray, y_capped: np.ndarray,
                         years: list[int]) -> dict:
    """New-business and gender factors from out-of-fold predictions of ``years``."""
    boot = H.ClusterBootstrap(C.N_BOOT, seed=C.SEED)
    nb, male = rows["_new_business"].to_numpy(dtype=bool), is_male(rows)
    est_nb, est_g = {}, {}
    for yr in years:
        sel = (rows["year"] == yr).to_numpy()
        part = rows.loc[sel]
        est_nb[yr] = year_effects(part, mu[sel], y_capped[sel], nb[sel], boot)
        est_g[yr] = year_effects(part, mu[sel], y_capped[sel], male[sel], boot)

    # New business: frequency always; claim size too where it was consistent (motor).
    freq = pooled([est_nb[y]["freq"] for y in years], C.NB_SHRINK["freq"])
    size = pooled([est_nb[y]["size"] for y in years], C.NB_SHRINK["size"])
    log_e = freq["posterior"] + (size["posterior"] if line in C.NB_WITH_SIZE else 0.0)
    share = float(mu[nb].sum() / mu.sum())        # new business share of the expected cost
    # The model is balanced on the training rows: if new business is under-predicted by e,
    # renewals are over-predicted. k keeps the training total: share x e x k + (1 - share) x k = 1.
    k = 1.0 / (share * np.exp(log_e) + 1.0 - share)
    gender = pooled([est_g[y]["cost"] for y in years], C.GENDER_SHRINK)   # male vs female, claim cost
    return {
        "new_business": {"by_year": {str(y): est_nb[y] for y in years}, "frequency": freq, "size": size,
                         "uses_size": line in C.NB_WITH_SIZE, "training_share_of_expected_cost": share,
                         "factor_new_business": float(np.exp(log_e) * k), "factor_renewal": float(k)},
        "gender": {"by_year": {str(y): est_g[y] for y in years}, "cost": gender,
                   "applied": line in C.GENDER_LINES, "male_vs_female": float(np.exp(gender["posterior"]))},
    }


def apply_adjustments(pred: np.ndarray, rows: pd.DataFrame, adj: dict, line: str) -> np.ndarray:
    """New-business factors on every row, then the gender factor with the line total unchanged."""
    nb = adj["new_business"]
    out = pred * np.where(rows["_new_business"].to_numpy(dtype=bool), nb["factor_new_business"], nb["factor_renewal"])
    if line in C.GENDER_LINES:
        g = out * np.where(is_male(rows), adj["gender"]["male_vs_female"], 1.0)
        out = g * out.sum() / g.sum()             # moves price between men and women only
    return out


# =================================================================================================
# Out-of-time check and final fit
# =================================================================================================
def holdout(line: str, train: pd.DataFrame, features: list[str]) -> tuple[list[dict], dict, dict]:
    """Fit on the years before HOLDOUT_YEAR and score the claims reported for HOLDOUT_YEAR."""
    year = C.HOLDOUT_YEAR
    past = train.loc[train["year"] < year].reset_index(drop=True)
    ho = train.loc[train["year"] == year].reset_index(drop=True)
    y = ho["claim_cost"].to_numpy(dtype=float)
    preds = {"developed prior alone": prior_alone(past, ho, line, year)}
    plain = fit_model(past, line, features, year, use_prior=False, refit=False)
    preds["LightGBM without prior"] = plain.predict(ho, reported=True)
    transfer = fit_model(past, line, features, year, use_prior=True, refit=False)
    preds["transfer LightGBM"] = transfer.predict(ho, reported=True)
    # Replay of the adjustments: estimated on the fit year alone, applied to the holdout year.
    mu, y_capped = out_of_fold(past, line, features, year, transfer.lf.model.n_trees)
    adj = estimate_adjustments(line, past, mu, y_capped, sorted(int(v) for v in past["year"].unique()))
    preds["transfer LightGBM + adjustments"] = apply_adjustments(preds["transfer LightGBM"], ho, adj, line)
    trees = {"LightGBM without prior": plain.lf.model.n_trees}
    trees.update({k: transfer.lf.model.n_trees for k in preds if k.startswith("transfer")})
    rows = [{"line": line, "model": name, "trees": trees.get(name), **P.score(y, mu_)} for name, mu_ in preds.items()]
    return rows, {name: (y, mu_) for name, mu_ in preds.items()}, adj


def final(line: str, train: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    """Fit on every training year at the 2025 cost level, price 2025, apply the adjustments."""
    year = C.TEST_YEAR
    fitted = fit_model(train, line, features, year, use_prior=True, refit=True)
    raw = fitted.predict(test, reported=False)                    # ultimate cost: completeness 1
    mu, y_capped = out_of_fold(train, line, features, year, fitted.lf.model.n_trees)
    adj = estimate_adjustments(line, train, mu, y_capped, list(C.TRAIN_YEARS))
    pred = apply_adjustments(raw, test, adj, line)
    # Training cost per year on cover at the 2025 level, to compare with the 2025 prediction.
    tr = P.index_sum_insured(train, year, C.SI_INDEX.get(line, 0.0))
    seen = (tr["exposure"].to_numpy(dtype=float) * P.completeness(tr)).sum()
    return raw, pred, fitted, adj, float(target(tr, line, year).sum() / seen)


# =================================================================================================
# Driver
# =================================================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lines", nargs="*", default=list(C.LINES))
    ap.add_argument("--skip-holdout", action="store_true", help="only the final fit and the submission")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    ho_rows, ho_pool, adjustments, summary, preds = [], {}, {}, {}, []

    for line in a.lines:
        with H.Timer(f"{line}: reading the model-ready tables and the prior"):
            train, test, features = load_line(line)
            dropped = [f for f in P.feature_columns(test) if f not in features and not f.startswith("_")]
            print(f"   {len(train):,} training rows, {len(test):,} rows to price, {len(features)} features"
                  + (f" (dropped, weather of the year: {', '.join(dropped)})" if dropped else ""))

        if not a.skip_holdout:
            with H.Timer(f"{line}: out-of-time check (fit before {C.HOLDOUT_YEAR}, score {C.HOLDOUT_YEAR})"):
                rows, pool, replay = holdout(line, train, features)
                ho_rows += rows
                for name, pair in pool.items():
                    ho_pool.setdefault(name, []).append(pair)
                adjustments.setdefault(line, {})["replay_fit_year_only"] = replay
                for r in rows:
                    print(f"   {r['model']:<32} D2 {r['d2']:.4f} (level known {r['d2_level_known']:.4f}), "
                          f"actual / expected {r['a_over_e']:.3f}")

        with H.Timer(f"{line}: final fit on {min(C.TRAIN_YEARS)}-{max(C.TRAIN_YEARS)}, price {C.TEST_YEAR}"):
            raw, pred, fitted, adj, train_avg = final(line, train, test, features)
            adjustments.setdefault(line, {})["final"] = adj
            fitted.lf.model.save(MODELS / f"transfer_{line}")
            fitted.lf.model.importance().to_csv(OUT / f"importance_{line}.csv", index=False)
            preds.append(pd.DataFrame({"row_id": test["row_id"].astype(str).to_numpy(), "line": line,
                                       "new_business": test["_new_business"].to_numpy(),
                                       "gender": test["_gender"].to_numpy(), "model": raw, "prediction": pred}))
            expo = test["exposure"].to_numpy(dtype=float)
            nb = adj["new_business"]
            summary[line] = {
                "features": len(features), "trees": fitted.lf.model.n_trees, "base": fitted.lf.model.base,
                "prior_centre": fitted.centre, "balance": fitted.lf.balance, "large_loss_cap": fitted.lf.cap,
                "large_loss_load": fitted.lf.load, "rows_2025": int(len(test)),
                "new_business_share_2025": float(np.average(test["_new_business"], weights=expo)),
                "factor_new_business": nb["factor_new_business"], "factor_renewal": nb["factor_renewal"],
                "male_vs_female": adj["gender"]["male_vs_female"], "gender_applied": line in C.GENDER_LINES,
                "cost_per_year_on_cover_2025_model": float(raw.sum() / expo.sum()),
                "cost_per_year_on_cover_2025_adjusted": float(pred.sum() / expo.sum()),
                "training_cost_per_year_on_cover_at_2025_level": train_avg,
            }
            print(f"   {fitted.lf.model.n_trees} trees, large-loss load {fitted.lf.load:.3f}; new business x"
                  f"{nb['factor_new_business']:.3f}, renewals x{nb['factor_renewal']:.3f}; men / women "
                  f"{adj['gender']['male_vs_female']:.3f}" + ("" if line in C.GENDER_LINES else " (not applied)"))
            print(f"   2025 cost per year on cover {pred.sum() / expo.sum():,.1f} (model alone "
                  f"{raw.sum() / expo.sum():,.1f}; training average at the 2025 level {train_avg:,.1f})")

    if ho_rows:
        for name, pairs in ho_pool.items():               # all lines pooled: the competition score
            y = np.concatenate([p[0] for p in pairs])
            mu = np.concatenate([p[1] for p in pairs])
            ho_rows.append({"line": "all lines", "model": name, "trees": None, **P.score(y, mu)})
        table = pd.DataFrame(ho_rows)
        table.to_csv(OUT / "holdout.csv", index=False)
        print(f"\nOut-of-time check, {C.HOLDOUT_YEAR} (reported claims), D2:")
        print(table.pivot(index="model", columns="line", values="d2").reindex(columns=[*a.lines, "all lines"])
              .reindex(list(ho_pool)).round(4).to_string())
    (OUT / "adjustments.json").write_text(json.dumps(adjustments, indent=1), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    if set(a.lines) != set(C.LINES):
        print("\nSubmission not written: only some lines were fitted.")
        return 0
    allp = pd.concat(preds, ignore_index=True)
    PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    allp.to_parquet(PREDICTIONS, index=False)              # model output and adjusted price per row
    expected = H.read_table("sample_submission", "clean", columns=["row_id"])["row_id"].astype(str)
    sub = allp.set_index("row_id").loc[expected.to_numpy(), ["prediction"]].reset_index()
    for name, ok in P.check_submission(sub, expected).items():   # raises if any check fails
        print(f"   [{'PASS' if ok else 'FAIL'}] {name}")
    best = [r for r in ho_rows if r["line"] == "all lines" and r["model"] == "transfer LightGBM + adjustments"]
    evidence = f"forward 2024 D2 {best[0]['d2']:.4f} (all lines)" if best else ""
    S.save(sub, C.ROOT, "transfer_lightgbm", "05_lightgbm_model.py",
           "LightGBM on top of the developed prior (with renewal_count), no weather of the year, "
           "new-business and gender factors", evidence)
    return 0


if __name__ == "__main__":
    sys.exit(main())
