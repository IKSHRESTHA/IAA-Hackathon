"""Step 8 - the 1st round's winning ensemble for this round, and its 2025 submission.

1. Members, per line, chosen by cross-validation. The 1st round's winner averaged six experts.
   Here some of them are weak (two years give few claims to learn claim sizes from), and a weak
   member drags a geometric mean down. Every combination of the six experts (63) is scored on the
   cross-validation predictions, by its D^2 with the line's level known in each year (the level is
   set in the next step), and the best mean over the two years wins. ``--members`` fixes them.
2. The members' geometric mean, recalibrated (``geo_experts.LinePool``): b0 + b1 x log(geo rate / L).
   The 1st round learnt both one year ahead, on two back-test years. Here two training years give
   one back-test, and 2024 alone would set the 2025 level badly (buildings came in at 0.59 of what
   2023 predicted, after a 2023 spike in fire costs). So the pool is fitted on the cross-validation
   predictions of both years. The slope test (``slope_check.csv``) found no consistent gain from
   b1, so it stays at 1 (``config.GEO_FIT_SLOPE``) and only the level b0 is fitted.
3. New business and gender: the effects the ensemble still misses, measured on its out-of-fold
   predictions with step 5's method (year by year, pooled, shrunk towards no effect).

Checks: every expert, the members' plain mean and the recalibrated ensemble on the cross-validation
predictions (by year, all lines pooled); the recalibration fitted on one year's folds and scored on
the other's; the forward back-test 2023 -> 2024 next to step 5's transfer LightGBM.

Output: the next version in ``submissions/`` (``v<N>_geo_ensemble_transfer[_<tag>].csv``),
``reports/08_geo/`` (members.csv, scores.csv, recalibration.csv, adjustments.json, summary.json).

Usage: python 08_geo_ensemble.py [--members total glm ...] [--tag NAME]
"""
from __future__ import annotations

import argparse
import importlib
import itertools
import json
import sys

import numpy as np
import pandas as pd

import config as C
import helpers as H

sys.path.insert(0, str(C.ROOT.parent))            # the shared modules sit in the folder above
import geo_experts as G  # noqa: E402
import pricing_lgbm as P  # noqa: E402
import submissions as S  # noqa: E402

S5 = importlib.import_module("05_lightgbm_model")    # step 5: the new-business and gender method
OUT = C.REPORTS / "08_geo"
EXPERTS_DIR = C.DATA / "experts"
RECAL = "recalibrated geo"
CROSS = "recalibrated geo (fit on the other year)"


def load(line: str, name: str) -> pd.DataFrame:
    path = EXPERTS_DIR / f"{line}_{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run 07_geo_experts.py first")
    return pd.read_parquet(path)


def with_geo(d: pd.DataFrame, members: list[str]) -> pd.DataFrame:
    return d.assign(geo=G.geo_mean(d, members))


def pool() -> G.LinePool:
    return G.LinePool(C.GEO_POOL_ALPHA, fit_slope=C.GEO_FIT_SLOPE)


def cross_year(cv: pd.DataFrame, fit_slope: bool = C.GEO_FIT_SLOPE) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Recalibration fitted on one year's folds, scored on the other year's: (claims, prediction)."""
    out = {}
    for fit_y, sc_y in ((2023, 2024), (2024, 2023)):
        sc = cv.loc[cv["year"] == sc_y]
        mu = G.LinePool(C.GEO_POOL_ALPHA, fit_slope=fit_slope).fit(cv.loc[cv["year"] == fit_y]).predict(sc)
        out[sc_y] = (sc["y"].to_numpy(dtype=float), mu)
    return out


def level_known(y: np.ndarray, mu: np.ndarray) -> float:
    """D^2 with the prediction rescaled to the actual total: how well the risks are ranked."""
    return P.d2_tweedie(y, mu * y.sum() / mu.sum())


def choose_members(line: str, cv: pd.DataFrame) -> tuple[list[str], list[dict]]:
    """The combination of experts that ranks this line's risks best, year by year (level known)."""
    recs = []
    for k in range(1, len(C.GEO_MEMBERS) + 1):
        for combo in itertools.combinations(C.GEO_MEMBERS, k):
            d = with_geo(cv, list(combo))
            mu = d["geo"].to_numpy(dtype=float) * G.reported_share(d)
            d2 = {y: level_known(d.loc[d["year"] == y, "y"].to_numpy(dtype=float), mu[(d["year"] == y).to_numpy()])
                  for y in C.TRAIN_YEARS}
            recs.append({"line": line, "members": "+".join(combo), "n": k, "d2_2023": d2[2023],
                         "d2_2024": d2[2024], "d2_mean": (d2[2023] + d2[2024]) / 2})
    best = max(recs, key=lambda r: r["d2_mean"])
    return best["members"].split("+"), recs


def slope_check(line: str, cv: pd.DataFrame) -> list[dict]:
    """Does a fitted slope b1 help? Fitted on one year's folds, scored on the other's, level known."""
    out = []
    plain = cv["geo"].to_numpy(dtype=float) * G.reported_share(cv)
    for year, (y, mu) in cross_year(cv, fit_slope=True).items():
        sel = (cv["year"] == year).to_numpy()
        out.append({"line": line, "year": year, "plain_geo": level_known(y, plain[sel]),
                    "with_fitted_slope": level_known(y, mu)})
    return out


def add(rows: list, pooled: dict, line: str, check: str, model: str, y: np.ndarray, mu: np.ndarray) -> None:
    """Score one model on one check, and keep the pair for the all-lines score."""
    rows.append({"line": line, "check": check, "model": model, **P.score(y, mu)})
    pooled.setdefault((check, model), []).append((y, mu))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--members", nargs="*", default=None, choices=list(C.GEO_MEMBERS),
                    help="fixed members for every line (default: chosen per line by cross-validation)")
    ap.add_argument("--tag", default="", help="suffix of the submission file name")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, pooled, recal, finals, adjustments, member_recs, chosen, slopes = [], {}, [], [], {}, [], {}, []

    for line in C.LINES:
        with H.Timer(f"{line}: members, geometric mean, recalibration, adjustments"):
            cv0, ho0, fin0 = (load(line, s) for s in ("cv", "ho2024", "final"))
            if a.members:
                members = list(a.members)
            else:
                members, recs = choose_members(line, cv0)
                member_recs += recs
            chosen[line] = members
            cv, ho, fin = (with_geo(d, members) for d in (cv0, ho0, fin0))
            slopes += slope_check(line, cv)
            for year in C.TRAIN_YEARS:                           # out-of-fold, by year
                part = cv.loc[cv["year"] == year]
                share, y = G.reported_share(part), part["y"].to_numpy(dtype=float)
                for m in [*C.GEO_MEMBERS, "geo"]:
                    add(rows, pooled, line, f"cv {year}", m, y, part[m].to_numpy(dtype=float) * share)
            for year, (y, mu) in cross_year(cv).items():         # recalibration across years
                add(rows, pooled, line, f"cv {year}", CROSS, y, mu)
            fitted = pool().fit(cv)
            mu_cv = fitted.predict(cv, reported=True)              # the ensemble's out-of-fold predictions
            for year in C.TRAIN_YEARS:
                sel = (cv["year"] == year).to_numpy()
                add(rows, pooled, line, f"cv {year}", RECAL, cv.loc[sel, "y"].to_numpy(dtype=float), mu_cv[sel])
            share, y = G.reported_share(ho), ho["y"].to_numpy(dtype=float)   # forward: fit 2023, price 2024
            for m in [*C.GEO_MEMBERS, "geo"]:
                add(rows, pooled, line, "forward 2024", m, y, ho[m].to_numpy(dtype=float) * share)
            add(rows, pooled, line, "forward 2024", RECAL, y, fitted.predict(ho, reported=True))

            adj = S5.estimate_adjustments(line, cv, mu_cv, cv["y_capped"].to_numpy(dtype=float), list(C.TRAIN_YEARS))
            adjustments[line] = adj
            model = fitted.predict(fin, reported=False)            # 2025 at the ultimate basis
            pred = S5.apply_adjustments(model, fin, adj, line)
            finals.append(pd.DataFrame({"row_id": fin["row_id"].astype(str).to_numpy(), "prediction": pred}))
            expo = fin["exposure"].sum()
            nb = adj["new_business"]
            recal.append({"line": line, "members": "+".join(members), **fitted.coefficients,
                          "factor_new_business": nb["factor_new_business"], "factor_renewal": nb["factor_renewal"],
                          "male_vs_female": adj["gender"]["male_vs_female"],
                          "model_per_year_on_cover_2025": model.sum() / expo,
                          "cost_per_year_on_cover_2025": pred.sum() / expo})
            print(f"   members {' + '.join(members)}; b0 {fitted.coefficients['b0']:+.3f}, b1 {fitted.coefficients['b1']:.3f}; "
                  f"new business x{nb['factor_new_business']:.3f}, renewals x{nb['factor_renewal']:.3f}, "
                  f"men / women {adj['gender']['male_vs_female']:.3f}; 2025 per year on cover {pred.sum() / expo:,.1f}")

    for (check, m), parts in pooled.items():                     # all lines pooled: the competition score
        rows.append({"line": "all lines", "check": check, "model": m,
                     **P.score(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))})
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "scores.csv", index=False)
    if member_recs:
        pd.DataFrame(member_recs).sort_values(["line", "d2_mean"], ascending=[True, False]).to_csv(
            OUT / "members.csv", index=False)
    sl = pd.DataFrame(slopes).assign(gain=lambda d: d["with_fitted_slope"] - d["plain_geo"])
    sl.to_csv(OUT / "slope_check.csv", index=False)
    print("\nSlope test (fitted on the other year's folds, level known per line):")
    print(sl.round(4).to_string(index=False))
    pd.DataFrame(recal).to_csv(OUT / "recalibration.csv", index=False)
    (OUT / "adjustments.json").write_text(json.dumps(adjustments, indent=1), encoding="utf-8")
    table = scores.loc[scores["line"] == "all lines"].pivot(index="model", columns="check", values="d2")
    print("\nD2, all lines pooled (reported claims; cv = out of fold; geo = the chosen members' mean):")
    print(table.reindex([*C.GEO_MEMBERS, "geo", RECAL, CROSS]).round(4).to_string())
    step5 = C.REPORTS / "05_model" / "holdout.csv"
    if step5.exists():                                           # the single transfer LightGBM, same back-test
        s5 = pd.read_csv(step5)
        s5 = s5.loc[s5["line"] == "all lines"].set_index("model")["d2"]
        print("step 5, forward 2024: " + ", ".join(f"{k} {v:.4f}" for k, v in s5.items()))

    expected = H.read_table("sample_submission", "clean", columns=["row_id"])["row_id"].astype(str)
    sub = pd.concat(finals, ignore_index=True).set_index("row_id").loc[expected.to_numpy()].reset_index()
    for name, ok in P.check_submission(sub, expected).items():   # raises if any check fails
        print(f"   [{'PASS' if ok else 'FAIL'}] {name}")
    cv_all = [table.loc[RECAL, f"cv {y}"] for y in C.TRAIN_YEARS]
    picked = "; ".join(f"{l}: {'+'.join(m)}" for l, m in chosen.items())
    path = S.save(sub, C.ROOT, "geo_ensemble_transfer" + (f"_{a.tag}" if a.tag else ""), "08_geo_ensemble.py",
                  f"geometric mean on developed priors ({picked}), "
                  + ("level and slope" if C.GEO_FIT_SLOPE else "level only (slope 1)")
                  + " from cross-validation, new-business and gender factors",
                  f"out-of-fold D2 {np.mean(cv_all):.4f}, forward 2024 D2 {table.loc[RECAL, 'forward 2024']:.4f} (all lines)")
    summary = {"members": chosen, "all_lines_d2": {f"{c} / {m}": float(v) for (m, c), v in table.stack().items()},
               "recalibration": recal, "submission": path.name}
    (OUT / f"summary{'_' + a.tag if a.tag else ''}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\n{len(sub):,} rows -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
