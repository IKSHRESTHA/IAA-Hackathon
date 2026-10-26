"""Step 6 - the 1st round's winning ensemble: the experts' geometric mean, recalibrated per line.

For each line the expert predictions of step 5 are combined row by row by an equal-weight
geometric mean. A Tweedie GLM with two parameters (``geo_experts.LinePool``) then sets the line's
level and slope. It is fitted on the 2023 and 2024 back-test predictions against the claims
reported for those years, and applied to the 2025 predictions.

Checks, before the submission is written:

* every expert and the plain geometric mean, scored on each back-test year (reported basis);
* the recalibration fitted on one back-test year and scored on the other, both ways, all lines
  pooled: the out-of-time estimate of the ensemble (the original reported 0.1380 for 2023 and
  0.1367 for 2024).

The step reads the saved expert predictions only, so it takes seconds. ``--members`` tries
another set of experts.

Output: the next version in ``submissions/`` (``v<N>_geo_ensemble[_<tag>].csv``), ``reports/06_geo/``
(scores.csv, recalibration.csv, summary.json).

Usage: python 06_geo_ensemble.py [--members total fs_gamma fs_lognorm peril fs_peril glm] [--tag NAME]
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
import submissions as S  # noqa: E402

OUT = C.REPORTS / "06_geo"
EXPERTS_DIR = C.DATA / "experts"
RECAL = "recalibrated geo (fit on the other year)"


def load(line: str, name: str, members: list[str]) -> pd.DataFrame:
    """Saved predictions of one line and fit set, with the members' geometric mean."""
    path = EXPERTS_DIR / f"{line}_{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run 05_geo_experts.py first")
    d = pd.read_parquet(path)
    missing = [m for m in members if m not in d]
    if missing:
        raise RuntimeError(f"{path.name} has no predictions for {missing}")
    d["geo"] = G.geo_mean(d, members)
    return d


def scored(line: str, year: int, model: str, y: np.ndarray, mu: np.ndarray) -> dict:
    return {"line": line, "year": int(year), "model": model, **P.score(y, mu)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--members", nargs="*", default=list(C.GEO_MEMBERS))
    ap.add_argument("--tag", default="", help="suffix of the submission file name")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, recal, finals, pooled = [], [], [], {}

    for line in C.LINES:
        with H.Timer(f"{line}: geometric mean of {len(a.members)} experts, recalibration"):
            h23, h24, fin = (load(line, s, a.members) for s in ("ho2023", "ho2024", "final"))
            for d in (h23, h24):                                 # each expert on its back-test year
                year, y, share = int(d["year"].iloc[0]), d["y"].to_numpy(dtype=float), G.reported_share(d)
                for m in [*a.members, "geo"]:
                    mu = d[m].to_numpy(dtype=float) * share       # what would be reported by now
                    rows.append(scored(line, year, m, y, mu))
                    pooled.setdefault((year, m), []).append((y, mu))
            for fit_d, sc_d in ((h23, h24), (h24, h23)):         # recalibration, out of time
                mu = G.LinePool(C.GEO_POOL_ALPHA).fit(fit_d).predict(sc_d, reported=True)
                year, y = int(sc_d["year"].iloc[0]), sc_d["y"].to_numpy(dtype=float)
                rows.append(scored(line, year, RECAL, y, mu))
                pooled.setdefault((year, RECAL), []).append((y, mu))
            pool = G.LinePool(C.GEO_POOL_ALPHA).fit(pd.concat([h23, h24], ignore_index=True))
            pred = pool.predict(fin, reported=False)             # 2025 at the ultimate basis
            finals.append(pd.DataFrame({"row_id": fin["row_id"].astype(str).to_numpy(), "prediction": pred}))
            expo = fin["exposure"].sum()
            recal.append({"line": line, **pool.coefficients, "geo_per_year_on_cover_2025": fin["geo"].sum() / expo,
                          "cost_per_year_on_cover_2025": pred.sum() / expo})
            print(f"   b0 {pool.coefficients['b0']:+.3f}, b1 {pool.coefficients['b1']:.3f}; 2025 cost per year on "
                  f"cover {pred.sum() / expo:,.1f} (plain geometric mean {fin['geo'].sum() / expo:,.1f})")

    for (year, m), parts in pooled.items():                      # all lines pooled: the competition score
        rows.append(scored("all lines", year, m, np.concatenate([p[0] for p in parts]),
                           np.concatenate([p[1] for p in parts])))
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "scores.csv", index=False)
    pd.DataFrame(recal).to_csv(OUT / "recalibration.csv", index=False)
    table = scores.loc[scores["line"] == "all lines"].pivot(index="model", columns="year", values="d2")
    table["mean"] = table.mean(axis=1)
    order = [*a.members, "geo", RECAL]
    print("\nBack-test D2, all lines pooled (reported claims):")
    print(table.reindex(order).round(4).to_string())

    expected = H.read_table("sample_submission", "clean", columns=["row_id"])["row_id"].astype(str)
    sub = pd.concat(finals, ignore_index=True).set_index("row_id").loc[expected.to_numpy()].reset_index()
    for name, ok in P.check_submission(sub, expected).items():   # raises if any check fails
        print(f"   [{'PASS' if ok else 'FAIL'}] {name}")
    cross = table.loc[RECAL]
    path = S.save(sub, C.ROOT, "geo_ensemble" + (f"_{a.tag}" if a.tag else ""), "06_geo_ensemble.py",
                  f"geometric mean of {', '.join(a.members)}, recalibrated per line (round 1 winning method)",
                  f"cross-year D2 2023 {cross[2023]:.4f}, 2024 {cross[2024]:.4f} (all lines)")
    summary = {"members": a.members, "cross_year_d2": {str(k): float(v) for k, v in cross.items()},
               "recalibration": recal, "submission": path.name}
    (OUT / f"summary{'_' + a.tag if a.tag else ''}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\nOut-of-time D2 of the ensemble: 2023 {cross[2023]:.4f}, 2024 {cross[2024]:.4f}, mean {cross['mean']:.4f}")
    print(f"{len(sub):,} rows -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
