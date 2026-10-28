"""Rebuild both submissions from the cached expert predictions (about a minute, no competition data needed).

    python rebuild_from_cache.py

writes outputs/submission_week1_v4_geo6.csv and outputs/pick3_ens_geo7_moe.csv and compares them with
the submitted files in submissions/.

It runs the same steps as ``scripts/experiments/build_robust_submission.py --method geo`` (the script that
made the submissions), on cache/experts7_<line>.parquet: the predictions of the seven experts exactly as
they were on 22 October 2026, for the three fit-sets of each line

    ho2023  experts trained on 2021-22, predicting the 2023 rows   (out-of-time back-test)
    ho2024  experts trained on 2021-23, predicting the 2024 rows   (out-of-time back-test)
    final   experts trained on 2021-24, predicting the 2025 rows   (the submission)

For each line: the geometric mean of the members' predictions is recalibrated (level and slope) with a
Tweedie (p = 1.5) GLM fitted on the 2023 and 2024 back-test predictions against the reported claims
(`iaa_pricing.stacking.LogLinearPool`), then applied to the 2025 predictions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from iaa_pricing import metrics as mt  # noqa: E402
from iaa_pricing.stacking import LogLinearPool  # noqa: E402

LINES = ["motor", "buildings", "contents"]
CORE6 = ["total", "fs_gamma", "fs_lognorm", "peril", "fs_peril", "glm"]
MODELS = {                                   # output file -> geometric-mean members
    "submission_week1_v4_geo6.csv": CORE6,                 # v4: the six experts
    "pick3_ens_geo7_moe.csv": CORE6 + ["fs_moe"],          # pick3: the six + claim-size mixture of experts
}


def with_geo(d: pd.DataFrame, members: list[str]) -> pd.DataFrame:
    """Equal-weight geometric mean of the members' predictions (as build_robust_submission.with_geo)."""
    d = d.copy()
    d["geo"] = np.exp(np.mean([np.log(d[m].to_numpy()) for m in members], axis=0))
    return d


def pool() -> LogLinearPool:
    """Level + slope on log(geo rate / set average rate); slope >= 0; offset log(exposure x level)."""
    return LogLinearPool(["geo"], a=None, exposure_term=False, nonneg=True)


def build(members: list[str], cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    finals, cv_parts = [], {2023: [], 2024: []}
    for line in LINES:
        c = cache[line]
        h23, h24, fin = (with_geo(c[c["set"] == s].reset_index(drop=True), members)
                         for s in ("ho2023", "ho2024", "final"))
        for fit_d, sc_d, year in ((h23, h24, 2024), (h24, h23, 2023)):        # cross-year check
            cv_parts[year].append(pd.DataFrame({"y": sc_d["y"].to_numpy(), "mu": pool().fit(fit_d).predict(sc_d)}))
        m = pool().fit(pd.concat([h23, h24], ignore_index=True))               # final fit on both years
        print(f"  {line:<10} calibration {({k: round(v, 3) for k, v in m.coefficients().items()})}")
        finals.append(pd.DataFrame({"row_id": fin["row_id"].to_numpy(), "order": fin["sub_order"].to_numpy(),
                                    "line": line, "exposure": fin["exposure"].to_numpy(),
                                    "prediction": m.predict(fin, reported=False)}))
    cv = {y: mt.d2_tweedie(pd.concat(p).y, pd.concat(p).mu, 1.5) for y, p in cv_parts.items()}
    print(f"  cross-year D2 (fit one back-test year, score the other): 2023 {cv[2023]:.4f}, "
          f"2024 {cv[2024]:.4f}, mean {np.mean(list(cv.values())):.4f}")
    out = pd.concat(finals, ignore_index=True).sort_values("order")       # sample-submission row order
    rate = out.groupby("line").apply(lambda g: g.prediction.sum() / g.exposure.sum(), include_groups=False)
    print("  2025 predicted cost per year on cover: " + ", ".join(f"{k} {v:,.1f}" for k, v in rate.items()))
    return out[["row_id", "prediction"]].reset_index(drop=True)


def main() -> int:
    cache = {line: pd.read_parquet(ROOT / "cache" / f"experts7_{line}.parquet") for line in LINES}
    (ROOT / "outputs").mkdir(exist_ok=True)
    for name, members in MODELS.items():
        print(f"\n{name}: geometric mean of {', '.join(members)}")
        sub = build(members, cache)
        if (sub["prediction"] <= 0).any() or sub["prediction"].isna().any() or sub["row_id"].duplicated().any():
            raise RuntimeError("integrity check failed")
        path = ROOT / "outputs" / name
        sub.to_csv(path, index=False)
        print(f"  {len(sub):,} rows -> {path}")
        ref = ROOT / "submissions" / name
        if ref.exists():
            r = pd.read_csv(ref)
            same_rows = r["row_id"].equals(sub["row_id"])
            diff = np.abs(sub["prediction"].to_numpy() / r["prediction"].to_numpy() - 1).max()
            print(f"  against the submitted file: same rows in the same order {same_rows}, "
                  f"largest relative difference {diff:.1e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
