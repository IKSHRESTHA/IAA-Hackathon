"""Build the robust stacked submission (v3) and its 2025 diagnostics.

Lessons from v2 (public leaderboard below v1): two meta-learner ingredients were not
supported by the raw data and over-adjusted 2025 rows (part-year rows up to 2.2x; four-
year claim histories pushed beyond the validated three-year range). v3 keeps only what
is robust: experts combined by a non-negative log-linear pool (or an equal-weight
geometric mean) with a per-line level learnt one year ahead. No exposure term, no
credibility term, no expert contrasts.

Usage: python scripts/experiments/build_robust_submission.py [--method nn|geo] [--with-fsp]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/experiments")
import stack_and_report as sr  # noqa: E402

from iaa_pricing import metrics as mt  # noqa: E402
from iaa_pricing.config import load_config  # noqa: E402
from iaa_pricing.guardrails import validate_submission  # noqa: E402
from iaa_pricing.io import read_table  # noqa: E402
from iaa_pricing.stacking import LogLinearPool  # noqa: E402

NN_EXPERTS = ["total", "glm", "peril", "fs_gamma", "fs_lognorm"]
GEO_MEMBERS = ["total", "fs_gamma", "fs_lognorm", "peril"]


def with_geo(d: pd.DataFrame, members: list[str]) -> pd.DataFrame:
    d = d.copy()
    d["geo"] = np.exp(np.mean([np.log(d[m].to_numpy()) for m in members], axis=0))
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["nn", "geo"], default="nn")
    ap.add_argument("--with-fsp", action="store_true")
    ap.add_argument("--members", nargs="*", default=None, help="geometric-mean members (overrides default)")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args(argv)
    cfg = load_config("configs/week1.yaml")
    finals, cv_parts = [], {2023: [], 2024: []}
    for line in sr.LINES:
        h23, h24, fin = (sr.load(cfg, line, s) for s in ("ho2023", "ho2024", "final"))
        nn_ex = NN_EXPERTS + (["fs_peril"] if a.with_fsp else [])
        geo_m = a.members or (GEO_MEMBERS + (["fs_peril"] if a.with_fsp else []))
        h23, h24, fin = (with_geo(d, geo_m) for d in (h23, h24, fin))
        ex = nn_ex if a.method == "nn" else ["geo"]

        def pool():
            return LogLinearPool(ex, a=None, exposure_term=False, nonneg=True)

        for fit_d, sc_d, H in ((h23, h24, 2024), (h24, h23, 2023)):     # cross-year check
            mu = pool().fit(fit_d).predict(sc_d)
            cv_parts[H].append(pd.DataFrame({"y": sc_d["y"].to_numpy(), "mu": mu}))
        m = pool().fit(pd.concat([h23, h24], ignore_index=True))
        print(line, {k: round(v, 3) for k, v in m.coefficients().items()})
        f = fin[["row_id", "line", "exposure", "total"]].copy()
        f["prediction"] = m.predict(fin, reported=False)
        finals.append(f)
    cv = {H: mt.d2_tweedie(pd.concat(p).y, pd.concat(p).mu, 1.5) for H, p in cv_parts.items()}
    print(f"cross-year pooled D2: 2023={cv[2023]:.4f} 2024={cv[2024]:.4f} mean={np.mean(list(cv.values())):.4f}")

    fin = pd.concat(finals, ignore_index=True)
    sub = fin[["row_id", "prediction"]]
    expected = read_table(cfg.data_dir, "sample_submission", columns=["row_id"])["row_id"]
    bad = [c for c in validate_submission(sub, expected) if not c.passed]
    if bad or (sub["prediction"] <= 0).any():
        raise RuntimeError(f"integrity failed: {bad}")
    sub = sub.set_index("row_id").loc[expected.to_numpy()].reset_index()
    tag = a.tag or (f"v3_{a.method}" + ("_fsp" if a.with_fsp else ""))
    path = cfg.output_dir / f"submission_week1_{tag}.csv"
    sub.to_csv(path, index=False)
    print("written", path)
    pp = fin.groupby("line").apply(lambda g: pd.Series({
        "pp_v3": g.prediction.sum() / g.exposure.sum(), "pp_total": g.total.sum() / g.exposure.sum()}),
        include_groups=False)
    print(pp.round(1).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
