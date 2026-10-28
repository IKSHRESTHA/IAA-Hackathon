"""Checks on a 2025 submission before it is uploaded.

* integrity: exact columns, every test row once, finite and strictly positive predictions;
* level: predicted 2025 pure premium per line against developed 2021-24 history (2025 cost
  level) and against the v1 submission;
* shape: correlation and ratio quantiles against v1 (large, unexplained moves get escalated).

Usage: python scripts/experiments/verify_final.py outputs/<new>.csv [outputs/submission_week1.csv]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from iaa_pricing.config import load_config
from iaa_pricing.guardrails import validate_submission
from iaa_pricing.io import read_table


def main(argv: list[str]) -> int:
    cfg = load_config("configs/week1.yaml")
    new = pd.read_csv(argv[0], dtype={"row_id": str})
    old = pd.read_csv(argv[1] if len(argv) > 1 else cfg.output_dir / "submission_week1.csv", dtype={"row_id": str})
    expected = read_table(cfg.data_dir, "sample_submission", columns=["row_id"])["row_id"]
    checks = validate_submission(new, expected)
    for c in checks:
        print(f"[{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}")
    if not all(c.passed for c in checks) or (new["prediction"] <= 0).any():
        print("BLOCK: submission integrity failed")
        return 1

    expo = []
    for line in ("motor", "buildings", "contents"):
        t = read_table(cfg.data_dir, f"test_{line}", columns=["row_id", "period_start", "period_end"])
        t["exposure"] = ((t["period_end"] - t["period_start"]).dt.days + 1) / 365.0
        t["line"] = line
        expo.append(t[["row_id", "line", "exposure"]])
    e = pd.concat(expo)
    d = e.merge(new.rename(columns={"prediction": "new"}), on="row_id").merge(
        old.rename(columns={"prediction": "old"}), on="row_id")
    hist = pd.read_csv(cfg.reports_dir / "developed_history.csv") if (cfg.reports_dir / "developed_history.csv").exists() else None
    print("\nline        PP_new   PP_v1   new/v1   corr(log)   ratio p05 / p50 / p95   PP_history(2025 level)")
    for line, g in d.groupby("line"):
        pp_new, pp_old = g["new"].sum() / g["exposure"].sum(), g["old"].sum() / g["exposure"].sum()
        r = g["new"] / g["old"]
        corr = np.corrcoef(np.log(g["new"]), np.log(g["old"]))[0, 1]
        ph = ""
        if hist is not None and "line" in hist:
            h = hist[hist["line"] == line]
            ph = f"{h['developed_cost'].sum() / h['exposure'].sum():8.1f}"
        print(f"{line:10s} {pp_new:8.1f} {pp_old:8.1f} {pp_new / pp_old:7.3f} {corr:10.3f}     "
              f"{r.quantile(.05):5.2f} / {r.quantile(.5):5.2f} / {r.quantile(.95):5.2f}       {ph}")
    print(f"\nrows {len(new):,}; min {new['prediction'].min():.3f}; max {new['prediction'].max():,.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
