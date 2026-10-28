"""Step 12: the final file: step 11's levels, with the ranking inside each segment improved.

v12 (step 11) sets the level of every segment. Its ranking inside a segment is v5's: a 50/50
geometric blend of the best public file and step 8's ensemble. This step changes that ranking in
two ways, both justified outside the public leaderboard, and then puts every segment back at the
levels version's total, so the levels stay exactly as step 11 set them.

1. This-year weather. The best file's buildings and contents model uses the weather of the year,
   and 2025 lies far outside the two training years (rainfall anomaly -128 mm against +-3 mm), so
   every 2025 row lands in the trees' edge leaves. The pricing repository refitted that model
   without this-year weather (and on the all-rows developed prior); in its policy-grouped 5-fold
   cross-validation the refit ranks better (buildings 0.0972 -> 0.0992, contents 0.1072 -> 0.1092,
   level fixed). Step 8's ensemble leaves this-year weather out already. With ``--no-weather``
   the best file's half of the buildings and contents blend is that refit (motor is unchanged).
2. Motor spread. Read through step 11's levels, the two public power-calibration files (0.16486
   and 0.16472) say motor prices are too compressed: risk differences inside a segment are about
   1.1 to 1.2 times wider on the log scale than the best file shows (buildings and contents: no
   change). The pricing repository's 2024 back-test pointed the other way (0.91), so the default
   is a mild 1.05: on unseen rows it gains about +0.0003 if the true widening is 1.1 and +0.0005
   if it is 1.2, and costs nothing if the best file's spread is right (v12's 50/50 blend is a
   little narrower than the best file, and 1.05 restores that).

Usage: python 12_final_assembly.py [--levels v12] [--weight 0.5] [--no-weather] [--spread motor=1.05]
       [--dry-run]   (prints the comparison with the levels version, saves nothing)
"""
from __future__ import annotations

import argparse
import importlib
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

S9 = importlib.import_module("09_rebase_to_best")    # segments, blend, rebase
S10 = importlib.import_module("10_level_probe")      # load_version
REF = C.DATA / "reference"
EXPERTS_DIR = C.DATA / "experts"
BEST = REF / "best_public_gbm100_submission.csv"
NO_WEATHER = REF / "best_no_year_weather_submission.csv"
NO_WEATHER_LINES = ("buildings", "contents")      # the refit leaves motor unchanged


def read_sub(path) -> pd.Series:
    return pd.read_csv(path, dtype={"row_id": str}).set_index("row_id")["prediction"]


def parse_spread(items: list[str] | None) -> dict[str, float]:
    """``motor=1.05`` -> {"motor": 1.05}; lines not named keep their spread (1.0)."""
    out = {}
    for item in items or []:
        line, value = item.split("=")
        if line not in C.LINES:
            raise ValueError(f"unknown line {line}")
        out[line] = float(value)
    return out


def assemble(levels: pd.Series, weight: float, no_weather: bool, spread: dict[str, float]) -> tuple[pd.DataFrame, list]:
    """The 2025 rows: blended ranking, optional wider motor, every segment at the levels version's total."""
    members = json.loads((C.REPORTS / "08_geo" / "summary.json").read_text(encoding="utf-8"))["members"]
    best, nw = read_sub(BEST), read_sub(NO_WEATHER) if no_weather else None
    parts, report = [], []
    for line in C.LINES:
        fin = pd.read_parquet(EXPERTS_DIR / f"{line}_final.parquet")
        ids = fin["row_id"].astype(str)
        ref = (nw if nw is not None and line in NO_WEATHER_LINES else best).loc[ids].to_numpy(dtype=float)
        seg = S9.segments(fin, line)
        pred = S9.blend(ref, G.geo_mean(fin, members[line]), weight)          # the ranking inside segments
        g = spread.get(line, 1.0)
        if g != 1.0:
            pred = pred ** g                                                  # wider risk differences (log scale)
        target = levels.loc[ids].to_numpy(dtype=float)
        pred = S9.rebase(pred, target, seg)                                   # each segment at the levels total
        parts.append(pd.DataFrame({"row_id": ids.to_numpy(), "prediction": pred}))
        d = np.log(pred / target)
        report.append({"line": line, "best_file": "no this-year weather" if nw is not None and line in NO_WEATHER_LINES
                       else "best public", "weight_best": weight, "spread": g,
                       "log_corr_with_levels_version": float(np.corrcoef(np.log(pred), np.log(target))[0, 1]),
                       "sd_log_change": float(d.std()), "max_abs_log_change": float(np.abs(d).max())})
    return pd.concat(parts, ignore_index=True), report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", default="v12", help="version whose segment totals are kept (default v12)")
    ap.add_argument("--weight", type=float, default=0.5, help="weight of the best file's ranking (0 to 1)")
    ap.add_argument("--no-weather", action="store_true", help="best file's buildings and contents without this-year weather")
    ap.add_argument("--spread", nargs="*", default=["motor=1.05"], help="power inside segments per line, e.g. motor=1.05")
    ap.add_argument("--dry-run", action="store_true", help="compare with the levels version, save nothing")
    a = ap.parse_args(argv)
    levels, levels_name = S10.load_version(a.levels)
    spread = parse_spread(a.spread)
    sub, report = assemble(levels, a.weight, a.no_weather, spread)
    print(pd.DataFrame(report).round(4).to_string(index=False))
    expected = H.read_table("sample_submission", "clean", columns=["row_id"])["row_id"].astype(str)
    sub = sub.set_index("row_id").loc[expected.to_numpy()].reset_index()
    P.check_submission(sub, expected)                                         # raises if any check fails
    if a.dry_run:
        lv = levels.loc[sub["row_id"]].to_numpy(dtype=float)
        print(f"max relative difference from {levels_name}: {np.max(np.abs(sub['prediction'].to_numpy() / lv - 1)):.2e}")
        return 0
    tags = (["nowx"] if a.no_weather else []) + [f"{k}{v:.2f}".replace(".", "p") for k, v in spread.items() if v != 1.0]
    tags += [] if a.weight == 0.5 else [f"w{round(100 * a.weight):02d}"]
    what = (f"segment totals of {levels_name}; ranking inside: {a.weight:.2f} best file"
            + (" (buildings and contents without this-year weather)" if a.no_weather else "")
            + f" + {1 - a.weight:.2f} step-8 ensemble" + "".join(f"; {k} spread x{v:.2f}" for k, v in spread.items() if v != 1.0))
    S.save(sub, C.ROOT, "final_" + ("_".join(tags) or "levels_only"), "12_final_assembly.py", what,
           f"levels as {a.levels}; ranking changes from cross-validation and the power-calibration scores")
    return 0


if __name__ == "__main__":
    sys.exit(main())
