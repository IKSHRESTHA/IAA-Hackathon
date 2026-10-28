"""Run the steps of one or both rounds, each in its own process, from the repository root.

Data steps (they need the case-study files in each round's data/raw/):
    01_data_reading.py, 02_data_cleaning.py, 03_feature_engineering.py
Model steps (they use data/clean and data/features, which are already in the folder):
    1st round: 04_lightgbm_model.py, then the round's winning ensemble: 05_geo_experts.py (slow:
               six experts on three fit sets) and 06_geo_ensemble.py
    2nd round: 04_developed_prior.py (reads the 1st round's data), 05_lightgbm_model.py, then the
               same ensemble on developed priors: 06_geo_priors.py, 07_geo_experts.py, 08_geo_ensemble.py,
               and 09_rebase_to_best.py (the best public file's levels, the ensemble's ranking inside)

Usage:
    python run_all.py                     # data steps of both rounds
    python run_all.py --models            # data steps, then the model steps
    python run_all.py --models-only       # model steps only (no raw files needed)
    python run_all.py --rounds "2nd round" --models-only
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = ("01_data_reading.py", "02_data_cleaning.py", "03_feature_engineering.py")
MODEL_STEPS = {
    "1st round": ("04_lightgbm_model.py", "05_geo_experts.py", "06_geo_ensemble.py"),
    "2nd round": ("04_developed_prior.py", "05_lightgbm_model.py", "06_geo_priors.py", "07_geo_experts.py",
                  "08_geo_ensemble.py", "09_rebase_to_best.py"),
}


def run(rnd: str, step: str, env: dict) -> int:
    """Run one step inside its round folder (the scripts expect it as working directory)."""
    t0 = time.time()
    print(f"\n######## {rnd} / {step}")
    r = subprocess.run([sys.executable, step], cwd=ROOT / rnd, env=env)
    if r.returncode:
        print(f"{rnd} / {step} failed (exit {r.returncode})")
    else:
        print(f"######## {rnd} / {step} done in {time.time() - t0:.0f}s")
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", nargs="*", default=["1st round", "2nd round"])
    ap.add_argument("--models", action="store_true", help="run the model steps after the data steps")
    ap.add_argument("--models-only", action="store_true", help="run only the model steps")
    a = ap.parse_args()
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONUTF8="1")    # same bytes on every run
    for rnd in a.rounds:
        steps = [] if a.models_only else list(STEPS)
        if a.models or a.models_only:
            steps += MODEL_STEPS[rnd]
        if not a.models_only:
            present = [p.name for p in (ROOT / rnd / "data" / "raw").glob("*.parquet")]
            if not present:
                print(f"{rnd}: data/raw is empty - copy the case-study files there first (see data/raw/README.md),"
                      f" or use --models-only")
                return 1
        for step in steps:
            code = run(rnd, step, env)
            if code:
                return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
