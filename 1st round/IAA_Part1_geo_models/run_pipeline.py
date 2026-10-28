"""Run the full pipeline of the two models from the competition files, step by step.

    python run_pipeline.py                     # every step: about 2.5 hours on 2 CPU cores, 7-8 GB RAM
    python run_pipeline.py --from twopart      # resume from a step
    python run_pipeline.py --only ensembles    # one step

Steps (each runs one script of the original code, from this folder, with src/ on the path):

  features   scripts/build_features.py                     cleaning, features, claims history (~3 min)
  experts    scripts/experiments/run_experts.py            total, glm, peril (+ frequency, large-loss helpers)
  twopart    scripts/experiments/run_twopart.py            fs_gamma, fs_lognorm (frequency x severity)
  peril      scripts/experiments/run_twopart.py --peril    fs_peril (frequency x severity per peril)
  moe        scripts/experiments/run_moe.py                fs_moe (claim-size mixture of experts; needs twopart)
  ensembles  scripts/experiments/build_robust_submission.py --method geo   both submissions, then checks

Each expert step fits every line (motor, buildings, contents) on three fit-sets: 2021-22 -> 2023,
2021-23 -> 2024 (out-of-time back-tests) and 2021-24 -> 2025 (the submission). Predictions go to
artifacts/week1/experts/, submissions to outputs/.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXP = "scripts/experiments"
CORE6 = ["total", "fs_gamma", "fs_lognorm", "peril", "fs_peril", "glm"]
STEPS: dict[str, list[list[str]]] = {
    "features": [["scripts/build_features.py", "--config", "configs/week1.yaml"]],
    "experts": [[f"{EXP}/run_experts.py"]],
    "twopart": [[f"{EXP}/run_twopart.py"]],
    "peril": [[f"{EXP}/run_twopart.py", "--peril"]],
    "moe": [[f"{EXP}/run_moe.py"]],
    "ensembles": [
        # v4: equal-weight geometric mean of the six experts -> outputs/submission_week1_v4_geo6.csv
        [f"{EXP}/build_robust_submission.py", "--method", "geo", "--members", *CORE6, "--tag", "v4_geo6"],
        # pick3: the six + the claim-size mixture of experts -> outputs/submission_week1_geo7_moe.csv
        [f"{EXP}/build_robust_submission.py", "--method", "geo", "--members", *CORE6, "fs_moe", "--tag", "geo7_moe"],
        # checks (integrity, 2025 level per line, shape) against the submitted files
        [f"{EXP}/verify_final.py", "outputs/submission_week1_v4_geo6.csv", "submissions/submission_week1_v4_geo6.csv"],
        [f"{EXP}/verify_final.py", "outputs/submission_week1_geo7_moe.csv", "submissions/pick3_ens_geo7_moe.csv"],
    ],
}
DATA_FILES = ["addresses", "claims", "customers", "policies_buildings", "policies_contents", "policies_motor",
              "sample_submission", "test_buildings", "test_contents", "test_motor", "weather"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--from", dest="start", choices=list(STEPS), help="start at this step")
    g.add_argument("--only", choices=list(STEPS), help="run only this step")
    a = ap.parse_args()
    missing = [f for f in DATA_FILES if not (ROOT / "data" / "week1" / f"{f}.parquet").exists()]
    if missing:
        print("Put the 11 competition files in data/week1/ first (see data/week1/README.md). Missing: "
              + ", ".join(f"{f}.parquet" for f in missing))
        return 1
    for d in ("artifacts/week1/experts", "reports/week1", "outputs"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    names = list(STEPS)
    todo = [a.only] if a.only else names[names.index(a.start):] if a.start else names
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONHASHSEED="0")
    for step in todo:
        for cmd in STEPS[step]:
            t0 = time.time()
            print(f"\n######## {step}: python {' '.join(cmd)}", flush=True)
            r = subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env)
            if r.returncode:
                print(f"######## {step} failed (exit {r.returncode}); fix it and resume with --from {step}")
                return r.returncode
            print(f"######## {step} done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
