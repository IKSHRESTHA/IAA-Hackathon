"""The pipeline that produces the two final prediction files, from the competition data to final/.

    python final/pipeline.py                 quick run (about a minute): the last stages, from the
                                             intermediate results committed in the repository
    python final/pipeline.py --full          full run (several hours): every stage, from the raw
                                             competition files in the repository
    python final/pipeline.py --week 2        one week only
    python final/pipeline.py --no-render     skip the PDF

Week 1 (developed market). Code: 1st round/IAA_Part1_geo_models/.
    full   run_pipeline.py: features, the six experts on three fit sets, ensembles (about 2.5 hours)
    quick  the ensemble stage only, on cache/experts7_<line>.parquet (the experts' predictions as
           they were on 22 October): geometric mean, recalibration per line

Week 2 (emerging market). Code: 1st round/ and 2nd round/.
    full   1st round steps 01-03 (the developed market's features, which the prior needs), then
           2nd round steps 01-03 (read, clean, features) and 04-08 (developed prior, LightGBM,
           expert ensemble on priors)
    both   09_rebase_to_best.py: v5, the best public file's segment totals with the ensemble's
           ranking inside
           11_joint_levels.py make --target public --scores-before v12: v13, v5 with every segment
           at its best level for the public rows, read from the public scores known when v13 was
           made

Then every rebuilt file is compared with the file that was submitted, the submitted files are
copied into final/1_predictions/ (checked against the sample submission, listed in manifest.csv),
and the write-up is rendered. Rebuilt files go to final/reproduced/.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

FINAL = Path(__file__).resolve().parent
REPO = FINAL.parent
R1, R2 = REPO / "1st round", REPO / "2nd round"
W1 = R1 / "IAA_Part1_geo_models"            # the code that made the Week 1 file
REBUILT = FINAL / "reproduced"
ENV = dict(os.environ, PYTHONHASHSEED="0", PYTHONUTF8="1")   # same bytes on every run

WEEK1_DATA = ["addresses", "claims", "customers", "policies_buildings", "policies_contents", "policies_motor",
              "sample_submission", "test_buildings", "test_contents", "test_motor", "weather"]

# The file submitted for each week, as kept in the repository.
SUBMITTED = {
    "week1": {
        "deliverable": "Week 1 prediction file (Part 1, developed market)",
        "file": "week1_part1_predictions.csv",
        "source": "1st round/IAA_Part1_geo_models/submissions/submission_week1_v4_geo6.csv",
        "sample": "1st round/data/raw/sample_submission.parquet",
        "model": "geometric mean of six experts per line, recalibrated per line",
        "public_score": 0.12851,
        "private_score": 0.13830,
        "note": "best private score of Part 1; tied with pick3_ens_geo7_moe.csv (7 experts), the simpler file is kept",
    },
    "week2": {
        "deliverable": "Week 2 prediction file (Part 2, emerging market)",
        "file": "week2_part2_predictions.csv",
        "source": "2nd round/submissions/v13_joint_levels_public.csv",
        "sample": "2nd round/data/raw/sample_submission.parquet",
        "model": "v13: transfer ensemble ranking, every segment at its best level for the public rows",
        "public_score": 0.17172,
        "private_score": None,
        "note": "best public score of Part 2; v12 (public 0.17095, best expected on unseen rows) is also selected on Kaggle",
    },
}


def run(script: str, *args: str, cwd: Path) -> None:
    """Run one pipeline script as a user would, from its own folder."""
    t0 = time.time()
    print(f"\n>>> {cwd.relative_to(REPO)}: python {script} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, script, *args], cwd=cwd, env=ENV, check=True)
    print(f">>> done in {(time.time() - t0) / 60:.1f} min", flush=True)


# ------------------------------------------------------------------------------------------ Week 1

def week1(full: bool) -> Path:
    out = REBUILT / SUBMITTED["week1"]["file"]
    if full:
        data = W1 / "data" / "week1"                 # that code reads the competition files from here
        for name in WEEK1_DATA:
            if not (data / f"{name}.parquet").exists():
                shutil.copyfile(R1 / "data" / "raw" / f"{name}.parquet", data / f"{name}.parquet")
        run("run_pipeline.py", cwd=W1)
        shutil.copyfile(W1 / "outputs" / "submission_week1_v4_geo6.csv", out)
    else:
        # The ensemble stage of rebuild_from_cache.py, written to final/reproduced/ so the committed
        # outputs/ stay untouched.
        print("\n>>> 1st round/IAA_Part1_geo_models: ensemble stage on the cached expert predictions", flush=True)
        sys.dont_write_bytecode = True               # leave no __pycache__ behind in that folder
        sys.path[:0] = [str(W1), str(W1 / "src")]
        rb = importlib.import_module("rebuild_from_cache")
        cache = {line: pd.read_parquet(W1 / "cache" / f"experts7_{line}.parquet") for line in rb.LINES}
        rb.build(rb.MODELS["submission_week1_v4_geo6.csv"], cache).to_csv(out, index=False)
    return out


# ------------------------------------------------------------------------------------------ Week 2

def week2(full: bool) -> Path:
    if full:
        run("run_all.py", "--rounds", "1st round", cwd=REPO)               # data steps 01-03
        run("run_all.py", "--rounds", "2nd round", "--models", cwd=REPO)   # 01-03, then 04-09
    else:
        run("09_rebase_to_best.py", cwd=R2)
    run("11_joint_levels.py", "make", "--target", "public", "--scores-before", "v12", cwd=R2)
    # Step 11 registers its file as a new version, or reports that it is identical to one already
    # there; either way the newest "joint_levels_public" version is the one it produced.
    reg = pd.read_csv(R2 / "submissions" / "versions.csv", dtype=str)
    rows = reg.loc[reg["model"] == "joint_levels_public"]
    made = rows.iloc[int(rows["version"].str.lstrip("v").astype(int).to_numpy().argmax())]
    out = REBUILT / SUBMITTED["week2"]["file"]
    shutil.copyfile(R2 / "submissions" / made["file"], out)
    return out


# ------------------------------------------------------------------------------- check and package

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(week: str, rebuilt: Path) -> dict:
    submitted = REPO / SUBMITTED[week]["source"]
    a, b = pd.read_csv(rebuilt, dtype={"row_id": str}), pd.read_csv(submitted, dtype={"row_id": str})
    same_rows = a["row_id"].equals(b["row_id"])
    diff = float(np.abs(a["prediction"].to_numpy() / b["prediction"].to_numpy() - 1).max()) if same_rows else np.inf
    same_bytes = sha256(rebuilt) == sha256(submitted)
    ok = same_rows and diff < 1e-9
    print(f"{week}: rebuilt file {'matches' if ok else 'DIFFERS FROM'} the submitted file "
          f"(same rows {same_rows}, largest relative difference {diff:.1e}, identical bytes {same_bytes})")
    return {"week": week, "rebuilt": str(rebuilt.relative_to(REPO)), "submitted": SUBMITTED[week]["source"],
            "same_rows": same_rows, "max_relative_difference": diff, "identical_bytes": same_bytes, "match": ok}


def check_against_sample(sub: pd.DataFrame, sample: pd.DataFrame) -> dict:
    """Raise if the file would be rejected or mis-scored; return a few facts about it."""
    if list(sub.columns) != ["row_id", "prediction"]:
        raise ValueError(f"columns are {list(sub.columns)}, expected ['row_id', 'prediction']")
    if len(sub) != len(sample) or not (sub["row_id"].to_numpy() == sample["row_id"].to_numpy()).all():
        raise ValueError("row_ids differ from the sample submission (or are in another order)")
    p = sub["prediction"].to_numpy(dtype=float)
    if not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError("a prediction is missing, infinite, zero or negative")
    line = sub["row_id"].str[0].map({"M": "motor", "B": "buildings", "C": "contents"})
    means = sub.groupby(line)["prediction"].mean()
    return {"rows": len(sub), **{f"mean_{k}": round(float(v), 2) for k, v in means.items()}}


def package() -> None:
    folder = FINAL / "1_predictions"
    folder.mkdir(exist_ok=True)
    rows = []
    for d in SUBMITTED.values():
        dst = folder / d["file"]
        shutil.copyfile(REPO / d["source"], dst)
        facts = check_against_sample(pd.read_csv(dst, dtype={"row_id": str}), pd.read_parquet(REPO / d["sample"]))
        rows.append({"deliverable": d["deliverable"], "file": d["file"], "source": d["source"], "model": d["model"],
                     **facts, "public_score": d["public_score"], "private_score": d["private_score"],
                     "sha256": sha256(dst), "note": d["note"]})
        print(f"{d['file']}: {facts['rows']:,} rows, checks against the sample submission passed")
    pd.DataFrame(rows).to_csv(folder / "manifest.csv", index=False)


def render() -> None:
    quarto = shutil.which("quarto")
    if quarto is None:
        print("quarto not found: install it from https://quarto.org to render the write-up")
        return
    subprocess.run([quarto, "render"], cwd=FINAL, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--full", action="store_true", help="rebuild everything from the raw competition files")
    ap.add_argument("--week", choices=["1", "2", "both"], default="both")
    ap.add_argument("--no-render", action="store_true", help="skip rendering the write-up")
    a = ap.parse_args()
    REBUILT.mkdir(exist_ok=True)
    rebuilt = {}
    if a.week in ("1", "both"):
        rebuilt["week1"] = week1(a.full)
    if a.week in ("2", "both"):
        rebuilt["week2"] = week2(a.full)
    print("\n=== check: rebuilt against submitted")
    checks = [compare(w, p) for w, p in rebuilt.items()]
    pd.DataFrame(checks).to_csv(REBUILT / "check.csv", index=False)
    print("\n=== package: final/1_predictions")
    package()
    if not a.no_render:
        print("\n=== render: final/2_writeup/writeup.pdf")
        render()
    return 0 if all(c["match"] for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
