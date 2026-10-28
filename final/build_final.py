"""Build the final submission folder.

1. Copy the best Week 1 and Week 2 prediction files into 1_predictions/ (byte for byte) and check
   them against each round's sample submission: same row ids in the same order, every prediction
   finite and positive.
2. Write 1_predictions/manifest.csv (source, scores, checksum of each file).
3. Write 4_code/IAA_Hackathon_code_<commit>.zip: the committed code, notebooks, reports and docs
   (no competition data, no model files), taken straight from git.
4. Render the Quarto write-up (2_writeup/writeup.qmd -> writeup.pdf) unless --no-render is given.

Run from anywhere:  python final/build_final.py  [--no-render]
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

FINAL = Path(__file__).resolve().parent
REPO = FINAL.parent

# The best file of each week. Week 1: best private score of Part 1 (0.13830; the pick3 variant
# tied, this one is the simpler of the two). Week 2: best public score of Part 2 (v13).
DELIVERABLES = [
    {
        "deliverable": "Week 1 prediction file (Part 1, developed market)",
        "file": "week1_part1_predictions.csv",
        "source": "1st round/IAA_Part1_geo_models/submissions/submission_week1_v4_geo6.csv",
        "sample": "1st round/data/raw/sample_submission.parquet",
        "model": "geometric mean of six experts per line, recalibrated per line",
        "public_score": 0.12851,
        "private_score": 0.13830,
        "note": "best private score of Part 1; tied with pick3_ens_geo7_moe.csv (7 experts), the simpler file is kept",
    },
    {
        "deliverable": "Week 2 prediction file (Part 2, emerging market)",
        "file": "week2_part2_predictions.csv",
        "source": "2nd round/submissions/v13_joint_levels_public.csv",
        "sample": "2nd round/data/raw/sample_submission.parquet",
        "model": "v13: transfer ensemble ranking, every segment at its best level for the public rows",
        "public_score": 0.17172,
        "private_score": None,
        "note": "best public score of Part 2; v12 (public 0.17095, best expected on unseen rows) is also ticked on Kaggle",
    },
]

# What goes into the code snapshot: text files that explain or reproduce the work.
CODE_SUFFIXES = {".py", ".ipynb", ".md", ".qmd", ".yml", ".yaml", ".toml", ".txt", ".json"}
SKIP_PARTS = {"__pycache__", "models", "cache", "outputs"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_predictions(sub: pd.DataFrame, sample: pd.DataFrame) -> dict:
    """Raise if the file would be rejected or mis-scored; return a few facts about it."""
    if list(sub.columns) != ["row_id", "prediction"]:
        raise ValueError(f"columns are {list(sub.columns)}, expected ['row_id', 'prediction']")
    if len(sub) != len(sample):
        raise ValueError(f"{len(sub)} rows, the sample submission has {len(sample)}")
    if not sub["row_id"].is_unique:
        raise ValueError("duplicated row_id")
    if not (sub["row_id"].to_numpy() == sample["row_id"].to_numpy()).all():
        raise ValueError("row_ids differ from the sample submission (or are in another order)")
    p = sub["prediction"].to_numpy(dtype=float)
    if not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError("a prediction is missing, infinite, zero or negative")
    line = sub["row_id"].str[0].map({"M": "motor", "B": "buildings", "C": "contents"})
    if line.isna().any():
        raise ValueError("a row_id does not start with M, B or C")
    means = sub.groupby(line)["prediction"].mean()
    return {"rows": len(sub), **{f"mean_{k}": round(float(v), 2) for k, v in means.items()}}


def build_predictions() -> pd.DataFrame:
    out_dir = FINAL / "1_predictions"
    out_dir.mkdir(exist_ok=True)
    rows = []
    for d in DELIVERABLES:
        src, dst = REPO / d["source"], out_dir / d["file"]
        shutil.copyfile(src, dst)
        if sha256(src) != sha256(dst):
            raise RuntimeError(f"copy of {src} is not identical")
        facts = check_predictions(pd.read_csv(dst), pd.read_parquet(REPO / d["sample"]))
        rows.append({"deliverable": d["deliverable"], "file": d["file"], "source": d["source"],
                     "model": d["model"], **facts, "public_score": d["public_score"],
                     "private_score": d["private_score"], "sha256": sha256(dst), "note": d["note"]})
        print(f"  {d['file']}: {facts['rows']:,} rows, checks passed")
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    return manifest


def git(*args: str) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True).stdout


def keep_in_snapshot(path: str) -> bool:
    p = PurePosixPath(path)
    if SKIP_PARTS & set(p.parts) or p.suffix == ".zip":
        return False
    if "data" in p.parts:  # competition data stays out; keep the folder notes and leaderboard evidence
        return p.name in {"README.md", "leaderboard_evidence.csv", "best_gender_factors.json"}
    if "submissions" in p.parts and p.suffix == ".csv":
        return p.name == "versions.csv"
    if "reports" in p.parts:
        return p.suffix in {".csv", ".json", ".md"}
    return p.suffix in CODE_SUFFIXES


def build_code_snapshot() -> Path:
    commit = git("rev-parse", "--short", "HEAD").decode().strip()
    dirty = [ln for ln in git("status", "--porcelain").decode().splitlines()
             if not ln[3:].startswith("final/")]
    if dirty:
        print(f"  note: {len(dirty)} uncommitted change(s) outside final/; the snapshot is the commit {commit}")
    # The repository holds about 1 GB of data, so read only the kept files, in one git call.
    paths = [p for p in git("ls-tree", "-r", "--name-only", "-z", "HEAD").decode().split("\0")
             if p and keep_in_snapshot(p)]
    blobs = subprocess.run(["git", "-C", str(REPO), "cat-file", "--batch"], check=True, capture_output=True,
                           input="".join(f"HEAD:{p}\n" for p in paths).encode()).stdout
    out = FINAL / "4_code" / f"IAA_Hackathon_code_{commit}.zip"
    out.parent.mkdir(exist_ok=True)
    for old in out.parent.glob("IAA_Hackathon_code_*.zip"):
        old.unlink()
    pos = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:  # each answer is "<sha> blob <size>\n<content>\n"
            header_end = blobs.index(b"\n", pos)
            size = int(blobs[pos:header_end].split()[2])
            z.writestr(f"IAA_Hackathon/{p}", blobs[header_end + 1:header_end + 1 + size])
            pos = header_end + 1 + size + 1
    print(f"  {out.name}: {len(paths)} files from commit {commit}, {out.stat().st_size / 1e6:.1f} MB")
    return out


def render() -> None:
    quarto = shutil.which("quarto")
    if quarto is None:
        print("  quarto not found: install it from https://quarto.org, then run `quarto render` in final/")
        return
    subprocess.run([quarto, "render"], cwd=FINAL, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-render", action="store_true", help="skip rendering the Quarto write-up")
    args = ap.parse_args()
    print("prediction files:")
    build_predictions()
    print("code snapshot:")
    build_code_snapshot()
    if not args.no_render:
        print("rendering the write-up:")
        render()


if __name__ == "__main__":
    sys.exit(main())
