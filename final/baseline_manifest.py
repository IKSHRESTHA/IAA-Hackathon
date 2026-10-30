"""Freeze the current model baseline and verify both submission files.

Usage:
    python final/baseline_manifest.py
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

FINAL = Path(__file__).resolve().parent
REPO = FINAL.parent
OUT = FINAL / "3_diagnostics" / "baseline_manifest.json"

SUBMISSIONS = {
    "week1": FINAL / "1_predictions" / "week1_part1_predictions.csv",
    "week2": FINAL / "1_predictions" / "week2_part2_predictions.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def package_versions() -> dict[str, str | None]:
    packages = ("numpy", "pandas", "lightgbm", "scikit-learn", "scipy", "glum")
    values: dict[str, str | None] = {}
    for package in packages:
        try:
            values[package] = version(package)
        except PackageNotFoundError:
            values[package] = None
    return values


def validate_submission(path: Path) -> dict[str, object]:
    frame = pd.read_csv(path, dtype={"row_id": "string"})
    prediction = pd.to_numeric(frame["prediction"], errors="coerce")
    checks = {
        "columns": frame.columns.tolist() == ["row_id", "prediction"],
        "unique_row_ids": bool(frame["row_id"].is_unique),
        "finite_predictions": bool(prediction.notna().all()),
        "positive_predictions": bool((prediction > 0).all()),
    }
    if not all(checks.values()):
        raise ValueError(f"{path}: submission checks failed: {checks}")
    return {
        "path": path.relative_to(REPO).as_posix(),
        "rows": int(len(frame)),
        "sha256": sha256(path),
        "mean_prediction": float(prediction.mean()),
        "checks": checks,
    }


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "purpose": "Final submission baseline",
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short").splitlines(),
        "packages": package_versions(),
        "submissions": {name: validate_submission(path) for name, path in SUBMISSIONS.items()},
    }
    OUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Baseline manifest written to {OUT}")
    for name, details in manifest["submissions"].items():
        print(f"{name}: {details['rows']:,} rows, sha256={details['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
