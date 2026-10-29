"""Versioned submission files: ``<round>/submissions/v<N>_<model>.csv``, with a register.

Every model step saves its submission here instead of overwriting one file, so each file ever
made stays available. ``versions.csv`` (and its readable copy ``README.md``) records, for each
version, the model, the script that made it, when, its checks and scores, and a checksum. A file
identical to an existing version is not saved again: re-running a step without changes adds
nothing.

When the environment variable ``SUBMISSIONS_EXPORT_DIR`` is set (``final/pipeline.py`` sets it),
every save also writes the file it produced to that folder, new or identical, and appends its
version to ``saved.csv`` there, so a caller can check exactly what a step made.
"""
from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

COLUMNS = ["version", "file", "model", "script", "created", "rows", "sha256", "evidence", "public_score",
           "description"]


def save(sub: pd.DataFrame, round_dir: Path, model: str, script: str, description: str = "",
         evidence: str = "") -> Path:
    """Save ``sub`` as the next version of ``round_dir``'s submissions; return its path.

    ``model`` names the model in the file name (letters, digits, underscores), ``evidence``
    summarises the scores behind it, ``description`` says what changed.
    """
    data = sub.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return _store(data, len(sub), round_dir, model, script, description, evidence)


def _store(data: bytes, rows: int, round_dir: Path, model: str, script: str, description: str,
           evidence: str) -> Path:
    folder = Path(round_dir) / "submissions"
    folder.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(data).hexdigest()[:16]
    reg = _register(folder)
    same = reg.loc[reg["sha256"] == digest]
    if len(same):                                            # nothing new: keep the existing version
        print(f"   identical to {same['version'].iloc[0]} ({same['file'].iloc[0]}): no new version")
        _export(data, same["version"].iloc[0], same["file"].iloc[0])
        return folder / same["file"].iloc[0]
    done = [int(m.group(1)) for f in [*reg["file"], *(p.name for p in folder.glob("v*_*.csv"))]
            if (m := re.match(r"v(\d+)_", str(f)))]
    version = f"v{max(done, default=0) + 1}"
    name = re.sub(r"[^a-z0-9_]+", "_", model.lower()).strip("_")
    path = folder / f"{version}_{name}.csv"
    path.write_bytes(data)
    _export(data, version, path.name)
    row = {"version": version, "file": path.name, "model": model, "script": script,
           "created": datetime.now().strftime("%Y-%m-%d %H:%M"), "rows": str(rows), "sha256": digest,
           "evidence": evidence, "public_score": "", "description": description}
    _write_register(folder, pd.concat([reg, pd.DataFrame([row])], ignore_index=True)[COLUMNS])
    print(f"   saved as {version}: {path}")
    return path


def _export(data: bytes, version: str, file: str) -> None:
    """Copy the produced file to ``SUBMISSIONS_EXPORT_DIR`` and log its version, if that is set."""
    folder = os.environ.get("SUBMISSIONS_EXPORT_DIR")
    if not folder:
        return
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    (out / file).write_bytes(data)
    with open(out / "saved.csv", "a", encoding="utf-8") as log:
        log.write(f"{version},{file}\n")


def record_score(round_dir: Path, version: str, score: float) -> None:
    """Write a version's public leaderboard score into the register."""
    folder = Path(round_dir) / "submissions"
    reg = _register(folder)
    if version not in set(reg["version"]):
        raise KeyError(f"{version} is not in {folder / 'versions.csv'}")
    reg.loc[reg["version"] == version, "public_score"] = f"{score:.5f}"
    _write_register(folder, reg)


def _register(folder: Path) -> pd.DataFrame:
    path = folder / "versions.csv"
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, dtype=str).fillna("").reindex(columns=COLUMNS, fill_value="")


def _write_register(folder: Path, reg: pd.DataFrame) -> None:
    reg.to_csv(folder / "versions.csv", index=False, lineterminator="\n")   # same bytes on every platform
    lines = ["# Submissions", "",
             "Every submission file made in this round, newest last. `versions.csv` holds the same table.", "",
             "| Version | File | Model | Script | Created | Evidence | Public score | Description |",
             "|---|---|---|---|---|---|---|---|"]
    for r in reg.itertuples(index=False):
        lines.append(f"| {r.version} | `{r.file}` | {r.model} | `{r.script}` | {r.created} | {r.evidence} "
                     f"| {r.public_score} | {r.description} |")
    (folder / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
