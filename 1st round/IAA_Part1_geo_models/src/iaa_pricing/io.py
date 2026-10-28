"""Typed loading of the competition parquet files.

Identifiers are forced to text and dates to ``datetime64`` so that downstream
joins never silently fail on dtype mismatches (e.g. ``"00123"`` vs ``123``).
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

from .config import DATE_COLUMNS, ID_COLUMNS, TABLES

log = logging.getLogger(__name__)


def table_path(data_dir: Path, name: str) -> Path:
    if name not in TABLES:
        raise KeyError(f"Unknown table {name!r}; expected one of {TABLES}")
    return Path(data_dir) / f"{name}.parquet"


def read_table(data_dir: Path, name: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Read one table with identifiers as text and dates parsed."""
    path = table_path(data_dir, name)
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    df = pd.read_parquet(path, columns=columns)
    for col in ID_COLUMNS:
        if col in df.columns:
            df[col] = _as_text_id(df[col])
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="raise")
    log.info("Loaded %s: %s rows x %s cols", name, f"{len(df):,}", df.shape[1])
    return df


def _as_text_id(s: pd.Series) -> pd.Series:
    """Cast an identifier column to text, preserving missing values.

    Float-typed ids (e.g. ``123.0`` after a NaN forced upcasting) are converted
    back to integer text first so ``123.0`` never becomes the string ``"123.0"``.
    """
    if pd.api.types.is_float_dtype(s):
        s = s.astype("Int64")
    out = s.astype("string")
    return out.str.strip()


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def data_fingerprints(data_dir: Path) -> dict[str, str]:
    """SHA-256 of every input file (recorded in the run manifest)."""
    return {
        name: file_sha256(table_path(data_dir, name))
        for name in TABLES
        if table_path(data_dir, name).exists()
    }
