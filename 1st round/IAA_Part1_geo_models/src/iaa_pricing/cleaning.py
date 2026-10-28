"""Deterministic, logged data-quality fixes.

Every fix returns the cleaned frame plus a list of :class:`Fix` records that the
pipeline writes to ``reports/<run>/data_quality_log.csv`` for human review.
Fixes were identified during profiling (see docs/data_quality.md):

* exposure recorded on a 360-day basis or in days instead of years (train only);
* sum insured inflated by ~1,000x on ~0.2% of home rows (train and test);
* inconsistent category coding: case variants, and timber coded ``T`` up to
  2022 and ``W`` from 2023 (test uses ``W`` only);
* nullable booleans stored as objects.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Canonical recoding of category synonyms (applied after lower-casing).
CATEGORY_SYNONYMS: dict[str, dict[str, str]] = {
    "construction_type": {
        "w": "timber", "t": "timber", "wood": "timber",
        "non-standard": "non_standard", "nonstandard": "non_standard",
    },
}

# Text columns that are categorical attributes (identifiers are excluded on purpose).
_NON_CATEGORICAL_TEXT = {
    "row_id", "policy_id", "customer_id", "address_id", "claim_id", "repairer_id",
    "third_party_id", "surname",
}
# Columns whose codes are upper case by convention.
_UPPER_CASE_CODES = {"credit_band", "energy_rating", "region", "make_model", "previous_insurer",
                     "campaign_code"}


@dataclass(frozen=True)
class Fix:
    table: str
    column: str
    rule: str
    n_rows: int
    detail: str = ""

    def as_dict(self) -> dict[str, object]:
        return {"table": self.table, "column": self.column, "rule": self.rule,
                "n_rows": self.n_rows, "detail": self.detail}


def normalise_categories(df: pd.DataFrame, table: str) -> tuple[pd.DataFrame, list[Fix]]:
    """Trim/case-fold categorical text and map known synonyms to canonical codes."""
    df = df.copy()
    fixes: list[Fix] = []
    text_cols = [
        c for c in df.columns
        if c not in _NON_CATEGORICAL_TEXT
        and (pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object)
        and not _is_boolean_like(df[c])
        and not pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    for col in text_cols:
        before = df[col].astype("string")
        after = before.str.strip()
        after = after.str.upper() if col in _UPPER_CASE_CODES else after.str.lower()
        if col in CATEGORY_SYNONYMS:
            after = after.replace(CATEGORY_SYNONYMS[col])
        changed = int((before.fillna("<NA>") != after.fillna("<NA>")).sum())
        if changed:
            fixes.append(Fix(table, col, "normalise_category_codes", changed,
                             f"levels now: {sorted(after.dropna().unique().tolist())[:12]}"))
        df[col] = after
    return df, fixes


def _is_boolean_like(s: pd.Series) -> bool:
    vals = pd.unique(s.dropna())
    return len(vals) > 0 and all(isinstance(v, (bool, np.bool_)) for v in vals)


def coerce_booleans(df: pd.DataFrame, table: str) -> tuple[pd.DataFrame, list[Fix]]:
    """Object columns holding True/False/None become float 1/0/NaN (model-ready)."""
    df = df.copy()
    fixes: list[Fix] = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            df[col] = s.astype("float32")
        elif s.dtype == object and _is_boolean_like(s):
            n_missing = int(s.isna().sum())
            df[col] = s.map({True: 1.0, False: 0.0}).astype("float32")
            fixes.append(Fix(table, col, "object_bool_to_float", n_missing,
                             "missing kept as NaN (informative-missing handled by models)"))
    return df, fixes


def fix_exposure(df: pd.DataFrame, table: str, days_in_year: float = 365.0) -> tuple[pd.DataFrame, list[Fix]]:
    """Recompute exposure from the period dates (inclusive day count / 365).

    The 2025 test files use exactly this definition, so the training data are
    aligned to it. Mismatched rows are classified for the data-quality log.
    """
    df = df.copy()
    days = (df["period_end"] - df["period_start"]).dt.days + 1
    if (days <= 0).any():
        raise ValueError(f"{table}: {int((days <= 0).sum())} rows end before they start")
    implied = days / days_in_year
    bad = ~np.isclose(df["exposure"], implied, atol=1e-3)
    fixes: list[Fix] = []
    if bad.any():
        e = df.loc[bad, "exposure"]
        in_days = np.isclose(e, days[bad], atol=1e-6)
        basis_360 = np.isclose(e, days[bad] / 360.0, atol=1e-3)
        detail = (f"days-unit={int(in_days.sum())}, 360-day basis={int(basis_360.sum())}, "
                  f"other={int((~in_days & ~basis_360).sum())}")
        fixes.append(Fix(table, "exposure", "recompute_from_dates", int(bad.sum()), detail))
        log.warning("%s: exposure recomputed on %d rows (%s)", table, int(bad.sum()), detail)
    df["exposure_reported"] = df["exposure"]
    df["exposure"] = implied.astype(float)
    return df, fixes


def fix_sum_insured(
    df: pd.DataFrame, table: str, ratio_to_median: float = 100.0, scale: float = 1000.0
) -> tuple[pd.DataFrame, list[Fix]]:
    """Undo ~1,000x unit errors in ``sum_insured``.

    A value more than ``ratio_to_median`` times the median is divided by
    ``scale``. Profiling showed a clean gap: genuine values never exceed ~3x the
    median and corrupted values are >= ~200x, and the written premium on the
    corrupted rows is typical, which confirms a recording error rather than risk.
    """
    if "sum_insured" not in df.columns:
        return df, []
    df = df.copy()
    med = float(df["sum_insured"].median())
    flag = df["sum_insured"] > ratio_to_median * med
    df["si_unit_error"] = flag.astype("float32")
    fixes: list[Fix] = []
    if flag.any():
        df.loc[flag, "sum_insured"] = df.loc[flag, "sum_insured"] / scale
        still = df.loc[flag, "sum_insured"] > ratio_to_median * med
        if still.any():
            raise ValueError(f"{table}: {int(still.sum())} sum_insured values remain implausible")
        fixes.append(Fix(table, "sum_insured", f"divide_by_{scale:g}_if_gt_{ratio_to_median:g}x_median",
                         int(flag.sum()), f"median={med:,.0f}"))
    return df, fixes


def fix_id_padding(
    df: pd.DataFrame, table: str, column: str = "customer_id", width: int = 8
) -> tuple[pd.DataFrame, list[Fix]]:
    """Restore leading zeros on all-digit identifiers shorter than ``width``.

    About 10% of motor ``customer_id`` values (train and test) were stored as
    numbers and lost their leading zeros (e.g. ``"01234567"`` became ``"1234567"``).
    Every padded value matches the customers table (verified by the data contracts).
    """
    if column not in df.columns:
        return df, []
    s = df[column].astype("string")
    short = s.str.fullmatch(r"\d+").fillna(False) & (s.str.len() < width)
    if not short.any():
        return df, []
    df = df.copy()
    df[column] = s.where(~short, s.str.zfill(width))
    return df, [Fix(table, column, f"zfill_{width}_lost_leading_zeros", int(short.sum()),
                    "all-digit ids shorter than the canonical width")]


def clean_policy_table(df: pd.DataFrame, table: str) -> tuple[pd.DataFrame, list[Fix]]:
    """Apply all policy-table fixes in a fixed order."""
    fixes: list[Fix] = []
    for step in (fix_id_padding, fix_exposure, fix_sum_insured, normalise_categories, coerce_booleans):
        df, f = step(df, table)
        fixes.extend(f)
    return df, fixes


def clean_reference_table(df: pd.DataFrame, table: str) -> tuple[pd.DataFrame, list[Fix]]:
    """Fixes for customers / addresses / weather."""
    fixes: list[Fix] = []
    for step in (normalise_categories, coerce_booleans):
        df, f = step(df, table)
        fixes.extend(f)
    return df, fixes


def fixes_to_frame(fixes: list[Fix]) -> pd.DataFrame:
    return pd.DataFrame([f.as_dict() for f in fixes],
                        columns=["table", "column", "rule", "n_rows", "detail"])


# ---------------------------------------------------------------------------
# Data-quality log (audit trail of every deterministic fix)
# ---------------------------------------------------------------------------
POLICY_TABLES: tuple[str, ...] = (
    "policies_motor", "policies_buildings", "policies_contents",
    "test_motor", "test_buildings", "test_contents",
)
REFERENCE_TABLES: tuple[str, ...] = ("customers", "addresses")


def data_quality_log(data_dir, tables: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Re-apply the cleaning to each available table and return every ``Fix`` record.

    Tables are processed one at a time (bounded memory). Missing tables are skipped and
    reported as an ``info`` row so the log always states what it covers.
    """
    from .io import read_table, table_path  # local import: io is a leaf module

    fixes: list[Fix] = []
    for name in tables or (*REFERENCE_TABLES, *POLICY_TABLES):
        if not table_path(data_dir, name).exists():
            fixes.append(Fix(name, "*", "table_missing", 0, "not present in data_dir"))
            continue
        df = read_table(data_dir, name)
        clean = clean_policy_table if name in POLICY_TABLES else clean_reference_table
        _, f = clean(df, name)
        fixes.extend(f)
        if not f:
            fixes.append(Fix(name, "*", "no_fix_needed", 0, f"{len(df):,} rows checked"))
        del df
    return fixes_to_frame(fixes)


def write_data_quality_log(log_frame: pd.DataFrame, path) -> None:
    """Write the data-quality log (CSV) creating the folder if needed."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    log_frame.to_csv(path, index=False)


def merge_data_quality_log(new: pd.DataFrame, path) -> pd.DataFrame:
    """Replace the rows of the tables in ``new`` within an existing log (or create it).

    Used by partial runs (e.g. rebuilding the features of one line) so they never
    truncate a complete log written by the full pipeline.
    """
    from pathlib import Path

    path = Path(path)
    if path.exists():
        old = pd.read_csv(path, dtype={"detail": str})
        old = old.loc[~old["table"].isin(set(new["table"]))]
        new = pd.concat([old, new], ignore_index=True)
    write_data_quality_log(new, path)
    return new
