"""Build the modelling feature frames (train + 2025 test) for each available line.

Writes ``<artifacts_dir>/features_<line>.parquet`` with metadata, every candidate
feature (raw text categoricals; encoding is fitted later on training rows only)
and an ``is_test`` flag. Lines are processed one at a time to bound memory.

Usage:
    python scripts/build_features.py --config configs/week1.yaml [--lines buildings contents]
"""
from __future__ import annotations

import argparse
import logging
try:
    import resource          # Unix only; used just to log peak memory
except ImportError:         # Windows
    resource = None
import sys
import time
from pathlib import Path

import pandas as pd

from iaa_pricing.cleaning import (
    clean_policy_table,
    clean_reference_table,
    fix_exposure,
    fixes_to_frame,
    merge_data_quality_log,
)
from iaa_pricing.config import LINES, load_config
from iaa_pricing.features import (
    CUSTOMER_COLUMNS,
    build_line_frame,
    claim_events,
    feature_spec,
    pad_customer_ids,
    policy_keys,
)
from iaa_pricing.io import read_table, table_path
from iaa_pricing.targets import claims_from_config

log = logging.getLogger("build_features")

_KEY_COLS = ["row_id", "policy_id", "customer_id", "address_id", "period_start", "period_end", "exposure"]


def _rss_mb() -> float:
    if resource is None:
        return float("nan")
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def load_all_policy_keys(data_dir: Path, lines: list[str], id_width: int) -> pd.DataFrame:
    """Training-row keys of every available line, exposure recomputed from dates."""
    parts = []
    for line in lines:
        k = read_table(data_dir, f"policies_{line}", columns=_KEY_COLS)
        k, _ = fix_exposure(k, f"policies_{line}")
        k = pad_customer_ids(k, id_width, table=f"policies_{line} keys")
        parts.append(policy_keys(k, line))
    return pd.concat(parts, ignore_index=True)


def load_weather(data_dir: Path) -> pd.DataFrame | None:
    """Weather with float32 measures (2.9M rows: halves memory vs float64)."""
    if not table_path(data_dir, "weather").exists():
        log.warning("weather.parquet not found: weather features will be absent")
        return None
    w = read_table(data_dir, "weather")
    for c in w.columns:
        if pd.api.types.is_float_dtype(w[c]):
            w[c] = w[c].astype("float32")
    w["year"] = w["year"].astype("int16")
    return w


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/week1.yaml")
    ap.add_argument("--lines", nargs="*", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    cfg.ensure_dirs()
    available = [l for l in LINES if table_path(cfg.data_dir, f"policies_{l}").exists()
                 and table_path(cfg.data_dir, f"test_{l}").exists()]
    lines = [l for l in (args.lines or available) if l in available]
    log.info("Lines available: %s; building: %s", available, lines)

    claims = claims_from_config(cfg)
    fixes = []  # every cleaning Fix, written to reports/<run>/data_quality_log.csv
    customers = read_table(cfg.data_dir, "customers", columns=list(CUSTOMER_COLUMNS))
    customers, f = clean_reference_table(customers, "customers")
    fixes += f
    addresses, f = clean_reference_table(read_table(cfg.data_dir, "addresses"), "addresses")
    fixes += f
    weather = load_weather(cfg.data_dir)  # numeric only: no category cleaning needed
    id_width = int(customers["customer_id"].str.len().max())
    keys = load_all_policy_keys(cfg.data_dir, available, id_width)
    events = claim_events(claims, keys)  # interval join done once, reused by every chunk
    del claims
    log.info("Reference data loaded; peak RSS %.0f MB", _rss_mb())

    for line in lines:
        t0 = time.time()
        frames = []
        for kind, is_test in (("policies", False), ("test", True)):
            # Cleaning uses whole-table statistics (e.g. the sum-insured median), so
            # clean first, then build year by year to bound peak memory.
            pol, f = clean_policy_table(read_table(cfg.data_dir, f"{kind}_{line}"), f"{kind}_{line}")
            fixes += f
            # Only this table's addresses are needed; filtering keeps every year of
            # those addresses, so long-run weather means are unchanged.
            wx = None if weather is None else weather[weather["address_id"].isin(set(pol["address_id"]))]
            years = pol["period_start"].dt.year
            for y in sorted(years.unique()):
                frames.append(build_line_frame(line, pol.loc[years == y], customers, addresses,
                                               wx, events, keys, cfg, is_test=is_test))
            del pol, years, wx
        frame = pd.concat(frames, ignore_index=True)
        del frames
        out = cfg.artifacts_dir / f"features_{line}.parquet"
        frame.to_parquet(out, index=False)
        spec = feature_spec(line, cfg, available=frame.columns)
        log.info(
            "%s: %s train + %s test rows, %d columns, %d model features (%d num / %d cat), "
            "%d review; %.0fs; peak RSS %.0f MB -> %s",
            line, f"{(~frame['is_test']).sum():,}", f"{frame['is_test'].sum():,}", frame.shape[1],
            len(spec.features), len(spec.numeric), len(spec.categorical), len(spec.review_features),
            time.time() - t0, _rss_mb(), out,
        )
        del frame
    dq = merge_data_quality_log(fixes_to_frame(fixes), cfg.reports_dir / "data_quality_log.csv")
    log.info("Data-quality log: %d fix records -> %s", len(dq), cfg.reports_dir / "data_quality_log.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
