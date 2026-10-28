"""Run configuration and domain constants.

All judgement-based assumptions are read from a YAML file so that each change is
reviewable in version control. Domain constants (table names, perils, value
domains) come from the competition data dictionary.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

LINES: tuple[str, ...] = ("motor", "buildings", "contents")

LINE_PREFIX: dict[str, str] = {"motor": "M", "buildings": "B", "contents": "C"}

PERILS: dict[str, tuple[str, ...]] = {
    "motor": ("own_damage", "third_party_prop", "bodily_injury", "theft", "glass_hail"),
    "buildings": ("storm", "flood", "escape_of_water", "subsidence", "fire", "accidental_damage"),
    "contents": ("theft", "escape_of_water", "accidental_damage", "flood"),
}

CLAIM_STATUSES: tuple[str, ...] = ("closed", "open", "repudiated")

# Identifiers must always be read as text (data dictionary instruction).
ID_COLUMNS: tuple[str, ...] = (
    "customer_id", "address_id", "policy_id", "row_id", "claim_id", "repairer_id",
    "third_party_id", "branch_id",
)

DATE_COLUMNS: tuple[str, ...] = (
    "period_start", "period_end", "loss_date", "report_date", "date_of_birth",
)

TABLES: tuple[str, ...] = (
    "customers", "addresses", "weather", "claims",
    "policies_motor", "policies_buildings", "policies_contents",
    "test_motor", "test_buildings", "test_contents", "sample_submission",
)

# Categorical value domains from the data dictionary (after cleaning to lower case).
DOMAINS: dict[str, tuple[Any, ...]] = {
    "marital_status": ("single", "married", "divorced", "widowed"),
    "credit_band": ("A", "B", "C", "D", "E"),
    "document_preference": ("email", "post", "app"),
    "region": tuple(f"R{i:02d}" for i in range(1, 13)),
    "urban_density": ("rural", "suburban", "urban", "dense_urban"),
    "soil_type": ("clay", "sand", "loam", "chalk", "rock"),
    "property_type": ("detached", "semi_detached", "terraced", "flat"),
    "roof_type": ("pitched", "flat"),
    "energy_rating": ("A", "B", "C", "D", "E", "F", "G"),
    "channel": ("direct", "comparison_site", "broker"),
    "payment_method": ("annual", "monthly"),
    "voluntary_excess": (0, 250, 500, 1000),
    "body_type": ("hatchback", "saloon", "estate", "suv", "coupe"),
    "transmission": ("manual", "automatic"),
    "fuel_type": ("petrol", "diesel", "hybrid", "electric"),
    "cover_type": ("comprehensive", "third_party_fire_theft", "third_party_only"),
    "overnight_parking": ("garage", "driveway", "street"),
    "vehicle_use": ("social", "commuting", "business"),
    "construction_type": ("brick", "stone", "timber", "non_standard"),
    "occupancy": ("owner_occupied", "let"),
    "tenure": ("owner", "renter"),
    "security_level": ("standard", "locks", "alarm_and_locks"),
    "renewal_decision": (
        "renewed", "lapsed_by_customer", "cancelled", "declined_by_insurer", "mid_term_change",
    ),
}


@dataclass(frozen=True)
class Config:
    """Typed view of a run configuration file."""

    run_name: str
    seed: int
    data_dir: Path
    artifacts_dir: Path
    reports_dir: Path
    output_dir: Path
    valuation_date: pd.Timestamp
    prediction_year: int
    train_years: tuple[int, ...]
    tweedie_power: float
    raw: dict[str, Any] = field(repr=False)
    source_path: Path | None = None

    # ---- convenience accessors -------------------------------------------------
    def section(self, name: str) -> dict[str, Any]:
        """Return a top-level YAML section (empty dict if absent)."""
        return dict(self.raw.get(name, {}) or {})

    @property
    def forbidden_features(self) -> dict[str, str]:
        return dict(self.section("features").get("forbidden", {}) or {})

    @property
    def review_required_features(self) -> dict[str, str]:
        return dict(self.section("features").get("review_required", {}) or {})

    def fingerprint(self) -> str:
        """Stable hash of the full configuration (recorded in the run manifest)."""
        payload = json.dumps(self.raw, sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def ensure_dirs(self) -> None:
        for p in (self.artifacts_dir, self.reports_dir, self.output_dir):
            p.mkdir(parents=True, exist_ok=True)


def _resolve(path_str: str, root: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (root / p)


def load_config(path: str | Path, root: Path = REPO_ROOT) -> Config:
    """Load and validate a YAML run configuration."""
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    required = ("run_name", "seed", "paths", "dates", "metric")
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"Config {path} is missing required sections: {missing}")

    paths, dates = raw["paths"], raw["dates"]
    cfg = Config(
        run_name=str(raw["run_name"]),
        seed=int(raw["seed"]),
        data_dir=_resolve(paths["data_dir"], root),
        artifacts_dir=_resolve(paths["artifacts_dir"], root),
        reports_dir=_resolve(paths["reports_dir"], root),
        output_dir=_resolve(paths["output_dir"], root),
        valuation_date=pd.Timestamp(dates["valuation_date"]),
        prediction_year=int(dates["prediction_year"]),
        train_years=tuple(int(y) for y in dates["train_years"]),
        tweedie_power=float(raw["metric"]["tweedie_power"]),
        raw=raw,
        source_path=path,
    )
    if not 1.0 < cfg.tweedie_power < 2.0:
        raise ValueError("Compound Poisson-gamma Tweedie power must lie in (1, 2)")
    if cfg.prediction_year <= max(cfg.train_years):
        raise ValueError("prediction_year must be after the last training year")
    return cfg
