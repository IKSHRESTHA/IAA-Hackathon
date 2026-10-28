"""Line-specific feature engineering, including as-at claims history.

Design principles
-----------------
* One frame per line holding identifiers/metadata plus every *candidate*
  feature. :func:`feature_spec` decides which candidates may enter a model, so
  the governance decision (forbidden / review-required / leakage) is made in one
  auditable place rather than scattered through modelling code.
* **No look-ahead.** Claims history for a row in year ``Y`` only uses claims
  with ``loss_date`` *and* ``report_date`` before 1 January ``Y`` (the pricing
  cut-off) and never repudiated claims. The 2025 test rows therefore see exactly
  the history known at the valuation date, as a live pricing exercise would.
* Categorical levels are fixed on training data by :class:`CategoryEncoder`;
  rare and unseen levels are pooled to ``__other__`` and missing values form
  their own ``__missing__`` level.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from .claims import match_claims_to_rows
from .config import PERILS, Config

log = logging.getLogger(__name__)

META_COLUMNS: tuple[str, ...] = (
    "row_id", "policy_id", "customer_id", "address_id", "line", "year",
    "period_start", "period_end", "exposure", "is_test",
)
WEATHER_VARS: tuple[str, ...] = (
    "hail_days", "wind_exposure_index", "freezing_days", "dry_summer_index",
    "wildfire_index", "rainfall_mm", "mean_temperature_c",
)
MOTOR_WEATHER_VARS: tuple[str, ...] = (
    "hail_days", "wind_exposure_index", "freezing_days", "rainfall_mm", "mean_temperature_c",
)
HOME_LINES: tuple[str, ...] = ("buildings", "contents")
WEATHER_PERILS: tuple[str, ...] = ("storm", "flood", "subsidence", "escape_of_water")
HISTORY_PERILS: tuple[str, ...] = ("flood", "subsidence", "escape_of_water")

CREDIT_ORD = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
ENERGY_ORD = {k: i for i, k in enumerate("ABCDEFG", start=1)}
SECURITY_ORD = {"standard": 0, "locks": 1, "alarm_and_locks": 2}

MISSING_LEVEL = "__missing__"
OTHER_LEVEL = "__other__"

# Never carried into the feature frame (absent from test, or PII with no use).
_DROP_ALWAYS: tuple[str, ...] = ("renewal_decision", "surname", "date_of_birth", "exposure_reported")

CUSTOMER_COLUMNS: tuple[str, ...] = (
    "customer_id", "date_of_birth", "occupation", "marital_status", "household_size",
    "credit_band", "prior_claims_5y", "licence_points", "years_at_address",
    "customer_since_year", "email_domain", "marketing_consent", "document_preference",
    "previous_insurer", "campaign_code", "branch_id", "marketing_segment",
)

# Excluded beyond the config's forbidden list (evidence in docs/eda_report.md).
_EXTRA_EXCLUDED: dict[str, str] = {
    "customer_service_contacts": (
        "proven leakage: in training, rows with claims have a shifted contacts distribution "
        "(0 contacts: 0.2% vs 67%), while the 2025 test distribution matches the no-claim rows"
    ),
    "si_unit_error": "data-quality flag, not a risk characteristic",
    "wx_fallback": "diagnostic flag; differs by construction between train and prediction year",
    "customer_since_year": "calendar-year proxy; customer_tenure is used instead",
    "year_built": "property_age (year - year_built) is used instead",
    "compulsory_excess": "insurer-set and constant within line; total_excess is used instead",
    "vehicle_colour": "noise (EDA: relativities 0.97-1.02, out-of-time Gini 0.008)",
}

# ---------------------------------------------------------------------------
# Candidate feature lists by group
# ---------------------------------------------------------------------------
_POLICY_NUMERIC = (
    "voluntary_excess", "total_excess", "missed_payments_12m", "new_business",
    "renewal_count", "days_quote_to_inception", "exposure",
)
_POLICY_CATEGORICAL = ("channel", "payment_method")
_PERIOD_NUMERIC = (
    "start_month", "mid_doy", "winter_fraction", "is_partial_year",
    "starts_after_jan1", "ends_before_year_end",
)
_CUSTOMER_NUMERIC = (
    "household_size", "prior_claims_5y", "licence_points", "years_at_address",
    "customer_tenure", "credit_band_ord",
)
_CUSTOMER_CATEGORICAL = ("occupation", "marital_status")
_ADDRESS_NUMERIC_HOME = (
    "flood_zone", "distance_to_river_km", "log_distance_to_river", "elevation_m",
    "property_age", "basement", "trees_within_5m", "listed_building", "bedrooms",
    "energy_rating_ord", "solar_panels",
)
_ADDRESS_CATEGORICAL_HOME = ("region", "urban_density", "soil_type", "property_type", "roof_type")
_ADDRESS_CATEGORICAL_MOTOR = ("region", "urban_density")

_LINE_NUMERIC: dict[str, tuple[str, ...]] = {
    "motor": (
        "driver_age", "customer_age", "years_licensed", "age_first_licensed", "young_driver",
        "ncb_years", "vehicle_value_group", "vehicle_age", "engine_size_cc", "seats",
        "annual_mileage", "log_annual_mileage", "mileage_zero", "driver_assist",
        "named_young_driver", "telematics_opt_in", "telematics_score",
    ),
    "buildings": (
        "policyholder_age", "sum_insured", "log_sum_insured", "si_per_bedroom",
        "days_unoccupied", "smoke_alarms", "accidental_damage_cover", "mortgage",
    ),
    "contents": (
        "policyholder_age", "sum_insured", "log_sum_insured", "si_per_bedroom",
        "high_value_items", "security_level_ord", "accidental_damage_cover",
        "works_from_home", "lodgers", "bicycle_cover",
    ),
}
_LINE_CATEGORICAL: dict[str, tuple[str, ...]] = {
    "motor": (
        "make_model", "body_type", "transmission", "vehicle_colour", "fuel_type",
        "cover_type", "overnight_parking", "vehicle_use",
    ),
    "buildings": ("construction_type", "occupancy"),
    "contents": ("tenure",),
}
_REVIEW_CATEGORICAL = (
    "email_domain", "marketing_segment", "campaign_code", "branch_id",
    "document_preference", "previous_insurer",
)
_REVIEW_NUMERIC = ("marketing_consent", "written_premium", "discount_applied")

_MONOTONE_COMMON = {
    "prior_claims_5y": +1, "credit_band_ord": +1, "voluntary_excess": -1, "total_excess": -1,
}
_MONOTONE_LINE: dict[str, dict[str, int]] = {
    # vehicle_value_group is NOT constrained: one-way frequency falls with value group
    # (independent review V9); a +1 constraint is only defensible for OD/theft severity.
    "motor": {"ncb_years": -1, "licence_points": +1},
    "buildings": {"sum_insured": +1, "log_sum_insured": +1, "flood_zone": +1},
    "contents": {"sum_insured": +1, "log_sum_insured": +1, "high_value_items": +1},
}


# ---------------------------------------------------------------------------
# Feature specification (governance)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureSpec:
    """Which candidate columns may enter a model for one line, and why not otherwise."""

    line: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    monotone: dict[str, int] = field(default_factory=dict)
    review_numeric: tuple[str, ...] = ()
    review_categorical: tuple[str, ...] = ()
    excluded: dict[str, str] = field(default_factory=dict)

    @property
    def features(self) -> list[str]:
        """Default model features (review-required features are *not* included)."""
        return list(self.numeric) + list(self.categorical)

    @property
    def review_features(self) -> list[str]:
        return list(self.review_numeric) + list(self.review_categorical)

    def with_review(self, names: list[str] | tuple[str, ...]) -> "FeatureSpec":
        """Spec that also uses the named review-required features (for challenger tests)."""
        unknown = set(names) - set(self.review_features)
        if unknown:
            raise ValueError(f"Not review-required features for {self.line}: {sorted(unknown)}")
        return replace(
            self,
            numeric=self.numeric + tuple(n for n in names if n in self.review_numeric),
            categorical=self.categorical + tuple(n for n in names if n in self.review_categorical),
        )

    def restrict_to(self, available: list[str] | pd.Index) -> "FeatureSpec":
        """Drop candidates that are absent from a built frame (e.g. no weather file)."""
        av = set(available)
        return replace(
            self,
            numeric=tuple(c for c in self.numeric if c in av),
            categorical=tuple(c for c in self.categorical if c in av),
            monotone={k: v for k, v in self.monotone.items() if k in av},
            review_numeric=tuple(c for c in self.review_numeric if c in av),
            review_categorical=tuple(c for c in self.review_categorical if c in av),
        )

    def monotone_vector(self, features: list[str]) -> list[int]:
        """LightGBM ``monotone_constraints`` aligned to ``features``."""
        return [int(self.monotone.get(f, 0)) for f in features]


def history_feature_names(line: str, history_years: int = 3) -> tuple[str, ...]:
    """Names of the as-at claims-history features produced for ``line``."""
    w = history_years
    names = [
        "hist_pol_n_1y", f"hist_pol_n_{w}y", f"hist_pol_cost_{w}y", f"hist_pol_expo_{w}y",
        "hist_pol_years_since_claim", f"hist_cust_n_{w}y", f"hist_cust_n_other_{w}y",
    ]
    if line in HOME_LINES:
        names += [f"hist_addr_n_{w}y", f"hist_addr_wx_n_{w}y"]
        names += [f"hist_pol_{p}_{w}y" for p in HISTORY_PERILS if p in PERILS[line]]
    return tuple(dict.fromkeys(names))  # de-duplicate (w == 1) while keeping order


def weather_feature_names(line: str) -> tuple[str, ...]:
    vars_ = MOTOR_WEATHER_VARS if line == "motor" else WEATHER_VARS
    return tuple(f"wx_{v}{suffix}" for v in vars_ for suffix in ("", "_mean", "_anom"))


def feature_spec(line: str, cfg: Config, available: list[str] | pd.Index | None = None) -> FeatureSpec:
    """Governed feature specification for ``line``.

    ``excluded`` = config ``features.forbidden`` + proven-leakage / redundant
    columns; ``review_*`` = config ``features.review_required`` minus anything
    excluded. If ``available`` is given the spec is restricted to those columns.
    """
    if line not in PERILS:
        raise KeyError(f"Unknown line {line!r}")
    fcfg = cfg.section("features")
    w = int(fcfg.get("history_years", 3))

    excluded = {**cfg.forbidden_features, **_EXTRA_EXCLUDED}
    review = [c for c in cfg.review_required_features if c not in excluded]

    address_num = _ADDRESS_NUMERIC_HOME if line in HOME_LINES else ()
    address_cat = _ADDRESS_CATEGORICAL_HOME if line in HOME_LINES else _ADDRESS_CATEGORICAL_MOTOR
    hist = history_feature_names(line, w)
    numeric = (
        _POLICY_NUMERIC + _PERIOD_NUMERIC + _CUSTOMER_NUMERIC + address_num
        + _LINE_NUMERIC[line] + weather_feature_names(line) + hist
    )
    categorical = _POLICY_CATEGORICAL + _CUSTOMER_CATEGORICAL + address_cat + _LINE_CATEGORICAL[line]

    monotone = {**_MONOTONE_COMMON, **_MONOTONE_LINE[line]}
    # More past claims never lowers the expected cost (bonus-malus logic).
    monotone.update({h: +1 for h in hist if h.startswith(("hist_pol_n", "hist_cust_n",
                                                          "hist_addr", "hist_pol_cost"))})
    monotone.update({f"hist_pol_{p}_{w}y": +1 for p in HISTORY_PERILS if p in PERILS[line]})

    def _clean(cols: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(c for c in cols if c not in excluded and c not in review))

    spec = FeatureSpec(
        line=line,
        numeric=_clean(numeric),
        categorical=_clean(categorical),
        monotone={k: v for k, v in monotone.items() if k not in excluded},
        review_numeric=tuple(c for c in _REVIEW_NUMERIC if c in review),
        review_categorical=tuple(c for c in _REVIEW_CATEGORICAL if c in review),
        excluded=excluded,
    )
    spec = replace(spec, monotone={k: v for k, v in spec.monotone.items()
                                   if k in spec.numeric or k in spec.review_numeric})
    return spec.restrict_to(available) if available is not None else spec


# ---------------------------------------------------------------------------
# Categorical encoding fixed on training data
# ---------------------------------------------------------------------------
def _as_levels(s: pd.Series) -> pd.Series:
    """Text levels with missing values mapped to an explicit level."""
    if isinstance(s.dtype, pd.CategoricalDtype):
        s = s.astype("object")
    out = s.astype("string").str.strip()
    return out.fillna(MISSING_LEVEL).astype(object)


@dataclass
class CategoryEncoder:
    """Fix categorical levels on training data; pool rare/unseen levels.

    Levels with training exposure below ``min_exposure`` and levels never seen in
    training become ``__other__``; missing values become ``__missing__``. The
    resulting pandas ``category`` dtype is identical for train and test, which
    LightGBM requires for consistent split encodings.
    """

    min_exposure: float = 200.0
    exposure_col: str = "exposure"
    categories_: dict[str, list[str]] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, columns: list[str] | tuple[str, ...]) -> "CategoryEncoder":
        self.categories_ = {}
        weights = df[self.exposure_col] if self.exposure_col in df else pd.Series(1.0, index=df.index)
        for col in columns:
            lv = _as_levels(df[col])
            expo = weights.groupby(lv.to_numpy()).sum()
            keep = sorted(
                str(k) for k, v in expo.items()
                if v >= self.min_exposure and k not in (MISSING_LEVEL, OTHER_LEVEL)
            )
            self.categories_[col] = keep + [OTHER_LEVEL, MISSING_LEVEL]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.categories_:
            raise RuntimeError("CategoryEncoder must be fitted before transform")
        out = df.copy()
        for col, levels in self.categories_.items():
            if col not in out.columns:
                continue
            lv = _as_levels(out[col])
            lv = lv.where(lv.isin(set(levels)), OTHER_LEVEL)
            out[col] = pd.Categorical(lv, categories=levels)
        return out

    def fit_transform(self, df: pd.DataFrame, columns: list[str] | tuple[str, ...]) -> pd.DataFrame:
        return self.fit(df, columns).transform(df)

    def to_dict(self) -> dict[str, Any]:
        return {"min_exposure": self.min_exposure, "exposure_col": self.exposure_col,
                "categories": self.categories_}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CategoryEncoder":
        enc = cls(min_exposure=float(d["min_exposure"]), exposure_col=str(d["exposure_col"]))
        enc.categories_ = {k: list(v) for k, v in d["categories"].items()}
        return enc


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def winter_fraction(start: pd.Series, end: pd.Series, months: tuple[int, ...] = (12, 1, 2)) -> pd.Series:
    """Share of covered days (inclusive period) falling in the given months.

    Vectorised via a cumulative daily calendar, so rows spanning several years
    are handled correctly too.
    """
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    if len(start) == 0:
        return pd.Series(dtype=float, index=start.index)
    origin = start.min() - pd.Timedelta(days=1)
    cal = pd.date_range(origin, end.max(), freq="D")
    cum = np.cumsum(np.isin(cal.month, months)).astype(np.int64)
    i_end = (end - origin).dt.days.to_numpy()
    i_before = (start - origin).dt.days.to_numpy() - 1
    winter_days = cum[i_end] - cum[i_before]
    days = i_end - i_before
    return pd.Series(winter_days / days, index=start.index, dtype=float)


def period_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features describing when in the year the row is on cover.

    The data use a 365-day policy year (exposure = days / 365): a full 2024 row
    runs 1 Jan - 30 Dec because 2024 is a leap year. "Full year" and "ends
    early" are therefore defined against ``1 Jan + 364 days``, not 31 Dec,
    otherwise every 2024 row would look truncated.
    """
    s, e = df["period_start"], df["period_end"]
    days = (e - s).dt.days + 1
    year_start = pd.to_datetime(s.dt.year.astype(str) + "-01-01")
    policy_year_end = year_start + pd.Timedelta(days=364)
    mid = s + (e - s) / 2
    return pd.DataFrame(
        {
            "start_month": s.dt.month,
            "mid_doy": mid.dt.dayofyear,
            "winter_fraction": winter_fraction(s, e),
            "is_partial_year": days < 365,
            "starts_after_jan1": s > year_start,
            "ends_before_year_end": e < policy_year_end,
        },
        index=df.index,
    ).astype("float32")


def pad_customer_ids(df: pd.DataFrame, width: int, table: str = "") -> pd.DataFrame:
    """Restore leading zeros lost when numeric-looking ids were stored as numbers.

    Profiling found ~10% of motor ``customer_id`` values (train and test) with
    6-7 digits instead of 8; every padded value matches the customers table.
    Applied defensively before joins (a logged fix belongs in ``cleaning.py``).
    """
    if "customer_id" not in df.columns:
        return df
    s = df["customer_id"].astype("string")
    short = s.str.len() < width
    if short.any():
        log.warning("%s: %d customer_id values lacked leading zeros; padded to width %d",
                    table or "table", int(short.sum()), width)
        df = df.copy()
        df["customer_id"] = s.str.zfill(width)
    return df


def _customer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if "date_of_birth" in df:
        out["customer_age"] = np.floor((df["period_start"] - df["date_of_birth"]).dt.days / 365.25)
    if "customer_since_year" in df:
        out["customer_tenure"] = df["year"] - df["customer_since_year"]
    if "credit_band" in df:
        out["credit_band_ord"] = df["credit_band"].map(CREDIT_ORD).astype("float32")
    return out.astype("float32")


def _address_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if "year_built" in df:
        out["property_age"] = df["year"] - df["year_built"]
    if "distance_to_river_km" in df:
        out["log_distance_to_river"] = np.log(df["distance_to_river_km"].clip(lower=0.005))
    if "energy_rating" in df:
        out["energy_rating_ord"] = df["energy_rating"].map(ENERGY_ORD).astype("float32")
    return out.astype("float32")


def _line_features(df: pd.DataFrame, line: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if "voluntary_excess" in df and "compulsory_excess" in df:
        out["total_excess"] = df["voluntary_excess"] + df["compulsory_excess"]
    if line == "motor":
        out["age_first_licensed"] = df["driver_age"] - df["years_licensed"]
        out["young_driver"] = (df["driver_age"] < 25).astype("float32")
        mileage = df["annual_mileage"].astype("float64")
        out["mileage_zero"] = (mileage == 0).astype("float32")
        mileage = mileage.where(mileage > 0)
        out["annual_mileage"] = mileage  # 0 recoded as missing (flag kept)
        out["log_annual_mileage"] = np.log(mileage)
        tel = df["telematics_score"].astype("float64")
        if "telematics_opt_in" in df:
            tel = tel.where(df["telematics_opt_in"] == 1)
        out["telematics_score"] = tel
    else:
        si = df["sum_insured"].astype("float64")
        out["log_sum_insured"] = np.log(si)
        if "bedrooms" in df:
            out["si_per_bedroom"] = si / df["bedrooms"].clip(lower=1)
        if line == "contents" and "security_level" in df:
            out["security_level_ord"] = df["security_level"].map(SECURITY_ORD).astype("float32")
    return out.astype("float32")


def weather_features(
    rows: pd.DataFrame,
    weather: pd.DataFrame | None,
    last_train_year: int,
    mode: str = "use_if_available",
    variables: tuple[str, ...] = WEATHER_VARS,
) -> pd.DataFrame:
    """Weather for each row's (address, year) plus long-run mean and anomaly.

    * ``wx_<v>_mean``: address mean over all available years up to
      ``last_train_year`` (the climate, a stable risk characteristic);
    * ``wx_<v>``: value for the row's year; for years after ``last_train_year``
      it is used only if present and ``mode == 'use_if_available'``, otherwise
      (and whenever the year is missing) it falls back to the address mean;
    * ``wx_<v>_anom``: ``wx_<v> - wx_<v>_mean`` (0 under fallback);
    * ``wx_fallback``: 1 where the fallback was applied (diagnostic only).
    """
    if weather is None:
        return pd.DataFrame(index=rows.index)
    if mode not in ("use_if_available", "address_mean"):
        raise ValueError(f"Unknown weather mode {mode!r}")
    vars_ = [v for v in variables if v in weather.columns]
    w = weather[["address_id", "year", *vars_]].copy()
    w["year"] = w["year"].astype(int)
    if w.duplicated(["address_id", "year"]).any():
        log.warning("weather has duplicate (address_id, year) keys; averaging them")
        w = w.groupby(["address_id", "year"], as_index=False)[vars_].mean()
    hist = w[w["year"] <= last_train_year]
    means = hist.groupby("address_id")[vars_].mean()

    key = rows[["address_id", "year"]].copy()
    key["year"] = key["year"].astype(int)
    cur = key.merge(w, on=["address_id", "year"], how="left")
    cur.index = rows.index
    mean_rows = key[["address_id"]].merge(means, left_on="address_id", right_index=True, how="left")
    mean_rows.index = rows.index

    if mode == "address_mean":
        cur.loc[key["year"] > last_train_year, vars_] = np.nan
    fallback = cur[vars_].isna().all(axis=1) & mean_rows[vars_].notna().any(axis=1)

    out = pd.DataFrame(index=rows.index)
    for v in vars_:
        val = cur[v].fillna(mean_rows[v])
        out[f"wx_{v}"] = val
        out[f"wx_{v}_mean"] = mean_rows[v]
        out[f"wx_{v}_anom"] = val - mean_rows[v]
    out["wx_fallback"] = fallback
    return out.astype("float32")


def claim_events(claims: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """Non-repudiated claims with the customer and the address at the time of loss.

    The customer comes from the policy (``policy_id -> customer_id`` in
    ``keys``); the address is that of the policy row covering the loss date,
    falling back to the policy's first row (e.g. losses just before the data
    starts). Claims whose policy is absent from ``keys`` keep a missing customer
    and address and therefore only feed policy-level history. Computing this
    once and passing it as ``claims`` to :func:`build_line_frame` avoids
    repeating the interval join for every chunk.
    """
    cols = ["claim_id", "policy_id", "line", "peril", "loss_date", "report_date", "net_incurred"]
    ev = claims.loc[~claims["is_repudiated"].astype(bool), cols].copy()

    pol = keys.sort_values("period_start")
    n_cust = pol.groupby("policy_id")["customer_id"].nunique()
    if (n_cust > 1).any():
        log.warning("%d policies change customer_id; latest customer used", int((n_cust > 1).sum()))
    cust_map = pol.drop_duplicates("policy_id", keep="last").set_index("policy_id")["customer_id"]
    ev["customer_id"] = ev["policy_id"].map(cust_map)

    matched = match_claims_to_rows(ev[["claim_id", "policy_id", "loss_date"]], keys)
    row_addr = keys.set_index("row_id")["address_id"]
    addr = matched.set_index("claim_id")["row_id"].map(row_addr)
    ev["address_id"] = ev["claim_id"].map(addr)
    first_addr = pol.drop_duplicates("policy_id", keep="first").set_index("policy_id")["address_id"]
    ev["address_id"] = ev["address_id"].fillna(ev["policy_id"].map(first_addr))

    unmapped = ev["customer_id"].isna().mean()
    if unmapped > 0:
        log.info("%.1f%% of claims have no customer mapping (policy table not loaded)", 100 * unmapped)
    return ev


def claims_history_features(
    rows: pd.DataFrame,
    line: str,
    claims: pd.DataFrame,
    all_policy_keys: pd.DataFrame,
    history_years: int = 3,
    cost_caps: dict[str, float] | None = None,
    cap_quantile: float = 0.995,
) -> pd.DataFrame:
    """As-at claims-history features for ``rows`` (aligned to ``rows.index``).

    For a row in year ``Y`` the cut-off is 1 January ``Y``: only non-repudiated
    claims with ``loss_date`` and ``report_date`` strictly before the cut-off
    count. Windows are calendar years ``[Y - k, Y - 1]`` on loss date.
    """
    w = history_years
    names = history_feature_names(line, w)
    out = pd.DataFrame(0.0, index=rows.index, columns=list(names), dtype="float64")
    out["hist_pol_years_since_claim"] = np.nan
    if rows.empty:
        return out.astype("float32")

    if {"customer_id", "address_id"} <= set(claims.columns):
        ev = claims  # already mapped by claim_events()
        if "is_repudiated" in ev.columns:
            ev = ev.loc[~ev["is_repudiated"].astype(bool)]
    else:
        ev = claim_events(claims, all_policy_keys)
    ev = ev.copy()
    if cost_caps is None:
        cost_caps = ev.groupby("line")["net_incurred"].quantile(cap_quantile).to_dict()
    ev["cost_capped"] = np.minimum(ev["net_incurred"], ev["line"].map(cost_caps).fillna(np.inf))

    keys = all_policy_keys
    pe = (
        keys.assign(kyear=keys["period_start"].dt.year)
        .groupby(["policy_id", "kyear"])["exposure"].sum()
        .reset_index()
    )
    line_perils = [p for p in HISTORY_PERILS if p in PERILS[line] and line in HOME_LINES]

    for y in sorted(rows["year"].astype(int).unique()):
        idx = rows.index[rows["year"].astype(int) == y]
        sub = rows.loc[idx]
        cutoff = pd.Timestamp(year=y, month=1, day=1)
        start_1 = pd.Timestamp(year=y - 1, month=1, day=1)
        start_w = pd.Timestamp(year=y - w, month=1, day=1)
        known = ev[(ev["report_date"] < cutoff) & (ev["loss_date"] < cutoff)]
        in_w = known[known["loss_date"] >= start_w]

        # Policy level ---------------------------------------------------------
        kp = known[known["policy_id"].isin(set(sub["policy_id"]))]
        kp_w = kp[kp["loss_date"] >= start_w]
        pol = pd.DataFrame(index=pd.Index(sub["policy_id"].unique(), name="policy_id"))
        pol["hist_pol_n_1y"] = kp[kp["loss_date"] >= start_1].groupby("policy_id").size()
        pol[f"hist_pol_n_{w}y"] = kp_w.groupby("policy_id").size()
        pol[f"hist_pol_cost_{w}y"] = kp_w.groupby("policy_id")["cost_capped"].sum()
        for p in line_perils:
            pol[f"hist_pol_{p}_{w}y"] = kp_w[kp_w["peril"] == p].groupby("policy_id").size()
        # Windowed like the counts, so every year sees the same look-back horizon
        # (otherwise 2025 rows would see 4 years of data vs 3 for 2024 rows).
        last = kp_w.groupby("policy_id")["loss_date"].max()
        pol["hist_pol_years_since_claim"] = (cutoff - last).dt.days / 365.25
        pe_y = pe[(pe["kyear"] >= y - w) & (pe["kyear"] <= y - 1)]
        pol[f"hist_pol_expo_{w}y"] = pe_y.groupby("policy_id")["exposure"].sum()
        fill_cols = [c for c in pol.columns if c != "hist_pol_years_since_claim"]
        pol[fill_cols] = pol[fill_cols].fillna(0.0)
        merged = sub[["policy_id"]].join(pol, on="policy_id")
        out.loc[idx, pol.columns] = merged[pol.columns].to_numpy(dtype="float64")

        # Customer level (all lines) ------------------------------------------
        cust_all = in_w.groupby("customer_id").size()
        cust_other = in_w[in_w["line"] != line].groupby("customer_id").size()
        out.loc[idx, f"hist_cust_n_{w}y"] = sub["customer_id"].map(cust_all).fillna(0).to_numpy()
        out.loc[idx, f"hist_cust_n_other_{w}y"] = sub["customer_id"].map(cust_other).fillna(0).to_numpy()

        # Address level (home lines, all customers) ---------------------------
        if line in HOME_LINES:
            home = in_w[in_w["line"].isin(HOME_LINES)]
            addr_all = home.groupby("address_id").size()
            addr_wx = home[home["peril"].isin(WEATHER_PERILS)].groupby("address_id").size()
            out.loc[idx, f"hist_addr_n_{w}y"] = sub["address_id"].map(addr_all).fillna(0).to_numpy()
            out.loc[idx, f"hist_addr_wx_n_{w}y"] = sub["address_id"].map(addr_wx).fillna(0).to_numpy()

    return out.astype("float32")


def policy_keys(policies: pd.DataFrame, line: str) -> pd.DataFrame:
    """Light key frame for one cleaned training policy table (see ``all_policy_keys``)."""
    cols = ["row_id", "policy_id", "customer_id", "address_id", "period_start", "period_end", "exposure"]
    k = policies[cols].copy()
    k["line"] = line
    return k


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def build_line_frame(
    line: str,
    policies: pd.DataFrame,
    customers: pd.DataFrame,
    addresses: pd.DataFrame,
    weather: pd.DataFrame | None,
    claims: pd.DataFrame,
    all_policy_keys: pd.DataFrame,
    cfg: Config,
    is_test: bool,
) -> pd.DataFrame:
    """One modelling frame for ``line``: metadata + all candidate features.

    ``policies`` must already be cleaned (:func:`cleaning.clean_policy_table`),
    ``customers``/``addresses``/``weather`` cleaned with
    :func:`cleaning.clean_reference_table` and ``claims`` prepared with
    :func:`claims.prepare_claims` (or already mapped by :func:`claim_events`,
    which is faster when the frame is built in chunks). ``all_policy_keys``
    covers the *training* rows of every available line (see :func:`policy_keys`).
    Rows are independent, so callers may build a large table year by year.
    """
    if line not in PERILS:
        raise KeyError(f"Unknown line {line!r}")
    fcfg = cfg.section("features")
    n0 = len(policies)
    if policies["row_id"].duplicated().any():
        raise ValueError(f"{line}: duplicate row_id in policy table")

    df = policies.drop(columns=[c for c in _DROP_ALWAYS if c in policies.columns]).copy()
    id_width = int(customers["customer_id"].astype("string").str.len().max())
    df = pad_customer_ids(df, id_width, table=f"{line}{' test' if is_test else ''}")
    for c in df.columns:  # downcast early to bound peak memory during the joins
        if c != "exposure" and pd.api.types.is_float_dtype(df[c]) and df[c].dtype != np.float32:
            df[c] = df[c].astype("float32")
    df["line"] = line
    df["year"] = df["period_start"].dt.year.astype(int)
    df["is_test"] = bool(is_test)

    cust = customers[[c for c in CUSTOMER_COLUMNS if c in customers.columns]]
    df = df.merge(cust, on="customer_id", how="left", validate="many_to_one")
    df = df.merge(addresses, on="address_id", how="left", validate="many_to_one")
    if len(df) != n0:
        raise AssertionError(f"{line}: row count changed during joins ({n0} -> {len(df)})")

    derived = [
        _customer_features(df), _address_features(df), period_features(df), _line_features(df, line),
    ]
    for block in derived:
        df[block.columns] = block  # overwrite raw where recoded (e.g. annual_mileage)
    df = df.drop(columns=[c for c in ("date_of_birth",) if c in df.columns])

    wx_vars = MOTOR_WEATHER_VARS if line == "motor" else WEATHER_VARS
    wx = weather_features(
        df, weather, last_train_year=max(cfg.train_years),
        mode=str(fcfg.get("weather_prediction_year", "use_if_available")), variables=wx_vars,
    )
    if not wx.empty:
        df[wx.columns] = wx

    hist = claims_history_features(
        df, line, claims, all_policy_keys,
        history_years=int(fcfg.get("history_years", 3)),
        cap_quantile=float(cfg.section("model").get("large_loss", {}).get("cap_quantile", 0.995)),
    )
    df[hist.columns] = hist

    # Model-ready dtypes: numeric -> float32 (ids, dates, text untouched).
    for c in df.columns:
        if c in META_COLUMNS:
            continue
        if pd.api.types.is_bool_dtype(df[c]) or (
            pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_float_dtype(df[c])
        ):
            df[c] = df[c].astype("float32")
        elif pd.api.types.is_float_dtype(df[c]) and df[c].dtype != np.float32:
            df[c] = df[c].astype("float32")

    meta = [c for c in META_COLUMNS if c in df.columns]
    df = df[meta + [c for c in df.columns if c not in meta]]
    if df["row_id"].duplicated().any() or len(df) != n0:
        raise AssertionError(f"{line}: feature frame lost or duplicated rows")
    return df


# ---------------------------------------------------------------------------
# Cost-level alignment and high-cardinality encoding (applied at model time)
# ---------------------------------------------------------------------------
def index_sum_insured(df: pd.DataFrame, target_year: int, annual_rate: float) -> pd.DataFrame:
    """Express ``sum_insured`` (and its derived features) in ``target_year`` money.

    Sums insured are indexed each year (+3% buildings, +2% contents, measured as the
    median year-on-year change on renewals). Claims are already on-levelled to the
    target year, so leaving historical sums insured unindexed would double count
    inflation (independent review V5).
    """
    if "sum_insured" not in df.columns or annual_rate == 0:
        return df
    out = df.copy()
    f = np.power(1.0 + annual_rate, float(target_year) - out["year"].astype(float))
    out["sum_insured"] = (out["sum_insured"].astype(float) * f).astype("float32")
    out["log_sum_insured"] = np.log(out["sum_insured"].astype(float)).astype("float32")
    if "si_per_bedroom" in out.columns and "bedrooms" in out.columns:
        out["si_per_bedroom"] = (out["sum_insured"] / out["bedrooms"].clip(lower=1)).astype("float32")
    return out


def credibility_encode(
    train: pd.DataFrame,
    others: list[pd.DataFrame],
    col: str,
    y_col: str = "y_capped",
    offset_col: str = "log_offset",
    count_col: str = "n_claims",
    k: float = 100.0,
    n_folds: int = 5,
    fold_key: str = "policy_id",
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Buhlmann-style credibility relativity of a high-cardinality factor (log scale).

    For level l: ``rel_l = sum(y) / sum(expected)`` where expected is the offset-only
    (no-factor) cost, and ``Z_l = N_l / (N_l + k)`` with ``N_l`` claims. The encoding
    is ``log(Z_l * rel_l + (1 - Z_l))``. Training rows get out-of-fold values (folds
    grouped by ``fold_key``) so no row sees its own claims; ``others`` use all training rows.
    """
    y = train[y_col].to_numpy(dtype=float)
    mu0 = np.exp(train[offset_col].to_numpy(dtype=float))
    mu0 = mu0 * y.sum() / mu0.sum()
    n = train[count_col].to_numpy(dtype=float) if count_col in train else (y > 0).astype(float)
    lv = train[col].astype(str).to_numpy()
    h = pd.util.hash_array(train[fold_key].astype(str).to_numpy()) % n_folds

    def table(mask: np.ndarray) -> pd.DataFrame:
        g = pd.DataFrame({"l": lv[mask], "y": y[mask], "e": mu0[mask], "n": n[mask]}).groupby("l").sum()
        z = g["n"] / (g["n"] + k)
        return np.log(z * (g["y"] / g["e"]) + (1 - z))

    enc_train = np.zeros(len(train))
    for f in range(n_folds):
        t = table(h != f)
        idx = h == f
        enc_train[idx] = pd.Series(lv[idx]).map(t).fillna(0.0).to_numpy()
    full = table(np.ones(len(train), dtype=bool))
    enc_others = [o[col].astype(str).map(full).fillna(0.0).to_numpy() for o in others]
    return enc_train.astype("float32"), [e.astype("float32") for e in enc_others]
