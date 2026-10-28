"""Row-level modelling targets and offsets ("trended, developed, on-levelled losses").

For each training row and a chosen cost level ``target_year`` (2025 for the
final model; the holdout year in back-tests) we build:

* ``y``           sum of claim ``net_ultimate`` (open claims developed for future
                  recoveries; suspect amounts corrected) x severity on-level factor
                  ``(1 + r_sev)^(T - loss_year)`` x any level-shift factor;
* ``y_capped``    the same with each on-levelled claim capped at the line's large-loss
                  cap (excess reloaded multiplicatively by ``large_loss_load``);
* ``y_<peril>``, ``n_<peril>``   peril-level costs (uncapped) and counts;
* ``comp_<peril>``, ``comp_line``  expected share of ultimate cost reported at the
                  valuation date (IBNR completeness, cost-weighted pattern);
* ``freq_trend``  ``(1 + r_freq)^(T - year)``;
* ``log_offset``  ``log(exposure x comp_line x freq_trend)``: the model estimates the
                  ultimate rate at cost level ``T``, and the IBNR shortfall and trend
                  are handled exactly, without inflating the observed losses.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from . import development as dev
from .claims import match_claims_to_rows, prepare_claims
from .config import PERILS, Config
from .io import read_table

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrendAssumptions:
    """Annual trend rates for one line (from ``trend.recommended`` in the config)."""

    frequency: float
    severity_by_peril: dict[str, float]
    level_shift: dict[str, Any] | None = None   # {"peril":..., "years": [...], "factor": ...}

    @classmethod
    def from_config(cls, cfg: Config, line: str) -> "TrendAssumptions":
        rec = (cfg.section("trend").get("recommended", {}) or {}).get(line, {}) or {}
        freq = float(rec.get("frequency", 0.0))
        if "severity_by_peril" in rec:
            sev = {p: float(rec["severity_by_peril"].get(p, 0.0)) for p in PERILS[line]}
        else:
            sev = {p: float(rec.get("severity", 0.0)) for p in PERILS[line]}
        shift = None
        if "bodily_injury_level_shift" in rec:
            s = rec["bodily_injury_level_shift"]
            shift = {"peril": "bodily_injury", "years": [int(y) for y in s["years"]],
                     "factor": float(s["factor"])}
        cap = float(cfg.section("trend").get("max_abs_annual_rate", 0.15))
        for name, r in [("frequency", freq), *sev.items()]:
            if abs(r) > cap:
                raise ValueError(f"{line} trend {name}={r:.3f} breaches guardrail {cap:.2f}")
        return cls(frequency=freq, severity_by_peril=sev, level_shift=shift)

    def claim_factor(self, peril: pd.Series, loss_year: pd.Series, target_year: int) -> np.ndarray:
        """Severity on-level factor for each claim to the cost level of ``target_year``."""
        r = peril.map(self.severity_by_peril).fillna(0.0).to_numpy(dtype=float)
        f = dev.trend_factor(loss_year.to_numpy(dtype=float), target_year, r)
        if self.level_shift is not None:
            hit = (peril == self.level_shift["peril"]).to_numpy() & loss_year.isin(
                self.level_shift["years"]).to_numpy()
            f = np.where(hit, f * self.level_shift["factor"], f)
        return f

    def row_frequency_factor(self, year: pd.Series, target_year: int) -> np.ndarray:
        return dev.trend_factor(year.to_numpy(dtype=float), target_year, self.frequency)


# Keys of the ``development:`` YAML section and how each is consumed. Unknown keys fail
# loudly (independent review V3: YAML edits must never be silently ignored).
_DEV_KEYS_USED = {
    "lag_unit_days", "tail_months", "min_claims_per_peril", "ibner_method",
    "open_claim_repudiation", "min_loss_date", "daily_lag_until_days", "pattern_method",
    "tail_min_mass", "tail_max_ks", "cost_cap_quantile", "min_row_completeness", "share_years",
    "mature_years", "immature_years", "repudiation_min_z", "ibner_min_z", "suspect_ratio",
    "suspect_treatment",
}
# Documented for the reader but not a model input (the observable window follows from
# ``min_loss_date`` and the valuation date).
_DEV_KEYS_INFORMATIONAL = {"max_lag_months"}
_SUSPECT_TREATMENTS = {"decimal_shift"}
_REPUDIATION_MODES = {"none", "estimate"}
_IBNER_METHODS = {"none", "ay_severity_ratio"}


def development_settings(cfg: Config) -> dict[str, dict[str, Any]]:
    """Resolve the ``development:`` YAML into explicit keyword arguments.

    Returns ``{"pattern": kwargs for reporting_pattern, "params": kwargs for
    estimate_development_params, "shares": kwargs for peril_cost_shares, "row": kwargs
    for row_completeness, "claims": kwargs for prepare_claims}``. Raises on unknown keys
    or unsupported values so a typo in the config cannot silently fall back to a default.
    """
    d = cfg.section("development")
    unknown = set(d) - _DEV_KEYS_USED - _DEV_KEYS_INFORMATIONAL
    if unknown:
        raise ValueError(f"Unknown development config keys (not consumed by the code): {sorted(unknown)}")
    treatment = str(d.get("suspect_treatment", "decimal_shift"))
    if treatment not in _SUSPECT_TREATMENTS:
        raise ValueError(f"development.suspect_treatment={treatment!r} not supported: {_SUSPECT_TREATMENTS}")
    repud = str(d.get("open_claim_repudiation", "estimate"))
    if repud not in _REPUDIATION_MODES:
        raise ValueError(f"development.open_claim_repudiation must be one of {_REPUDIATION_MODES}")
    ibner = str(d.get("ibner_method", "none"))
    if ibner not in _IBNER_METHODS:
        raise ValueError(f"development.ibner_method must be one of {_IBNER_METHODS}")

    bucket = float(d.get("lag_unit_days", 30.4375))
    cap_q = float(d.get("cost_cap_quantile", 0.995))
    suspect_ratio = float(d.get("suspect_ratio", 5.0))
    floor = bool((cfg.section("claims") or {}).get("floor_net_cost_at_zero", True))
    return {
        "pattern": {
            "max_days": max(int(round(float(d.get("tail_months", 120)) * bucket)), 3650),
            "cost_cap_quantile": cap_q,
            "method": d.get("pattern_method", "auto"),
            "min_loss_date": d.get("min_loss_date"),
            "daily_until": int(d.get("daily_lag_until_days", 60)),
            "bucket_days": bucket,
            "tail_min_mass": float(d.get("tail_min_mass", 0.002)),
            "tail_max_ks": float(d.get("tail_max_ks", 0.03)),
            "min_claims_per_peril": int(d.get("min_claims_per_peril", 150)),
        },
        "params": {
            "mature_years": [int(y) for y in d.get("mature_years", [2021, 2022])],
            "immature_years": [int(y) for y in d.get("immature_years", [2023, 2024])],
            "repudiation": repud,
            "repudiation_min_z": float(d.get("repudiation_min_z", 2.0)),
            "ibner_method": ibner,
            "ibner_min_z": float(d.get("ibner_min_z", 2.0)),
            "suspect_ratio": suspect_ratio,
        },
        "shares": {
            "years": [int(y) for y in d.get("share_years", [2021, 2022])],
            "cap_quantile": cap_q,
        },
        "row": {"min_value": float(d.get("min_row_completeness", 1e-6))},
        "claims": {"floor_at_zero": floor, "suspect_ratio": suspect_ratio},
    }


def claims_from_config(cfg: Config, raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Read (unless given) and prepare the claims table with the configured settings."""
    raw = read_table(cfg.data_dir, "claims") if raw is None else raw
    return prepare_claims(raw, **development_settings(cfg)["claims"])


def ay_completeness(patterns: Mapping[str, pd.DataFrame], valuation_date: pd.Timestamp,
                    years: list[int]) -> pd.DataFrame:
    """Loss-year completeness (losses uniform over each calendar year) by line x peril.

    ``patterns`` maps a column suffix (``"count"``, ``"cost"``) to a reporting pattern;
    returns ``line, peril, loss_year, comp_<suffix>...`` as used by the IBNER test.
    """
    rows = pd.DataFrame({
        "row_id": [str(y) for y in years],
        "period_start": pd.to_datetime([f"{y}-01-01" for y in years]),
        "period_end": pd.to_datetime([f"{y}-12-31" for y in years]),
    })
    out = None
    for suffix, pattern in patterns.items():
        recs = []
        for line in sorted(pattern["line"].unique()):
            rc = dev.row_completeness(rows, pattern, line, valuation_date)
            for col in [c for c in rc.columns if c.startswith("comp_")]:
                for y, v in zip(years, rc[col]):
                    recs.append({"line": line, "peril": col[5:], "loss_year": int(y), f"comp_{suffix}": float(v)})
        frame = pd.DataFrame(recs)
        out = frame if out is None else out.merge(frame, on=["line", "peril", "loss_year"])
    return out


@dataclass
class DevelopmentInputs:
    """Everything the target builder needs from the reserving layer (fitted once)."""

    claims_ult: pd.DataFrame          # prepared claims + net_ultimate
    pattern_cost: pd.DataFrame        # cost-weighted reporting pattern
    peril_shares: dict[str, dict[str, float]]
    params: dict[str, Any]
    settings: dict[str, dict[str, Any]] = field(default_factory=dict)   # resolved config (audit)
    pattern_count: pd.DataFrame | None = None   # count-weighted pattern (frequency experts)

    @classmethod
    def fit(cls, claims_prepared: pd.DataFrame, cfg: Config) -> "DevelopmentInputs":
        """Fit patterns, open-claim parameters and peril shares with every setting explicit."""
        st = development_settings(cfg)
        V = cfg.valuation_date
        pattern = dev.reporting_pattern(claims_prepared, V, weight="cost", **st["pattern"])
        pattern_n = dev.reporting_pattern(claims_prepared, V, weight="count", **st["pattern"])
        ay = None
        if st["params"]["ibner_method"] != "none":
            ay = ay_completeness({"count": pattern_n, "cost": pattern}, V, list(cfg.train_years))
        params = dev.estimate_development_params(claims_prepared, ay_completeness=ay, **st["params"])
        claims_ult = dev.open_claim_ultimate(claims_prepared, params)
        shares = {line: dev.peril_cost_shares(claims_ult, line, st["shares"]["years"],
                                              cap_quantile=st["shares"]["cap_quantile"])
                  for line in PERILS}
        return cls(claims_ult=claims_ult, pattern_cost=pattern, peril_shares=shares, params=params,
                   settings=st, pattern_count=pattern_n)


def large_loss_cap(claim_costs: pd.Series, quantile: float | None) -> float:
    """Claim-level cap at ``quantile`` of positive on-levelled claim costs (inf = no cap)."""
    pos = claim_costs[claim_costs > 0]
    if quantile is None or quantile >= 1.0 or pos.empty:
        return float("inf")
    return float(pos.quantile(quantile))


def build_targets(
    line: str,
    rows: pd.DataFrame,
    dev_inputs: DevelopmentInputs,
    cfg: Config,
    target_year: int,
    cap_quantile: float | None = None,
    fit_years: list[int] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Targets and offsets for the training ``rows`` of one line at cost level ``target_year``.

    Returns ``(frame, summary)`` where ``frame`` is keyed by ``row_id`` and
    ``summary`` records the cap, the large-loss load and totals (for the run report).
    The cap is the ``cap_quantile`` of on-levelled claim costs and the large-loss load is
    ``sum(y) / sum(y_capped)``, both computed only on ``fit_years`` (default: all years),
    and the load only on those of them that are mature (mean cost completeness >= 0.95).
    Pass the training years in a back-test so nothing is learnt from the holdout year.
    """
    perils = list(PERILS[line])
    trend = TrendAssumptions.from_config(cfg, line)
    c = dev_inputs.claims_ult
    c = c.loc[(c["line"] == line) & ~c["is_repudiated"].astype(bool)]
    m = match_claims_to_rows(c, rows)
    m = m.loc[m["match_status"] == "matched"].copy()

    m["cost_T"] = m["net_ultimate"].to_numpy(dtype=float) * trend.claim_factor(
        m["peril"], m["loss_year"], target_year)
    fit_mask = m["loss_year"].isin(fit_years) if fit_years is not None else pd.Series(True, index=m.index)
    cap = large_loss_cap(m.loc[fit_mask, "cost_T"], cap_quantile)
    m["cost_T_capped"] = np.minimum(m["cost_T"], cap)

    g = m.groupby("row_id")
    agg = pd.DataFrame({
        "y": g["cost_T"].sum(),
        "y_capped": g["cost_T_capped"].sum(),
        "n_claims": g.size(),
        "y_reported": g["net_incurred"].sum(),
    })
    pc = m.pivot_table(index="row_id", columns="peril", values="cost_T", aggfunc="sum", fill_value=0.0)
    pn = m.pivot_table(index="row_id", columns="peril", values="claim_id", aggfunc="size", fill_value=0)
    pc = pc.reindex(columns=perils, fill_value=0.0).add_prefix("y_")
    pn = pn.reindex(columns=perils, fill_value=0).add_prefix("n_")
    # Experts' targets: capped cost by peril, and the large-loss part (count and excess).
    pcc = m.pivot_table(index="row_id", columns="peril", values="cost_T_capped", aggfunc="sum",
                        fill_value=0.0).reindex(columns=perils, fill_value=0.0).add_prefix("yc_")
    m["is_large"] = (m["cost_T"] > cap).astype(float)
    m["excess"] = np.maximum(m["cost_T"] - cap, 0.0)
    agg["n_large"] = m.groupby("row_id")["is_large"].sum()
    agg["y_excess"] = m.groupby("row_id")["excess"].sum()

    out = rows[["row_id", "policy_id", "period_start", "period_end", "exposure"]].copy()
    out["year"] = out["period_start"].dt.year.astype(int)
    out = out.join(agg, on="row_id").join(pc, on="row_id").join(pn, on="row_id").join(pcc, on="row_id")
    num_cols = [c_ for c_ in out.columns if c_.startswith(("y", "n_"))]
    out[num_cols] = out[num_cols].fillna(0.0)

    min_comp = float((dev_inputs.settings.get("row") or {}).get("min_value", 1e-6))
    comp = dev.row_completeness(rows, dev_inputs.pattern_cost, line, cfg.valuation_date,
                                min_value=min_comp)
    comp = comp.set_index("row_id").reindex(out["row_id"]).reset_index(drop=True)
    for p in perils:
        out[f"comp_{p}"] = comp[f"comp_{p}"].to_numpy()
    out["comp_line"] = dev.line_completeness(
        out[[f"comp_{p}" for p in perils]], dev_inputs.peril_shares[line]).to_numpy()
    if dev_inputs.pattern_count is not None:  # count completeness for frequency experts
        compn = dev.row_completeness(rows, dev_inputs.pattern_count, line, cfg.valuation_date,
                                     min_value=min_comp)
        compn = compn.set_index("row_id").reindex(out["row_id"]).reset_index(drop=True)
        for p in perils:
            out[f"compn_{p}"] = compn[f"comp_{p}"].to_numpy()
    out["freq_trend"] = trend.row_frequency_factor(out["year"], target_year)
    out["log_offset"] = np.log(out["exposure"] * out["comp_line"] * out["freq_trend"])

    # Large-loss load from mature years (developed basis).
    by_year = out.groupby("year").agg(y=("y", "sum"), yc=("y_capped", "sum"),
                                      comp=("comp_line", "mean"))
    cand = by_year if fit_years is None else by_year.loc[by_year.index.isin(fit_years)]
    load_years = cand.index[cand["comp"] >= 0.95].tolist() or cand.index.tolist()
    sel = by_year.loc[by_year.index.isin(load_years)]
    load = float(sel["y"].sum() / sel["yc"].sum()) if sel["yc"].sum() > 0 else 1.0

    summary = {
        "line": line, "target_year": int(target_year), "cap": cap, "cap_quantile": cap_quantile,
        "large_loss_load": load, "load_years": [int(y) for y in load_years],
        "n_rows": int(len(out)), "n_claims_matched": int(len(m)),
        "y_total": float(out["y"].sum()), "y_capped_total": float(out["y_capped"].sum()),
        "freq_trend_rate": trend.frequency,
        "peril_loads": {p: float(out.loc[out["year"].isin(load_years), f"y_{p}"].sum()
                                 / max(out.loc[out["year"].isin(load_years), f"yc_{p}"].sum(), 1e-9))
                        for p in perils},
        "mean_excess": float(m.loc[(m["is_large"] > 0) & fit_mask, "excess"].mean())
        if ((m["is_large"] > 0) & fit_mask).any() else 0.0,
        "n_large_fit": int(((m["is_large"] > 0) & fit_mask).sum()),
    }
    log.info("%s targets @%d: cap=%.0f load=%.4f rows=%d", line, target_year, cap, load, len(out))
    return out, summary


def developed_experience(targets: pd.DataFrame) -> pd.DataFrame:
    """Year-level developed experience at the target-year cost level (A/E and guardrails).

    ``y`` is already on-levelled for severity; frequency trend and IBNR live in the
    offset, so the ratio estimator divides by both (independent review V10):
    ``developed_cost = sum(y) * sum(exposure) / sum(exposure * comp_line * freq_trend)``.
    ``comp`` is the pure IBNR completeness (no trend) for reporting.
    """
    ft = targets["freq_trend"] if "freq_trend" in targets.columns else 1.0
    t = targets.assign(ec=targets["exposure"] * targets["comp_line"],
                       ect=targets["exposure"] * targets["comp_line"] * ft)
    g = t.groupby("year")
    out = g.agg(exposure=("exposure", "sum"), ec=("ec", "sum"), ect=("ect", "sum"), y=("y", "sum"),
                n_claims=("n_claims", "sum"))
    out["comp"] = out["ec"] / out["exposure"]
    out["freq_trend"] = out["ect"] / out["ec"]
    out["developed_cost"] = out["y"] * out["exposure"] / out["ect"]
    out["developed_pp"] = out["developed_cost"] / out["exposure"]
    return out.drop(columns=["ec", "ect"]).reset_index()


def summarise_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Compact, JSON-friendly view of the development parameters for the run report."""
    keep = ("recovery_ratio", "repudiation_prob", "ibner_factor", "suspect_corrections")
    return {k: params.get(k) for k in keep if k in params}


def claim_level_frame(
    line: str,
    rows: pd.DataFrame,
    dev_inputs: DevelopmentInputs,
    cfg: Config,
    target_year: int,
) -> pd.DataFrame:
    """One record per matched, non-repudiated claim with its on-levelled cost at ``target_year``.

    Used by the severity experts (two-part frequency x severity models). Columns:
    ``claim_id, row_id, peril, loss_year, report_delay_days, cost_T`` (developed for open
    claims via ``net_ultimate``, trended, BI level shift applied).
    """
    trend = TrendAssumptions.from_config(cfg, line)
    c = dev_inputs.claims_ult
    c = c.loc[(c["line"] == line) & ~c["is_repudiated"].astype(bool)]
    m = match_claims_to_rows(c, rows)
    m = m.loc[m["match_status"] == "matched"].copy()
    m["cost_T"] = m["net_ultimate"].to_numpy(dtype=float) * trend.claim_factor(
        m["peril"], m["loss_year"], target_year)
    m = m.loc[m["cost_T"] > 0]
    return m[["claim_id", "row_id", "peril", "loss_year", "report_delay_days", "cost_T"]].reset_index(drop=True)
