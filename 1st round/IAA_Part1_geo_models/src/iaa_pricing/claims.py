"""Claim-level net cost and claim -> policy-row matching.

Target definition (competition): the ultimate cost of all claims with a loss
date inside the row's period, net of the policy excess (already deducted in
``paid_to_date`` / ``incurred_to_date``) and net of recoveries, excluding
repudiated (declined) claims.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import PERILS

log = logging.getLogger(__name__)


def prepare_claims(claims: pd.DataFrame, floor_at_zero: bool = True,
                   suspect_ratio: float = 5.0) -> pd.DataFrame:
    """Add analysis columns to the raw claims table.

    Adds ``net_incurred`` (incurred - recoveries; zero for repudiated claims),
    ``report_delay_days``, ``loss_year`` and status flags. Claims whose incurred
    exceeds ``suspect_ratio`` x the largest *other* claim of the same peril are
    flagged ``suspect_amount`` (e.g. a decimal-shift in a case reserve) for
    human review; their value is left untouched here and handled by the
    large-loss cap in modelling.
    """
    c = claims.copy()
    for col in ("line", "peril", "status"):
        c[col] = c[col].astype("string").str.strip().str.lower()
    c["is_repudiated"] = c["status"].eq("repudiated")
    c["is_open"] = c["status"].eq("open")
    net = c["incurred_to_date"] - c["recoveries"]
    if floor_at_zero:
        net = net.clip(lower=0.0)
    c["net_incurred"] = np.where(c["is_repudiated"], 0.0, net)
    c["report_delay_days"] = (c["report_date"] - c["loss_date"]).dt.days
    c["loss_year"] = c["loss_date"].dt.year
    c["report_year"] = c["report_date"].dt.year

    # Suspect-amount screen: compare each claim with the 2nd largest of its peril.
    second = (
        c.groupby(["line", "peril"])["incurred_to_date"]
        .transform(lambda s: s.nlargest(2).iloc[-1] if len(s) > 1 else np.inf)
    )
    largest = c.groupby(["line", "peril"])["incurred_to_date"].transform("max")
    c["suspect_amount"] = (c["incurred_to_date"] == largest) & (
        c["incurred_to_date"] > suspect_ratio * second
    )
    if c["suspect_amount"].any():
        log.warning("Suspect claim amounts flagged for review: %s",
                    c.loc[c["suspect_amount"], "claim_id"].tolist())
    return c


def match_claims_to_rows(claims: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Attach the policy row whose period contains the claim's loss date.

    A policy keeps its ``policy_id`` across years and mid-term changes, so the
    join is ``policy_id`` equality plus ``period_start <= loss_date <= period_end``
    (both ends inclusive). If periods overlap on a boundary day the later row
    wins (the post-change risk was on cover). Returns the claims with
    ``row_id`` and ``match_status`` in {matched, no_row_for_date, unknown_policy}.
    """
    keys = rows[["row_id", "policy_id", "period_start", "period_end"]]
    cand = claims[["claim_id", "policy_id", "loss_date"]].merge(keys, on="policy_id", how="inner")
    inside = (cand["loss_date"] >= cand["period_start"]) & (cand["loss_date"] <= cand["period_end"])
    cand = cand.loc[inside]
    n_multi = int(cand["claim_id"].duplicated().sum())
    if n_multi:
        log.warning("%d claims fall in overlapping rows; later row kept", n_multi)
    cand = cand.sort_values(["claim_id", "period_start"]).drop_duplicates("claim_id", keep="last")

    out = claims.merge(cand[["claim_id", "row_id"]], on="claim_id", how="left")
    known = out["policy_id"].isin(set(rows["policy_id"]))
    out["match_status"] = np.select(
        [out["row_id"].notna(), known], ["matched", "no_row_for_date"], default="unknown_policy"
    )
    return out


def aggregate_to_rows(matched: pd.DataFrame, rows: pd.DataFrame, line: str,
                      cost_col: str = "net_incurred") -> pd.DataFrame:
    """Row-level claim counts and costs, in total and by peril.

    Only non-repudiated, matched claims count. Returns one record per row in
    ``rows`` (zeros where no claims), indexed like ``rows['row_id']``.
    """
    m = matched.loc[(matched["match_status"] == "matched") & ~matched["is_repudiated"]]
    perils = PERILS[line]
    agg = m.groupby("row_id").agg(
        n_claims=("claim_id", "size"),
        cost=(cost_col, "sum"),
        n_open=("is_open", "sum"),
    )
    by_peril_cost = m.pivot_table(index="row_id", columns="peril", values=cost_col,
                                  aggfunc="sum", fill_value=0.0)
    by_peril_n = m.pivot_table(index="row_id", columns="peril", values="claim_id",
                               aggfunc="size", fill_value=0)
    by_peril_cost = by_peril_cost.reindex(columns=list(perils), fill_value=0.0)
    by_peril_n = by_peril_n.reindex(columns=list(perils), fill_value=0)
    by_peril_cost.columns = [f"cost_{p}" for p in perils]
    by_peril_n.columns = [f"n_{p}" for p in perils]

    out = rows[["row_id"]].set_index("row_id")
    out = out.join(agg).join(by_peril_cost).join(by_peril_n).fillna(0.0)
    count_cols = ["n_claims", "n_open"] + list(by_peril_n.columns)
    out[count_cols] = out[count_cols].astype("int32")
    return out.reset_index()


def matching_summary(matched: pd.DataFrame) -> pd.DataFrame:
    """Counts of match outcomes by line and loss year (for the data-quality log)."""
    return (
        matched.groupby(["line", "loss_year", "match_status"], observed=True)
        .size().unstack("match_status", fill_value=0).reset_index()
    )
