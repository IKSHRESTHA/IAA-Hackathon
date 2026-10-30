"""Run prediction-time availability and claims-history leakage checks.

Usage:
    python final/leakage_audit.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
R2 = REPO / "2nd round"
DATA = R2 / "data"
LINES = ("motor", "buildings", "contents")
CUTOFF = pd.Timestamp("2025-01-01")
HISTORY_YEARS = 3
TOLERANCE = 1e-5
# Decided after the period or driven by its claims, so never available when pricing 2025.
FORBIDDEN = {"renewal_decision", "customer_service_contacts"}


def read(name: str, layer: str, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(DATA / layer / f"{name}.parquet", columns=columns)


def policy_rows() -> pd.DataFrame:
    parts = []
    columns = ["row_id", "policy_id", "customer_id", "address_id", "period_start", "period_end"]
    for line in LINES:
        for prefix in ("policies", "test"):
            table = read(f"{prefix}_{line}", "clean", columns=columns)
            parts.append(table.assign(line=line, year=table["period_start"].dt.year))
    return pd.concat(parts, ignore_index=True)


def assert_close(name: str, actual: pd.Series, expected: pd.Series) -> None:
    if pd.api.types.is_float_dtype(actual):
        equal = np.isclose(actual.fillna(0), expected.fillna(0), atol=TOLERANCE, rtol=TOLERANCE)
        equal &= actual.isna().to_numpy() == expected.isna().to_numpy()
    else:
        equal = actual.fillna(0).to_numpy() == expected.fillna(0).to_numpy()
    if not bool(np.all(equal)):
        bad = int((~equal).sum())
        raise AssertionError(f"{name}: {bad:,} rows differ from an as-of recomputation")


def check_history(line: str, rows: pd.DataFrame, claims: pd.DataFrame) -> list[str]:
    features = read(f"test_{line}", "features")
    test_rows = rows.loc[rows["line"] == line].loc[
        rows.loc[rows["line"] == line, "year"].eq(2025)
    ].copy()
    test_rows = test_rows.set_index("row_id").loc[features["row_id"].astype(str)].reset_index()
    ev = claims.loc[
        (claims["match_status"] == "matched")
        & (claims["status"] != "repudiated")
        & claims["loss_date"].lt(CUTOFF)
        & claims["report_date"].lt(CUTOFF)
    ].copy()
    ev = ev.merge(rows[["row_id", "customer_id"]], on="row_id", how="left", validate="many_to_one")
    ev["cost"] = ev["net_incurred"].clip(lower=0)
    caps = ev.groupby("line")["cost"].transform("quantile", 0.995)
    ev["cost"] = ev["cost"].clip(upper=caps)
    start = pd.Timestamp("2022-01-01")
    ev = ev.loc[ev["loss_date"].ge(start)]
    expected: dict[str, pd.Series] = {}
    own = ev.loc[ev["line"] == line]
    expected["hist_pol_claims_1y"] = (
        test_rows["policy_id"].map(own.loc[own["loss_date"].ge(pd.Timestamp("2024-01-01"))]
        .groupby("policy_id").size()).fillna(0)
    )
    expected["hist_pol_claims_3y"] = test_rows["policy_id"].map(own.groupby("policy_id").size()).fillna(0)
    expected["hist_pol_cost_3y"] = test_rows["policy_id"].map(own.groupby("policy_id")["cost"].sum()).fillna(0)
    expected["hist_cust_claims_3y"] = test_rows["customer_id"].map(ev.groupby("customer_id").size()).fillna(0)
    expected["hist_cust_claims_other_lines_3y"] = test_rows["customer_id"].map(
        ev.loc[ev["line"] != line].groupby("customer_id").size()
    ).fillna(0)
    latest = own.groupby("policy_id")["loss_date"].max()
    expected["hist_pol_years_since_claim"] = (
        test_rows["policy_id"].map(latest).rsub(CUTOFF).dt.days / 365.25
    )
    expected["hist_years_available"] = pd.Series(2.0, index=test_rows.index)
    for column, values in expected.items():
        if column in features.columns:
            assert_close(f"{line}.{column}", features[column], values)
    return [f"{line}: checked {sum(c in features.columns for c in expected)} as-of history columns"]


def main() -> int:
    claims = read("claims", "clean")
    rows = policy_rows()
    messages = []
    if claims["claim_id"].duplicated().any():
        raise AssertionError("claims: duplicate claim_id values")
    if (claims["report_date"] < claims["loss_date"]).any():
        raise AssertionError("claims: report_date precedes loss_date")
    if rows["row_id"].duplicated().any():
        raise AssertionError("policy rows: duplicate row_id values")
    if not claims["match_status"].isin({"matched", "before_first_row", "after_last_row", "unknown_policy"}).all():
        raise AssertionError("claims: unknown match_status value")
    for line in LINES:
        train = read(f"policies_{line}", "features")
        test = read(f"test_{line}", "features")
        target_columns = {"n_claims", "claim_cost", "n_open", "comp_cost", "comp_count"}
        leaked = target_columns.intersection(test.columns)
        if leaked:
            raise AssertionError(f"{line}: target columns present in test features: {sorted(leaked)}")
        forbidden = FORBIDDEN.intersection(train.columns) | FORBIDDEN.intersection(test.columns)
        if forbidden:
            raise AssertionError(f"{line}: leakage columns present in features: {sorted(forbidden)}")
        for column in ("comp_cost", "comp_count"):
            if column in train and not train[column].between(0, 1).all():
                raise AssertionError(f"{line}.{column}: completeness outside [0, 1]")
        history = check_history(line, rows, claims)
        if history[0].split(": checked ")[1].startswith("0 "):
            print(f"warning: {history[0]}")
        messages.extend(history)
    result = {"status": "passed", "checks": messages}
    out = REPO / "final" / "3_diagnostics" / "leakage_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
