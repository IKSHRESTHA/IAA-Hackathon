
from __future__ import annotations

import sys

import pandas as pd

import config as C
import helpers as H

OUT = C.REPORTS / "01_reading"


def read_all() -> dict[str, pd.DataFrame]:
    """All case-study tables, typed (``config.TABLES`` plus any optional table present)."""
    names = list(C.TABLES) + [t for t in getattr(C, "OPTIONAL_TABLES", ()) if H.table_path(t).exists()]
    data = {}
    for name in names:
        data[name] = H.read_table(name, "raw")
        print(f"  {name:<22} {len(data[name]):>10,} rows x {data[name].shape[1]:>2} columns")
    return data


def schema_check(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Columns expected by the data dictionary vs columns present, with their types."""
    recs = []
    for name, df in data.items():
        expected = {c: t for c, t, _ in C.DICTIONARY.get(name, [])}
        for c in df.columns:
            recs.append({"table": name, "column": c, "in_dictionary": c in expected,
                         "dictionary_type": expected.get(c), "dtype_read": str(df[c].dtype),
                         "status": "ok" if c in expected else "extra column"})
        for c, t in expected.items():
            if c not in df.columns:
                recs.append({"table": name, "column": c, "in_dictionary": True, "dictionary_type": t,
                             "dtype_read": None, "status": "MISSING"})
    return pd.DataFrame(recs)


def _check(recs: list, name: str, check: str, n_fail: int, n_total: int, detail: str = "") -> None:
    recs.append({"table": name, "check": check, "failures": int(n_fail), "checked": int(n_total),
                 "passed": n_fail == 0, "detail": detail})


def key_checks(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Unique keys and links between tables (on the data as delivered)."""
    recs: list[dict] = []
    for name, keys in C.PRIMARY_KEYS.items():
        if name in data:
            df = data[name]
            _check(recs, name, f"unique key ({', '.join(keys)})", df.duplicated(keys).sum(), len(df))
    policy_tables = [t for t in C.POLICY_TABLES if t in data]
    rows = pd.concat([data[t][["row_id", "policy_id", "customer_id", "address_id", "period_start",
                               "period_end"]].assign(table=t) for t in policy_tables], ignore_index=True)
    _check(recs, "all policy tables", "row_id unique across all policy files",
           rows["row_id"].duplicated().sum(), len(rows))
    cust = set(data["customers"]["customer_id"])
    addr = set(data["addresses"]["address_id"])
    width = C.ID_WIDTH.get("customer_id")
    for t in policy_tables:
        r = rows.loc[rows["table"] == t]
        miss = ~r["customer_id"].isin(cust)
        short = r["customer_id"].str.len() < width if width else pd.Series(False, index=r.index)
        _check(recs, t, "customer_id found in customers", miss.sum(), len(r),
               f"{int((miss & short).sum()):,} of the misses are shorter than {width} digits" if miss.any() else "")
        _check(recs, t, "address_id found in addresses", (~r["address_id"].isin(addr)).sum(), len(r))
        _check(recs, t, "period_end >= period_start", (r["period_end"] < r["period_start"]).sum(), len(r))
        _check(recs, t, "period within one calendar year",
               (r["period_start"].dt.year != r["period_end"].dt.year).sum(), len(r))
        yrs = r["period_start"].dt.year
        allowed = (C.TEST_YEAR,) if t.startswith("test_") else C.TRAIN_YEARS
        _check(recs, t, f"year in {allowed}", (~yrs.isin(allowed)).sum(), len(r))
        prefix = C.LINE_PREFIX[t.split("_", 1)[1]]
        _check(recs, t, f"row_id and policy_id start with '{prefix}'",
               (~r["row_id"].str.startswith(prefix) | ~r["policy_id"].str.startswith(prefix)).sum(), len(r))
    # policy rows of one policy never overlap
    s = rows.sort_values(["policy_id", "period_start"])
    prev_end = s.groupby("policy_id")["period_end"].shift()
    _check(recs, "all policy tables", "rows of a policy do not overlap", (s["period_start"] <= prev_end).sum(), len(s))
    # claims
    cl = data["claims"]
    _check(recs, "claims", "policy_id found in policy files", (~cl["policy_id"].isin(set(rows["policy_id"]))).sum(), len(cl))
    _check(recs, "claims", "report_date >= loss_date", (cl["report_date"] < cl["loss_date"]).sum(), len(cl))
    _check(recs, "claims", f"report_date <= valuation date {C.VALUATION_DATE}",
           (cl["report_date"] > pd.Timestamp(C.VALUATION_DATE)).sum(), len(cl))
    _check(recs, "claims", "peril belongs to the line",
           (~pd.Series([p in C.PERILS.get(l, ()) for l, p in zip(cl["line"], cl["peril"])], index=cl.index)).sum(), len(cl))
    _check(recs, "claims", "line matches the policy prefix",
           (cl["policy_id"].str[0] != cl["line"].map(C.LINE_PREFIX)).sum(), len(cl))
    _check(recs, "claims", "paid_to_date <= incurred_to_date", (cl["paid_to_date"] > cl["incurred_to_date"] + 0.01).sum(), len(cl))
    _check(recs, "claims", "recoveries <= incurred_to_date", (cl["recoveries"] > cl["incurred_to_date"] + 0.01).sum(), len(cl))
    _check(recs, "claims", "no negative amounts", (cl[["paid_to_date", "incurred_to_date", "recoveries"]] < 0).any(axis=1).sum(), len(cl))
    dup = cl.duplicated([c for c in cl.columns if c != "claim_id"], keep="first")
    _check(recs, "claims", "no claim repeated under another claim_id", dup.sum(), len(cl),
           f"claim_id lengths of the repeats: {({int(k): int(v) for k, v in cl.loc[dup, 'claim_id'].str.len().value_counts().items()})}" if dup.any() else "")
    # weather covers every (address, year) on cover
    w = data["weather"]
    need = rows.assign(year=rows["period_start"].dt.year)[["address_id", "year"]].drop_duplicates()
    have = need.merge(w[["address_id", "year"]], on=["address_id", "year"], how="left", indicator=True)
    _check(recs, "weather", "weather for every address-year on cover", (have["_merge"] != "both").sum(), len(need),
           f"years in weather: {sorted(w['year'].unique().tolist())}")
    # 2025 rows = sample submission
    ss = data["sample_submission"]
    test_ids = pd.concat([data[t]["row_id"] for t in policy_tables if t.startswith("test_")])
    _check(recs, "sample_submission", "same row_ids as the test files",
           len(set(ss["row_id"]) ^ set(test_ids)), len(ss))
    if "reference_policies" in data:
        ref = data["reference_policies"]
        _check(recs, "reference_policies", "every reference policy is a 2025 row",
               (~ref["row_id"].isin(set(test_ids))).sum(), len(ref))
    return pd.DataFrame(recs)


def dictionary_markdown(schema: pd.DataFrame, prof: pd.DataFrame) -> str:
    lines = [f"# Data dictionary - {C.ROUND} ({C.MARKET})", "",
             f"Files of the case study *{C.COMPETITION}*, as read by `01_data_reading.py`.", ""]
    for name in schema["table"].unique():
        p = prof.loc[prof["table"] == name].set_index("column")
        lines += [f"## `{name}`", "", "| Column | Type | Meaning | Missing % | Distinct | Values |", "|---|---|---|---:|---:|---|"]
        meaning = {c: (t, m) for c, t, m in C.DICTIONARY.get(name, [])}
        for c in p.index:
            t, m = meaning.get(c, ("?", "not in the dictionary"))
            r = p.loc[c]
            if pd.notna(r.get("median", None)) and t not in ("id", "category", "boolean"):
                vals = f"{r['min']:g} to {r['max']:g} (median {r['median']:g})" if isinstance(r["min"], float) else ""
            else:
                vals = str(r["top_values"])[:90]
            lines.append(f"| `{c}` | {t} | {m} | {r['missing_pct']:.2f} | {int(r['distinct']):,} | {vals} |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with H.Timer(f"{C.ROUND}: reading the case-study files"):
        data = read_all()
    with H.Timer("schema against the data dictionary"):
        schema = schema_check(data)
        schema.to_csv(OUT / "schema_check.csv", index=False)
        bad = schema.loc[schema["status"] != "ok"]
        print(bad.to_string(index=False) if len(bad) else "  every column matches the dictionary")
    with H.Timer("profile of every column"):
        prof = pd.concat([H.profile(df, name) for name, df in data.items()], ignore_index=True)
        prof.to_csv(OUT / "column_profile.csv", index=False)
        miss = prof.loc[prof["missing_pct"] > 0, ["table", "column", "missing_pct"]]
        print(miss.to_string(index=False))
    with H.Timer("keys and links between tables"):
        keys = key_checks(data)
        keys.to_csv(OUT / "key_checks.csv", index=False)
        print(keys.loc[~keys["passed"]].to_string(index=False) if (~keys["passed"]).any() else "  all key checks pass")
        print(f"  {int(keys['passed'].sum())} of {len(keys)} checks pass")
    (OUT / "data_dictionary.md").write_text(dictionary_markdown(schema, prof), encoding="utf-8")
    print(f"\nReports written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
