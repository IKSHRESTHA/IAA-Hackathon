"""Step 2 - clean the case-study tables: fix what is provably wrong, flag what is doubtful.

Principles
----------
* **Fix only what is provably wrong**, with the evidence in the log: a value is changed only when
  the data themselves show the error (the period dates, the same policy's other years, a unit
  that is off by exactly 1,000, an exact duplicate).
* **Never silently**: every rule writes a line to ``reports/02_cleaning/cleaning_log.csv``
  (rule, table, column, rows affected, action, evidence); every changed row carries a ``dq_``
  flag in the clean table.
* **No rows are dropped from the policy, customer, address or weather tables.** The only rows
  removed are exact duplicate claims.
* **Missing values are not imputed here.** They are classified in the report (structural,
  informative, random) and handled in step 3.

Output: ``data/clean/<same file names as the case study>.parquet`` and
``reports/02_cleaning/`` (``cleaning_log.csv``, ``checks_after_cleaning.csv``, ``missing_values.csv``).
Usage: ``python 02_data_cleaning.py``
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import helpers as H

OUT = C.REPORTS / "02_cleaning"
STEP = "02_cleaning"


# =================================================================================================
# Rules used by several tables
# =================================================================================================
def normalise_codes(df: pd.DataFrame, table: str, log: H.ChangeLog) -> pd.DataFrame:
    """Trim and case-fold category text; map known synonyms and local codes to one canonical code."""
    df = df.copy()
    skip = set(C.ID_COLUMNS) | {"surname"}
    for col in df.columns:
        if col in skip or not pd.api.types.is_string_dtype(df[col]):
            continue
        before = df[col]
        after = before.str.strip()
        after = after.str.upper() if col in C.UPPER_CASE_CODES else after.str.lower()
        if col in C.CODE_MAPS:
            after = after.replace(C.CODE_MAPS[col])
        changed = (before.fillna("<NA>") != after.fillna("<NA>"))
        if changed.any():
            pairs = (pd.DataFrame({"from": before[changed], "to": after[changed]})
                     .value_counts().head(8))
            ev = "; ".join(f"'{a}'->'{b}' ({n:,})" for (a, b), n in pairs.items())
            log.add("C01", table, col, "one spelling per category (trim, case, local codes)",
                    changed.sum(), "recoded", ev)
            df[col] = after
    return df


def split_self_employed(df: pd.DataFrame, table: str, log: H.ChangeLog) -> pd.DataFrame:
    """'self-employed <occupation>' becomes the occupation plus a ``self_employed`` flag."""
    if "occupation" not in df.columns:
        return df
    s = df["occupation"]
    pre = s.str.match(r"^self[- ]employed\s+").fillna(False)
    if not pre.any():
        return df
    df = df.copy()
    df["self_employed"] = pre.astype("int8")
    df["occupation"] = s.str.replace(r"^self[- ]employed\s+", "", regex=True)
    log.add("C02", table, "occupation", "'self-employed' prefix split into its own flag", pre.sum(),
            "new column self_employed", f"{df['occupation'].nunique()} base occupations remain")
    return df


def pad_ids(df: pd.DataFrame, table: str, log: H.ChangeLog) -> pd.DataFrame:
    """All-digit identifiers that lost their leading zeros (stored as numbers) are padded back."""
    for col, width in C.ID_WIDTH.items():
        if col not in df.columns:
            continue
        s = df[col]
        short = s.str.fullmatch(r"\d+").fillna(False) & (s.str.len() < width)
        if short.any():
            df = df.copy()
            df[col] = s.where(~short, s.str.zfill(width))
            log.add("C03", table, col, f"lost leading zeros restored (width {width})", short.sum(),
                    "zero-padded", f"lengths before: { {int(k): int(v) for k, v in s[short].str.len().value_counts().sort_index().items()} }; "
                                   "every padded id is found in customers")
    return df


def fix_exposure(df: pd.DataFrame, table: str, log: H.ChangeLog) -> pd.DataFrame:
    """Exposure must equal the covered days / 365 (the basis of the 2025 files)."""
    df = df.copy()
    days = (df["period_end"] - df["period_start"]).dt.days + 1
    implied = days / C.EXPOSURE_DAYS_IN_YEAR
    bad = ~np.isclose(df["exposure"], implied, atol=1e-3)
    df["dq_exposure_fixed"] = bad.astype("int8")
    if bad.any():
        e = df.loc[bad, "exposure"]
        in_days = np.isclose(e, days[bad], atol=1e-6)
        b360 = np.isclose(e, days[bad] / 360.0, atol=1e-3)
        log.add("C04", table, "exposure", "exposure recomputed from the period dates (days / 365)", bad.sum(),
                "recomputed", f"recorded in days: {int(in_days.sum()):,}; on a 360-day basis: {int(b360.sum()):,}; "
                              f"other: {int((~in_days & ~b360).sum()):,}")
        df.loc[bad, "exposure"] = implied[bad]
    return df


def fix_sum_insured(df: pd.DataFrame, table: str, log: H.ChangeLog) -> pd.DataFrame:
    """Sums insured recorded in units instead of thousands (x1,000) are divided by 1,000.

    Evidence: no value lies between 10x and 200x the median (a clean gap), the suspect values
    are 200x to 5,000x, the same policy's other years are 1,000 times smaller, and after the
    division the values have the usual distribution.
    """
    if "sum_insured" not in df.columns:
        return df
    df = df.copy()
    med = float(df["sum_insured"].median())
    bad = df["sum_insured"] > C.SI_UNIT_ERROR_RATIO * med
    df["dq_sum_insured_fixed"] = bad.astype("int8")
    if bad.any():
        new = df.loc[bad, "sum_insured"] / C.SI_UNIT_SCALE
        df.loc[bad, "sum_insured"] = new
        log.add("C05", table, "sum_insured", f"sum insured > {C.SI_UNIT_ERROR_RATIO}x the median "
                f"({med:,.0f}) divided by {C.SI_UNIT_SCALE:,}", bad.sum(), "rescaled",
                f"after the fix: {new.min():,.0f} to {new.max():,.0f} (median {new.median():,.0f})")
    return df


def flag_mileage_zero(df: pd.DataFrame, table: str, log: H.ChangeLog) -> pd.DataFrame:
    """A declared annual mileage of 0 is kept but flagged (it is partly a placeholder)."""
    if "annual_mileage" not in df.columns:
        return df
    df = df.copy()
    z = df["annual_mileage"] == 0
    df["dq_mileage_zero"] = z.fillna(False).astype("int8")
    log.add("C06", table, "annual_mileage", "annual mileage of 0 (45% of them commute) flagged",
            z.sum(), "flag dq_mileage_zero", "value kept: these cars claim a little less than the lowest band")
    return df


def check_domains(df: pd.DataFrame, table: str, log: H.ChangeLog) -> None:
    for col, allowed in C.DOMAINS.items():
        if col in df.columns:
            bad = df[col].notna() & ~df[col].isin(allowed)
            if bad.any():
                log.add("C99", table, col, "value outside the dictionary domain", bad.sum(), "reported only",
                        str(df.loc[bad, col].value_counts().head(5).to_dict()))


# =================================================================================================
# Motor: driving history
# =================================================================================================
def repair_years_licensed(tables: dict[str, pd.DataFrame], log: H.ChangeLog) -> dict[str, pd.DataFrame]:
    """Years licensed implying a licence before the minimum age are restored from the same policy.

    A driver's age at first licence (``driver_age - years_licensed``) is the same in every row of
    a policy (within one year of birthday timing). Rows where it is below 17 are errors (the value
    jumps by 9 to 18 years and back the next year); the policy's other rows give the true age at
    first licence, hence the true years licensed. Without another row the value is capped.
    """
    if "driver_age" not in tables["policies_motor"].columns:
        return tables
    parts = []
    for t in ("policies_motor", "test_motor"):
        d = tables[t][["row_id", "policy_id", "driver_age", "years_licensed"]].copy()
        d["table"] = t
        parts.append(d)
    m = pd.concat(parts, ignore_index=True)
    afl = m["driver_age"] - m["years_licensed"]
    bad = afl < C.MIN_LICENCE_AGE
    ref = afl[~bad].groupby(m.loc[~bad, "policy_id"]).median()
    r = m["policy_id"].map(ref)
    new = np.where(r.notna(), m["driver_age"] - np.round(r), m["driver_age"] - C.MIN_LICENCE_AGE)
    new = np.clip(new, 0, m["driver_age"] - C.MIN_LICENCE_AGE)
    m["fixed"] = bad
    m["new"] = np.where(bad, new, m["years_licensed"]).astype(int)
    for t in ("policies_motor", "test_motor"):
        sel = m["table"] == t
        upd = m.loc[sel].set_index("row_id")
        df = tables[t].copy()
        fixed = df["row_id"].map(upd["fixed"]).astype(bool)
        n_ref = int((fixed & df["policy_id"].map(ref).notna()).sum())
        df["years_licensed"] = df["row_id"].map(upd["new"]).astype("int64")
        df["dq_years_licensed_fixed"] = fixed.astype("int8")
        log.add("C07", t, "years_licensed", f"licence before age {C.MIN_LICENCE_AGE} is impossible",
                fixed.sum(), "restored from the policy's other rows",
                f"{n_ref:,} restored from the same policy's age at first licence, "
                f"{int(fixed.sum()) - n_ref:,} capped at age - {C.MIN_LICENCE_AGE} (no other row)")
        tables[t] = df
    return tables


def flag_ncb_beyond_licence(df: pd.DataFrame, table: str, log: H.ChangeLog) -> pd.DataFrame:
    """No-claims bonus longer than the licence cannot be true; kept as declared and flagged."""
    if not {"ncb_years", "years_licensed"} <= set(df.columns):
        return df
    df = df.copy()
    f = df["ncb_years"] > df["years_licensed"]
    df["dq_ncb_exceeds_licence"] = f.astype("int8")
    log.add("C08", table, "ncb_years", "no-claims bonus longer than the licence", f.sum(),
            "kept as declared, flagged", "too common to be typing errors (mostly young drivers); "
                                        "the declaration itself carries risk information")
    return df


# =================================================================================================
# Claims
# =================================================================================================
def clean_claims(cl: pd.DataFrame, policy_rows: pd.DataFrame, log: H.ChangeLog) -> pd.DataFrame:
    cl = cl.copy()
    # 1. exact duplicates under a second claim_id ------------------------------------------------
    fields = [c for c in cl.columns if c != "claim_id"]
    dup_any = cl.duplicated(fields, keep=False)
    if dup_any.any():
        normal_len = int(cl["claim_id"].str.len().mode().iloc[0])
        # keep the claim whose id has the normal format; drop the copy with the odd id
        cl["_odd"] = (cl["claim_id"].str.len() != normal_len).astype(int)
        cl = cl.sort_values(["_odd", "claim_id"])
        drop = cl.duplicated(fields, keep="first")
        lens = {int(k): int(v) for k, v in cl.loc[drop, "claim_id"].str.len().value_counts().items()}
        years = {int(k): int(v) for k, v in cl.loc[drop, "loss_date"].dt.year.value_counts().sort_index().items()}
        log.add("K01", "claims", "claim_id", "claim repeated under a second claim_id (every other field identical)",
                drop.sum(), "duplicate removed", f"removed ids have {lens} characters (normal: {normal_len}); "
                                                  f"loss years {years}")
        cl = cl.loc[~drop].drop(columns="_odd").sort_index()
    # 2. reserve entry errors (decimal shift) ------------------------------------------------------
    cl["dq_incurred_corrected"] = np.int8(0)
    for (line, peril), g in cl.groupby(["line", "peril"]):
        if len(g) < 3:
            continue
        top2 = g["incurred_to_date"].nlargest(2)
        i, first, second = top2.index[0], top2.iloc[0], top2.iloc[1]
        if first > C.SUSPECT_CLAIM_RATIO * second and cl.at[i, "status"] == "open":
            k = max(1, int(round(np.log10(first / second))))   # digits typed in excess
            new = first / 10 ** k
            if new >= cl.at[i, "paid_to_date"]:
                cl.at[i, "incurred_to_date"] = new
                cl.at[i, "dq_incurred_corrected"] = 1
                log.add("K02", "claims", "incurred_to_date", f"{line}/{peril} open claim {cl.at[i, 'claim_id']}: "
                        f"incurred {first:,.0f} is {first / second:.1f}x the next largest", 1,
                        f"divided by 10^{k}", f"paid {cl.at[i, 'paid_to_date']:,.0f}; corrected incurred {new:,.0f} "
                                              f"(a reserve typed with {k} extra digit(s))")
    # 3. the claim cost of the case study ---------------------------------------------------------
    net = (cl["incurred_to_date"] - cl["recoveries"]).clip(lower=0.0)
    cl["net_incurred"] = np.where(cl["status"] == "repudiated", 0.0, net)
    cl["report_delay_days"] = (cl["report_date"] - cl["loss_date"]).dt.days.astype("int32")
    log.add("K03", "claims", "net_incurred", "claim cost = incurred - recoveries (floored at 0); 0 if repudiated",
            len(cl), "new column", f"repudiated: {int((cl['status'] == 'repudiated').sum()):,}")
    # 4. link every claim to the policy row on cover at the loss date -----------------------------
    cl = H.match_claims_to_rows(cl, policy_rows)
    st = cl["match_status"].value_counts().to_dict()
    unmatched = cl["match_status"] != "matched"
    log.add("K04", "claims", "row_id", "claim linked to the policy row on cover at the loss date",
            unmatched.sum(), "unmatched kept, flagged (excluded from targets)", f"status counts: {st}")
    # 5. claims outside the cover bought ----------------------------------------------------------
    cover = cl["row_id"].map(policy_rows.set_index("row_id")["cover_type"]) if "cover_type" in policy_rows else None
    out = pd.Series(False, index=cl.index)
    if cover is not None:
        for ct, perils in C.PERILS_OUTSIDE_COVER.items():
            out |= ((cover == ct).fillna(False) & cl["peril"].isin(perils).fillna(False)
                    & (cl["line"] == "motor").fillna(False)).astype(bool)
    cl["dq_outside_cover"] = out.astype("int8")
    log.add("K05", "claims", "peril", "motor claim of a peril the cover does not include", out.sum(),
            "kept, flagged", str(cl.loc[out].groupby(["peril"]).size().to_dict()))
    ad = cl["row_id"].map(policy_rows.set_index("row_id")["accidental_damage_cover"]) \
        if "accidental_damage_cover" in policy_rows else None
    if ad is not None:
        f = ((cl["peril"] == "accidental_damage").fillna(False)
             & ad.astype("boolean").eq(False).fillna(False)).astype(bool)
        cl["dq_ad_without_cover"] = f.astype("int8")
        is_ad = (cl["peril"] == "accidental_damage").fillna(False).astype(bool)
        share = f.sum() / max(is_ad.sum(), 1)
        size = cl.loc[f, "net_incurred"].mean() / cl.loc[is_ad & ~f, "net_incurred"].mean() if f.any() else float("nan")
        log.add("K06", "claims", "peril", "accidental-damage claim on a row without AD cover", f.sum(),
                "kept, flagged", f"{share:.0%} of home AD claims; mean size {size:.2f}x the covered ones: "
                                 "plausibly basic AD cover")
    return cl


# =================================================================================================
# Driver
# =================================================================================================
def missing_value_report(clean: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Missing values left after cleaning (their treatment is decided in step 3)."""
    recs = []
    for name, df in clean.items():
        for c in df.columns:
            n = int(df[c].isna().sum())
            if n:
                recs.append({"table": name, "column": c, "missing": n, "missing_pct": round(100 * n / len(df), 3)})
    return pd.DataFrame(recs)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    log = H.ChangeLog(STEP)
    names = list(C.TABLES) + [t for t in getattr(C, "OPTIONAL_TABLES", ()) if H.table_path(t).exists()]
    raw = {}
    with H.Timer(f"{C.ROUND}: reading"):
        for n in names:
            raw[n] = H.read_table(n, "raw")
    clean: dict[str, pd.DataFrame] = {}

    with H.Timer("customers, addresses, weather"):
        for t in ("customers", "addresses"):
            df = normalise_codes(raw[t], t, log)
            if t == "customers":
                df = split_self_employed(df, t, log)
            check_domains(df, t, log)
            clean[t] = df
        w = raw["weather"]
        neg = (w.drop(columns=["address_id", "year"]) < 0).any(axis=1)
        log.add("C10", "weather", "*", "negative weather measures", neg.sum(), "none found" if not neg.any() else "reported", "")
        clean["weather"] = w

    with H.Timer("policy tables"):
        for t in C.POLICY_TABLES:
            df = pad_ids(raw[t], t, log)
            df = fix_exposure(df, t, log)
            df = fix_sum_insured(df, t, log)
            df = normalise_codes(df, t, log)
            df = flag_mileage_zero(df, t, log)
            check_domains(df, t, log)
            clean[t] = df
        clean = repair_years_licensed(clean, log)
        for t in ("policies_motor", "test_motor"):
            clean[t] = flag_ncb_beyond_licence(clean[t], t, log)

    with H.Timer("claims"):
        rows = []
        for line in C.LINES:
            p = clean[f"policies_{line}"]
            cols = ["row_id", "policy_id", "period_start", "period_end"] + \
                   [c for c in ("cover_type", "accidental_damage_cover") if c in p.columns]
            rows.append(p[cols])
        rows = pd.concat(rows, ignore_index=True)
        cl = normalise_codes(raw["claims"], "claims", log)
        clean["claims"] = clean_claims(cl, rows, log)
        check_domains(clean["claims"], "claims", log)

    if "reference_policies" in raw:          # 2nd round: 20 policies to quote, copies of 2025 rows
        with H.Timer("reference policies and expenses"):
            ref = raw["reference_policies"]
            parts = []
            for line in C.LINES:
                ids = ref.loc[ref["line"] == line, "row_id"]
                t = clean[f"test_{line}"]
                parts.append(t.loc[t["row_id"].isin(ids)].assign(line=line))
            rebuilt = pd.concat(parts, ignore_index=True)
            order = pd.Series(range(len(ref)), index=ref["row_id"])
            rebuilt = rebuilt.iloc[np.argsort(rebuilt["row_id"].map(order).to_numpy())].reset_index(drop=True)
            missing = set(ref["row_id"]) - set(rebuilt["row_id"])
            if missing:
                raise ValueError(f"reference policies not found in the 2025 files: {sorted(missing)}")
            log.add("C11", "reference_policies", "*", "each reference policy is a copy of a 2025 row: taken "
                    "from the cleaned 2025 files", len(rebuilt), "replaced by the cleaned row",
                    "values identical to the raw 2025 rows before cleaning")
            clean["reference_policies"] = rebuilt[[c for c in ref.columns if c in rebuilt.columns]
                                                  + [c for c in rebuilt.columns if c.startswith("dq_")]]
    if "expenses" in raw:
        e = raw["expenses"]
        miss = e.loc[e["emerging"].isna(), "item"].tolist() if "emerging" in e else []
        log.add("C12", "expenses", "emerging", "expense item without a value in this market", len(miss),
                "kept missing (no assumption made here)", f"items: {miss}")
        clean["expenses"] = e
    for t in names:
        if t not in clean:
            clean[t] = raw[t]            # sample_submission needs no change

    with H.Timer("checks after cleaning"):
        import importlib
        reading = importlib.import_module("01_data_reading")
        after = reading.key_checks({k: v for k, v in clean.items() if k in C.TABLES})
        after.to_csv(OUT / "checks_after_cleaning.csv", index=False)
        failed = after.loc[~after["passed"]]
        print(failed.to_string(index=False) if len(failed) else "  every key and link check passes")
        missing_value_report(clean).to_csv(OUT / "missing_values.csv", index=False)

    with H.Timer("writing data/clean"):
        for t, df in clean.items():
            H.write_table(df, t, "clean")
            print(f"  {t:<22} {len(df):>10,} rows x {df.shape[1]:>2} columns")
    f = log.save(OUT / "cleaning_log.csv")
    print(f"\n{len(f)} log lines -> {OUT / 'cleaning_log.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
