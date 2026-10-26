"""Write the three notebooks of each round (they call the round's own scripts and show the evidence).

Usage: python make_notebooks.py            # writes 1st round/*.ipynb and 2nd round/*.ipynb
       python make_notebooks.py --execute  # ... and runs them, saving the outputs inside
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
ROUNDS = ("1st round", "2nd round")


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


SETUP = """
import importlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as C
import helpers as H

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 120)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3})
print(f"{C.ROUND} - {C.MARKET} - {C.COMPETITION}")
"""


# =================================================================================================
def nb_reading(rnd: str):
    cells = [
        md(f"""
# {rnd} - Step 1: reading and understanding the data

This notebook reads every file of the case study (names unchanged), checks it against the data
dictionary, profiles every column and checks the keys and links between tables. **Nothing is
changed here**: the purpose is to know the data before cleaning it. The same code runs as a script:
`python 01_data_reading.py` (reports in `reports/01_reading/`).
"""),
        code(SETUP + '\nreading = importlib.import_module("01_data_reading")'),
        md("## 1. The files\nIdentifiers are read as text, dates as dates, and True/False/None columns as nullable booleans."),
        code("""
data = reading.read_all()
pd.DataFrame([{"table": k, "rows": len(v), "columns": v.shape[1]} for k, v in data.items()])
"""),
        md("## 2. Columns against the data dictionary\nEvery column should be in `config.DICTIONARY` (type and meaning), and every dictionary column in the file."),
        code("""
schema = reading.schema_check(data)
print(schema["status"].value_counts().to_string())
schema.loc[schema["status"] != "ok"]
"""),
        md("## 3. Profile of every column\nMissing values, distinct values, range and the most frequent values. The full table is `reports/01_reading/column_profile.csv`."),
        code("""
prof = pd.concat([H.profile(df, n) for n, df in data.items()], ignore_index=True)
miss = prof.loc[prof["missing_pct"] > 0, ["table", "column", "missing_pct", "distinct"]]
miss = miss.loc[~miss["table"].isin(["reference_policies"])].sort_values("missing_pct")
fig, ax = plt.subplots(figsize=(8, 0.28 * len(miss) + 1))
ax.barh(miss["table"] + "." + miss["column"], miss["missing_pct"], color="#4C72B0")
ax.set_xlabel("% of rows missing"); ax.set_title("Columns with missing values")
plt.tight_layout(); plt.show()
"""),
        code("""
prof.loc[prof["table"].isin(["policies_motor", "policies_buildings", "policies_contents"]),
         ["table", "column", "dtype", "missing_pct", "distinct", "min", "median", "max", "top_values"]]
"""),
        md("## 4. Keys and links between tables\nUnique keys, every row's customer and address, every claim's policy, weather for every address-year on cover, the 2025 rows against the sample submission."),
        code("""
keys = reading.key_checks(data)
print(f"{int(keys['passed'].sum())} of {len(keys)} checks pass")
keys.loc[~keys["passed"]]
"""),
        md("## 5. What will need cleaning\n### 5.1 Exposure against the period dates\nThe 2025 files define exposure as covered days / 365. Training rows that disagree were recorded in days or on a 360-day basis."),
        code("""
recs = []
for t in [t for t in C.POLICY_TABLES if t.startswith("policies_")] + [t for t in C.POLICY_TABLES if t.startswith("test_")]:
    d = data[t]
    days = (d["period_end"] - d["period_start"]).dt.days + 1
    bad = ~np.isclose(d["exposure"], days / 365, atol=1e-3)
    recs.append({"table": t, "rows": len(d), "exposure != days/365": int(bad.sum()),
                 "of which in days": int(np.isclose(d.loc[bad, "exposure"], days[bad]).sum()),
                 "of which days/360": int(np.isclose(d.loc[bad, "exposure"], days[bad] / 360, atol=1e-3).sum())})
pd.DataFrame(recs)
"""),
        md("### 5.2 Sums insured\nOn a log scale the unit errors stand apart: values hundreds to thousands of times the median (recorded in units instead of thousands), with nothing between 10x and 100x. Anything above 100x the median (dashed line) is rescaled in step 2."),
        code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
for ax, line in zip(axes, ["buildings", "contents"]):
    si = pd.concat([data[f"policies_{line}"]["sum_insured"], data[f"test_{line}"]["sum_insured"]])
    r = np.log10(si / si.median())
    ax.hist(r, bins=120, color="#4C72B0")
    ax.set_yscale("log"); ax.set_xlabel("log10(sum insured / median)"); ax.set_title(line)
    ax.axvline(np.log10(C.SI_UNIT_ERROR_RATIO), color="#C44E52", ls="--")
plt.tight_layout(); plt.show()
"""),
        md("### 5.3 Identifiers\nAll-digit customer ids must have 8 digits. Shorter ones lost their leading zeros when stored as numbers."),
        code("""
for t in C.POLICY_TABLES:
    l = data[t]["customer_id"].str.len().value_counts().sort_index()
    print(f"{t:<20} customer_id lengths: {({int(k): int(v) for k, v in l.items()})}")
"""),
        md("### 5.4 Claims\nClaims by loss year and peril; claims stored twice under a second `claim_id`; reporting delays (the slow perils are the ones still largely unreported at 31 December 2024)."),
        code("""
cl = data["claims"]
display(pd.crosstab([cl["line"], cl["peril"]], cl["loss_date"].dt.year))
fields = [c for c in cl.columns if c != "claim_id"]
dup = cl.duplicated(fields, keep=False)
id_len = cl.loc[dup, "claim_id"].str.len().value_counts().sort_index()
print(f"claims identical in every field but the claim_id: {int(dup.sum())} records "
      f"(id lengths: {({int(k): int(v) for k, v in id_len.items()})})")
if dup.any():
    first = cl.loc[dup].sort_values(fields).head(4)
    display(first)
delay = (cl["report_date"] - cl["loss_date"]).dt.days
cl.assign(delay=delay).groupby(["line", "peril"])["delay"].describe(percentiles=[0.5, 0.9, 0.99])[["count", "50%", "90%", "99%", "max"]]
"""),
        md("Largest claim of each peril against the next one: a claim far above the rest with a small paid amount is a reserve typed with an extra digit."),
        code("""
top = (cl.sort_values("incurred_to_date", ascending=False).groupby(["line", "peril"])
       .head(2).groupby(["line", "peril"])["incurred_to_date"].apply(list))
pd.DataFrame({"largest": top.str[0], "second": top.str[1], "ratio": top.str[0] / top.str[1]}).round(1)
"""),
        md("### 5.5 Driver history\nThe age at first licence (driver age - years licensed) cannot be below 17. In the 1st round some rows break this, and the same policy's other years show the true value. The 2nd round has age bands only."),
        code("""
m = data["policies_motor"]
if "driver_age" in m:
    afl = m["driver_age"] - m["years_licensed"]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(afl, bins=range(0, 30), color="#4C72B0"); ax.axvline(C.MIN_LICENCE_AGE, color="#C44E52", ls="--")
    ax.set_yscale("log"); ax.set_xlabel("age at first licence"); plt.tight_layout(); plt.show()
    print(f"rows with a licence before {C.MIN_LICENCE_AGE}: {int((afl < C.MIN_LICENCE_AGE).sum()):,}")
f = m["ncb_years"] > m["years_licensed"]
band = m["driver_age_band"] if "driver_age_band" in m else pd.cut(m["driver_age"], [16, 24, 34, 49, 64, 120])
print("no-claims bonus longer than the licence, share by driver age:")
print(f.groupby(band).mean().round(3).to_string())
"""),
        md("### 5.6 Weather\nOne row per address and year, including 2025. The 2025 values partly lie outside the training years' range."),
        code("""
w = data["weather"]
w.groupby("year")[[c for c in C.WEATHER_VARS]].mean().round(3)
"""),
        md("## 6. Summary\nThe issues found here are fixed in step 2 (`02_data_cleaning`), each with a logged rule; the reports of this step are in `reports/01_reading/` (`schema_check.csv`, `column_profile.csv`, `key_checks.csv`, `data_dictionary.md`)."),
    ]
    return cells


# =================================================================================================
def nb_cleaning(rnd: str):
    cells = [
        md(f"""
# {rnd} - Step 2: cleaning

Rules: **fix only what is provably wrong** (the data themselves show the error), **log every
change** (`reports/02_cleaning/cleaning_log.csv`) and flag every changed row (`dq_` columns),
**never drop** policy, customer, address or weather rows (only exact duplicate claims are removed),
and **do not impute** missing values (step 3 decides how each one is handled).

The script `python 02_data_cleaning.py` writes `data/clean/` (same file names as the case study).
This notebook applies the same functions rule by rule and shows each rule's effect.
"""),
        code(SETUP + '\ncleaning = importlib.import_module("02_data_cleaning")\nraw = {n: H.read_table(n, "raw") for n in C.TABLES}\nlog = H.ChangeLog("notebook")'),
        md("## 1. Identifiers: lost leading zeros"),
        code("""
def lengths(s):
    return {int(k): int(v) for k, v in s.str.len().value_counts().sort_index().items()}

for t in C.POLICY_TABLES:
    before = lengths(raw[t]["customer_id"])
    after = lengths(cleaning.pad_ids(raw[t], t, log)["customer_id"])
    print(f"{t:<20} customer_id lengths before {before} -> after {after}")
"""),
        md("## 2. Exposure recomputed from the dates"),
        code("""
fixed = {t: cleaning.fix_exposure(raw[t], t, log) for t in C.POLICY_TABLES}
t = "policies_motor"
d = fixed[t]
bad = d["dq_exposure_fixed"] == 1
pd.DataFrame({"exposure as recorded": raw[t].loc[bad, "exposure"].head(8).to_numpy(),
              "days on cover": ((d["period_end"] - d["period_start"]).dt.days + 1)[bad].head(8).to_numpy(),
              "exposure fixed": d.loc[bad, "exposure"].head(8).round(4).to_numpy()})
"""),
        md("## 3. Sums insured recorded in units instead of thousands"),
        code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
for ax, line in zip(axes, ["buildings", "contents"]):
    t = f"policies_{line}"
    after = cleaning.fix_sum_insured(raw[t], t, log)
    bins = np.histogram_bin_edges(np.log10(raw[t]["sum_insured"].dropna()), bins=100)  # same bins for both
    ax.hist(np.log10(raw[t]["sum_insured"]), bins=bins, alpha=0.5, label="raw", color="#C44E52")
    ax.hist(np.log10(after["sum_insured"]), bins=bins, alpha=0.6, label="cleaned", color="#4C72B0")
    ax.set_yscale("log"); ax.set_xlabel("log10(sum insured)"); ax.set_title(line); ax.legend()
plt.tight_layout(); plt.show()
"""),
        md("## 4. One spelling per category"),
        code("""
cust = cleaning.normalise_codes(raw["customers"], "customers", log)
cust = cleaning.split_self_employed(cust, "customers", log)
b = cleaning.normalise_codes(raw["policies_buildings"], "policies_buildings", log)
pd.DataFrame({"raw": raw["policies_buildings"]["construction_type"].value_counts(),
              "cleaned": b["construction_type"].value_counts()}).fillna(0).astype(int)
"""),
        code("""
print("occupations:", cust["occupation"].nunique(), "levels after cleaning (raw:", raw["customers"]["occupation"].nunique(), ")")
if "self_employed" in cust:
    print("self-employed customers:", int(cust["self_employed"].sum()))
"""),
        md("## 5. Driver history: impossible years licensed (1st round)\nA driver's age at first licence is the same in every row of a policy. Rows where it falls below 17 take it from the policy's other rows; a policy without another row is capped at age - 17."),
        code("""
tabs = {t: cleaning.pad_ids(raw[t], t, log) for t in ("policies_motor", "test_motor")}
if "driver_age" in tabs["policies_motor"]:
    rep = cleaning.repair_years_licensed(dict(tabs), log)
    m0, m1 = tabs["policies_motor"], rep["policies_motor"]
    ex = m1.loc[m1["dq_years_licensed_fixed"] == 1, "policy_id"].iloc[0]
    view = m0.loc[m0["policy_id"] == ex, ["policy_id", "period_start", "driver_age", "years_licensed"]].copy()
    view["years_licensed cleaned"] = m1.loc[m1["policy_id"] == ex, "years_licensed"].to_numpy()
    view["age at first licence (raw)"] = view["driver_age"] - view["years_licensed"]
    display(view)
else:
    print("2nd round: age bands only, the check is not possible")
"""),
        md("## 6. Claims\nExact duplicates under a second id, the reserve entry error, the claim cost of the case study, the link to the policy row on cover, and claims outside the cover bought."),
        code("""
rows = []
for line in C.LINES:
    p = cleaning.pad_ids(raw[f"policies_{line}"], f"policies_{line}", log)
    rows.append(p[["row_id", "policy_id", "period_start", "period_end"] +
                  [c for c in ("cover_type", "accidental_damage_cover") if c in p]])
rows = pd.concat(rows, ignore_index=True)
claims = cleaning.clean_claims(cleaning.normalise_codes(raw["claims"], "claims", log), rows, log)
print(f"claims: {len(raw['claims']):,} raw -> {len(claims):,} clean")
claims["match_status"].value_counts()
"""),
        md("## 7. The cleaning log of the full run\nWritten by `python 02_data_cleaning.py`: one line per rule and table."),
        code("""
log_full = pd.read_csv(C.REPORTS / "02_cleaning" / "cleaning_log.csv")
log_full[["rule_id", "table", "column", "rule", "rows_affected", "action", "evidence"]]
"""),
        md("## 8. After cleaning\nEvery key and link check of step 1 is run again on the cleaned tables, and the missing values left are listed (step 3 tests whether each gap is informative)."),
        code("""
after = pd.read_csv(C.REPORTS / "02_cleaning" / "checks_after_cleaning.csv")
print(f"{int(after['passed'].sum())} of {len(after)} checks pass")
display(after.loc[~after["passed"]])
mv = pd.read_csv(C.REPORTS / "02_cleaning" / "missing_values.csv")
mv.loc[mv["table"] != "reference_policies"].sort_values("missing_pct", ascending=False).head(25)
"""),
        code("""
pd.DataFrame([{"table": n, "rows": len(H.read_table(n, "clean", columns=[H.read_table(n, "clean").columns[0]]))}
              for n in C.TABLES])
"""),
    ]
    return cells


# =================================================================================================
def nb_features(rnd: str, demo_line: str):
    cells = [
        md(f"""
# {rnd} - Step 3: feature engineering and selection

1. **Join** each policy row with its customer, address and the weather of its address and year.
2. **Extract information**: season on cover, property age, age at first licence, sum insured per
   bedroom, the address's climate next to the year's weather, claims history known at the start
   of each year (policy, customer across lines, address), the other lines the customer holds.
3. **Targets**: each row's claims by peril (count and cost) and the share already reported.
4. **Selection**: every column must earn its place. Rules first (leakage, personal data,
   governance, constants, exact redundancy), then out-of-time tests against random *probe*
   columns, then near-duplicates. Every decision is in `reports/03_features/feature_register.csv`.

The script `python 03_feature_engineering.py` builds all three lines; this notebook rebuilds one
line ({demo_line}) to show the steps, then reads the full run's register and outputs.
"""),
        code(SETUP + '\nfe = importlib.import_module("03_feature_engineering")'),
        md("## 1. Reference tables, claims and reporting patterns"),
        code("""
ref = {t: H.read_table(t, "clean") for t in ("customers", "addresses", "weather")}
claims = H.read_table("claims", "clean")
all_rows = pd.concat([H.read_table(f"{k}_{l}", "clean", columns=["row_id", "policy_id", "customer_id", "address_id",
                                                                  "period_start", "exposure"]).assign(line=l)
                      for l in C.LINES for k in ("policies", "test")], ignore_index=True)
ev = fe.claim_events(claims, all_rows)
nonrep = claims.loc[claims["status"] != "repudiated"]
cap = nonrep.groupby(["line", "peril"])["net_incurred"].transform(lambda s: s.quantile(C.LARGE_LOSS_QUANTILE))
nonrep = nonrep.assign(w=np.minimum(nonrep["net_incurred"], cap))
min_loss = f"{min(C.TRAIN_YEARS)}-01-01"
patterns = {"cost": H.reporting_pattern(nonrep, C.VALUATION_DATE, min_loss, weight="w"),
            "count": H.reporting_pattern(nonrep, C.VALUATION_DATE, min_loss)}
fig, axes = plt.subplots(1, 3, figsize=(13, 3.2), sharey=True)
for ax, line in zip(axes, C.LINES):
    for p in C.PERILS[line]:
        F = patterns["cost"][(line, p)]
        ax.plot(np.arange(min(len(F), 731)), F[:731], label=p)
    ax.set_title(line); ax.set_xlabel("days after the loss"); ax.legend(fontsize=7)
axes[0].set_ylabel("share of cost reported")
plt.tight_layout(); plt.show()
"""),
        md(f"## 2. Building the {demo_line} table"),
        code(f'df = fe.build_line("{demo_line}", ref, ev, all_rows, patterns)\ndf.shape'),
        md("### Extracted columns"),
        code("""
new = ["start_month", "winter_fraction", "starts_after_jan1", "ends_before_year_end", "property_age",
       "age_first_licensed", "si_per_bedroom"] + [c for c in df.columns if c.startswith(("wx_", "hist_", "holds_"))]
new = [c for c in new if c in df]
df.loc[df["is_test"] == 0, new].describe().T.round(3)
"""),
        md("### Claims history is known at the start of the year\nFor a row of year Y only claims with loss date *and* report date before 1 January Y count, so a 2025 row sees exactly what was known on 31 December 2024. One policy across its years:"),
        code("""
pol = df.loc[(df["is_test"] == 0) & (df["hist_pol_claims_1y"] > 0), "policy_id"].iloc[0]
cols = ["policy_id", "year", "n_claims", "hist_pol_claims_1y", "hist_pol_claims_3y", "hist_pol_years_since_claim",
        "hist_pol_exposure_3y", "hist_cust_claims_3y"]
df.loc[df["policy_id"] == pol, cols].sort_values("year")
"""),
        md("### Targets and reporting completeness\nThe last training year is not fully reported at 31 December 2024; `comp_cost` (share of the row's expected cost already reported) and `comp_count` (share of its claims) let a model use it without bias."),
        code("""
tr = df.loc[df["is_test"] == 0]
tr.groupby("year").agg(rows=("row_id", "size"), exposure=("exposure", "sum"), claims=("n_claims", "sum"),
                       cost=("claim_cost", "sum"), comp_count=("comp_count", "mean"),
                       comp_cost=("comp_cost", "mean")).round(3)
"""),
        md("## 3. Selection: every column must earn its place\nThe full run's register, all lines. `reason` says why a column is kept or dropped; the `_z` columns are the out-of-time z-scores of each test. A column passes a test when it beats every random probe *and* its z-score reaches the line's calibrated threshold (see the calibration below); it is kept if it passes at least one test."),
        code("""
reg = pd.read_csv(C.REPORTS / "03_features" / "feature_register.csv")
pd.crosstab(reg["reason"], reg["line"], margins=True)
"""),
        code("""
zc = [c for c in reg.columns if c.endswith("_z")]
for line in C.LINES:
    r = reg.loc[(reg["line"] == line) & (reg["decision"] == "keep") & (reg["reason"] == "adds value")].copy()
    r["best z"] = r[zc].max(axis=1)
    r = r.sort_values("best z")
    thr = float(reg.loc[reg["line"] == line, "z_threshold"].dropna().iloc[0])
    fig, ax = plt.subplots(figsize=(7, 0.22 * len(r) + 1))
    ax.barh(r["column"], r["best z"].clip(upper=60), color="#4C72B0")
    ax.axvline(thr, color="#C44E52", ls="--")
    ax.set_title(f"{line}: kept features, strongest z-score (capped at 60); threshold {thr}")
    plt.tight_layout(); plt.show()
"""),
        md("Dropped columns and why:"),
        code("""
reg.loc[reg["decision"] == "drop", ["line", "column", "reason", "evidence"]].sort_values(["line", "reason", "column"])
"""),
        md("Calibration on the random probes. Each line's z threshold is the lowest of 1.0 / 1.5 / 2.0 / 2.5 at which at most 1 of the 20 probes would pass a test against the other probes (a false-keep rate of about 5%): thin data get a stricter threshold because their z-scores are noisier."),
        code("""
reg.groupby("line")[["z_threshold", "probe_false_passes"]].first()
"""),
        md("Missing values: a flag is added only where missing rows claim differently from the rest."),
        code("""
pd.read_csv(C.REPORTS / "03_features" / "missing_values.csv")
"""),
        md("## 4. The outputs\nOne training and one 2025 table per line, named as in the case study. The 2025 tables have the same features and no targets."),
        code("""
ss = H.read_table("sample_submission", "clean")
out = []
ids = []
for line in C.LINES:
    for kind in ("policies", "test"):
        f = H.read_table(f"{kind}_{line}", "features")
        out.append({"file": f"data/features/{kind}_{line}.parquet", "rows": len(f), "columns": f.shape[1]})
        if kind == "test":
            ids.append(f["row_id"])
ids = pd.concat(ids)
print("2025 rows equal the sample submission:", set(ids) == set(ss["row_id"]), "| duplicates:", int(ids.duplicated().sum()))
leak = [c for c in reg.loc[reg["reason"] == "leakage", "column"].unique()]
print("leakage columns in any output:", [c for c in leak for line in C.LINES
                                         if c in H.read_table(f"policies_{line}", "features").columns])
pd.DataFrame(out)
"""),
    ]
    return cells


def write(execute: bool, rounds: list[str]) -> None:
    for rnd, demo in (("1st round", "buildings"), ("2nd round", "buildings")):
        if rnd not in rounds:
            continue
        folder = ROOT / rnd
        books = {"01_data_reading.ipynb": nb_reading(rnd), "02_data_cleaning.ipynb": nb_cleaning(rnd),
                 "03_feature_engineering.ipynb": nb_features(rnd, demo)}
        for name, cells in books.items():
            nb = nbf.v4.new_notebook()
            nb["cells"] = cells
            nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
            path = folder / name
            nbf.write(nb, path)
            print("wrote", path)
            if execute:
                subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                                "--inplace", "--ExecutePreprocessor.timeout=3600", name], cwd=folder, check=True)
                print("  executed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--rounds", nargs="*", default=list(ROUNDS))
    a = ap.parse_args()
    write(a.execute, a.rounds)
