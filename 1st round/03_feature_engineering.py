"""Step 3: feature engineering and selection: one model-ready table per line.

1. **Join** each policy row with its customer, its address and the weather of its address and year.
2. **Extract information** the raw columns hold only implicitly: the season on cover, property
   age, age at first licence, sum insured per bedroom, the climate of the address (long-run
   weather) next to the weather of the year, the claims history known at the start of each year
   (policy, customer across lines, address), and the other lines the customer holds.
3. **Targets** (training rows): the claims of each row by peril (number and cost, net of
   recoveries, repudiated excluded) and the share of them already reported at the valuation date.
4. **Selection** - every column is tested from several angles and kept only if it adds value:
   leakage, personal data, governance, constant, exact redundancy (a column fully determined by
   another), near-duplicates (rank correlation >= 0.95), and an out-of-time test of predictive
   value against random "probe" columns: a feature is useful when it predicts the claims of a
   later year better than pure noise does, alone (one-way) or inside a model (permutation).
   Every decision is written to ``reports/03_features/feature_register.csv``.

Output (names as in the case study): ``data/features/policies_<line>.parquet`` (training rows:
keys, exposure, kept features, targets) and ``data/features/test_<line>.parquet`` (2025 rows:
keys, exposure, kept features). Usage: ``python 03_feature_engineering.py [--lines motor ...]``
"""
from __future__ import annotations

import argparse
import hashlib
import sys

import lightgbm as lgb

import numpy as np
import pandas as pd

import config as C
import helpers as H

OUT = C.REPORTS / "03_features"
KEYS = ["row_id", "policy_id", "customer_id", "address_id", "year", "period_start", "period_end"]
BASE = ["exposure"]
WEATHER_PERILS = ("storm", "flood", "subsidence", "escape_of_water")
HOME_HISTORY_PERILS = ("flood", "subsidence", "escape_of_water")
# Features with a documented physical link to one peril (tested on that peril's claims as well).
PERIL_DRIVERS = {
    "flood": ["flood_zone", "distance_to_river_km", "elevation_m", "basement", "wx_rainfall_mm",
              "wx_rainfall_mm_mean"],
    "subsidence": ["soil_type", "trees_within_5m", "wx_dry_summer_index", "wx_dry_summer_index_mean",
                   "property_age"],
    "fire": ["construction_type", "smoke_alarms", "wx_wildfire_index", "wx_wildfire_index_mean"],
    "escape_of_water": ["property_age", "wx_freezing_days", "wx_freezing_days_mean", "days_unoccupied",
                        "wx_mean_temperature_c", "wx_mean_temperature_c_mean"],
    "storm": ["roof_type", "wx_wind_exposure_index", "wx_wind_exposure_index_mean", "property_type",
              "trees_within_5m"],
    "theft": ["urban_density", "security_level", "overnight_parking", "high_value_items", "bicycle_cover"],
    "glass_hail": ["wx_hail_days", "wx_hail_days_mean"],
    "accidental_damage": ["accidental_damage_cover", "works_from_home", "lodgers", "household_size"],
}


# =================================================================================================
# 1-2. Joining and extracting information
# =================================================================================================
def load_line(line: str) -> pd.DataFrame:
    tr = H.read_table(f"policies_{line}", "clean").assign(is_test=0)
    te = H.read_table(f"test_{line}", "clean").assign(is_test=1)
    df = pd.concat([tr, te], ignore_index=True)
    df["year"] = df["period_start"].dt.year.astype("int16")
    return df


def join_reference(df: pd.DataFrame, customers: pd.DataFrame, addresses: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    df = df.merge(customers, on="customer_id", how="left", validate="many_to_one")
    df = df.merge(addresses, on="address_id", how="left", validate="many_to_one")
    assert len(df) == n, "a join changed the number of rows"
    return df


def winter_fraction(start: pd.Series, end: pd.Series) -> np.ndarray:
    """Share of the covered days falling in December, January or February."""
    origin = start.min() - pd.Timedelta(days=1)
    cal = pd.date_range(origin, end.max(), freq="D")
    cum = np.cumsum(np.isin(cal.month, (12, 1, 2))).astype(np.int64)
    i_end = (end - origin).dt.days.to_numpy()
    i_before = (start - origin).dt.days.to_numpy() - 1
    return (cum[i_end] - cum[i_before]) / (i_end - i_before)


def extract_features(df: pd.DataFrame, line: str) -> pd.DataFrame:
    """Derived columns: season, ages and durations, ratios, ordinal codes."""
    s, e = df["period_start"], df["period_end"]
    year_start = pd.to_datetime(s.dt.year.astype(str) + "-01-01")
    new = {
        "start_month": s.dt.month.astype("int8"),
        "winter_fraction": winter_fraction(s, e).astype("float32"),
        # The case study's policy year has 365 days: a full 2024 row runs 1 Jan to 30 Dec.
        "starts_after_jan1": (s > year_start).astype("int8"),
        "ends_before_year_end": (e < year_start + pd.Timedelta(days=364)).astype("int8"),
    }
    if "year_built" in df:
        new["property_age"] = (df["year"] - df["year_built"]).astype("float32")
    if "driver_age" in df:
        new["age_first_licensed"] = (df["driver_age"] - df["years_licensed"]).astype("float32")
    if "sum_insured" in df and "bedrooms" in df:
        new["si_per_bedroom"] = (df["sum_insured"] / df["bedrooms"].clip(lower=1)).astype("float32")
    out = df.assign(**new)
    for col, codes in C.ORDINAL_CODES.items():          # letters -> ordered numbers
        if col in out:
            out[col] = out[col].map(codes).astype("float32")
    for col in [c for c in getattr(C, "AGE_BANDS", {}) if c in out]:     # 2nd round: age bands
        out[col] = out[col].map(C.AGE_BANDS[col]).astype("float32")
    return out


def weather_features(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """The weather of the row's address and year, and the address's climate (mean of the
    training years). The anomaly (year - climate) is their difference, so it is not stored."""
    w = weather.copy()
    vars_ = [v for v in C.WEATHER_VARS if v in w]
    clim = w.loc[w["year"].isin(C.TRAIN_YEARS)].groupby("address_id")[vars_].mean().add_suffix("_mean")
    cur = w.set_index(["address_id", "year"])[vars_]
    key = pd.MultiIndex.from_arrays([df["address_id"], df["year"].astype(int)])
    out = {f"wx_{v}": cur[v].reindex(key).to_numpy(dtype="float32") for v in vars_}
    cl = clim.reindex(df["address_id"])
    out.update({f"wx_{c}": cl[c].to_numpy(dtype="float32") for c in clim.columns})
    return df.assign(**out)


def claim_events(claims: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Non-repudiated claims linked to a policy row, with that row's customer and address."""
    ev = claims.loc[(claims["match_status"] == "matched") & (claims["status"] != "repudiated"),
                    ["claim_id", "row_id", "policy_id", "line", "peril", "loss_date", "report_date",
                     "status", "net_incurred"]]
    return ev.merge(rows[["row_id", "customer_id", "address_id"]], on="row_id", how="left")


def history_features(df: pd.DataFrame, line: str, ev: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Claims history known at the start of each row's year (no look-ahead).

    Only claims with loss date *and* report date before 1 January of the row's year count, so a
    2025 row sees exactly what was known on 31 December 2024. Windows are calendar years.
    """
    w = C.HISTORY_YEARS
    first_year = min(C.TRAIN_YEARS)
    caps = ev.groupby("line")["net_incurred"].quantile(C.LARGE_LOSS_QUANTILE)
    ev = ev.assign(cost=np.minimum(ev["net_incurred"], ev["line"].map(caps)))
    pexp = rows.assign(y=rows["period_start"].dt.year).groupby(["policy_id", "y"])["exposure"].sum()
    names = ["hist_pol_claims_1y", f"hist_pol_claims_{w}y", f"hist_pol_cost_{w}y", "hist_pol_years_since_claim",
             f"hist_pol_exposure_{w}y", f"hist_cust_claims_{w}y", f"hist_cust_claims_other_lines_{w}y",
             "hist_years_available"]
    home = line in ("buildings", "contents")
    if home:
        names += [f"hist_addr_claims_{w}y", f"hist_addr_weather_claims_{w}y"] + \
                 [f"hist_pol_{p}_{w}y" for p in HOME_HISTORY_PERILS if p in C.PERILS[line]]
    out = pd.DataFrame(np.nan, index=df.index, columns=names, dtype="float64")
    for y in sorted(df["year"].unique()):
        idx = df.index[df["year"] == y]
        sub = df.loc[idx]
        cut = pd.Timestamp(year=int(y), month=1, day=1)
        known = ev.loc[(ev["loss_date"] < cut) & (ev["report_date"] < cut)]
        k_w = known.loc[known["loss_date"] >= pd.Timestamp(year=int(y) - w, month=1, day=1)]
        k_1 = known.loc[known["loss_date"] >= pd.Timestamp(year=int(y) - 1, month=1, day=1)]
        own = k_w.loc[k_w["line"] == line]
        vals = {
            "hist_pol_claims_1y": sub["policy_id"].map(k_1.loc[k_1["line"] == line].groupby("policy_id").size()),
            f"hist_pol_claims_{w}y": sub["policy_id"].map(own.groupby("policy_id").size()),
            f"hist_pol_cost_{w}y": sub["policy_id"].map(own.groupby("policy_id")["cost"].sum()),
            "hist_pol_years_since_claim": sub["policy_id"].map((cut - own.groupby("policy_id")["loss_date"].max()).dt.days / 365.25),
            f"hist_cust_claims_{w}y": sub["customer_id"].map(k_w.groupby("customer_id").size()),
            f"hist_cust_claims_other_lines_{w}y": sub["customer_id"].map(k_w.loc[k_w["line"] != line].groupby("customer_id").size()),
        }
        e = pexp.loc[(pexp.index.get_level_values(1) >= y - w) & (pexp.index.get_level_values(1) < y)]
        vals[f"hist_pol_exposure_{w}y"] = sub["policy_id"].map(e.groupby(level=0).sum())
        if home:
            hk = k_w.loc[k_w["line"].isin(["buildings", "contents"])]
            vals[f"hist_addr_claims_{w}y"] = sub["address_id"].map(hk.groupby("address_id").size())
            vals[f"hist_addr_weather_claims_{w}y"] = sub["address_id"].map(
                hk.loc[hk["peril"].isin(WEATHER_PERILS)].groupby("address_id").size())
            for p in HOME_HISTORY_PERILS:
                if p in C.PERILS[line]:
                    vals[f"hist_pol_{p}_{w}y"] = sub["policy_id"].map(own.loc[own["peril"] == p].groupby("policy_id").size())
        for k, v in vals.items():
            v = v.astype(float)
            out.loc[idx, k] = v.to_numpy() if k == "hist_pol_years_since_claim" else v.fillna(0.0).to_numpy()
        out.loc[idx, "hist_years_available"] = float(min(max(int(y) - first_year, 0), w))
    return pd.concat([df, out.astype("float32")], axis=1)


def holdings_features(df: pd.DataFrame, line: str, all_rows: pd.DataFrame) -> pd.DataFrame:
    """Which other lines the customer holds in the same year (train and 2025 rows alike)."""
    held = all_rows.assign(y=all_rows["period_start"].dt.year).groupby(["customer_id", "y"])["line"].agg(set)
    key = pd.MultiIndex.from_arrays([df["customer_id"], df["year"].astype(int)])
    s = held.reindex(key)
    new = {f"holds_{o}": np.array([o in x if isinstance(x, set) else False for x in s], dtype="int8")
           for o in C.LINES if o != line}
    return df.assign(**new)


def targets(df: pd.DataFrame, line: str, ev: pd.DataFrame, patterns_cost: dict, patterns_count: dict,
            shares: dict) -> pd.DataFrame:
    """Claims of each training row by peril, and the share already reported (IBNR completeness)."""
    tr = df["is_test"] == 0
    e = ev.loc[ev["line"] == line]
    g = e.groupby("row_id")
    new = {"n_claims": df["row_id"].map(g.size()), "claim_cost": df["row_id"].map(g["net_incurred"].sum()),
           "n_open": df["row_id"].map(e.loc[e["status"] == "open"].groupby("row_id").size())}
    for p in C.PERILS[line]:
        ep = e.loc[e["peril"] == p].groupby("row_id")
        new[f"n_{p}"] = df["row_id"].map(ep.size())
        new[f"cost_{p}"] = df["row_id"].map(ep["net_incurred"].sum())
    out = {k: np.where(tr, v.fillna(0.0).to_numpy(dtype=float), np.nan) for k, v in new.items()}
    rows = df.loc[tr, ["period_start", "period_end"]]
    comp_cost = np.zeros(int(tr.sum()))
    comp_count = np.zeros(int(tr.sum()))
    for p in C.PERILS[line]:
        cp = H.row_completeness(rows, patterns_cost[(line, p)], C.VALUATION_DATE)
        cn = H.row_completeness(rows, patterns_count[(line, p)], C.VALUATION_DATE)
        col = np.full(len(df), np.nan)
        col[tr.to_numpy()] = cp
        out[f"comp_{p}"] = col
        comp_cost += shares["cost"][p] * cp
        comp_count += shares["count"][p] * cn
    for name, arr in (("comp_cost", comp_cost), ("comp_count", comp_count)):
        col = np.full(len(df), np.nan)
        col[tr.to_numpy()] = arr
        out[name] = col
    return df.assign(**{k: v.astype("float32") if k.startswith("comp") else v for k, v in out.items()})


def peril_shares(ev: pd.DataFrame, line: str) -> dict[str, dict[str, float]]:
    """Cost and count shares of each peril in the mature years (all but the last training year)."""
    e = ev.loc[(ev["line"] == line) & (ev["loss_date"].dt.year < max(C.TRAIN_YEARS))]
    cap = e.groupby("peril")["net_incurred"].transform(lambda s: s.quantile(C.LARGE_LOSS_QUANTILE))
    cost = np.minimum(e["net_incurred"], cap).groupby(e["peril"]).sum().reindex(C.PERILS[line], fill_value=0)
    cnt = e.groupby("peril").size().reindex(C.PERILS[line], fill_value=0)
    return {"cost": (cost / cost.sum()).to_dict(), "count": (cnt / cnt.sum()).to_dict()}


# =================================================================================================
# 4. Selection
# =================================================================================================
def source_columns(df: pd.DataFrame) -> list[str]:
    drop = set(KEYS) | {"is_test", "line"}
    return [c for c in df.columns if c not in drop and not c.startswith(("n_", "cost_", "comp_"))
            and c not in ("claim_cost", "n_claims", "n_open")]


def rule_based_decisions(df: pd.DataFrame, line: str) -> dict[str, tuple[str, str, str]]:
    """Decisions that need no model: leakage, personal data, governance, keys, constants and
    exact redundancy, each verified on the data. Returns {column: (decision, reason, evidence)}."""
    d: dict[str, tuple[str, str, str]] = {}
    tr = df.loc[df["is_test"] == 0]
    for c, why in C.LEAKAGE.items():
        if c in df:
            d[c] = ("drop", "leakage", why)
    for c, why in C.PERSONAL_DATA.items():
        if c in df:
            d[c] = ("drop", "personal data", why)
    for c, why in C.GOVERNANCE.items():
        if c in df:
            d[c] = ("drop", "governance", why)
    for c in df.columns:
        if c.startswith("dq_") and c not in getattr(C, "DQ_FLAGS_AS_FEATURES", ("dq_ncb_exceeds_licence", "dq_mileage_zero")):
            d[c] = ("drop", "data-quality flag", "records a correction (kept in data/clean); no risk meaning")
    # a vehicle code that does not identify a vehicle cannot carry vehicle risk
    if {"make_model", "body_type", "fuel_type"} <= set(df.columns):
        per = df.groupby("make_model")[["body_type", "fuel_type"]].nunique().mean()
        tot = df[["body_type", "fuel_type"]].nunique()
        if (per >= 0.9 * tot).all():
            d["make_model"] = ("drop", "not an identifier",
                               f"every code appears with {per['body_type']:.1f} of {tot['body_type']} body types and "
                               f"{per['fuel_type']:.1f} of {tot['fuel_type']} fuel types: a random label, not a vehicle")
    # exact redundancy, verified
    if "customer_since_year" in df:
        eq = float(((df["year"] - df["customer_since_year"]) == df["renewal_count"]).mean())
        d["customer_since_year"] = ("drop", "redundant", f"year - customer_since_year == renewal_count on {eq:.2%} of rows")
    if "new_business" in df:
        eq = float((df["new_business"].astype(bool) == (df["renewal_count"] == 0)).mean())
        d["new_business"] = ("drop", "redundant", f"new_business == (renewal_count == 0) on {eq:.2%} of rows")
    if "year_built" in df:
        d["year_built"] = ("drop", "redundant", "replaced by property_age = year - year_built")
    for c in source_columns(df):
        if c in d:
            continue
        s = tr[c]
        top = s.value_counts(dropna=False, normalize=True)
        if len(top) and top.iloc[0] >= 0.995:
            d[c] = ("drop", "constant", f"one value on {top.iloc[0]:.2%} of training rows ({top.index[0]})")
    return d


N_PROBES = 20
PROBE_LEVELS = (2, 3, 5, 8, 12, 20, 30, 50, 100, 300)


def make_probes(row_ids: pd.Series) -> pd.DataFrame:
    """Random columns with no relation to claims: the yardstick of 'no value'. Each row's
    probe values come from a hash of its row_id, so they are the same in every fold."""
    h = np.array([int(hashlib.md5(f"probe:{r}".encode()).hexdigest()[:16], 16) for r in row_ids], dtype=np.uint64)
    cols = {}
    with np.errstate(over="ignore"):
        for i in range(N_PROBES):
            u = (H._mix64(h ^ np.uint64(0x9E3779B97F4A7C15 * (i + 1) % (1 << 64))) >> np.uint64(11)).astype(float) / float(1 << 53)
            if i < 10:
                cols[f"probe_num_{i}"] = u.astype("float32")
            else:
                k = PROBE_LEVELS[i - 10]
                cols[f"probe_cat_{i - 10}"] = np.floor(u * k).astype(int).astype(str)
    return pd.DataFrame(cols, index=row_ids.index)


def is_categorical(s: pd.Series) -> bool:
    return not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s))


def _model_frame(X: pd.DataFrame, cats: list[str], fit: np.ndarray) -> pd.DataFrame:
    """LightGBM-ready copy: categories fixed on the fit rows (levels with >= 50 rows), numbers
    and booleans as float32 (missing as NaN)."""
    cols = {}
    for c in X.columns:
        if c in cats:
            s = X[c].astype("string")
            lv = s[fit].value_counts()
            keep = lv.index[lv >= 50].tolist()
            cols[c] = pd.Categorical(s.where(s.isin(keep)), categories=keep)
        else:
            cols[c] = X[c].astype("Float64").to_numpy(dtype="float32", na_value=np.nan)
    return pd.DataFrame(cols, index=X.index)


GBM = {"learning_rate": 0.1, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
       "lambda_l2": 10.0, "verbosity": -1, "num_threads": 2, "deterministic": True, "force_col_wise": True,
       "cat_smooth": 20.0, "min_data_per_group": 100}


class Accumulator:
    """Pools a test's deviances (and their bootstrap replicates) over the folds."""

    def __init__(self, boot: H.ClusterBootstrap):
        self.boot = boot
        self.sums: dict[tuple[str, str], np.ndarray] = {}

    def add(self, feature: str, test: str, codes, W, d_ref: np.ndarray, d_alt: np.ndarray) -> None:
        r = np.concatenate([[d_ref.sum(), d_alt.sum()], self.boot.replicate_sums(codes, W, d_ref),
                            self.boot.replicate_sums(codes, W, d_alt)])
        k = (feature, test)
        self.sums[k] = self.sums.get(k, 0) + r

    def table(self, kind: dict[str, str]) -> pd.DataFrame:
        """One-way tests: gain = 1 - alt/ref (alt = with the feature). Model tests: loss = alt/ref - 1
        (alt = feature shuffled). Both in % of the reference deviance, with bootstrap SE."""
        B = self.boot.B
        recs: dict[str, dict] = {}
        for (f, t), r in self.sums.items():
            ref, alt, rb, ab = r[0], r[1], r[2:2 + B], r[2 + B:]
            if kind[t] == "gain":
                stat, reps = 1 - alt / ref, 1 - ab / rb
            else:
                stat, reps = alt / ref - 1, ab / rb - 1
            recs.setdefault(f, {})[t] = 100 * stat
            recs[f][f"{t}_se"] = 100 * float(np.std(reps))
        return pd.DataFrame(recs).T


def fit_gbm(X: pd.DataFrame, y: np.ndarray, init: np.ndarray, fi: np.ndarray, groups: pd.Series,
            objective: str, leaves: int, min_leaf: int) -> lgb.Booster:
    params = {**GBM, "objective": objective, "num_leaves": leaves, "min_data_in_leaf": min_leaf, "seed": C.SEED}
    es = H.stable_hash(groups.iloc[fi], "es") < 2000              # 20% of fit customers stop the trees
    cats = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
    dtr = lgb.Dataset(X.iloc[fi[~es]], y[fi[~es]], init_score=init[fi[~es]], categorical_feature=cats)
    des = lgb.Dataset(X.iloc[fi[es]], y[fi[es]], init_score=init[fi[es]], categorical_feature=cats, reference=dtr)
    return lgb.train(params, dtr, 1500, valid_sets=[des], callbacks=[lgb.early_stopping(50, verbose=False)])


MAX_PERMUTATION_ROWS = 120_000   # test rows used per fold for the permutation tests (speed)


def permutation_into(acc: Accumulator, test_name: str, b: lgb.Booster, X: pd.DataFrame, y: np.ndarray,
                     init: np.ndarray, ti: np.ndarray, groups: pd.Series, power: float) -> None:
    """Shuffle each column on the test rows and record the model's deviance with and without it.

    At most ``MAX_PERMUTATION_ROWS`` test rows are used (whole customers, chosen by a hash)."""
    g = groups.iloc[ti]
    if len(ti) > MAX_PERMUTATION_ROWS:
        keep = H.stable_hash(g, "perm") < int(10_000 * MAX_PERMUTATION_ROWS / len(ti))
        ti, g = ti[keep], g[keep]
    codes, W = acc.boot.weights(g)
    Xt, yt, it = X.iloc[ti].reset_index(drop=True), y[ti], init[ti]
    base = H.tweedie_deviance(yt, np.exp(b.predict(Xt, raw_score=True) + it), power)
    perm = np.random.default_rng(C.SEED).permutation(len(Xt))
    for c in X.columns:
        keep = Xt[c]
        v = keep.to_numpy()[perm]
        Xt[c] = pd.Categorical(v, categories=keep.cat.categories) if isinstance(keep.dtype, pd.CategoricalDtype) else v
        d = H.tweedie_deviance(yt, np.exp(b.predict(Xt, raw_score=True) + it), power)
        Xt[c] = keep                                    # restore the column
        acc.add(c, test_name, codes, W, base, d)


def year_relativity(year: np.ndarray, y: np.ndarray, base: np.ndarray, fit: np.ndarray) -> np.ndarray:
    """Each year's actual / expected on the fit rows, relative to all fit rows (1 for a year
    without fit rows, e.g. an out-of-time test year, whose level is rescaled anyway)."""
    s = pd.DataFrame({"y": year[fit], "a": y[fit], "b": base[fit]}).groupby("y").sum()
    rel = (s["a"] / s["b"]) / (y[fit].sum() / base[fit].sum())
    return pd.Series(year).map(rel).fillna(1.0).to_numpy(float)


def signal_tests(df: pd.DataFrame, line: str, features: list[str], ev: pd.DataFrame) -> pd.DataFrame:
    """Out-of-time value of every candidate, pooled over the folds of ``config.SIGNAL_FOLDS``.

    For each fold (learn on the earlier years, score the next one) and each candidate:

    * ``frequency_oneway``  claim counts, one-way relativities (Poisson deviance gain, %);
    * ``severity_oneway``   claim sizes, one-way relativities (gamma deviance gain, %);
    * ``<peril>_oneway``    one peril's claim counts, for features with a physical link to it;
    * ``frequency_model``   loss of a LightGBM frequency model when the column is shuffled (%);
    * ``severity_model``    the same for a LightGBM claim-size model.

    Each statistic comes with a customer-bootstrap standard error.
    """
    boot = H.ClusterBootstrap(100, seed=C.SEED)
    acc = Accumulator(boot)
    kind = {"frequency_oneway": "gain", "severity_oneway": "gain", "frequency_model": "loss", "severity_model": "loss"}
    kind.update({f"{p}_oneway": "gain" for p in C.PERILS[line]})
    e_line = ev.loc[(ev["line"] == line) & (ev["net_incurred"] > 0)]
    for fit_years, test_years, half in C.SIGNAL_FOLDS:
        tr = df.loc[(df["is_test"] == 0) & df["year"].isin([*fit_years, *test_years])].reset_index(drop=True)
        fit = tr["year"].isin(fit_years).to_numpy()
        test = tr["year"].isin(test_years).to_numpy()
        if half is not None:            # cross-fit: learn on one half of the customers, score the other
            in_half = (H.stable_hash(tr["customer_id"], "half") < 5000) == bool(half)
            fit, test = fit & in_half, test & ~in_half
        X = pd.concat([tr[features], make_probes(tr["row_id"])], axis=1)
        cols = list(X.columns)
        pos = pd.Series(np.arange(len(tr)), index=tr["row_id"])
        e = e_line.loc[e_line["row_id"].isin(pos.index)]
        epos = pos.loc[e["row_id"]].to_numpy()
        cap = float(e.loc[fit[epos], "net_incurred"].quantile(C.LARGE_LOSS_QUANTILE))
        sev_y = np.minimum(e["net_incurred"].to_numpy(float), cap)
        sev_fit, sev_test = fit[epos], test[epos]
        year = tr["year"].to_numpy()
        n = tr["n_claims"].to_numpy(float)
        # Baselines include the level of each year (from the fit rows when the year is among them):
        # a column then gets no credit for merely telling the years apart.
        expo_n = (tr["exposure"] * tr["comp_count"]).to_numpy(float) * year_relativity(year, n, (tr["exposure"] * tr["comp_count"]).to_numpy(float), fit)
        # claim-size baseline: the mean size of the claim's peril (perils differ by 10x and more)
        peril_mean = pd.Series(sev_y[sev_fit]).groupby(e["peril"].to_numpy()[sev_fit]).mean()
        sev_peril = e["peril"].map(peril_mean).fillna(float(sev_y[sev_fit].mean())).to_numpy(float)
        sev_base = sev_peril * year_relativity(year[epos], sev_y, sev_peril, sev_fit)
        mean_sev = float(sev_y[sev_fit].mean())
        peril_y, peril_e = {}, {}
        for p in C.PERILS[line]:
            peril_y[p] = tr[f"n_{p}"].to_numpy(float)
            e_p = (tr["exposure"] * tr[f"comp_{p}"]).to_numpy(float)
            peril_e[p] = e_p * year_relativity(year, peril_y[p], e_p, fit)
        cust = tr["customer_id"]
        rc, rW = boot.weights(cust[test])
        cc, cW = boot.weights(pd.Series(cust.to_numpy()[epos][sev_test]))
        for c in cols:
            lv = H.binned(X[c], fit)
            d0, d1 = H.oneway_deviances(lv, n, expo_n, fit, test, 1.0, 30.0)
            acc.add(c, "frequency_oneway", rc, rW, d0, d1)
            lvc = lv.iloc[epos].reset_index(drop=True)
            d0, d1 = H.oneway_deviances(lvc, sev_y, sev_base, sev_fit, sev_test, 2.0, 30.0 * mean_sev)
            acc.add(c, "severity_oneway", cc, cW, d0, d1)
            for p in C.PERILS[line]:
                if c in PERIL_DRIVERS.get(p, []) or c.startswith("probe_"):
                    y_p = peril_y[p]
                    d0, d1 = H.oneway_deviances(lv, y_p, peril_e[p], fit, test, 1.0, 10.0)
                    acc.add(c, f"{p}_oneway", rc, rW, d0, d1)
        cats = [c for c in cols if is_categorical(X[c])]
        Xm = _model_frame(X, cats, fit)
        del X
        init_n = np.log(np.maximum(expo_n, 1e-9) * n[fit].sum() / expo_n[fit].sum())
        bf = fit_gbm(Xm, n, init_n, np.flatnonzero(fit), cust, "poisson", 15, 400)
        permutation_into(acc, "frequency_model", bf, Xm, n, init_n, np.flatnonzero(test), cust, 1.0)
        Xmc = Xm.iloc[epos].reset_index(drop=True)
        init_s = np.log(sev_base)
        bs = fit_gbm(Xmc, sev_y, init_s, np.flatnonzero(sev_fit), pd.Series(cust.to_numpy()[epos]), "gamma", 7, 200)
        permutation_into(acc, "severity_model", bs, Xmc, sev_y, init_s, np.flatnonzero(sev_test),
                         pd.Series(cust.to_numpy()[epos]), 2.0)
        print(f"    fold fit {fit_years} -> test {test_years}{'' if half is None else f' (customer half {half})'}: frequency {bf.best_iteration} trees, "
              f"claim size {bs.best_iteration} trees; {int(test.sum()):,} test rows, {int(sev_test.sum()):,} test claims")
    return acc.table(kind)


REUSE_TESTS = False
FIXED_Z = None                  # --z-threshold: a fixed threshold instead of the calibrated one
Z_GRID = (1.0, 1.5, 2.0, 2.5)   # candidate z thresholds, from lenient to strict
MAX_PROBE_PASSES = 1            # ... the lowest one at which at most 1 of the 20 probes would pass


def _passes(stats: pd.DataFrame, tests: list[str], rows: pd.Index, ref: pd.Index, zmin: float) -> pd.DataFrame:
    """Test-by-test pass flags: z >= zmin and the statistic above every probe in ``ref``."""
    out = {}
    for t in tests:
        z = stats.loc[rows, t] / stats.loc[rows, f"{t}_se"].replace(0, np.nan)
        thr = stats.loc[ref, t].max()
        out[t] = (z >= zmin) & (stats.loc[rows, t] > (thr if pd.notna(thr) else -np.inf))
    return pd.DataFrame(out).fillna(False)


def decide(stats: pd.DataFrame) -> pd.DataFrame:
    """A column passes a test when it beats every random probe and its z-score clears the line's
    threshold. The threshold is calibrated on the probes themselves: the lowest value of
    ``Z_GRID`` at which at most ``MAX_PROBE_PASSES`` probe would pass any test against the other
    probes (leave-one-out), i.e. a false-keep rate of about 5% or less for a column of pure noise."""
    stats = stats[[c for c in stats.columns if not c.endswith("_z")]].copy()
    tests = [c for c in stats.columns if not c.endswith("_se")]
    probes = stats.index[stats.index.str.startswith("probe_")]
    feats = stats.index[~stats.index.str.startswith("probe_")]
    chosen, false_passes = Z_GRID[-1], None
    grid = (FIXED_Z,) if FIXED_Z is not None else Z_GRID
    for zmin in grid:
        n = sum(bool(_passes(stats, tests, pd.Index([p]), probes.drop(p), zmin).any(axis=1).iloc[0]) for p in probes)
        if n <= MAX_PROBE_PASSES or FIXED_Z is not None:
            chosen, false_passes = zmin, n
            break
    if false_passes is None:
        false_passes = sum(bool(_passes(stats, tests, pd.Index([p]), probes.drop(p), chosen).any(axis=1).iloc[0])
                           for p in probes)
    for t in tests:
        stats[f"{t}_z"] = stats[t] / stats[f"{t}_se"].replace(0, np.nan)
    passed = _passes(stats, tests, feats, probes, chosen)
    out = stats.loc[feats].copy()
    out["tests_passed"] = passed.sum(axis=1).astype(int)
    out["passed_which"] = passed.apply(lambda r: ", ".join(r.index[r.astype(bool)]), axis=1)
    out.attrs.update({"z_threshold": chosen, "probe_false_passes": false_passes, "n_probes": len(probes)})
    return out


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    t = pd.crosstab(a.astype("string").fillna("NA"), b.astype("string").fillna("NA")).to_numpy(float)
    if min(t.shape) < 2:
        return 0.0
    n = t.sum()
    e = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / n
    chi2 = ((t - e) ** 2 / e).sum()
    return float(np.sqrt(chi2 / (n * (min(t.shape) - 1))))


def redundancy(df: pd.DataFrame, features: list[str], stats: pd.DataFrame) -> list[tuple[str, str, float, str]]:
    """Near-duplicate pairs among the useful features: |Spearman| >= 0.95 between numbers, or
    Cramer's V >= 0.95 between columns with at most 50 levels. The one with the weaker
    out-of-time evidence (sum of its positive z-scores) is redundant."""
    tr = df.loc[df["is_test"] == 0, features].sample(min(int((df["is_test"] == 0).sum()), 200_000), random_state=C.SEED)
    zcols = [c for c in stats.columns if c.endswith("_z")]
    strength = stats[zcols].clip(lower=0).sum(axis=1)
    num = [c for c in features if not is_categorical(tr[c])]
    out = []
    if num:
        vals = tr[num].astype("Float64").astype("float64")
        rho = vals.rank().corr()
        modes = {c: vals[c].mode(dropna=True) for c in num}
        for i, a in enumerate(num):
            for b in num[i + 1:]:
                r = rho.loc[a, b]
                if abs(r) < 0.95:
                    continue
                # Two mostly-zero columns (e.g. past claim counts and past claim costs) agree on
                # the zeros by construction: judge them on the rows where either is not at its mode.
                ma, mb = modes[a], modes[b]
                off = ((vals[a] != (ma.iloc[0] if len(ma) else np.nan)) | (vals[b] != (mb.iloc[0] if len(mb) else np.nan)))
                if off.mean() < 0.5 and off.sum() > 100:
                    r = vals.loc[off, [a, b]].rank().corr().iloc[0, 1]
                if abs(r) >= 0.95:
                    out.append((a, b, float(r), "rank correlation"))
    low = [c for c in features if tr[c].nunique(dropna=False) <= 50]
    for i, a in enumerate(low):
        for b in low[i + 1:]:
            if a in num and b in num:
                continue
            v = cramers_v(tr[a], tr[b])
            if v >= 0.95:
                out.append((a, b, v, "Cramer's V"))
    res = []
    for a, b, r, how in out:
        weak, strong = (a, b) if strength.get(a, 0) < strength.get(b, 0) else (b, a)
        res.append((weak, strong, r, how))
    return res


def missing_flags(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Is a column's missingness informative? Claim frequency of rows missing it vs the rest."""
    tr = df.loc[df["is_test"] == 0]
    e = (tr["exposure"] * tr["comp_count"]).to_numpy(float)
    n = tr["n_claims"].to_numpy(float)
    recs = []
    binary = {c: tr[c].astype("Float64").fillna(-1).to_numpy() for c in features
              if not is_categorical(tr[c]) and tr[c].nunique(dropna=True) == 2}
    for c in features:
        m = tr[c].isna().to_numpy()
        if m.mean() < 0.005 or m.all():
            continue
        twin = [k for k, v in binary.items() if k != c and ((v == np.nanmax(v)) == m).all() | ((v == np.nanmin(v)) == m).all()]
        twin += [k for k in features if k != c and not is_categorical(tr[k]) and k not in binary
                 and ((tr[k].astype("Float64").fillna(-1).to_numpy() == 0) == m).all()]
        if twin:   # telematics_score: missing exactly when telematics_opt_in is 0; years since claim: no claim
            recs.append({"column": c, "missing_pct": round(100 * m.mean(), 2), "informative": False,
                         "note": f"missing exactly when {twin[0]} says so: structural, no flag needed"})
            continue
        f1, f0 = n[m].sum() / e[m].sum(), n[~m].sum() / e[~m].sum()
        rel = f1 / f0
        se = np.sqrt(1 / max(n[m].sum(), 1) + 1 / max(n[~m].sum(), 1))
        recs.append({"column": c, "missing_pct": round(100 * m.mean(), 2), "claims_missing": int(n[m].sum()),
                     "frequency_relativity": round(rel, 3), "z": round(np.log(rel) / se, 1),
                     "informative": bool(abs(np.log(rel)) > 3 * se and abs(rel - 1) > 0.05)})
    return pd.DataFrame(recs)


# =================================================================================================
# Driver
# =================================================================================================
def build_line(line: str, ref: dict, ev: pd.DataFrame, all_rows: pd.DataFrame, patterns: dict) -> pd.DataFrame:
    with H.Timer(f"{line}: joining and extracting information"):
        df = load_line(line)
        df = join_reference(df, ref["customers"], ref["addresses"])
        df = extract_features(df, line)
        df = weather_features(df, ref["weather"])
        df = history_features(df, line, ev, all_rows.loc[all_rows["line"] == line])
        df = holdings_features(df, line, all_rows)
        shares = peril_shares(ev, line)
        df = targets(df, line, ev, patterns["cost"], patterns["count"], shares)
        print(f"  {len(df):,} rows, {df.shape[1]} columns")
    return df


def select(df: pd.DataFrame, line: str, ev: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    reg = {}
    rules = rule_based_decisions(df, line)
    cands = [c for c in source_columns(df) if c not in rules and c not in BASE]
    with H.Timer(f"{line}: out-of-time value of {len(cands)} candidates ({len(C.SIGNAL_FOLDS)} folds)"):
        saved = OUT / f"signal_tests_{line}.csv"
        raw_stats = None
        if REUSE_TESTS and saved.exists():
            prev = pd.read_csv(saved, index_col=0)
            if set(cands) <= set(prev.index):
                raw_stats = prev.loc[[c for c in prev.index if c in cands or c.startswith("probe_")]]
                print("  reusing the saved test results")
        if raw_stats is None:
            raw_stats = signal_tests(df, line, cands, ev)
            raw_stats.to_csv(saved)
        stats = decide(raw_stats)
        print(f"  z threshold {stats.attrs['z_threshold']}: {stats.attrs['probe_false_passes']} of "
              f"{stats.attrs['n_probes']} random probes would pass (leave-one-out)")
    test_years = sorted({y for _, ts, _ in C.SIGNAL_FOLDS for y in ts})
    untestable = [c for c in cands
                  if all(df.loc[(df["is_test"] == 0) & (df["year"] == y), c].nunique(dropna=False) <= 1 for y in test_years)]
    red = redundancy(df, [c for c in cands if stats.loc[c, "tests_passed"] > 0], stats)
    redundant = {}
    for weak, strong, r, how in red:
        if weak not in redundant and strong not in redundant:
            redundant[weak] = (strong, r, how)
    for c in source_columns(df):
        if c in BASE:
            reg[c] = ("keep", "base", "exposure: offset and partial-year effect", None)
        elif c in rules:
            dec, why, ev_ = rules[c]
            reg[c] = (dec, why, ev_, None)
        elif c in redundant:
            strong, r, how = redundant[c]
            reg[c] = ("drop", "near-duplicate", f"{how} {r:+.3f} with {strong}, which predicts better", stats.loc[c])
        elif c in untestable:
            reg[c] = ("drop", "not testable", "constant within every test year (a calendar-year proxy); "
                      "the window length is known from renewal_count and hist_pol_exposure_3y", stats.loc[c])
        elif stats.loc[c, "tests_passed"] > 0:
            reg[c] = ("keep", "adds value", f"beats the random probes (z >= {stats.attrs['z_threshold']}) on: "
                      f"{stats.loc[c, 'passed_which']}", stats.loc[c])
        else:
            reg[c] = ("drop", "no value", f"no test beats the random probes at z >= {stats.attrs['z_threshold']}", stats.loc[c])
    recs = []
    for c, (dec, why, evid, st) in reg.items():
        r = {"line": line, "column": c, "decision": dec, "reason": why, "evidence": evid}
        if st is not None:
            r.update({k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
                      for k, v in st.items() if k not in ("passed_which",) and not k.endswith("_se")})
        recs.append(r)
    register = pd.DataFrame(recs)
    register["z_threshold"] = stats.attrs["z_threshold"]
    register["probe_false_passes"] = stats.attrs["probe_false_passes"]
    order = ["frequency_oneway", "severity_oneway", "frequency_model", "severity_model"] + \
            [f"{p}_oneway" for p in C.PERILS[line]]
    stat_cols = [c for t in order for c in (t, f"{t}_z") if c in register.columns]
    lead = ["line", "column", "decision", "reason", "evidence"]
    tail = ["tests_passed", "z_threshold", "probe_false_passes"]
    rest = [c for c in register.columns if c not in lead + stat_cols + tail]
    register = register[lead + stat_cols + rest + [c for c in tail if c in register.columns]]
    kept = [c for c, v in reg.items() if v[0] == "keep" and c not in BASE]
    return register, kept


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lines", nargs="*", default=list(C.LINES))
    ap.add_argument("--reuse-tests", action="store_true",
                    help="reuse reports/03_features/signal_tests_<line>.csv (same candidates) instead of re-running the tests")
    ap.add_argument("--z-threshold", type=float, default=None,
                    help="keep columns at this z (e.g. 1.0 keeps weaker evidence) instead of the calibrated threshold")
    a = ap.parse_args(argv)
    global REUSE_TESTS, FIXED_Z
    REUSE_TESTS, FIXED_Z = a.reuse_tests, a.z_threshold
    OUT.mkdir(parents=True, exist_ok=True)
    with H.Timer(f"{C.ROUND}: reference tables and claims"):
        ref = {t: H.read_table(t, "clean") for t in ("customers", "addresses", "weather")}
        claims = H.read_table("claims", "clean")
        all_rows = []
        for line in C.LINES:
            for kind in ("policies", "test"):
                r = H.read_table(f"{kind}_{line}", "clean", columns=["row_id", "policy_id", "customer_id",
                                                                    "address_id", "period_start", "exposure"])
                all_rows.append(r.assign(line=line))
        all_rows = pd.concat(all_rows, ignore_index=True)
        ev = claim_events(claims, all_rows)
        nonrep = claims.loc[claims["status"] != "repudiated"]
        cap = nonrep.groupby(["line", "peril"])["net_incurred"].transform(lambda s: s.quantile(C.LARGE_LOSS_QUANTILE))
        nonrep = nonrep.assign(w=np.minimum(nonrep["net_incurred"], cap))
        min_loss = f"{min(C.TRAIN_YEARS)}-01-01"
        patterns = {"cost": H.reporting_pattern(nonrep, C.VALUATION_DATE, min_loss, weight="w"),
                    "count": H.reporting_pattern(nonrep, C.VALUATION_DATE, min_loss)}
    registers, missing = [], []
    for line in a.lines:
        df = build_line(line, ref, ev, all_rows, patterns)
        register, kept = select(df, line, ev)
        miss = missing_flags(df, kept)
        flags = miss.loc[miss["informative"], "column"].tolist() if len(miss) else []
        for c in flags:
            df[f"{c}_missing"] = df[c].isna().astype("int8")
            register.loc[len(register)] = {"line": line, "column": f"{c}_missing", "decision": "keep",
                                           "reason": "informative missingness",
                                           "evidence": "claim frequency of rows missing it differs (see missing_values.csv)"}
        kept += [f"{c}_missing" for c in flags]
        registers.append(register)
        missing.append(miss.assign(line=line))
        tcols = ["n_claims", "claim_cost", "n_open"] + [f"{k}_{p}" for p in C.PERILS[line] for k in ("n", "cost")] + \
                ["comp_count", "comp_cost"] + [f"comp_{p}" for p in C.PERILS[line]]
        with H.Timer(f"{line}: writing data/features"):
            for kind, flag in (("policies", 0), ("test", 1)):
                cols = KEYS + BASE + kept + (tcols if flag == 0 else [])
                part = H.compact_dtypes(df.loc[df["is_test"] == flag, cols].reset_index(drop=True),
                                        keep=("exposure",))
                H.write_table(part, f"{kind}_{line}", "features")
                print(f"  {kind}_{line}: {len(part):,} rows x {part.shape[1]} columns "
                      f"({len(kept)} features)")
        print(register["decision"].value_counts().to_string())
        del df
    reg = pd.concat(registers, ignore_index=True)
    miss = pd.concat(missing, ignore_index=True)
    for name, new in (("feature_register.csv", reg), ("missing_values.csv", miss)):
        path = OUT / name
        if path.exists():               # keep the other lines' rows when only some lines are rebuilt
            old = pd.read_csv(path)
            new = pd.concat([old.loc[~old["line"].isin(a.lines)], new], ignore_index=True)
        order = {l: i for i, l in enumerate(C.LINES)}
        new = new.sort_values("line", key=lambda s: s.map(order), kind="stable")
        if name == "feature_register.csv":
            tests = ["frequency_oneway", "severity_oneway", "frequency_model", "severity_model"] + \
                    list(dict.fromkeys(f"{p}_oneway" for l in C.LINES for p in C.PERILS[l]))
            stat_cols = [t for t in tests if t in new] + [f"{t}_z" for t in tests if f"{t}_z" in new]
            lead = ["line", "column", "decision", "reason", "evidence"]
            tail = ["tests_passed", "z_threshold", "probe_false_passes"]
            new = new[lead + stat_cols + [c for c in new.columns if c not in lead + stat_cols + tail] + tail]
        new.to_csv(path, index=False)
    print(f"\nRegister -> {OUT / 'feature_register.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
