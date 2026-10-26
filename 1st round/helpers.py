"""Helpers shared by the three steps of this round.

* typed reading and writing of the case-study tables (identifiers as text, real dates, booleans);
* :class:`ChangeLog`, the audit trail every cleaning rule and feature decision writes to;
* :func:`profile`, a per-column profile;
* :func:`match_claims_to_rows`, which attaches each claim to the policy row on cover at the loss;
* :func:`reporting_pattern` and :func:`row_completeness`: the share of a row's claims already
  reported at the valuation date (claims of the last months are still coming in);
* :func:`oneway_gain` and :func:`permutation_gain`: out-of-time tests of a feature's value.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config as C

LAYERS = {"raw": C.RAW, "clean": C.CLEAN, "features": C.FEATURES}


# =================================================================================================
# Reading and writing
# =================================================================================================
def table_path(name: str, layer: str = "raw"):
    return LAYERS[layer] / f"{name}.parquet"


def read_table(name: str, layer: str = "raw", columns: list[str] | None = None) -> pd.DataFrame:
    """Read a table; raw tables get proper types (ids as text, dates, nullable booleans)."""
    df = pd.read_parquet(table_path(name, layer), columns=columns)
    return apply_types(df) if layer == "raw" else df


def _text_id(s: pd.Series) -> pd.Series:
    if pd.api.types.is_float_dtype(s):          # 123.0 must become "123", not "123.0"
        s = s.astype("Int64")
    return s.astype("string").str.strip()


def _is_bool_like(s: pd.Series) -> bool:
    vals = pd.unique(s.dropna())
    return len(vals) > 0 and all(isinstance(v, (bool, np.bool_)) for v in vals)


def apply_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if c in C.ID_COLUMNS:
            df[c] = _text_id(df[c])
        elif c in C.DATE_COLUMNS:
            df[c] = pd.to_datetime(df[c])
        elif pd.api.types.is_bool_dtype(df[c]) or (df[c].dtype == object and _is_bool_like(df[c])):
            df[c] = df[c].astype("boolean")       # True / False / <NA>
        elif df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype("string")
    return df


def write_table(df: pd.DataFrame, name: str, layer: str) -> None:
    """Parquet with strong zstd compression (about 40% smaller than the default, as fast to read;
    slower to write, which happens once)."""
    path = table_path(name, layer)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd", compression_level=19)


def compact_dtypes(df: pd.DataFrame, keep: tuple[str, ...] = ()) -> pd.DataFrame:
    """Smaller numeric types without losing information: float64 -> float32 (7 significant
    digits: enough for every measure here), whole numbers -> the smallest integer type."""
    out = {}
    for c in df.columns:
        s = df[c]
        if c in keep:
            out[c] = s
        elif pd.api.types.is_float_dtype(s) and s.dtype == np.float64:
            out[c] = s.astype("float32")
        elif pd.api.types.is_integer_dtype(s) and not pd.api.types.is_extension_array_dtype(s):
            out[c] = pd.to_numeric(s, downcast="integer")
        else:
            out[c] = s
    return pd.DataFrame(out, index=df.index)


def stable_hash(values: pd.Series, salt: str = "") -> np.ndarray:
    """Deterministic 0..9999 bucket of each value (same on every machine and run)."""
    u = pd.unique(values.astype(str))
    table = {v: int(hashlib.md5(f"{salt}:{v}".encode()).hexdigest()[:8], 16) % 10_000 for v in u}
    return values.astype(str).map(table).to_numpy()


# =================================================================================================
# Audit trail
# =================================================================================================
@dataclass
class ChangeLog:
    """Every rule applied to the data: what, where, how many rows, and the evidence."""

    step: str
    rows: list[dict] = field(default_factory=list)

    def add(self, rule_id: str, table: str, column: str, rule: str, n_rows: int, action: str,
            evidence: str = "") -> None:
        self.rows.append({"step": self.step, "rule_id": rule_id, "table": table, "column": column,
                          "rule": rule, "rows_affected": int(n_rows), "action": action,
                          "evidence": evidence})
        print(f"  [{rule_id}] {table}.{column}: {rule} -> {n_rows:,} rows ({action})")

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["step", "rule_id", "table", "column", "rule",
                                                "rows_affected", "action", "evidence"])

    def save(self, path) -> pd.DataFrame:
        f = self.frame()
        path.parent.mkdir(parents=True, exist_ok=True)
        f.to_csv(path, index=False)
        return f


class Timer:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        print(f"\n== {self.label}")
        return self

    def __exit__(self, *exc):
        print(f"   ({time.time() - self.t0:.1f}s)")


# =================================================================================================
# Profiling
# =================================================================================================
def profile(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """One row per column: type, missing, distinct values, range and most frequent values."""
    recs = []
    for c in df.columns:
        s = df[c]
        r = {"table": table, "column": c, "dtype": str(s.dtype), "rows": len(s),
             "missing": int(s.isna().sum()), "missing_pct": round(100 * s.isna().mean(), 3),
             "distinct": int(s.nunique(dropna=True))}
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            x = s.dropna().astype(float)
            if len(x):
                q = x.quantile([0.0, 0.01, 0.5, 0.99, 1.0]).to_numpy()
                r.update({"min": q[0], "p01": q[1], "median": q[2], "p99": q[3], "max": q[4],
                          "mean": float(x.mean()), "zeros_pct": round(100 * float((x == 0).mean()), 3)})
        elif pd.api.types.is_datetime64_any_dtype(s):
            r.update({"min": str(s.min().date()) if s.notna().any() else None,
                      "max": str(s.max().date()) if s.notna().any() else None})
        top = s.astype("string").value_counts(dropna=True).head(6)
        r["top_values"] = "; ".join(f"{k} ({v:,})" for k, v in top.items())
        recs.append(r)
    return pd.DataFrame(recs)


# =================================================================================================
# Claims and policy rows
# =================================================================================================
def match_claims_to_rows(claims: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """``row_id`` of the policy row on cover at each claim's loss date, and a match status.

    A policy keeps its ``policy_id`` across years and mid-term changes, so a claim belongs to the
    row of the same policy whose period contains the loss date (both ends inclusive). Status:
    ``matched``; ``before_first_row`` / ``after_last_row`` (the loss falls outside every period
    of a known policy); ``unknown_policy``.
    """
    keys = rows[["row_id", "policy_id", "period_start", "period_end"]]
    cand = claims[["claim_id", "policy_id", "loss_date"]].merge(keys, on="policy_id", how="inner")
    inside = (cand["loss_date"] >= cand["period_start"]) & (cand["loss_date"] <= cand["period_end"])
    hit = cand.loc[inside].sort_values(["claim_id", "period_start"]).drop_duplicates("claim_id", keep="last")
    out = claims.merge(hit[["claim_id", "row_id"]], on="claim_id", how="left")
    span = rows.groupby("policy_id").agg(first=("period_start", "min"), last=("period_end", "max"))
    first = out["policy_id"].map(span["first"])
    last = out["policy_id"].map(span["last"])
    out["match_status"] = np.select(
        [out["row_id"].notna(), first.isna(), out["loss_date"] < first, out["loss_date"] > last],
        ["matched", "unknown_policy", "before_first_row", "after_last_row"], default="between_rows")
    return out


# =================================================================================================
# Reporting completeness (claims of the last months are not all reported yet)
# =================================================================================================
def reporting_pattern(claims: pd.DataFrame, valuation_date: str, min_loss_date: str,
                      weight: str | None = None, min_claims: int = 50) -> dict[tuple[str, str], np.ndarray]:
    """Share of claims reported within ``d`` days of the loss, F(d), per line and peril.

    A claim is in the data only if it was reported by the valuation date V, so a loss on day t is
    seen only with a delay d <= V - t (right truncation). The reverse-hazard (Lynden-Bell)
    estimator corrects for this: h(d) = (claims reported at delay d) / (claims with delay <= d
    and window >= d), F(d - 1) = F(d) (1 - h(d)), F(W) = 1 at the longest observable delay W.
    ``weight`` (a column) gives a cost-weighted pattern. A peril with fewer than ``min_claims``
    claims uses its line's pooled pattern.
    """
    V = pd.Timestamp(valuation_date)
    c = claims.loc[(claims["loss_date"] >= pd.Timestamp(min_loss_date)) & (claims["report_date"] <= V)]
    W = int((V - pd.Timestamp(min_loss_date)).days)

    def estimate(g: pd.DataFrame) -> np.ndarray:
        d = (g["report_date"] - g["loss_date"]).dt.days.clip(0, W).to_numpy()
        T = (V - g["loss_date"]).dt.days.clip(0, W).to_numpy()
        w = np.ones(len(g)) if weight is None else g[weight].to_numpy(float)
        at = np.bincount(d, weights=w, minlength=W + 1)
        cum_d = np.cumsum(at)                                        # weight with delay <= k
        t_at = np.bincount(T, weights=w, minlength=W + 1)
        before = np.cumsum(t_at) - t_at                              # weight with window < k
        risk = cum_d - before
        h = np.divide(at, risk, out=np.zeros(W + 1), where=risk > 0)
        log1m = np.log(np.clip(1.0 - h, 1e-12, 1.0))
        # F(k) = prod_{j > k} (1 - h(j))
        tail = np.concatenate([np.cumsum(log1m[::-1])[::-1][1:], [0.0]])
        return np.exp(tail)

    out = {}
    for line, gl in c.groupby("line"):
        pooled = estimate(gl)
        for peril, gp in gl.groupby("peril"):
            out[(line, peril)] = estimate(gp) if len(gp) >= min_claims else pooled
    return out


def row_completeness(rows: pd.DataFrame, F: np.ndarray, valuation_date: str) -> np.ndarray:
    """Average of F(V - t) over the row's days t (0 for days after V): share already reported."""
    V = pd.Timestamp(valuation_date)
    M = len(F) - 1
    G = np.concatenate([[0.0], np.cumsum(F)])                          # G[k + 1] = sum_{j <= k} F(j)

    def cum(k):
        k = np.asarray(k, dtype=np.int64)
        inside = G[np.clip(k, -1, M) + 1]
        return np.where(k < 0, 0.0, inside + np.maximum(k - M, 0))   # F = 1 beyond the window

    hi = (V - rows["period_start"]).dt.days.to_numpy()
    lo = (V - rows["period_end"]).dt.days.to_numpy()
    n = hi - lo + 1
    return np.clip((cum(hi) - cum(lo - 1)) / n, 0.0, 1.0)


# =================================================================================================
# Out-of-time value of a feature
# =================================================================================================
def tweedie_deviance(y: np.ndarray, mu: np.ndarray, p: float) -> np.ndarray:
    """Unit deviance: Poisson (p = 1), gamma (p = 2) or compound Poisson-gamma (1 < p < 2)."""
    y = np.asarray(y, float)
    mu = np.maximum(np.asarray(mu, float), 1e-12)
    if p == 1.0:
        return 2 * (np.where(y > 0, y * np.log(np.maximum(y, 1e-12) / mu), 0.0) - (y - mu))
    if p == 2.0:
        return 2 * (-np.log(np.maximum(y, 1e-12) / mu) + (y - mu) / mu)
    return 2 * (np.power(y, 2 - p) / ((1 - p) * (2 - p)) - y * np.power(mu, 1 - p) / (1 - p)
                + np.power(mu, 2 - p) / (2 - p))


def binned(x: pd.Series, fit: np.ndarray, n_bins: int = 10) -> pd.Series:
    """Levels of a feature for one-way tables, fixed on the ``fit`` rows.

    Numbers with more than ``n_bins`` distinct values are cut at quantiles; a value holding at
    least 20% of the rows (e.g. the zeros of a claims-history count) keeps its own level, so
    zero-inflated columns are not swallowed by one bin. Categories, flags and missing values
    (level ``missing``) are kept as they are.
    """
    if (pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x)
            and x.nunique() > n_bins):
        v = x.astype("Float64").to_numpy(dtype=float, na_value=np.nan)
        vf = v[fit]
        vf = vf[~np.isnan(vf)]
        lab = np.full(len(v), "missing", dtype=object)
        ok = ~np.isnan(v)
        if len(vf) == 0:
            lab[ok] = "all"
            return pd.Series(lab, index=x.index)
        vals, counts = np.unique(vf, return_counts=True)
        spike = vals[np.argmax(counts)] if counts.max() >= 0.2 * len(vf) else None
        rest = vf if spike is None else vf[vf != spike]
        edges = np.unique(np.quantile(rest, np.linspace(0, 1, n_bins + 1)[1:-1])) if len(rest) else np.array([])
        lab[ok] = np.char.add("b", np.searchsorted(edges, v[ok], side="left").astype(str))
        if spike is not None:
            lab[ok & (v == spike)] = f"={spike:g}"
        return pd.Series(lab, index=x.index)
    return x.astype("string").fillna("missing").astype(str)


def oneway_deviances(level: pd.Series, y: np.ndarray, expected: np.ndarray, fit: np.ndarray,
                     test: np.ndarray, power: float, k: float) -> tuple[np.ndarray, np.ndarray]:
    """Test-year deviance of each row without and with a credibility-smoothed one-way relativity.

    Relativities ``(Y_l + k) / (E_l + k)`` are fitted on the ``fit`` rows (``Y_l`` the level's
    actual, ``E_l`` its expected under no rating, scaled to the fit total; ``k`` in units of the
    target, so a level with little data stays near 1). On the test rows both predictions are
    scaled to the actual test total (level known): the comparison measures ranking only.
    """
    e_fit = expected[fit] * y[fit].sum() / expected[fit].sum()
    tab = pd.DataFrame({"l": level[fit].to_numpy(), "y": y[fit], "e": e_fit}).groupby("l").sum()
    rel = (tab["y"] + k) / (tab["e"] + k)
    r = level[test].map(rel).fillna(1.0).to_numpy()
    mu0 = expected[test] * y[test].sum() / expected[test].sum()
    mu1 = mu0 * r
    mu1 *= y[test].sum() / mu1.sum()
    return tweedie_deviance(y[test], mu0, power), tweedie_deviance(y[test], mu1, power)


def _mix64(x: np.ndarray) -> np.ndarray:
    """SplitMix64 finaliser: a fast, well-spread 64-bit hash of unsigned integers."""
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


_POISSON1_CDF = np.cumsum([np.exp(-1.0) / np.prod(np.arange(1, k + 1)) for k in range(12)])


class ClusterBootstrap:
    """Customer-clustered bootstrap that can be accumulated over several test years.

    Each customer gets, for each replicate ``b``, a Poisson(1) weight computed from a hash of
    (customer, b): the same customer has the same weight in every fold, so sums over folds stay
    a valid bootstrap of the pooled statistic.
    """

    def __init__(self, n_boot: int = 100, seed: int = 0):
        self.B = n_boot
        self.salt = np.uint64(seed * 7919 + 17)

    def weights(self, clusters: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        codes, uniq = pd.factorize(clusters.astype(str))
        h = np.array([int(hashlib.md5(u.encode()).hexdigest()[:16], 16) for u in uniq], dtype=np.uint64)
        b = np.arange(self.B, dtype=np.uint64)[:, None]
        with np.errstate(over="ignore"):
            r = _mix64(h[None, :] ^ _mix64(b * np.uint64(0x9E3779B97F4A7C15) + self.salt))
        u = (r >> np.uint64(11)).astype(np.float64) / float(1 << 53)
        return codes, np.searchsorted(_POISSON1_CDF, u).astype(np.float32)

    @staticmethod
    def replicate_sums(codes: np.ndarray, W: np.ndarray, values: np.ndarray) -> np.ndarray:
        s = np.bincount(codes, weights=values, minlength=W.shape[1]).astype(np.float32)
        return (W @ s).astype(np.float64)
