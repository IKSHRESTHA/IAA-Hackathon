"""Claims development: reporting delay (IBNR), open-claim development, trend.

Actuarial intent
----------------
Training targets are valued at the valuation date ``V`` (31 Dec 2024). Recent
rows look cheaper than they will be at ultimate for two reasons:

* **IBNR.** A claim with loss date ``L`` and reporting delay ``D`` is in the data
  only if ``L + D <= V`` (right truncation of ``D`` at ``T = V - L``). The delay
  CDF ``F`` is estimated per line x peril with the reverse-hazard product-limit
  estimator (Lynden-Bell 1971; Lagakos et al. 1988; the continuous-time
  chain-ladder), optionally with a parametric tail beyond the observable window.
  A row covering ``[s, e]`` has expected completeness ``mean_{t in [s, e]} F(V - t)``
  (losses assumed uniform over the row's period).
* **Development of reported open claims.** Expected future recoveries, the
  chance of later repudiation and case-estimate bias (IBNER) are applied
  claim by claim in :func:`open_claim_ultimate`.

Frequency / severity trend is estimated on the *developed* experience.

Assumptions are listed in each docstring and the evidence is in
``docs/reserving_analysis.md``.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, special

from .config import PERILS

log = logging.getLogger(__name__)

PARAMETRIC_FAMILIES: tuple[str, ...] = ("lognormal", "weibull", "lognormal_mix")
_N_PARAMS = {"lognormal": 2, "weibull": 2, "lognormal_mix": 5}
_METHODS = ("auto", "npmle", "npmle_tail", "parametric")


# =============================================================================
# 1. Reporting-delay pattern
# =============================================================================
def _default_min_loss_date(valuation_date: pd.Timestamp) -> pd.Timestamp:
    """First day of the four loss years ending at the valuation date."""
    return pd.Timestamp(year=valuation_date.year - 3, month=1, day=1)


def _delay_sample(
    claims: pd.DataFrame,
    valuation_date: pd.Timestamp,
    weight: str,
    cost_cap_quantile: float,
    cost_col: str,
    min_loss_date: pd.Timestamp,
) -> pd.DataFrame:
    """Observed delays ``d``, truncation points ``T = V - L`` and weights ``w``.

    Repudiated claims are excluded because the target excludes them. Claims with
    loss dates before ``min_loss_date`` are excluded: in this data they are a
    selected sample (Dec-2020 losses all have delays >= 37 days), which would
    bias the long-delay tail.
    """
    if weight not in ("count", "cost"):
        raise ValueError("weight must be 'count' or 'cost'")
    keep = (
        ~claims["is_repudiated"].astype(bool)
        & (claims["report_date"] <= valuation_date)
        & (claims["loss_date"] >= min_loss_date)
        & (claims["loss_date"] <= valuation_date)
    )
    cols = ["line", "peril", "loss_date", "report_date"] + ([cost_col] if weight == "cost" else [])
    c = claims.loc[keep, cols]
    out = pd.DataFrame({
        "line": c["line"].astype(str).to_numpy(),
        "peril": c["peril"].astype(str).to_numpy(),
        "d": (c["report_date"] - c["loss_date"]).dt.days.to_numpy(dtype=np.int64),
        "T": (valuation_date - c["loss_date"]).dt.days.to_numpy(dtype=np.int64),
    })
    if (out["d"] < 0).any():
        raise ValueError("report_date before loss_date on some claims")
    if weight == "count":
        out["w"] = 1.0
    else:
        cost = c[cost_col].to_numpy(dtype=float)
        cap = (
            pd.Series(cost).groupby([out["line"], out["peril"]])
            .transform(lambda s: s.quantile(cost_cap_quantile)).to_numpy()
        )
        out["w"] = np.minimum(np.maximum(cost, 0.0), cap)
    return out


def _npmle_cdf(
    d: np.ndarray, T: np.ndarray, w: np.ndarray, W: int,
    daily_until: int = 60, bucket_days: float = 30.4375,
) -> tuple[np.ndarray, np.ndarray]:
    """Reverse-hazard product-limit estimate of F(k) = P(D <= k | D <= W), k = 0..W.

    Discrete reverse hazard ``h_k = P(D = k | D <= k)``; risk set
    ``R_k = sum w_i [d_i <= k <= T_i]``. Lags up to ``daily_until`` use daily
    hazards; longer lags use a piecewise-constant hazard on ``bucket_days``-wide
    buckets, ``h_b = sum_{k in b} n_k / sum_{k in b} R_k`` (the MLE of a
    piecewise-constant reverse hazard), which avoids the instability of the
    daily estimator where the risk set is small (lags close to ``W``).
    ``F(k) = prod_{j > k} (1 - h_j)`` with ``F(W) = 1``. Returns ``(F, R)``.
    """
    if daily_until < 1:
        raise ValueError("daily_until must be >= 1 so that F(0) = 0 is preserved")
    d = np.asarray(d, dtype=np.int64)
    T = np.minimum(np.asarray(T, dtype=np.int64), W)
    w = np.asarray(w, dtype=float)
    n = np.bincount(d, weights=w, minlength=W + 1)[: W + 1]
    m = np.bincount(T, weights=w, minlength=W + 1)[: W + 1]
    A = np.cumsum(n)                                     # sum w [d <= k]
    B = np.concatenate([[0.0], np.cumsum(m)[:-1]])       # sum w [T < k]
    R = np.maximum(A - B, 0.0)                           # sum w [d <= k <= T]
    k = np.arange(W + 1)
    bucket = np.where(
        k <= daily_until, k,
        daily_until + 1 + np.floor((k - daily_until - 1) / bucket_days).astype(np.int64),
    )
    nb = np.bincount(bucket, weights=n)
    Rb = np.bincount(bucket, weights=R)
    hb = np.divide(nb, Rb, out=np.zeros_like(nb), where=Rb > 0)
    h = np.clip(hb[bucket], 0.0, 1.0)
    surv = np.cumprod((1.0 - h)[::-1])[::-1]              # prod_{j >= k} (1 - h_j)
    F = np.append(surv[1:], 1.0)                          # prod_{j > k}
    return np.maximum.accumulate(F), R


def _par_cdf_sf(family: str, theta: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CDF and survival function of a positive continuous delay distribution."""
    x = np.asarray(x, dtype=float)
    pos = x > 0
    lx = np.log(np.where(pos, x, 1.0))
    if family == "lognormal":
        z = (lx - theta[0]) / np.exp(theta[1])
        cdf, sf = special.ndtr(z), special.ndtr(-z)
    elif family == "weibull":
        hz = np.exp(np.exp(theta[1]) * (lx - theta[0]))   # (x / scale) ** shape
        cdf, sf = -np.expm1(-hz), np.exp(-hz)
    elif family == "lognormal_mix":
        p = special.expit(theta[0])
        z1 = (lx - theta[1]) / np.exp(theta[2])
        z2 = (lx - theta[3]) / np.exp(theta[4])
        cdf = p * special.ndtr(z1) + (1 - p) * special.ndtr(z2)
        sf = p * special.ndtr(-z1) + (1 - p) * special.ndtr(-z2)
    else:
        raise ValueError(f"unknown family {family!r}")
    return np.where(pos, cdf, 0.0), np.where(pos, sf, 1.0)


def _par_nll(theta, family, ud, nd, uT, mT) -> float:
    """Right-truncated, day-discretised negative log-likelihood.

    ``P(D = d) = F(d) - F(d - 1)`` (computed on the survival scale in the upper
    tail for precision), divided by ``P(D <= T) = F(T)``.
    """
    Fd, Sd = _par_cdf_sf(family, theta, ud)
    Fd1, Sd1 = _par_cdf_sf(family, theta, ud - 1)
    p = np.where(Fd > 0.5, Sd1 - Sd, Fd - Fd1)
    FT, _ = _par_cdf_sf(family, theta, uT)
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(FT))):
        return 1e300
    ll = nd @ np.log(np.maximum(p, 1e-300)) - mT @ np.log(np.maximum(FT, 1e-300))
    return float(-ll)


def _fit_parametric(d: np.ndarray, T: np.ndarray, w: np.ndarray, family: str) -> dict[str, Any]:
    """Weighted right-truncated MLE. Weights are rescaled to sum to the claim count."""
    w = np.asarray(w, dtype=float)
    pos = w > 0
    d, T, w = np.asarray(d)[pos], np.asarray(T)[pos], w[pos]
    w = w * (len(w) / w.sum())
    ud, inv = np.unique(d, return_inverse=True)
    nd = np.bincount(inv, weights=w)
    uT, invT = np.unique(T, return_inverse=True)
    mT = np.bincount(invT, weights=w)
    ld = np.log(d.astype(float))
    mu0 = float(np.average(ld, weights=w))
    s0 = float(np.sqrt(np.average((ld - mu0) ** 2, weights=w))) or 1.0
    if family == "lognormal":
        starts = [np.array([mu0, np.log(s0)])]
    elif family == "weibull":
        k0 = 1.28 / s0
        starts = [np.array([mu0 + 0.5772 / k0, np.log(k0)])]
    else:
        q99 = float(np.quantile(d, 0.99))
        core = ld[d <= q99]
        mc, sc = float(core.mean()), float(core.std()) or 1.0
        starts = [
            np.array([4.6, mc, np.log(sc), np.log(450.0), np.log(0.25)]),   # late-report bump
            np.array([2.2, mc, np.log(sc), mu0 + 1.5, np.log(s0)]),         # heavy second component
        ]
    best = None
    for x0 in starts:
        res = optimize.minimize(_par_nll, x0, args=(family, ud, nd, uT, mT), method="Nelder-Mead",
                                options={"maxiter": 20000, "maxfev": 20000, "xatol": 1e-7, "fatol": 1e-7})
        if best is None or res.fun < best.fun:
            best = res
    k = _N_PARAMS[family]
    return {"family": family, "theta": best.x.tolist(), "nll": float(best.fun),
            "aic": float(2 * best.fun + 2 * k), "converged": bool(best.success)}


def _resolve_method(method: str | Mapping[str, str], line: str, peril: str) -> str:
    if isinstance(method, Mapping):
        m = method.get(f"{line}/{peril}", method.get(peril, method.get("default", "auto")))
    else:
        m = method
    if m not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {m!r}")
    return m


def reporting_pattern(
    claims: pd.DataFrame,
    valuation_date: str | pd.Timestamp,
    weight: str = "count",
    max_days: int = 3650,
    cost_cap_quantile: float = 0.995,
    method: str | Mapping[str, str] = "auto",
    *,
    min_loss_date: str | pd.Timestamp | None = None,
    daily_until: int = 60,
    bucket_days: float = 30.4375,
    tail_min_mass: float = 0.002,
    tail_max_ks: float = 0.03,
    families: Iterable[str] = PARAMETRIC_FAMILIES,
    min_claims_per_peril: int = 150,
    cost_col: str = "net_incurred",
) -> pd.DataFrame:
    """Reporting-delay CDF by line x peril, corrected for right truncation.

    Parameters
    ----------
    claims : prepared claims (:func:`iaa_pricing.claims.prepare_claims`).
    weight : ``"count"`` (claim numbers) or ``"cost"`` (cost-weighted: each claim
        weighted by ``cost_col`` capped at the peril's ``cost_cap_quantile``, so the
        pattern is the share of ultimate cost reported by each lag).
    method : ``"npmle"`` (non-parametric, no mass beyond the observable window
        ``W = V - min_loss_date``); ``"npmle_tail"`` (non-parametric body rescaled by
        ``F_par(W)`` plus the best-AIC right-truncated parametric fit beyond ``W``);
        ``"parametric"``; or ``"auto"``, which adds the tail only when the best-AIC
        family fits the body (max |F_par(.|D<=W) - F_npmle| <= ``tail_max_ks``) and
        its mass beyond ``W`` is at least ``tail_min_mass``. A mapping
        ``{"line/peril" | peril | "default": method}`` sets methods per peril.
    min_loss_date : earliest loss date used (default: 1 Jan of ``V.year - 3``).

    Returns
    -------
    DataFrame ``line, peril, day, cdf`` for ``day = 0..max_days``; the CDF is
    non-decreasing, ``cdf(0) = 0`` when no delay is 0 and ``cdf(max_days) = 1``
    (no reporting assumed after ``max_days``). Fit diagnostics per peril are in
    ``.attrs["diagnostics"]`` (list of dicts).

    Assumptions: delays are independent of loss date within a peril (stationary
    reporting); perils with fewer than ``min_claims_per_peril`` claims use the
    pooled line pattern.
    """
    V = pd.Timestamp(valuation_date)
    mld = _default_min_loss_date(V) if min_loss_date is None else pd.Timestamp(min_loss_date)
    sample = _delay_sample(claims, V, weight, cost_cap_quantile, cost_col, mld)
    W = int((V - mld).days)
    if max_days < W:
        raise ValueError(f"max_days={max_days} is shorter than the observable window W={W}")
    families = tuple(families)
    day = np.arange(max_days + 1)
    frames, diags = [], []
    for (line, peril), g in sample.groupby(["line", "peril"], sort=True):
        n_claims = int(len(g))
        src = g
        pooled = n_claims < min_claims_per_peril
        if pooled:
            src = sample.loc[sample["line"] == line]
            log.warning("%s/%s: %d claims < %d; using pooled line pattern",
                        line, peril, n_claims, min_claims_per_peril)
        d, T, w = src["d"].to_numpy(), src["T"].to_numpy(), src["w"].to_numpy()
        F_np, R = _npmle_cdf(d, T, w, W, daily_until, bucket_days)
        meth = _resolve_method(method, line, peril)
        diag: dict[str, Any] = {
            "line": line, "peril": peril, "weight": weight, "n_claims": n_claims,
            "pooled_with_line": pooled, "window_days": W, "max_delay": int(d.max()),
            "npmle_cdf_30": float(F_np[30]), "npmle_cdf_90": float(F_np[90]),
            "npmle_cdf_365": float(F_np[365]), "npmle_cdf_730": float(F_np[min(730, W)]),
        }
        fits = []
        if meth != "npmle":
            for fam in families:
                fit = _fit_parametric(d, T, w, fam)
                Fp, _ = _par_cdf_sf(fam, np.asarray(fit["theta"]), day.astype(float))
                cond = Fp[: W + 1] / Fp[W]
                fit["ks_vs_npmle"] = float(np.max(np.abs(cond - F_np)))
                fit["tail_mass_beyond_window"] = float(1.0 - Fp[W] / Fp[max_days])
                fit["cdf"] = Fp
                fits.append(fit)
                diag[f"{fam}_aic"] = fit["aic"]
                diag[f"{fam}_ks"] = fit["ks_vs_npmle"]
                diag[f"{fam}_tail_mass"] = fit["tail_mass_beyond_window"]
        best = min(fits, key=lambda f: f["aic"]) if fits else None
        if meth == "auto":
            credible = best["ks_vs_npmle"] <= tail_max_ks
            material = best["tail_mass_beyond_window"] >= tail_min_mass
            meth = "npmle_tail" if (credible and material) else "npmle"
            if material and not credible:
                log.warning("%s/%s: parametric tail %.4f ignored (KS %.3f > %.3f)", line, peril,
                            best["tail_mass_beyond_window"], best["ks_vs_npmle"], tail_max_ks)
        if meth == "npmle":
            cdf = np.ones(max_days + 1)
            cdf[: W + 1] = F_np
        else:
            Fp = best["cdf"] / best["cdf"][max_days]
            if meth == "parametric":
                cdf = Fp
            else:  # npmle_tail
                cdf = np.empty(max_days + 1)
                cdf[: W + 1] = F_np * Fp[W]
                cdf[W + 1:] = Fp[W + 1:]
        cdf = np.clip(np.maximum.accumulate(cdf), 0.0, 1.0)
        cdf[-1] = 1.0
        diag.update({
            "method": meth, "best_family": best["family"] if best else None,
            "best_theta": best["theta"] if best else None,
            "cdf_at_window": float(cdf[W]),
            "cdf_365": float(cdf[365]), "cdf_730": float(cdf[730]),
        })
        diags.append(diag)
        frames.append(pd.DataFrame({"line": line, "peril": peril, "day": day, "cdf": cdf}))
    out = pd.concat(frames, ignore_index=True)
    out.attrs["diagnostics"] = diags
    return out


# =============================================================================
# 2. Row-level completeness
# =============================================================================
def _cum_cdf(cdf: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """``G(k) = sum_{j=0}^{k} F(j)`` with ``F = 0`` for ``j < 0`` and ``F = 1`` beyond the grid."""
    M = len(cdf) - 1
    G = np.concatenate([[0.0], np.cumsum(cdf)])           # G[k + 1] = sum_{j <= k}
    k = np.asarray(lags, dtype=np.int64)
    inside = G[np.clip(k, -1, M) + 1]
    return np.where(k < 0, 0.0, inside + np.maximum(k - M, 0).astype(float))


def row_completeness(
    rows: pd.DataFrame,
    pattern: pd.DataFrame,
    line: str,
    valuation_date: str | pd.Timestamp,
    *,
    min_value: float = 1e-6,
) -> pd.DataFrame:
    """Expected share of each row's ultimate claims reported by the valuation date.

    For a row covering ``[s, e]`` (inclusive) with losses uniform over the period,
    ``comp = (1 / n_days) * sum_{t=s}^{e} F(V - t)``, computed with a cumulative
    sum of ``F`` so 1.4M rows take well under a second. Days after ``V`` count as
    unreported. Count or cost completeness follows the ``pattern`` passed. Values
    are floored at ``min_value`` (a 1-day row on ``V`` has ``F(0) = 0``).

    Returns ``row_id`` plus one ``comp_<peril>`` column per peril of ``line``.
    """
    V = pd.Timestamp(valuation_date)
    pat = pattern.loc[pattern["line"] == line]
    if pat.empty:
        raise ValueError(f"pattern has no rows for line {line!r}")
    perils = [p for p in PERILS.get(line, ()) if p in set(pat["peril"])]
    missing = sorted(set(PERILS.get(line, ())) - set(perils))
    if missing:
        log.warning("%s: no reporting pattern for perils %s", line, missing)
    perils += sorted(set(pat["peril"]) - set(perils))
    lag_lo = (V - rows["period_end"]).dt.days.to_numpy(dtype=np.int64)   # latest day in period
    lag_hi = (V - rows["period_start"]).dt.days.to_numpy(dtype=np.int64)
    n_days = lag_hi - lag_lo + 1
    if (n_days <= 0).any():
        raise ValueError("rows with period_end before period_start")
    out = pd.DataFrame({"row_id": rows["row_id"].to_numpy()})
    for p in perils:
        cdf = pat.loc[pat["peril"] == p].sort_values("day")["cdf"].to_numpy(dtype=float)
        comp = (_cum_cdf(cdf, lag_hi) - _cum_cdf(cdf, lag_lo - 1)) / n_days
        out[f"comp_{p}"] = np.clip(comp, min_value, 1.0)
    return out


def peril_cost_shares(
    claims: pd.DataFrame,
    line: str,
    years: Iterable[int],
    *,
    cost_col: str | None = None,
    cap_quantile: float | None = None,
) -> dict[str, float]:
    """Share of ultimate cost by peril for ``line`` over (mature) loss ``years``.

    Uses ``net_ultimate`` if present (suspect amount corrected, expected recoveries
    deducted), else ``net_incurred`` with ``suspect_amount`` claims dropped.
    Optional per-peril cap at ``cap_quantile``. Mature years should be used so
    IBNR barely distorts the mix. Returns ``{peril: share}`` summing to 1.
    """
    years = list(years)
    c = claims.loc[(claims["line"] == line) & claims["loss_year"].isin(years) & ~claims["is_repudiated"]]
    col = cost_col or ("net_ultimate" if "net_ultimate" in c.columns else "net_incurred")
    if col == "net_incurred" and "suspect_amount" in c.columns:
        c = c.loc[~c["suspect_amount"].astype(bool)]
    cost = c[col].astype(float)
    if cap_quantile is not None:
        cap = cost.groupby(c["peril"]).transform(lambda s: s.quantile(cap_quantile))
        cost = np.minimum(cost, cap)
    tot = cost.groupby(c["peril"]).sum().reindex(list(PERILS[line]), fill_value=0.0)
    if tot.sum() <= 0:
        raise ValueError(f"no cost for {line} in years {years}")
    shares = tot / tot.sum()
    return {str(k): float(v) for k, v in shares.items()}


def line_completeness(row_comp: pd.DataFrame, peril_weights: Mapping[str, float]) -> pd.Series:
    """Weighted line-level completeness ``sum_p w_p comp_p`` (weights renormalised).

    With peril *cost* shares and the cost-weighted pattern this is the expected
    share of a row's ultimate cost reported by ``V`` (assumes the row's peril mix
    equals the portfolio mix). Aligned to ``row_comp``'s index.
    """
    w = {p: float(v) for p, v in peril_weights.items() if float(v) > 0}
    absent = [p for p in w if f"comp_{p}" not in row_comp.columns]
    if absent:
        raise KeyError(f"row_comp lacks completeness columns for perils {absent}")
    tot = sum(w.values())
    if tot <= 0:
        raise ValueError("peril weights must have a positive sum")
    out = np.zeros(len(row_comp))
    for p, v in w.items():
        out += (v / tot) * row_comp[f"comp_{p}"].to_numpy(dtype=float)
    return pd.Series(out, index=row_comp.index, name="comp_line")


# =============================================================================
# 3. Open-claim development to ultimate
# =============================================================================
def _nested(df: pd.DataFrame, value: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for r in df.itertuples(index=False):
        out.setdefault(str(r.line), {})[str(r.peril)] = float(getattr(r, value))
    return out


def _suspect_corrections(c: pd.DataFrame, suspect_ratio: float) -> dict[str, dict[str, Any]]:
    """Robust incurred for open claims flagged ``suspect_amount``.

    Primary: undo a decimal shift, i.e. the smallest ``k`` in 1..3 with
    ``paid <= incurred / 10**k <= suspect_ratio * largest other claim of the peril``.
    Alternative (reported for review): paid x median incurred/paid of open claims
    of the same peril and loss year.
    """
    out: dict[str, dict[str, Any]] = {}
    if "suspect_amount" not in c.columns:
        return out
    for r in c.loc[c["suspect_amount"].astype(bool) & c["is_open"]].itertuples(index=False):
        same = c.loc[(c["line"] == r.line) & (c["peril"] == r.peril) & (c["claim_id"] != r.claim_id)]
        largest_other = float(same["incurred_to_date"].max())
        peers = same.loc[same["is_open"] & (same["loss_year"] == r.loss_year) & (same["paid_to_date"] > 0)]
        ratio = float((peers["incurred_to_date"] / peers["paid_to_date"]).median()) if len(peers) else np.nan
        alt = float(r.paid_to_date * ratio) if np.isfinite(ratio) else None
        corrected, how = None, None
        for k in (1, 2, 3):
            v = r.incurred_to_date / 10 ** k
            if r.paid_to_date <= v <= suspect_ratio * largest_other:
                corrected, how = float(v), f"decimal_shift_div_10^{k}"
                break
        if corrected is None:
            corrected, how = (alt, "paid_x_median_incurred_to_paid") if alt else (
                float(min(r.incurred_to_date, largest_other)), "capped_at_largest_other")
        out[str(r.claim_id)] = {
            "line": str(r.line), "peril": str(r.peril), "loss_year": int(r.loss_year),
            "paid_to_date": float(r.paid_to_date), "incurred_reported": float(r.incurred_to_date),
            "incurred_corrected": corrected, "method": how,
            "largest_other_incurred": largest_other,
            "alt_paid_x_median_ratio": alt, "median_incurred_to_paid_peers": ratio,
            "needs_human_review": True,
        }
    return out


def _ibner_regression(cells: pd.DataFrame) -> dict[str, Any]:
    """WLS of log developed AY severity on peril effects, peril-specific linear
    trends and the open share of incurred: ``beta`` ~ proportional case-estimate
    bias of open claims (``incurred = (1 + beta) x ultimate``)."""
    cells = cells.loc[(cells["sev"] > 0) & (cells["n"] > 0)].copy()
    keys = (cells["line"] + "/" + cells["peril"]).to_numpy()
    uk = np.unique(keys)
    t = cells["loss_year"].to_numpy(float) - cells["loss_year"].mean()
    X = [(keys == k).astype(float) for k in uk] + [(keys == k) * t for k in uk]
    X.append(cells["open_share"].to_numpy(float))
    X = np.column_stack(X)
    y = np.log(cells["sev"].to_numpy(float))
    w = cells["n"].to_numpy(float)
    keep = np.linalg.matrix_rank(X) == X.shape[1]
    if not keep or len(y) <= X.shape[1]:
        return {"beta": None, "se": None, "z": None, "df": int(len(y) - X.shape[1]), "n_cells": int(len(y))}
    XtW = X.T * w
    cov = np.linalg.inv(XtW @ X)
    beta = cov @ XtW @ y
    resid = y - X @ beta
    df = len(y) - X.shape[1]
    sigma2 = float(np.sum(w * resid ** 2) / df)
    se = float(np.sqrt(cov[-1, -1] * sigma2))
    return {"beta": float(beta[-1]), "se": se, "z": float(beta[-1] / se), "df": int(df),
            "n_cells": int(len(y))}


def estimate_development_params(
    claims: pd.DataFrame,
    *,
    mature_years: Iterable[int] = (2021, 2022),
    immature_years: Iterable[int] = (2023, 2024),
    repudiation: str = "estimate",
    repudiation_min_z: float = 2.0,
    ibner_method: str = "none",
    ibner_min_z: float = 2.0,
    ay_completeness: pd.DataFrame | None = None,
    suspect_ratio: float = 5.0,
) -> dict[str, Any]:
    """Estimate the parameters used by :func:`open_claim_ultimate` (JSON-serialisable).

    * ``recovery_ratio[line][peril]`` = sum(recoveries) / sum(incurred) on closed
      claims of all loss years from ``min(mature_years)``. Recoveries are booked
      only at closure, so an open claim's expected future recovery is this ratio
      times its (gross) incurred. Diagnostics by loss year and size quintile.
    * ``repudiation_prob[line][peril]``: chance an open claim is later repudiated,
      ``max(0, sum_y (r_mature N_y - R_y)) / sum_y O_y`` over ``immature_years``
      (``r_mature`` = repudiations / reported claims in mature years), set to 0
      unless the shortfall is significant (``z >= repudiation_min_z``) or
      ``repudiation='none'``.
    * ``ibner_factor[line][peril]`` = ultimate / incurred for open claims: 1.0
      unless ``ibner_method='ay_severity_ratio'`` and the pooled AY-severity test
      (see :func:`_ibner_regression`) is significant at ``ibner_min_z``.
      ``ay_completeness`` (``line, peril, loss_year, comp_count, comp_cost``)
      converts reported AY severity to developed severity for the test.
    * ``suspect_corrections``: robust incurred for suspect open claims, flagged
      for human review.
    """
    mature_years, immature_years = list(mature_years), list(immature_years)
    c = claims.loc[claims["loss_year"] >= min(mature_years + immature_years)].copy()
    c["line"] = c["line"].astype(str)
    c["peril"] = c["peril"].astype(str)
    suspects = _suspect_corrections(c, suspect_ratio)
    gross = c["incurred_to_date"].astype(float).copy()
    for cid, s in suspects.items():
        gross[c["claim_id"] == cid] = s["incurred_corrected"]
    c["gross"] = gross

    # ---- recoveries -------------------------------------------------------
    cl = c.loc[c["status"] == "closed"].assign(has_rec=lambda x: x["recoveries"] > 0)
    rec = cl.groupby(["line", "peril"]).agg(
        n_closed=("claim_id", "size"), incidence=("has_rec", "mean"),
        recoveries=("recoveries", "sum"), gross=("incurred_to_date", "sum"),
    ).reset_index()
    rec["ratio"] = rec["recoveries"] / rec["gross"]
    given = (cl.loc[cl["has_rec"]].assign(r=lambda x: x["recoveries"] / x["incurred_to_date"])
             .groupby(["line", "peril"])["r"].mean().rename("mean_ratio_given_recovery"))
    rec = rec.merge(given.reset_index(), on=["line", "peril"], how="left").fillna(
        {"mean_ratio_given_recovery": 0.0})
    by_year = (cl.groupby(["line", "peril", "loss_year"])[["recoveries", "incurred_to_date"]].sum()
               .assign(ratio=lambda x: x["recoveries"] / x["incurred_to_date"]).reset_index())
    with_rec = rec.loc[rec["recoveries"] > 0, ["line", "peril"]]
    size_rows = []
    for r in with_rec.itertuples(index=False):
        s = cl.loc[(cl["line"] == r.line) & (cl["peril"] == r.peril)]
        q = pd.qcut(s["incurred_to_date"], 5, labels=False, duplicates="drop")
        g = s.groupby(q).agg(lo=("incurred_to_date", "min"), hi=("incurred_to_date", "max"),
                             incidence=("has_rec", "mean"), rec=("recoveries", "sum"),
                             gross=("incurred_to_date", "sum"))
        for qi, gg in g.iterrows():
            size_rows.append({"line": r.line, "peril": r.peril, "size_quintile": int(qi) + 1,
                              "lo": float(gg.lo), "hi": float(gg.hi), "incidence": float(gg.incidence),
                              "ratio": float(gg.rec / gg.gross)})

    # ---- repudiation ------------------------------------------------------
    cnt = c.groupby(["line", "peril", "loss_year"]).agg(
        n=("claim_id", "size"), rep=("is_repudiated", "sum"), open=("is_open", "sum"),
    ).reset_index()
    open_zero_paid = c.loc[c["is_open"]].groupby(["line", "peril"])["paid_to_date"].apply(
        lambda s: int((s <= 0).sum()))
    rep_rows = []
    for (line, peril), g in cnt.groupby(["line", "peril"]):
        m = g.loc[g["loss_year"].isin(mature_years)]
        im = g.loc[g["loss_year"].isin(immature_years)]
        r_m = float(m["rep"].sum() / m["n"].sum()) if m["n"].sum() else 0.0
        expected = float(r_m * im["n"].sum())
        observed = float(im["rep"].sum())
        n_open = float(im["open"].sum())
        var = r_m * (1 - r_m) * im["n"].sum() + (im["n"].sum() ** 2) * r_m * (1 - r_m) / max(m["n"].sum(), 1)
        z = (expected - observed) / np.sqrt(var) if var > 0 else 0.0
        p_raw = max(expected - observed, 0.0) / n_open if n_open else 0.0
        adopt = repudiation == "estimate" and z >= repudiation_min_z
        rep_rows.append({
            "line": line, "peril": peril, "mature_rate": r_m,
            "immature_rate": observed / im["n"].sum() if im["n"].sum() else np.nan,
            "expected_immature": expected, "observed_immature": observed, "open_immature": n_open,
            "z_shortfall": float(z), "p_raw": float(min(p_raw, 1.0)),
            "p_adopted": float(min(p_raw, 1.0)) if adopt else 0.0,
            "open_with_zero_paid": int(open_zero_paid.get((line, peril), 0)),
        })
    rep_df = pd.DataFrame(rep_rows)

    # ---- IBNER ------------------------------------------------------------
    nr = c.loc[~c["is_repudiated"]]
    cells = nr.groupby(["line", "peril", "loss_year"]).agg(
        n=("claim_id", "size"), n_open=("is_open", "sum"), gross=("gross", "sum"),
    ).reset_index()
    open_gross = nr.loc[nr["is_open"]].groupby(["line", "peril", "loss_year"])["gross"].sum()
    cells = cells.merge(open_gross.rename("open_gross").reset_index(),
                        on=["line", "peril", "loss_year"], how="left").fillna({"open_gross": 0.0})
    sev_closed = nr.loc[nr["status"] == "closed"].groupby(["line", "peril", "loss_year"])["gross"].mean()
    sev_open = nr.loc[nr["is_open"]].groupby(["line", "peril", "loss_year"])["gross"].mean()
    cells = cells.merge(sev_closed.rename("sev_closed").reset_index(), on=["line", "peril", "loss_year"],
                        how="left").merge(sev_open.rename("sev_open").reset_index(),
                                          on=["line", "peril", "loss_year"], how="left")
    cells["open_share"] = cells["open_gross"] / cells["gross"]
    cells["sev_reported"] = cells["gross"] / cells["n"]
    cells["ibnr_adj"] = 1.0
    if ay_completeness is not None:
        a = ay_completeness[["line", "peril", "loss_year", "comp_count", "comp_cost"]]
        cells = cells.merge(a, on=["line", "peril", "loss_year"], how="left")
        cells["ibnr_adj"] = (cells["comp_count"] / cells["comp_cost"]).fillna(1.0)
    cells["sev"] = cells["sev_reported"] * cells["ibnr_adj"]
    test = _ibner_regression(cells.loc[cells["loss_year"].isin(mature_years + immature_years)])
    test["developed"] = ay_completeness is not None
    factor = 1.0
    if ibner_method == "ay_severity_ratio" and test["z"] is not None and abs(test["z"]) >= ibner_min_z:
        factor = float(1.0 / (1.0 + test["beta"]))
    elif ibner_method not in ("none", "ay_severity_ratio"):
        raise ValueError("ibner_method must be 'none' or 'ay_severity_ratio'")
    test["factor_adopted"] = factor
    ibner = {line: {p: factor for p in PERILS.get(line, ())} for line in sorted(c["line"].unique())}

    def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
        return [{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}
                for r in df.replace({np.nan: None}).to_dict("records")]

    return {
        "recovery_ratio": _nested(rec, "ratio"),
        "repudiation_prob": _nested(rep_df, "p_adopted"),
        "ibner_factor": ibner,
        "suspect_corrections": suspects,
        "settings": {"mature_years": mature_years, "immature_years": immature_years,
                     "repudiation": repudiation, "repudiation_min_z": repudiation_min_z,
                     "ibner_method": ibner_method, "ibner_min_z": ibner_min_z},
        "diagnostics": {
            "recovery": _records(rec),
            "recovery_by_year": _records(by_year),
            "recovery_by_size": size_rows,
            "repudiation": _records(rep_df),
            "ibner_cells": _records(cells),
            "ibner_test": test,
        },
    }


def open_claim_ultimate(claims: pd.DataFrame, params: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Add ``net_ultimate`` (expected ultimate net cost) and ``ultimate_adjustment``.

    * closed: ``net_ultimate = net_incurred`` (unchanged);
    * repudiated: 0;
    * open: ``gross = incurred`` (suspect amounts replaced by
      ``params['suspect_corrections']``), then
      ``net_ultimate = (1 - p_repud) * max(gross * ibner * (1 - recovery_ratio), 0)``.

    ``ultimate_adjustment = net_ultimate - net_incurred``; ``ultimate_basis``
    records which rule was applied (audit trail). Perils without a parameter use
    ratio 0, probability 0 and factor 1.
    """
    if params is None:
        params = estimate_development_params(claims)
    c = claims.copy()
    line, peril = c["line"].astype(str), c["peril"].astype(str)

    def lookup(key: str, default: float) -> np.ndarray:
        table = params.get(key, {}) or {}
        return np.array([float(table.get(l, {}).get(p, default)) for l, p in zip(line, peril)])

    rec = lookup("recovery_ratio", 0.0)
    rep = lookup("repudiation_prob", 0.0)
    ibn = lookup("ibner_factor", 1.0)
    gross = c["incurred_to_date"].to_numpy(dtype=float).copy()
    corrected = np.zeros(len(c), dtype=bool)
    for cid, s in (params.get("suspect_corrections", {}) or {}).items():
        hit = (c["claim_id"] == cid).to_numpy() & c["is_open"].to_numpy(dtype=bool)
        gross[hit] = float(s["incurred_corrected"])
        corrected |= hit
    is_open = c["is_open"].to_numpy(dtype=bool)
    is_rep = c["is_repudiated"].to_numpy(dtype=bool)
    open_ult = (1.0 - rep) * np.maximum(gross * ibn * (1.0 - rec), 0.0)
    net = c["net_incurred"].to_numpy(dtype=float)
    c["net_ultimate"] = np.where(is_rep, 0.0, np.where(is_open, open_ult, net))
    c["ultimate_adjustment"] = c["net_ultimate"] - net
    c["ultimate_basis"] = np.select(
        [is_rep, is_open & corrected, is_open], ["repudiated", "open_suspect_corrected", "open_developed"],
        default="closed_as_reported")
    return c


# =============================================================================
# 4. Trend
# =============================================================================
def _loglinear_wls(t: np.ndarray, y: np.ndarray, var: np.ndarray) -> tuple[float, float, float, float]:
    """Fit ``y = a + b t`` by WLS (weights ``1/var``). Returns ``b``, quasi SE
    (sampling SE inflated by ``sqrt(max(1, dispersion))``), sampling SE, dispersion."""
    w = 1.0 / var
    X = np.column_stack([np.ones_like(t), t - np.average(t, weights=w)])
    XtW = X.T * w
    cov = np.linalg.inv(XtW @ X)
    beta = cov @ XtW @ y
    df = len(y) - 2
    phi = float(np.sum(w * (y - X @ beta) ** 2) / df) if df > 0 else np.nan
    se_s = float(np.sqrt(cov[1, 1]))
    se = se_s * np.sqrt(max(1.0, phi)) if df > 0 else se_s
    return float(beta[1]), float(se), se_s, phi


def estimate_trend(
    rows_with_claims: pd.DataFrame,
    *,
    perils: Mapping[str, Iterable[str]] | None = None,
    line_col: str = "line",
    year_col: str = "year",
    exposure_col: str = "exposure",
    years: Iterable[int] | None = None,
    min_claims: int = 30,
) -> pd.DataFrame:
    """Annual log-linear frequency, severity and pure-premium trend on developed experience.

    ``rows_with_claims`` has one record per policy row with ``line``, ``year``,
    ``exposure`` and, per peril, reported claim counts ``n_<peril>``, developed
    costs of reported claims ``cost_<peril>`` (e.g. capped ``net_ultimate``) and
    optionally count / cost completeness ``compn_<peril>`` / ``compc_<peril>``
    (default 1). Year aggregates are developed for IBNR by the ratio estimator
    ``freq_y = sum n / sum(exposure * compn)`` and ``pp_y = sum cost / sum(exposure * compc)``;
    ``sev_y = pp_y / freq_y``. The line (``peril = "all"``) adds developed perils.

    Each series is fitted by WLS on ``log`` with sampling variances ``1/N``
    (frequency), ``CV^2/N`` (severity) and ``(1 + CV^2)/N`` (pure premium), where
    ``CV`` is the claim-level coefficient of variation from rows with exactly one
    claim. ``se`` is the quasi-likelihood SE (sampling SE inflated when the
    between-year scatter exceeds sampling noise, e.g. weather years); with 4
    years it rests on 2 degrees of freedom. ``annual_rate = exp(b) - 1``.
    Year-level developed experience is in ``.attrs["experience"]`` (list of records).
    """
    df = rows_with_claims
    if years is not None:
        df = df.loc[df[year_col].isin(list(years))]
    results, exp_rows = [], []
    for line, g in df.groupby(line_col, sort=True):
        line = str(line)
        pl = list(perils[line]) if perils is not None else list(PERILS.get(line, ()))
        pl = [p for p in pl if f"n_{p}" in g.columns and f"cost_{p}" in g.columns]
        e = g[exposure_col].to_numpy(dtype=float)
        yr = g[year_col].to_numpy()
        uy = np.unique(yr)
        E = np.bincount(np.searchsorted(uy, yr), weights=e, minlength=len(uy))
        series: dict[str, dict[str, np.ndarray]] = {}
        cv2: dict[str, float] = {}
        tot = {k: np.zeros(len(uy)) for k in ("N", "devN", "C", "devC")}
        idx = np.searchsorted(uy, yr)
        single_costs = []
        for p in pl:
            n = g[f"n_{p}"].to_numpy(dtype=float)
            cost = g[f"cost_{p}"].to_numpy(dtype=float)
            compn = g[f"compn_{p}"].to_numpy(dtype=float) if f"compn_{p}" in g.columns else np.ones(len(g))
            compc = g[f"compc_{p}"].to_numpy(dtype=float) if f"compc_{p}" in g.columns else np.ones(len(g))
            N = np.bincount(idx, weights=n, minlength=len(uy))
            C = np.bincount(idx, weights=cost, minlength=len(uy))
            En = np.bincount(idx, weights=e * compn, minlength=len(uy))
            Ec = np.bincount(idx, weights=e * compc, minlength=len(uy))
            devN, devC = N * E / En, C * E / Ec
            one = cost[n == 1]
            single_costs.append(one)
            cv2[p] = float(np.var(one) / np.mean(one) ** 2) if len(one) > 30 and np.mean(one) > 0 else 1.0
            series[p] = {"N": N, "devN": devN, "devC": devC}
            for k, v in (("N", N), ("devN", devN), ("C", C), ("devC", devC)):
                tot[k] += v
            for i, y in enumerate(uy):
                exp_rows.append({"line": line, "peril": p, "year": int(y), "exposure": E[i],
                                 "n_reported": N[i], "n_developed": devN[i], "cost_reported": C[i],
                                 "cost_developed": devC[i], "comp_count": En[i] / E[i],
                                 "comp_cost": Ec[i] / E[i]})
        allc = np.concatenate(single_costs) if single_costs else np.array([])
        cv2["all"] = float(np.var(allc) / np.mean(allc) ** 2) if len(allc) > 30 else 1.0
        series["all"] = {"N": tot["N"], "devN": tot["devN"], "devC": tot["devC"]}
        for i, y in enumerate(uy):
            exp_rows.append({"line": line, "peril": "all", "year": int(y), "exposure": E[i],
                             "n_reported": tot["N"][i], "n_developed": tot["devN"][i],
                             "cost_reported": tot["C"][i], "cost_developed": tot["devC"][i],
                             "comp_count": tot["N"][i] / tot["devN"][i] if tot["devN"][i] else np.nan,
                             "comp_cost": tot["C"][i] / tot["devC"][i] if tot["devC"][i] else np.nan})
        for p, s in series.items():
            ok = (s["N"] > 0) & (s["devC"] > 0) & (E > 0)
            if ok.sum() < 3 or s["N"].sum() < min_claims:
                continue
            t = uy[ok].astype(float)
            N, devN, devC, Ey = s["N"][ok], s["devN"][ok], s["devC"][ok], E[ok]
            comps = {
                "frequency": (np.log(devN / Ey), 1.0 / N),
                "severity": (np.log(devC / devN), cv2[p] / N),
                "pure_premium": (np.log(devC / Ey), (1.0 + cv2[p]) / N),
            }
            for comp, (y_, v_) in comps.items():
                b, se_b, se_s, phi = _loglinear_wls(t, y_, v_)
                results.append({
                    "line": line, "peril": p, "component": comp,
                    "annual_rate": float(np.expm1(b)), "se": float(np.exp(b) * se_b),
                    "n_years": int(ok.sum()), "se_sampling": float(np.exp(b) * se_s),
                    "dispersion": phi, "n_claims": float(N.sum()),
                    "first_year": int(t.min()), "last_year": int(t.max()),
                })
    out = pd.DataFrame(results, columns=["line", "peril", "component", "annual_rate", "se", "n_years",
                                         "se_sampling", "dispersion", "n_claims", "first_year", "last_year"])
    out.attrs["experience"] = [{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}
                               for r in exp_rows]
    return out


def trend_factor(years: Any, target_year: int, annual_rate: Any) -> np.ndarray:
    """Multiplicative trend ``(1 + r) ** (target_year - year)`` (broadcasts)."""
    r = np.asarray(annual_rate, dtype=float)
    if np.any(r <= -1.0):
        raise ValueError("annual_rate must exceed -100%")
    return np.power(1.0 + r, float(target_year) - np.asarray(years, dtype=float))
