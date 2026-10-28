"""Scoring and diagnostic metrics.

The competition metric is Tweedie D^2 with power 1.5:

    D^2 = 1 - sum_i d(y_i, mu_i) / sum_i d(y_i, b_i)

where d is the Tweedie unit deviance and b the baseline prediction. With
p = 1.5 the unit deviance simplifies to ``4 (sqrt(mu) - sqrt(y))^2 / sqrt(mu)``,
which shows why under-predicting a row that has a large claim is expensive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ArrayLike = np.ndarray | pd.Series

_EPS = 1e-12


def _arr(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=float)


def tweedie_unit_deviance(y: ArrayLike, mu: ArrayLike, power: float = 1.5) -> np.ndarray:
    """Unit deviance d(y, mu) for a Tweedie distribution with 1 < power < 2."""
    y, mu = _arr(y), _arr(mu)
    if not 1.0 < power < 2.0:
        raise ValueError("power must lie in (1, 2)")
    if np.any(y < 0):
        raise ValueError("y must be non-negative")
    if np.any(~np.isfinite(mu)) or np.any(mu <= 0):
        raise ValueError("mu must be finite and strictly positive")
    p = power
    return 2.0 * (
        np.power(y, 2 - p) / ((1 - p) * (2 - p))
        - y * np.power(mu, 1 - p) / (1 - p)
        + np.power(mu, 2 - p) / (2 - p)
    )


def mean_tweedie_deviance(
    y: ArrayLike, mu: ArrayLike, power: float = 1.5, weight: ArrayLike | None = None
) -> float:
    d = tweedie_unit_deviance(y, mu, power)
    return float(np.average(d, weights=None if weight is None else _arr(weight)))


def d2_tweedie(
    y: ArrayLike,
    mu: ArrayLike,
    power: float = 1.5,
    baseline: ArrayLike | None = None,
    weight: ArrayLike | None = None,
) -> float:
    """Tweedie D^2.

    ``baseline=None`` uses the constant (weighted) mean of ``y`` and reproduces
    ``sklearn.metrics.d2_tweedie_score``. Pass an explicit baseline (e.g.
    :func:`constant_rate_baseline`) to score relative to exposure x average rate.
    """
    y, mu = _arr(y), _arr(mu)
    w = np.ones_like(y) if weight is None else _arr(weight)
    if baseline is None:
        baseline = np.full_like(y, np.average(y, weights=w))
    num = np.sum(w * tweedie_unit_deviance(y, mu, power))
    den = np.sum(w * tweedie_unit_deviance(y, _arr(baseline), power))
    return float(1.0 - num / den)


def constant_rate_baseline(y: ArrayLike, exposure: ArrayLike) -> np.ndarray:
    """Exposure x (total loss / total exposure): the 'constant rate' benchmark."""
    y, e = _arr(y), _arr(exposure)
    return e * (y.sum() / e.sum())


def bootstrap_d2(
    y: ArrayLike,
    mu: ArrayLike,
    clusters: ArrayLike,
    power: float = 1.5,
    n_boot: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """Cluster bootstrap of D^2 (constant-mean baseline).

    Resamples whole clusters (e.g. policies) so correlated rows of the same risk
    are not treated as independent. Uses per-cluster sufficient statistics so
    each replicate is O(n_clusters).
    """
    y, mu = _arr(y), _arr(mu)
    p = power
    codes, _ = pd.factorize(pd.Series(clusters))
    k = codes.max() + 1
    dev_model = np.bincount(codes, tweedie_unit_deviance(y, mu, p), minlength=k)
    s_y = np.bincount(codes, y, minlength=k)
    s_y2p = np.bincount(codes, np.power(y, 2 - p), minlength=k)
    n = np.bincount(codes, minlength=k).astype(float)

    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        w = rng.multinomial(k, np.full(k, 1.0 / k)).astype(float)
        sy, sy2p, nn, dm = w @ s_y, w @ s_y2p, w @ n, w @ dev_model
        c = sy / nn
        dev_base = 2.0 * (
            sy2p / ((1 - p) * (2 - p)) - c ** (1 - p) * sy / (1 - p) + nn * c ** (2 - p) / (2 - p)
        )
        out[b] = 1.0 - dm / dev_base
    return {
        "d2": d2_tweedie(y, mu, p),
        "boot_mean": float(out.mean()),
        "boot_se": float(out.std(ddof=1)),
        "ci_low": float(np.quantile(out, 0.025)),
        "ci_high": float(np.quantile(out, 0.975)),
    }


def _lorenz_points(y: np.ndarray, e: np.ndarray, rate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ordered Lorenz curve with tied rates merged into one straight segment.

    Merging ties makes the curve independent of the input row order: the chord
    through a tied group is the average over all orderings of that group. Rates equal
    to a relative 1e-10 count as tied, so ``pred / exposure`` rounding noise on a
    constant rate does not create a spurious ordering.
    """
    order = np.argsort(rate, kind="mergesort")
    r, ee, yy = rate[order], e[order], y[order]
    starts = np.flatnonzero(np.r_[True, ~np.isclose(r[1:], r[:-1], rtol=1e-10, atol=0.0)])
    e_grp = np.add.reduceat(ee, starts)
    y_grp = np.add.reduceat(yy, starts)
    cum_e = np.concatenate([[0.0], np.cumsum(e_grp) / e.sum()])
    cum_y = np.concatenate([[0.0], np.cumsum(y_grp) / max(y.sum(), _EPS)])
    return cum_e, cum_y


def bootstrap_d2_paired(
    y: ArrayLike,
    preds: dict[str, ArrayLike],
    clusters: ArrayLike,
    power: float = 1.5,
    n_boot: int = 200,
    seed: int = 0,
    reference: str | None = None,
) -> pd.DataFrame:
    """Paired cluster bootstrap of D^2 for several models on the same rows.

    Every model is scored on the *same* resampled clusters (e.g. customers, so a
    customer's motor and home rows move together), which gives valid intervals for the
    difference between models. With one model it reproduces :func:`bootstrap_d2` for
    the same ``seed``. Returns one row per model and, if ``reference`` is given, one row
    per ``<model> - <reference>`` difference.
    """
    y = _arr(y)
    p = power
    codes, _ = pd.factorize(pd.Series(clusters))
    k = codes.max() + 1
    s_y = np.bincount(codes, y, minlength=k)
    s_y2p = np.bincount(codes, np.power(y, 2 - p), minlength=k)
    n = np.bincount(codes, minlength=k).astype(float)
    dev = {m: np.bincount(codes, tweedie_unit_deviance(y, _arr(mu), p), minlength=k)
           for m, mu in preds.items()}

    rng = np.random.default_rng(seed)
    reps = {m: np.empty(n_boot) for m in preds}
    for b in range(n_boot):
        w = rng.multinomial(k, np.full(k, 1.0 / k)).astype(float)
        sy, sy2p, nn = w @ s_y, w @ s_y2p, w @ n
        c = sy / nn
        dev_base = 2.0 * (
            sy2p / ((1 - p) * (2 - p)) - c ** (1 - p) * sy / (1 - p) + nn * c ** (2 - p) / (2 - p)
        )
        for m in preds:
            reps[m][b] = 1.0 - (w @ dev[m]) / dev_base

    def _row(name: str, point: float, r: np.ndarray) -> dict[str, float | str]:
        return {"model": name, "d2": point, "boot_se": float(r.std(ddof=1)),
                "ci_low": float(np.quantile(r, 0.025)), "ci_high": float(np.quantile(r, 0.975))}

    point = {m: d2_tweedie(y, mu, p) for m, mu in preds.items()}
    rows = [_row(m, point[m], reps[m]) for m in preds]
    if reference is not None:
        for m in preds:
            if m != reference:
                rows.append(_row(f"{m}-{reference}", point[m] - point[reference], reps[m] - reps[reference]))
    return pd.DataFrame(rows)


def normalized_gini(
    y: ArrayLike, pred: ArrayLike, exposure: ArrayLike | None = None, normalize: bool = False
) -> float:
    """Exposure-weighted Gini of the ordered Lorenz curve (Frees, Meyers & Cummings).

    Risks are sorted by predicted *rate* (pred / exposure); the curve plots the
    cumulative share of losses against cumulative share of exposure. 0 = no ranking
    power; larger is better. Tied predicted rates are merged into one segment, so the
    result does not depend on row order (a constant prediction scores exactly 0).

    By default this is the *un-normalised* Lorenz Gini used in insurance (comparable
    across models on the same data). ``normalize=True`` divides by the Gini of the
    oracle ranking (sorting on the actual rate), giving a value <= 1.
    """
    y, pred = _arr(y), _arr(pred)
    e = np.ones_like(y) if exposure is None else _arr(exposure)
    cum_e, cum_y = _lorenz_points(y, e, pred / np.maximum(e, _EPS))
    g = float(1.0 - 2.0 * np.trapezoid(cum_y, cum_e))
    if normalize:
        oe, oy = _lorenz_points(y, e, y / np.maximum(e, _EPS))
        g_oracle = float(1.0 - 2.0 * np.trapezoid(oy, oe))
        return g / g_oracle if g_oracle > 0 else float("nan")
    return g


def _midpoint_bands(e_sorted: np.ndarray, n_bins: int) -> np.ndarray:
    """Band (1..n_bins) of each sorted row from the mid-point of its exposure interval.

    A row belongs to the band containing the centre of its cumulative-exposure
    interval, so boundary rows are not pushed into the next band.
    """
    cum = np.cumsum(e_sorted)
    mid = (cum - e_sorted / 2.0) / cum[-1]
    return np.clip(np.floor(mid * n_bins).astype(int), 0, n_bins - 1) + 1


def lift_table(
    y: ArrayLike, pred: ArrayLike, exposure: ArrayLike, n_bins: int = 10
) -> pd.DataFrame:
    """Actual vs expected by bands of predicted rate with (approximately) equal exposure."""
    df = pd.DataFrame({"y": _arr(y), "pred": _arr(pred), "e": _arr(exposure)})
    df["rate"] = df["pred"] / df["e"].clip(lower=_EPS)
    df = df.sort_values("rate", kind="mergesort")
    df["band"] = _midpoint_bands(df["e"].to_numpy(), n_bins)
    g = df.groupby("band", observed=True).agg(
        exposure=("e", "sum"), actual=("y", "sum"), expected=("pred", "sum"), rows=("y", "size")
    )
    g["actual_rate"] = g["actual"] / g["exposure"]
    g["expected_rate"] = g["expected"] / g["exposure"]
    g["ae"] = g["actual"] / g["expected"]
    return g.reset_index()


def double_lift_table(
    y: ArrayLike,
    pred_new: ArrayLike,
    pred_cur: ArrayLike,
    exposure: ArrayLike,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Double-lift chart: bands of new/current ratio, each model rebased to actual."""
    y, a, b, e = _arr(y), _arr(pred_new), _arr(pred_cur), _arr(exposure)
    a = a * y.sum() / a.sum()
    b = b * y.sum() / b.sum()
    df = pd.DataFrame({"y": y, "a": a, "b": b, "e": e, "ratio": a / np.maximum(b, _EPS)})
    df = df.sort_values("ratio", kind="mergesort")
    df["band"] = _midpoint_bands(df["e"].to_numpy(), n_bins)
    g = df.groupby("band", observed=True).agg(
        exposure=("e", "sum"), actual=("y", "sum"), new=("a", "sum"), current=("b", "sum")
    )
    g["ae_new"] = g["actual"] / g["new"]
    g["ae_current"] = g["actual"] / g["current"]
    return g.reset_index()


def actual_vs_expected(
    df: pd.DataFrame, by: str | list[str], actual: str, expected: str, exposure: str
) -> pd.DataFrame:
    """One-way (or multi-way) A/E table for model monitoring and review."""
    g = df.groupby(by, observed=True, dropna=False).agg(
        exposure=(exposure, "sum"), actual=(actual, "sum"), expected=(expected, "sum"),
        rows=(actual, "size"),
    )
    g["ae"] = g["actual"] / g["expected"].replace(0, np.nan)
    return g.reset_index()


def population_stability_index(
    expected: pd.Series, actual: pd.Series, n_bins: int = 10, eps: float = 1e-4
) -> float:
    """PSI between a reference (training) and a new (test) distribution.

    Numeric columns are binned on reference quantiles; others use their levels.
    Missing values form their own bin. Rule of thumb: <0.1 stable, >0.25 shift.
    """
    ref, new = expected, actual
    if pd.api.types.is_numeric_dtype(ref) and ref.nunique(dropna=True) > n_bins:
        qs = np.unique(np.nanquantile(ref.astype(float), np.linspace(0, 1, n_bins + 1)))
        qs[0], qs[-1] = -np.inf, np.inf
        ref_b = pd.cut(ref.astype(float), qs, include_lowest=True).astype("string")
        new_b = pd.cut(new.astype(float), qs, include_lowest=True).astype("string")
    else:
        ref_b, new_b = ref.astype("string"), new.astype("string")
    ref_p = ref_b.fillna("<NA>").value_counts(normalize=True)
    new_p = new_b.fillna("<NA>").value_counts(normalize=True)
    idx = ref_p.index.union(new_p.index)
    rp = ref_p.reindex(idx, fill_value=0).to_numpy() + eps
    np_ = new_p.reindex(idx, fill_value=0).to_numpy() + eps
    return float(np.sum((np_ - rp) * np.log(np_ / rp)))
