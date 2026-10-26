"""The method of the best 1st-round submissions: six experts, their geometric mean, and a per-line
recalibration (IAA_Part1_geo_models: private D^2 0.13830, public 0.12851).

Five experts are LightGBM models and one is a GLM. Each predicts the expected ultimate claim cost:

    total       Tweedie GBM on the capped total cost, x large-loss load
    glm         Tweedie GLM on banded numeric factors and categories, x large-loss load
    peril       one Tweedie GBM per peril (its own reported share and large-loss load), summed
    fs_gamma    Poisson claim-frequency GBM x Gamma claim-size GBM, x large-loss load
    fs_lognorm  the same frequency x lognormal claim size with a risk-specific variance, uncapped
    fs_peril    claim frequency x claim size for each peril, summed

The ensemble is the row-wise geometric mean of the experts. For each line a Tweedie GLM with two
parameters then recalibrates its level and slope:

    log E[cost] = log(exposure x L) + b0 + b1 x log(geo rate / L),      b1 >= 0

L is the average rate of the geometric mean in the scored year. b0 and b1 are fitted on the 2023
and 2024 back-test predictions against the claims reported for those years, then applied to 2025.

Differences from the original code, on purpose:

* the frequency trend divides the offsets (the original multiplied by it; its README documents
  that defect, which only touched contents);
* the features are those selected by step 3 of this folder, plus the original's credibility-
  weighted ``make_model``;
* reporting completeness comes from this folder's step 3 (no parametric tail beyond four years).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from glum import GeneralizedLinearRegressor, TweedieDistribution

import pricing_lgbm as P

EPS = 1e-9
MISSING = "__missing__"        # GLM level for a missing value
OTHER = "__other__"            # GLM level for a rare or unseen category


# =================================================================================================
# 1. Claims to ultimate: open claims developed for recoveries and later repudiation
# =================================================================================================
def development_params(claims: pd.DataFrame, mature_years: tuple[int, ...], immature_years: tuple[int, ...],
                       min_z: float = 2.0) -> dict[str, dict]:
    """Recovery ratio and later-repudiation probability of open claims, per (line, peril).

    * Recovery ratio = recoveries / incurred on closed claims. Recoveries are booked at closure,
      so an open claim will still lose this share of its incurred amount.
    * Repudiation: if the recent (immature) years have fewer repudiated claims than the rate of
      the mature years predicts, and the shortfall is significant (z >= min_z), it is spread over
      the open claims as the probability that each is still declined.
    """
    c = claims.loc[claims["loss_date"].dt.year >= min(mature_years + immature_years)].copy()
    c["loss_year"] = c["loss_date"].dt.year
    closed = c.loc[c["status"] == "closed"].groupby(["line", "peril"])[["recoveries", "incurred_to_date"]].sum()
    recovery = (closed["recoveries"] / closed["incurred_to_date"]).fillna(0.0)
    repudiation = {}
    for (line, peril), d in c.groupby(["line", "peril"]):
        mature, recent = d.loc[d["loss_year"].isin(mature_years)], d.loc[d["loss_year"].isin(immature_years)]
        n_m, n_r = len(mature), len(recent)
        rate = float((mature["status"] == "repudiated").sum() / n_m) if n_m else 0.0
        expected = rate * n_r                                         # repudiations a mature year would show
        observed = float((recent["status"] == "repudiated").sum())
        n_open = float((recent["status"] == "open").sum())
        var = rate * (1 - rate) * n_r + n_r ** 2 * rate * (1 - rate) / max(n_m, 1)
        z = (expected - observed) / np.sqrt(var) if var > 0 else 0.0
        p_raw = min(max(expected - observed, 0.0) / n_open, 1.0) if n_open else 0.0
        repudiation[(line, peril)] = {"z": float(z), "p_raw": float(p_raw), "p": float(p_raw) if z >= min_z else 0.0}
    return {"recovery_ratio": {k: float(v) for k, v in recovery.items()}, "repudiation": repudiation}


def claims_to_ultimate(claims: pd.DataFrame, params: dict[str, dict]) -> pd.DataFrame:
    """Add ``net_ultimate``: closed claims as reported, repudiated 0, open claims developed.

    Open: (1 - repudiation probability) x incurred x (1 - recovery ratio).
    """
    c = claims.copy()
    keys = list(zip(c["line"].astype(str), c["peril"].astype(str)))
    rec = np.array([params["recovery_ratio"].get(k, 0.0) for k in keys])
    rep = np.array([params["repudiation"].get(k, {}).get("p", 0.0) for k in keys])
    is_open = (c["status"] == "open").to_numpy()
    is_rep = (c["status"] == "repudiated").to_numpy()
    open_ult = (1.0 - rep) * np.maximum(c["incurred_to_date"].to_numpy(dtype=float) * (1.0 - rec), 0.0)
    net = c["net_incurred"].to_numpy(dtype=float)
    c["net_ultimate"] = np.where(is_rep, 0.0, np.where(is_open, open_ult, net))
    c["loss_year"] = c["loss_date"].dt.year
    return c


# =================================================================================================
# 2. Targets of one fit set: developed, on-levelled, capped claim costs
# =================================================================================================
def severity_factor(claims: pd.DataFrame, line: str, trend: dict, target_year: int,
                    level_shift: dict | None) -> np.ndarray:
    """On-level factor of each claim's size, from its loss year to the cost level of ``target_year``."""
    rate = claims["peril"].astype(str).map(lambda p: P.trend_rates(trend, line, p)[1]).to_numpy(dtype=float)
    f = np.power(1.0 + rate, target_year - claims["loss_year"].to_numpy(dtype=float))
    if level_shift and level_shift["line"] == line:               # e.g. bodily injury 2021-22 x 0.82
        hit = (claims["peril"] == level_shift["peril"]).to_numpy() & claims["loss_year"].isin(level_shift["years"]).to_numpy()
        f = np.where(hit, f * level_shift["factor"], f)
    return f


def frequency_trend(year: pd.Series, line: str, trend: dict, target_year: int) -> np.ndarray:
    """(1 + frequency trend) ** (target_year - year): how many more claims the target year expects."""
    return np.power(1.0 + float(trend[line]["frequency"]), target_year - year.to_numpy(dtype=float))


@dataclass
class Targets:
    rows: pd.DataFrame            # row_id, y, y_capped, n_claims, y_<p>, yc_<p>, n_<p>
    claims: pd.DataFrame          # one record per claim with a cost: row_id, peril, loss_year, cost_T
    cap: float                    # claim-level large-loss cap
    load: float                   # total / capped total on the mature fit years
    peril_loads: dict[str, float]
    load_years: list[int]


def build_targets(rows: pd.DataFrame, claims_ult: pd.DataFrame, line: str, perils: tuple[str, ...],
                  trend: dict, level_shift: dict | None, target_year: int, cap_quantile: float,
                  fit_years: list[int]) -> Targets:
    """Claim costs of ``rows`` at the cost level of ``target_year``.

    The cap and the large-loss loads come from the claims of ``fit_years`` only, so a back-test
    learns nothing from its holdout year. The loads use the mature fit years (mean reported share
    at least 95%): in a young year the large claims are still open or unreported.
    """
    m = claims_ult.loc[(claims_ult["line"] == line) & (claims_ult["status"] != "repudiated")
                       & (claims_ult["match_status"] == "matched")
                       & claims_ult["row_id"].isin(rows["row_id"])].copy()
    m["cost_T"] = m["net_ultimate"].to_numpy(dtype=float) * severity_factor(m, line, trend, target_year, level_shift)
    fit = m["loss_year"].isin(fit_years).to_numpy()
    cap = float(np.quantile(m.loc[fit & (m["cost_T"] > 0).to_numpy(), "cost_T"], cap_quantile))
    m["cost_T_capped"] = np.minimum(m["cost_T"], cap)

    g = m.groupby("row_id")
    agg = pd.DataFrame({"y": g["cost_T"].sum(), "y_capped": g["cost_T_capped"].sum(), "n_claims": g.size()})
    for name, col, how in (("y_", "cost_T", "sum"), ("yc_", "cost_T_capped", "sum"), ("n_", "claim_id", "size")):
        piv = m.pivot_table(index="row_id", columns="peril", values=col, aggfunc=how, fill_value=0)
        agg = agg.join(piv.reindex(columns=list(perils), fill_value=0).add_prefix(name))
    out = rows[["row_id", "year", "comp_cost"]].join(agg, on="row_id")
    tcols = [c for c in out.columns if c not in ("row_id", "year", "comp_cost")]
    out[tcols] = out[tcols].fillna(0.0).astype(float)

    by_year = out.groupby("year").agg(comp=("comp_cost", "mean"), **{c: (c, "sum") for c in tcols})
    cand = by_year.loc[by_year.index.isin(fit_years)]
    mature = cand.index[cand["comp"] >= 0.95].tolist() or cand.index.tolist()
    sel = by_year.loc[mature]
    load = float(sel["y"].sum() / sel["y_capped"].sum())
    peril_loads = {p: float(sel[f"y_{p}"].sum() / sel[f"yc_{p}"].sum()) if sel[f"yc_{p}"].sum() > 0 else 1.0
                   for p in perils}
    claims = m.loc[m["cost_T"] > 0, ["claim_id", "row_id", "peril", "loss_year", "cost_T"]].reset_index(drop=True)
    return Targets(out.drop(columns=["year", "comp_cost"]), claims, cap, load, peril_loads,
                   [int(y) for y in mature])


def credibility_encode(train: pd.DataFrame, others: list[pd.DataFrame], col: str, y: np.ndarray,
                       offset: np.ndarray, n_claims: np.ndarray, k: float = 100.0,
                       n_folds: int = 5) -> tuple[np.ndarray, list[np.ndarray]]:
    """Credibility-weighted log relativity of a many-level factor (Buhlmann style).

    Level l: Z = N / (N + k) with N its claims, and the encoding log(Z x A/E + 1 - Z), where A/E
    compares its claims with the offset-only expectation. Training rows get out-of-fold values
    (folds of policies), so no row sees its own claims. ``others`` use every training row.
    """
    mu0 = np.exp(offset)
    mu0 = mu0 * y.sum() / mu0.sum()
    lv = train[col].astype(str).to_numpy()
    fold = pd.util.hash_array(train["policy_id"].astype(str).to_numpy()) % n_folds

    def table(mask: np.ndarray) -> pd.Series:
        g = pd.DataFrame({"l": lv[mask], "y": y[mask], "e": mu0[mask], "n": n_claims[mask]}).groupby("l").sum()
        z = g["n"] / (g["n"] + k)
        return np.log(z * (g["y"] / g["e"]) + (1 - z))

    enc = np.zeros(len(train))
    for f in range(n_folds):
        idx = fold == f
        enc[idx] = pd.Series(lv[idx]).map(table(~idx)).fillna(0.0).to_numpy()
    full = table(np.ones(len(train), dtype=bool))
    return enc.astype("float32"), [o[col].astype(str).map(full).fillna(0.0).to_numpy(dtype="float32") for o in others]


# =================================================================================================
# 3. The GLM expert: a classical tariff structure
# =================================================================================================
@dataclass
class BandedGLM:
    """Tweedie GLM (log link, offset) on banded numeric factors and grouped categories.

    Numeric columns are cut at training quantiles (10 bands; columns with few values keep each
    value), missing values get their own band, and categories seen on less than ``min_exposure``
    years of cover are pooled. Each factor then has a table of relativities an actuary can review.
    """

    numeric: list[str]
    categorical: list[str]
    alpha: float = 5e-4             # ridge strength (glum scaling)
    n_bins: int = 10
    min_exposure: float = 200.0
    power: float = P.TWEEDIE_POWER
    edges_: dict[str, np.ndarray] = field(default_factory=dict)
    levels_: dict[str, list[str]] = field(default_factory=dict)
    model_: GeneralizedLinearRegressor | None = field(default=None, repr=False)

    def _band(self, X: pd.DataFrame) -> dict[str, pd.Categorical]:
        out = {}
        for c, e in self.edges_.items():
            v = X[c].to_numpy(dtype=float)
            idx = np.clip(np.searchsorted(e, v, side="right") - 1, 0, len(e) - 2)
            names = [f"b{i:02d}" for i in range(len(e) - 1)]
            lab = np.where(np.isfinite(v), np.array(names, dtype=object)[idx], MISSING)
            out[c] = pd.Categorical(lab, categories=names + [MISSING])
        return out

    def design(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = self._band(X)
        for c, keep in self.levels_.items():
            s = X[c].astype("string")
            lab = s.where(s.isin(keep), OTHER).fillna(MISSING)             # rare / unseen / missing
            cols[c] = pd.Categorical(lab.astype(object), categories=keep + [OTHER, MISSING])
        return pd.DataFrame(cols, index=X.index)

    def fit(self, X: pd.DataFrame, y: np.ndarray, offset: np.ndarray, exposure: np.ndarray) -> "BandedGLM":
        self.edges_ = {}
        for c in self.numeric:
            v = X[c].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            u = np.unique(v)
            if len(u) <= self.n_bins:                                      # few values: one band each
                self.edges_[c] = np.concatenate([[-np.inf], (u[:-1] + u[1:]) / 2, [np.inf]])
            else:
                q = np.unique(np.quantile(v, np.linspace(0, 1, self.n_bins + 1)))
                q[0], q[-1] = -np.inf, np.inf
                self.edges_[c] = q
        self.levels_ = {}
        for c in self.categorical:
            expo = pd.Series(exposure).groupby(X[c].astype("string").to_numpy()).sum()
            self.levels_[c] = sorted(str(k) for k, v in expo.items() if v >= self.min_exposure)
        self.model_ = GeneralizedLinearRegressor(family=TweedieDistribution(self.power), link="log",
                                                 alpha=self.alpha, l1_ratio=0.0, fit_intercept=True,
                                                 drop_first=False, max_iter=200)
        self.model_.fit(self.design(X), np.asarray(y, dtype=float), offset=np.asarray(offset, dtype=float))
        return self

    def predict(self, X: pd.DataFrame, offset: np.ndarray) -> np.ndarray:
        return self.model_.predict(self.design(X), offset=np.asarray(offset, dtype=float))


# =================================================================================================
# 4. The experts
# =================================================================================================
@dataclass
class FitData:
    """One fit set of one line: a back-test, a cross-validation fold, or the final fit.

    ``prior`` / ``prior_score`` hold centred log priors by component ("cost", "freq", "sev",
    "cost_<peril>", "freq_<peril>", "sev_<peril>") for the training and scored rows. They enter the
    offsets, so a model only learns how the data depart from the prior (2nd round: priors fitted
    on the developed market). Without priors every offset is the plain one of the 1st round.
    """

    line: str
    perils: tuple[str, ...]
    train: pd.DataFrame           # training rows: keys, exposure, reported shares, targets
    score: pd.DataFrame           # rows to predict
    X: pd.DataFrame               # model matrix of the training rows
    Xs: pd.DataFrame              # model matrix of the scored rows (same category levels)
    claims: pd.DataFrame          # claims for the claim-size models: pos (row in X), policy_id, peril, cost_T
    targets: Targets
    freq_trend: np.ndarray        # training rows: (1 + frequency trend) ** (target year - year)
    monotone: list[int]           # LightGBM monotone constraints aligned to X.columns
    params: dict[str, Any]        # LightGBM settings of the line
    refit: bool                   # final fit: refit every model on all rows with 1.1x trees
    seed: int
    score_trend: np.ndarray | None = None       # scored rows of earlier years (cross-validation folds)
    prior: dict[str, np.ndarray] = field(default_factory=dict)
    prior_score: dict[str, np.ndarray] = field(default_factory=dict)
    min_freq_claims: int = 50     # fewer claims of a peril: no frequency model (the prior alone if any)
    min_size_claims: int = 300    # fewer claims of a peril: no claim-size model (the prior or the mean)

    def lp(self, key: str | None) -> np.ndarray:
        return self.prior[key] if key in self.prior else np.zeros(len(self.train))

    def lps(self, key: str | None) -> np.ndarray:
        return self.prior_score[key] if key in self.prior_score else np.zeros(len(self.score))

    def offset(self, share: str, prior: str | None = None) -> np.ndarray:
        """log(exposure x reported share / frequency trend) + log prior, training rows."""
        e = self.train["exposure"].to_numpy(dtype=float)
        return np.log(e * P.completeness(self.train, share) / self.freq_trend) + self.lp(prior)

    def score_offset(self, prior: str | None = None) -> np.ndarray:
        """Scored rows at their ultimate cost in the target year: log(exposure) + log prior."""
        o = np.log(self.score["exposure"].to_numpy(dtype=float)) + self.lps(prior)
        return o if self.score_trend is None else o - np.log(self.score_trend)   # earlier-year rows

    def claim_prior(self, key: str) -> np.ndarray:
        """The log prior of the row each claim belongs to (claim-size models)."""
        return self.lp(key)[self.claims["pos"].to_numpy()]

    def score_share(self, share: str) -> np.ndarray:
        """Reported share of the scored rows (back-test and cross-validation rows); 1 for 2025 rows."""
        return P.completeness(self.score, share) if share in self.score else np.ones(len(self.score))

    @property
    def Xc(self) -> pd.DataFrame:
        """Model matrix of the claims: the features of the row each claim belongs to."""
        return self.X.iloc[self.claims["pos"].to_numpy()].reset_index(drop=True)


def _gbm(d: FitData, y: np.ndarray, offset: np.ndarray, overrides: dict | None = None, monotone: bool = True,
         X: pd.DataFrame | None = None, policy: pd.Series | None = None) -> tuple[P.TweedieLGBM, float]:
    """Early-stopped (and in the final fit, refitted) LightGBM, with its balance factor."""
    X = d.X if X is None else X
    policy = d.train["policy_id"] if policy is None else policy
    y = np.asarray(y, dtype=float)
    m = P.fit_early_stopped({**d.params, **(overrides or {})}, X, y, offset, policy, refit=d.refit,
                            seed=d.seed, monotone=d.monotone if monotone else None)
    return m, P.balance_factor(y, m.predict(X, offset))


def prior_alone(y: np.ndarray, offset: np.ndarray, score_offset: np.ndarray) -> np.ndarray:
    """The prior re-levelled to this market's data: used where a peril has too few claims."""
    return np.exp(score_offset) * float(np.sum(y) / np.exp(offset).sum())


def expert_total(d: FitData) -> tuple[dict, dict]:
    """Tweedie GBM on the capped total cost, x the line's large-loss load."""
    y = d.train["y_capped"].to_numpy(dtype=float)
    m, b = _gbm(d, y, d.offset("comp_cost", "cost"))
    pred = m.predict(d.Xs, d.score_offset("cost")) * b * d.targets.load
    return {"total": pred}, {"trees": m.n_trees, "balance": b}


def expert_glm(d: FitData, glm_settings: dict) -> tuple[dict, dict]:
    """Banded Tweedie GLM on the same features, target and offset as ``total``."""
    numeric = [c for c in d.X.columns if not isinstance(d.X[c].dtype, pd.CategoricalDtype)]
    categorical = [c for c in d.X.columns if c not in numeric]
    y, off = d.train["y_capped"].to_numpy(dtype=float), d.offset("comp_cost", "cost")
    g = BandedGLM(numeric, categorical, **glm_settings).fit(d.X, y, off, d.train["exposure"].to_numpy(dtype=float))
    b = P.balance_factor(y, g.predict(d.X, off))
    return {"glm": g.predict(d.Xs, d.score_offset("cost")) * b * d.targets.load}, {"balance": b}


def expert_peril(d: FitData) -> tuple[dict, dict]:
    """One Tweedie GBM per peril on its capped cost, with its own reported share and load."""
    total, reported, info = np.zeros(len(d.score)), np.zeros(len(d.score)), {}
    for p in d.perils:
        y = d.train[f"yc_{p}"].to_numpy(dtype=float)
        if y.sum() <= 0:
            continue
        off, so = d.offset(f"comp_{p}", f"cost_{p}"), d.score_offset(f"cost_{p}")
        if f"cost_{p}" in d.prior and d.train[f"n_{p}"].sum() < d.min_freq_claims:
            pred, trees = prior_alone(y, off, so), 0                 # too few claims: the prior alone
        else:
            m, b = _gbm(d, y, off)
            pred, trees = m.predict(d.Xs, so) * b, m.n_trees
        pred = pred * d.targets.peril_loads[p]
        total += pred
        reported += pred * d.score_share(f"comp_{p}")           # what would be reported by now
        info[p] = {"trees": trees, "load": d.targets.peril_loads[p]}
    return {"peril": total, "peril_rep": reported}, info


def expert_frequency_severity(d: FitData, severity: dict) -> tuple[dict, dict]:
    """``fs_gamma`` and ``fs_lognorm``: one claim-frequency model x two claim-size models.

    Claim sizes are learnt on claim records (each with its row's features), so they are not
    swamped by the ~90% of rows without a claim. The lognormal size has a risk-specific variance:
    one GBM for the mean of log cost m(x), one for the residual variance s2(x), and
    E[size] = exp(m + s2 / 2), so risks with heavier tails get a higher mean. It is uncapped.
    """
    freq_m, bf = _gbm(d, d.train["n_claims"], d.offset("comp_count", "freq"), {"objective": "poisson"})
    freq = freq_m.predict(d.Xs, d.score_offset("freq")) * bf     # claims expected in the target year
    Xc, pol = d.Xc, d.claims["policy_id"].reset_index(drop=True)
    cost = d.claims["cost_T"].to_numpy(dtype=float)
    cp, sp = d.claim_prior("sev"), d.lps("sev")                   # claim-size prior (0 without one)
    zc, zs = np.zeros(len(Xc)), np.zeros(len(d.Xs))

    gam, bg = _gbm(d, np.minimum(cost, d.targets.cap), cp, {**severity, "objective": "gamma"},
                   monotone=False, X=Xc, policy=pol)
    out = {"fs_gamma": freq * gam.predict(d.Xs, sp) * bg * d.targets.load}

    log_cost = np.log(cost)
    mean_m = P.fit_early_stopped({**d.params, **severity, "objective": "regression"}, Xc, log_cost, cp, pol,
                                 refit=d.refit, seed=d.seed)            # mean of log cost (identity link)
    mu_c = mean_m.predict(Xc, cp)
    resid2 = np.maximum((log_cost - mu_c) ** 2, 1e-12)
    var_m, bv = _gbm(d, resid2, zc, {**severity, "objective": "gamma", "num_leaves": 7, "min_data_in_leaf": 1000},
                     monotone=False, X=Xc, policy=pol)
    # Calibrate the lognormal mean to the average claim size of the training claims.
    calib = cost.sum() / np.exp(mu_c + var_m.predict(Xc, zc) * bv / 2).sum()
    size_ln = np.exp(mean_m.predict(d.Xs, sp) + var_m.predict(d.Xs, zs) * bv / 2) * calib
    out["fs_lognorm"] = freq * size_ln
    info = {"freq_trees": freq_m.n_trees, "gamma_trees": gam.n_trees, "lognorm_trees": [mean_m.n_trees, var_m.n_trees],
            "lognorm_calibration": float(calib), "claims": int(len(Xc))}
    return out, info


def expert_peril_frequency_severity(d: FitData) -> tuple[dict, dict]:
    """``fs_peril``: claim frequency x claim size for each peril (own reported share), summed."""
    total, reported, info = np.zeros(len(d.score)), np.zeros(len(d.score)), {}
    Xc = d.Xc                                                   # built once for every peril
    for p in d.perils:
        n_p = d.train[f"n_{p}"].to_numpy(dtype=float)
        off, so = d.offset(f"compn_{p}", f"freq_{p}"), d.score_offset(f"freq_{p}")
        if n_p.sum() >= d.min_freq_claims:
            fm, bf = _gbm(d, n_p, off, {"objective": "poisson"})
            freq, freq_trees = fm.predict(d.Xs, so) * bf, fm.n_trees
        elif f"freq_{p}" in d.prior and n_p.sum() > 0:
            freq, freq_trees = prior_alone(n_p, off, so), 0         # few claims: the prior alone
        else:
            continue                                                # no model and no prior
        sel = (d.claims["peril"] == p).to_numpy()
        y_cap = np.minimum(d.claims.loc[sel, "cost_T"].to_numpy(dtype=float), d.targets.cap)
        cp, sp = d.claim_prior(f"sev_{p}")[sel], d.lps(f"sev_{p}")
        if sel.sum() >= d.min_size_claims:
            sev_set = {"objective": "gamma", "num_leaves": 7, "learning_rate": 0.03,
                       "min_data_in_leaf": max(100, min(1000, int(sel.sum()) // 30))}
            sm, bs = _gbm(d, y_cap, cp, sev_set, monotone=False, X=Xc.loc[sel].reset_index(drop=True),
                          policy=d.claims.loc[sel, "policy_id"].reset_index(drop=True))
            size, size_trees = sm.predict(d.Xs, sp) * bs, sm.n_trees
        elif f"sev_{p}" in d.prior and sel.sum() > 0:
            size, size_trees = prior_alone(y_cap, cp, sp), 0
        else:                                                      # few claims: the peril's mean size
            size, size_trees = np.full(len(d.Xs), float(y_cap.mean()) if len(y_cap) else 0.0), 0
        pred = freq * size * d.targets.peril_loads[p]
        total += pred
        reported += pred * d.score_share(f"comp_{p}")
        info[p] = {"freq_trees": freq_trees, "size_trees": size_trees, "claims": int(sel.sum())}
    return {"fs_peril": total, "fs_peril_rep": reported}, info


EXPERT_STEPS = ("total", "glm", "peril", "fs", "fs_peril")   # fs gives both fs_gamma and fs_lognorm


def run_experts(d: FitData, steps: tuple[str, ...], glm_settings: dict, severity: dict) -> tuple[pd.DataFrame, dict]:
    """Every requested expert on one fit set: one prediction column per expert, plus info."""
    out = pd.DataFrame({"row_id": d.score["row_id"].astype(str).to_numpy()})
    info: dict[str, Any] = {}
    for step in steps:
        if step == "total":
            cols, info[step] = expert_total(d)
        elif step == "glm":
            cols, info[step] = expert_glm(d, glm_settings)
        elif step == "peril":
            cols, info[step] = expert_peril(d)
        elif step == "fs":
            cols, info[step] = expert_frequency_severity(d, severity)
        elif step == "fs_peril":
            cols, info[step] = expert_peril_frequency_severity(d)
        else:
            raise ValueError(f"unknown expert step {step!r}")
        for k, v in cols.items():
            out[k] = v
    return out, info


# =================================================================================================
# 5. The ensemble: geometric mean, then a level and slope per line
# =================================================================================================
def geo_mean(df: pd.DataFrame, members: list[str]) -> np.ndarray:
    """Row-wise equal-weight geometric mean of the members' predictions."""
    return np.exp(np.mean([np.log(np.maximum(df[m].to_numpy(dtype=float), EPS)) for m in members], axis=0))


def reported_share(df: pd.DataFrame) -> np.ndarray:
    """Share of a scored row's ultimate cost reported by the valuation date (1 for 2025 rows).

    It comes from the peril expert's mix: sum over perils of prediction x reported share,
    divided by the prediction, so a row with more slow-reporting perils gets a lower share.
    """
    c = (df["peril_rep"] / df["peril"]).to_numpy(dtype=float)
    c = np.where(np.isfinite(c), c, 1.0)
    return np.where(df["set"].eq("final").to_numpy(), 1.0, np.clip(c, 1e-4, 1.0))


@dataclass
class LinePool:
    """log E[cost] = log(exposure x L) + b0 + b1 x log(geo rate / L), with b1 >= 0.

    L is the average rate of ``geo`` within each scored set, so the pool only learns how far the
    level was off (b0) and how much to stretch the ranking (b1). It is fitted on back-test or
    cross-validation rows against reported claims (the reported share enters the offset). With
    ``fit_slope=False`` the slope stays at 1 and only the level is fitted, in closed form: the
    Tweedie score equation gives b0 = log(sum(y m^(1-p)) / sum(m^(2-p))).
    """

    alpha: float = 1e-5
    power: float = P.TWEEDIE_POWER
    fit_slope: bool = True
    b0_: float = 0.0
    b1_: float = 1.0

    @staticmethod
    def _level(df: pd.DataFrame) -> np.ndarray:
        g = df.groupby("set")
        return (g["geo"].transform("sum") / g["exposure"].transform("sum")).to_numpy(dtype=float)

    def _x(self, df: pd.DataFrame) -> np.ndarray:
        rate = df["geo"].to_numpy(dtype=float) / df["exposure"].to_numpy(dtype=float)
        return np.log(np.maximum(rate / self._level(df), EPS))

    def _offset(self, df: pd.DataFrame, reported: bool) -> np.ndarray:
        off = np.log(df["exposure"].to_numpy(dtype=float) * self._level(df))
        return off + (np.log(reported_share(df)) if reported else 0.0)

    def fit(self, df: pd.DataFrame) -> "LinePool":
        x, off, y = self._x(df), self._offset(df, True), df["y"].to_numpy(dtype=float)
        if self.fit_slope:
            m = GeneralizedLinearRegressor(family=TweedieDistribution(self.power), link="log", alpha=self.alpha,
                                           l1_ratio=0.0, fit_intercept=True, lower_bounds=np.array([0.0]))
            m.fit(pd.DataFrame({"log_geo": x}), y, offset=off)
            self.b0_, self.b1_ = float(m.intercept_), float(m.coef_[0])
        else:                                                    # level only: b1 = 1
            mu = np.exp(off + x)
            p = self.power
            self.b0_ = float(np.log(np.sum(y * mu ** (1 - p)) / np.sum(mu ** (2 - p))))
            self.b1_ = 1.0
        return self

    def predict(self, df: pd.DataFrame, reported: bool = True) -> np.ndarray:
        return np.exp(self.b0_ + self.b1_ * self._x(df) + self._offset(df, reported))

    @property
    def coefficients(self) -> dict[str, float]:
        return {"b0": self.b0_, "b1": self.b1_}
