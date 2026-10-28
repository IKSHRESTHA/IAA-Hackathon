"""Step 11 - every segment's best level, read from all the public scores at once.

Step 10 reads one probe at a time. But any scored file is a probe of the segment levels, as long as
it differs from another scored file only by one factor per segment (line x new business, and
gender for motor and contents). For Tweedie p = 1.5 the public score of a file is

    D2 = K - (4 / d0) * sum_s B_s (c_s + 1)

where B_s is the sum of mu^0.5 over segment s divided by the rows, c_s = A_s / B_s is the file's
best scale for the segment (A_s = sum of y mu^-0.5 on the public rows), d0 is the null deviance
per row and K does not depend on the file. Scaling segment s by f turns c_s into c_s / f and B_s
into B_s f^0.5. So files of one family (the same predictions up to segment factors) give exact
linear equations in the unknown best levels. Files of different families are linked through the
best segment total, which is about the same whatever the ranking inside the segment, plus a term
for each family's own ranking (prior sd ``FAMILY_SD``).

Evidence:
* ``data/reference/leaderboard_evidence.csv``: the pricing repository's scored files, summarised
  per segment once with the ``evidence`` mode. Its trio files v1, v2, v6, v8a and v8b are the same
  three experts with different new-business, gender and buildings factors, so they are one exact
  family (their base is the repository's final file, v6 with both v8 changes).
* every version in ``submissions/versions.csv`` that has a public score and differs from the base
  (v5) only by segment factors: they join its family. Other versions are not used.

The GBM-weight path (prior tilt, GLM tilt, 40%, 50% and 65% GBM) is a blend of the trio's own
experts, so it joins the trio family with a little noise (``PATH_SD``) for its small ranking
changes. Left out of the fit, its five scores are predicted from the levels alone with one constant
miss (-0.0020 each, the trio family's ranking term): the levels explain the whole path. Fitted with
its own family term at a tight sd, the estimate swings to extremes (contents women x0.43); with
``FAMILY_SD`` = 1e-3 the readings agree within a few percent.

Public luck. The public score covers a sample of the 2025 rows, and a segment's best level on
that sample differs from its best level on the other rows by the sample's luck: sd
sqrt(phi / (share x sum of mu^0.5)), with phi the line's claim dispersion (from the out-of-fold
predictions of 2023-24) and share the public share of rows (``PUBLIC_SHARE``, 30% unless the
leaderboard page says otherwise). At 30%, that is 8% for motor new-business men and 24% for
buildings renewals. The final ranking scores the other rows, so the fit separates the two:

* u_s: the segment's best level on rows not yet scored (prior: a line level, a new-business and a
  gender effect per line, sd ``PRIOR_SD``);
* e_s: the public rows' luck (sd as above); every public score sees u_s + e_s;
* a link vector per other family (sd ``LINK_SD``), K and the family terms.

The model is linear in them, so the posterior is exact for a given null deviance d0. d0 is read
from the scores too: the fit is repeated over a grid and the value with the highest posterior
(evidence x a prior of sd ``D0_SD`` around the model's own value) is kept.

    fit:       python 11_joint_levels.py fit [--public-share 0.3]
    make:      python 11_joint_levels.py make                     (levels for rows not yet scored)
               python 11_joint_levels.py make --target public     (levels for the public rows)
               python 11_joint_levels.py make --set motor_m=1.08 motor_f=0.93
               python 11_joint_levels.py make --target public --scores-before v12
                          (rebuilds v13: only the scores known before v12 and v13 were made)
    evidence:  python 11_joint_levels.py evidence --ref-repo "C:/Users/.../iaa-pricing-part2"

A ``--set`` key names a segment or a group of segments by its parts: ``motor_new`` is both motor
new-business segments, ``contents_m`` both contents male segments, ``buildings`` the whole line.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

sys.path.insert(0, str(C.ROOT.parent))            # the shared modules sit in the folder above
import geo_experts as G  # noqa: E402
import submissions as S  # noqa: E402

S10 = importlib.import_module("10_level_probe")   # 2025 rows, versions, null deviance, checks
EVIDENCE = C.DATA / "reference" / "leaderboard_evidence.csv"
OUT = C.REPORTS / "11_joint"
BASE = "v5"                       # the family the levels are read against (the best file's levels)
GENDER_LINES = ("motor", "contents")
ROUNDING_SD = 1.5e-5              # public scores have five decimals; a little room for rounding
PATH_SD = 2e-4                    # blends of the trio's experts: small ranking changes on top of levels
LINK_SD = 0.03                    # another family's best segment total may sit this far from the base's
PUBLIC_SHARE = 0.30               # share of the 2025 rows behind the public score (assumed; see --public-share)
D0_SD = 0.15                      # prior sd (log) of the null deviance around the model's own value
FAMILY_SD = 1e-3                  # how much a different ranking can move the public score
PRIOR_SD = {                      # prior sd of the log best level vs the base, per effect
    "line": {"motor": 0.05, "buildings": 0.08, "contents": 0.08},
    "new_business": 0.15,
    "gender": 0.25,               # men vs women: the leaderboard has shown gender effects this large
    "segment": 0.04,
}

# The pricing repository's scored files (its docs/validation.md and MODEL_EXPERIMENTS.md). The trio
# family shares one set of predictions; ``factors`` turn its base file into each scored file.
_R_NB = 1.0370567 / 1.1016        # motor new business priced on claim frequency only (v6, v8a)
_FIRE = 0.94818                   # buildings fire-severity factor (v1 to v6, and v8b)
_NO_GENDER = {"motor_m": 1 / 1.078792, "motor_f": 1 / 0.921255,      # v1, v2: before the gender factors
              "contents_f": 1 / 1.093206, "contents_m": 1 / 0.906891}
_TRIO = "reference/submission_emerging.csv"
REF_FILES = [
    # name, file in the repository, public score, family, noise sd, factors on the file, use
    ("trio_v8b", _TRIO, 0.16319, "trio", ROUNDING_SD, {"buildings": _FIRE}, "fit"),
    ("trio_v8a", _TRIO, 0.16199, "trio", ROUNDING_SD, {"motor_new": _R_NB}, "fit"),
    ("trio_v6", _TRIO, 0.16172, "trio", ROUNDING_SD, {"motor_new": _R_NB, "buildings": _FIRE}, "fit"),
    ("trio_v2", _TRIO, 0.15835, "trio", ROUNDING_SD, {"motor_new": _R_NB, "buildings": _FIRE, **_NO_GENDER},
     "fit"),
    ("trio_v1", _TRIO, 0.15786, "trio", ROUNDING_SD,
     {"motor_new": 1 / 1.1016, "buildings": _FIRE, "buildings_new": 1 / 0.975, "contents_new": 1 / 1.185,
      **_NO_GENDER}, "fit"),
    # the equal-weight trio as scored: the same experts refitted, 1-4% apart inside segments
    ("trio_equal", "conservative_blend_refinement/equal_baseline_transfer_expert_submission.csv", 0.16350,
     "trio", 1e-4, {}, "fit"),
    # the GBM-weight path: geometric blends of the same experts, so the trio family (PATH_SD)
    ("prior_tilt", "conservative_blend_refinement/prior_tilt_transfer_expert_submission.csv", 0.16334,
     "trio", PATH_SD, {}, "fit"),
    ("glm_tilt", "conservative_blend_refinement/glm_tilt_transfer_expert_submission.csv", 0.16338,
     "trio", PATH_SD, {}, "fit"),
    ("gbm40", "gbm_tilt_refinement/gbm_40_winner_transfer_expert_submission.csv", 0.16376, "trio", PATH_SD, {}, "fit"),
    ("gbm50", "gbm_tilt_refinement/gbm_50_transfer_expert_submission.csv", 0.16412, "trio", PATH_SD, {}, "fit"),
    ("gbm65", "gbm_tilt_refinement_next/gbm_65_transfer_expert_submission.csv", 0.16458, "trio", PATH_SD, {}, "fit"),
    # the 100% GBM: the best file, another ranking, its own family
    ("gbm100", "gbm_100_transfer_expert_notebook/gbm_100_transfer_expert_notebook_submission.csv", 0.16529,
     "gbm100", ROUNDING_SD, {}, "fit"),
]


# =================================================================================================
# Segments
# =================================================================================================
def segment_keys(meta: pd.DataFrame) -> np.ndarray:
    """Segment of each 2025 row: line_new or line_renewal, plus _f or _m for motor and contents."""
    line = meta["line"].to_numpy().astype(object)
    key = line + "_" + np.where(meta["_new_business"].to_numpy(dtype=bool), "new", "renewal")
    gender = meta["_gender"].astype(str).to_numpy().astype(object)
    return np.where(np.isin(line, GENDER_LINES), key + "_" + gender, key)


def factor_vector(segments: list[str], spec: dict[str, float]) -> np.ndarray:
    """Factor per segment from keys such as motor_new or contents_m (every part must match)."""
    f = np.ones(len(segments))
    for key, x in spec.items():
        parts = set(key.split("_"))
        hit = [i for i, s in enumerate(segments) if parts <= set(s.split("_"))]
        if not hit:
            raise KeyError(f"{key} matches no segment of {segments}")
        f[hit] *= x
    return f


def summarise(pred: pd.Series, meta: pd.DataFrame, keys: np.ndarray, segments: list[str]) -> pd.DataFrame:
    """Total prediction and sum of sqrt(prediction) per segment."""
    mu = pred.loc[meta.index].to_numpy(dtype=float)
    if not np.all(np.isfinite(mu)) or (mu <= 0).any():
        raise ValueError("predictions must be finite and positive on every 2025 row")
    df = pd.DataFrame({"segment": keys, "total": mu, "sqrt_sum": np.sqrt(mu)}).groupby("segment").sum()
    return df.reindex(segments)


# =================================================================================================
# Evidence: per-segment totals of every scored file
# =================================================================================================
def build_evidence(ref_repo: Path, meta: pd.DataFrame, keys: np.ndarray, segments: list[str]) -> pd.DataFrame:
    """Summarise the pricing repository's scored files once (they are not copied here)."""
    parts = []
    for name, rel, score, family, noise, spec, use in REF_FILES:
        sub = pd.read_csv(ref_repo / rel, dtype={"row_id": str}).set_index("row_id")["prediction"]
        f = factor_vector(segments, spec)
        summ = summarise(sub, meta, keys, segments)
        summ["total"] *= f                                    # factors are constant inside a segment,
        summ["sqrt_sum"] *= np.sqrt(f)                        # so the summaries scale exactly
        parts.append(summ.reset_index().assign(file=name, family=family, public=score, noise_sd=noise, use=use))
    ev = pd.concat(parts, ignore_index=True)
    ev["rows"] = ev["segment"].map(pd.Series(keys).value_counts())
    return ev[["file", "family", "public", "noise_sd", "use", "segment", "rows", "total", "sqrt_sum"]]


def same_family(sub: pd.Series, base: pd.Series, meta: pd.DataFrame, keys: np.ndarray) -> bool:
    """True when ``sub`` is ``base`` times one factor per segment."""
    r = pd.Series(np.log(sub.loc[meta.index].to_numpy(dtype=float) / base.loc[meta.index].to_numpy(dtype=float)))
    spread = r.groupby(keys).agg(lambda x: x.max() - x.min())
    return bool((spread < 1e-9).all())


def register_files(meta: pd.DataFrame, keys: np.ndarray, segments: list[str], base: pd.Series):
    """Every registered version: its summary, whether it is in the base family, and its score."""
    reg = pd.read_csv(C.ROOT / "submissions" / "versions.csv", dtype=str)
    out = []
    for _, r in reg.iterrows():
        sub = pd.read_csv(C.ROOT / "submissions" / r["file"], dtype={"row_id": str}).set_index("row_id")["prediction"]
        score = pd.to_numeric(r.get("public_score"), errors="coerce")
        out.append({"version": r["version"], "file": r["file"], "public": score,
                    "in_family": same_family(sub, base, meta, keys),
                    "summary": summarise(sub, meta, keys, segments)})
    return out


# =================================================================================================
# The model: linear in the best levels, so the posterior is exact
# =================================================================================================
def prior_cov(segments: list[str]) -> np.ndarray:
    """Prior covariance of the log best levels: line, new business and gender effects, plus noise."""
    cols = []
    for line in C.LINES:
        in_line = np.array([s.split("_")[0] == line for s in segments], dtype=float)
        is_new = np.array(["new" in s.split("_") for s in segments], dtype=float)
        cols += [PRIOR_SD["line"][line] * in_line, PRIOR_SD["new_business"] * in_line * is_new]
        if line in GENDER_LINES:
            sign = np.array([s.endswith("_m") - s.endswith("_f") for s in segments], dtype=float)
            cols.append(PRIOR_SD["gender"] / 2 * in_line * sign)
    Z = np.array(cols).T
    return Z @ Z.T + PRIOR_SD["segment"] ** 2 * np.eye(len(segments))


class Fit:
    """Posterior of each segment's best level, on the public rows and on rows not yet seen.

    Parameters: u (the best level on unseen rows: what the private board rewards), e (the public
    rows' own luck, sd ``public_sd`` per segment), a link vector per other family (``LINK_SD``:
    its best totals may sit a few percent from the base's), K and the family terms. Every public
    score sees u + e.
    """

    def __init__(self, files: list[dict], base: pd.DataFrame, segments: list[str], d0: float, n_rows: int,
                 public_sd: np.ndarray):
        nS = len(segments)
        fams = sorted({f["family"] for f in files if f["family"] != BASE})
        self.segments, self.d0, self.n, self.fams, self.files = segments, d0, n_rows, fams, files
        self.B0 = base["sqrt_sum"].to_numpy() / n_rows                  # the base's B per segment
        self.L0 = base["total"].to_numpy()
        iu, ie, il = 0, nS, 2 * nS                                       # where each block sits
        iK = il + nS * len(fams)
        iR = iK + 1
        nP = iR + len(fams)
        X, y, w = np.zeros((len(files), nP)), np.zeros(len(files)), np.zeros(len(files))
        for i, f in enumerate(files):
            B = f["summary"]["sqrt_sum"].to_numpy() / n_rows
            L = f["summary"]["total"].to_numpy()
            c = -(4 / d0) * B * self.L0 / L                              # c_s = level x base total / file total
            X[i, iu:iu + nS] = c
            X[i, ie:ie + nS] = c                                          # the public rows' luck
            if f["family"] != BASE:
                j = fams.index(f["family"])
                X[i, il + j * nS:il + (j + 1) * nS] = c                  # the family's own best totals
                X[i, iR + j] = 1.0                                        # and its ranking
            X[i, iK] = 1.0
            y[i] = f["public"] + (4 / d0) * B.sum()
            w[i] = 1 / f["noise_sd"] ** 2
        S0, m0 = np.zeros((nP, nP)), np.zeros(nP)
        m0[iu:iu + nS] = 1.0
        S0[iu:iu + nS, iu:iu + nS] = prior_cov(segments)
        S0[ie:ie + nS, ie:ie + nS] = np.diag(public_sd ** 2)
        S0[il:iK, il:iK] = np.eye(iK - il) * LINK_SD ** 2
        S0[iK, iK] = 100.0                                                # K (about 1.4): no real prior
        S0[iR:, iR:] = np.eye(len(fams)) * FAMILY_SD ** 2
        Sy = X @ S0 @ X.T + np.diag(1 / w)                               # evidence: how well this d0 explains the scores
        r = y - X @ m0
        self.log_evidence = float(-0.5 * (r @ np.linalg.solve(Sy, r) + np.linalg.slogdet(Sy)[1]))
        P0 = np.linalg.inv(S0)
        self.cov = np.linalg.inv(P0 + X.T @ (w[:, None] * X))
        self.mean = self.cov @ (P0 @ m0 + X.T @ (w * y))
        self.fitted = X @ self.mean - (y - np.array([f["public"] for f in files]))
        self.iu, self.ie, self.public_sd = iu, ie, public_sd
        self.u = self.mean[iu:iu + nS]                                    # best level on unseen rows
        self.u_pub = self.u + self.mean[ie:ie + nS]                       # best level on the public rows
        self.u_sd = np.sqrt(np.diag(self.cov)[iu:iu + nS])
        J = np.zeros((nS, nP))
        J[:, iu:iu + nS], J[:, ie:ie + nS] = np.eye(nS), np.eye(nS)
        self.u_pub_sd = np.sqrt(np.diag(J @ self.cov @ J.T))

    def gain(self, f: np.ndarray, board: str = "public") -> tuple[float, float]:
        """Expected change of the score, and its sd, when the base is scaled by f per segment.

        ``board="public"`` scores the public rows; ``"private"`` rows not yet seen, whose own luck
        averages out (the sd is then the uncertainty about the 2025 levels only)."""
        nS = len(self.segments)
        a = np.zeros(len(self.mean))
        coef = -(4 / self.d0) * self.B0 * (f ** -0.5 - 1)
        a[self.iu:self.iu + nS] = coef
        if board == "public":
            a[self.ie:self.ie + nS] = coef
        const = -(4 / self.d0) * np.sum(self.B0 * (f ** 0.5 - 1))
        return float(a @ self.mean + const), float(np.sqrt(max(a @ self.cov @ a, 0.0)))


def line_dispersion(segments: list[str]) -> dict[str, float]:
    """Tweedie dispersion of each line, from step 8's out-of-fold predictions of 2023-24."""
    members = {}
    summary = C.REPORTS / "08_geo" / "summary.json"
    if summary.exists():
        members = {r["line"]: r["members"].split("+") for r in json.loads(summary.read_text())["recalibration"]}
    out = {}
    for line in sorted({s.split("_")[0] for s in segments}):
        cv = pd.read_parquet(C.DATA / "experts" / f"{line}_cv.parquet")
        mu = G.geo_mean(cv, members.get(line, ["total"])) * G.reported_share(cv)
        y = cv["y"].to_numpy(dtype=float)
        m = mu * y.sum() / mu.sum()
        out[line] = float(np.mean((y - m) ** 2 / m ** 1.5))
    return out


def null_deviance_per_row(mu: np.ndarray, meta: pd.DataFrame) -> float:
    return S10.null_deviance(mu, meta["line"].to_numpy()) / len(mu)


def fit_levels(files, base_sum, segments, n_rows, public_sd, d0_center: float) -> Fit:
    """The fit at the null deviance with the highest posterior (scores x a prior around ``d0_center``)."""
    grid = d0_center * np.exp(np.linspace(np.log(0.7), np.log(1.6), 19))
    fits = [Fit(files, base_sum, segments, d, n_rows, public_sd) for d in grid]
    logpost = np.array([f.log_evidence for f in fits]) - 0.5 * (np.log(grid / d0_center) / D0_SD) ** 2
    best = fits[int(np.argmax(logpost))]
    best.d0_table = pd.DataFrame({"d0": grid, "log_posterior": logpost - logpost.max()})
    best.d0_center = d0_center
    return best


# =================================================================================================
# Run
# =================================================================================================
def known_score(version: str, scores_before: str | None) -> bool:
    """True when ``version``'s public score counts: all of them, or only those of versions made
    before ``scores_before`` (so an old version can be rebuilt with the scores known then)."""
    return scores_before is None or int(version.lstrip("v")) < int(scores_before.lstrip("v"))


def load_fit(base_version: str, public_share: float, scores_before: str | None = None):
    """Everything the modes need: rows, segments, the base, the scored files and the fit."""
    meta = S10.rows_2025()
    base, base_name = S10.load_version(base_version)
    meta = meta.loc[base.index]
    keys = segment_keys(meta)
    segments = sorted(set(keys))
    mu = base.to_numpy(dtype=float)
    base_sum = summarise(base, meta, keys, segments)
    files = []
    if EVIDENCE.exists():
        ev = pd.read_csv(EVIDENCE)
        for name, g in ev.groupby("file", sort=False):
            r = g.iloc[0]
            if r.get("use", "fit") == "fit":
                files.append({"name": name, "family": r["family"], "public": float(r["public"]),
                              "noise_sd": float(r["noise_sd"]), "summary": g.set_index("segment").reindex(segments)})
    reg = register_files(meta, keys, segments, base)
    for r in reg:                                   # only exact relatives of the base enter the fit
        if r["in_family"] and np.isfinite(r["public"]) and known_score(r["version"], scores_before):
            files.append({"name": r["version"], "family": BASE, "public": float(r["public"]),
                          "noise_sd": ROUNDING_SD, "summary": r["summary"]})
    phi = line_dispersion(segments)
    public_sd = np.array([np.sqrt(phi[s.split("_")[0]] / (public_share * base_sum.loc[s, "sqrt_sum"]))
                          for s in segments])
    # the null deviance prior is centred on the model's own value at the best public levels
    first = Fit(files, base_sum, segments, null_deviance_per_row(mu, meta), len(mu), public_sd)
    level = pd.Series(keys, index=meta.index).map(dict(zip(segments, first.u_pub))).loc[base.index].to_numpy()
    fit = fit_levels(files, base_sum, segments, len(mu), public_sd, null_deviance_per_row(mu * level, meta))
    fit.phi, fit.public_share = phi, public_share
    return meta, keys, segments, base, base_name, base_sum, reg, fit


def fit_base_score(fit: Fit) -> float:
    """The base's own public score (it is in the evidence)."""
    return next(f["public"] for f in fit.files if f["family"] == BASE and np.allclose(
        f["summary"]["total"].to_numpy(), fit.L0))


def report(segments, base_sum, reg, fit, base_name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base_score = fit_base_score(fit)
    print(f"public share of the 2025 rows (assumed): {fit.public_share:.0%}; claim dispersion "
          + ", ".join(f"{k} {v:,.0f}" for k, v in fit.phi.items()))
    print(f"null deviance per row: {fit.d0:.1f} (the model's own value at the best levels: {fit.d0_center:.1f}); "
          "log posterior by value:")
    print("   " + "  ".join(f"{d:.0f}:{p:.1f}" for d, p in zip(fit.d0_table["d0"], fit.d0_table["log_posterior"])))
    ev = pd.DataFrame({"file": [f["name"] for f in fit.files], "family": [f["family"] for f in fit.files],
                       "public": [f["public"] for f in fit.files], "fitted": fit.fitted})
    print("\nscored files and the model's fit of them:")
    print(ev.round(5).to_string(index=False))
    lv = pd.DataFrame({"segment": segments, "share_of_total": base_sum["total"].to_numpy() / base_sum["total"].sum(),
                       "B_per_row": fit.B0, "public_luck_sd": fit.public_sd, "best_public": fit.u_pub,
                       "sd_public": fit.u_pub_sd, "best_unseen": fit.u, "sd_unseen": fit.u_sd})
    print(f"\nbest level of each segment as a factor on {base_name}: on the public rows (what the leaderboard"
          " rewards) and on rows not yet scored (what the final ranking rewards):")
    print(lv.round(3).to_string(index=False))
    lv.to_csv(OUT / "levels.csv", index=False)
    ev.to_csv(OUT / "evidence_fit.csv", index=False)
    rows = [(r["version"], r["file"], r["public"], r["summary"]["total"].to_numpy() / base_sum["total"].to_numpy())
            for r in reg if r["in_family"]]
    rows += [("", "best levels for the public rows", np.nan, fit.u_pub),
             ("", "best levels for unseen rows", np.nan, fit.u)]
    out = []
    for version, name, public, f in rows:
        gp, sp = fit.gain(f, "public")
        gq, sq = fit.gain(f, "private")
        out.append({"version": version, "file": name, "public": public, "expected_public": base_score + gp,
                    "sd": sp, "unseen_change": gq, "unseen_sd": sq})
    t = pd.DataFrame(out)
    print(f"\nexpected scores ({base_name} = {base_score:.5f}); unseen_change is the expected change on rows not"
          " yet scored:")
    print(t.round(5).to_string(index=False))
    t.to_csv(OUT / "expected_scores.csv", index=False)


def make(args, meta, keys, segments, base, base_name, fit) -> int:
    if args.set:
        spec = {k: float(v) for k, v in (s.split("=") for s in args.set)}
        f = factor_vector(segments, spec)
        tag = args.name or "levels_" + "_".join(f"{k}{v:.2f}".replace(".", "p") for k, v in spec.items())
        what = ", ".join(f"{k} x{v:.3f}" for k, v in spec.items())
    else:
        target = fit.u_pub if args.target == "public" else fit.u
        f = target ** args.shrink
        tag = args.name or f"joint_levels_{args.target}" + (
            "" if args.shrink == 1 else f"_s{round(100 * args.shrink):03d}")
        rows = "the public rows" if args.target == "public" else "rows not yet scored"
        what = f"every segment at its best level for {rows} from step 11" + (
            "" if args.shrink == 1 else f", {args.shrink:.0%} of the way on the log scale")
    gp, sp = fit.gain(f, "public")
    gq, sq = fit.gain(f, "private")
    seg_of_row = pd.Series(keys, index=meta.index).map(dict(zip(segments, f)))
    sub = (base * seg_of_row.loc[base.index]).rename("prediction").reset_index()
    expected = S10.H.read_table("sample_submission", "clean", columns=["row_id"])["row_id"].astype(str)
    sub = sub.set_index("row_id").loc[expected.to_numpy()].reset_index()
    S10.P.check_submission(sub, expected)
    for s, x in zip(segments, f):
        print(f"   {s:20s} x{x:.3f}")
    evidence = (f"expected public {fit_base_score(fit) + gp:.5f} (sd {sp:.5f}); unseen rows {gq:+.5f} "
                f"(sd {sq:.5f}) vs {base_name}")
    print(evidence)
    S.save(sub, C.ROOT, tag, "11_joint_levels.py", f"{base_name} with {what}", evidence)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["fit", "make", "evidence"])
    ap.add_argument("--base", default=BASE, help="version the levels are read against (default v5)")
    ap.add_argument("--target", default="private", choices=["private", "public"],
                    help="make: levels for rows not yet scored (default) or for the public rows")
    ap.add_argument("--shrink", type=float, default=1.0, help="make: share of the log move to the target level")
    ap.add_argument("--set", nargs="*", help="make: explicit factors, e.g. motor_m=1.08 contents_f=1.04")
    ap.add_argument("--name", default=None, help="make: model name in the register")
    ap.add_argument("--public-share", type=float, default=PUBLIC_SHARE,
                    help="share of the 2025 rows behind the public score (the leaderboard page states it)")
    ap.add_argument("--ref-repo", type=Path, help="evidence: the pricing repository (iaa-pricing-part2)")
    ap.add_argument("--scores-before", default=None, metavar="VERSION",
                    help="use only the public scores of versions made before this one (v12 rebuilds v12 and v13)")
    a = ap.parse_args(argv)
    if a.mode == "evidence":
        if a.ref_repo is None:
            ap.error("evidence needs --ref-repo")
        meta = S10.rows_2025()
        keys = segment_keys(meta)
        ev = build_evidence(a.ref_repo, meta, keys, sorted(set(keys)))
        EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
        ev.to_csv(EVIDENCE, index=False)
        print(f"{ev['file'].nunique()} scored files, {len(ev)} rows -> {EVIDENCE}")
        return 0
    meta, keys, segments, base, base_name, base_sum, reg, fit = load_fit(a.base, a.public_share, a.scores_before)
    if a.mode == "fit":
        report(segments, base_sum, reg, fit, base_name)
        return 0
    return make(a, meta, keys, segments, base, base_name, fit)


if __name__ == "__main__":
    sys.exit(main())
