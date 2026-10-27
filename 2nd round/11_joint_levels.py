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

Unknowns: u_s, the best total of segment s over the base's (prior: a line level, a new-business
effect and a gender effect per line, sd ``PRIOR_SD``), K, and the family terms. The model is
linear in them, so the posterior is exact.

    fit:       python 11_joint_levels.py fit
    make:      python 11_joint_levels.py make --shrink 0.5
               python 11_joint_levels.py make --set motor_m=1.08 motor_f=0.93 contents_m=0.90 contents_f=1.04
    evidence:  python 11_joint_levels.py evidence --ref-repo "C:/Users/.../iaa-pricing-part2"

A ``--set`` key names a segment or a group of segments by its parts: ``motor_new`` is both motor
new-business segments, ``contents_m`` both contents male segments, ``buildings`` the whole line.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

sys.path.insert(0, str(C.ROOT.parent))            # the shared modules sit in the folder above
import submissions as S  # noqa: E402

S10 = importlib.import_module("10_level_probe")   # 2025 rows, versions, null deviance, checks
EVIDENCE = C.DATA / "reference" / "leaderboard_evidence.csv"
OUT = C.REPORTS / "11_joint"
BASE = "v5"                       # the family the levels are read against (the best file's levels)
GENDER_LINES = ("motor", "contents")
ROUNDING_SD = 1.5e-5              # public scores have five decimals; a little room for rounding
PATH_SD = 2e-4                    # blends of the trio's experts: small ranking changes on top of levels
FAMILY_SD = 1e-3                  # how much a different ranking can move the public score
PRIOR_SD = {                      # prior sd of the log best level vs the base, per effect
    "line": {"motor": 0.05, "buildings": 0.08, "contents": 0.08},
    "new_business": 0.12,
    "gender": 0.15,               # men vs women; the leaderboard has shown that gender moves the score
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
    """Posterior of u (best segment total over the base's), K and the family terms."""

    def __init__(self, files: list[dict], base: pd.DataFrame, segments: list[str], d0: float, n_rows: int):
        self.segments, self.d0, self.n = segments, d0, n_rows
        self.B0 = base["sqrt_sum"].to_numpy() / n_rows                  # the base's B per segment
        self.L0 = base["total"].to_numpy()
        fams = sorted({f["family"] for f in files if f["family"] != BASE})
        nS, nF = len(segments), len(fams)
        X, y, w = np.zeros((len(files), nS + 1 + nF)), np.zeros(len(files)), np.zeros(len(files))
        for i, f in enumerate(files):
            B = f["summary"]["sqrt_sum"].to_numpy() / n_rows
            L = f["summary"]["total"].to_numpy()
            X[i, :nS] = -(4 / d0) * B * self.L0 / L                     # c_s = u_s x base total / file total
            X[i, nS] = 1.0                                                # K
            if f["family"] != BASE:
                X[i, nS + 1 + fams.index(f["family"])] = 1.0             # the family's ranking term
            y[i] = f["public"] + (4 / d0) * B.sum()
            w[i] = 1 / f["noise_sd"] ** 2
        P0 = np.zeros((nS + 1 + nF, nS + 1 + nF))
        P0[:nS, :nS] = np.linalg.inv(prior_cov(segments))
        P0[nS, nS] = 1e-12                                                # K: no prior
        P0[nS + 1:, nS + 1:] = np.eye(nF) / FAMILY_SD ** 2
        m0 = np.r_[np.ones(nS), 0.0, np.zeros(nF)]
        prec = P0 + X.T @ (w[:, None] * X)
        self.cov = np.linalg.inv(prec)
        self.mean = self.cov @ (P0 @ m0 + X.T @ (w * y))
        self.files, self.fitted = files, X @ self.mean - (y - np.array([f["public"] for f in files]))
        self.u, self.u_sd = self.mean[:nS], np.sqrt(np.diag(self.cov)[:nS])

    def predict(self, summary: pd.DataFrame) -> float:
        """Expected public score of a file of another family (its family term at 0)."""
        B = summary["sqrt_sum"].to_numpy() / self.n
        L = summary["total"].to_numpy()
        K = self.mean[len(self.segments)]
        return float(K - (4 / self.d0) * np.sum(B * (self.u * self.L0 / L + 1)))

    def gain(self, f: np.ndarray) -> tuple[float, float]:
        """Expected change of the public score, and its sd, when the base is scaled by f per segment."""
        a = np.zeros(len(self.mean))
        a[:len(self.segments)] = -(4 / self.d0) * self.B0 * (f ** -0.5 - 1)
        const = -(4 / self.d0) * np.sum(self.B0 * (f ** 0.5 - 1))
        return float(a @ self.mean + const), float(np.sqrt(a @ self.cov @ a))


# =================================================================================================
# Run
# =================================================================================================
def load_fit(base_version: str):
    """Everything the modes need: rows, segments, the base, the scored files and the fit."""
    meta = S10.rows_2025()
    base, base_name = S10.load_version(base_version)
    meta = meta.loc[base.index]
    keys = segment_keys(meta)
    segments = sorted(set(keys))
    mu = base.to_numpy(dtype=float)
    d0 = S10.null_deviance(mu, meta["line"].to_numpy()) / len(mu)
    base_sum = summarise(base, meta, keys, segments)
    files, checks = [], []
    if EVIDENCE.exists():
        ev = pd.read_csv(EVIDENCE)
        for name, g in ev.groupby("file", sort=False):
            r = g.iloc[0]
            item = {"name": name, "family": r["family"], "public": float(r["public"]),
                    "noise_sd": float(r["noise_sd"]), "summary": g.set_index("segment").reindex(segments)}
            (checks if r.get("use", "fit") == "check" else files).append(item)
    reg = register_files(meta, keys, segments, base)
    for r in reg:                                   # only exact relatives of the base enter the fit
        if r["in_family"] and np.isfinite(r["public"]):
            files.append({"name": r["version"], "family": BASE, "public": float(r["public"]),
                          "noise_sd": ROUNDING_SD, "summary": r["summary"]})
    fit = Fit(files, base_sum, segments, d0, len(mu))
    fit.checks = checks
    return meta, keys, segments, base, base_name, base_sum, reg, fit


def report(segments, base_sum, reg, fit, base_name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ev = pd.DataFrame({"file": [f["name"] for f in fit.files], "family": [f["family"] for f in fit.files],
                       "public": [f["public"] for f in fit.files], "fitted": fit.fitted})
    print(f"null deviance per row (model-based, on {base_name}): {fit.d0:.1f}")
    print("\nscored files and the model's fit of them:")
    print(ev.round(5).to_string(index=False))
    if fit.checks:
        ck = pd.DataFrame({"file": [c["name"] for c in fit.checks], "public": [c["public"] for c in fit.checks],
                           "predicted": [fit.predict(c["summary"]) for c in fit.checks]})
        ck["miss"] = ck["public"] - ck["predicted"]
        print("\ncheck: files the fit does not use (their ranking differs too), predicted from the levels alone:")
        print(ck.round(5).to_string(index=False))
        ck.to_csv(OUT / "check_files.csv", index=False)
    rows = []
    for i, s in enumerate(segments):
        f = np.ones(len(segments))
        f[i] = fit.u[i]
        g, _ = fit.gain(f)
        rows.append({"segment": s, "share_of_total": base_sum["total"].iloc[i] / base_sum["total"].sum(),
                     "B_per_row": fit.B0[i], "best_vs_base": fit.u[i], "sd": fit.u_sd[i], "gain_alone": g})
    lv = pd.DataFrame(rows)
    print(f"\nbest level of each segment, as a factor on {base_name} (posterior mean and sd):")
    print(lv.round(4).to_string(index=False))
    lv.to_csv(OUT / "levels.csv", index=False)
    ev.to_csv(OUT / "evidence_fit.csv", index=False)
    print(f"\nexpected public score of each version not yet scored ({base_name} = {fit_base_score(fit):.5f}):")
    for r in reg:
        if not r["in_family"] or np.isfinite(r["public"]):
            continue
        f = r["summary"]["total"].to_numpy() / base_sum["total"].to_numpy()
        g, sd = fit.gain(f)
        print(f"   {r['version']:4s} {r['file']:42s} {fit_base_score(fit) + g:.5f}  ({g:+.5f}, sd {sd:.5f})")
    for label, f in (("every segment at its best level", fit.u), ("half way (log scale)", fit.u ** 0.5)):
        g, sd = fit.gain(f)
        print(f"   {label:47s} {fit_base_score(fit) + g:.5f}  ({g:+.5f}, sd {sd:.5f})")


def fit_base_score(fit: Fit) -> float:
    """The base's own public score (it is in the evidence)."""
    return next(f["public"] for f in fit.files if f["family"] == BASE and np.allclose(
        f["summary"]["total"].to_numpy(), fit.L0))


def make(args, meta, keys, segments, base, base_name, fit) -> int:
    if args.set:
        spec = {k: float(v) for k, v in (s.split("=") for s in args.set)}
        f = factor_vector(segments, spec)
        tag = args.name or "levels_" + "_".join(f"{k}{v:.2f}".replace(".", "p") for k, v in spec.items())
        what = ", ".join(f"{k} x{v:.3f}" for k, v in spec.items())
    else:
        f = fit.u ** args.shrink
        tag = args.name or f"joint_levels_s{round(100 * args.shrink):03d}"
        what = f"every segment moved {args.shrink:.0%} of the way (log scale) to its best level from step 11"
    g, sd = fit.gain(f)
    seg_of_row = pd.Series(keys, index=meta.index).map(dict(zip(segments, f)))
    sub = (base * seg_of_row.loc[base.index]).rename("prediction").reset_index()
    expected = S10.H.read_table("sample_submission", "clean", columns=["row_id"])["row_id"].astype(str)
    sub = sub.set_index("row_id").loc[expected.to_numpy()].reset_index()
    S10.P.check_submission(sub, expected)
    for s, x in zip(segments, f):
        print(f"   {s:20s} x{x:.3f}")
    print(f"expected public {fit_base_score(fit) + g:.5f} ({g:+.5f}, sd {sd:.5f})")
    S.save(sub, C.ROOT, tag, "11_joint_levels.py", f"{base_name} with {what}",
           f"expected public {fit_base_score(fit) + g:.5f} (sd {sd:.5f}) from every scored file")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["fit", "make", "evidence"])
    ap.add_argument("--base", default=BASE, help="version the levels are read against (default v5)")
    ap.add_argument("--shrink", type=float, default=0.5, help="make: share of the log move to the best level")
    ap.add_argument("--set", nargs="*", help="make: explicit factors, e.g. motor_m=1.08 contents_f=1.04")
    ap.add_argument("--name", default=None, help="make: model name in the register")
    ap.add_argument("--ref-repo", type=Path, help="evidence: the pricing repository (iaa-pricing-part2)")
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
    meta, keys, segments, base, base_name, base_sum, reg, fit = load_fit(a.base)
    if a.mode == "fit":
        report(segments, base_sum, reg, fit, base_name)
        return 0
    return make(a, meta, keys, segments, base, base_name, fit)


if __name__ == "__main__":
    sys.exit(main())
