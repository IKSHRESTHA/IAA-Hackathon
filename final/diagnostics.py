"""Model diagnostics for the two submitted files: where the models are right, and what drives them.

    python final/diagnostics.py          about 5 minutes; writes final/3_diagnostics/

Calibration and lift, both weeks. Out-of-time predictions are cut into ten bands of equal exposure,
ordered by predicted cost per year on cover. Each band shows its predicted and actual cost per year
and their ratio (A/E). A model that ranks risks well shows actual cost rising from band 1 to band 10;
a well-calibrated one keeps A/E close to its overall value in every band.
    Week 1: the submitted six-expert ensemble, recalibrated on one back-test year and scored on the
            other, from the experts' cached predictions for 2023 and 2024.
    Week 2: the step-8 ensemble on the 2024 forward test (fitted on 2023) and on the 2023-24
            out-of-fold rows, plus A/E by segment (line x new business x gender).

SHAP, Week 2. Exact TreeSHAP values from LightGBM (pred_contrib) on the 2025 rows, for the
developed-market prior (step 4) and for the LightGBM fitted on top of it (step 5). They are on the
log scale: a SHAP value of 0.1 means about 10% more expected cost for that row. Both models are
refitted with the steps' own code and settings, and the prior is checked against the one stored.

GLM, Week 2. The GLM expert of step 7, refitted on 2023-24 with its tuned ridge: every factor level's
relativity and the exposure behind it. Its predictions are checked against the stored ones. The
ridge pulls the coefficients towards the developed-market prior, so textbook standard errors do not
apply; the exposure column shows how much data each level rests on.

SHAP, Week 1. The Week 1 code saved its experts' predictions but not their trees, so the
total-cost GBM expert (the ensemble's main tree model) is refitted here with that code's own
functions, saved to final/3_diagnostics/week1_models/ and checked against its cached 2025
predictions. The first run takes about an hour for this; later runs load the saved trees.

    python final/diagnostics.py --parts lift glm     run only some parts (lift, shap, glm, week1)
    python final/diagnostics.py --refit-week1        refit the Week 1 trees even if they are saved
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FINAL = Path(__file__).resolve().parent
REPO = FINAL.parent
R2 = REPO / "2nd round"
W1 = REPO / "1st round" / "IAA_Part1_geo_models"
OUT = FINAL / "3_diagnostics"
BANDS = 10
TOP = 8                                   # features shown per line in the SHAP charts

sys.dont_write_bytecode = True            # leave no __pycache__ in the round folders
sys.path[:0] = [str(FINAL), str(FINAL / "src"), str(R2), str(REPO), str(W1), str(W1 / "src")]
import final_figures as ff  # noqa: E402  chart style and colours shared with the write-up
import config as C  # noqa: E402  the 2nd round's settings
import geo_experts as G  # noqa: E402
import helpers as H  # noqa: E402
import pipeline as PL  # noqa: E402  the final pipeline: Week 1's file list
import pricing_lgbm as P  # noqa: E402

S4 = importlib.import_module("04_developed_prior")
S5 = importlib.import_module("05_lightgbm_model")
S7 = importlib.import_module("07_geo_experts")
RB = importlib.import_module("rebuild_from_cache")      # Week 1's ensemble stage
WEEK1_MODELS = OUT / "week1_models"                      # the refitted Week 1 trees
WEEK1_TARGET_ENCODE = {"motor": ["make_model"]}           # as in the Week 1 expert step


# ------------------------------------------------------------------------------ calibration and lift

def lift(frame: pd.DataFrame, check: str) -> pd.DataFrame:
    """Ten bands of equal exposure, ordered by predicted cost per year on cover."""
    y, mu, expo = (frame[c].to_numpy(dtype=float) for c in ("y", "mu", "exposure"))
    order = np.argsort(mu / expo, kind="stable")
    share = np.cumsum(expo[order]) / expo.sum()
    band = np.clip(np.ceil(share * BANDS), 1, BANDS).astype(int)
    g = pd.DataFrame({"band": band, "y": y[order], "mu": mu[order], "exposure": expo[order]}).groupby("band").sum()
    return pd.DataFrame({
        "check": check, "band": g.index,
        "predicted_per_year": g["mu"] / g["exposure"],
        "actual_per_year": g["y"] / g["exposure"],
        "a_over_e": g["y"] / g["mu"],
    }).reset_index(drop=True)


def summary_row(frame: pd.DataFrame, bands: pd.DataFrame, check: str) -> dict:
    y, mu = frame["y"].to_numpy(dtype=float), frame["mu"].to_numpy(dtype=float)
    ae = bands["a_over_e"].to_numpy()
    return {"check": check, "rows": len(frame), "d2": P.d2_tweedie(y, mu), "a_over_e": y.sum() / mu.sum(),
            "lift_top_over_bottom": bands["actual_per_year"].iloc[-1] / bands["actual_per_year"].iloc[0],
            "band_a_over_e_min": ae.min(), "band_a_over_e_max": ae.max()}


def week1_backtests() -> dict[str, pd.DataFrame]:
    """The submitted ensemble on each back-test year, recalibrated on the other year (as when it was built)."""
    members = RB.MODELS["submission_week1_v4_geo6.csv"]
    parts = {2023: [], 2024: []}
    for line in RB.LINES:
        c = pd.read_parquet(W1 / "cache" / f"experts7_{line}.parquet")
        h23, h24 = (RB.with_geo(c[c["set"] == s].reset_index(drop=True), members) for s in ("ho2023", "ho2024"))
        for fit_d, sc_d, year in ((h23, h24, 2024), (h24, h23, 2023)):
            parts[year].append(pd.DataFrame({"line": line, "y": sc_d["y"].to_numpy(dtype=float),
                                             "mu": RB.pool().fit(fit_d).predict(sc_d),
                                             "exposure": sc_d["exposure"].to_numpy(dtype=float)}))
    return {f"Week 1, back-test {year}": pd.concat(p, ignore_index=True) for year, p in parts.items()}


def week2_tests() -> dict[str, pd.DataFrame]:
    """The step-8 ensemble's predictions on the forward test and the out-of-fold rows, reported basis."""
    members = json.loads((C.REPORTS / "08_geo" / "summary.json").read_text(encoding="utf-8"))["members"]
    out = {}
    for name, label in (("ho2024", "Week 2, forward test 2024"), ("cv", "Week 2, out of fold 2023-24")):
        parts = []
        for line in C.LINES:
            d = pd.read_parquet(C.DATA / "experts" / f"{line}_{name}.parquet")
            mu = G.geo_mean(d, members[line]) * G.reported_share(d)
            parts.append(d[["year", "exposure", "_new_business", "_gender", "y", "n_claims"]].assign(line=line, mu=mu))
        out[label] = pd.concat(parts, ignore_index=True)
    return out


def segments(frame: pd.DataFrame, check: str) -> pd.DataFrame:
    """A/E by line x new business x gender (gender only where it is priced: motor and contents)."""
    seg = frame["line"] + np.where(frame["_new_business"], ", new business", ", renewals")
    priced = frame["line"].isin(["motor", "contents"])
    seg = seg.where(~priced, seg + ", " + frame["_gender"].map({"m": "men", "f": "women"}).fillna("unknown"))
    g = frame.assign(segment=seg).groupby("segment")[["y", "mu", "exposure", "n_claims"]].sum()
    return pd.DataFrame({"check": check, "segment": g.index, "claims": g["n_claims"].astype(int),
                         "exposure_share": g["exposure"] / g["exposure"].sum(),
                         "a_over_e": g["y"] / g["mu"]}).reset_index(drop=True)


# ------------------------------------------------------------------------------------------- SHAP

def shap_values(model: P.TweedieLGBM, X: pd.DataFrame) -> pd.DataFrame:
    """Exact TreeSHAP per row and feature; the extra column ``base`` is the model's expected value."""
    contrib = model.booster.predict(X[model.features], pred_contrib=True, num_iteration=model.n_trees)
    return pd.DataFrame(contrib, columns=[*model.features, "base"], index=X.index)


def shap_table(shap: pd.DataFrame, X: pd.DataFrame, model_name: str, line: str) -> pd.DataFrame:
    """Mean |SHAP| per feature, with the direction of its effect."""
    rows = []
    for f in shap.columns.drop("base"):
        s, v = shap[f], X[f]
        if isinstance(v.dtype, pd.CategoricalDtype):
            by_level = s.groupby(v.astype("string").fillna("missing")).mean()
            direction = f"highest: {by_level.idxmax()}, lowest: {by_level.idxmin()}"
        else:
            ok = v.notna().to_numpy()
            x, sv = v.to_numpy(dtype=float)[ok], s.to_numpy()[ok]
            if len(np.unique(x)) < 2 or np.ptp(sv) == 0:
                direction = "no variation in the 2025 rows"
            else:
                rho = pd.Series(x).corr(pd.Series(sv), method="spearman")
                direction = (f"higher value, higher cost (rank corr {rho:+.2f})" if rho > 0
                             else f"higher value, lower cost (rank corr {rho:+.2f})")
        rows.append({"model": model_name, "line": line, "feature": f, "mean_abs_shap": float(s.abs().mean()),
                     "direction": direction})
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False, ignore_index=True)


def prior_shap(line: str) -> tuple[pd.DataFrame, float]:
    """Refit step 4's developed prior, check it against the stored one, and explain it on the 2025 rows."""
    dev = S4.harmonise(*S4.read_market(C.DEVELOPED, line, ("policies",)), line, developed=True)
    perils = [f"cost_{p}" for p in S4.C1.PERILS[line]]
    targets = pd.read_parquet(S4.DEV_DATA / "features" / f"policies_{line}.parquet",
                              columns=["row_id", "comp_cost", *perils])
    dev = dev.merge(targets, on="row_id", how="inner", validate="one_to_one")
    em = S4.harmonise(*S4.read_market(C.ROOT, line, ("policies", "test")), line, developed=False)
    features, _ = S4.prior_features(line, dev, em)
    model, levels = S4.fit_prior(line, dev, features)
    del dev
    stored = pd.read_parquet(C.DATA / "prior" / f"prior_{line}.parquet").set_index("row_id")["log_prior"]
    lp = S4.log_prior(model, levels, em)
    gap = float(np.abs(lp - stored.loc[em["row_id"].astype(str)].to_numpy()).max())
    rows = em.loc[em["row_id"].astype(str).isin(H.read_table(f"test_{line}", "features", columns=["row_id"])["row_id"].astype(str))]
    X, _ = P.prepare(rows, model.features, levels)
    return shap_table(shap_values(model, X), X, "developed prior (step 4)", line), gap


def transfer_shap(line: str) -> tuple[pd.DataFrame, float]:
    """Refit step 5's LightGBM on top of the prior and explain it on the 2025 rows."""
    train, test, features = S5.load_line(line)
    fitted = S5.fit_model(train, line, features, C.TEST_YEAR, use_prior=True, refit=True)
    model = fitted.lf.model
    X, _ = P.prepare(test, model.features, fitted.lf.levels)
    shap = shap_values(model, X)
    off = S5.offset(test, fitted.centre, fitted.use_prior, reported=False)
    check = np.abs(shap.sum(axis=1).to_numpy() + model.base + off - np.log(model.predict(X, off))).max()
    return shap_table(shap, X, "LightGBM on the prior (step 5)", line), float(check)


# -------------------------------------------------------------------------------------------- GLM

def glm_tables() -> tuple[pd.DataFrame, dict[str, float]]:
    """Refit step 7's GLM expert for the lines whose ensemble uses it; relativities and a check."""
    members = json.loads((C.REPORTS / "08_geo" / "summary.json").read_text(encoding="utf-8"))["members"]
    claims = H.read_table("claims", "clean")
    dev = G.development_params(claims, C.DEV_MATURE_YEARS, C.DEV_IMMATURE_YEARS, C.REPUDIATION_MIN_Z)
    claims_ult = G.claims_to_ultimate(claims, dev)
    nonrep = claims.loc[claims["status"] != "repudiated"]
    count_pattern = H.reporting_pattern(nonrep, C.VALUATION_DATE, f"{min(C.TRAIN_YEARS)}-01-01")
    tables, gaps = [], {}
    for line in [ln for ln in C.LINES if "glm" in members[ln]]:
        settings = json.loads((C.REPORTS / "07_geo" / f"experts_{line}_final.json").read_text(encoding="utf-8"))
        glm_settings = {**C.GEO_GLM, "alpha": settings["glm_alpha"]}
        train_all, test, features, keys = S7.load_line(line, count_pattern)
        tg = G.build_targets(train_all, claims_ult, line, C.PERILS[line], C.TREND, C.LEVEL_SHIFT,
                             C.TEST_YEAR, C.CAP_QUANTILE, list(C.TRAIN_YEARS))
        rows = train_all.merge(tg.rows, on="row_id", how="left", validate="one_to_one")
        d = S7.make_fit(line, rows, test, features, keys, tg, C.TEST_YEAR, settings["params"], final=True)
        stored = pd.read_parquet(C.DATA / "experts" / f"{line}_final.parquet")["glm"].to_numpy(dtype=float)
        gaps[line] = float(np.abs(G.expert_glm(d, glm_settings)[0]["glm"] / stored - 1).max())
        numeric = [c for c in d.X.columns if not isinstance(d.X[c].dtype, pd.CategoricalDtype)]
        categorical = [c for c in d.X.columns if c not in numeric]
        y, off = d.train["y_capped"].to_numpy(dtype=float), d.offset("comp_cost", "cost")
        expo = d.train["exposure"].to_numpy(dtype=float)
        g = G.BandedGLM(numeric, categorical, **glm_settings).fit(d.X, y, off, expo)
        design = g.design(d.X)
        coef = pd.Series(g.model_.coef_, index=g.model_.feature_names_)
        for factor in design.columns:
            level_expo = pd.Series(expo).groupby(design[factor].astype(str).to_numpy()).sum()
            for level in design[factor].cat.categories:
                name = next((n for n in coef.index if n in (f"{factor}__{level}", f"{factor}[{level}]")), None)
                if name is None or level_expo.get(level, 0.0) == 0:
                    continue
                tables.append({"line": line, "factor": factor, "level": level, "coefficient": coef[name],
                               "relativity": float(np.exp(coef[name])), "years_on_cover": float(level_expo[level])})
    out = pd.DataFrame(tables)
    if len(out):    # relativities within each factor, relative to its largest level by exposure
        ref = out.loc[out.groupby(["line", "factor"])["years_on_cover"].idxmax()].set_index(["line", "factor"])["coefficient"]
        out["relativity_to_main_level"] = np.exp(out["coefficient"] - out.set_index(["line", "factor"]).index.map(ref))
    return out, gaps


# ------------------------------------------------------------------------------- Week 1 trees

def week1_inputs() -> None:
    """The competition files and features the Week 1 code reads, made if missing (as pipeline.py --full)."""
    data = W1 / "data" / "week1"
    for name in PL.WEEK1_DATA:
        if not (data / f"{name}.parquet").exists():
            shutil.copyfile(REPO / "1st round" / "data" / "raw" / f"{name}.parquet", data / f"{name}.parquet")
    if not all((W1 / "artifacts" / "week1" / f"features_{line}.parquet").exists() for line in RB.LINES):
        subprocess.run([sys.executable, "run_pipeline.py", "--only", "features"], cwd=W1, check=True,
                       env=dict(os.environ, PYTHONHASHSEED="0", PYTHONPATH=str(W1 / "src")))


def week1_total_expert(refit: bool) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Refit (or load) Week 1's total-cost GBM expert per line, check it, and explain it on the 2025 rows."""
    from iaa_pricing.claims import prepare_claims
    from iaa_pricing.config import load_config
    from iaa_pricing.experts import _fit_es
    from iaa_pricing.io import read_table
    from iaa_pricing.modeling import balance_factor, load_line_data
    from iaa_pricing.models.gbm import TweedieGBM
    from iaa_pricing.targets import DevelopmentInputs

    week1_inputs()
    cfg = load_config(W1 / "configs" / "week1.yaml")
    devin = DevelopmentInputs.fit(prepare_claims(read_table(cfg.data_dir, "claims")), cfg)
    tables, checks = [], {}
    for line in RB.LINES:
        print(f"   Week 1 {line}: total-cost expert", flush=True)
        data = load_line_data(line, cfg, devin, target_year=cfg.prediction_year, cap_quantile=0.995,
                              target_encode=WEEK1_TARGET_ENCODE.get(line))
        train, score = data.train, data.test
        y, off = train["y_capped"].to_numpy(), train["log_offset"].to_numpy()
        folder = WEEK1_MODELS / line
        if refit or not (folder / "booster.txt").exists():
            model = _fit_es(data.spec, cfg, train, y, off, {}, refit=True, seed=cfg.seed)   # as the expert step
            model.save(folder)
        else:
            model = TweedieGBM.load(folder)
        trend = score["freq_trend"].to_numpy() if "freq_trend" in score else np.ones(len(score))
        pred = (model.predict(score, np.log(score["exposure"].to_numpy() * trend))
                * balance_factor(y, model.predict(train, off)) * float(data.target_summary["large_loss_load"]))
        cached = pd.read_parquet(W1 / "cache" / f"experts7_{line}.parquet", columns=["row_id", "set", "total"])
        ref = cached.loc[cached["set"] == "final"].set_index("row_id")["total"].loc[score["row_id"]].to_numpy()
        checks[line] = {"trees": int(model.best_iteration_), "max_rel_diff": float(np.abs(pred / ref - 1).max()),
                        "log_corr": float(np.corrcoef(np.log(pred), np.log(ref))[0, 1])}
        shap = model.contributions(score).rename(columns={"bias": "base"})
        tables.append(shap_table(shap, score[model.features], "total-cost GBM expert (Week 1)", line))
    return pd.concat(tables, ignore_index=True), checks


# ----------------------------------------------------------------------------------------- charts

def chart_lift(bands: pd.DataFrame, checks: list[str], path: Path) -> None:
    ff.use_style()
    fig, axes = plt.subplots(1, len(checks), figsize=(6.6, 2.6), layout="constrained", sharey=False)
    for ax, check in zip(np.atleast_1d(axes), checks):
        b = bands.loc[bands["check"] == check]
        scale = b["actual_per_year"].mean()
        ax.plot(b["band"], b["predicted_per_year"] / scale, color=ff.BLUE, lw=2, marker="o", ms=5, label="predicted")
        ax.plot(b["band"], b["actual_per_year"] / scale, color=ff.ORANGE, lw=0, marker="D", ms=5, label="actual")
        ax.set_title(check, fontsize=8.5)
        ax.set_xticks(range(1, BANDS + 1))
        ax.set_xlabel("band of predicted cost (1 = lowest)")
    np.atleast_1d(axes)[0].set_ylabel("cost per year on cover\n(average = 1)")
    fig.legend(*np.atleast_1d(axes)[0].get_legend_handles_labels(), loc="outside lower center", ncols=2)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def chart_shap(table: pd.DataFrame, model_name: str, path: Path, week: str = "Week 2") -> None:
    ff.use_style()
    lines = list(dict.fromkeys(table["line"]))
    fig, axes = plt.subplots(1, len(lines), figsize=(6.6, 2.9), layout="constrained")
    for ax, line in zip(np.atleast_1d(axes), lines):
        t = table.loc[table["line"] == line].head(TOP)[::-1]
        ax.barh(t["feature"], t["mean_abs_shap"], color=ff.BLUE, height=0.6)
        ax.set_title(line, fontsize=8.5)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("mean |SHAP| (log cost)", fontsize=7.5)
    fig.suptitle(f"{week}, {model_name}: top features on the 2025 rows", fontsize=9, x=0.01, ha="left",
                 fontweight="bold")
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------------------- report

def md(df: pd.DataFrame, floats: str = "{:.3f}") -> str:
    shown = df.copy()
    for c in shown.columns:
        if pd.api.types.is_float_dtype(shown[c]):
            shown[c] = shown[c].map(floats.format)
    return shown.to_markdown(index=False, disable_numparse=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--parts", nargs="*", default=["lift", "shap", "glm", "week1"],
                    choices=["lift", "shap", "glm", "week1"])
    ap.add_argument("--refit-week1", action="store_true", help="refit the Week 1 trees even if they are saved")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    report = ["# Model diagnostics", "",
              "Written by `final/diagnostics.py`, which explains each check. Every number comes from the "
              "pipeline's own data and code.", ""]
    checks: dict[str, float] = {}
    week1_checks: dict[str, dict] = {}

    if "lift" in a.parts:
        print("calibration and lift ...", flush=True)
        tests = {**week1_backtests(), **week2_tests()}
        bands = pd.concat([lift(f, k) for k, f in tests.items()], ignore_index=True)
        summary = pd.DataFrame([summary_row(f, bands.loc[bands["check"] == k], k) for k, f in tests.items()])
        segs = pd.concat([segments(tests[k], k) for k in ("Week 2, forward test 2024", "Week 2, out of fold 2023-24")],
                         ignore_index=True)
        bands.to_csv(OUT / "lift_bands.csv", index=False, lineterminator="\n")
        summary.to_csv(OUT / "calibration_summary.csv", index=False, lineterminator="\n")
        segs.to_csv(OUT / "segments_ae.csv", index=False, lineterminator="\n")
        chart_lift(bands, ["Week 1, back-test 2024", "Week 2, forward test 2024"], OUT / "lift.png")
        report += ["## Calibration and lift", "", "![lift](lift.png)", "", md(summary), "",
                   "`lift_bands.csv` has every band. A/E below 1 means the model predicted more than came in.", "",
                   "## A/E by segment, Week 2", "", md(segs), ""]

    shap_parts = []
    if "shap" in a.parts:
        print("SHAP: developed prior (refit on every 1st-round row) ...", flush=True)
        prior = [prior_shap(line) for line in C.LINES]
        print("SHAP: LightGBM on the prior ...", flush=True)
        transfer = [transfer_shap(line) for line in C.LINES]
        checks["prior refit against the stored prior (largest log difference)"] = max(g for _, g in prior)
        checks["SHAP additivity, step 5: SHAP sum + offset against the log prediction"] = max(g for _, g in transfer)
        chart_shap(pd.concat([t for t, _ in prior]), "developed prior (step 4)", OUT / "shap_prior.png")
        chart_shap(pd.concat([t for t, _ in transfer]), "LightGBM on the prior (step 5)", OUT / "shap_transfer.png")
        shap_parts += [t for t, _ in prior + transfer]
        report += ["## SHAP, Week 2", "", "![shap prior](shap_prior.png)", "", "![shap transfer](shap_transfer.png)", ""]

    if "week1" in a.parts:
        print("SHAP: Week 1 total-cost expert ...", flush=True)
        week1, week1_checks = week1_total_expert(a.refit_week1)
        chart_shap(week1, "total-cost GBM expert", OUT / "shap_week1.png", week="Week 1")
        shap_parts.append(week1)
        report += ["## SHAP, Week 1", "", "![shap week 1](shap_week1.png)", "",
                   "The refitted trees against the expert's cached 2025 predictions:", "",
                   md(pd.DataFrame([{"line": k, **v} for k, v in week1_checks.items()]), "{:.2e}"), ""]

    if shap_parts:
        shap = pd.concat(shap_parts, ignore_index=True)
        shap.to_csv(OUT / "shap_importance.csv", index=False, lineterminator="\n")
        report += ["The three strongest features of each model and line (all of them in `shap_importance.csv`):", "",
                   md(shap.groupby(["model", "line"]).head(3)), ""]

    if "glm" in a.parts:
        print("GLM relativities ...", flush=True)
        glm, glm_gaps = glm_tables()
        glm.to_csv(OUT / "glm_relativities.csv", index=False, lineterminator="\n")
        for line, v in glm_gaps.items():
            checks[f"GLM refit against the stored {line} GLM predictions (largest relative difference)"] = v
        report += ["## GLM relativities, Week 2", "",
                   f"The GLM expert is in the ensemble for {', '.join(sorted(glm['line'].unique()))}. "
                   "`glm_relativities.csv` lists every factor level with its relativity to the factor's "
                   "largest level and the years on cover behind it.", ""]

    if checks:
        report[4:4] = ["## Reproduction checks", "",
                       md(pd.DataFrame({"check": list(checks), "value": list(checks.values())}), "{:.1e}"), ""]
    (OUT / "report.md").write_text("\n".join(report), encoding="utf-8", newline="\n")
    print(f"written to {OUT}")
    for k, v in checks.items():
        print(f"   {k}: {v:.1e}")
    for k, v in week1_checks.items():
        print(f"   Week 1 {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
