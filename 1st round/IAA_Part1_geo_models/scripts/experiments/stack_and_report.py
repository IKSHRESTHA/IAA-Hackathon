"""Score every expert and every meta-learner configuration out of time, then build the
stacked 2025 submission.

Protocol (no tuning on the year being scored):
  * per-expert scores on the 2023 and 2024 back-test years (reported basis);
  * meta-learners validated across years: fit on 2023 -> score 2024, fit on 2024 -> score 2023;
  * the configuration with the best average cross-year pooled D^2 is refitted on both years
    and applied to the 2025 expert predictions.

Usage: python scripts/experiments/stack_and_report.py [--write-submission]
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from iaa_pricing import metrics as mt
from iaa_pricing.config import load_config
from iaa_pricing.guardrails import validate_submission
from iaa_pricing.io import read_table
from iaa_pricing.claims import prepare_claims
from iaa_pricing.stacking import (GatedPool, LogLinearPool, all_history, choose_credibility,
                                  expert_scores, row_completeness)

LINES = ["motor", "buildings", "contents"]
EXPERTS = ["total", "glm", "peril", "total_plus_large", "peril_plus_large", "fs_gamma", "fs_lognorm",
           "fs_peril", "catboost"]
SINGLE = EXPERTS + ["tariff"]   # scored individually
META_CONFIGS = {
    "meta_total_cred": ["total"],
    "meta_core": ["total", "glm", "peril"],
    "meta_core_tilt": ["total", "glm", "peril", "total_plus_large/total"],
    "meta_core_2p": ["total", "glm", "peril", "fs_gamma", "fs_lognorm"],
    "meta_fs": ["fs_gamma", "fs_lognorm", "fs_peril"],
    "meta_core_fs": ["total", "glm", "peril", "fs_gamma", "fs_lognorm", "fs_peril"],
    "meta_all": ["total", "glm", "peril", "fs_gamma", "fs_lognorm", "fs_peril", "catboost",
                 "total_plus_large/total"],
    "meta_all_tariff": ["total", "glm", "peril", "fs_gamma", "fs_lognorm", "fs_peril", "catboost",
                        "total_plus_large/total", "tariff"],
}
GATED = ("meta_core_fs",)      # expert sets also tried with cluster-gated weights
NONNEG = ("meta_core_2p", "meta_core_fs", "meta_fs")   # also tried with weights >= 0
_CLAIMS = None


def all_configs() -> dict[str, list[str]]:
    out = dict(META_CONFIGS)
    out.update({f"gated_{k[5:]}": v for k, v in META_CONFIGS.items() if k in GATED})
    out.update({f"nn_{k[5:]}": v for k, v in META_CONFIGS.items() if k in NONNEG})
    return out


def make_pool(name: str, ex: list[str], line: str, a: float):
    if name.startswith("gated_"):
        return GatedPool(ex, line, a=a)
    return LogLinearPool(ex, a=a, nonneg=name.startswith("nn_"))


def load(cfg, line, s):
    """Expert predictions for one line/set, merged with optional expert families and
    the all-history credibility inputs."""
    global _CLAIMS
    p = cfg.artifacts_dir / "experts" / f"{line}_{s}.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    for suffix in ("_2p", "_cb", "_fsp", "_fsg", "_moe", "_seg", "_xgb", "_hgb", "_rf", "_xgbt"):
        q = cfg.artifacts_dir / "experts" / f"{line}_{s}{suffix}.parquet"
        if q.exists():
            d = d.merge(pd.read_parquet(q), on="row_id", how="left", validate="one_to_one")
    if _CLAIMS is None:
        _CLAIMS = prepare_claims(read_table(cfg.data_dir, "claims"))
    keys = pd.read_parquet(cfg.artifacts_dir / f"features_{line}.parquet",
                           columns=["row_id", "policy_id", "period_start", "exposure", "is_test", "written_premium"])
    # Current tariff as a candidate "prior expert" (earned premium = annual premium x exposure).
    wp = keys.set_index("row_id")["written_premium"].astype(float)
    d["tariff"] = d["row_id"].map(wp).to_numpy() * d["exposure"].to_numpy()
    d["const"] = d["exposure"].to_numpy(dtype=float)          # constant-rate benchmark
    keys = keys.loc[~keys["is_test"].astype(bool)]
    # Products held: does the customer also hold the other lines in the same year?
    # (train rows for back-test years, 2025 test rows for the final set)
    others = [l for l in LINES if l != line]
    is_final = s == "final"
    combo = np.zeros(len(d), dtype=int)
    for i, o in enumerate(others):
        k_o = pd.read_parquet(cfg.artifacts_dir / f"features_{o}.parquet", columns=["customer_id", "year", "is_test"])
        k_o = k_o.loc[k_o["is_test"].astype(bool) == is_final]
        held = set(zip(k_o["customer_id"], k_o["year"].astype(int)))
        combo += (2 ** i) * np.array([(c, int(y)) in held for c, y in zip(d["customer_id"], d["year"])], dtype=int)
    d["products"] = combo.astype(str)
    h = all_history(d, _CLAIMS.loc[_CLAIMS["line"] == line], keys)
    d["hist_all_n"], d["hist_all_expo"] = h["hist_all_n"].to_numpy(), h["hist_all_expo"].to_numpy()
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-submission", action="store_true")
    a = ap.parse_args(argv)
    pd.set_option("display.width", 250)
    cfg = load_config("configs/week1.yaml")
    rep = cfg.reports_dir / "experts"
    rep.mkdir(parents=True, exist_ok=True)

    score_rows, pooled_parts, finals, coefs = [], {}, [], {}
    lines_done = []
    for line in LINES:
        h23, h24, fin = load(cfg, line, "ho2023"), load(cfg, line, "ho2024"), load(cfg, line, "final")
        if h23 is None or h24 is None:
            print(f"{line}: expert predictions not ready, skipped")
            continue
        lines_done.append(line)
        experts = [e for e in SINGLE if e in h23 and e in h24 and (fin is None or e in fin)
                   and h23[e].notna().all() and h24[e].notna().all()]
        # 1) single experts
        for H, d in ((2023, h23), (2024, h24)):
            sc = expert_scores(d, experts)
            sc["line"], sc["holdout_year"], sc["kind"] = line, H, "expert"
            score_rows.append(sc)
            c = row_completeness(d)
            for k in experts:
                pooled_parts.setdefault((H, k), []).append(
                    pd.DataFrame({"y": d["y"].to_numpy(), "mu": d[k].to_numpy() * c}))
        # 2) meta-learners, cross-year. The credibility constant is chosen once per training
        #    year on the champion expert and reused by every configuration (1-D, stable).
        a_fit = {H: choose_credibility(fit_d, ["total"])
                 for fit_d, H in ((h23, 2024), (h24, 2023))}
        for name, ex in all_configs().items():
            ex = [e for e in ex if all(x in experts for x in e.split("/"))]
            for fit_d, sc_d, H in ((h23, h24, 2024), (h24, h23, 2023)):
                acred = a_fit[H]
                m = make_pool(name, ex, line, acred).fit(fit_d)
                mu = m.predict(sc_d)
                y, e = sc_d["y"].to_numpy(), sc_d["exposure"].to_numpy()
                score_rows.append(pd.DataFrame([{
                    "model": name, "d2": mt.d2_tweedie(y, mu, 1.5), "gini": mt.normalized_gini(y, mu, e),
                    "ae": float(y.sum() / mu.sum()), "line": line, "holdout_year": H, "kind": "meta",
                    "cred_a": acred}]))
                pooled_parts.setdefault((H, name), []).append(pd.DataFrame({"y": y, "mu": mu}))
        # 3) final meta on both years for every configuration (choice made after pooling)
        both = pd.concat([h23, h24], ignore_index=True)
        if fin is not None:
            a_both = choose_credibility(both, ["total"])
            for name, ex in all_configs().items():
                ex = [e for e in ex if all(x in experts for x in e.split("/"))]
                acred = a_both
                gated = name.startswith("gated_")
                m = make_pool(name, ex, line, acred).fit(both)
                coefs[(line, name)] = {"a": acred, **({} if gated else m.coefficients())}
                f = fin[["row_id", "line", "exposure"]].copy()
                f["model"] = name
                f["prediction"] = m.predict(fin, reported=False)
                finals.append(f)
            for k in experts:
                f = fin[["row_id", "line", "exposure"]].copy()
                f["model"], f["prediction"] = k, fin[k].to_numpy()
                finals.append(f)

    scores = pd.concat(score_rows, ignore_index=True)
    pooled = []
    for (H, k), parts in pooled_parts.items():
        d = pd.concat(parts)
        if len(parts) == len(lines_done):
            pooled.append({"holdout_year": H, "model": k, "pooled_d2": mt.d2_tweedie(d.y, d.mu, 1.5),
                           "ae": float(d.y.sum() / d.mu.sum())})
    pooled = pd.DataFrame(pooled).pivot(index="model", columns="holdout_year", values="pooled_d2")
    pooled["mean"] = pooled.mean(axis=1)
    pooled = pooled.sort_values("mean", ascending=False)

    scores.to_csv(rep / "expert_and_meta_scores.csv", index=False)
    pooled.to_csv(rep / "pooled_d2.csv")
    (rep / "meta_coefficients.json").write_text(json.dumps({f"{l}|{n}": v for (l, n), v in coefs.items()}, indent=1))

    print("\n=== D2 by line, model and back-test year (reported basis; metas are cross-year) ===")
    tab = scores.pivot_table(index=["line", "model"], columns="holdout_year", values="d2")
    tab["mean"] = tab.mean(axis=1)
    print(tab.round(4).to_string())
    print("\n=== A/E by line, model and year ===")
    print(scores.pivot_table(index=["line", "model"], columns="holdout_year", values="ae").round(3).to_string())
    print("\n=== POOLED D2 (all lines, as the competition scores) ===")
    print(pooled.round(4).to_string())

    if finals and a.write_submission and len(lines_done) == 3:
        fin = pd.concat(finals, ignore_index=True)
        best = pooled.index[0]
        sub = fin[fin.model == best][["row_id", "prediction"]]
        expected = read_table(cfg.data_dir, "sample_submission", columns=["row_id"])["row_id"]
        chk = validate_submission(sub, expected)
        bad = [c for c in chk if not c.passed]
        if bad:
            raise RuntimeError(f"submission integrity failed: {bad}")
        sub = sub.set_index("row_id").loc[expected.to_numpy()].reset_index()
        path = cfg.output_dir / f"submission_week1_stack_{best}.csv"
        sub.to_csv(path, index=False)
        fin.to_parquet(cfg.artifacts_dir / "experts" / "final_all_models.parquet", index=False)
        print(f"\nSubmission written with {best}: {path} ({len(sub):,} rows)")
        print(fin.groupby(["line", "model"]).apply(lambda g: g.prediction.sum() / g.exposure.sum(),
                                                   include_groups=False).unstack().round(1).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
