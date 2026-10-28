"""Score every model out of time and export one calibrated 2025 submission per model.

Model kinds (all fitted with the competition loss, Tweedie p=1.5, on the back-test years):
  single  one expert, recalibrated (level + slope) one year ahead
  geo     equal-weight geometric mean of experts, recalibrated
  pool    learned expert weights (log-linear pool), optional credibility / exposure terms
  gated   pool whose weights vary by risk cluster
Cross-year protocol: fit on 2023 -> score 2024 and fit on 2024 -> score 2023; the 2025 file
uses the fit on both years. Outputs reports/<run>/experts/all_models_table.csv and
outputs/all_models/submission_<model>.csv.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/experiments")
import stack_and_report as sr  # noqa: E402

from iaa_pricing import metrics as mt  # noqa: E402
from iaa_pricing.config import load_config  # noqa: E402
from iaa_pricing.guardrails import validate_submission  # noqa: E402
from iaa_pricing.io import read_table  # noqa: E402
from iaa_pricing.stacking import GatedPool, LogLinearPool, choose_credibility  # noqa: E402

CORE6 = ["total", "fs_gamma", "fs_lognorm", "peril", "fs_peril", "glm"]
ALT = ["fs_xgb", "fs_hgb", "fs_rf", "total_xgb"]
MODELS: dict[str, tuple] = {
    # --- benchmarks
    "bench_constant_rate": ("single", "const"),
    "bench_current_tariff": ("single", "tariff"),
    # --- single experts
    "glm_tweedie": ("single", "glm"),
    "glm_freq_x_sev": ("single", "fs_glm"),
    "gbm_total_lightgbm": ("single", "total"),
    "gbm_total_plus_large_loss": ("single", "total_plus_large"),
    "gbm_total_xgboost": ("single", "total_xgb"),
    "gbm_per_peril_tweedie": ("single", "peril"),
    "gbm_per_peril_plus_large": ("single", "peril_plus_large"),
    "policy_segments_kmeans": ("single", "seg_moe"),
    "fs_lightgbm_gamma": ("single", "fs_gamma"),
    "fs_lightgbm_lognormal_tail": ("single", "fs_lognorm"),
    "fs_lightgbm_per_peril": ("single", "fs_peril"),
    "fs_claim_size_moe": ("single", "fs_moe"),
    "fs_xgboost": ("single", "fs_xgb"),
    "fs_histgradboost": ("single", "fs_hgb"),
    "fs_random_forest": ("single", "fs_rf"),
    # --- equal-weight ensembles
    "ens_v3_geo4": ("geo", ["total", "fs_gamma", "fs_lognorm", "peril"]),
    "ens_v4_geo6": ("geo", CORE6),
    "ens_geo7_moe": ("geo", CORE6 + ["fs_moe"]),
    "ens_geo_fs_family": ("geo", ["fs_gamma", "fs_lognorm", "fs_peril", "fs_moe", "fs_xgb", "fs_hgb"]),
    "ens_geo6_plus_alt": ("geo", CORE6 + ["fs_xgb", "fs_hgb", "total_xgb"]),
    "ens_geo_all": ("geo", CORE6 + ["fs_glm", "fs_moe", "seg_moe", "fs_xgb", "fs_hgb", "fs_rf", "total_xgb"]),
    # --- learned-weight meta-learners
    "meta_nn_weights": ("pool", CORE6, False, False, True),
    "meta_nn_weights_credibility": ("pool", CORE6, True, False, True),
    "meta_nn_weights_exposure": ("pool", CORE6, False, True, True),
    "meta_free_weights_cred_expo_v2style": ("pool", CORE6 + ["total_plus_large/total"], True, True, False),
    "meta_gated_by_cluster": ("gated", CORE6),
}


def available(d: pd.DataFrame, cols: list[str]) -> bool:
    return all(all(x in d for x in c.split("/")) for c in cols)


def make(kind_spec, fit_d, line):
    kind = kind_spec[0]
    if kind == "single":
        return LogLinearPool([kind_spec[1]], a=None, exposure_term=False, nonneg=True)
    if kind == "geo":
        return LogLinearPool(["_geo"], a=None, exposure_term=False, nonneg=True)
    if kind == "pool":
        _, ex, cred, expo, nn = kind_spec
        a = choose_credibility(fit_d, ["total"]) if cred else None
        return LogLinearPool(ex, a=a, exposure_term=expo, nonneg=nn)
    if kind == "gated":
        return GatedPool(kind_spec[1], line, a=None)
    raise ValueError(kind)


def main() -> int:
    cfg = load_config("configs/week1.yaml")
    out_dir = cfg.output_dir / "all_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = read_table(cfg.data_dir, "sample_submission", columns=["row_id"])["row_id"]
    parts, per_line, finals = {}, [], {}
    for line in sr.LINES:
        sets = {s: sr.load(cfg, line, s) for s in ("ho2023", "ho2024", "final")}
        for name, spec in MODELS.items():
            cols = [spec[1]] if spec[0] == "single" else list(spec[1])
            if not all(available(d, cols) for d in sets.values()):
                continue
            if spec[0] == "geo":
                for d in sets.values():
                    d["_geo"] = np.exp(np.mean([np.log(np.maximum(d[m].to_numpy(dtype=float), 1e-9))
                                                for m in cols], axis=0))
            h23, h24, fin = sets["ho2023"], sets["ho2024"], sets["final"]
            for fit_d, sc_d, H in ((h23, h24, 2024), (h24, h23, 2023)):
                mu = make(spec, fit_d, line).fit(fit_d).predict(sc_d)
                y = sc_d["y"].to_numpy()
                per_line.append({"model": name, "line": line, "holdout_year": H,
                                 "d2": mt.d2_tweedie(y, mu, 1.5), "ae": y.sum() / mu.sum()})
                parts.setdefault((name, H), []).append(pd.DataFrame({"y": y, "mu": mu}))
            both = pd.concat([h23, h24], ignore_index=True)
            m = make(spec, both, line).fit(both)
            finals.setdefault(name, []).append(pd.DataFrame({"row_id": fin["row_id"].to_numpy(),
                                                             "prediction": m.predict(fin, reported=False)}))
    pl = pd.DataFrame(per_line)
    rows = []
    for name in MODELS:
        if len(parts.get((name, 2023), [])) != len(sr.LINES):
            continue
        r = {"model": name, "kind": MODELS[name][0]}
        for H in (2023, 2024):
            a = pd.concat(parts[(name, H)])
            r[f"pooled_{H}"] = mt.d2_tweedie(a.y, a.mu, 1.5)
        r["pooled_mean"] = (r["pooled_2023"] + r["pooled_2024"]) / 2
        for line in sr.LINES:
            r[f"{line}_mean"] = pl[(pl.model == name) & (pl.line == line)]["d2"].mean()
        sub = pd.concat(finals[name], ignore_index=True)
        ok = all(c.passed for c in validate_submission(sub, expected)) and bool((sub["prediction"] > 0).all())
        if ok:
            sub = sub.set_index("row_id").loc[expected.to_numpy()].reset_index()
            sub.to_csv(out_dir / f"submission_{name}.csv", index=False)
        r["file_ok"] = ok
        rows.append(r)
    tab = pd.DataFrame(rows).sort_values("pooled_mean", ascending=False)
    tab.to_csv(cfg.reports_dir / "experts" / "all_models_table.csv", index=False)
    pd.set_option("display.width", 250)
    print(tab.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
