"""Fit all experts for each line on three fit-sets (2023 and 2024 back-tests, final 2025)
and save their predictions for the stacking meta-learner.

Usage: python scripts/experiments/run_experts.py --lines motor buildings contents [--sets ho2023 ho2024 final]
       [--smoke]   (tiny subsample, to test the code path)
"""
from __future__ import annotations
import argparse, json, logging, sys, time
import numpy as np, pandas as pd
from iaa_pricing.claims import prepare_claims
from iaa_pricing.config import load_config
from iaa_pricing.experts import FitSet, run_experts
from iaa_pricing.io import read_table
from iaa_pricing.modeling import load_line_data
from iaa_pricing.targets import DevelopmentInputs

TE = {"motor": ["make_model"]}

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", nargs="*", default=["motor", "buildings", "contents"])
    ap.add_argument("--sets", nargs="*", default=["ho2023", "ho2024", "final"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for n in ("iaa_pricing.io", "iaa_pricing.claims", "iaa_pricing.development", "iaa_pricing.cleaning"):
        logging.getLogger(n).setLevel(logging.ERROR)
    cfg = load_config("configs/week1.yaml")
    out_dir = cfg.artifacts_dir / "experts"; out_dir.mkdir(parents=True, exist_ok=True)
    devin = DevelopmentInputs.fit(prepare_claims(read_table(cfg.data_dir, "claims")), cfg)
    for line in a.lines:
        for s in a.sets:
            t0 = time.time()
            if s == "final":
                data = load_line_data(line, cfg, devin, target_year=cfg.prediction_year, cap_quantile=0.995,
                                      target_encode=TE.get(line))
                fs = FitSet("final", data.train, data.test, True)
            else:
                H = int(s[2:])
                data = load_line_data(line, cfg, devin, target_year=H, cap_quantile=0.995, max_train_year=H,
                                      fit_years=[y for y in cfg.train_years if y < H],
                                      target_encode=TE.get(line), encode_fit_max_year=H - 1)
                fs = FitSet(s, data.train[data.train.year < H], data.train[data.train.year == H], False)
            if a.smoke:
                rng = np.random.default_rng(0)
                fs = FitSet(fs.name, fs.train.sample(60000, random_state=1), fs.score.sample(20000, random_state=2), fs.is_final)
            res = run_experts(data, fs, cfg, total_trees=None)
            tag = f"{line}_{s}" + ("_smoke" if a.smoke else "")
            res.predictions.to_parquet(out_dir / f"{tag}.parquet", index=False)
            (out_dir / f"{tag}.json").write_text(json.dumps(res.info, indent=1, default=str))
            print(f"### {tag} done in {time.time()-t0:.0f}s: " + json.dumps({k: v for k, v in res.info.items() if k in ('total', 'large', 'freq')}, default=str), flush=True)
            del data, fs, res
    return 0

if __name__ == "__main__":
    sys.exit(main())
