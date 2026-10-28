"""Fit the two-part (frequency x severity) experts for each line and fit-set; saves
artifacts/<run>/experts/<line>_<set>_2p.parquet for the stacking meta-learner."""
from __future__ import annotations
import argparse, json, logging, sys, time
import numpy as np, pandas as pd
from iaa_pricing.claims import prepare_claims
from iaa_pricing.config import load_config
from iaa_pricing.experts import FitSet
from iaa_pricing.experts_twopart import run_twopart, run_twopart_glm, run_twopart_peril
from iaa_pricing.io import read_table
from iaa_pricing.modeling import load_line_data
from iaa_pricing.targets import DevelopmentInputs

TE = {"motor": ["make_model"]}

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", nargs="*", default=["motor", "buildings", "contents"])
    ap.add_argument("--sets", nargs="*", default=["ho2023", "ho2024", "final"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--peril", action="store_true", help="per-peril frequency x severity (fs_peril)")
    ap.add_argument("--glm", action="store_true", help="GLM frequency x severity (fs_glm)")
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
                T = cfg.prediction_year
                data = load_line_data(line, cfg, devin, target_year=T, cap_quantile=0.995, target_encode=TE.get(line))
                fs = FitSet("final", data.train, data.test, True)
            else:
                T = int(s[2:])
                data = load_line_data(line, cfg, devin, target_year=T, cap_quantile=0.995, max_train_year=T,
                                      fit_years=[y for y in cfg.train_years if y < T],
                                      target_encode=TE.get(line), encode_fit_max_year=T - 1)
                fs = FitSet(s, data.train[data.train.year < T], data.train[data.train.year == T], False)
            if a.smoke:
                fs = FitSet(fs.name, fs.train.sample(80000, random_state=1), fs.score.sample(20000, random_state=2), fs.is_final)
            if a.glm:
                pred, info = run_twopart_glm(data, fs, cfg, devin, T)
                tag = f"{line}_{s}_fsg" + ("_smoke" if a.smoke else "")
            elif a.peril:
                pred, info = run_twopart_peril(data, fs, cfg, devin, T)
                tag = f"{line}_{s}_fsp" + ("_smoke" if a.smoke else "")
            else:
                pred, info = run_twopart(data, fs, cfg, devin, T)
                tag = f"{line}_{s}_2p" + ("_smoke" if a.smoke else "")
            pred.to_parquet(out_dir / f"{tag}.parquet", index=False)
            (out_dir / f"{tag}.json").write_text(json.dumps(info, indent=1, default=str))
            print(f"### {tag} done in {time.time()-t0:.0f}s: {json.dumps(info, default=str)}", flush=True)
            del data, fs
    return 0

if __name__ == "__main__":
    sys.exit(main())
