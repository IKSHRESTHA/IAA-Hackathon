# IAA AI-Assisted Hackathon 2026, Part 1: code of the two best private-score models

| Submitted file | Model | Public | Private |
|---|---|---:|---:|
| `submission_week1_v4_geo6.csv` | **v4**: geometric mean of 6 experts, recalibrated per line | 0.12851 | **0.13830** |
| `pick3_ens_geo7_moe.csv` | **pick3**: the same 6 experts + a claim-size mixture-of-experts (7) | 0.12883 | **0.13830** |

Out-of-time check before submitting (fit the calibration on one back-test year, score the other,
Tweedie D² with power 1.5, all lines pooled): v4 0.1380 (2023) / 0.1367 (2024); pick3 0.1380 / 0.1368.

This is the code exactly as it ran on 22 October 2026, when both files were made, and it reproduces
them: from the cached expert predictions in a minute (relative difference 2e-15), or from the raw
competition files through every step.

## Quick start: rebuild both submissions in a minute (no competition data needed)

```bash
pip install -r requirements.txt          # Python 3.11+ (built with 3.13)
python rebuild_from_cache.py
```

It reads `cache/experts7_<line>.parquet` (the seven experts' predictions as they were on 22 October),
writes `outputs/submission_week1_v4_geo6.csv` and `outputs/pick3_ens_geo7_moe.csv`, and compares
them with the submitted files in `submissions/` (largest relative difference 2e-15).

## Full pipeline from the competition files (about 2.5 hours on 2 CPU cores, 7-8 GB RAM)

Put the 11 competition files in `data/week1/` (names in `data/week1/README.md`), then:

```bash
python run_pipeline.py                   # every step; resume with --from <step>, or --only <step>
```

| Step | Script | Output | Time |
|---|---|---|---:|
| `features` | `scripts/build_features.py` | `artifacts/week1/features_<line>.parquet` | 3 min |
| `experts` | `scripts/experiments/run_experts.py` | experts `total`, `glm`, `peril` | 72 min |
| `twopart` | `scripts/experiments/run_twopart.py` | experts `fs_gamma`, `fs_lognorm` | 20 min |
| `peril` | `scripts/experiments/run_twopart.py --peril` | expert `fs_peril` | 55 min |
| `moe` | `scripts/experiments/run_moe.py` | expert `fs_moe` (uses the `twopart` frequency) | 3 min |
| `ensembles` | `scripts/experiments/build_robust_submission.py --method geo` | `outputs/submission_week1_v4_geo6.csv`, `outputs/submission_week1_geo7_moe.csv` | 2 min |

The `ensembles` step also runs `verify_final.py` on both files (integrity, 2025 level per line, and
the comparison with the submitted files). The steps run from this folder with `src/` on the Python
path; to run one script by hand: `PYTHONPATH=src python scripts/experiments/run_experts.py --lines
contents --sets ho2024` (Windows PowerShell: `$env:PYTHONPATH="src"` first). Every expert script
has `--smoke` for a quick test on a sample.

## How the models work

**1. Cleaning and features** (`build_features.py`, `cleaning.py`, `features.py`). Exposure recorded in
days or on a 360-day basis is recomputed from the dates, sums insured recorded in units are rescaled,
customer ids that lost their leading zeros are restored, category spellings are unified. Features:
the rating factors, the address and its weather, and claims history as known on 1 January of each
year (no look-ahead). `customer_service_contacts` and `renewal_decision` leak the outcome and are
excluded.

**2. Claims to ultimate** (`claims.py`, `development.py`, `targets.py`). Claims are valued at 31
December 2024, so the recent years are incomplete: a reporting pattern per peril, corrected for
truncation, gives each row the share of its claims already reported (it enters the model offsets);
open claims are developed; one case reserve typed with an extra digit (`CL30973963`) is divided by
10; costs are trended to 2025; claims are capped at the 99.5th percentile and the excess is loaded
back.

**3. Seven experts**, each trained three times per line: on 2021-22 to predict 2023 (`ho2023`), on
2021-23 to predict 2024 (`ho2024`) and on 2021-24 to predict 2025 (`final`). All are LightGBM with an
exact log offset (exposure x reported share x trend), except the GLM.

| Expert | What it is | Code |
|---|---|---|
| `total` | Tweedie GBM on capped total cost x large-loss load | `experts.py` |
| `glm` | Tweedie GLM with banded numeric factors | `experts.py`, `models/glm.py` |
| `peril` | one Tweedie GBM per peril (own reporting pattern and large-loss load), summed | `experts.py` |
| `fs_gamma` | Poisson frequency GBM x Gamma severity GBM | `experts_twopart.py` |
| `fs_lognorm` | the same frequency x lognormal severity with a risk-specific variance (heavier tails, higher mean) | `experts_twopart.py` |
| `fs_peril` | frequency x severity for each peril, summed | `experts_twopart.py` |
| `fs_moe` (pick3 only) | frequency x claim-size mixture of experts: claims clustered by size, a gating GBM gives each policy's chance of a small / medium / large claim, one Gamma GBM per size cluster | `experts_moe.py` |

**4. The ensemble** (`build_robust_submission.py`, `stacking.py: LogLinearPool`). For each row the
members' predictions are combined by an equal-weight geometric mean, `geo = exp(mean(log
prediction))`. Then, per line, a Tweedie (p = 1.5) GLM recalibrates level and slope:

```
log(expected cost) = log(exposure x L) + b0 + b1 x log(geo rate / L)      with b1 >= 0
```

where `L` is the average rate of `geo` in the scored year. It is fitted on the 2023 and 2024
back-test predictions against the reported claims (the reported share in the offset), then applied
to the 2025 predictions at the ultimate basis. Fitted values:

| | v4 b0 | v4 b1 | pick3 b0 | pick3 b1 |
|---|---:|---:|---:|---:|
| motor | -0.001 | 1.042 | 0.000 | 1.040 |
| buildings | 0.014 | 1.096 | 0.008 | 1.104 |
| contents | 0.098 | 1.089 | 0.097 | 1.089 |

## Files

```text
rebuild_from_cache.py       both submissions from the cache (1 minute)
run_pipeline.py             every step from the competition files (about 2.5 hours)
configs/week1.yaml          every assumption: paths, dates, reserving, trend, caps, GBM settings
src/iaa_pricing/            the library: cleaning, features, reserving, targets, experts, ensemble
scripts/build_features.py
scripts/experiments/        run_experts, run_twopart, run_moe, build_robust_submission, verify_final,
                            stack_and_report (loads the experts' predictions),
                            export_all_models (the script that wrote pick3, with 26 other models)
cache/experts7_<line>.parquet   predictions of the 7 experts for 2023, 2024 (back-tests) and 2025
submissions/                the two files as made on 22 October (full precision)
data/week1/                 put the competition files here
outputs/                    rebuilt submissions are written here
```

## Notes

* **Exact code.** Everything is the 22 October snapshot (git commit `33afbd4`), unchanged except one
  line so `build_features.py` also runs on Windows (it imported a Unix-only module to log memory).
  The features step was re-run from this folder and gives the same feature files.
* **pick3** was written by `export_all_models.py`, which builds 28 models in one go. The
  `ensembles` step builds it with `build_robust_submission.py` and `fs_moe` added to the members:
  same predictions (relative difference 2e-15), one model only.
* **A defect found a day later, kept here on purpose.** The contents frequency trend (+1.4% a year)
  enters the experts' offsets with the wrong sign (`* freq_trend` instead of `/ freq_trend` in
  `experts.py`, `experts_twopart.py` and `targets.py`). Motor and buildings have no frequency trend
  and are unaffected; for contents the line recalibration absorbs most of it (its intercept, 0.098,
  is about +10%), but the contents relativities may be slightly distorted. Fixing it changes the contents predictions,
  so the code keeps it to reproduce the submitted files.
* **Retraining on another machine** gives very close, not necessarily bit-identical, expert
  predictions (LightGBM results can depend on the platform). The cache rebuild is exact.
* The copies uploaded to Kaggle may have had their predictions rounded to fewer decimals; the files
  in `submissions/` are the full-precision originals.
