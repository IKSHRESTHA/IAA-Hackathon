# Week 1: the code that made the submitted file

This folder is the code exactly as it ran on 22 October 2026, when the Week 1 file was made (git
commit `33afbd4` of the original repository). It is kept unchanged, apart from one line that lets
`build_features.py` run on Windows, so that it still reproduces the submitted file.

| Submitted file | Model | Public | Private |
|---|---|---:|---:|
| `submission_week1_v4_geo6.csv` | geometric mean of 6 experts, recalibrated per line | 0.12851 | 0.13830 |
| `pick3_ens_geo7_moe.csv` | the same 6 plus a claim-size mixture of experts | 0.12883 | 0.13830 |

The final submission uses the first file. Before submitting, the out-of-time check (calibration
fitted on one back-test year, scored on the other, all lines pooled) gave 0.1380 for 2023 and 0.1367
for 2024 (pick3: 0.1380 and 0.1368).

## Rebuild both files in a minute

```bash
python -m pip install -r requirements.txt     # or final/requirements.txt at the repository root
python rebuild_from_cache.py
```

This reads `cache/experts7_<line>.parquet`, the seven experts' predictions as they were on 22
October, writes both files to `outputs/` and compares them with `submissions/`. The largest relative
difference is about 3e-15. `final/pipeline.py` at the repository root runs the same stage.

## Full pipeline from the competition files

About 2.5 hours on 2 CPU cores and 7-8 GB of memory. The 11 competition files go in `data/week1/`
(`python final/pipeline.py --full --week 1` copies them from `1st round/data/raw/`), then:

```bash
python run_pipeline.py          # every step; resume with --from <step>, or run one with --only <step>
```

| Step | Script | Output | Time |
|---|---|---|---:|
| `features` | `scripts/build_features.py` | `artifacts/week1/features_<line>.parquet` | 3 min |
| `experts` | `scripts/experiments/run_experts.py` | experts `total`, `glm`, `peril` | 72 min |
| `twopart` | `scripts/experiments/run_twopart.py` | experts `fs_gamma`, `fs_lognorm` | 20 min |
| `peril` | `scripts/experiments/run_twopart.py --peril` | expert `fs_peril` | 55 min |
| `moe` | `scripts/experiments/run_moe.py` | expert `fs_moe` | 3 min |
| `ensembles` | `scripts/experiments/build_robust_submission.py --method geo` | both submissions, then `verify_final.py` | 2 min |

The steps run from this folder with `src/` on the Python path. To run one script by hand:
`PYTHONPATH=src python scripts/experiments/run_experts.py --lines contents --sets ho2024` (in
PowerShell, set `$env:PYTHONPATH="src"` first). Every expert script has `--smoke` for a quick test on
a sample. Retraining on another machine gives very close but not always bit-identical predictions,
because LightGBM results can depend on the platform; the cache rebuild is exact.

## How it works

1. Cleaning and features (`build_features.py`, `cleaning.py`, `features.py`): exposure recomputed
   from the dates, sums insured rescaled, customer ids padded, spellings unified. Features are the
   rating factors, the address and its weather, and claims history known on 1 January of each year.
   `customer_service_contacts` and `renewal_decision` leak the outcome and are left out.
2. Claims to ultimate (`claims.py`, `development.py`, `targets.py`): a reporting pattern per peril,
   corrected for truncation, gives each row the share of its claims already reported, which enters
   the offsets. Open claims are developed, the mistyped reserve on `CL30973963` is divided by 10,
   costs are trended to 2025, and claims above the 99.5th percentile are capped with the excess
   loaded back.
3. Seven experts, each trained on 2021-22 to price 2023, on 2021-23 to price 2024, and on 2021-24 to
   price 2025. All are LightGBM with an exact log offset except the GLM.

   | Expert | What it is | Code |
   |---|---|---|
   | `total` | Tweedie GBM on capped total cost × large-loss load | `experts.py` |
   | `glm` | Tweedie GLM with banded numeric factors | `experts.py`, `models/glm.py` |
   | `peril` | one Tweedie GBM per peril, summed | `experts.py` |
   | `fs_gamma` | Poisson frequency GBM × Gamma claim-size GBM | `experts_twopart.py` |
   | `fs_lognorm` | the same frequency × lognormal claim size with a risk-specific variance | `experts_twopart.py` |
   | `fs_peril` | frequency × claim size for each peril, summed | `experts_twopart.py` |
   | `fs_moe` (pick3 only) | frequency × a mixture of claim-size models, one per size cluster | `experts_moe.py` |

4. The ensemble (`build_robust_submission.py`, `stacking.py`): an equal-weight geometric mean of the
   members' predictions, then per line a Tweedie GLM that recalibrates level and slope,
   `log E = log(exposure × L) + b0 + b1 × log(geo rate / L)` with `L` the scored year's average rate,
   fitted on the 2023 and 2024 back-tests.

   | | v4 b0 | v4 b1 | pick3 b0 | pick3 b1 |
   |---|---:|---:|---:|---:|
   | motor | -0.001 | 1.042 | 0.000 | 1.040 |
   | buildings | 0.014 | 1.096 | 0.008 | 1.104 |
   | contents | 0.098 | 1.089 | 0.097 | 1.089 |

## Files

```
rebuild_from_cache.py          both submissions from the cache (a minute)
run_pipeline.py                every step from the competition files (about 2.5 hours)
configs/week1.yaml             every assumption: paths, dates, reserving, trend, caps, GBM settings
src/iaa_pricing/               cleaning, features, reserving, targets, experts, ensemble, guardrails
scripts/build_features.py
scripts/experiments/           run_experts, run_twopart, run_moe, build_robust_submission,
                               verify_final, stack_and_report, export_all_models
cache/experts7_<line>.parquet  the 7 experts' predictions for 2023, 2024 and 2025 as of 22 October
submissions/                   the two files as made, at full precision
data/week1/                    the competition files, for the full pipeline
outputs/                       rebuilt submissions
```

## Known issue, kept on purpose

The contents frequency trend (+1.4% a year) enters the experts' offsets with the wrong sign
(`* freq_trend` instead of `/ freq_trend` in `experts.py`, `experts_twopart.py` and `targets.py`).
Motor and buildings have no frequency trend and are not affected. For contents, the recalibration
absorbs most of it (its intercept, 0.098, is about +10%), but the contents relativities may be
slightly off. Fixing it would change the predictions, so this code keeps it to reproduce
the submitted file; the rebuild in `1st round/05_geo_experts.py` has it fixed.

`pick3` was first written by `export_all_models.py`, which builds 28 models in one go; the
`ensembles` step builds the same file with `build_robust_submission.py` (relative difference 2e-15).
The copies uploaded to Kaggle may have had their predictions rounded; the files in `submissions/`
are the full-precision originals.
