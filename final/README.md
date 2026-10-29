# IAA AI-Assisted Hackathon 2026: final submission

Team Actuarial Noob: Krishna Kumar Shrestha (solo). Deadline 30 October 2026, 11:59:59 PM EDT.

This folder holds the two prediction files I submitted, the write-up, and `pipeline.py`, which runs
the pipeline that produced both files. The repository includes the competition data and every
intermediate result, so a fresh clone can rebuild the files without downloading anything else.

| Form item | Where |
|---|---|
| Week 1 predictions | `1_predictions/week1_part1_predictions.csv` (public 0.12851, private 0.13830) |
| Week 2 predictions | `1_predictions/week2_part2_predictions.csv` (v13, public 0.17172) |
| Write-up | `2_writeup/writeup.pdf`, pages 1-5 (page 6 is an appendix) |
| Code | this repository, https://github.com/IKSHRESTHA/IAA-Hackathon |
| Video | recorded separately |

`1_predictions/manifest.csv` gives each file's source in the repository, its scores and its SHA-256
checksum.

`governance_report.py`, `leakage_audit.py`, `baseline_manifest.py` and `validation_report.py` write
the governance reports (leakage, feature governance, tail sensitivity, validation, uncertainty
and population drift) under `3_diagnostics/`.

## Reproduce the two files

Set up once (Python 3.13 or 3.14; the clone is about 850 MB because it holds the data):

```bash
git clone https://github.com/IKSHRESTHA/IAA-Hackathon.git
cd IAA-Hackathon
python -m pip install -r requirements.txt
```

Quarto is only needed to render the write-up PDF. Add `--no-render` to any run below to skip it.
Everything here was tested on Windows 11 with Python 3.14.2, LightGBM 4.7.0, pandas 3.0.5, numpy
2.5.3 and Quarto 1.8.27.

### Quick run (about a minute)

This starts from the intermediate results stored in the repository
(the experts' predictions and the developed-market priors) and runs the last stages of both
pipelines:

```bash
python final/pipeline.py
```

It ends by comparing what it built with the files that were submitted:

```
week1: rebuilt file matches the submitted file (same rows True, largest relative difference 2.7e-15 against a tolerance of 1e-05, identical bytes False)
week2: rebuilt file matches the submitted file (same rows True, largest relative difference 0.0e+00 against a tolerance of 1e-05, identical bytes True)
```

Week 2 comes out byte for byte, and step 11 also prints `identical to v13 ... no new version`.
Week 1 differs only around the 15th significant digit, which is floating-point rounding.
The rebuilt files go to `final/reproduced/`, with the comparison in `check.csv`, and
`reproduced/steps/` keeps the exact bytes each round step produced. The first run on a
new machine can take longer while Python loads and the system scans the compiled libraries.

### Full run (from the raw competition files)

```bash
python final/pipeline.py --full --week 2     # Week 2 only (about 80 minutes on 8 cores)
python final/pipeline.py --full --week 1     # Week 1 only (about 2.5 hours on 2 cores)
python final/pipeline.py --full              # both
```

This reruns every stage in the tables below and replaces the intermediate results in the
repository with fresh ones, so do it on a clean clone. A step whose output differs from a file
already registered saves it as a new numbered version rather than overwriting anything. The check
at the end passes when the largest relative difference from the submitted file is below 1e-5.

I ran the Week 2 full run on a fresh clone. Steps 01 and 02 rebuilt their outputs byte for byte.
The feature tests in step 03 moved in the fifth decimal of their standard errors, and none of the
keep-or-drop decisions changed. The transfer LightGBM (v2) came out identical, the ensemble and v5
matched to within 3e-15, and the final Week 2 file to within 9e-7, so steps 8, 9 and 11 saved new
versions (v16, v17, v18) next to v4, v5 and v13. The Week 1 full run, also on a fresh clone, took
about 8 hours here alongside other jobs, and its file matched the submitted one to within 3.4e-12.

## The pipeline

Week 1, developed market. The code that made the submitted file is in
`1st round/IAA_Part1_geo_models/`, and `run_pipeline.py` there runs these stages:

| Stage | Script | Output | Run |
|---|---|---|---|
| features | `scripts/build_features.py` | cleaned data, features, claims history known at 1 January | full |
| experts | `scripts/experiments/run_experts.py`, `run_twopart.py`, `run_moe.py` | each expert's predictions for 2023 and 2024 (back-tests) and 2025 | full |
| ensemble | `scripts/experiments/build_robust_submission.py` | geometric mean of six experts, recalibrated per line: the Week 1 file | full and quick |

The quick run does the ensemble stage on `cache/experts7_<line>.parquet`, the experts' predictions
as they were on 22 October, through `rebuild_from_cache.py`.

Week 2, emerging market. The scripts sit in `1st round/` and `2nd round/`, and `run_all.py` at the
root runs a round's steps in order:

| Stage | Script | Output | Run |
|---|---|---|---|
| developed-market data | `1st round/01_data_reading.py` to `03_feature_engineering.py` | the developed market's features, which the prior is fitted on | full |
| emerging-market data | `2nd round/01_data_reading.py` to `03_feature_engineering.py` | cleaned data and features | full |
| developed prior | `04_developed_prior.py`, `05_lightgbm_model.py` | prior log rates; the transfer LightGBM (v2) | full |
| ensemble on priors | `06_geo_priors.py`, `07_geo_experts.py`, `08_geo_ensemble.py` | experts' predictions and their ensemble (v4) | full |
| rebase | `09_rebase_to_best.py` | v5: the best public file's segment totals, the ensemble's ranking inside | full and quick |
| segment levels | `11_joint_levels.py make --target public --scores-before v12` | v13, the Week 2 file | full and quick |

Two inputs come from outside the code, and the repository keeps them as files. The public
leaderboard scores are recorded in `2nd round/submissions/versions.csv` and
`2nd round/data/reference/leaderboard_evidence.csv`; step 11 reads the segment levels from them,
and `--scores-before v12` limits it to the scores known when v13 was made. The best public file
before this work (0.16529) and its back-test predictions came from my earlier repository
(`iaa-pricing-part2`) and sit in `2nd round/data/reference/`; v5 takes its segment totals from
that file.

## Why these two files

For Week 1 I used `submission_week1_v4_geo6.csv`, the geometric mean of six experts recalibrated
per line. Its private score of 0.13830 ties with `pick3_ens_geo7_moe.csv`, which adds a seventh
expert, so I kept the simpler one.

For Week 2 I used v13 because it has the best public score. I also selected v12 (public 0.17095) on
Kaggle for the final ranking. It is the best expected file on the rows not yet scored, and v13 only
beats it if the true levels are as extreme as the public rows suggest. `11_joint_levels.py make
--scores-before v12` (without `--target public`) rebuilds v12 the same way.

## The approach in short

Week 1 is a developed market with four years of data (2021-24, 163,150 claims). I fixed the data
errors with a logged reason for each, built the target as the ultimate claim cost (share reported
in the offset, costs trended to 2025, large losses capped and loaded back), and priced each line
with six experts: a total-cost GBM, a GLM, per-peril GBMs and three frequency × claim-size models.
Their geometric mean, recalibrated per line, beat every single model in both back-test years.

Week 2 is an emerging market with two years (2023-24, 9,258 claims), and 77% of the 2025 book is
new business. I fit a model on the developed market using the factors both markets share and put it
in the offset, so the local models only learn how this market differs. The back-test could rank
risks but could not set segment levels, so I read the levels from the public scores with an exact
Tweedie formula and modelled the luck of the public sample separately.

The write-up has the numbers, decisions, use of AI, assumptions, limitations and guardrails.

## Where things are

```
IAA-Hackathon/
  final/          this folder
  1st round/      developed market: steps 01-06, and IAA_Part1_geo_models/ (the Week 1 code)
  2nd round/      emerging market: steps 01-12
  run_all.py      runs a round's steps in order
  pricing_lgbm.py, geo_experts.py, submissions.py    code shared by the rounds
  README.md, KNOWLEDGE.md                             full technical notes and lessons learned
```

The 2nd round keeps every submission it made in `submissions/`, with a register (`versions.csv`:
script, date, evidence, public score and checksum of each version). Each round keeps the evidence
behind each step in `reports/`.

```
final/
  pipeline.py
  _quarto.yml, _variables.yml, src/final_figures.py   the write-up's settings and figures
  1_predictions/    the two submitted files and manifest.csv
  2_writeup/        writeup.qmd and writeup.pdf
  reproduced/       written by pipeline.py, not kept in git
```
