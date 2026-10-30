# IAA AI-Assisted Hackathon 2026

My pricing work for both parts of the hackathon. Week 1 is a developed market with four years of
data; Week 2 is an emerging market with two. In both, the task is to predict the 2025 claim cost of
every motor, buildings and contents policy, scored by the Tweedie deviance D² with power 1.5.

Start with [`final/`](final/README.md): the two submitted prediction files, the write-up, the
presentation, the model diagnostics, and `final/pipeline.py`, which reruns the pipeline that
produced both files.

| | Week 1 (developed market) | Week 2 (emerging market) |
|---|---|---|
| Training data | 2021-24, 163,150 claims | 2023-24, 9,258 claims |
| Rows to price | 626,639 | 209,063 |
| Submitted file | geometric mean of six experts, recalibrated per line | v13: ensemble ranking, segment levels for the 2025 mix |
| Public / private D² | 0.12851 / 0.13830 | 0.17172 / pending |

## What is where

```
final/              submission files, write-up, presentation, pipeline.py, diagnostics
1st round/          developed market
  01-03               read, clean and build features (scripts and executed notebooks)
  04-06               a LightGBM baseline, and a rebuild of the Week 1 ensemble
  IAA_Part1_geo_models/   the exact code that made the submitted Week 1 file
2nd round/          emerging market
  01-03               read, clean and build features
  04-08               developed-market prior, transfer LightGBM, ensemble on priors
  09-12               segment levels and the final files
run_all.py          runs a round's steps in order
pricing_lgbm.py     LightGBM tools the model steps share (offset, trend, large losses, D²)
geo_experts.py      the six experts and the per-line recalibration
submissions.py      numbered submission files and their register
KNOWLEDGE.md        what I learned along the way, kept as the project's memory
```

Each round has `data/raw/` (the competition files), `data/clean/`, `data/features/` (one training
and one 2025 table per line), `reports/` (the evidence behind each step) and its settings in
`config.py`. The 2nd round also keeps `data/prior/`, `data/experts/` and `submissions/`, because the
final file is built from them. Fitted models and the 1st round's rebuild files are written on each
run and not kept in git.

## Run it

```bash
python -m pip install -r requirements.txt           # Python 3.13 or 3.14
python final/pipeline.py                            # rebuild both submitted files (about a minute)
python run_all.py                                   # data steps 01-03 of both rounds
python run_all.py --models                          # the same, then the model steps
python run_all.py --rounds "2nd round" --models-only
```

`run_all.py` runs each script from its round folder, as the scripts expect. Everything needed is in
the repository, so every step runs from a fresh clone; only the 1st round's step 6 needs step 5's
expert predictions first. On this machine the 1st round's data steps take about 50 minutes and the
2nd round's data and model steps about 25. Steps 01 and 02 give the same bytes on every run; the
feature tests in step 03 can move in the fifth decimal of their standard errors, which changed no
feature decision in a full rerun.

## The data

Every fix is logged with the rows it changed, and changed rows carry a `dq_` flag. No policy,
customer, address or weather row is dropped, and nothing is imputed.

| Issue | 1st round | 2nd round |
|---|---:|---:|
| customer ids that lost their leading zeros | 123,664 rows | none |
| exposure recorded in days or on a 360-day basis | 13,880 | 595 |
| sums insured recorded in units instead of thousands | 3,527 | 349 |
| impossible years licensed, restored from the same policy's other years | 3,717 | none |
| claims stored twice under a second id (removed) | 604 | none |
| reserve typed with an extra digit | 1 claim | none |
| category spellings and local codes unified | construction type, occupation | construction type, occupation, gender, self-employed flag |
| claims outside every period of their policy (kept, flagged, left out of targets) | 413 | 36 |

A column becomes a feature only if it beats 20 random "probe" columns out of time, after rules have
dropped leakage, personal data, governance columns, constants and duplicates. That kept 53, 44 and
49 features (motor, buildings, contents) in the 1st round and 28, 14 and 9 in the 2nd, where two
years of a small book cannot prove weaker effects. Each round's
`reports/03_features/feature_register.csv` lists every column with its decision and tests.

## The models

Week 1 combines six experts per line (a total-cost GBM, a GLM, per-peril GBMs and three frequency ×
claim-size models) by geometric mean, recalibrated per line; out of time it beats one LightGBM in
both back-test years (0.1386 and 0.1375 against 0.1364 and 0.1338). Week 2 starts from a LightGBM
fitted on the developed market with the factors both markets share, used as an offset, so the local
models only learn how this market differs (forward test 0.1300 against 0.1026 without it). The
2025 book is 77% new business, so steps 9 to 12 keep the tested ranking and set the ten segment
levels from the public scores, with a model of the public sample's noise.

The write-up in `final/2_writeup/` and `KNOWLEDGE.md` explain the decisions and the evidence.

## Submissions

Every model step saves its 2025 file as the next numbered version in its round's `submissions/`
folder and never overwrites one. `versions.csv` records the model, script, date, out-of-time
evidence, public score and checksum of each version, and `submissions/README.md` shows the same
table. A run that reproduces an existing file adds no version. Every file is checked before it is
saved: one row per sample row id, in order, with every prediction finite and positive.

## Using the feature tables

* Training rows are in `data/features/policies_<line>.parquet`, rows to price in
  `data/features/test_<line>.parquet`. Keys (`row_id`, `policy_id`, `customer_id`, `address_id`,
  `year`, `period_start`, `period_end`) are not features; group cross-validation by `customer_id`.
* Targets: `claim_cost` (all perils, net of recoveries, repudiated claims left out), `n_claims`, and
  `n_<peril>` / `cost_<peril>` per peril.
* Offset: `log(exposure × comp_count)` for counts and `log(exposure × comp_cost)` for cost, because
  the last training year is not fully reported. The 2025 rows need no completeness.
* Text columns are categories and missing values stay missing; a `<column>_missing` flag exists
  where missingness itself predicts claims. Trend and large-loss treatment are left to the model
  steps, and dropped columns are still in `data/clean/`.
