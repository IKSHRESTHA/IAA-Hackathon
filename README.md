# IAA AI-Assisted Hackathon 2026

My pricing work for both parts of the hackathon. Week 1 is a developed market with four years of
data; Week 2 is an emerging market with two. In both, the task is to predict the 2025 claim cost of
every motor, buildings and contents policy, scored by the Tweedie deviance D² with power 1.5.

Start with [`final/`](final/README.md). It holds the two prediction files I submitted, the write-up,
and `final/pipeline.py`, which reruns the pipeline that produced both files.

| | Week 1 (developed market) | Week 2 (emerging market) |
|---|---|---|
| Training data | 2021-24, 163,150 claims | 2023-24, 9,258 claims |
| Rows to price | 626,639 | 209,063 |
| Submitted file | geometric mean of six experts, recalibrated per line | v13: ensemble ranking, segment levels for the 2025 mix |
| Public / private D² | 0.12851 / 0.13830 | 0.17172 / pending |

## What is where

```
final/              the submission, the write-up and pipeline.py
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
make_notebooks.py   writes, and with --execute runs, the notebooks of steps 1-3
fit_example.py      a short LightGBM example on the feature tables
KNOWLEDGE.md        what I learned along the way, kept as the project's memory
```

Each round folder has the same layout: `data/raw/` (the competition files), `data/clean/`,
`data/features/` (one training and one 2025 table per line), `models/`, `submissions/` (every file
made, with `versions.csv`), and `reports/`, which holds the evidence behind each step. The round's
settings are in its `config.py`.

## Run it

```bash
python -m pip install -r final/requirements.txt     # Python 3.13 or 3.14
python final/pipeline.py                            # rebuild both submitted files (about a minute)
python run_all.py                                   # data steps 01-03 of both rounds
python run_all.py --models                          # the same, then the model steps
python run_all.py --rounds "2nd round" --models-only
```

The round scripts expect their round folder as the working directory, and `run_all.py` takes care
of that. The competition files, cleaned tables, features and intermediate results are all in the
repository, so every step runs from a fresh clone. On this machine the 1st round's data steps took
about 50 minutes (most of it the feature tests on 2.4 million rows) and the 2nd round's data and
model steps about 25. Fitting the Week 1 experts from scratch takes about 2.5 hours.

Steps 01 and 02 give the same bytes on every run. The feature tests in step 03 can move in the fifth
decimal of their standard errors; in a full rerun that changed none of the keep-or-drop decisions.

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

Step 3 adds what a model cannot build for itself: claims history known at 1 January (policy,
customer, address), the other lines the customer holds, the address's climate next to the year's
weather, season on cover, property age, age at first licence, sum insured per bedroom, and the
targets with the share of each row's claims already reported at 31 December 2024.

A column stays only if it adds value. Rules drop leakage, personal data, governance columns,
constants and duplicates. The rest are tested out of time against 20 random "probe" columns, alone
and inside a small LightGBM, with a pass mark that lets at most 1 in 20 noise columns through.

| Features kept (plus exposure) | Motor | Buildings | Contents |
|---|---:|---:|---:|
| 1st round | 53 | 44 | 49 |
| 2nd round | 28 | 14 | 9 |

The 2nd round keeps fewer because two years of a small book (about 1,000 claims per home line)
cannot prove weaker effects. Each round's `reports/03_features/feature_register.csv` lists every
column with its decision and test results.

## The models

**Week 1.** A single LightGBM per line (step 4) scores 0.1342 on the 2024 holdout. The submitted
file combines six experts per line: a Tweedie GBM on total cost, a Tweedie GLM, one GBM per peril,
and three frequency × claim-size models. Their geometric mean is recalibrated per line with a
two-parameter Tweedie GLM fitted on the two back-test years.

| Back-test D², all lines | 2023 | 2024 |
|---|---:|---:|
| one LightGBM on total cost | 0.1364 | 0.1338 |
| best single expert (`fs_peril`) | 0.1379 | 0.1365 |
| geometric mean, recalibrated on the other year | 0.1386 | 0.1375 |

The file I submitted came from the code in `1st round/IAA_Part1_geo_models/` (22 October). Steps 5
and 6 rebuild the same method on this folder's cleaned data, and its 2025 predictions have a log
correlation of 0.996 to 0.998 with the submitted file.

**Week 2.** Two years of data are too few to learn a pricing model from scratch, so step 4 fits a
LightGBM on every Week 1 row, using only the factors that mean the same thing in both markets, and
step 5 puts its log rate in the offset of the Week 2 models.

| Forward test, fit 2023 and score 2024, all lines | D² |
|---|---:|
| LightGBM on Week 2 data only | 0.1026 |
| developed-market prior alone | 0.1259 |
| LightGBM on top of the prior | 0.1277 |
| ensemble on priors (steps 6-8) | 0.1300 |

Steps 6 to 8 rebuild the Week 1 experts on priors of their own. Cross-validation over customers
chose one-split trees, a strong GLM ridge, a member set per line and a recalibration slope of 1.

The ensemble ranked risks better but scored lower in public (v4, 0.16198 against 0.16529 for my
earlier single model), because the 2025 book is 77% new business against 13% in the 2024 back-test.
Steps 9 to 12 keep the tested ranking and set the ten segment levels (line × new business × gender)
from the public scores, with a model of the public sample's noise. v12 aims at the rows not yet
scored and v13 at the public rows. `KNOWLEDGE.md` and the write-up explain the reasoning.

## Submissions

Every model step saves its 2025 file as the next version in its round's `submissions/` folder and
never overwrites one. `versions.csv` there records the model, script, date, out-of-time evidence,
public score and a checksum for each version; `submissions/README.md` shows the same table. A run
that reproduces an existing file adds no version. Every file is checked before it is saved: one row
per sample row id, in its order, with every prediction finite and positive. Public scores go into
the register with `submissions.record_score(round_dir, "v4", 0.16198)`.

## Using the feature tables

* `python fit_example.py "2nd round"` (or `"1st round"`) fits one LightGBM Tweedie model per line and
  writes a submission: a working starting point.
* Training rows are in `data/features/policies_<line>.parquet`, rows to price in
  `data/features/test_<line>.parquet`.
* Keys (`row_id`, `policy_id`, `customer_id`, `address_id`, `year`, `period_start`, `period_end`) are
  not features. Group cross-validation folds by `customer_id`.
* Targets: `claim_cost` (all perils, net of recoveries, repudiated claims left out), `n_claims`, and
  `n_<peril>` / `cost_<peril>` per peril; `n_open` counts open claims.
* Offset: `log(exposure × comp_count)` for claim counts and `log(exposure × comp_cost)` for cost.
  The last training year is not fully reported, and the completeness columns correct for it. The
  2025 rows need no completeness.
* Text columns are categories, and missing values stay missing. A `<column>_missing` flag exists
  where missingness itself predicts claims.
* The feature tables leave out trend and large-loss treatment, which are modelling choices made in
  the model steps. Dropped columns are still in `data/clean/`.
