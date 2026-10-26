# IAA Hackathon: clean data and a LightGBM price for both rounds

Data reading, cleaning, feature engineering and a LightGBM pricing model for the IAA AI-Assisted
Hackathon 2026, one folder per round. Steps 1-3 build the data, each as a script that does the
work and an executed notebook that shows the evidence. The model steps price 2025 with LightGBM
(see [The LightGBM model](#the-lightgbm-model)) and with the 1st round's winning ensemble (see
[The 1st round's winning ensemble](#the-1st-rounds-winning-ensemble-in-both-rounds)). Every
submission is kept as a numbered version (see [Submissions](#submissions)):

| Step | Script | Notebook | Output |
|---|---|---|---|
| 1. Read and understand | `01_data_reading.py` | `01_data_reading.ipynb` | `reports/01_reading/` (data dictionary, column profile, key checks) |
| 2. Clean | `02_data_cleaning.py` | `02_data_cleaning.ipynb` | `data/clean/` + `reports/02_cleaning/cleaning_log.csv` |
| 3. Engineer and select features | `03_feature_engineering.py` | `03_feature_engineering.ipynb` | `data/features/` + `reports/03_features/feature_register.csv` |
| 4. LightGBM model (1st round) | `04_lightgbm_model.py` | | `submissions/v<N>_lightgbm.csv` + `reports/04_model/` |
| 5-6. Winning ensemble (1st round) | `05_geo_experts.py`, `06_geo_ensemble.py` | | `data/experts/`, `submissions/v<N>_geo_ensemble.csv` + `reports/05_geo/`, `06_geo/` |
| 4. Developed prior (2nd round) | `04_developed_prior.py` | | `data/prior/` + `reports/04_prior/` |
| 5. LightGBM model (2nd round) | `05_lightgbm_model.py` | | `submissions/v<N>_transfer_lightgbm.csv` + `reports/05_model/` |
| 6-8. Ensemble on developed priors (2nd round) | `06_geo_priors.py`, `07_geo_experts.py`, `08_geo_ensemble.py` | | `data/prior/`, `data/experts/`, `submissions/v<N>_geo_ensemble_transfer.csv` + `reports/06_geo/` to `08_geo/` |

**File and column names are those of the case study** in every layer: `data/raw/` holds the
competition files, `data/clean/` the same tables with errors fixed, and `data/features/` one
training table `policies_<line>` and one 2025 table `test_<line>` per line, model-ready.

[`KNOWLEDGE.md`](KNOWLEDGE.md) summarises everything learned in the earlier work (data issues,
leakage, what worked in the models and what did not).

## Folder layout

```
IAA Hackathon/
  README.md, KNOWLEDGE.md, requirements.txt
  run_all.py            runs the steps of one or both rounds (--models adds the model steps)
  pricing_lgbm.py       LightGBM tools shared by the model steps (offset, trend, large losses, D2)
  geo_experts.py        the 1st round's winning ensemble: claims to ultimate, six experts, recalibration
  submissions.py        numbered submission files and their register
  make_notebooks.py     writes (and with --execute runs) the six notebooks
  1st round/            developed market: 2021-2024, 2025 to price
  2nd round/            emerging market: 2023-2024, 2025 to price, plus expenses and reference policies
     config.py          everything specific to the round: files, data dictionary, codes, thresholds,
                        and at the end the model settings
     helpers.py         reading, logging, profiling, claim matching, reporting completeness, tests
     01_data_reading.py / .ipynb
     02_data_cleaning.py / .ipynb
     03_feature_engineering.py / .ipynb
     04_lightgbm_model.py            1st round: the model and its 2025 submission
     05_geo_experts.py, 06_geo_ensemble.py               1st round: the winning ensemble
     04_developed_prior.py           2nd round: LightGBM on the 1st round, applied to this round
     05_lightgbm_model.py            2nd round: the model on top of that prior, and its submission
     06_geo_priors.py, 07_geo_experts.py, 08_geo_ensemble.py   2nd round: the ensemble on priors
     data/raw/          the case-study files (put them here)
     data/clean/        cleaned tables, same names
     data/features/     policies_<line>.parquet, test_<line>.parquet
     data/prior/        2nd round: the developed priors' log rates for every row
     data/experts/      the ensemble experts' predictions (back-tests, cross-validation, 2025)
     models/            the fitted LightGBM boosters
     submissions/       every submission made (v1, v2, ...) and their register
     outputs/           other model outputs (2nd round: predictions_lightgbm.parquet)
     reports/           01_reading/, 02_cleaning/, 03_features/, then one folder per model step
```

## Run it

```bash
python -m pip install -r requirements.txt        # Python 3.13 (3.12 works)
python run_all.py                                # both rounds; or: python run_all.py --rounds "2nd round"
python run_all.py --notebooks                    # ... and re-run the notebooks
python run_all.py --models-only                  # the model steps only (no raw files needed)
```

or step by step inside a round's folder: `python 01_data_reading.py`, `python
02_data_cleaning.py`, `python 03_feature_engineering.py`. The cleaned and model-ready tables are
already included; to re-run the steps, first copy the case-study files into each round's
`data/raw/` with their original names (listed in `data/raw/README.md`).

The round scripts expect the round folder to be the working directory. Use `run_all.py` from the
repository root, or open a round folder in the terminal and run its scripts there. This keeps
the imports simple and makes the data and report locations unambiguous.

Time on 2 CPU cores: 1st round about 30 minutes (steps 1-2 take 3 minutes; the feature tests run on
2.4 million rows), 2nd round a few minutes. Re-running step 3 with `--reuse-tests` reuses the
saved test results and takes 5 minutes (1st round) or seconds. Every run gives the same bytes.
The single LightGBM models take about 8 minutes (1st round) and 4 (2nd round) on 8 cores. The
ensembles take longer: the 1st round's experts about 2.5 hours (the three lines in parallel), the
2nd round's priors and experts about 40 minutes; the ensemble steps themselves take seconds.

## Download

| Zip | Contents |
|---|---|
| `IAA_Hackathon.zip` | code, notebooks, reports and docs of both rounds, and the 2nd round's `data/clean/` and `data/features/` |
| `IAA_Hackathon_1st_round_features_1of3.zip` ... `3of3` | the 1st round's model-ready tables (`data/features/`) |
| `IAA_Hackathon_1st_round_clean_1of3.zip` ... `3of3` | the 1st round's cleaned tables (`data/clean/`), only needed to build other features |

Unzip them all in the same place: each one fills its part of the `IAA Hackathon/` folder. The
case-study files themselves (`data/raw/`) are not included.

## What was done

**Cleaning** (each rule logged with the rows it changed and its evidence; changed rows carry a
`dq_` flag; no policy, customer, address or weather row is dropped; nothing is imputed):

| Issue | 1st round | 2nd round |
|---|---:|---:|
| customer ids that lost their leading zeros | 123,664 rows | - |
| exposure recorded in days or on a 360-day basis | 13,880 | 595 |
| sums insured recorded in units instead of thousands | 3,527 | 349 |
| impossible years licensed, restored from the same policy's other years | 3,717 | - |
| claims stored twice under a second id (removed) | 604 | - |
| reserve typed with an extra digit | 1 claim | - |
| category spellings and local codes unified | construction type, occupation | construction type, occupation, gender, self-employed flag |
| claims outside every period of their policy (kept, flagged, excluded from targets) | 413 | 36 |

**Feature engineering** adds what a model cannot build itself: claims history known at the start
of each year (policy, customer across lines, address), the other lines the customer holds, the
address's climate next to the year's weather, season on cover, property age, age at first
licence, sum insured per bedroom, and the targets (claims by peril, count and cost) with the share
already reported at 31 December 2024 (`comp_count`, `comp_cost`).

**Selection.** Every column is kept only if it adds value. Rules drop leakage, personal data,
governance columns, constants and exact duplicates. Out-of-time tests then compare each remaining
column with 20 random "probe" columns, both alone and inside a small throwaway LightGBM used only
to measure value. Near-duplicates go last. Each line's pass mark is calibrated on the probes, so at
most 1 in 20 pure-noise columns would pass.

| Features kept (+ exposure) | Motor | Buildings | Contents |
|---|---:|---:|---:|
| 1st round | 53 | 44 | 49 |
| 2nd round | 28 | 14 | 9 |

The 2nd round keeps fewer columns because two years of a small book (about 1,000 buildings and
1,000 contents claims) cannot prove weaker effects. Every candidate is in the register with its
test results. To keep weaker evidence, rebuild with `python 03_feature_engineering.py
--reuse-tests --z-threshold 1.0` (seconds).

## The LightGBM model

The model steps follow the emerging-market pricing repository (`iaa-pricing-part2`) and keep only
its LightGBM parts. In the 2nd round, that repository's best public score (0.16529) came from one
LightGBM per line fitted on top of a developed-market prior, with no GLM and no blend. The shared
code is in `pricing_lgbm.py`; each round's settings are at the end of its `config.py`.

In both rounds:

* The offset is log(exposure x share of the claims reported at 31 December 2024). The trees then
  learn only how risky a row is, not how long it was on cover or how much of its claims are still
  to come.
* Claims are projected to the cost level of the year priced with yearly trend rates per peril
  (motor severity 0-7.9% by peril, buildings severity 6%, contents frequency 1.4% and severity
  3.4%). In the 1st round, bodily injury claims of 2021-22 are multiplied by 0.82 to put them on
  the settlement level seen from 2023. Sums insured are restated in the same year's money.
* Row costs above the 99.5% quantile are capped, and the capped amount comes back as one load per
  line (1.02 to 1.06).
* The number of trees is chosen on 20% of the policies, then the model is refitted on every row
  with 10% more trees. The fitted total is rescaled to the actual total.
* Each step first checks itself out of time: fit on the years before 2024, score 2024. The 2024
  claims are not all reported yet, so each prediction is multiplied by its row's reported share
  before the comparison.

In the 2nd round only:

* Step 4 fits a LightGBM on every 1st-round row (200 trees) with the factors that mean the same
  thing in both markets: 30 for motor, 24 for buildings, 26 for contents. Region, weather, claims
  history and the other lines held are left out, and exact 1st-round ages become this round's age
  bands. Its log rate goes into the offset of step 5, so the 2nd-round trees only learn how this
  market differs. They need 20 to 110 trees.
* The prior always gets `renewal_count`. New business costs 1.10 times as much per year on cover
  in developed buildings (1.45 in motor), but the 1st round's step 3 dropped the column for
  buildings because the history features already told new customers apart, and the prior leaves
  those out. New business is 77% of the 2025 book against 12-15% of the training years, so this
  sets much of the 2025 level.
* The weather of the row's own year is dropped: the 2025 values lie far outside the two training
  years. The address climate stays.
* New-business and gender factors are measured on out-of-fold predictions, year by year, pooled
  over the two years and shrunk towards no effect (prior spread 15% for frequency, 10% for claim
  size, 30% for gender). For 2025 they are x1.04 (motor), x0.98 (buildings) and x1.15 (contents) on
  new business. Gender applies to motor and contents only, and it moves price between men and
  women without changing the line's total.

Out of time (fit before 2024, score the claims reported for 2024), Tweedie D^2 with power 1.5:

| 1st round | Motor | Buildings | Contents | All lines |
|---|---:|---:|---:|---:|
| LightGBM | 0.0862 | 0.1146 | 0.0917 | 0.1342 |

| 2nd round | Motor | Buildings | Contents | All lines |
|---|---:|---:|---:|---:|
| developed prior alone | 0.0838 | 0.0704 | 0.0985 | 0.1259 |
| LightGBM without the prior | 0.0801 | 0.0122 | 0.0757 | 0.1026 |
| transfer LightGBM | 0.0842 | 0.0754 | 0.0989 | 0.1277 |
| transfer LightGBM + factors estimated on 2023 alone | 0.0855 | 0.0749 | 0.1016 | 0.1286 |

The prior carries most of the 2nd round: without it LightGBM loses 0.025 of pooled D^2, and
buildings drops to 0.012. Buildings claims of 2024 came in at 0.59 of what 2023 predicted (2023
had a spike in fire costs), so its level-known D^2 (0.111) is far above the one shown. These are
holdout scores, not leaderboard scores, and one year of holdout moves by about 0.002 from noise
alone. The 2nd-round submission is close to the pricing repository's files: log correlation with
its best one (0.16529) is 0.986 for motor, 0.963 for buildings and 0.987 for contents, and its
total is 2.7% lower.

Each model step saves its submission as the next version in `submissions/` (see
[Submissions](#submissions)), in the order of the sample submission. The reports hold
`holdout.csv`, `summary.json` and the feature importance per line, plus `adjustments.json` in the
2nd round. The starter file `submission_2nd_round.csv` from `fit_example.py` stays where it was;
a byte-for-byte copy is the 2nd round's v1.

## Using the features with a boosting model

* `python fit_example.py "2nd round"` (or `"1st round"`) fits one LightGBM Tweedie model per line
  from these tables and writes a submission CSV: a working starting point. The model steps above
  are the full version.
* Training rows: `data/features/policies_<line>.parquet`; rows to price: `data/features/test_<line>.parquet`.
* Keys (`row_id`, `policy_id`, `customer_id`, `address_id`, `year`, `period_start`, `period_end`)
  are not features. Use `customer_id` to group cross-validation folds.
* Targets: `claim_cost` (all perils, net of recoveries, repudiated claims excluded),
  `n_claims`, and per peril `n_<peril>` / `cost_<peril>`; `n_open` counts open claims.
* Offset: `log(exposure x comp_count)` for claim counts, `log(exposure x comp_cost)` for cost.
  The last training year is not fully reported; the completeness columns correct for it. The
  2025 rows need no completeness (their target is the ultimate cost).
* Text columns are categories (use them as `category` dtype); missing values are left missing
  (boosting handles them); a `<column>_missing` flag exists where missingness itself predicts claims.
* The feature tables leave out trend and large-loss treatment, which are modelling choices; the
  model steps add them. The dropped columns are still in `data/clean/`.
