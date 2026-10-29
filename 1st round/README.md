# 1st round: developed market

The data steps for Part 1 (four training years, 2021-2024, and 626,639 rows to price for 2025), a
LightGBM baseline, and a rebuild of the ensemble I submitted. The submitted file itself came from
the code in [`IAA_Part1_geo_models/`](IAA_Part1_geo_models/README.md).

## Run it

From this folder, in order (or `python run_all.py` from the repository root):

```bash
python 01_data_reading.py          # read and profile the files: reports/01_reading/
python 02_data_cleaning.py         # fix what is provably wrong: data/clean/, reports/02_cleaning/
python 03_feature_engineering.py   # build and test features: data/features/, reports/03_features/
python 04_lightgbm_model.py        # one LightGBM per line: submissions/v1, reports/04_model/
python 05_geo_experts.py           # the six experts on three fit sets (about 2.5 hours): data/experts/
python 06_geo_ensemble.py          # their geometric mean, recalibrated: submissions/v2, reports/06_geo/
```

`03_feature_engineering.py --reuse-tests` reuses the saved test results and takes about 5 minutes
instead of most of an hour. The notebooks of steps 1 to 3 show the same work with its evidence;
to rerun one, from this folder: `python -m jupyter nbconvert --to notebook --execute --inplace 01_data_reading.ipynb`.
Steps 4 to 6 write their files (`submissions/`, `data/experts/`, `models/`) locally; git keeps only
their reports.

## Data

| Table | Raw rows | Clean rows | Clean columns |
|---|---:|---:|---:|
| `addresses` | 575,983 | 575,983 | 16 |
| `claims` | 163,150 | 162,546 | 19 |
| `customers` | 545,240 | 545,240 | 18 |
| `policies_buildings` | 707,726 | 707,726 | 29 |
| `policies_contents` | 721,617 | 721,617 | 30 |
| `policies_motor` | 979,968 | 979,968 | 43 |
| `sample_submission` | 626,639 | 626,639 | 2 |
| `test_buildings` | 184,155 | 184,155 | 28 |
| `test_contents` | 187,498 | 187,498 | 29 |
| `test_motor` | 254,986 | 254,986 | 42 |
| `weather` | 2,879,915 | 2,879,915 | 9 |

| Model-ready table (`data/features/`) | Rows | Columns | Features |
|---|---:|---:|---:|
| `policies_motor` / `test_motor` | 979,968 / 254,986 | 81 / 61 | 53 + exposure |
| `policies_buildings` / `test_buildings` | 707,726 / 184,155 | 75 / 52 | 44 + exposure |
| `policies_contents` / `test_contents` | 721,617 / 187,498 | 74 / 57 | 49 + exposure |

## Cleaning

`reports/02_cleaning/cleaning_log.csv` lists every rule with the rows it changed and its evidence.
In short:

* Spellings unified: occupation (531,778 rows) and construction type (560,124 training and 145,548
  test rows).
* Motor customer ids that lost their leading zeros padded back: 98,110 training and 25,554 test
  rows. Every padded id exists in `customers`.
* Exposure recorded in days or on a 360-day basis recomputed: 5,596 motor, 4,078 buildings and
  4,206 contents rows.
* Sums insured recorded in units rescaled: 1,395 buildings and 1,402 contents training rows, 370 and
  360 test rows.
* Impossible years licensed restored from the same policy's other rows: 2,962 training and 670 test
  rows; 85 test rows with no other row capped at age - 17.
* Kept as declared and flagged: zero annual mileage (11,928 and 3,108 rows) and a no-claims bonus
  longer than the licence (127,410 and 34,873 rows), which is too common to be a typing error.
* Claims: 604 copies stored under a second id removed, one reserve typed with an extra digit divided
  by 10, net incurred added (1,705 repudiated claims), 413 claims outside every policy period kept
  but left out of the targets, and 1,653 accidental-damage claims on rows without that cover flagged.

No policy, customer, address or weather row is removed; the only rows removed are the duplicate
claims. Missing values are not imputed.

## Feature selection

Every column of the joined table is kept or dropped with a reason, in
`reports/03_features/feature_register.csv` (one row per column, with its test results). The tests
compare each column with 20 random probe columns, alone and inside a small LightGBM, on three
out-of-time folds (learn on the earlier years, score 2022, 2023 and 2024).

| Reason | Motor | Buildings | Contents |
|---|---:|---:|---:|
| adds value (kept) | 53 | 43 | 48 |
| informative missingness (kept) | 0 | 1 | 1 |
| exposure (kept) | 1 | 1 | 1 |
| no value | 22 | 22 | 18 |
| governance | 9 | 9 | 9 |
| redundant, near-duplicate or constant | 6 | 8 | 7 |
| leakage, personal data, data-quality flags and others | 8 | 7 | 7 |

The pass mark is the lowest z at which at most 1 of the 20 probes would pass: z of at least 1 for
motor and contents and 1.5 for buildings.

## Targets in the training tables

* `n_claims` and `claim_cost`: claims with a loss date in the row's period, cost net of the excess
  and recoveries, repudiated claims left out, valued at 31 December 2024. `n_open` counts the open
  ones.
* `n_<peril>` and `cost_<peril>`: the same by peril.
* `comp_count` and `comp_cost`: the share of the row's claims already reported at 31 December 2024,
  from reporting-delay patterns corrected for truncation (`comp_<peril>` by peril). Use
  `log(exposure × comp_count)` or `log(exposure × comp_cost)` as the offset.

## Models

| Version | Step | What it is | Out-of-time D², all lines |
|---|---|---|---|
| v1 | `04_lightgbm_model.py` | one LightGBM Tweedie model per line | 2024 holdout 0.1342 |
| v2 | `05_geo_experts.py`, `06_geo_ensemble.py` | geometric mean of six experts, recalibrated per line | 2023 0.1386, 2024 0.1375 |

v2 rebuilds the submitted method on this folder's cleaned data. Its 2025 predictions have a log
correlation of 0.996 to 0.998 with the submitted file per line.
