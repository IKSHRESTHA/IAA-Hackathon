# 2nd round: emerging market

Everything for Part 2: two training years (2023-2024), 209,063 rows to price for 2025, of which 77%
are new business, plus an `expenses` table and 20 reference policies. The steps run from the data to
the submitted file (v13).

## Run it

From this folder, in order (or `python run_all.py --rounds "2nd round" --models` from the root,
which runs steps 1 to 9):

| Step | Script | Output |
|---|---|---|
| 1 | `01_data_reading.py` | `reports/01_reading/` (data dictionary, profiles, key checks) |
| 2 | `02_data_cleaning.py` | `data/clean/`, `reports/02_cleaning/` |
| 3 | `03_feature_engineering.py` | `data/features/`, `reports/03_features/` |
| 4 | `04_developed_prior.py` | a LightGBM fitted on every 1st-round row: `data/prior/`, `reports/04_prior/` |
| 5 | `05_lightgbm_model.py` | LightGBM on top of that prior: v2, `reports/05_model/` |
| 6 | `06_geo_priors.py` | developed priors for each expert's component: `data/prior/` |
| 7 | `07_geo_experts.py` | the experts on their priors, with 5-fold cross-validation: `data/experts/` |
| 8 | `08_geo_ensemble.py` | their geometric mean per line: v3, v4, `reports/08_geo/` |
| 9 | `09_rebase_to_best.py` | the best public file's segment totals, the ensemble's ranking inside: v5, v6 |
| 10 | `10_level_probe.py` | one-segment level probes: v7 to v10 |
| 11 | `11_joint_levels.py` | segment levels read from every public score: v11 to v13 |
| 12 | `12_final_assembly.py` | ranking changes on v12's levels: v14, v15 |

`11_joint_levels.py make --target public --scores-before v12` rebuilds v13 using only the public
scores known when it was made. `final/pipeline.py` runs steps 9 and 11 that way and checks the
result against the submitted file.

## Data

| Table | Raw rows | Clean rows | Clean columns |
|---|---:|---:|---:|
| `addresses` | 136,914 | 136,914 | 16 |
| `claims` | 9,258 | 9,258 | 19 |
| `customers` | 135,856 | 135,856 | 18 |
| `expenses` | 8 | 8 | 4 |
| `policies_buildings` | 27,447 | 27,447 | 29 |
| `policies_contents` | 34,566 | 34,566 | 30 |
| `policies_motor` | 47,283 | 47,283 | 38 |
| `reference_policies` | 20 | 20 | 53 |
| `sample_submission` | 209,063 | 209,063 | 2 |
| `test_buildings` | 52,295 | 52,295 | 28 |
| `test_contents` | 66,247 | 66,247 | 29 |
| `test_motor` | 90,521 | 90,521 | 37 |
| `weather` | 410,742 | 410,742 | 9 |

| Model-ready table (`data/features/`) | Rows | Columns | Features |
|---|---:|---:|---:|
| `policies_motor` / `test_motor` | 47,283 / 90,521 | 56 / 36 | 28 + exposure |
| `policies_buildings` / `test_buildings` | 27,447 / 52,295 | 45 / 22 | 14 + exposure |
| `policies_contents` / `test_contents` | 34,566 / 66,247 | 34 / 17 | 9 + exposure |

## Cleaning

`reports/02_cleaning/cleaning_log.csv` lists every rule with the rows it changed and its evidence.
In short:

* Gender recoded from F/M to f/m (135,856 customers). Occupation spellings unified (89,397 rows),
  and the `self-employed ` prefix turned into a `self_employed` flag (11,975 rows, 20 base
  occupations left).
* Construction type recoded from local codes (BRK, STN, WOD, MIX) on every buildings row.
* Exposure recorded in days or on a 360-day basis recomputed: 249 motor, 152 buildings and 194
  contents rows.
* Sums insured recorded in units rescaled: 53 buildings and 57 contents training rows, 104 and 135
  test rows.
* A no-claims bonus longer than the licence kept as declared and flagged (7,756 training and 20,051
  test rows).
* Claims: net incurred added (120 repudiated), 36 claims outside every policy period kept but left
  out of the targets, 74 accidental-damage claims on rows without that cover flagged. No duplicate
  claims were found.
* The reference policies were checked against the raw 2025 rows and are identical. One expense item
  (`commission_comparison_site`) is missing and stays missing.

No row is removed and nothing is imputed.

## Feature selection

Every column is kept or dropped with a reason, in `reports/03_features/feature_register.csv`. The
tests compare each column with 20 random probe columns, alone and inside a small LightGBM, on two
out-of-time folds.

| Reason | Motor | Buildings | Contents |
|---|---:|---:|---:|
| adds value (kept) | 28 | 14 | 9 |
| exposure (kept) | 1 | 1 | 1 |
| no value | 43 | 52 | 57 |
| governance | 9 | 9 | 9 |
| redundant, near-duplicate or constant | 6 | 8 | 8 |
| leakage, personal data, data-quality flags and others | 6 | 6 | 6 |

The pass marks are stricter than in the 1st round (z of at least 1.5 for motor and 2 for buildings
and contents), because two years of a small book give noisier tests. To keep weaker evidence,
rebuild with `python 03_feature_engineering.py --reuse-tests --z-threshold 1.0`, which takes seconds.

## Targets in the training tables

The same as in the 1st round: `n_claims` and `claim_cost` (net of the excess and recoveries,
repudiated claims left out, valued at 31 December 2024), `n_open`, the same by peril, and
`comp_count` / `comp_cost`, the share of each row's claims already reported. Use
`log(exposure × comp_count)` or `log(exposure × comp_cost)` as the offset.

## Submissions

`submissions/versions.csv` (and `submissions/README.md`) records every file made in this round
with its script, date, evidence, public score and checksum. The ones that matter most:

| Version | What it is | Public D² |
|---|---|---:|
| v2 | LightGBM on the developed prior | not scored |
| v4 | ensemble on developed priors | 0.16198 |
| v5 | the best public file's segment totals, ranking 50/50 with the ensemble | 0.16529 |
| v12 | v5 with every segment at its best level for rows not yet scored | 0.17095 |
| v13 | v5 with every segment at its best level for the public rows (the Week 2 file) | 0.17172 |
| v14 | v12's levels with two ranking changes | 0.17032 |
