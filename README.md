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
| 9. Best public levels, ensemble ranking (2nd round) | `09_rebase_to_best.py` | | `submissions/v<N>_best_levels_ranked_w<weight>.csv` + `reports/09_rebase/` |
| 10. Segment level probes (2nd round) | `10_level_probe.py` | | `submissions/v<N>_level_<segment>_x<factor>.csv` |
| 11. Every public score at once (2nd round) | `11_joint_levels.py` | | `submissions/v<N>_joint_levels_s<shrink>.csv` + `reports/11_joint/` |
| 12. Final assembly (2nd round) | `12_final_assembly.py` | | `submissions/v<N>_final_<changes>.csv` |

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

## The 1st round's winning ensemble, in both rounds

The best 1st-round private score (0.13830, public 0.12851) came from the code in
`IAA_Part1_geo_models`: six experts per line, their row-by-row geometric mean, and a small
recalibration per line. `geo_experts.py` holds that method, and each round has its own steps.

| Expert | What it is |
|---|---|
| `total` | Tweedie LightGBM on the capped total cost, x the line's large-loss load |
| `glm` | Tweedie GLM on banded numeric factors and grouped categories |
| `peril` | one Tweedie LightGBM per peril, with its own reported share and load, summed |
| `fs_gamma` | claim-frequency LightGBM (Poisson) x claim-size LightGBM (Gamma) |
| `fs_lognorm` | the same frequency x a lognormal claim size whose variance depends on the risk |
| `fs_peril` | frequency x claim size for each peril, summed |

Claims are developed first: an open claim loses the recovery ratio seen on closed claims, and a
share of it is declined later if recent years show significantly too few repudiations. Claim sizes
are learnt on claim records, from mature loss years only. The recalibration is a Tweedie GLM with
two parameters per line, log E = log(exposure x L) + b0 + b1 x log(geo rate / L), where L is the
mean's average rate in the scored year.

Three things differ from the original on purpose. The frequency trend divides the offsets (the
original multiplied by it, a defect its README admits; it only touched contents). The features are
this folder's step-3 selection plus the original's credibility-weighted `make_model`: step 3
dropped `make_model` as a random label, but its 2021-22 claim frequency by code predicts 2023-24
(correlation 0.22, against 0.02 with shuffled labels). Reporting completeness comes from step 3.

**1st round** (`05_geo_experts.py`, `06_geo_ensemble.py`). The experts are fitted on 2021-22 to
price 2023, on 2021-23 to price 2024, and on every year to price 2025. The recalibration is fitted
on the two back-tests; the check fits it on one back-test year and scores the other.

| 1st round, all lines pooled | 2023 | 2024 |
|---|---:|---:|
| `total` (one LightGBM) | 0.1364 | 0.1338 |
| best single expert (`fs_peril`) | 0.1379 | 0.1365 |
| geometric mean of the six | 0.1386 | 0.1373 |
| recalibrated, fitted on the other year | **0.1386** | **0.1375** |
| the original code, as it reported itself | 0.1380 | 0.1367 |

The frequency x claim-size experts beat a single cost model in every back-test, and the six
together beat each of them. The 2025 file (v2) follows the original winning file closely: log
correlation 0.998 (motor), 0.996 (buildings), 0.996 (contents), total 0.3% lower. The fitted
slopes (b1 1.04 motor, 1.08 buildings, 1.09 contents) match the original's.

**2nd round** (`06_geo_priors.py`, `07_geo_experts.py`, `08_geo_ensemble.py`). The round's best
recipe is the developed prior, so every expert starts from a prior of its own component. Step 6
fits claim frequency, claim size, and the cost, frequency and size of each peril on every 1st-round
row, with the factors both markets share (as in step 4). Step 7 puts them in the experts' offsets.
Where a peril has too few claims here (under 50, or under 300 for claim sizes), its prior alone,
re-levelled, replaces the model. Two training years give only one forward back-test, so step 7
also fits every expert on five folds of customers over 2023-2024. Those out-of-fold predictions
drive the tuning and the ensemble.

What cross-validation decided:

* LightGBM settings: stumps (one split per tree) for motor and contents, 7 leaves with large
  leaves and a strong L2 penalty for buildings. On top of the prior, simple trees do best.
* GLM ridge: strong (alpha 10 motor, 30 buildings, 1 contents). With the prior in the offset, a
  strong ridge means staying close to the developed market; a weak one overfits buildings badly.
* Members: here all six together score worse than `total` alone, because two years give few claims
  to learn claim sizes from. Each line keeps the combination that ranks its risks best (level
  known, mean of the two years): motor `total` + `fs_peril` + `glm`, buildings `fs_lognorm` +
  `peril`, contents `total` + `glm`.
* Recalibration: a fitted slope b1 changed the other year's D^2 by -0.0035 to +0.0008 with no
  consistent sign, so b1 stays at 1 and only the level is fitted, on both years' folds.
* New-business and gender factors, as in step 5, from the ensemble's out-of-fold predictions.

| 2nd round, all lines pooled | Out of fold 2023 | Out of fold 2024 | Forward 2024 |
|---|---:|---:|---:|
| `total` (transfer LightGBM, tuned) | 0.1301 | 0.1386 | 0.1278 |
| `glm` | 0.1300 | 0.1389 | 0.1279 |
| geometric mean of the chosen members | 0.1339 | 0.1398 | **0.1300** |
| step 5's transfer LightGBM (v2) | | | 0.1277 |

The forward check fits on 2023 and prices 2024: the ensemble gains 0.0023 over step 5's model. The
members were chosen on out-of-fold rows that include 2024, so that gain is a little optimistic.
The 2025 file (v4) stays close to the pricing repository's best public file: log correlation 0.989
(motor), 0.946 (buildings), 0.983 (contents), total 1.5% lower.

**What the leaderboard said, and step 9.** v4 scored 0.16198 in public, below the best known file
(0.16529, the pricing repository's single transfer LightGBM), although it ranks the 2024 risks
better. The difference is in the levels. That file's new-business, buildings and gender levels were
settled with leaderboard probes, and v4 departs from them (buildings 6% lower, motor renewals 7%
higher). The 2024 back-test cannot judge those levels: 2025 is 77% new business, 2024 about 13%.

`09_rebase_to_best.py` keeps the best file's total in every segment (line x new business x gender
for motor and contents; line x new business for buildings) and replaces only the ranking inside
each segment by a geometric blend of that file and the ensemble. On 2024, every segment at its
actual level and rows weighted to the 2025 mix, the ranking D^2 is 0.1016 for the best file, 0.1067
for a 50/50 blend (v5), 0.1078 for 25/75 (v6) and 0.1080 for the ensemble alone. The reference
files are in `data/reference/`.

**Step 10: levels from probes.** v5 changed a quarter of the rows by more than 10% and scored
exactly the best file's 0.16529, so the ranking inside segments hardly moves the public score; the
levels of the big segments do. For Tweedie p = 1.5 the effect of a level is exact algebra: scaling a
segment by c changes the public deviance by 4 [(c^-0.5 - 1) A + (c^0.5 - 1) B], with
A = sum y mu^-0.5 and B = sum mu^0.5 over the segment's public rows, so the best level is A / B and
one probe with a known score change pins it. `10_level_probe.py make` saves a probe, and
`10_level_probe.py read` turns its score into the segment's best level.

The pricing repository's own probe (motor new business +6.2%: +0.00147) puts the best motor
new-business level about 18% above the best file; with the null deviance off by 30% either way it
stays between +11% and +21%. v7 applies +15% (expected public about 0.1671). v8 to v10 probe the
next segments one at a time on the same base, so each score reads on its own.

**Step 11: every public score at once.** Any scored file is a level probe if it differs from another
scored file only by one factor per segment. The pricing repository's trio files (v1, v2, v6, v8a,
v8b) are the same three experts with different new-business, gender and buildings factors, and its
GBM-weight path (prior tilt to 65% GBM) blends those same experts. `11_joint_levels.py` puts these
11 public scores, the best file's and v5's into one model that is linear in each segment's best
level (prior: a line level, a new-business and a gender effect per line). It fits all 13 scores
within 0.0001. Left out of the fit, the five path files are predicted with one constant miss
(-0.0020 each, the trio's own ranking), so the levels explain the whole path.

What it reads, as factors on v5: motor new business x1.18 overall, made of men x1.32 and women
x1.01; motor renewals men x1.16 and women x0.88; contents new business x0.86; buildings x1.02. The
gender split is the lever nobody had pushed: adding the gender factors gained 0.0034, about 4.6
times what that step would gain if the factors were already right, and in the training data the
male effect beyond the prior rose from +0.19 (2023) to +0.28 (2024). v11 puts every segment at its
best level: expected public 0.1692 (+0.0039). Under every other reading tried (without v1 and v2,
without the path, null deviance x0.8 or x1.25, family sd 3e-4 to 3e-3, a tight gender prior) its
expected gain stays between +0.0026 and +0.0050. The same model expects v9 to lose 0.0007 and v8
and v10 to change nothing, so those probes are not worth a submission.

**Public luck, and what the final ranking rewards.** v11 scored 0.17036, above its expected 0.1692,
and v8 and v10 gained a little where the model expected nothing. But the public score covers only
a sample of the 2025 rows, and the final ranking scores the others. A segment's best level on the
public sample differs from its best level on the other rows by that sample's luck, with sd
sqrt(phi / (share x sum of mu^0.5)): at a 30% public share, 8% for motor new-business men, 13%
for buildings new business and 24% for buildings renewals (phi from 2023-24: motor 1,748,
buildings 3,843, contents 721). Step 11 now fits that luck separately, and reads the null
deviance from the scores (about 140 per row, against the model's own 120). Two files follow:

* v12, the best levels for rows not yet scored: expected public 0.1708, and on unseen rows
  +0.0041 against v5 (v11: +0.0040). It keeps what is well clear of the luck (motor men about
  x1.36, women x0.98 new business and x0.86 renewal, contents new business down) and pulls the
  rest back towards v5.
* v13, the best levels for the public rows: expected public 0.1714, but only +0.0035 on unseen
  rows, below v11. It chases buildings renewals to x0.81 and motor renewal men to x1.40, both
  inside the public luck.

The order holds for a public share from 20% to 50%. For the final selection, tick v12 (and v11).
v13 only raises the public number; if no file is ticked, Kaggle picks the best public ones, which
would be v13.

With v12 and v13 scored, the ten segments are worked out: the best public levels would score
0.1718 (v13 has 0.17172) and v12 already sits at the best levels for unseen rows. Widening or
narrowing the prior on the true levels does not explain the scores better (the evidence peaks at
the current widths for a public share of 20%, 30% and 50%). If the true levels are as far out as
the public rows say (prior twice as wide), v13 does as well as v12 on unseen rows; so the final
selection ticks both.

**Step 12: the ranking inside segments.** With the levels settled, two changes to the ranking are
backed by evidence from outside the public leaderboard. `12_final_assembly.py` rebuilds v12's
50/50 blend with them and puts every segment back at v12's total (without changes it reproduces
v12 to 3e-15):

* This-year weather. The best file's buildings and contents model splits on the weather of the
  year, and 2025 lies far outside 2023-24 (rainfall anomaly -128 mm against +-3 mm). The pricing
  repository's refit without it ranks better in its 5-fold cross-validation (buildings 0.0972 ->
  0.0992, contents 0.1072 -> 0.1092), and our own ensemble, which never used it, already ranked
  buildings better on 2024. The refit replaces the best file's half of the buildings and contents
  blend (prices move 9% and 6% inside segments).
* Motor spread. Read through step 11's levels, the two power-calibration scores say motor risk
  differences are 1.1 to 1.2 times wider (log scale) than the best file shows. Against v12, a
  motor widening of 1.05 inside each segment gains on unseen rows +0.0003 if the true widening is
  1.1, +0.0005 if 1.2, and nothing either way if the best file's spread is right (v12's blend is a
  little narrower than the best file). The pricing repository's 2024 back-test pointed the other
  way (0.91), hence the mild 1.05 rather than the 1.2 that fits the scores best.

v14 has both. Its public score can move either way by about 0.0005 from the luck of the public
rows alone, so the leaderboard cannot confirm these changes; they rest on the cross-validation
and on the spread reading. Only a drop of more than about 0.001 below v12 would argue against it.

A permutation test closes the search for missing factors: correcting the out-of-fold predictions
one factor at a time, real factors pass a both-years test no more often than the same factors
shuffled at random (motor 8 of 45 against 7.5, buildings 7 of 37 against 10.7, contents 11 of
38 against 12.2), and shuffled buildings factors gain up to +0.006 by luck alone. Only gender
(already priced) and, at the edge, contents `works_from_home` beat chance. Factors chosen for
agreeing across the two years and then scored on one of them looked strong (buildings +0.01 to
+0.02); chosen inside one year and scored on the other they lose (buildings -0.008 to -0.054).

Tested in this round and not adopted: a ridge correction on top of the out-of-fold predictions
with every allowed factor (fitted on one year, it loses on the other at every penalty: motor
-0.0002 to -0.06; contents at best +0.0001); motor policies that end before the year end (new
business: twice the expected cost, but on 30 claims, with no claim-driven pattern; renewals 0.84,
worth under 0.0001).

After each new public score: `submissions.record_score(...)`, then `python 11_joint_levels.py fit`
(levels and the expected score of every version) and `python 11_joint_levels.py make` (levels for
unseen rows; `--target public` for the public rows; `--set motor_m=1.08 motor_f=0.93` for a chosen
probe; `--public-share` once the leaderboard page states the share). The
pricing repository's files are summarised once in `data/reference/leaderboard_evidence.csv`
(`python 11_joint_levels.py evidence --ref-repo <path>` rebuilds it).

To try another combination without refitting the experts, run the ensemble step with fixed
members; each run is saved as a new version:

```bash
python 06_geo_ensemble.py --members total fs_gamma fs_peril --tag three    # in "1st round"
python 08_geo_ensemble.py --members total glm --tag total_glm               # in "2nd round"
```

## Submissions

Every model step saves its 2025 file as the next version in its round's `submissions/` folder,
named after the model, and never overwrites one. `submissions/versions.csv`, and the same table in
`submissions/README.md`, records for each version the model, the script, the date, the
out-of-time evidence, a description and a checksum. A run that reproduces an existing file adds no
version. Every file is checked before it is saved: one row per row_id of the sample submission, in
its order, and every prediction finite and positive.

| Round | Version | Model | Out-of-time evidence (all lines) |
|---|---|---|---|
| 1st | v1 | `lightgbm`: one LightGBM per line (step 4) | 2024 holdout 0.1342 |
| 1st | v2 | `geo_ensemble`: the winning ensemble (steps 5-6) | 2023 0.1386, 2024 0.1375 |
| 2nd | v1 | `starter_lightgbm`: the earlier `fit_example.py` file, unchanged | none; public 0.12519 |
| 2nd | v2 | `transfer_lightgbm`: LightGBM on the developed prior (step 5) | forward 2024 0.1286 |
| 2nd | v3 | `geo_ensemble_transfer`: first ensemble setup (slope fitted) | kept for comparison |
| 2nd | v4 | `geo_ensemble_transfer`: steps 6-8 as they are now | forward 2024 0.1300 (plain mean); public 0.16198 |
| 2nd | v5 | `best_levels_ranked_w50`: best public file's levels, 50/50 ranking (step 9) | 2024 ranking 0.1067 vs 0.1016; public 0.16529 |
| 2nd | v6 | `best_levels_ranked_w25`: the same, 25/75 ranking (step 9) | 2024 ranking 0.1078 vs 0.1016 |
| 2nd | v7 | `level_motor_new_x1p150`: v5, motor new business x1.15 (step 10) | expected public 0.1671 (step 11) |
| 2nd | v8 | `level_buildings_new_x1p080`: v5, buildings new business x1.08 | public 0.16552 |
| 2nd | v9 | `level_contents_new_x1p100`: v5, contents new business x1.10 | expected 0.1647: skip |
| 2nd | v10 | `level_motor_renewal_x1p070`: v5, motor renewals x1.07 | public 0.16556 |
| 2nd | v11 | `joint_levels_s100`: v5, every segment at its best level from all scores (step 11) | public 0.17036 (expected 0.1692) |
| 2nd | v12 | `joint_levels_private`: v5, best levels for rows not yet scored (step 11) | public 0.17095 (expected 0.1708); best expected final score |
| 2nd | v13 | `joint_levels_public`: v5, best levels for the public rows (step 11) | public 0.17172 (expected 0.1714); below v12 on unseen rows |
| 2nd | v14 | `final_nowx_motor1p05`: v12's levels; buildings and contents ranked without this-year weather; motor spread x1.05 (step 12) | expected public near v12's; about +0.0006 over v12 on unseen rows |

Public scores go into the register with `submissions.record_score(round_dir, "v4", 0.16198)`.

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
