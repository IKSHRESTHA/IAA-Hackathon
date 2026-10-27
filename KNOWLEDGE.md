# What we learned so far (summary of all earlier work)

A compact record of everything the earlier rounds of work established about the data and the
models, so this folder can start from it instead of rediscovering it. Numbers come from the
earlier repositories and from the audit run for this folder (the cleaning logs in
`1st round/reports/02_cleaning/` and `2nd round/reports/02_cleaning/` reproduce the data ones).

## 1. The case study

| | 1st round (Part 1) | 2nd round (Part 2) |
|---|---|---|
| Market | developed | emerging (the insurer writes it from 2023) |
| Training years | 2021-2024 | 2023-2024 |
| Rows to price (2025) | 626,639 (motor 254,986, buildings 184,155, contents 187,498) | 209,063 (motor 90,521, buildings 52,295, contents 66,247) |
| Claims | 163,150 (after cleaning 162,546) | 9,258 |
| Extra deliverable | | office premium of 20 reference policies (`expenses`, `reference_policies`) |

* **Target of a row:** the *ultimate* cost of the claims whose loss date falls inside the row's
  period, net of the excess (already deducted in paid and incurred) and of recoveries, excluding
  repudiated claims.
* **Score:** Tweedie deviance D² with power 1.5, all rows of the three lines pooled.
* **Claims are valued at 31 December 2024**, so the claims of 2024 are not all reported yet.
  Bodily injury is the slowest peril (about half of the 2024 bodily-injury cost is reported),
  then subsidence; most perils are 95-98% reported within a year.
* **Files and columns** keep the case study's names everywhere in this folder: `policies_<line>`
  (training rows), `test_<line>` (2025 rows), `claims`, `customers`, `addresses`, `weather`,
  `sample_submission`, and in the 2nd round `expenses` and `reference_policies`.

## 2. Data quality: every issue found

| # | Issue | 1st round | 2nd round | Treatment |
|---|---|---|---|---|
| 1 | `customer_id` lost its leading zeros (stored as a number) | 98,110 training + 25,554 test motor rows | none | zero-padded back to 8 digits; every padded id then exists in `customers` |
| 2 | `exposure` recorded in days or on a 360-day basis | 13,880 training rows | 595 training rows | recomputed as (period_end - period_start + 1) / 365, the 2025 files' definition |
| 3 | `sum_insured` recorded in units instead of thousands (x1,000) | 3,527 rows (train + test) | 349 rows | divided by 1,000: the error values are 200-5,000x the median, genuine ones stay below 10x, and the same policy's other years are 1,000 times smaller |
| 4 | `construction_type` spelled several ways (`Brick`, `BRICK`, `W`, `T`, ...) or as local codes (`BRK`, `STN`, `WOD`, `MIX`) | all buildings rows | all buildings rows | one code each: brick, stone, timber, non_standard |
| 5 | `occupation` spelled several ways, with a `self-employed ` prefix | case only | 80 spellings | one spelling; the prefix becomes a `self_employed` flag (20 base occupations) |
| 6 | `gender` coded F/M | | all customers | f/m |
| 7 | **The same claim stored twice under a second `claim_id`** (11 characters instead of 10; every other field identical; 600 of the 604 in loss year 2022) | **604 claims (new in this pass)** | none | the copy is removed; 2022 claim counts were inflated by about 1.4% |
| 8 | A reserve typed with one digit too many (open bodily-injury claim `CL30973963`: incurred 11.9M against 1.1M for the next largest, paid 0.74M) | 1 claim | none | divided by 10 (1.19M) and flagged |
| 9 | Years licensed impossible (licence before age 17): in one year the value is up to 18 years too high, then returns to normal | 2,962 training + 755 test rows | not checkable (age bands) | **restored from the same policy's other years** (age at first licence is constant within a policy); 85 test rows without another row capped at age - 17 |
| 10 | No-claims bonus longer than the licence | 13% of motor rows | 16-22% (38-48% of drivers under 35) | kept as declared and flagged: too common to be typing errors, and the flag itself predicts claims |
| 11 | Annual mileage of 0 | 1.2% of motor rows | (no mileage) | kept and flagged (partly a placeholder: 45% of these cars commute) |
| 12 | Claims whose loss date is outside every period of their policy (mostly before the first row) | 413 | 36 | kept in `claims` with `match_status`, excluded from the targets |
| 13 | Motor claims of a peril the cover does not include (own damage on third-party covers) | 4 | 0 | kept, flagged |
| 14 | Accidental-damage claims on rows without AD cover | 1,653 (13% of home AD claims) | 74 (16%) | kept, flagged (mean size 1.03-1.04x the covered ones: plausibly basic cover) |
| 15 | Missing values | about 2.5% in 15 fields, at random | about 12% in 15 fields, at random | not imputed; `flood_zone` missingness is informative (missing addresses lie closer to rivers and lower) and gets a flag |
| 16 | `smoke_alarms` stored as True / False / None | yes | yes | read as a nullable boolean |
| 17 | `engine_size_cc` has the same distribution for electric cars as for petrol ones | yes | yes | kept (the generator ignores fuel); the feature tests decide its value |
| 18 | Leap year: a full 2024 row runs 1 January - 30 December (365-day policy year) | yes | yes | features use a 365-day year |
| 19 | 2025 weather lies partly outside the training range (wind exposure above every training year) | yes | yes | weather of the year kept, plus the address's climate (long-run mean) |

## 3. Columns that must not, or need not, be features

* **Leakage:** `renewal_decision` (an outcome after the period, absent from the 2025 files) and
  `customer_service_contacts` (driven by the same period's claims: 0 contacts on 0.2% of rows with
  a claim against 67% without; the 2025 file looks like the no-claim rows).
* **Exact redundancy (verified on 100% of rows):** `customer_since_year` = year - `renewal_count`;
  `new_business` = (`renewal_count` = 0); `discount_applied` is set by channel x new business;
  `compulsory_excess` is a constant per line (250 motor, 350 home); `date_of_birth` gives the same
  age as `driver_age` / `policyholder_age` (99.9%); `year_built` = year - property age.
* **Governance:** `written_premium` (the current tariff: a model using it would copy the old
  price; it is also annual, not pro-rated), `discount_applied`, and marketing or service
  attributes (`marketing_segment`, `campaign_code`, `email_domain`, `previous_insurer`,
  `branch_id`, `document_preference`, `marketing_consent`). `surname` is personal data.
* **Gender** is a permitted rating factor in the emerging market (it improved the public score by
  0.0034; removing it cost 0.0027), absent from the developed market.

## 4. Modelling lessons

**1st round (public leaderboard 0.12902, best of all our submissions).** A non-negative log-linear
pool of six experts per line (total-cost GBM, GLM, per-peril GBMs, frequency x gamma, frequency x
lognormal, per-peril frequency x severity). What mattered: the reserving layer (reporting
completeness in the offset, open-claim development, the reserve correction), as-at claims history
(a claim last year multiplies next year's frequency by 1.7-2.1), sums insured restated in 2025
money, large losses capped and reloaded, `make_model` encoded with credibility.

**2nd round (public 0.16319, then 0.16346 expected with both probes; first place 0.16446).**
* Transfer from the developed market (a developed-market model as an offset) beat every
  emerging-only model: 2024 holdout D² with the level known 0.136 against 0.115.
* Credibility-weighted new-business factors (+0.0005) and gender factors (+0.0034) helped; the
  unshrunk new-business effect hurt (-0.0011).
* Tried and not adopted: deep learning (Performer, MLP, TabM: no gain), discrimination-free prices
  (-0.0027), importance weighting towards the 2025 mix, forward factor search (winner's curse),
  hyper-parameter tuning (no out-of-time gain), the written premium as a signal (its effect flips
  sign between years).
* Last experiments: a peril-mix transfer model was +0.004 on the 2024 holdout (within noise, not
  submitted); an emerging-only multi-peril model was started and paused for this data work.

## 5. Why this folder exists

The models had reached a plateau, and the next gain is more likely to come from the data than from
another model. This folder rebuilds the data foundation for both rounds: every file read with the
right types, every error fixed with evidence and logged, information extracted from the raw
columns, and only the columns that predict claims out of time kept (see each round's
`reports/03_features/feature_register.csv`).

## 6. LightGBM lessons from the pricing repository

The emerging-market repository (`iaa-pricing-part2`) kept testing after the probes above. Its
public scores, all for the 2nd round:

| Submission | Public D² |
|---|---:|
| equal blend of prior, transfer GLM and transfer GBM | 0.16350 |
| blend tilted to the prior / to the GLM | 0.16334 / 0.16338 |
| 40% / 50% / 65% transfer GBM | 0.16376 / 0.16412 / 0.16458 |
| **100% transfer GBM** | **0.16529** (best) |
| 100% GBM, power calibration per line / overall | 0.16472 / 0.16486 |
| blend split by new business and renewals | 0.16272 |

* More weight on the transfer GBM helped at every step. Reshaping its predictions afterwards
  (power calibration) and splitting the blend by segment both helped on the holdout and hurt on
  the leaderboard.
* Tuned multi-seed GBMs and an XGBoost challenger never beat it (XGBoost was well below on the
  holdout); CatBoost could not be installed.
* Not submitted there: dropping the weather of the row's own year (2025 lies outside the two
  training years) and fitting the prior on every developed row (out of time, buildings 0.056 to
  0.066 and contents 0.098 to 0.101). This folder's 2nd-round model uses both.

What the LightGBM steps of this folder added (README, "The LightGBM model"):

* The prior needs `renewal_count` even where the 1st round's selection dropped it. There, the
  history features carried the new-business effect (x1.10 in developed buildings), and the prior
  leaves history out. With the flag, the 2025 buildings level rises by about 2%.
* The developed market's reporting patterns change the 2nd round's completeness by 0.5% at most,
  so the 2nd round keeps its own.
* 2nd-round buildings claims of 2024 came in at 0.59 of what 2023 predicted, because of a fire-cost
  spike in 2023 (283 per year on cover against 93). A one-year holdout cannot judge the buildings
  level.

## 7. The 1st round's winning ensemble, rebuilt here for both rounds

The six-expert geometric mean of `IAA_Part1_geo_models` (private 0.13830) now runs on this
folder's data (README, "The 1st round's winning ensemble").

* 1st round: it reproduces the original. Out of time, 0.1386 (2023) and 0.1375 (2024) against the
  original's own 0.1380 and 0.1367, and 0.0033 above a single LightGBM on 2024. The 2025 file has
  a log correlation of 0.996-0.998 with the original winning file per line.
* Frequency x claim-size experts beat a single cost model in the 1st round, every line and year.
  In the 2nd round they lose: two years give too few claims to learn claim sizes, even with a
  developed prior for the size. There, all six together score below `total` alone (contents,
  cross-year 0.1071 against 0.1087), and choosing members per line by cross-validation recovers
  the gain (0.1120 with `total` + `glm`).
* With a developed prior in the offset, the 2nd round wants very simple local models: LightGBM
  stumps (motor, contents) and a strong GLM ridge (alpha 1 to 30). A weak ridge overfits buildings
  (out-of-fold D^2 below 0).
* The 1st round's recalibration slope (b1 about 1.04-1.09) holds out of time there. In the 2nd round
  a slope fitted on one year does not carry to the other (-0.0035 to +0.0008), so it is fixed at 1.
* `make_model` is not a random label for risk: its 2021-22 claim frequency by code predicts
  2023-24 (correlation 0.22; 0.02 with shuffled labels). It enters as a credibility-weighted
  relativity.
* Developing open claims (recoveries still to come) lowers the 1st round's open incurred by about
  1.3%; no later-repudiation effect was significant in either round.

## 8. Leaderboard lesson of the 2nd-round ensemble

* v4 (the ensemble on developed priors) scored 0.16198 in public against 0.16529 for the pricing
  repository's single transfer LightGBM, although on the 2024 back-test it scores higher (0.1300
  against 0.1215 for that model on the same rows and claims).
* The 2024 back-test is mostly renewals (13% new business) and 2025 mostly new business (77%), so
  it cannot judge the level of new business, of buildings or of gender, which the best file
  settled with leaderboard probes. v4 departed from those levels (buildings 6% lower, motor
  renewals 7% higher) and lost.
* What the back-test can judge is the ranking inside a segment. There the ensemble is clearly
  better (0.1080 against 0.1016, rows weighted to the 2025 mix), so step 9 keeps the best file's
  segment totals and blends only the ranking inside (v5 50/50, v6 25/75).
* v5 (the best file's segment totals, half of the ranking replaced) scored exactly 0.16529: the
  public score barely reacts to ranking inside segments, and strongly to segment levels.
* For Tweedie p = 1.5 a segment's best level follows from one probe in closed form (README, step
  10). The pricing repository's motor new-business probe (+6.2% gave +0.00147) puts the best motor
  new-business level about 18% above the best file (+11% to +21% with the null deviance off by
  30%): 2025's motor new business is riskier than any model fitted on 2023-24 says.

## 9. Reading every public score at once (2nd round, step 11)

* Thirteen scored files share their segment structure: the pricing repository's trio files (v1, v2,
  v6, v8a, v8b: the same experts with different new-business, gender and buildings factors), its
  GBM-weight path (blends of those experts), its best file and our v5. One model, linear in each
  segment's best level, fits all thirteen within 0.0001. The five path files, left out, were
  predicted with one constant miss (-0.0020), so levels explain the whole path from 0.16334 to
  0.16458.
* The reading, as factors on v5: motor new business men x1.32, women x1.01; motor renewals men
  x1.16, women x0.88; contents new business x0.86; buildings x1.02. It holds within a few percent
  without v1 and v2, without the path, with the null deviance x0.8 or x1.25, and with family terms
  from 3e-4 to 3e-3. Only the contents gender split moves around, and contents has little leverage.
* Gender is the lever that was under-used. Adding the gender factors gained 0.0034, about 4.6 times
  what the step would gain if the factors were right. The male effect beyond the prior rose from
  +0.19 (2023) to +0.28 (2024), and on its own 2024 back-test the best file still under-prices men
  by about 15% in motor (contents: over-prices them by about 10%).
* Tested and not adopted: an age-specific gender factor for motor (men 17-24 are +0.39 beyond the
  model, 50-64 are -0.15) helps 2023 rows and hurts 2024 rows in cross-validation; only the young-
  male part repeats in both years. Seasonality and the 2025 start dates (2025 new business starts in
  January with 0.9 years on cover, against mid-year and 0.47 in 2023-24): motor claim frequency is
  flat by month, home claims peak in December and January, worth a few percent at most, which the
  levels absorb anyway.
* Probes v8 to v10 are not worth a submission (expected -0.0001, -0.0007, 0.0000). v11, every
  segment at its best level, is expected at 0.1692.
* Scores after step 11: v8 0.16552, v10 0.16556, v11 0.17036 (expected 0.1692). The level reading
  works; v11 did better than expected because the true levels sit further out than the model said.
* The public score is a sample of the 2025 rows. A segment's best level on it carries that sample's
  luck: sd sqrt(phi / (share x sum of mu^0.5)), 8% for motor new-business men and 13% to 24% for
  buildings at a 30% share. Chasing every public gain fits that luck: the best public levels
  (v13, expected 0.1714) are expected to do worse on unseen rows than v11; the best levels for
  unseen rows (v12, expected public 0.1708) do slightly better. Select v12 (and v11) for the final.
* Real gains on unseen rows now need new information, not more probes of the same segments: a
  ridge correction with every allowed factor on top of the out-of-fold predictions loses across
  years at every penalty, and the end-of-cover pattern for motor new business rests on 30 claims.
* v12 scored 0.17095 and v13 0.17172, both a little above forecast. Within the ten segments the
  public ceiling is about 0.1718, and v12 is the best expected file for the final ranking; tick
  v12 and v13 (v13 wins only if the true levels are as extreme as the public rows suggest).
* Permutation test: real factors pass the both-years correction test as often as shuffled ones,
  and shuffled buildings factors gain up to +0.006 by luck. Factor searches on two lumpy years
  find noise; only gender clearly beats chance. Selecting factors for agreeing across the two
  years and then scoring one of them is circular (it looked like +0.01 to +0.02 on buildings).
* From here, extra public probes mostly fit the public sample's luck: each one can add a little
  public score but nothing on the rows that decide the final ranking.
* Two ranking changes with evidence outside the leaderboard (step 12, v14): the best file's
  buildings and contents model uses this-year weather that 2025 puts far out of range, and its
  refit without it ranks better in cross-validation; and the power-calibration scores, read
  through step 11's levels, say motor prices are too compressed (true spread 1.1 to 1.2 times
  wider). A mild 1.05 widening is free if that reading is wrong. Ranking changes like these move
  the public score by less than its luck (about 0.0005), so the leaderboard cannot confirm them.
* v14 scored 0.17032 (v12 0.17095). Public luck on a ranking change is dominated by buildings: sd
  0.00071 for v14's buildings change alone, against 0.00025 for the motor spread. A buildings
  ranking change can never be judged on the public score; a motor one barely can.
