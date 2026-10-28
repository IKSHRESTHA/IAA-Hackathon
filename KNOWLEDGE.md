# What I learned

The project's memory. Every idea tried, what the test said, and why it was kept or dropped. Each new
piece of work starts by reading this file, and each finished test is written back into it, so the
same idea is not tried twice. Numbers come from the reports in each round's `reports/` folder and
from the earlier repositories this work grew out of.

## 1. The case study

| | Week 1 (Part 1) | Week 2 (Part 2) |
|---|---|---|
| Market | developed | emerging (the insurer writes it from 2023) |
| Training years | 2021-2024 | 2023-2024 |
| Rows to price (2025) | 626,639 (motor 254,986, buildings 184,155, contents 187,498) | 209,063 (motor 90,521, buildings 52,295, contents 66,247) |
| Claims | 163,150 (162,546 after cleaning) | 9,258 |
| Extra tables | | `expenses` and 20 `reference_policies` to quote |

* The target of a row is the ultimate cost of the claims with a loss date in its period, net of the
  excess (already taken off paid and incurred) and of recoveries, with repudiated claims left out.
* The score is the Tweedie deviance D² with power 1.5, all rows of the three lines pooled.
* Claims are valued at 31 December 2024, so 2024 is not fully reported. Bodily injury is the slowest
  peril (about half of its 2024 cost is in), then subsidence; most perils are 95-98% reported within
  a year.
* File and column names follow the case study everywhere: `policies_<line>`, `test_<line>`,
  `claims`, `customers`, `addresses`, `weather`, `sample_submission`, and in Week 2 `expenses` and
  `reference_policies`.

## 2. Data issues

| # | Issue | Week 1 | Week 2 | Treatment |
|---|---|---|---|---|
| 1 | `customer_id` lost its leading zeros | 98,110 training + 25,554 test motor rows | none | padded back to 8 digits; every padded id then exists in `customers` |
| 2 | `exposure` in days or on a 360-day basis | 13,880 training rows | 595 | recomputed as (period_end - period_start + 1) / 365, the 2025 files' definition |
| 3 | `sum_insured` in units instead of thousands | 3,527 rows | 349 | divided by 1,000: these values are 200 to 5,000 times the median, and the same policy's other years are 1,000 times smaller |
| 4 | `construction_type` spelled several ways or as local codes | all buildings rows | all buildings rows | one code each: brick, stone, timber, non_standard |
| 5 | `occupation` spellings, with a `self-employed ` prefix | case only | 80 spellings | one spelling; the prefix becomes a `self_employed` flag |
| 6 | `gender` coded F/M | | all customers | f/m |
| 7 | the same claim stored twice under a second `claim_id` (11 characters instead of 10) | 604 claims, 600 of them in 2022 | none | copy removed; 2022 claim counts had been 1.4% too high |
| 8 | a reserve typed with an extra digit (`CL30973963`: incurred 11.9m, next largest 1.1m) | 1 claim | none | divided by 10 and flagged |
| 9 | years licensed impossible (licence before 17), too high in one year only | 2,962 training + 755 test rows | not checkable | restored from the same policy's other years; 85 test rows with no other row capped at age - 17 |
| 10 | no-claims bonus longer than the licence | 13% of motor rows | 16-22% | kept and flagged: too common to be typing errors, and the flag predicts claims |
| 11 | annual mileage of 0 | 1.2% of motor rows | no mileage | kept and flagged |
| 12 | claims outside every period of their policy | 413 | 36 | kept with `match_status`, left out of targets |
| 13 | motor claims of a peril the cover does not include | 4 | 0 | kept, flagged |
| 14 | accidental-damage claims without AD cover | 1,653 | 74 | kept, flagged (mean size 1.03-1.04 times the covered ones) |
| 15 | missing values | about 2.5% in 15 fields | about 12% in 15 fields | not imputed; missing `flood_zone` predicts claims and gets a flag |
| 16 | `smoke_alarms` stored as True / False / None | yes | yes | read as a nullable boolean |
| 17 | `engine_size_cc` has the same distribution for electric cars | yes | yes | kept; the feature tests decide |
| 18 | a full 2024 row runs 1 January to 30 December | yes | yes | features use a 365-day year |
| 19 | 2025 weather partly outside the training range | yes | yes | Week 1 keeps it next to the address's climate; Week 2 drops this-year weather |

## 3. Columns that must not, or need not, be features

* Leakage: `renewal_decision` (decided after the period, absent from the 2025 files) and
  `customer_service_contacts` (driven by the period's claims: zero contacts on 0.2% of rows with a
  claim against 67% without).
* Exact redundancy, checked on every row: `customer_since_year` = year - `renewal_count`;
  `new_business` = (`renewal_count` = 0); `discount_applied` is set by channel and new business;
  `compulsory_excess` is a constant per line; `date_of_birth` gives the same age as `driver_age` or
  `policyholder_age`; `year_built` = year - property age.
* Governance: `written_premium` (a model using it would copy the old tariff), `discount_applied`,
  marketing and service fields, and `surname` (personal data).
* Gender is a permitted rating factor in the emerging market only.

## 4. Week 1

* What mattered: the reserving layer (reported share in the offset, open-claim development, the
  reserve correction), claims history known at 1 January (a claim last year multiplies next year's
  frequency by 1.7 to 2.1), sums insured in 2025 money, large losses capped and loaded back, and
  `make_model` encoded with credibility.
* `make_model` is not a random label: its 2021-22 claim frequency by code predicts 2023-24
  (correlation 0.22, against 0.02 with shuffled labels).
* Frequency × claim-size experts beat a single cost model in every line and year, and the six
  experts together beat each one. Back-test D²: 0.1386 (2023) and 0.1375 (2024) against 0.1364 and
  0.1338 for one LightGBM.
* The recalibration slope (b1 about 1.04 to 1.09) holds when fitted on one year and scored on the
  other.
* Developing open claims (recoveries still to come) lowers open incurred by about 1.3%; no
  later-repudiation effect was significant.
* A sign error in the submitted code: the contents frequency trend multiplies the offset instead of
  dividing it. The recalibration absorbs most of it (intercept about +10%). It is kept in
  `IAA_Part1_geo_models/` so that code still reproduces the submitted file, and fixed in the rebuild.
* Submitted file: public 0.12851, private 0.13830. A seven-expert version scored the same privately.

## 5. Week 2: models

* Transfer from the developed market beats every model built on Week 2 data alone. Forward D²:
  0.1277 with the prior against 0.1026 without; buildings 0.075 against 0.012.
* The prior needs `renewal_count` even where Week 1's selection dropped it. In Week 1 the history
  features carried the new-business effect, and the prior leaves history out.
* Week 1's reporting patterns change Week 2's completeness by 0.5% at most, so Week 2 keeps its own.
* With a prior in the offset, simple local models do best: one-split trees for motor and contents,
  a strong GLM ridge (alpha 1 to 30; a weak one overfits buildings below D² = 0).
* All six experts together lose here, because two years give too few claims to learn claim sizes
  (contents: 0.1071 against 0.1087 for the total-cost model alone). Each line keeps its best set
  out of all 63 combinations.
* A recalibration slope fitted on one year does not carry to the other (-0.0035 to +0.0008), so it
  is fixed at 1.
* Credibility-weighted new-business factors helped (+0.0005 public); unshrunk ones hurt (-0.0011).
* Buildings 2024 came in at 0.59 of what 2023 predicted, after a 2023 fire-cost spike (283 per year
  on cover against 93). One year of holdout cannot judge the buildings level.
* Tried and not adopted: deep learning (Performer, MLP, TabM), discrimination-free prices (-0.0027
  public), importance weighting towards the 2025 mix, forward factor search, hyper-parameter tuning
  in the earlier repository (no out-of-time gain), the written premium as a signal (its effect flips sign between years), XGBoost,
  power calibration and blends split by segment (both helped the holdout and hurt the leaderboard).

## 6. Week 2: levels and the leaderboard

* The 2024 back-test is 13% new business and 2025 is 77%. The back-test can judge the ranking inside
  a segment, but not the level of new business, buildings or gender.
* The ensemble (v4) ranked 2024 better than my earlier single transfer LightGBM and scored lower in
  public (0.16198 against 0.16529). v5 kept that file's segment totals with a 50/50 blend of
  rankings and scored exactly 0.16529, although it changed a quarter of the rows by more than 10%.
  The public score hardly reacts to ranking inside segments.
* For Tweedie p = 1.5, scaling a segment by c changes the deviance by exactly
  4 [(c^-0.5 - 1) A + (c^0.5 - 1) B], with A = sum y mu^-0.5 and B = sum mu^0.5 over its public rows,
  so the best level is A / B. Any scored file that differs from another only by segment factors
  measures it.
* One model, linear in the ten best levels, fits 13 public scores within 0.0001. Read as factors on
  v5, motor new business needs men about x1.37 and women about x0.97. Gender was the lever nobody
  had pushed: in the training data the male effect beyond the prior rose from +0.19 (2023) to +0.28
  (2024).
* The public score is a sample of 2025, so a segment's best public level carries luck: sd
  sqrt(phi / (share × sum of mu^0.5)), 8% for motor new-business men and 13% to 24% for buildings at
  a 30% share. v13 aims at the public rows; v12 pulls the noisy segments back for the rows not yet
  scored. v12 scored 0.17095, v13 0.17172. Both are selected for the final ranking.
* Factor searches on two lumpy years find noise. In a permutation test real factors passed a
  both-years check no more often than shuffled ones (motor 8 of 45 against 7.5 by chance, buildings
  7 of 37 against 10.7, contents 11 of 38 against 12.2), and shuffled buildings factors gained up to
  +0.006 by luck. Choosing factors for agreeing across the two years and then scoring one of them is
  circular. Only gender clearly beats chance.
* Ranking changes move the public score less than its noise. v14 (no this-year weather in buildings
  and contents, motor spread x1.05) scored 0.17032, 0.0006 below v12, against a luck sd of 0.0008,
  most of it from buildings. A buildings ranking change can never be judged on the public score.
* Also tested and not adopted: an age-specific gender factor for motor (only the young-male part
  repeats in both years), a ridge correction on top of the out-of-fold predictions (loses across
  years at every penalty), and motor policies ending before the year end (rests on 30 claims).

## 7. The earlier pricing repository

The best public file before this work (0.16529) came from my emerging-market repository
(`iaa-pricing-part2`): one transfer LightGBM per line. Its public scores along the way:

| Submission | Public D² |
|---|---:|
| equal blend of prior, transfer GLM and transfer GBM | 0.16350 |
| blend tilted to the prior / to the GLM | 0.16334 / 0.16338 |
| 40% / 50% / 65% transfer GBM | 0.16376 / 0.16412 / 0.16458 |
| 100% transfer GBM | 0.16529 |
| 100% GBM, power calibration per line / overall | 0.16472 / 0.16486 |
| blend split by new business and renewals | 0.16272 |

More weight on the transfer GBM helped at every step. Its refit without this-year weather ranked
better in its own cross-validation (buildings 0.0972 to 0.0992, contents 0.1072 to 0.1092).
