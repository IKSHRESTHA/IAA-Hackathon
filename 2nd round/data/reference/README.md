# Reference: the best public 2nd-round file

Copied from the emerging-market pricing repository (`iaa-pricing-part2`), unchanged:

* `best_public_gbm100_submission.csv`: `gbm_100_transfer_expert_notebook_submission.csv`, public
  score 0.16529 (the best known). One transfer LightGBM per line on the pipeline prior, with
  new-business factors (motor 1.1016, buildings 0.975, contents 1.185) and gender factors.
* `best_ho2024_<line>.parquet`: its experts' 2024 back-test predictions (fit on 2023): claims `y`,
  reported share `comp_line`, `prior` and `tr_gbm` (the model behind the file).
* `best_gender_factors.json`: the gender factors that file applies (motor and contents).

`09_rebase_to_best.py` keeps that file's level in every segment and changes only the ranking inside.
* `best_no_year_weather_submission.csv`: `gbm_no_year_weather/gbm_full_prior_no_year_weather_submission.csv`,
  never scored. The same model refitted without this-year weather and on the all-rows developed
  prior, for buildings and contents (motor rows are identical to the best file). In the pricing
  repository's policy-grouped 5-fold cross-validation it ranks better: buildings 0.0972 -> 0.0992,
  contents 0.1072 -> 0.1092 (level fixed). `12_final_assembly.py` uses it for the best file's half
  of the buildings and contents ranking.
* `leaderboard_evidence.csv`: the pricing repository's scored files, summarised per segment for
  `11_joint_levels.py`.
