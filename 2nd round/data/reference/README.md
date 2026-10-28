# Reference files from the earlier pricing repository

Copied unchanged from my emerging-market repository (`iaa-pricing-part2`):

* `best_public_gbm100_submission.csv`: that repository's `gbm_100_transfer_expert_notebook_submission.csv`,
  public score 0.16529, the best before this work. One transfer LightGBM per line on the pipeline
  prior, with new-business factors (motor 1.1016, buildings 0.975, contents 1.185) and gender factors.
  Step 9 keeps its level in every segment and changes only the ranking inside.
* `best_ho2024_<line>.parquet`: its experts' 2024 back-test predictions (fitted on 2023), with the
  claims `y`, the reported share `comp_line`, `prior` and `tr_gbm`, the model behind the file.
* `best_gender_factors.json`: the gender factors that file applies to motor and contents.
* `best_no_year_weather_submission.csv`: the same model refitted without this-year weather and on
  the all-rows developed prior, for buildings and contents (motor rows are the same). It was never
  scored. In that repository's 5-fold cross-validation it ranks better: buildings 0.0972 to 0.0992,
  contents 0.1072 to 0.1092. `12_final_assembly.py` uses it.
* `leaderboard_evidence.csv`: that repository's scored files, summarised per segment for
  `11_joint_levels.py`.
