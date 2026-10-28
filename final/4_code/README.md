# Code

All code is under version control at **https://github.com/IKSHRESTHA/IAA-Hackathon**. The git
history shows how the solution was built, step by step and submission by submission.

`IAA_Hackathon_code_<commit>.zip` is a snapshot of that repository at the named commit: every
script, notebook, config, report and document, taken straight from git. It leaves out the
competition data, fitted model files and cached predictions (about 1 GB). `python final/build_final.py`
rebuilds it from the current commit.

## Where each submitted file comes from

| File | Made by | Rebuild |
|---|---|---|
| Week 1: `week1_part1_predictions.csv` | `1st round/IAA_Part1_geo_models/` (the 22 October code): six experts per line, geometric mean, per-line recalibration | `python rebuild_from_cache.py` in that folder: rebuilds the file from the cached expert predictions in a minute (largest relative difference 2e-15). `python run_pipeline.py` reruns everything from the competition files (about 2.5 hours). |
| Week 2: `week2_part2_predictions.csv` (v13) | `2nd round/11_joint_levels.py make --target public`, on top of v5 from `09_rebase_to_best.py` | Steps 1-11 of `2nd round/` (below). v13 was made on 26 October with the public scores up to v11 in the register. Rerunning now also reads the later scores (v12-v14), so it writes a new version instead of v13 byte for byte. The register keeps v13's checksum (`864b10b305758004`). |

## Pipeline map

Run from the repository root with `python run_all.py` (both rounds; `--models` adds the model
steps), or step by step inside a round's folder. Setup: Python 3.13, `pip install -r requirements.txt`.

| Step | 1st round (developed market) | 2nd round (emerging market) |
|---|---|---|
| Read and profile | `01_data_reading.py` / `.ipynb` | `01_data_reading.py` / `.ipynb` |
| Clean (every fix logged) | `02_data_cleaning.py` / `.ipynb` | `02_data_cleaning.py` / `.ipynb` |
| Features, selected against noise probes | `03_feature_engineering.py` / `.ipynb` | `03_feature_engineering.py` / `.ipynb` |
| Single LightGBM | `04_lightgbm_model.py` | `04_developed_prior.py`, `05_lightgbm_model.py` (transfer) |
| Expert ensemble | `05_geo_experts.py`, `06_geo_ensemble.py` | `06_geo_priors.py`, `07_geo_experts.py`, `08_geo_ensemble.py` |
| Segment levels and ranking | | `09_rebase_to_best.py`, `10_level_probe.py`, `11_joint_levels.py`, `12_final_assembly.py` |

Shared code at the root: `pricing_lgbm.py` (offset, trend, large losses, D²), `geo_experts.py`
(the six experts and the recalibration), `submissions.py` (numbered submission files and their
register). Each round's settings are in its `config.py`.

## Version control and records

* Every submission is saved as a new numbered version in `<round>/submissions/` and is never
  overwritten. `versions.csv` records the script, date, out-of-time evidence, public score and
  checksum of each one.
* `README.md` at the root explains every step and result; `KNOWLEDGE.md` records what was learned
  (data issues, what worked, what did not).
* Each step's evidence is in `<round>/reports/` (cleaning logs, feature tests, back-tests,
  cross-validation, level fits).
