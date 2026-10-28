# IAA AI-Assisted Hackathon 2026: final submission

Krishna Kumar Shrestha (solo team). Deadline 30 October 2026, 11:59:59 PM EDT.

| Form item | Where |
|---|---|
| Week 1 predictions | `1_predictions/week1_part1_predictions.csv` (public 0.12851, private 0.13830) |
| Week 2 predictions | `1_predictions/week2_part2_predictions.csv` (v13, public 0.17172) |
| Write-up | `2_writeup/writeup.pdf`, pages 1-5 (page 6 is an appendix) |
| Code | https://github.com/IKSHRESTHA/IAA-Hackathon, snapshot zip in `4_code/` |
| Video | recorded separately |

`1_predictions/manifest.csv` gives each prediction file's source in the repository, its scores and
its SHA-256 checksum.

## Why these two files

For Week 1 I used `submission_week1_v4_geo6.csv`, the geometric mean of six experts recalibrated
per line. Its private score of 0.13830 ties with `pick3_ens_geo7_moe.csv`, which adds a seventh
expert, so I kept the simpler one.

For Week 2 I used v13 because it has the best public score. I also selected v12 (public 0.17095) on
Kaggle for the final ranking. It is the best expected file on the rows not yet scored, and v13 only
beats it if the true levels are as extreme as the public rows suggest. To submit v12 instead, point
its `source` line in `build_final.py` at `2nd round/submissions/v12_joint_levels_private.csv` and
rerun the script.

## The approach in short

Week 1 is a developed market with four years of data (2021-24, 163,150 claims). I fixed the data
errors with a logged reason for each, built the target as the ultimate claim cost (share reported
in the offset, costs trended to 2025, large losses capped and loaded back), and priced each line
with six experts: a total-cost GBM, a GLM, per-peril GBMs and three frequency × claim-size models.
Their geometric mean, recalibrated per line, beat every single model in both back-test years.

Week 2 is an emerging market with two years (2023-24, 9,258 claims), and 77% of the 2025 book is
new business. I fit a model on the developed market using the factors both markets share and put it
in the offset, so the local models only learn how this market differs. The back-test could rank
risks but could not set segment levels, so I read the levels from the public scores with an exact
Tweedie formula and modelled the luck of the public sample separately. v12 targets the rows not yet
scored and v13 the public rows.

The write-up has the numbers, decisions, use of AI, assumptions, limitations and guardrails.

## The code

Each round folder runs the same steps, and `python run_all.py` at the repository root runs them all
(Python 3.13, `pip install -r requirements.txt`; the competition files go in each round's
`data/raw/`).

| Step | 1st round | 2nd round |
|---|---|---|
| Read, clean, build features | `01` to `03` scripts, each with a notebook | same |
| Single LightGBM | `04_lightgbm_model.py` | `04_developed_prior.py`, `05_lightgbm_model.py` |
| Expert ensemble | `05_geo_experts.py`, `06_geo_ensemble.py` | `06_geo_priors.py` to `08_geo_ensemble.py` |
| Segment levels | | `09_rebase_to_best.py` to `12_final_assembly.py` |

Shared code sits at the root: `pricing_lgbm.py`, `geo_experts.py` and `submissions.py`.

The Week 1 file came from `1st round/IAA_Part1_geo_models/`. In that folder,
`python rebuild_from_cache.py` rebuilds it from cached expert predictions in about a minute, and
`python run_pipeline.py` reruns everything from the competition files (about 2.5 hours).

The Week 2 file came from `2nd round/11_joint_levels.py make --target public`, on top of v5 from
`09_rebase_to_best.py`. It was made on 26 October with the public scores up to v11. Running it now
also reads the later scores, so it writes a new version rather than v13 itself.

Every submission is saved as a numbered version in `<round>/submissions/` and never overwritten.
`versions.csv` there records the script, date, evidence, public score and checksum of each one, and
`<round>/reports/` holds the evidence behind every step.

## Rebuilding this folder

`python final/build_final.py` copies both prediction files, checks them against the sample
submissions (same row ids in the same order, every prediction positive), writes the manifest and the
code zip, and renders the write-up. Add `--no-render` to skip the PDF. Rendering needs Quarto and a
Jupyter `python3` kernel with pandas, matplotlib and tabulate.

```
final/
  README.md
  build_final.py        builds everything below
  _quarto.yml, _variables.yml
  src/final_figures.py  figures and tables for the write-up
  1_predictions/        the two prediction files and manifest.csv
  2_writeup/            writeup.qmd and writeup.pdf
  4_code/               code snapshot zip (no competition data)
```
