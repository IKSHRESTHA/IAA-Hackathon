# IAA AI-Assisted Hackathon 2026: final submission

Team Actuarial Noob: Krishna Kumar Shrestha (solo). Deadline 30 October 2026, 11:59:59 PM EDT.

This folder holds what I submitted and the pipeline that produced it. The repository includes the
competition data and every intermediate result, so a fresh clone rebuilds both files without
downloading anything else.

| Form item | Where |
|---|---|
| Week 1 predictions | `1_predictions/week1_part1_predictions.csv` (public 0.12851, private 0.13830) |
| Week 2 predictions | `1_predictions/week2_part2_predictions.csv` (v13, public 0.17172) |
| Write-up | `2_writeup/writeup.pdf`, pages 1-5 (the appendix starts on page 6) |
| Video | recorded separately; `presentation/` has the slides and the 5-minute script |
| Code | this repository, https://github.com/IKSHRESTHA/IAA-Hackathon |

`1_predictions/manifest.csv` gives each file's source, scores and SHA-256 checksum.

## Reproduce

```bash
git clone https://github.com/IKSHRESTHA/IAA-Hackathon.git
cd IAA-Hackathon
python -m pip install -r requirements.txt        # Python 3.13 or 3.14; the clone is about 850 MB

python final/pipeline.py                         # quick run, about a minute
python final/pipeline.py --full --week 2         # from the raw data: about 80 minutes on 8 cores
python final/pipeline.py --full --week 1         # from the raw data: 2.5 to 8 hours
python final/diagnostics.py                      # SHAP, lift, GLM: about 25 minutes the first time
```

The quick run starts from the intermediate results in the repository, runs the last stages of both
pipelines, renders the write-up (Quarto; `--no-render` skips it). The output includes:

```
week1: rebuilt file matches the submitted file (same rows True, largest relative difference 3.6e-15 ...)
week2: rebuilt file matches the submitted file (same rows True, largest relative difference 0.0e+00 ...)
```

A full run replaces the intermediate results with fresh ones, so run it on a clean clone. A step
whose output differs from a registered file saves a new numbered version instead of overwriting
one, and the check passes when the largest relative difference is below 1e-5. Rebuilt files go to
`reproduced/`. Tested on Windows 11 with Python 3.14.2, LightGBM 4.7.0, pandas 3.0.5, numpy 2.5.3
and Quarto 1.8.27.

What has been checked on fresh clones:

| Check | Result |
|---|---|
| Quick run | Week 2 byte for byte; Week 1 within 3.6e-15 |
| Week 2 full run from raw data | within 9e-7; steps 01 and 02 byte for byte, no feature decision changed |
| Week 1 full run from raw data | within 3.4e-12 |
| Diagnostics refits against the stored models | Week 2 prior identical, GLMs within 2e-15, Week 1 expert within 7e-16 |
| Notebooks of steps 1 to 3, both rounds | all six rerun without errors |

## The pipeline

Week 1 (developed market). The submitted file came from `1st round/IAA_Part1_geo_models/`:
features, six experts on three fit sets, then their geometric mean recalibrated per line. The quick
run repeats the last stage on the experts' cached predictions (`cache/experts7_<line>.parquet`).

Week 2 (emerging market), with `run_all.py` running each round's steps in order:

| Steps | What they make | Run |
|---|---|---|
| `1st round/01` to `03` | the developed market's features, which the prior is fitted on | full |
| `2nd round/01` to `03` | cleaned data and features | full |
| `04`, `05` | developed-market prior; LightGBM on top of it (v2) | full |
| `06` to `08` | experts on their priors and their ensemble (v4) | full |
| `09` | v5: the best public file's segment totals, the ensemble's ranking inside | full and quick |
| `11 make --target public --scores-before v12` | v13, the Week 2 file | full and quick |

Two inputs come from outside the code and are kept as files. The public leaderboard scores are in
`2nd round/submissions/versions.csv` and `2nd round/data/reference/leaderboard_evidence.csv`;
`--scores-before v12` limits step 11 to the scores known when v13 was made. The best public file
before this work (0.16529) came from my earlier repository and sits in `2nd round/data/reference/`.

## Diagnostics and governance

`diagnostics.py` writes `3_diagnostics/report.md` with its tables and charts: calibration and lift
in ten bands, A/E by segment, SHAP (LightGBM's exact TreeSHAP) for the Week 1 total-cost expert and
both Week 2 tree models, and the Week 2 GLM's relativities. Week 1's ensemble keeps A/E between 0.87
and 1.04 in every band; Week 2's ranks well but its 2024 level was 14% high, mostly buildings. The
Week 1 trees are refitted with that code's own functions and saved in `3_diagnostics/week1_models/`
(not in git), so later runs take a few minutes.

`baseline_manifest.py`, `leakage_audit.py`, `governance_report.py` and `validation_report.py` write
the governance reports (baseline, leakage, feature governance, tail sensitivity, validation,
bootstrap uncertainty and population drift) to the same folder.

## Why these two files

Week 1: `submission_week1_v4_geo6.csv`, the six-expert ensemble. Its private score of 0.13830 ties
with a seven-expert version, so I kept the simpler one.

Week 2: v13, the best public score. I also selected v12 (public 0.17095) on Kaggle: it is the best
expected file on the rows not yet scored, and v13 only beats it if the true levels are as extreme
as the public rows suggest. `11_joint_levels.py make --scores-before v12` rebuilds v12.

The write-up explains the approach, decisions, use of AI, assumptions, limitations and guardrails.

## Folder

```
final/
  pipeline.py, diagnostics.py            run the pipeline; check the final models
  baseline_manifest.py, leakage_audit.py, governance_report.py, validation_report.py
  _quarto.yml, _variables.yml, src/      the write-up's settings and figures
  1_predictions/                         the two submitted files and manifest.csv
  2_writeup/                             writeup.qmd, references.bib, writeup.pdf
  3_diagnostics/                         diagnostics and governance reports
  presentation/                          slides and the video script
  reproduced/                            written by pipeline.py, not in git
```
