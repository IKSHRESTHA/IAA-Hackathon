# Final submission: IAA AI-Assisted Hackathon 2026

**Team:** Krishna Kumar Shrestha (solo team) · **Member:** Krishna Kumar Shrestha
(the write-up reads these from `_variables.yml`)

Submission deadline: **30 October 2026, 11:59:59 PM EDT**.

## What goes into the submission form

| Form item | Where it is | Notes |
|---|---|---|
| Team name, members | this file and `_variables.yml` | Krishna Kumar Shrestha, solo team |
| Week 1 prediction file | `1_predictions/week1_part1_predictions.csv` | 626,639 rows; public 0.12851, **private 0.13830** (best of Part 1) |
| Week 2 prediction file | `1_predictions/week2_part2_predictions.csv` | 209,063 rows; v13, **public 0.17172** (best of Part 2) |
| Video demo (5 min) | not in this folder | recorded separately |
| Write-up (5 pages) | `2_writeup/writeup.pdf`, source `2_writeup/writeup.qmd` | graded part is pages 1-5; page 6 is an appendix |
| Code under version control | https://github.com/IKSHRESTHA/IAA-Hackathon, snapshot in `4_code/` | map of the code in `4_code/README.md` |

`1_predictions/manifest.csv` lists, for each prediction file, its source in the repository, model,
scores and SHA-256 checksum.

## Which files and why

For Week 1 I used `submission_week1_v4_geo6.csv`, the geometric mean of six experts recalibrated
per line. Its private score of 0.13830 ties with `pick3_ens_geo7_moe.csv`, which adds a seventh
expert, so I kept the simpler one.

For Week 2 I used v13 because it has the best public score. I also selected v12 (public 0.17095) on
Kaggle for the final ranking. It is the best expected file on the rows not yet scored, and v13 only
beats it if the true levels are as extreme as the public rows suggest (write-up, section 4). To
submit v12 instead, point its `source` line in `build_final.py` at
`2nd round/submissions/v12_joint_levels_private.csv` and rerun the script.

## Rebuild this folder

```bash
python final/build_final.py              # copy + check prediction files, code snapshot, render write-up
python final/build_final.py --no-render  # the same without rendering
```

The script copies both prediction files byte for byte and checks them against each round's sample
submission (same row ids in the same order, every prediction finite and positive). It then writes
`manifest.csv`, the code snapshot zip, and renders the write-up. Rendering needs Quarto (1.4 or
later) and a Jupyter `python3` kernel with pandas, matplotlib and tabulate. The figures and
appendix tables are read from the repository's own registers and reports at render time.

## Layout

```
final/
  README.md            this file
  build_final.py       builds everything below
  _quarto.yml          Quarto project (renders the write-up)
  _variables.yml       team name and members, used by the write-up
  src/final_figures.py figures and tables for the write-up
  1_predictions/       the two prediction files + manifest.csv
  2_writeup/           writeup.qmd -> writeup.pdf
  4_code/              README.md (code map) + code snapshot zip
```
