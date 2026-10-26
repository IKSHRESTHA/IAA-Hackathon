# Submissions

Every submission file made in this round, newest last. `versions.csv` holds the same table.

| Version | File | Model | Script | Created | Evidence | Description |
|---|---|---|---|---|---|---|
| v1 | `v1_lightgbm.csv` | lightgbm | `04_lightgbm_model.py` | 2026-10-26 14:24 | 2024 holdout D2 0.1342 (all lines) | one LightGBM Tweedie model per line on the step-3 features |
| v2 | `v2_geo_ensemble.csv` | geo_ensemble | `06_geo_ensemble.py` | 2026-10-26 15:51 | cross-year D2 2023 0.1386, 2024 0.1375 (all lines) | geometric mean of total, fs_gamma, fs_lognorm, peril, fs_peril, glm, recalibrated per line (round 1 winning method) |
