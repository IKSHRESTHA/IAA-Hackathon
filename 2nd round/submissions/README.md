# Submissions

Every submission file made in this round, newest last. `versions.csv` holds the same table.

| Version | File | Model | Script | Created | Evidence | Description |
|---|---|---|---|---|---|---|
| v1 | `v1_starter_lightgbm.csv` | starter_lightgbm | `fit_example.py` | 2026-10-26 14:24 | not checked out of time | starter: one LightGBM per line, no developed prior, no trend or large-loss treatment |
| v2 | `v2_transfer_lightgbm.csv` | transfer_lightgbm | `05_lightgbm_model.py` | 2026-10-26 14:24 | forward 2024 D2 0.1286 (all lines) | LightGBM on top of the developed prior (with renewal_count), no weather of the year, new-business and gender factors |
| v3 | `v3_geo_ensemble_transfer.csv` | geo_ensemble_transfer | `08_geo_ensemble.py` | 2026-10-26 15:13 | out-of-fold D2 0.1381, forward 2024 D2 0.1349 (all lines) | geometric mean on developed priors (motor: total+fs_peril+glm; buildings: fs_lognorm+peril+fs_peril; contents: total+glm), slope from cross-validation, new-business and gender factors |
| v4 | `v4_geo_ensemble_transfer.csv` | geo_ensemble_transfer | `08_geo_ensemble.py` | 2026-10-26 15:16 | out-of-fold D2 0.1375, forward 2024 D2 0.1347 (all lines) | geometric mean on developed priors (motor: total+fs_peril+glm; buildings: fs_lognorm+peril; contents: total+glm), level only (slope 1) from cross-validation, new-business and gender factors |
