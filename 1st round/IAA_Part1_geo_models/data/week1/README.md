# Competition files for the full Week 1 pipeline

`run_pipeline.py` reads the 11 Part 1 files from here, under their original names: `addresses`,
`claims`, `customers`, `policies_buildings`, `policies_contents`, `policies_motor`,
`sample_submission`, `test_buildings`, `test_contents`, `test_motor` and `weather` (parquet).

They are the same files as in `1st round/data/raw/`, and `python final/pipeline.py --full --week 1`
copies them here. `rebuild_from_cache.py` does not need them.
