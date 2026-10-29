"""Build a non-leaderboard, gender-neutral Week 2 submission.

The existing v4 transfer ensemble is used as the validated risk ranking. This
model removes the explicit gender relativity from the final prediction and
rebases each line to preserve its validated portfolio total. No public
leaderboard files are read.

Usage:
    python final/best_practice_model.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
R2 = REPO / "2nd round"
FINAL = REPO / "final"
SOURCE = R2 / "submissions" / "v4_geo_ensemble_transfer.csv"
OUTPUT = FINAL / "1_predictions" / "week2_part2_best_practice_no_gender.csv"
REPORT = FINAL / "3_diagnostics" / "best_practice_model.json"
SAMPLE = R2 / "data" / "raw" / "sample_submission.parquet"
LINES = {"M": "motor", "B": "buildings", "C": "contents"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    predictions = pd.read_csv(SOURCE, dtype={"row_id": "string"})
    sample = pd.read_parquet(SAMPLE, columns=["row_id"])
    sample["row_id"] = sample["row_id"].astype("string")
    if not predictions["row_id"].equals(sample["row_id"]):
        raise ValueError("v4 prediction rows do not match the Week 2 sample order")

    recal = pd.read_csv(R2 / "reports" / "08_geo" / "recalibration.csv")
    gender_factor = recal.set_index("line")["male_vs_female"].to_dict()
    pred = predictions["prediction"].to_numpy(dtype=float)
    line = predictions["row_id"].str[0].map(LINES).to_numpy()
    output = pred.copy()
    changes = []
    for prefix, line_name in LINES.items():
        mask = line == line_name
        factor = float(gender_factor.get(line_name, 1.0))
        if line_name not in {"motor", "contents"}:
            factor = 1.0
        before_total = float(output[mask].sum())
        neutral = output[mask] / np.where(np.isclose(factor, 1.0), 1.0, factor)
        neutral *= before_total / neutral.sum()
        output[mask] = neutral
        changes.append({
            "line": line_name,
            "gender_factor_removed": factor,
            "rows": int(mask.sum()),
            "source_total": before_total,
            "output_total": float(neutral.sum()),
            "mean_relative_change": float(np.mean(neutral / pred[mask] - 1.0)),
            "max_abs_relative_change": float(np.max(np.abs(neutral / pred[mask] - 1.0))),
        })

    result = pd.DataFrame({"row_id": predictions["row_id"], "prediction": output})
    if not np.isfinite(output).all() or (output <= 0).any():
        raise ValueError("best-practice predictions must be finite and strictly positive")
    if result["row_id"].duplicated().any():
        raise ValueError("best-practice predictions contain duplicate row IDs")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False, lineterminator="\n")
    metadata = {
        "model_name": "week2_best_practice_no_gender",
        "source_model": str(SOURCE.relative_to(REPO)),
        "leaderboard_inputs_used": False,
        "public_score_level_overlay_used": False,
        "explicit_gender_adjustment_used": False,
        "data_quality_features_removed": False,
        "data_quality_features_note": "Requires retraining from raw features; not changed by this prediction-only overlay.",
        "validation_evidence": {
            "forward_2024_d2": 0.13472227517461532,
            "out_of_fold_d2_2023": 0.1319129530889488,
            "out_of_fold_d2_2024": 0.14307616215597285,
        },
        "changes": changes,
        "output": str(OUTPUT.relative_to(REPO)),
        "sha256": sha256(OUTPUT),
        "rows": int(len(result)),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
