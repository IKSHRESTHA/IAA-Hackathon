"""Create governance, tail-sensitivity, validation, and drift reports."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
R2 = REPO / "2nd round"
OUT = REPO / "final" / "3_diagnostics"
LINES = ("motor", "buildings", "contents")


def feature_governance() -> pd.DataFrame:
    register = pd.read_csv(R2 / "reports" / "03_features" / "feature_register.csv")
    register["risk_class"] = np.select(
        [
            register["column"].str.startswith("dq_"),
            register["column"].eq("gender"),
            register["column"].isin(["occupation", "marital_status"]),
        ],
        ["data_quality", "sensitive", "potential_proxy"],
        default="ordinary",
    )
    return register.loc[register["risk_class"] != "ordinary",
                        ["line", "column", "decision", "reason", "risk_class"]]


def tail_sensitivity() -> pd.DataFrame:
    rows = []
    for line in LINES:
        frame = pd.read_parquet(R2 / "data" / "features" / f"policies_{line}.parquet",
                                columns=["claim_cost"])
        positive = frame.loc[frame["claim_cost"] > 0, "claim_cost"].astype(float)
        for quantile in (0.99, 0.995, 0.999):
            cap = float(positive.quantile(quantile))
            capped = positive.clip(upper=cap)
            rows.append({
                "line": line,
                "cap_quantile": quantile,
                "cap": cap,
                "positive_claims": int(len(positive)),
                "uncapped_cost": float(positive.sum()),
                "capped_cost": float(capped.sum()),
                "tail_load": float(positive.sum() / capped.sum()),
                "tail_share_uncapped": float((positive - capped).sum() / positive.sum()),
            })
    return pd.DataFrame(rows)


def validation_summary() -> pd.DataFrame:
    sources = [
        (R2 / "reports" / "05_model" / "holdout.csv", "transfer_holdout"),
        (R2 / "reports" / "08_geo" / "scores.csv", "ensemble_scores"),
    ]
    output = []
    for path, source in sources:
        if path.exists():
            frame = pd.read_csv(path)
            frame["source"] = source
            output.append(frame)
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def drift_summary() -> pd.DataFrame:
    records = []
    for line in LINES:
        train = pd.read_parquet(R2 / "data" / "features" / f"policies_{line}.parquet")
        test = pd.read_parquet(R2 / "data" / "features" / f"test_{line}.parquet")
        for column in ("exposure", "new_business", "gender", "dq_ncb_exceeds_licence"):
            if column not in train.columns or column not in test.columns:
                continue
            a, b = train[column], test[column]
            if pd.api.types.is_numeric_dtype(a):
                records.append({
                    "line": line, "column": column, "kind": "numeric",
                    "train_mean": float(a.mean()), "test_mean": float(b.mean()),
                    "relative_change": float(b.mean() / a.mean() - 1) if a.mean() else None,
                })
            else:
                train_dist = a.astype("string").value_counts(normalize=True)
                test_dist = b.astype("string").value_counts(normalize=True)
                levels = train_dist.index.union(test_dist.index)
                distribution_shift = float(
                    0.5 * (train_dist.reindex(levels, fill_value=0)
                           - test_dist.reindex(levels, fill_value=0)).abs().sum()
                )
                records.append({
                    "line": line, "column": column, "kind": "categorical",
                    "train_mean": None, "test_mean": None,
                    "relative_change": distribution_shift,
                })
    return pd.DataFrame(records)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    governance = feature_governance()
    tails = tail_sensitivity()
    validation = validation_summary()
    drift = drift_summary()
    governance.to_csv(OUT / "feature_governance.csv", index=False, lineterminator="\n")
    tails.to_csv(OUT / "tail_sensitivity.csv", index=False, lineterminator="\n")
    validation.to_csv(OUT / "validation_summary.csv", index=False, lineterminator="\n")
    drift.to_csv(OUT / "population_drift.csv", index=False, lineterminator="\n")
    metadata = {
        "status": "complete",
        "governance_rows": int(len(governance)),
        "tail_rows": int(len(tails)),
        "validation_rows": int(len(validation)),
        "drift_rows": int(len(drift)),
        "reports": [
            "feature_governance.csv",
            "tail_sensitivity.csv",
            "validation_summary.csv",
            "population_drift.csv",
        ],
    }
    (OUT / "governance_report.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
