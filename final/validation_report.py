"""Report paired customer-bootstrap uncertainty for the stored out-of-sample checks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
R2 = REPO / "2nd round"
OUT = REPO / "final" / "3_diagnostics"
LINES = ("motor", "buildings", "contents")
SEED = 20260515
REPS = 500


POWER = 1.5  # Tweedie power of the competition metric
FLOOR = 1e-12  # keeps log and power terms finite


def deviance(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Row-wise Tweedie deviance at POWER."""
    p = POWER
    return 2 * (np.power(y, 2 - p) / ((1 - p) * (2 - p))
                - y * np.power(mu, 1 - p) / (1 - p)
                + np.power(mu, 2 - p) / (2 - p))


def d2(y: np.ndarray, mu: np.ndarray) -> float:
    """Share of the null (mean-only) Tweedie deviance that the prediction removes."""
    y = np.asarray(y, dtype=float)
    mu = np.maximum(np.asarray(mu, dtype=float), FLOOR)
    return float(1 - deviance(y, mu).sum() / deviance(y, np.full_like(y, y.mean())).sum())


def evaluate(name: str) -> pd.DataFrame:
    """Bootstrap D2 by customer for one stored out-of-sample check."""
    members = json.loads((R2 / "reports" / "08_geo" / "summary.json").read_text())["members"]
    parts = []
    for line in LINES:
        frame = pd.read_parquet(R2 / "data" / "experts" / f"{line}_{name}.parquet")
        logs = []
        for member in members[line]:
            logs.append(np.log(np.maximum(frame[member].to_numpy(dtype=float), FLOOR)))
        prediction = np.exp(np.mean(logs, axis=0)) * frame["comp_cost"].fillna(1).to_numpy(dtype=float)
        parts.append(pd.DataFrame({
            "customer_id": frame["customer_id"].astype(str),
            "y": frame["y"].to_numpy(dtype=float),
            "mu": prediction,
        }))
    frame = pd.concat(parts, ignore_index=True)
    customers = frame["customer_id"].unique()
    rng = np.random.default_rng(SEED)
    estimates = np.empty(REPS)
    y = frame["y"].to_numpy(dtype=float)
    mu = frame["mu"].to_numpy(dtype=float)
    for i in range(REPS):
        sampled = rng.choice(customers, size=len(customers), replace=True)
        counts = pd.Series(sampled.astype(str)).value_counts()
        weight = frame["customer_id"].astype(str).map(counts).fillna(0).to_numpy(dtype=float)
        estimates[i] = d2(np.repeat(y, weight.astype(int)), np.repeat(mu, weight.astype(int)))
    return pd.DataFrame([{
        "check": name,
        "rows": int(len(frame)),
        "customers": int(len(customers)),
        "d2": d2(frame["y"], frame["mu"]),
        "bootstrap_se": float(estimates.std(ddof=1)),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
    }])


def main() -> int:
    result = pd.concat([evaluate("ho2024"), evaluate("cv")], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / "validation_uncertainty.csv", index=False, lineterminator="\n")
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
