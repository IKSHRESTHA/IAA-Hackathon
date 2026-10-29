"""Train a non-leaderboard four-model Week 2 ensemble.

The ensemble contains:
* the stored transfer GBM and GLM experts;
* a spline-based Tweedie generalized additive model;
* a random forest regressor.

Weights are selected only on the 2023 -> 2024 forward split. No leaderboard
scores or public segment levels are read.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import TweedieRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pricing_lgbm import d2_tweedie

warnings.filterwarnings("ignore", category=UserWarning)

R2 = REPO / "2nd round"
OUT = REPO / "final" / "1_predictions" / "week2_part2_four_model.csv"
REPORT = REPO / "final" / "3_diagnostics" / "four_model_ensemble.json"
LINES = ("motor", "buildings", "contents")
SEED = 20260515
DROP = {"row_id", "policy_id", "customer_id", "address_id", "exposure", "year",
        "gender", "dq_ncb_exceeds_licence"}


def features(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    candidates = [c for c in test.columns if c in train.columns and c not in DROP]
    return [c for c in candidates if not c.startswith("wx_") or c.endswith("_mean")]


def matrices(frame: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    X = frame[cols].copy()
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in cols if c not in numeric]
    X = X.astype(object).where(pd.notna(X), np.nan)
    return X, numeric, categorical


def gam_model(numeric: list[str], categorical: list[str], alpha: float) -> Pipeline:
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("splines", SplineTransformer(n_knots=5, degree=2, include_bias=False)),
    ])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    prep = ColumnTransformer([("num", num, numeric), ("cat", cat, categorical)],
                             sparse_threshold=0.0)
    return Pipeline([
        ("prep", prep),
        ("model", TweedieRegressor(power=1.5, link="log", alpha=alpha,
                                   max_iter=500, tol=1e-5)),
    ])


def rate_target(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    completeness = frame["comp_cost"].fillna(1.0).to_numpy(dtype=float)
    denominator = np.maximum(frame["exposure"].to_numpy(dtype=float) * completeness, 1e-6)
    y = frame["claim_cost"].fillna(0.0).to_numpy(dtype=float) / denominator
    weight = denominator
    cap = np.quantile(y[y > 0], 0.995) if np.any(y > 0) else 1.0
    return np.minimum(y, cap), weight


def fit_candidates(line: str) -> tuple[pd.DataFrame, dict]:
    train = pd.read_parquet(R2 / "data" / "features" / f"policies_{line}.parquet")
    test = pd.read_parquet(R2 / "data" / "features" / f"test_{line}.parquet")
    cols = features(train, test)
    X, numeric, categorical = matrices(train, cols)
    test_X, _, _ = matrices(test, cols)
    fit_mask = train["year"].eq(2023).to_numpy()
    score_mask = train["year"].eq(2024).to_numpy()
    y, sample_weight = rate_target(train)
    results = []
    final = []

    expert_cv = pd.read_parquet(R2 / "data" / "experts" / f"{line}_ho2024.parquet")
    expert_final = pd.read_parquet(R2 / "data" / "experts" / f"{line}_final.parquet")
    for name in ("total", "glm"):
        results.append(expert_cv[name].to_numpy(dtype=float) * expert_cv["comp_cost"].to_numpy(dtype=float))
        final.append(expert_final[name].to_numpy(dtype=float))

    gam_alphas = (0.1, 1.0, 10.0)
    best_gam = None
    best_score = -np.inf
    for alpha in gam_alphas:
        model = gam_model(numeric, categorical, alpha)
        model.fit(X.loc[fit_mask], y[fit_mask], model__sample_weight=sample_weight[fit_mask])
        pred = model.predict(X.loc[score_mask])
        pred = np.maximum(
            pred
            * train.loc[score_mask, "exposure"].to_numpy(dtype=float)
            * train.loc[score_mask, "comp_cost"].fillna(1.0).to_numpy(dtype=float),
            1e-8,
        )
        score = d2_tweedie(
            train.loc[score_mask, "claim_cost"].to_numpy(dtype=float), pred
        )
        if score > best_score:
            best_score, best_gam = score, alpha
    gam = gam_model(numeric, categorical, float(best_gam))
    gam.fit(X.loc[fit_mask], y[fit_mask], model__sample_weight=sample_weight[fit_mask])
    results.append(np.maximum(
        gam.predict(X.loc[score_mask])
        * train.loc[score_mask, "exposure"].to_numpy(dtype=float)
        * train.loc[score_mask, "comp_cost"].fillna(1.0).to_numpy(dtype=float),
        1e-8,
    ))
    final.append(np.maximum(gam.predict(test_X) * test["exposure"], 1e-8))

    rf = RandomForestRegressor(
        n_estimators=160, max_features=0.7, min_samples_leaf=80,
        max_depth=14, random_state=SEED, n_jobs=2,
    )
    rf_X = pd.get_dummies(X, dummy_na=True).replace([np.inf, -np.inf], np.nan)
    rf_X = rf_X.fillna(rf_X.median(numeric_only=True)).fillna(0.0)
    rf.fit(rf_X.loc[fit_mask], y[fit_mask], sample_weight=sample_weight[fit_mask])
    rf_cv_X = rf_X.loc[score_mask]
    rf_test_X = pd.get_dummies(test_X, dummy_na=True).reindex(columns=rf_X.columns, fill_value=0.0)
    rf_test_X = rf_test_X.fillna(rf_X.median(numeric_only=True)).fillna(0.0)
    results.append(np.maximum(
        rf.predict(rf_cv_X)
        * train.loc[score_mask, "exposure"].to_numpy(dtype=float)
        * train.loc[score_mask, "comp_cost"].fillna(1.0).to_numpy(dtype=float),
        1e-8,
    ))
    final.append(np.maximum(rf.predict(rf_test_X) * test["exposure"], 1e-8))

    cv = np.column_stack(results)
    cv_y = train.loc[score_mask, "claim_cost"].to_numpy(dtype=float)
    cv_exposure = train.loc[score_mask, "exposure"].to_numpy(dtype=float)
    # Log-space grid search over simplex; the score is forward validation only.
    best = (-np.inf, None)
    for w1 in np.linspace(0, 1, 11):
        for w2 in np.linspace(0, 1 - w1, 11):
            for w3 in np.linspace(0, 1 - w1 - w2, 11):
                w = np.array([w1, w2, w3, 1 - w1 - w2 - w3])
                pred = np.exp(np.log(np.maximum(cv, 1e-8)) @ w)
                pred *= cv_y.sum() / pred.sum()
                score = d2_tweedie(cv_y, pred)
                if score > best[0]:
                    best = (score, w)
    weights = np.asarray(best[1], dtype=float)
    full = np.column_stack(final)
    prediction = np.exp(np.log(np.maximum(full, 1e-8)) @ weights)
    # Preserve the GBM/GLM portfolio level without using leaderboard information.
    prediction *= (full[:, :2].mean(axis=1).sum() / prediction.sum())
    output = pd.DataFrame({"row_id": test["row_id"].astype(str), "prediction": prediction})
    report = {
        "line": line, "features": len(cols), "numeric_features": len(numeric),
        "categorical_features": len(categorical), "gam_alpha": float(best_gam),
        "weights": dict(zip(("gbm", "glm", "gam", "random_forest"), weights.tolist())),
        "forward_d2": float(best[0]),
        "component_forward_d2": dict(zip(
            ("gbm", "glm", "gam", "random_forest"),
            [d2_tweedie(cv_y, cv[:, i]) for i in range(cv.shape[1])],
        )),
        "rows": int(len(output)),
    }
    return output, report


def main() -> int:
    parts, reports = [], []
    for line in LINES:
        print(f"fitting {line}", flush=True)
        out, report = fit_candidates(line)
        parts.append(out)
        reports.append(report)
    result = pd.concat(parts, ignore_index=True)
    sample = pd.read_parquet(R2 / "data" / "raw" / "sample_submission.parquet", columns=["row_id"])
    sample["row_id"] = sample["row_id"].astype(str)
    result = result.set_index("row_id").loc[sample["row_id"]].reset_index()
    if not result["row_id"].is_unique or not np.isfinite(result["prediction"]).all() or (result["prediction"] <= 0).any():
        raise ValueError("invalid four-model submission")
    result.to_csv(OUT, index=False, lineterminator="\n")
    metadata = {"model": "gbm_glm_spline_gam_random_forest", "leaderboard_inputs_used": False,
                "gender_adjustment_used": False, "reports": reports,
                "output": str(OUT.relative_to(REPO)), "rows": int(len(result))}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
