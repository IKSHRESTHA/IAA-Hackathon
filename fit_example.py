
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("round", nargs="?", default="2nd round", choices=("1st round", "2nd round"))
ROUND = parser.parse_args().round
DATA = Path(__file__).resolve().parent / ROUND / "data"
KEYS = ["row_id", "policy_id", "customer_id", "address_id", "year", "period_start", "period_end"]
PARAMS = dict(objective="tweedie", tweedie_variance_power=1.5, learning_rate=0.03, num_leaves=31,
              min_data_in_leaf=200, bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8,
              seed=1, verbose=-1)


def prepare(df: pd.DataFrame, train: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Text columns become categories (with the training levels), True/False/missing becomes 1/0/NaN."""
    X = df[features].copy()
    for c in features:
        if X[c].dtype == "string":
            X[c] = pd.Categorical(X[c], categories=sorted(train[c].dropna().unique()))
        elif X[c].dtype == "boolean":
            X[c] = X[c].astype("float32")
    return X


preds = []
for line in ("motor", "buildings", "contents"):
    train = pd.read_parquet(DATA / "features" / f"policies_{line}.parquet")   # 2021/2023-2024 rows + targets
    test = pd.read_parquet(DATA / "features" / f"test_{line}.parquet")        # 2025 rows to price
    features = [c for c in test.columns if c not in KEYS + ["exposure"]]       # the rest of test = features
    X, X_test = prepare(train, train, features), prepare(test, train, features)

    y = train["claim_cost"].to_numpy()
    seen = (train["exposure"] * train["comp_cost"]).to_numpy()   # years on cover x share of cost reported so far
    base = y.sum() / seen.sum()                                    # average cost per year on cover
    offset = np.log(seen * base)

    # number of trees: early stopping on 20% of the customers (all rows of a customer on the same side)
    valid = (pd.util.hash_pandas_object(train["customer_id"], index=False) % 5 == 0).to_numpy()
    d_fit = lgb.Dataset(X[~valid], y[~valid], init_score=offset[~valid])
    d_val = lgb.Dataset(X[valid], y[valid], init_score=offset[valid], reference=d_fit)
    trial = lgb.train(PARAMS, d_fit, num_boost_round=3000, valid_sets=[d_val],
                      callbacks=[lgb.early_stopping(100, verbose=False)])

    model = lgb.train(PARAMS, lgb.Dataset(X, y, init_score=offset), num_boost_round=trial.best_iteration)
    fitted = np.exp(model.predict(X, raw_score=True) + offset)
    scale = y.sum() / fitted.sum()                                 # re-balance: fitted total = actual total
    pred = scale * np.exp(model.predict(X_test, raw_score=True) + np.log(test["exposure"].to_numpy() * base))
    preds.append(pd.DataFrame({"row_id": test["row_id"], "prediction": pred}))
    print(f"{line}: {len(features)} features, {trial.best_iteration} trees, "
          f"2025 cost per year on cover {pred.sum() / test['exposure'].sum():,.1f} (training average {base:,.1f})")

preds = pd.concat(preds, ignore_index=True)
for layer in ("clean", "raw"):                                     # keep the sample submission's row order
    ss = DATA / layer / "sample_submission.parquet"
    if ss.exists():
        preds = pd.read_parquet(ss)[["row_id"]].merge(preds, on="row_id", how="left")
        break
out = Path(__file__).resolve().parent / f"submission_{ROUND.replace(' ', '_')}.csv"
preds.to_csv(out, index=False)
print(f"{len(preds):,} rows ({int(preds['prediction'].isna().sum())} missing) -> {out}")
