"""LightGBM Tweedie model with an exact log-offset.

The offset carries ``log(exposure x IBNR completeness x frequency trend)`` so the
trees learn only the *risk relativity*. LightGBM skips ``boost_from_average`` when
an ``init_score`` is supplied, so a fitted intercept (``base_``) is added to the
offset, which makes the first iteration calibrated.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class TweedieGBM:
    features: list[str]
    categorical: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    monotone: list[int] | None = None
    num_boost_round: int = 3000
    early_stopping_rounds: int = 150
    booster_: lgb.Booster | None = field(default=None, repr=False)
    base_: float = 0.0
    best_iteration_: int | None = None
    evals_: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def identity_link(self) -> bool:
        """L2-type objectives model the mean directly (no log link)."""
        return str(self.params.get("objective", "tweedie")) in ("regression", "regression_l2", "l2", "huber")

    # ------------------------------------------------------------------ fit
    def _lgb_params(self, seed: int) -> dict[str, Any]:
        p = {"objective": "tweedie", "metric": "tweedie", "verbose": -1, "seed": seed,
             "num_threads": 2, "force_col_wise": True, **self.params}
        for k in ("num_boost_round", "early_stopping_rounds"):
            p.pop(k, None)
        if self.monotone is not None and any(self.monotone):
            p["monotone_constraints"] = list(self.monotone)
            p.setdefault("monotone_constraints_method", "advanced")
        return p

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        offset: np.ndarray,
        X_es: pd.DataFrame | None = None,
        y_es: np.ndarray | None = None,
        offset_es: np.ndarray | None = None,
        num_boost_round: int | None = None,
        seed: int = 0,
        log_every: int = 0,
    ) -> "TweedieGBM":
        y = np.asarray(y, dtype=float)
        offset = np.asarray(offset, dtype=float)
        if (not self.identity_link and np.any(y < 0)) or not np.all(np.isfinite(offset)):
            raise ValueError("targets must be >= 0 (log-link objectives) and offsets finite")
        if self.identity_link:
            self.base_ = float(np.mean(y - offset))
        else:
            self.base_ = float(np.log(y.sum() / np.exp(offset).sum()))
        dtrain = lgb.Dataset(X[self.features], label=y, init_score=offset + self.base_,
                             categorical_feature=self.categorical or "auto", free_raw_data=True)
        valid_sets, names, callbacks = [dtrain], ["train"], []
        rounds = int(num_boost_round or self.num_boost_round)
        if X_es is not None:
            des = lgb.Dataset(X_es[self.features], label=np.asarray(y_es, dtype=float),
                              init_score=np.asarray(offset_es, dtype=float) + self.base_,
                              reference=dtrain, categorical_feature=self.categorical or "auto")
            valid_sets, names = [dtrain, des], ["train", "early_stop"]
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))
        self.evals_ = {}
        callbacks.append(lgb.record_evaluation(self.evals_))
        if log_every:
            callbacks.append(lgb.log_evaluation(log_every))
        self.booster_ = lgb.train(self._lgb_params(seed), dtrain, num_boost_round=rounds,
                                  valid_sets=valid_sets, valid_names=names, callbacks=callbacks)
        self.best_iteration_ = int(self.booster_.best_iteration or self.booster_.current_iteration())
        log.info("GBM fitted: %d trees (base %.4f)", self.best_iteration_, self.base_)
        return self

    # -------------------------------------------------------------- predict
    def predict_raw(self, X: pd.DataFrame, num_iteration: int | None = None) -> np.ndarray:
        """Log relativity (excluding offset and intercept)."""
        if self.booster_ is None:
            raise RuntimeError("model not fitted")
        return self.booster_.predict(X[self.features], raw_score=True,
                                     num_iteration=num_iteration or self.best_iteration_)

    def predict(self, X: pd.DataFrame, offset: np.ndarray, num_iteration: int | None = None) -> np.ndarray:
        """Expected cost = exp(raw + base + offset) (identity link: raw + base + offset)."""
        eta = self.predict_raw(X, num_iteration) + self.base_ + np.asarray(offset, dtype=float)
        return eta if self.identity_link else np.exp(eta)

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """SHAP (TreeSHAP) contributions on the log scale; last column is the bias."""
        c = self.booster_.predict(X[self.features], pred_contrib=True,
                                  num_iteration=self.best_iteration_)
        return pd.DataFrame(c, columns=[*self.features, "bias"], index=X.index)

    def importance(self) -> pd.DataFrame:
        gain = self.booster_.feature_importance("gain", iteration=self.best_iteration_)
        split = self.booster_.feature_importance("split", iteration=self.best_iteration_)
        imp = pd.DataFrame({"feature": self.features, "gain": gain, "split": split})
        imp["gain_share"] = imp["gain"] / imp["gain"].sum()
        return imp.sort_values("gain", ascending=False, ignore_index=True)

    # ----------------------------------------------------------- persistence
    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.booster_.save_model(str(path / "booster.txt"), num_iteration=self.best_iteration_)
        meta = {"features": self.features, "categorical": self.categorical, "params": self.params,
                "monotone": self.monotone, "base": self.base_, "best_iteration": self.best_iteration_}
        (path / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    @classmethod
    def load(cls, path: Path) -> "TweedieGBM":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        m = cls(features=meta["features"], categorical=meta["categorical"], params=meta["params"],
                monotone=meta["monotone"])
        m.booster_ = lgb.Booster(model_file=str(path / "booster.txt"))
        m.base_, m.best_iteration_ = float(meta["base"]), int(meta["best_iteration"])
        return m
