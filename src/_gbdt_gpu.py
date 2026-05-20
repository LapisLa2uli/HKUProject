"""Optional XGBoost CUDA training for Poisson regression and binary classification."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

_REG_DEFAULTS = {
    "objective": "count:poisson",
    "tree_method": "hist",
    "learning_rate": 0.06,
    "max_depth": 8,
    "max_leaves": 63,
    "subsample": 1.0,
    "lambda": 0.0,
    "seed": 42,
    "verbosity": 0,
}
_CLF_DEFAULTS = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "learning_rate": 0.08,
    "max_depth": 8,
    "max_leaves": 48,
    "subsample": 1.0,
    "lambda": 0.0,
    "seed": 42,
    "verbosity": 0,
}


def want_gpu(env_prefix: str = "MODEL") -> bool:
    """True if CUDA XGBoost should be used (env ``{PREFIX}_USE_GPU`` / ``{PREFIX}_CPU``)."""
    if os.environ.get(f"{env_prefix}_CPU", "").strip().lower() in ("1", "true", "yes"):
        return False
    if os.environ.get(f"{env_prefix}_USE_GPU", "auto").strip().lower() == "0":
        return False
    return cuda_xgboost_available()


def cuda_xgboost_available() -> bool:
    try:
        import xgboost as xgb

        X = np.random.rand(32, 4).astype(np.float32)
        y = np.abs(np.random.randn(32)).astype(np.float32)
        dm = xgb.DMatrix(X, label=y)
        xgb.train(
            {"tree_method": "hist", "device": "cuda", "objective": "reg:squarederror"},
            dm,
            num_boost_round=1,
            verbose_eval=False,
        )
        return True
    except Exception:
        return False


def _as_float32(X: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.nan_to_num(X, nan=0.0), dtype=np.float32)


def fit_regressor(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
    *,
    use_gpu: bool,
    num_boost_round: int = 600,
    params: dict[str, Any] | None = None,
) -> tuple[Any, str]:
    if use_gpu:
        import xgboost as xgb

        p = {**_REG_DEFAULTS, "device": "cuda", **(params or {})}
        dtrain = xgb.DMatrix(_as_float32(X), label=y.astype(np.float32), weight=sample_weight)
        booster = xgb.train(p, dtrain, num_boost_round=num_boost_round)
        return booster, "xgboost_cuda"
    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=(params or {}).get("learning_rate", 0.06),
        max_iter=num_boost_round,
        max_leaf_nodes=63,
        min_samples_leaf=40,
        l2_regularization=0.0,
        early_stopping=False,
        random_state=42,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model, "sklearn_cpu"


def fit_classifier(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
    *,
    use_gpu: bool,
    num_boost_round: int = 200,
    params: dict[str, Any] | None = None,
) -> tuple[Any, str]:
    if use_gpu:
        import xgboost as xgb

        p = {**_CLF_DEFAULTS, "device": "cuda", **(params or {})}
        dtrain = xgb.DMatrix(
            _as_float32(X), label=y.astype(np.float32), weight=sample_weight
        )
        booster = xgb.train(p, dtrain, num_boost_round=num_boost_round)
        return booster, "xgboost_cuda"
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=(params or {}).get("learning_rate", 0.08),
        max_iter=num_boost_round,
        max_leaf_nodes=48,
        min_samples_leaf=80,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model, "sklearn_cpu"


def predict(model: Any, backend: str, X: np.ndarray) -> np.ndarray:
    if backend == "xgboost_cuda":
        import xgboost as xgb

        return model.predict(xgb.DMatrix(_as_float32(X)))
    return model.predict(X)


def predict_proba_positive(model: Any, backend: str, X: np.ndarray) -> np.ndarray:
    if backend == "xgboost_cuda":
        import xgboost as xgb

        return model.predict(xgb.DMatrix(_as_float32(X)))
    return model.predict_proba(X)[:, 1]
