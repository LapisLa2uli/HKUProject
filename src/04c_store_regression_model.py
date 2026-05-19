"""Step 4c: Store-level demand model (third model).

Modeling grain: (warehouse, store, product_id) x day.
Target: daily ``qty_ea`` (same as hurdle).

March validation: Poisson GBDT on the dense panel (lags + calendar + CNY).
April forecast (default): **gap + DOW pattern continuation** (``_pattern_forecast``),
which schedules sparse order days and puts robust order size on those days only.
This avoids Poisson "smearing" (tiny positive qty on every day) for intermittent stores.

Set ``STORE_REG_APRIL_MODE=recursive`` to use the legacy recursive Poisson April path.

Optimizations:
    - Vectorized lag/rolling on the dense panel (2D reshape).
    - Optional XGBoost GPU training when CUDA is available (falls back to sklearn CPU).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from _cny import (
    add_cny_features,
    detect_warehouse_closure,
    make_sample_weights,
    mask_qty_for_features,
)
from _metrics import bias_ratio, evaluate_arrays, wmape
from _pattern_forecast import compute_tuple_order_patterns, pattern_forecast_april
from _report_paths import (
    STORE_REG_APRIL_DAILY,
    STORE_REG_APRIL_MONTHLY_STORE,
    STORE_REG_APRIL_MONTHLY_WCP,
    STORE_REG_VALIDATION_MONTHLY,
    STORE_REG_VALIDATION_OVERALL,
    STORE_REG_VALIDATION_PER_CUSTOMER,
    STORE_REG_VALIDATION_PER_WAREHOUSE,
    ensure_report_dirs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "processed"
MARTS = PROCESSED / "marts"

JAN_START = pd.Timestamp("2026-01-01")
TRAIN_END = pd.Timestamp("2026-02-28")
VALID_START = pd.Timestamp("2026-03-01")
VALID_END = pd.Timestamp("2026-03-28")
HISTORY_END = VALID_END
FORECAST_START = pd.Timestamp("2026-04-01")
FORECAST_END = pd.Timestamp("2026-04-30")

LAG_DAYS = [1, 2, 3, 7, 14, 21, 28]
ROLL_WINDOWS = [7, 14, 28]
HISTORY_SPAN = max(LAG_DAYS) + max(ROLL_WINDOWS)
MIN_NONZERO_DAYS_TRAIN = 3

ID_COLS = ["warehouse", "store", "product_id"]
CATEGORICAL_RAW = ID_COLS + ["customer_id", "temperature_zone"]
CATEGORICAL_ENC = [f"{c}_enc" for c in CATEGORICAL_RAW]

CALENDAR_COLS = ["dow", "dom", "month", "weekofyear", "is_weekend"]
CNY_COLS = ["is_cny_window", "is_warehouse_closed", "days_to_cny", "days_from_cny"]


class RegressorModel(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray: ...


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _april_mode() -> str:
    mode = os.environ.get("STORE_REG_APRIL_MODE", "pattern").strip().lower()
    if mode not in ("pattern", "recursive"):
        log(f"Unknown STORE_REG_APRIL_MODE={mode!r}; using pattern")
        return "pattern"
    return mode


def _want_gpu() -> bool:
    if os.environ.get("STORE_REG_CPU", "").strip().lower() in ("1", "true", "yes"):
        return False
    return os.environ.get("STORE_REG_USE_GPU", "auto").strip().lower() != "0"


def _cuda_xgboost_available() -> bool:
    try:
        import xgboost as xgb

        X = np.random.rand(32, 4).astype(np.float32)
        y = np.abs(np.random.randn(32)).astype(np.float32)
        dm = xgb.DMatrix(X, label=y)
        xgb.train(
            {"tree_method": "hist", "device": "cuda", "objective": "count:poisson"},
            dm,
            num_boost_round=1,
            verbose_eval=False,
        )
        return True
    except Exception:
        return False


def load_panel_base() -> pd.DataFrame:
    df = pd.read_csv(
        MARTS / "mart_warehouse_store_product_day.csv",
        parse_dates=["create_date"],
        dtype={
            "warehouse": str,
            "customer_id": str,
            "store": str,
            "product_id": str,
            "temperature_zone": str,
        },
    )
    df = df.rename(columns={"create_date": "date"})
    df["qty_ea"] = pd.to_numeric(df["qty_ea"], errors="coerce").fillna(0.0)
    return df


def build_dense_panel(df: pd.DataFrame, history_end: pd.Timestamp) -> pd.DataFrame:
    df = df[df["date"] <= history_end].copy()

    nonzero_train = (
        df[(df["date"] <= TRAIN_END) & (df["qty_ea"] > 0)]
        .groupby(ID_COLS)
        .size()
    )
    keep = nonzero_train[nonzero_train >= MIN_NONZERO_DAYS_TRAIN].index
    keep_df = pd.DataFrame(list(keep), columns=ID_COLS)
    df = df.merge(keep_df, on=ID_COLS, how="inner")

    name_map = (
        df.groupby(ID_COLS, dropna=False)
        .agg(
            product_name=("product_name", "first"),
            temperature_zone=("temperature_zone", "first"),
            customer_id=("customer_id", "first"),
        )
        .reset_index()
    )

    history_start = df["date"].min()
    all_dates = pd.date_range(history_start, history_end, freq="D")
    tuples = keep_df.copy()
    tuples["_key"] = 1
    dates_df = pd.DataFrame({"date": all_dates, "_key": 1})
    grid = tuples.merge(dates_df, on="_key").drop(columns="_key")

    panel = grid.merge(
        df[ID_COLS + ["date", "qty_ea"]], on=ID_COLS + ["date"], how="left"
    )
    panel["qty_ea"] = panel["qty_ea"].fillna(0.0)
    panel = panel.merge(name_map, on=ID_COLS, how="left")
    panel = panel.sort_values(ID_COLS + ["date"]).reset_index(drop=True)
    return panel


def is_dense_regular_panel(df: pd.DataFrame) -> bool:
    counts = df.groupby(ID_COLS, observed=True).size()
    return len(counts) > 0 and counts.nunique() == 1


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = df["date"]
    df["dow"] = d.dt.dayofweek.astype(np.int16)
    df["dom"] = d.dt.day.astype(np.int16)
    df["month"] = d.dt.month.astype(np.int16)
    df["weekofyear"] = d.dt.isocalendar().week.astype(np.int16)
    df["is_weekend"] = (df["dow"] >= 5).astype(np.int8)
    return df


def _rolling_nanmean(a: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean along axis=1; uses values strictly before the last column (shift-1)."""
    n_g, n_d = a.shape
    out = np.zeros((n_g, n_d), dtype=np.float64)
    for t in range(n_d):
        lo = max(0, t - window)
        if t == 0:
            continue
        sl = a[:, lo:t]
        with np.errstate(all="ignore"):
            out[:, t] = np.nanmean(sl, axis=1)
    return out


def _rolling_nanstd(a: np.ndarray, window: int) -> np.ndarray:
    n_g, n_d = a.shape
    out = np.zeros((n_g, n_d), dtype=np.float64)
    for t in range(n_d):
        lo = max(0, t - window)
        if t <= lo:
            continue
        sl = a[:, lo:t]
        with np.errstate(all="ignore"):
            out[:, t] = np.nanstd(sl, axis=1)
    return out


def _rolling_nanmax(a: np.ndarray, window: int) -> np.ndarray:
    n_g, n_d = a.shape
    out = np.zeros((n_g, n_d), dtype=np.float64)
    for t in range(n_d):
        lo = max(0, t - window)
        if t == 0:
            continue
        sl = a[:, lo:t]
        with np.errstate(all="ignore"):
            out[:, t] = np.nanmax(sl, axis=1)
    return out


def _rolling_nonzero_ratio(a: np.ndarray, window: int) -> np.ndarray:
    n_g, n_d = a.shape
    pos = (a > 0).astype(np.float64)
    out = np.zeros((n_g, n_d), dtype=np.float64)
    for t in range(n_d):
        lo = max(0, t - window)
        if t == 0:
            continue
        sl = pos[:, lo:t]
        out[:, t] = sl.mean(axis=1)
    return out


def add_lag_features_dense(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized lags/rolls when every tuple has the same daily calendar."""
    dates = pd.DatetimeIndex(df["date"].unique()).sort_values()
    n_d = len(dates)
    n_g = len(df) // n_d
    if n_g * n_d != len(df):
        raise ValueError("panel is not a regular dense grid")

    qty_m = df["qty_ea_masked"].to_numpy(dtype=np.float64).reshape(n_g, n_d)
    qty_a = df["qty_ea"].to_numpy(dtype=np.float64).reshape(n_g, n_d)

    out = df.copy()
    for lag in LAG_DAYS:
        col = np.zeros((n_g, n_d), dtype=np.float64)
        if lag < n_d:
            col[:, lag:] = qty_m[:, :-lag]
        out[f"lag_{lag}"] = col.ravel()

    shifted_m = np.empty_like(qty_m)
    shifted_m[:, 0] = np.nan
    shifted_m[:, 1:] = qty_m[:, :-1]

    for w in ROLL_WINDOWS:
        out[f"rmean_{w}"] = _rolling_nanmean(shifted_m, w).ravel()
        out[f"rstd_{w}"] = _rolling_nanstd(shifted_m, w).ravel()
        out[f"rmax_{w}"] = _rolling_nanmax(shifted_m, w).ravel()
        out[f"nonzero_ratio_{w}"] = _rolling_nonzero_ratio(qty_a, w).ravel()
    return out


def add_lag_features_slow(
    df: pd.DataFrame,
    qty_col: str = "qty_ea_masked",
    qty_actual_col: str = "qty_ea",
) -> pd.DataFrame:
    df = df.sort_values(ID_COLS + ["date"]).copy()
    g_masked = df.groupby(ID_COLS, sort=False, observed=True)[qty_col]
    g_actual = df.groupby(ID_COLS, sort=False, observed=True)[qty_actual_col]
    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = g_masked.shift(lag)
    shifted = g_masked.shift(1)
    shifted_actual = g_actual.shift(1)
    grp = [df[c] for c in ID_COLS]
    for w in ROLL_WINDOWS:
        df[f"rmean_{w}"] = shifted.groupby(grp).transform(
            lambda s: s.rolling(w, min_periods=1).mean()
        )
        df[f"rstd_{w}"] = shifted.groupby(grp).transform(
            lambda s: s.rolling(w, min_periods=1).std()
        )
        df[f"rmax_{w}"] = shifted.groupby(grp).transform(
            lambda s: s.rolling(w, min_periods=1).max()
        )
        df[f"nonzero_ratio_{w}"] = shifted_actual.groupby(grp).transform(
            lambda s: (s > 0).astype(np.float64).rolling(w, min_periods=1).mean()
        )
    return df


def build_encoders(panel: pd.DataFrame) -> dict[str, dict[str, int]]:
    encoders = {}
    for col in CATEGORICAL_RAW:
        vals = panel[col].astype(str).fillna("__nan__").unique().tolist()
        encoders[col] = {v: i for i, v in enumerate(sorted(vals))}
    return encoders


def apply_encoders(df: pd.DataFrame, encoders: dict[str, dict[str, int]]) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL_RAW:
        df[f"{col}_enc"] = (
            df[col].astype(str).fillna("__nan__").map(encoders[col]).fillna(-1).astype(np.int32)
        )
    return df


def feature_columns() -> list[str]:
    cols = list(CATEGORICAL_ENC)
    cols += [f"lag_{l}" for l in LAG_DAYS]
    for w in ROLL_WINDOWS:
        cols += [f"rmean_{w}", f"rstd_{w}", f"rmax_{w}", f"nonzero_ratio_{w}"]
    cols += CALENDAR_COLS + CNY_COLS
    return cols


def fit_model_sklearn(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.06,
        max_iter=600,
        max_leaf_nodes=63,
        min_samples_leaf=40,
        l2_regularization=0.0,
        early_stopping=False,
        random_state=42,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def fit_model_xgb(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> Any:
    import xgboost as xgb

    dtrain = xgb.DMatrix(X, label=y, weight=sample_weight)
    params = {
        "objective": "count:poisson",
        "tree_method": "hist",
        "device": "cuda",
        "learning_rate": 0.06,
        "max_depth": 8,
        "max_leaves": 63,
        "subsample": 1.0,
        "lambda": 0.0,
        "seed": 42,
        "verbosity": 0,
    }
    booster = xgb.train(params, dtrain, num_boost_round=600)
    return booster


def fit_model(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
    *,
    use_gpu: bool,
) -> tuple[Any, str]:
    if use_gpu:
        return fit_model_xgb(X, y, sample_weight), "xgboost_cuda"
    return fit_model_sklearn(X, y, sample_weight), "sklearn_cpu"


def predict_model(model: Any, backend: str, X: np.ndarray) -> np.ndarray:
    if backend == "xgboost_cuda":
        import xgboost as xgb

        return np.clip(model.predict(xgb.DMatrix(X)), 0, None)
    return np.clip(model.predict(X), 0, None)


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metrics = evaluate_arrays(y_true, y_pred)
    log(
        f"  [{name}] MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}  "
        f"SMAPE={metrics['smape']:.3f}  bias={metrics['bias']:+.3f}"
    )
    return metrics


def _lag_roll_from_buffer(masked_buf: np.ndarray, actual_buf: np.ndarray) -> dict[str, np.ndarray]:
    """Features for the next forecast day from trailing buffers (n_tuples, HISTORY_SPAN)."""
    n_g = masked_buf.shape[0]
    feats: dict[str, np.ndarray] = {}
    for lag in LAG_DAYS:
        if lag <= masked_buf.shape[1]:
            feats[f"lag_{lag}"] = masked_buf[:, -lag].astype(np.float64)
        else:
            feats[f"lag_{lag}"] = np.zeros(n_g, dtype=np.float64)

    shifted_m = masked_buf
    shifted_a = actual_buf
    for w in ROLL_WINDOWS:
        sl_m = shifted_m[:, -w:] if w <= shifted_m.shape[1] else shifted_m
        with np.errstate(all="ignore"):
            feats[f"rmean_{w}"] = np.nanmean(sl_m, axis=1)
            feats[f"rstd_{w}"] = np.nanstd(sl_m, axis=1)
            feats[f"rmax_{w}"] = np.nanmax(sl_m, axis=1)
        sl_a = shifted_a[:, -w:] if w <= shifted_a.shape[1] else shifted_a
        feats[f"nonzero_ratio_{w}"] = (sl_a > 0).mean(axis=1)
    return feats


def _recursive_forecast_one_warehouse_fast(
    model: Any,
    backend: str,
    panel_wh: pd.DataFrame,
    forecast_dates: pd.DatetimeIndex,
    encoders: dict[str, dict[str, int]],
    closure: pd.DataFrame,
    feat_cols: list[str],
    warehouse: str,
) -> pd.DataFrame:
    panel_wh = panel_wh.sort_values(ID_COLS + ["date"]).reset_index(drop=True)
    hist_dates = pd.DatetimeIndex(panel_wh["date"].unique()).sort_values()
    n_d = len(hist_dates)
    n_g = len(panel_wh) // n_d
    if n_g * n_d != len(panel_wh):
        raise ValueError(f"warehouse {warehouse}: panel is not a regular grid")

    meta = (
        panel_wh.groupby(ID_COLS, observed=True)
        .agg(customer_id=("customer_id", "first"), temperature_zone=("temperature_zone", "first"))
        .reset_index()
    )

    qty = panel_wh["qty_ea"].to_numpy(dtype=np.float64).reshape(n_g, n_d)
    closed = panel_wh["is_warehouse_closed"].to_numpy(dtype=np.int8).reshape(n_g, n_d)
    masked = qty.copy()
    masked[closed == 1] = np.nan

    keep_hist = hist_dates >= (forecast_dates.min() - pd.Timedelta(days=HISTORY_SPAN))
    idx_keep = np.where(keep_hist)[0]
    masked_buf = masked[:, idx_keep]
    actual_buf = qty[:, idx_keep]
    if masked_buf.shape[1] < HISTORY_SPAN:
        pad = HISTORY_SPAN - masked_buf.shape[1]
        masked_buf = np.pad(masked_buf, ((0, 0), (pad, 0)), constant_values=np.nan)
        actual_buf = np.pad(actual_buf, ((0, 0), (pad, 0)), constant_values=0.0)
    elif masked_buf.shape[1] > HISTORY_SPAN:
        masked_buf = masked_buf[:, -HISTORY_SPAN:]
        actual_buf = actual_buf[:, -HISTORY_SPAN:]

    static_enc = np.column_stack(
        [
            meta[col].astype(str).fillna("__nan__").map(encoders[col]).fillna(-1).astype(np.float64).values
            for col in CATEGORICAL_RAW
        ]
    )

    wh_closure = closure[closure["warehouse"] == warehouse].copy()
    wh_closure["date"] = pd.to_datetime(wh_closure["date"]).dt.normalize()

    pred_rows: list[np.ndarray] = []
    date_list: list[pd.Timestamp] = []

    for d in forecast_dates:
        cal_row = add_calendar(pd.DataFrame({"date": [d]}))
        cny_row = add_cny_features(
            cal_row.assign(warehouse=warehouse, qty_ea=0.0), wh_closure
        )
        lag_roll = _lag_roll_from_buffer(masked_buf, actual_buf)
        cal_block = np.tile(
            cal_row[CALENDAR_COLS].to_numpy(dtype=np.float64), (n_g, 1)
        )
        cny_block = np.tile(
            cny_row[CNY_COLS].to_numpy(dtype=np.float64), (n_g, 1)
        )
        X = np.column_stack(
            [
                static_enc,
                *[lag_roll[f"lag_{lag}"] for lag in LAG_DAYS],
                *[lag_roll[f"rmean_{w}"] for w in ROLL_WINDOWS],
                *[lag_roll[f"rstd_{w}"] for w in ROLL_WINDOWS],
                *[lag_roll[f"rmax_{w}"] for w in ROLL_WINDOWS],
                *[lag_roll[f"nonzero_ratio_{w}"] for w in ROLL_WINDOWS],
                cal_block,
                cny_block,
            ]
        )
        X = np.nan_to_num(X, nan=0.0)
        preds = predict_model(model, backend, X)

        is_closed = int(cny_row["is_warehouse_closed"].iloc[0]) == 1
        if is_closed:
            preds = np.zeros_like(preds)
        masked_next_arr = np.where(is_closed, np.nan, preds.astype(np.float64))
        actual_next_arr = preds.astype(np.float64)

        pred_rows.append(preds.astype(np.float64))
        date_list.append(d)

        masked_buf = np.hstack([masked_buf[:, 1:], masked_next_arr.reshape(-1, 1)])
        actual_buf = np.hstack([actual_buf[:, 1:], actual_next_arr.reshape(-1, 1)])

    pred_mat = np.column_stack(pred_rows)
    out = meta.copy()
    out["customer_id"] = meta["customer_id"].values
    long = out.loc[out.index.repeat(len(forecast_dates))].reset_index(drop=True)
    long["date"] = np.tile(np.array(forecast_dates, dtype="datetime64[ns]"), n_g)
    long["qty_ea"] = pred_mat.ravel()
    return long[ID_COLS + ["customer_id", "date", "qty_ea"]]


def recursive_forecast(
    model: Any,
    backend: str,
    panel: pd.DataFrame,
    forecast_dates: pd.DatetimeIndex,
    encoders: dict[str, dict[str, int]],
    closure: pd.DataFrame,
) -> pd.DataFrame:
    feat_cols = feature_columns()
    warehouses = sorted(panel["warehouse"].astype(str).unique())
    parts: list[pd.DataFrame] = []
    t0 = time.perf_counter()
    for wi, wh in enumerate(warehouses):
        panel_wh = panel[panel["warehouse"] == wh]
        parts.append(
            _recursive_forecast_one_warehouse_fast(
                model, backend, panel_wh, forecast_dates, encoders, closure, feat_cols, wh
            )
        )
        if wi == 0 or (wi + 1) % 5 == 0 or wi == len(warehouses) - 1:
            elapsed = time.perf_counter() - t0
            log(
                f"  recursive April {wi + 1}/{len(warehouses)} warehouses "
                f"({elapsed:.1f}s elapsed)"
            )
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    ensure_report_dirs()
    t_start = time.perf_counter()
    use_gpu = _want_gpu() and _cuda_xgboost_available()
    if use_gpu:
        log("Training backend: XGBoost CUDA")
    else:
        log("Training backend: sklearn HistGradientBoosting (CPU)")

    log("Loading store-level mart...")
    raw = load_panel_base()

    log("Building dense daily panel (Jan-Mar)...")
    panel = build_dense_panel(raw, HISTORY_END)
    log(f"  panel rows: {len(panel):,}  unique tuples: {panel.groupby(ID_COLS).ngroups:,}")

    log("CNY closure + masked qty for features...")
    closure = detect_warehouse_closure(raw)
    panel = add_cny_features(panel, closure)
    panel["qty_ea_masked"] = mask_qty_for_features(panel)

    log("Adding calendar + lag features...")
    t_feat = time.perf_counter()
    panel = add_calendar(panel)
    if is_dense_regular_panel(panel):
        panel_feat = add_lag_features_dense(panel)
        log(f"  used vectorized dense lag/roll ({time.perf_counter() - t_feat:.1f}s)")
    else:
        panel_feat = add_lag_features_slow(panel)
        panel_feat = add_calendar(panel_feat)
        log(f"  used pandas lag/roll ({time.perf_counter() - t_feat:.1f}s)")

    encoders = build_encoders(panel_feat)
    panel_feat = apply_encoders(panel_feat, encoders)

    feat_cols = feature_columns()
    train_mask = panel_feat["date"] <= TRAIN_END
    valid_mask = (panel_feat["date"] >= VALID_START) & (panel_feat["date"] <= VALID_END)

    train_df = panel_feat.loc[train_mask]
    valid_df = panel_feat.loc[valid_mask]
    log(f"  train rows: {len(train_df):,}  valid rows: {len(valid_df):,}")

    w_train = make_sample_weights(train_df)
    X_train = np.nan_to_num(train_df[feat_cols].to_numpy(dtype=np.float64), nan=0.0)
    y_train = train_df["qty_ea"].to_numpy(dtype=np.float64)

    valid_open = valid_df[valid_df["is_warehouse_closed"] != 1]
    X_valid = np.nan_to_num(valid_open[feat_cols].to_numpy(dtype=np.float64), nan=0.0)
    y_valid = valid_open["qty_ea"].to_numpy(dtype=np.float64)

    log("Training validation model (Jan+Feb -> Mar)...")
    t_train = time.perf_counter()
    model_val, backend = fit_model(X_train, y_train, w_train, use_gpu=use_gpu)
    log(f"  train time: {time.perf_counter() - t_train:.1f}s")
    valid_pred = predict_model(model_val, backend, X_valid)

    log("Validation metrics (open days only):")
    overall = evaluate("overall_daily", y_valid, valid_pred)

    diag = valid_open[ID_COLS + ["customer_id", "date", "qty_ea"]].copy()
    diag["pred"] = np.clip(valid_pred, 0, None)

    per_wh = (
        diag.groupby("warehouse", observed=True)[["qty_ea", "pred"]]
        .apply(lambda g: pd.Series(evaluate_arrays(g["qty_ea"].values, g["pred"].values)))
        .reset_index()
    )
    per_cust = (
        diag.groupby("customer_id", observed=True)[["qty_ea", "pred"]]
        .apply(lambda g: pd.Series(evaluate_arrays(g["qty_ea"].values, g["pred"].values)))
        .reset_index()
    )

    monthly = (
        diag.groupby(ID_COLS + ["customer_id"])
        .agg(actual=("qty_ea", "sum"), pred=("pred", "sum"))
        .reset_index()
    )
    monthly_overall = evaluate(
        "monthly (wh,store,prod)", monthly["actual"].values, monthly["pred"].values
    )
    wmape_m = wmape(monthly["actual"].values, monthly["pred"].values)
    bias_m = bias_ratio(monthly["actual"].values, monthly["pred"].values)
    log(f"  monthly WMAPE={wmape_m:.4f}  bias_ratio={bias_m:+.4f}")

    per_wh.to_csv(STORE_REG_VALIDATION_PER_WAREHOUSE, index=False, encoding="utf-8-sig")
    per_cust.to_csv(STORE_REG_VALIDATION_PER_CUSTOMER, index=False, encoding="utf-8-sig")
    monthly.to_csv(STORE_REG_VALIDATION_MONTHLY, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [{"metric": f"daily_{k}", "value": v} for k, v in overall.items()]
        + [{"metric": f"monthly_{k}", "value": v} for k, v in monthly_overall.items()]
        + [{"metric": "monthly_wmape", "value": wmape_m}, {"metric": "monthly_bias_ratio", "value": bias_m}]
        + [{"metric": "training_backend_xgboost_cuda", "value": float(use_gpu)}]
    ).to_csv(STORE_REG_VALIDATION_OVERALL, index=False, encoding="utf-8-sig")

    forecast_dates = pd.date_range(FORECAST_START, FORECAST_END, freq="D")
    april_mode = _april_mode()
    t_rec = time.perf_counter()
    if april_mode == "recursive":
        log("Retraining on full history (Jan+Feb+Mar) for recursive April...")
        full_df = panel_feat
        w_full = make_sample_weights(full_df)
        X_full = np.nan_to_num(full_df[feat_cols].to_numpy(dtype=np.float64), nan=0.0)
        y_full = full_df["qty_ea"].to_numpy(dtype=np.float64)
        t_train = time.perf_counter()
        model_full, backend = fit_model(X_full, y_full, w_full, use_gpu=use_gpu)
        log(f"  train time: {time.perf_counter() - t_train:.1f}s")
        log("April forecast: recursive Poisson (STORE_REG_APRIL_MODE=recursive)...")
        daily_fc = recursive_forecast(
            model_full, backend, panel, forecast_dates, encoders, closure
        )
    else:
        log("April forecast: gap + DOW pattern (intermittent order days)...")
        patterns = compute_tuple_order_patterns(
            panel,
            history_end=HISTORY_END,
            jan_start=JAN_START,
        )
        daily_fc = pattern_forecast_april(
            patterns,
            forecast_dates,
            closure,
            forecast_start=FORECAST_START,
            forecast_end=FORECAST_END,
        )
        log(
            f"  pattern tuples={len(patterns):,}  "
            f"mean gap={patterns['mean_gap_days'].mean():.2f}d  "
            f"mean pos_rate={patterns['pos_rate'].mean():.3f}"
        )
    log(f"  April forecast total: {time.perf_counter() - t_rec:.1f}s")

    name_lookup = panel[ID_COLS + ["product_name", "temperature_zone"]].drop_duplicates(ID_COLS)
    daily_fc = daily_fc.merge(name_lookup, on=ID_COLS, how="left")
    daily_fc.to_csv(STORE_REG_APRIL_DAILY, index=False, encoding="utf-8-sig")

    monthly_store = (
        daily_fc.groupby(ID_COLS + ["customer_id", "product_name", "temperature_zone"])["qty_ea"]
        .sum()
        .reset_index()
        .rename(columns={"qty_ea": "predicted_qty_ea_april"})
    )
    monthly_store.to_csv(STORE_REG_APRIL_MONTHLY_STORE, index=False, encoding="utf-8-sig")

    monthly_wcp = (
        daily_fc.groupby(["warehouse", "customer_id", "product_id", "product_name", "temperature_zone"])[
            "qty_ea"
        ]
        .sum()
        .reset_index()
        .rename(columns={"qty_ea": "predicted_qty_ea_april"})
    )
    monthly_wcp.to_csv(STORE_REG_APRIL_MONTHLY_WCP, index=False, encoding="utf-8-sig")

    log("Done.")
    log(f"April daily forecast rows: {len(daily_fc):,}")
    log(f"April total predicted qty_ea: {daily_fc['qty_ea'].sum():,.1f}")
    log(f"Total runtime: {(time.perf_counter() - t_start) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
