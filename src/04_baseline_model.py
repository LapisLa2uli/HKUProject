"""Step 4: Baseline forecasting model for April demand.

Modeling grain: (warehouse, customer_id, product_id) x day.

Sliding protocol (see ``_sliding_window``):
    - **Lookback:** last 14 warehouse-**open** days (CNY closure days skipped).
    - **Horizon:** predict the next **7** calendar days; slide forward 7 days.
    - **Train / validate:** samples whose 7-day horizon lies in Jan–Feb (train) or Mar (validate).
    - **April:** same weekly sliding; after each week, predictions feed the next lookback.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from _cny import add_cny_features, detect_warehouse_closure
from _gbdt_gpu import fit_regressor, predict, want_gpu
from _metrics import evaluate_arrays, wmape, bias_ratio
from _sliding_window import (
    HORIZON_DAYS,
    LOOKBACK_OPEN_DAYS,
    SLIDE_STEP_DAYS,
    SLIDE_STEP_TRAIN_DAYS,
    anchors_with_horizon_in,
    build_sliding_samples,
    sample_weights_sliding,
    sliding_feature_columns,
    sliding_forecast_period,
)
from _report_paths import (
    BASELINE_APRIL_DAILY,
    BASELINE_APRIL_MONTHLY_CP,
    BASELINE_APRIL_MONTHLY_WCP,
    BASELINE_APRIL_MONTHLY_WH,
    BASELINE_VALIDATION_MONTHLY,
    BASELINE_VALIDATION_OVERALL,
    BASELINE_VALIDATION_PER_CUSTOMER,
    BASELINE_VALIDATION_PER_WAREHOUSE,
    ensure_report_dirs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "processed"
MARTS = PROCESSED / "marts"
MODELS = PROJECT_ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

JAN_START = pd.Timestamp("2026-01-01")
TRAIN_END = pd.Timestamp("2026-02-28")
VALID_START = pd.Timestamp("2026-03-01")
VALID_END = pd.Timestamp("2026-03-28")
HISTORY_END = VALID_END
FORECAST_START = pd.Timestamp("2026-04-01")
FORECAST_END = pd.Timestamp("2026-04-30")

LAG_DAYS = [1, 2, 3, 7, 14, 21, 28]
ROLL_WINDOWS = [7, 14, 28]
MIN_NONZERO_DAYS_TRAIN = 3

ID_COLS = ["warehouse", "customer_id", "product_id"]
CATEGORICAL_RAW = ID_COLS + ["temperature_zone"]
CATEGORICAL_ENC = [f"{c}_enc" for c in CATEGORICAL_RAW]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_panel_base() -> pd.DataFrame:
    df = pd.read_csv(
        MARTS / "mart_warehouse_customer_product_day.csv",
        parse_dates=["create_date"],
        dtype={
            "warehouse": str,
            "customer_id": str,
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
        )
        .reset_index()
    )

    df = (
        df.groupby(ID_COLS + ["date"], as_index=False, observed=True)["qty_ea"]
        .sum()
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


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = df["date"]
    df["dow"] = d.dt.dayofweek.astype(np.int16)
    df["dom"] = d.dt.day.astype(np.int16)
    df["month"] = d.dt.month.astype(np.int16)
    df["weekofyear"] = d.dt.isocalendar().week.astype(np.int16)
    df["is_weekend"] = (df["dow"] >= 5).astype(np.int8)
    return df


def add_lag_features(
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
    grp = [df["warehouse"], df["customer_id"], df["product_id"]]
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
    cols += [
        "dow",
        "dom",
        "month",
        "weekofyear",
        "is_weekend",
        "is_cny_window",
        "is_warehouse_closed",
        "days_to_cny",
        "days_from_cny",
    ]
    return cols


def fit_model(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    sample_weight: np.ndarray | None = None,
    *,
    use_gpu: bool | None = None,
) -> tuple[object, str]:
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy(dtype=np.float64)
    if isinstance(y, pd.Series):
        y = y.to_numpy(dtype=np.float64)
    if use_gpu is None:
        use_gpu = want_gpu("BASELINE")
    return fit_regressor(X, y, sample_weight, use_gpu=use_gpu, num_boost_round=600)


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metrics = evaluate_arrays(y_true, y_pred)
    log(
        f"  [{name}] MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}  "
        f"SMAPE={metrics['smape']:.3f}  bias={metrics['bias']:+.3f}"
    )
    return metrics


def main() -> int:
    ensure_report_dirs()
    use_gpu = want_gpu("BASELINE")
    log(f"Training backend: {'XGBoost CUDA' if use_gpu else 'sklearn HistGradientBoosting (CPU)'}")
    log("Loading base panel...")
    raw = load_panel_base()

    log("Building dense daily panel (Jan-Mar)...")
    panel = build_dense_panel(raw, HISTORY_END)
    log(f"  panel rows: {len(panel):,}  unique tuples: {panel.groupby(ID_COLS).ngroups:,}")

    log("CNY closure detection...")
    closure = detect_warehouse_closure(raw)
    panel = add_cny_features(panel, closure)

    encoders = build_encoders(panel)
    static_enc = apply_encoders(
        panel[ID_COLS + ["temperature_zone"]].drop_duplicates(ID_COLS),
        encoders,
    )
    feat_cols = sliding_feature_columns(CATEGORICAL_ENC)

    train_anchors = anchors_with_horizon_in(
        JAN_START, TRAIN_END, step_days=SLIDE_STEP_TRAIN_DAYS
    )
    valid_anchors = anchors_with_horizon_in(
        VALID_START, VALID_END, step_days=SLIDE_STEP_TRAIN_DAYS
    )
    log(
        f"Sliding windows: lookback={LOOKBACK_OPEN_DAYS} open days, "
        f"horizon={HORIZON_DAYS}d, train_step={SLIDE_STEP_TRAIN_DAYS}d, "
        f"april_step={SLIDE_STEP_DAYS}d"
    )
    log(f"  train anchors: {len(train_anchors)}  valid anchors: {len(valid_anchors)}")

    log("Building sliding training samples (Jan-Feb horizons)...")
    train_df = build_sliding_samples(
        panel, ID_COLS, CATEGORICAL_ENC, static_enc, train_anchors, closure
    )
    log(f"  train sample rows: {len(train_df):,}")

    w_train = sample_weights_sliding(train_df)
    X_train = train_df[feat_cols].fillna(0.0)
    y_train = train_df["qty_ea"].astype(float)

    log("Building sliding validation samples (March horizons)...")
    valid_df = build_sliding_samples(
        panel, ID_COLS, CATEGORICAL_ENC, static_enc, valid_anchors, closure
    )
    valid_open = valid_df[valid_df["is_warehouse_closed"] != 1].copy()
    X_valid = valid_open[feat_cols].fillna(0.0)
    y_valid = valid_open["qty_ea"].astype(float)

    log("Training validation model (sliding Jan-Feb -> Mar)...")
    model_val, backend_val = fit_model(
        X_train.to_numpy(dtype=np.float64), y_train.to_numpy(), sample_weight=w_train, use_gpu=use_gpu
    )
    valid_pred = predict(model_val, backend_val, X_valid.to_numpy(dtype=np.float64))

    log("Validation metrics (open March horizon days):")
    overall = evaluate("overall_daily", y_valid.values, valid_pred)

    diag = valid_open[ID_COLS + ["date", "qty_ea"]].copy()
    diag["pred"] = np.clip(valid_pred, 0, None)

    log("Per-warehouse validation metrics:")
    per_wh = (
        diag.groupby("warehouse")
        .apply(lambda g: pd.Series(evaluate(f"wh={g.name}", g["qty_ea"].values, g["pred"].values)))
        .reset_index()
    )
    log("Per-customer validation metrics:")
    per_cust = (
        diag.groupby("customer_id")
        .apply(lambda g: pd.Series(evaluate(f"cust={g.name}", g["qty_ea"].values, g["pred"].values)))
        .reset_index()
    )

    monthly = (
        diag.groupby(ID_COLS)
        .agg(actual=("qty_ea", "sum"), pred=("pred", "sum"))
        .reset_index()
    )
    monthly_overall = evaluate("monthly (wh,cust,prod)", monthly["actual"].values, monthly["pred"].values)
    wmape_m = wmape(monthly["actual"].values, monthly["pred"].values)
    bias_m = bias_ratio(monthly["actual"].values, monthly["pred"].values)
    log(f"  monthly WMAPE={wmape_m:.4f}  bias_ratio={bias_m:+.4f}")

    per_wh.to_csv(BASELINE_VALIDATION_PER_WAREHOUSE, index=False, encoding="utf-8-sig")
    per_cust.to_csv(BASELINE_VALIDATION_PER_CUSTOMER, index=False, encoding="utf-8-sig")
    monthly.to_csv(BASELINE_VALIDATION_MONTHLY, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [{"metric": f"daily_{k}", "value": v} for k, v in overall.items()]
        + [{"metric": f"monthly_{k}", "value": v} for k, v in monthly_overall.items()]
        + [{"metric": "monthly_wmape", "value": wmape_m}, {"metric": "monthly_bias_ratio", "value": bias_m}]
        + [{"metric": "sliding_lookback_open_days", "value": float(LOOKBACK_OPEN_DAYS)}]
        + [{"metric": "sliding_horizon_days", "value": float(HORIZON_DAYS)}]
        + [{"metric": "sliding_step_train_days", "value": float(SLIDE_STEP_TRAIN_DAYS)}]
        + [{"metric": "sliding_step_april_days", "value": float(SLIDE_STEP_DAYS)}]
        + [{"metric": "training_backend_xgboost_cuda", "value": float(use_gpu)}]
    ).to_csv(BASELINE_VALIDATION_OVERALL, index=False, encoding="utf-8-sig")

    log("Retraining on sliding samples through March (Jan-Mar horizons)...")
    full_anchors = anchors_with_horizon_in(
        JAN_START, HISTORY_END, step_days=SLIDE_STEP_TRAIN_DAYS
    )
    full_df = build_sliding_samples(
        panel, ID_COLS, CATEGORICAL_ENC, static_enc, full_anchors, closure
    )
    w_full = sample_weights_sliding(full_df)
    model_full, backend_full = fit_model(
        full_df[feat_cols].fillna(0.0).to_numpy(dtype=np.float64),
        full_df["qty_ea"].to_numpy(dtype=np.float64),
        sample_weight=w_full,
        use_gpu=use_gpu,
    )

    log("Sliding April forecast (weekly blocks, predictions extend lookback)...")
    april_anchors = anchors_with_horizon_in(FORECAST_START, FORECAST_END)

    def _predict_batch(batch: pd.DataFrame) -> np.ndarray:
        Xb = batch[feat_cols].fillna(0.0).to_numpy(dtype=np.float64)
        return predict(model_full, backend_full, Xb)

    daily_fc = sliding_forecast_period(
        panel,
        ID_COLS,
        CATEGORICAL_ENC,
        static_enc,
        april_anchors,
        _predict_batch,
        closure,
    )

    name_lookup = panel[ID_COLS + ["product_name", "temperature_zone"]].drop_duplicates(ID_COLS)
    daily_fc = daily_fc.merge(name_lookup, on=ID_COLS, how="left")
    daily_fc.to_csv(BASELINE_APRIL_DAILY, index=False, encoding="utf-8-sig")

    monthly_wcp = (
        daily_fc.groupby(ID_COLS + ["product_name", "temperature_zone"])  
        ["qty_ea"].sum().reset_index().rename(columns={"qty_ea": "predicted_qty_ea_april"})
    )
    monthly_wcp.to_csv(BASELINE_APRIL_MONTHLY_WCP, index=False, encoding="utf-8-sig")

    monthly_cp = (
        daily_fc.groupby(["customer_id", "product_id", "product_name", "temperature_zone"])
        ["qty_ea"].sum().reset_index().rename(columns={"qty_ea": "predicted_qty_ea_april"})
    )
    monthly_cp.to_csv(BASELINE_APRIL_MONTHLY_CP, index=False, encoding="utf-8-sig")

    monthly_wh = (
        daily_fc.groupby("warehouse")["qty_ea"]
        .sum()
        .reset_index()
        .rename(columns={"qty_ea": "predicted_qty_ea_april"})
        .sort_values("predicted_qty_ea_april", ascending=False)
    )
    monthly_wh.to_csv(BASELINE_APRIL_MONTHLY_WH, index=False, encoding="utf-8-sig")

    log("Done.")
    log(f"April daily forecast rows: {len(daily_fc):,}")
    log(f"April total predicted qty_ea: {daily_fc['qty_ea'].sum():,.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
