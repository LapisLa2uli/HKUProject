"""Step 4: Baseline forecasting model for April demand.

Modeling grain: (warehouse, customer_id, product_id) x day.
No external data used here -- only history from Jan + Feb + Mar.

Pipeline:
    1) Build a dense daily panel for (warehouse, customer, product) tuples that
       have at least ``MIN_NONZERO_DAYS_TRAIN`` non-zero days in Jan+Feb.
    2) Engineer lag/rolling/calendar features (all using only past info).
    3) Train HistGradientBoostingRegressor on (Jan+Feb), evaluate on Mar.
    4) Re-train on (Jan+Feb+Mar), recursively forecast Apr 1-30.
    5) Aggregate forecasts to monthly per (customer, product) and per
       (warehouse, customer, product), and write report cards.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

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
from _metrics import evaluate_arrays
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
    X: pd.DataFrame,
    y: pd.Series,
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


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metrics = evaluate_arrays(y_true, y_pred)
    log(
        f"  [{name}] MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}  "
        f"SMAPE={metrics['smape']:.3f}  bias={metrics['bias']:+.3f}"
    )
    return metrics


def recursive_forecast(
    model: HistGradientBoostingRegressor,
    panel: pd.DataFrame,
    forecast_dates: pd.DatetimeIndex,
    encoders: dict[str, dict[str, int]],
    closure: pd.DataFrame,
) -> pd.DataFrame:
    feat_cols = feature_columns()

    tuple_meta = (
        panel[ID_COLS + ["temperature_zone"]]
        .drop_duplicates(ID_COLS)
        .reset_index(drop=True)
    )

    rolling = panel[ID_COLS + ["date", "qty_ea", "temperature_zone"]].copy()
    out_rows: list[pd.DataFrame] = []

    keep_from = forecast_dates.min() - pd.Timedelta(days=max(LAG_DAYS) + max(ROLL_WINDOWS))
    rolling = rolling[rolling["date"] >= keep_from].copy()

    for d in forecast_dates:
        future = tuple_meta.copy()
        future["date"] = d
        future["qty_ea"] = 0.0
        ext = pd.concat([rolling, future], ignore_index=True)
        ext = add_cny_features(ext, closure)
        ext["qty_ea_masked"] = mask_qty_for_features(ext)
        ext = add_lag_features(ext)
        ext = add_calendar(ext)
        ext = apply_encoders(ext, encoders)
        slice_d = ext[ext["date"] == d].copy()
        X = slice_d[feat_cols].fillna(0.0)
        preds = np.clip(model.predict(X), 0, None)
        slice_d["qty_ea"] = preds
        rolling = pd.concat(
            [rolling, slice_d[ID_COLS + ["date", "qty_ea", "temperature_zone"]]],
            ignore_index=True,
        )
        rolling = rolling[rolling["date"] >= d - pd.Timedelta(days=max(LAG_DAYS) + max(ROLL_WINDOWS))]
        out_rows.append(slice_d[ID_COLS + ["date", "qty_ea"]])

    return pd.concat(out_rows, ignore_index=True)


def main() -> int:
    ensure_report_dirs()
    log("Loading base panel...")
    raw = load_panel_base()

    log("Building dense daily panel (Jan-Mar)...")
    panel = build_dense_panel(raw, HISTORY_END)
    log(f"  panel rows: {len(panel):,}  unique tuples: {panel.groupby(ID_COLS).ngroups:,}")

    log("CNY closure + masked qty for features...")
    closure = detect_warehouse_closure(raw)
    panel = add_cny_features(panel, closure)
    panel["qty_ea_masked"] = mask_qty_for_features(panel)

    log("Adding calendar + lag features...")
    panel_feat = add_lag_features(panel)
    panel_feat = add_calendar(panel_feat)

    encoders = build_encoders(panel_feat)
    panel_feat = apply_encoders(panel_feat, encoders)

    feat_cols = feature_columns()
    train_mask = panel_feat["date"] <= TRAIN_END
    valid_mask = (panel_feat["date"] >= VALID_START) & (panel_feat["date"] <= VALID_END)

    train_df = panel_feat.loc[train_mask].copy()
    valid_df = panel_feat.loc[valid_mask].copy()
    log(f"  train rows: {len(train_df):,}  valid rows: {len(valid_df):,}")

    w_train = make_sample_weights(train_df)
    X_train = train_df[feat_cols].fillna(0.0)
    y_train = train_df["qty_ea"].astype(float)

    valid_open = valid_df[valid_df["is_warehouse_closed"] != 1].copy()
    X_valid = valid_open[feat_cols].fillna(0.0)
    y_valid = valid_open["qty_ea"].astype(float)

    log("Training validation model (Jan+Feb -> Mar)...")
    model_val = fit_model(X_train, y_train, sample_weight=w_train)
    valid_pred = model_val.predict(X_valid)

    log("Validation metrics (open days only):")
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

    per_wh.to_csv(BASELINE_VALIDATION_PER_WAREHOUSE, index=False, encoding="utf-8-sig")
    per_cust.to_csv(BASELINE_VALIDATION_PER_CUSTOMER, index=False, encoding="utf-8-sig")
    monthly.to_csv(BASELINE_VALIDATION_MONTHLY, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [{"metric": f"daily_{k}", "value": v} for k, v in overall.items()]
        + [{"metric": f"monthly_{k}", "value": v} for k, v in monthly_overall.items()]
    ).to_csv(BASELINE_VALIDATION_OVERALL, index=False, encoding="utf-8-sig")

    log("Retraining on full history (Jan+Feb+Mar) for April forecast...")
    full_df = panel_feat.copy()
    w_full = make_sample_weights(full_df)
    X_full = full_df[feat_cols].fillna(0.0)
    y_full = full_df["qty_ea"].astype(float)
    model_full = fit_model(X_full, y_full, sample_weight=w_full)

    log("Recursive forecast for April...")
    forecast_dates = pd.date_range(FORECAST_START, FORECAST_END, freq="D")
    daily_fc = recursive_forecast(model_full, panel, forecast_dates, encoders, closure)

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
