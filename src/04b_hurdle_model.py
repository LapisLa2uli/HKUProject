"""Step 4b: Store-grain hurdle model with sliding-window train/validate/forecast.

Grain: (warehouse, store, product_id) x day.

- **Lookback:** 14 warehouse-open days (CNY closure skipped).
- **Horizon:** next 7 calendar days; slide forward 7 days.
- **Model:** classifier P(order) + Poisson size μ; pred = μ if P(order) >= 0.5 else 0.
- **April:** same weekly sliding forecast (predictions extend the lookback).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from _cny import (
    add_cny_features,
    detect_warehouse_closure,
    make_sample_weights,
    mask_qty_for_features,
)
from _gbdt_gpu import (
    fit_classifier,
    fit_regressor,
    predict as gbdt_predict,
    predict_proba_positive,
    want_gpu,
)
from _metrics import (
    bias_ratio,
    evaluate_arrays,
    evaluate_arrays_positive_actual,
    intermittent_hurdle_loss_report,
    smape_fracs,
    store_planning_loss_report,
    wmape,
)
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
    HURDLE_APRIL_CALIBRATION,
    HURDLE_APRIL_DAILY_NETWORK,
    HURDLE_APRIL_DAILY_PATTERN,
    HURDLE_APRIL_DAILY_STORE,
    HURDLE_APRIL_MONTHLY_CP,
    HURDLE_APRIL_MONTHLY_STORE,
    HURDLE_APRIL_MONTHLY_WCP,
    HURDLE_APRIL_MONTHLY_WH,
    HURDLE_APRIL_STORE_CALIBRATION,
    HURDLE_VALIDATION_CLASSIFIER,
    HURDLE_VALIDATION_INTERMITTENT,
    HURDLE_VALIDATION_PLANNING,
    HURDLE_VALIDATION_MONTHLY,
    HURDLE_VALIDATION_OVERALL,
    HURDLE_VALIDATION_PER_CUSTOMER,
    HURDLE_VALIDATION_PER_WAREHOUSE,
    ensure_report_dirs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "processed"
MARTS = PROCESSED / "marts"

TRAIN_END = pd.Timestamp("2026-02-28")
VALID_START = pd.Timestamp("2026-03-01")
VALID_END = pd.Timestamp("2026-03-28")
HISTORY_END = VALID_END
FORECAST_START = pd.Timestamp("2026-04-01")
FORECAST_END = pd.Timestamp("2026-04-30")
JAN_START = pd.Timestamp("2026-01-01")
JAN_END = pd.Timestamp("2026-01-31")

LAG_DAYS = [1, 2, 3, 4, 5, 6, 7, 10, 14]
ROLL_WINDOWS = [7, 14]
# Match ``04_baseline_model`` so hurdle covers the same demand universe (store grain).
MIN_NONZERO_DAYS_TRAIN = 3

ID_COLS = ["warehouse", "store", "product_id"]
STORE_KEYS = ["warehouse", "customer_id", "store"]
CATEGORICAL_RAW = ID_COLS + ["customer_id", "temperature_zone"]
CATEGORICAL_ENC = [f"{c}_enc" for c in CATEGORICAL_RAW]

ORDER_PROB_THRESHOLD = 0.5
PLANNING_MATCH_WINDOW_DAYS = 2
PLANNING_QTY_REL_TOL = 0.15

# P(order): blend classifier proba with empirical prior (tuned on March validation).
P_ORDER_BLEND_LAMBDA = 0.10
P_ORDER_PRIOR_K = 0.35
P_ORDER_MIN = 8e-5
ROLL_QTY_MIN_FRAC_SEED = 0.35  # rolling buffer qty >= max(pred, frac * seed_qty)
# When tuning prior blend: tiny weight on decision-rule SMAPE (see tune_prior_hyperparams).
PRIOR_TUNE_DECISION_SMAPE_WEIGHT = 0.04

# April network-level calibration toward (Q_Jan + Q_Mar) / 2 on dense panel.
APRIL_SCALE_MIN = 0.5
APRIL_SCALE_MAX = 3.0

# Pattern forecast: learn cadence from recent history only (no predict-from-yesterday recursion).
PATTERN_LOOKBACK_DAYS = 28
PATTERN_MIN_GAP_DAYS = 1
PATTERN_MAX_GAP_DAYS = 28
PATTERN_DOW_MIN_FRAC = 0.35  # min dow_pos_rate / pos_rate to keep a gap-scheduled order day
# Blend short lookback with Jan–Mar dense-panel tuple stats (pos rate + order size).
PATTERN_LONG_HISTORY_BLEND = 0.35
PATTERN_QTY_MEDIAN_WEIGHT = 0.55  # weight on median vs mean among short-window positive days
PATTERN_QTY_LONG_SHRINK = 0.30  # pull robust short estimate toward Jan–Mar median positive qty
# Per-store April scale toward each store's (Jan mean daily total + Mar mean daily total) / 2.
STORE_CALIBRATION_CLIP_LO = 0.15
STORE_CALIBRATION_CLIP_HI = 3.5

# April daily exports (visualization uses pattern + store-cal, not network-scaled).
HURDLE_DAILY_PATTERN_CSV = HURDLE_APRIL_DAILY_PATTERN.name
HURDLE_DAILY_STORE_CSV = HURDLE_APRIL_DAILY_STORE.name
HURDLE_DAILY_NETWORK_CSV = HURDLE_APRIL_DAILY_NETWORK.name


def write_hurdle_daily_csv(
    daily_fc: pd.DataFrame,
    path: Path,
    name_lookup: pd.DataFrame,
) -> None:
    extra = [
        c
        for c in ("customer_id", "product_name", "temperature_zone")
        if c in daily_fc.columns
    ]
    base = daily_fc.drop(columns=extra, errors="ignore")
    out = base.merge(name_lookup, on=ID_COLS, how="left")
    try:
        out.to_csv(path, index=False, encoding="utf-8-sig")
    except PermissionError:
        log(f"WARNING: could not write {path.name} (file locked?); skipping")


def tuple_recursive_priors(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-tuple Jan–Feb order-day rate and recent mean daily qty (lag stabilization)."""
    pt_train = panel[panel["date"] <= TRAIN_END]
    pos = pt_train.groupby(ID_COLS, as_index=False).agg(
        pos_rate_train=("qty_ea", lambda s: float((s > 0).mean())),
    )
    seed_start = HISTORY_END - pd.Timedelta(days=13)
    seed = (
        panel[(panel["date"] >= seed_start) & (panel["date"] <= HISTORY_END)]
        .groupby(ID_COLS, as_index=False)["qty_ea"]
        .mean()
        .rename(columns={"qty_ea": "seed_qty"})
    )
    out = pos.merge(seed, on=ID_COLS, how="outer")
    out["pos_rate_train"] = out["pos_rate_train"].fillna(0.03).clip(1e-6, 1.0)
    out["seed_qty"] = out["seed_qty"].fillna(0.0)
    return out


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# #region agent log
def _agent_debug_log(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict,
    run_id: str = "pre-fix",
) -> None:
    import json

    path = PROJECT_ROOT / "debug-164b5f.log"
    payload = {
        "sessionId": "164b5f",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
        "runId": run_id,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")


# #endregion


def mean_network_daily_total(df: pd.DataFrame, qty_col: str = "qty_ea") -> float:
    """Mean over calendar days of sum(qty) across all panel rows."""
    if df.empty:
        return 0.0
    return float(df.groupby("date")[qty_col].sum().mean())


def compute_q_targets(panel: pd.DataFrame) -> dict[str, float]:
    """Q_Jan, Q_Mar, Q_target = (Q_Jan + Q_Mar) / 2 on dense panel actuals."""
    jan = panel[(panel["date"] >= JAN_START) & (panel["date"] <= JAN_END)]
    mar = panel[(panel["date"] >= VALID_START) & (panel["date"] <= VALID_END)]
    q_jan = mean_network_daily_total(jan)
    q_mar = mean_network_daily_total(mar)
    q_target = 0.5 * (q_jan + q_mar)
    return {"Q_Jan": q_jan, "Q_Mar": q_mar, "Q_target": q_target}


def compute_store_daily_targets(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-store mean daily total qty (sum SKUs); target = average of Jan and Mar means."""
    daily = (
        panel.groupby(STORE_KEYS + ["date"], as_index=False)["qty_ea"]
        .sum()
        .rename(columns={"qty_ea": "day_total"})
    )
    jan = daily[(daily["date"] >= JAN_START) & (daily["date"] <= JAN_END)]
    mar = daily[(daily["date"] >= VALID_START) & (daily["date"] <= VALID_END)]
    jan_m = jan.groupby(STORE_KEYS, as_index=False)["day_total"].mean().rename(
        columns={"day_total": "jan_mean_daily"}
    )
    mar_m = mar.groupby(STORE_KEYS, as_index=False)["day_total"].mean().rename(
        columns={"day_total": "mar_mean_daily"}
    )
    out = jan_m.merge(mar_m, on=STORE_KEYS, how="outer")
    out["jan_mean_daily"] = out["jan_mean_daily"].fillna(0.0)
    out["mar_mean_daily"] = out["mar_mean_daily"].fillna(0.0)
    out["target_daily_mean"] = 0.5 * (out["jan_mean_daily"] + out["mar_mean_daily"])
    return out


def calibrate_april_per_store(
    daily_fc: pd.DataFrame,
    store_targets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scale each store's April rows so mean daily store total matches historic target (clipped)."""
    apr_daily = (
        daily_fc.groupby(STORE_KEYS + ["date"], as_index=False)["qty_ea"]
        .sum()
        .rename(columns={"qty_ea": "apr_day_total"})
    )
    raw_mean = apr_daily.groupby(STORE_KEYS, as_index=False)["apr_day_total"].mean().rename(
        columns={"apr_day_total": "apr_mean_daily_raw"}
    )
    merged = raw_mean.merge(store_targets, on=STORE_KEYS, how="left")
    merged["target_daily_mean"] = merged["target_daily_mean"].fillna(merged["apr_mean_daily_raw"])
    merged["jan_mean_daily"] = merged["jan_mean_daily"].fillna(0.0)
    merged["mar_mean_daily"] = merged["mar_mean_daily"].fillna(0.0)

    t = merged["target_daily_mean"].to_numpy(dtype=np.float64)
    r = merged["apr_mean_daily_raw"].to_numpy(dtype=np.float64)
    scale_raw = np.ones(len(merged), dtype=np.float64)
    mask_pos = (t > 1e-9) | (r > 1e-9)
    scale_raw[mask_pos] = np.where(
        t[mask_pos] > 1e-9,
        t[mask_pos] / np.maximum(r[mask_pos], 1e-9),
        0.0,
    )
    tiny_both = (~mask_pos) | ((t <= 1e-9) & (r <= 1e-9))
    scale_raw[tiny_both] = 1.0

    merged["store_scale_raw"] = scale_raw
    merged["store_scale"] = np.clip(scale_raw, STORE_CALIBRATION_CLIP_LO, STORE_CALIBRATION_CLIP_HI)
    merged["store_scale_clipped"] = merged["store_scale"] != merged["store_scale_raw"]

    scale_map = merged.set_index(STORE_KEYS)["store_scale"]
    idx = pd.MultiIndex.from_frame(daily_fc[STORE_KEYS])
    mult = idx.map(scale_map)
    if mult.isna().any():
        mult = mult.fillna(1.0)
    mult_arr = mult.astype(np.float64).values

    out = daily_fc.copy()
    out["qty_ea"] = out["qty_ea"].astype(np.float64) * mult_arr
    out["mu_qty"] = out["mu_qty"].astype(np.float64) * mult_arr
    mu = np.maximum(out["mu_qty"].astype(np.float64).values, 1e-9)
    out["p_order"] = np.clip(out["qty_ea"].values / mu, 0.0, 1.0)
    return out, merged


def _mean_interorder_gap(hit_dates: pd.Series) -> float:
    if len(hit_dates) < 2:
        return 7.0
    deltas = np.diff(hit_dates.sort_values().values.astype("datetime64[D]")).astype(np.int64)
    return float(np.clip(np.mean(deltas), PATTERN_MIN_GAP_DAYS, PATTERN_MAX_GAP_DAYS))


def compute_tuple_order_patterns(
    panel: pd.DataFrame,
    lookback_end: pd.Timestamp | None = None,
    lookback_days: int = PATTERN_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Per-tuple cadence: short lookback plus blended Jan–Mar stats for pos_rate and order size."""
    lookback_end = lookback_end or HISTORY_END
    lb_start = lookback_end - pd.Timedelta(days=lookback_days - 1)
    hist = panel[panel["date"] <= lookback_end].copy()
    window = hist[(hist["date"] >= lb_start) & (hist["date"] <= lookback_end)].copy()

    meta = (
        panel.groupby(ID_COLS, as_index=False)
        .agg(
            customer_id=("customer_id", "first"),
            temperature_zone=("temperature_zone", "first"),
        )
    )

    win_agg = window.groupby(ID_COLS, as_index=False).agg(
        pos_rate_short=("qty_ea", lambda s: float((s > 0).mean())),
        mean_qty_pos=("qty_ea", lambda s: float(s[s > 0].mean()) if (s > 0).any() else 0.0),
        median_qty_short=("qty_ea", lambda s: float(s[s > 0].median()) if (s > 0).any() else 0.0),
    )

    long_hist = hist[(hist["date"] >= JAN_START) & (hist["date"] <= HISTORY_END)]
    long_agg = long_hist.groupby(ID_COLS, as_index=False).agg(
        pos_rate_long=("qty_ea", lambda s: float((s > 0).mean())),
        median_qty_long=("qty_ea", lambda s: float(s[s > 0].median()) if (s > 0).any() else 0.0),
    )

    last_order = (
        hist.loc[hist["qty_ea"] > 0, ID_COLS + ["date"]]
        .groupby(ID_COLS, as_index=False)["date"]
        .max()
        .rename(columns={"date": "last_order_date"})
    )

    gap_rows: list[dict] = []
    window["dow"] = window["date"].dt.dayofweek.astype(np.int16)
    for keys, g in window.groupby(ID_COLS):
        hit_dates = g.loc[g["qty_ea"] > 0, "date"]
        row = dict(zip(ID_COLS, keys if isinstance(keys, tuple) else (keys,)))
        row["mean_gap_days"] = _mean_interorder_gap(hit_dates)
        gap_rows.append(row)
    gaps = pd.DataFrame(gap_rows)

    dow = (
        window.groupby(ID_COLS + ["dow"], as_index=False)["qty_ea"]
        .apply(lambda s: float((s > 0).mean()))
        .rename(columns={"qty_ea": "dow_pos_rate"})
    )
    dow_wide = dow.pivot_table(index=ID_COLS, columns="dow", values="dow_pos_rate", fill_value=0.0)
    dow_wide.columns = [f"dow_{int(c)}" for c in dow_wide.columns]
    dow_wide = dow_wide.reset_index()
    for d in range(7):
        col = f"dow_{d}"
        if col not in dow_wide.columns:
            dow_wide[col] = 0.0

    patterns = meta.merge(win_agg, on=ID_COLS, how="left")
    patterns = patterns.merge(long_agg, on=ID_COLS, how="left")
    patterns = patterns.merge(gaps, on=ID_COLS, how="left")
    patterns = patterns.merge(dow_wide, on=ID_COLS, how="left")
    patterns = patterns.merge(last_order, on=ID_COLS, how="left")

    patterns["pos_rate_short"] = patterns["pos_rate_short"].fillna(0.0).clip(0.0, 1.0)
    patterns["mean_qty_pos"] = patterns["mean_qty_pos"].fillna(0.0)
    patterns["median_qty_short"] = patterns["median_qty_short"].fillna(patterns["mean_qty_pos"])
    patterns["pos_rate_long"] = patterns["pos_rate_long"].fillna(patterns["pos_rate_short"]).clip(0.0, 1.0)
    patterns["median_qty_long"] = patterns["median_qty_long"].fillna(patterns["median_qty_short"])

    w = PATTERN_LONG_HISTORY_BLEND
    patterns["pos_rate"] = (1.0 - w) * patterns["pos_rate_short"] + w * patterns["pos_rate_long"]
    patterns["pos_rate"] = patterns["pos_rate"].clip(0.0, 1.0)

    m = patterns["mean_qty_pos"]
    med_s = patterns["median_qty_short"]
    med_l = patterns["median_qty_long"]
    alpha = PATTERN_QTY_MEDIAN_WEIGHT
    beta = PATTERN_QTY_LONG_SHRINK
    base = alpha * med_s + (1.0 - alpha) * m
    patterns["mean_qty_pos"] = ((1.0 - beta) * base + beta * med_l).clip(lower=0.0)

    patterns["mean_gap_days"] = patterns["mean_gap_days"].fillna(7.0).clip(
        PATTERN_MIN_GAP_DAYS, PATTERN_MAX_GAP_DAYS
    )
    for d in range(7):
        patterns[f"dow_{d}"] = patterns[f"dow_{d}"].fillna(patterns["pos_rate"])

    patterns = patterns.drop(
        columns=["pos_rate_short", "median_qty_short", "pos_rate_long", "median_qty_long"],
        errors="ignore",
    )
    return patterns


def _scheduled_order_days(
    last_order: pd.Timestamp,
    gap_days: int,
    forecast_start: pd.Timestamp,
    forecast_end: pd.Timestamp,
    pos_rate: float,
    dow_rates: np.ndarray,
) -> set[pd.Timestamp]:
    """Order dates in [forecast_start, forecast_end] from gap phase anchored at last_order."""
    if pd.isna(last_order) or pos_rate <= 0:
        return set()
    gap = int(np.clip(gap_days, PATTERN_MIN_GAP_DAYS, PATTERN_MAX_GAP_DAYS))
    out: set[pd.Timestamp] = set()
    cur = pd.Timestamp(last_order).normalize()
    while True:
        cur = cur + pd.Timedelta(days=gap)
        if cur > forecast_end:
            break
        if cur < forecast_start:
            continue
        dow = int(cur.dayofweek)
        dow_p = float(dow_rates[dow]) if dow < len(dow_rates) else pos_rate
        if pos_rate <= 0 or dow_p >= pos_rate * PATTERN_DOW_MIN_FRAC:
            out.add(cur)
    return out


def pattern_forecast_april(
    patterns: pd.DataFrame,
    forecast_dates: pd.DatetimeIndex,
    closure: pd.DataFrame,
) -> pd.DataFrame:
    """April daily forecast by continuing gap + DOW cadence from lookback (no recursive lags)."""
    order_flags: list[pd.DataFrame] = []
    for _, pat in patterns.iterrows():
        dow_rates = np.array([pat[f"dow_{d}"] for d in range(7)], dtype=np.float64)
        gap = int(round(pat["mean_gap_days"]))
        days = _scheduled_order_days(
            pat["last_order_date"],
            gap,
            FORECAST_START,
            FORECAST_END,
            float(pat["pos_rate"]),
            dow_rates,
        )
        if not days:
            continue
        order_flags.append(
            pd.DataFrame(
                {
                    "warehouse": pat["warehouse"],
                    "store": pat["store"],
                    "product_id": pat["product_id"],
                    "date": list(days),
                    "is_order": 1,
                }
            )
        )

    orders = (
        pd.concat(order_flags, ignore_index=True)
        if order_flags
        else pd.DataFrame(columns=ID_COLS + ["date", "is_order"])
    )

    tuples = patterns[ID_COLS + ["customer_id", "mean_qty_pos"]].drop_duplicates(ID_COLS)
    tuples["_key"] = 1
    dates_df = pd.DataFrame({"date": forecast_dates, "_key": 1})
    grid = tuples.merge(dates_df, on="_key").drop(columns="_key")
    grid = grid.merge(orders, on=ID_COLS + ["date"], how="left")
    grid["is_order"] = grid["is_order"].fillna(0).astype(int)
    grid["qty_ea"] = np.where(grid["is_order"] == 1, grid["mean_qty_pos"].astype(np.float64), 0.0)
    grid["p_order"] = grid["is_order"].astype(np.float64)
    grid["mu_qty"] = np.where(grid["is_order"] == 1, grid["mean_qty_pos"].astype(np.float64), 0.0)

    wh_days = grid[["warehouse", "date"]].drop_duplicates()
    wh_days = add_cny_features(wh_days.assign(qty_ea=0.0), closure)
    grid = grid.merge(
        wh_days[["warehouse", "date", "is_warehouse_closed"]],
        on=["warehouse", "date"],
        how="left",
    )
    closed = grid["is_warehouse_closed"].fillna(0).astype(int) == 1
    grid.loc[closed, ["qty_ea", "p_order", "mu_qty"]] = 0.0

    return grid[ID_COLS + ["customer_id", "date", "qty_ea", "p_order", "mu_qty"]]


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


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = df["date"]
    df["dow"] = d.dt.dayofweek.astype(np.int16)
    df["dom"] = d.dt.day.astype(np.int16)
    df["month"] = d.dt.month.astype(np.int16)
    df["weekofyear"] = d.dt.isocalendar().week.astype(np.int16)
    df["is_weekend"] = (df["dow"] >= 5).astype(np.int8)
    dow = df["dow"].astype(np.float64)
    dom = df["dom"].astype(np.float64)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    df["dom_sin"] = np.sin(2 * np.pi * (dom - 1.0) / 30.0)
    df["dom_cos"] = np.cos(2 * np.pi * (dom - 1.0) / 30.0)
    return df


def add_cadence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Closure-aware days_since_last_order and trailing gap stats (single O(n) pass)."""
    df = df.sort_values(ID_COLS + ["date"]).reset_index(drop=True)
    if "order_signal" not in df.columns:
        df["order_signal"] = (df["qty_ea"] > 0).astype(np.float64)

    keys = pd.MultiIndex.from_frame(df[ID_COLS])
    gid = pd.factorize(keys, sort=False)[0].astype(np.int32)
    n = len(df)
    closed = df["is_warehouse_closed"].fillna(0).astype(int).values == 1
    sig = df["order_signal"].values.astype(np.float64)
    hit = (sig >= ORDER_PROB_THRESHOLD) & (~closed)
    hit_prev = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if gid[i] == gid[i - 1]:
            hit_prev[i] = hit[i - 1]
    t = df["date"].values.astype("datetime64[ns]").astype(np.int64) // 10**9

    days_since = np.zeros(n, dtype=np.float64)
    mean28 = np.full(n, 7.0, dtype=np.float64)
    mean56 = np.full(n, 7.0, dtype=np.float64)
    std28 = np.zeros(n, dtype=np.float64)

    open_zeros = 0.0
    prev = 0.0
    last_hit_t: int | None = None
    gaps: list[float] = []
    prev_gid = np.int32(-1)

    for i in range(n):
        g = gid[i]
        if g != prev_gid:
            open_zeros = 0.0
            prev = 0.0
            last_hit_t = None
            gaps = []
            prev_gid = g

        g28 = gaps[-28:] if gaps else []
        g56 = gaps[-56:] if gaps else []
        if len(g28) >= 2:
            mean28[i] = float(np.mean(g28))
            std28[i] = float(np.std(g28, ddof=0))
        elif len(g28) == 1:
            mean28[i] = float(g28[0])
        if len(g56) >= 2:
            mean56[i] = float(np.mean(g56))
        elif len(g56) == 1:
            mean56[i] = float(g56[0])

        if closed[i]:
            days_since[i] = prev
        else:
            if hit_prev[i]:
                days_since[i] = 0.0
                open_zeros = 1.0
            else:
                days_since[i] = open_zeros
                open_zeros += 1.0
        prev = days_since[i]

        if hit[i] and (not closed[i]) and last_hit_t is not None:
            gaps.append(max(1.0, float((t[i] - last_hit_t) / 86400.0)))
            if len(gaps) > 120:
                gaps.pop(0)
        if hit[i] and not closed[i]:
            last_hit_t = int(t[i])

    df["days_since_last_order"] = days_since
    df["mean_gap_28"] = mean28
    df["mean_gap_56"] = mean56
    df["std_gap_28"] = std28
    df["expected_gap_phase"] = df["days_since_last_order"] / np.maximum(df["mean_gap_28"], 1.0)
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
    grp = [df["warehouse"], df["store"], df["product_id"]]
    for w in ROLL_WINDOWS:
        rmean = shifted.groupby(grp, sort=False).rolling(w, min_periods=1).mean()
        df[f"rmean_{w}"] = rmean.droplevel([0, 1, 2])
        rstd = shifted.groupby(grp, sort=False).rolling(w, min_periods=1).std()
        df[f"rstd_{w}"] = rstd.droplevel([0, 1, 2])
        rmax = shifted.groupby(grp, sort=False).rolling(w, min_periods=1).max()
        df[f"rmax_{w}"] = rmax.droplevel([0, 1, 2])
        nz = (shifted_actual > 0).astype(np.float64)
        nzr = nz.groupby(grp, sort=False).rolling(w, min_periods=1).mean()
        df[f"nonzero_ratio_{w}"] = nzr.droplevel([0, 1, 2])
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
        "dow_sin",
        "dow_cos",
        "dom_sin",
        "dom_cos",
        "is_cny_window",
        "is_warehouse_closed",
        "days_to_cny",
        "days_from_cny",
        "days_since_last_order",
        "mean_gap_28",
        "mean_gap_56",
        "std_gap_28",
        "expected_gap_phase",
    ]
    return cols


def fit_hurdle(
    X: pd.DataFrame | np.ndarray,
    y_qty: pd.Series | np.ndarray,
    sample_weight: np.ndarray,
    *,
    use_gpu: bool = False,
) -> tuple[object, object, str, str]:
    if isinstance(X, pd.DataFrame):
        X_arr = X.to_numpy(dtype=np.float64)
    else:
        X_arr = np.asarray(X, dtype=np.float64)
    y_arr = y_qty.to_numpy(dtype=np.float64) if isinstance(y_qty, pd.Series) else np.asarray(y_qty, dtype=np.float64)
    y_bin = (y_arr > 0).astype(np.int8)

    if use_gpu:
        clf, clf_b = fit_classifier(X_arr, y_bin, sample_weight, use_gpu=True, num_boost_round=200)
        pos = y_bin == 1
        reg, reg_b = fit_regressor(
            X_arr[pos],
            y_arr[pos],
            sample_weight[pos] if sample_weight is not None else None,
            use_gpu=True,
            num_boost_round=350,
            params={"learning_rate": 0.08},
        )
        return clf, reg, clf_b, reg_b

    clf = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.08,
        max_iter=200,
        max_leaf_nodes=48,
        min_samples_leaf=80,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_arr, y_bin, sample_weight=sample_weight)
    pos = y_bin == 1
    reg = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.08,
        max_iter=350,
        max_leaf_nodes=48,
        min_samples_leaf=200,
        l2_regularization=0.0,
        early_stopping=False,
        random_state=42,
    )
    reg.fit(X_arr[pos], y_arr[pos], sample_weight=sample_weight[pos])
    return clf, reg, "sklearn_cpu", "sklearn_cpu"


def predict_hurdle(
    clf: object,
    reg: object,
    X: pd.DataFrame | np.ndarray,
    pos_rate_train: np.ndarray | None = None,
    *,
    clf_backend: str = "sklearn_cpu",
    reg_backend: str = "sklearn_cpu",
    blend_lambda: float | None = None,
    prior_k: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_arr = X.to_numpy(dtype=np.float64) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float64)
    if clf_backend == "xgboost_cuda":
        p_raw = predict_proba_positive(clf, clf_backend, X_arr)
    else:
        p_raw = clf.predict_proba(X_arr)[:, 1]
    mu = np.clip(gbdt_predict(reg, reg_backend, X_arr), 0, None)
    lam = P_ORDER_BLEND_LAMBDA if blend_lambda is None else float(blend_lambda)
    k = P_ORDER_PRIOR_K if prior_k is None else float(prior_k)
    if pos_rate_train is None or lam <= 0:
        p = p_raw
    else:
        pr = np.clip(np.asarray(pos_rate_train, dtype=np.float64), 1e-6, 1.0)
        p_prior = np.clip(pr * k, P_ORDER_MIN, pr)
        p = (1.0 - lam) * p_raw + lam * p_prior
        p = np.clip(p, 0.0, 1.0)
    order = (p >= ORDER_PROB_THRESHOLD).astype(np.float64)
    pred = np.clip(order * mu, 0, None)
    return pred, p, mu


def tune_prior_hyperparams(
    clf: object,
    reg: object,
    valid_open: pd.DataFrame,
    feat_cols: list[str],
    *,
    clf_backend: str = "sklearn_cpu",
    reg_backend: str = "sklearn_cpu",
) -> tuple[float, float]:
    """Pick (blend_lambda, prior_k) from March validation.

    Primary objective: monthly tuple totals bias_ratio nearest 1.
    Secondary (tie-break / gentle pull): lower hard-hurdle SMAPE on daily rows
    (same rule as ``predict_hurdle``: μ if P(order) >= threshold else 0).
    """
    X = valid_open[feat_cols].fillna(0.0)
    pr = valid_open["pos_rate_train"].values
    y_daily = valid_open["qty_ea"].astype(np.float64).values
    keys = ID_COLS + ["customer_id"]
    actual_m = (
        valid_open.groupby(keys, as_index=False)["qty_ea"]
        .sum()
        .rename(columns={"qty_ea": "actual"})
    )

    lambdas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    ks = [0.20, 0.30, 0.35, 0.45, 0.55]
    best_lam, best_k = 0.0, P_ORDER_PRIOR_K
    best_score = float("inf")
    best_br_err = float("inf")

    for lam in lambdas:
        for k in ks:
            pred, p_ord, _ = predict_hurdle(
                clf, reg, X, pr, blend_lambda=lam, prior_k=k,
                clf_backend=clf_backend, reg_backend=reg_backend,
            )
            pred_df = valid_open[keys].copy()
            pred_df["pred"] = pred
            pred_m = pred_df.groupby(keys, as_index=False)["pred"].sum()
            merged = actual_m.merge(pred_m, on=keys, how="inner")
            if merged["actual"].sum() <= 0:
                continue
            br = bias_ratio(merged["actual"].values, merged["pred"].values)
            br_err = abs(br - 1.0)
            sm_dec = evaluate_arrays(y_daily, pred)["smape"]
            score = br_err + PRIOR_TUNE_DECISION_SMAPE_WEIGHT * sm_dec
            if score < best_score - 1e-9 or (
                abs(score - best_score) <= 1e-9 and br_err < best_br_err
            ):
                best_score = score
                best_br_err = br_err
                best_lam, best_k = lam, k

    log(
        f"  tuned P(order) blend: lambda={best_lam:.2f}  prior_k={best_k:.2f}  "
        f"(score=monthly|bias_ratio-1|+{PRIOR_TUNE_DECISION_SMAPE_WEIGHT}*hard_SMAPE "
        f"→ {best_score:.4f}; monthly |bias_ratio-1| alone → {best_br_err:.4f})"
    )
    return best_lam, best_k


def soften_april_first_day_spike(daily_fc: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Scale down Apr 1 if its network daily total is far above the rest of April (recursive cold start)."""
    daily_tot = daily_fc.groupby("date")["qty_ea"].sum()
    d0 = FORECAST_START
    if d0 not in daily_tot.index:
        return daily_fc, 1.0
    d0_total = float(daily_tot.loc[d0])
    later = daily_tot[daily_tot.index > d0]
    if later.empty or d0_total <= 0:
        return daily_fc, 1.0
    peer_mean = float(later.mean())
    if peer_mean <= 0 or d0_total <= peer_mean * 1.15:
        return daily_fc, 1.0
    scale = peer_mean / d0_total
    out = daily_fc.copy()
    mask = out["date"] == d0
    out.loc[mask, "qty_ea"] = out.loc[mask, "qty_ea"].astype(np.float64) * scale
    mu = np.maximum(out.loc[mask, "mu_qty"].astype(np.float64).values, 1e-9)
    out.loc[mask, "p_order"] = np.clip(out.loc[mask, "qty_ea"].values / mu, 0.0, 1.0)
    log(
        f"  April 1 spike soften: scale={scale:.4f}  "
        f"(day1 total {d0_total:,.0f} -> {d0_total * scale:,.0f}, peer mean {peer_mean:,.0f})"
    )
    return out, scale


def calibrate_april_forecast(
    daily_fc: pd.DataFrame,
    q_target: float,
) -> tuple[pd.DataFrame, float, float]:
    """Uniform scale on April qty_ea and mu_qty so mean network daily total matches q_target."""
    q_raw = mean_network_daily_total(daily_fc)
    if q_raw <= 0:
        log("WARNING: Q_apr_raw is zero; skipping April calibration")
        return daily_fc, 1.0, q_raw

    scale_raw = q_target / q_raw
    scale = float(np.clip(scale_raw, APRIL_SCALE_MIN, APRIL_SCALE_MAX))
    if abs(scale - scale_raw) > 1e-6:
        log(
            f"WARNING: April scale {scale_raw:.4f} clipped to [{APRIL_SCALE_MIN}, {APRIL_SCALE_MAX}] "
            f"-> {scale:.4f}"
        )

    out = daily_fc.copy()
    out["qty_ea"] = out["qty_ea"].astype(np.float64) * scale
    out["mu_qty"] = out["mu_qty"].astype(np.float64) * scale
    mu = np.maximum(out["mu_qty"].astype(np.float64).values, 1e-9)
    out["p_order"] = np.clip(out["qty_ea"].values / mu, 0.0, 1.0)
    return out, scale, q_raw


def evaluate_reg(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metrics = evaluate_arrays(y_true, y_pred)
    log(
        f"  [{name}] MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}  "
        f"SMAPE={metrics['smape']:.3f}  bias={metrics['bias']:+.3f}"
    )
    return metrics


def evaluate_reg_masked(name: str, metrics: dict[str, float]) -> None:
    log(
        f"  [{name}] MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}  "
        f"SMAPE={metrics['smape']:.3f}  bias={metrics['bias']:+.3f}"
    )


def evaluate_classifier(y_true: np.ndarray, p_pred: np.ndarray) -> dict[str, float]:
    y = y_true.astype(int)
    out: dict[str, float] = {}
    out["log_loss"] = float(log_loss(y, np.clip(p_pred, 1e-15, 1 - 1e-15), labels=[0, 1]))
    out["brier"] = float(brier_score_loss(y, p_pred))
    if np.unique(y).size > 1:
        out["roc_auc"] = float(roc_auc_score(y, p_pred))
    else:
        out["roc_auc"] = float("nan")
    log(
        f"  [clf] log_loss={out['log_loss']:.4f}  brier={out['brier']:.4f}  "
        f"roc_auc={out['roc_auc']}"
    )
    return out


def engineer_features(
    panel: pd.DataFrame,
    encoders: dict[str, dict[str, int]],
    closure: pd.DataFrame,
) -> pd.DataFrame:
    had_sig = "order_signal" in panel.columns
    sig = panel["order_signal"].reset_index(drop=True) if had_sig else None
    df = add_cny_features(panel.copy(), closure)
    df["qty_ea_masked"] = mask_qty_for_features(df)
    if had_sig:
        df["order_signal"] = sig.astype(np.float64).values
    else:
        df["order_signal"] = (df["qty_ea"] > 0).astype(np.float64)
    df = add_lag_features(df)
    df = add_calendar(df)
    df = add_cadence_features(df)
    df = apply_encoders(df, encoders)
    return df


def recursive_forecast(
    clf: HistGradientBoostingClassifier,
    reg: HistGradientBoostingRegressor,
    panel: pd.DataFrame,
    forecast_dates: pd.DatetimeIndex,
    encoders: dict[str, dict[str, int]],
    closure: pd.DataFrame,
    tuple_priors: pd.DataFrame,
    *,
    future_qty_mode: str = "zero",
) -> pd.DataFrame:
    feat_cols = feature_columns()
    meta_cols = ID_COLS + ["temperature_zone", "customer_id"]
    tuple_meta = panel[meta_cols].drop_duplicates(ID_COLS).reset_index(drop=True)
    tuple_meta = tuple_meta.merge(tuple_priors, on=ID_COLS, how="left")
    tuple_meta["pos_rate_train"] = tuple_meta["pos_rate_train"].fillna(0.03).clip(1e-6, 1.0)
    tuple_meta["seed_qty"] = tuple_meta["seed_qty"].fillna(0.0)

    rolling = panel[ID_COLS + ["date", "qty_ea", "temperature_zone", "customer_id"]].copy()
    rolling["order_signal"] = (rolling["qty_ea"] > 0).astype(np.float64)
    out_rows: list[pd.DataFrame] = []

    keep_from = forecast_dates.min() - pd.Timedelta(days=max(LAG_DAYS) + max(ROLL_WINDOWS))
    rolling = rolling[rolling["date"] >= keep_from].copy()

    for di, d in enumerate(forecast_dates):
        if di == 0 or (di + 1) % 6 == 0 or di == len(forecast_dates) - 1:
            log(f"  recursive forecast {d.date()} ({di + 1}/{len(forecast_dates)})")
        future = tuple_meta.copy()
        future["date"] = d
        seed_vals = future["seed_qty"].values
        # Baseline-style placeholder: do not inject seed_qty into the forecast day's row.
        future["qty_ea"] = 0.0
        future["order_signal"] = 0.0
        future_only = future.drop(columns=["seed_qty", "pos_rate_train"])
        ext = pd.concat([rolling, future_only], ignore_index=True)
        ext = engineer_features(ext, encoders, closure)
        slice_d = ext[ext["date"] == d].copy()
        slice_d = slice_d.merge(
            tuple_meta[ID_COLS + ["seed_qty", "pos_rate_train"]],
            on=ID_COLS,
            how="left",
        )
        slice_d["pos_rate_train"] = slice_d["pos_rate_train"].fillna(0.03)
        slice_d["seed_qty"] = slice_d["seed_qty"].fillna(0.0)

        X = slice_d[feat_cols].fillna(0.0)
        p_raw_dbg = clf.predict_proba(X)[:, 1]
        pred, p, mu = predict_hurdle(clf, reg, X, slice_d["pos_rate_train"].values)
        slice_d["qty_ea"] = pred
        slice_d["p_order"] = p
        slice_d["mu_qty"] = mu

        # #region agent log
        if di <= 2:
            feat = slice_d[feat_cols].fillna(0.0)
            _agent_debug_log(
                "A",
                "04b_hurdle_model.py:recursive_forecast",
                "apr_day_features_and_pred",
                {
                    "di": di,
                    "date": str(d.date()),
                    "future_qty_mode": future_qty_mode,
                    "sum_pred": float(np.sum(pred)),
                    "mean_pred": float(np.mean(pred)),
                    "mean_p_raw": float(np.mean(p_raw_dbg)),
                    "mean_p": float(np.mean(p)),
                    "mean_mu": float(np.mean(mu)),
                    "mean_seed_qty": float(np.mean(seed_vals)),
                    "mean_lag_1": float(feat["lag_1"].mean()) if "lag_1" in feat else None,
                    "mean_rmean_7": float(feat["rmean_7"].mean()) if "rmean_7" in feat else None,
                    "mean_days_since_last_order": float(feat["days_since_last_order"].mean())
                    if "days_since_last_order" in feat
                    else None,
                    "frac_order_signal_future": float((seed_vals > 0).mean())
                    if future_qty_mode == "seed"
                    else 0.0,
                },
            )
        # #endregion

        seed_arr = slice_d["seed_qty"].values
        qty_roll = np.maximum(pred, ROLL_QTY_MIN_FRAC_SEED * seed_arr)

        order_hit = (p >= ORDER_PROB_THRESHOLD) & (qty_roll > 0)
        slice_roll = slice_d[ID_COLS + ["date", "temperature_zone", "customer_id"]].copy()
        # #region agent log
        if di <= 1:
            _agent_debug_log(
                "E",
                "04b_hurdle_model.py:recursive_forecast",
                "order_hit_rate",
                {
                    "di": di,
                    "future_qty_mode": future_qty_mode,
                    "frac_order_hit": float(np.mean(order_hit)),
                    "mean_qty_roll": float(np.mean(qty_roll)),
                },
            )
        # #endregion
        slice_roll["qty_ea"] = qty_roll
        slice_roll["order_signal"] = order_hit.astype(np.float64)

        rolling = pd.concat([rolling, slice_roll], ignore_index=True)
        rolling = rolling[rolling["date"] >= d - pd.Timedelta(days=max(LAG_DAYS) + max(ROLL_WINDOWS))]
        out_rows.append(
            slice_d[
                ID_COLS + ["customer_id", "date", "qty_ea", "p_order", "mu_qty"]
            ]
        )

    return pd.concat(out_rows, ignore_index=True)


def main() -> int:
    global P_ORDER_BLEND_LAMBDA, P_ORDER_PRIOR_K

    ensure_report_dirs()
    use_gpu = want_gpu("HURDLE")
    log(f"Training backend: {'XGBoost CUDA' if use_gpu else 'sklearn HistGradientBoosting (CPU)'}")
    log("Loading store-level mart...")
    raw = load_panel_base()

    log("Building closure table from raw (warehouse-day totals)...")
    closure = detect_warehouse_closure(raw)

    log("Building dense panel...")
    panel = build_dense_panel(raw, HISTORY_END)
    nt = panel.groupby(ID_COLS).ngroups
    log(
        f"  panel rows: {len(panel):,}  tuples: {nt:,}  "
        f"(min {MIN_NONZERO_DAYS_TRAIN}+ nonzero train days per tuple)"
    )

    q_stats = compute_q_targets(panel)
    log(
        f"  network daily mean (dense panel): Q_Jan={q_stats['Q_Jan']:,.1f}  "
        f"Q_Mar={q_stats['Q_Mar']:,.1f}  Q_target={q_stats['Q_target']:,.1f}"
    )

    log("Sliding-window samples + encodings...")
    panel = add_cny_features(panel, closure)
    encoders = build_encoders(panel)
    static_enc = apply_encoders(
        panel[ID_COLS + ["customer_id", "temperature_zone"]].drop_duplicates(ID_COLS),
        encoders,
    )
    feat_cols = sliding_feature_columns(CATEGORICAL_ENC)
    tuple_priors = tuple_recursive_priors(panel)

    train_anchors = anchors_with_horizon_in(
        JAN_START, TRAIN_END, step_days=SLIDE_STEP_TRAIN_DAYS
    )
    valid_anchors = anchors_with_horizon_in(
        VALID_START, VALID_END, step_days=SLIDE_STEP_TRAIN_DAYS
    )
    log(
        f"  lookback={LOOKBACK_OPEN_DAYS} open days, horizon={HORIZON_DAYS}d, "
        f"train_step={SLIDE_STEP_TRAIN_DAYS}d, april_step={SLIDE_STEP_DAYS}d  "
        f"train_anchors={len(train_anchors)}  valid_anchors={len(valid_anchors)}"
    )

    train_df = build_sliding_samples(
        panel, ID_COLS, CATEGORICAL_ENC, static_enc, train_anchors, closure
    )
    valid_df = build_sliding_samples(
        panel, ID_COLS, CATEGORICAL_ENC, static_enc, valid_anchors, closure
    )
    log(f"  train sample rows: {len(train_df):,}  valid sample rows: {len(valid_df):,}")

    w_train = sample_weights_sliding(train_df)
    X_train = train_df[feat_cols].fillna(0.0)
    y_train = train_df["qty_ea"].astype(float)

    valid_open = valid_df[valid_df["is_warehouse_closed"] != 1].copy()
    valid_open = valid_open.merge(tuple_priors, on=ID_COLS, how="left")
    valid_open["pos_rate_train"] = valid_open["pos_rate_train"].fillna(0.03)

    X_valid = valid_open[feat_cols].fillna(0.0)
    y_valid = valid_open["qty_ea"].astype(float)

    log("Training hurdle on sliding Jan-Feb horizons...")
    clf_val, reg_val, clf_b, reg_b = fit_hurdle(X_train, y_train, w_train, use_gpu=use_gpu)

    log("Tuning P(order) prior blend on March sliding validation...")
    blend_lam, prior_k = tune_prior_hyperparams(
        clf_val, reg_val, valid_open, feat_cols, clf_backend=clf_b, reg_backend=reg_b
    )

    pred_val, p_val, mu_val = predict_hurdle(
        clf_val,
        reg_val,
        X_valid,
        valid_open["pos_rate_train"].values,
        blend_lambda=blend_lam,
        prior_k=prior_k,
        clf_backend=clf_b,
        reg_backend=reg_b,
    )

    log("Validation (regression, daily, hard hurdle):")
    overall = evaluate_reg(
        f"overall_daily_hard_hurdle (p_order>={ORDER_PROB_THRESHOLD})",
        y_valid.values,
        pred_val,
    )

    pos_m, n_pos = evaluate_arrays_positive_actual(y_valid.values, pred_val)
    evaluate_reg_masked(f"overall_daily_actual_gt0_only (n={n_pos:,})", pos_m)

    sf = smape_fracs(y_valid.values, pred_val)
    log(
        "  [dense SMAPE drivers] "
        f"frac(y=0 & pred>0)={sf['frac_y0_pred_pos']:.3f} → each row adds SMAPE=2.0; "
        f"frac(both 0)={sf['frac_both_zero']:.3f}; frac(y>0 & pred=0)={sf['frac_ypos_pred0']:.3f}"
    )

    y_bin = (y_valid.values > 0).astype(int)
    log("Validation (classifier, raw proba for diagnostics):")
    if clf_b == "xgboost_cuda":
        p_raw_val = predict_proba_positive(clf_val, clf_b, X_valid.to_numpy(dtype=np.float64))
    else:
        p_raw_val = clf_val.predict_proba(X_valid)[:, 1]
    clf_metrics = evaluate_classifier(y_bin, p_raw_val)

    diag = valid_open[ID_COLS + ["customer_id", "date", "qty_ea"]].copy()
    diag["pred"] = np.clip(pred_val, 0, None)
    diag["p_order"] = p_val
    diag["mu_qty"] = mu_val

    log("Validation (intermittent loss — tuple×month freq + segment volume + qty on order days)...")
    iloss = intermittent_hurdle_loss_report(
        diag,
        tuple_keys=ID_COLS + ["customer_id"],
        decision_tau=ORDER_PROB_THRESHOLD,
    )
    log(
        f"  segments={iloss.get('n_segments', 0):,}  "
        f"freq_count_MAE_soft={iloss.get('freq_count_mae_soft', float('nan')):.3f}  "
        f"freq_count_MAE_hard={iloss.get('freq_count_mae_hard', float('nan')):.3f}  "
        f"freq_rate_MAE_soft={iloss.get('freq_rate_mae_soft', float('nan')):.4f}  "
        f"seg_vol_WMAPE={iloss.get('segment_volume_wmape', float('nan')):.3f}  "
        f"seg_vol_bias_ratio={iloss.get('segment_volume_bias_ratio', float('nan')):+.3f}  "
        f"qty_on_order_MAE={iloss.get('positive_day_mae', float('nan')):.3f}  "
        f"composite={iloss.get('composite_intermittent_loss', float('nan')):.4f}"
    )
    iloss_rows = [
        {"metric": k, "value": v}
        for k, v in iloss.items()
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
    ]
    pd.DataFrame(iloss_rows).to_csv(
        HURDLE_VALIDATION_INTERMITTENT, index=False, encoding="utf-8-sig"
    )

    log(
        "Validation (store planning loss — product+qty within "
        f"±{PLANNING_MATCH_WINDOW_DAYS}d; timing slip unpunished in window)..."
    )
    ploss = store_planning_loss_report(
        diag,
        store_keys=STORE_KEYS,
        window_days=PLANNING_MATCH_WINDOW_DAYS,
        qty_rel_tol=PLANNING_QTY_REL_TOL,
        min_pred_qty=0.0,
    )
    log(
        f"  stores={ploss.get('n_stores', 0):,}  "
        f"planning_window_loss={ploss.get('planning_window_loss', float('nan')):.4f}  "
        f"strict_daily_wmape={ploss.get('strict_daily_wmape', float('nan')):.4f}  "
        f"composite={ploss.get('composite_planning_loss', float('nan')):.4f}  "
        f"matched_vol={ploss.get('matched_volume_rate', float('nan')):.3f}  "
        f"mean_|day_slip|={ploss.get('mean_abs_day_slip_matched', float('nan')):.2f}"
    )
    ploss_rows = [
        {"metric": k, "value": v}
        for k, v in ploss.items()
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
    ]
    pd.DataFrame(ploss_rows).to_csv(
        HURDLE_VALIDATION_PLANNING, index=False, encoding="utf-8-sig"
    )

    per_wh = diag.groupby("warehouse", observed=True)[["qty_ea", "pred"]].apply(
        lambda g: pd.Series(evaluate_arrays(g["qty_ea"].values, g["pred"].values)),
    ).reset_index()

    per_cust = diag.groupby("customer_id", observed=True)[["qty_ea", "pred"]].apply(
        lambda g: pd.Series(evaluate_arrays(g["qty_ea"].values, g["pred"].values)),
    ).reset_index()

    for _, row in per_wh.iterrows():
        log(
            f"  [wh={row['warehouse']}] MAE={row['mae']:.3f}  RMSE={row['rmse']:.3f}  "
            f"SMAPE={row['smape']:.3f}  bias={row['bias']:+.3f}"
        )
    for _, row in per_cust.iterrows():
        log(
            f"  [cust={row['customer_id']}] MAE={row['mae']:.3f}  RMSE={row['rmse']:.3f}  "
            f"SMAPE={row['smape']:.3f}  bias={row['bias']:+.3f}"
        )

    monthly = (
        diag.groupby(ID_COLS + ["customer_id"])
        .agg(actual=("qty_ea", "sum"), pred=("pred", "sum"))
        .reset_index()
    )
    monthly_overall = evaluate_reg(
        "monthly (wh,store,prod)", monthly["actual"].values, monthly["pred"].values
    )

    wmape_m = wmape(monthly["actual"].values, monthly["pred"].values)
    bias_m = bias_ratio(monthly["actual"].values, monthly["pred"].values)
    log(f"  monthly WMAPE={wmape_m:.4f}  bias_ratio={bias_m:+.4f}")

    pd.DataFrame([clf_metrics]).to_csv(
        HURDLE_VALIDATION_CLASSIFIER, index=False, encoding="utf-8-sig"
    )
    per_wh.to_csv(HURDLE_VALIDATION_PER_WAREHOUSE, index=False, encoding="utf-8-sig")
    per_cust.to_csv(HURDLE_VALIDATION_PER_CUSTOMER, index=False, encoding="utf-8-sig")
    monthly.to_csv(HURDLE_VALIDATION_MONTHLY, index=False, encoding="utf-8-sig")
    cal_rows = [
        {"metric": f"daily_{k}", "value": v} for k, v in overall.items()
    ] + [{"metric": f"monthly_{k}", "value": v} for k, v in monthly_overall.items()]
    cal_rows += [
        {"metric": f"daily_actual_gt0_{k}", "value": v} for k, v in pos_m.items()
    ]
    cal_rows += [
        {"metric": "daily_n_actual_gt0", "value": float(n_pos)},
        {"metric": "dense_frac_y0_pred_pos", "value": sf["frac_y0_pred_pos"]},
        {"metric": "dense_frac_both_zero", "value": sf["frac_both_zero"]},
        {"metric": "dense_frac_ypos_pred0", "value": sf["frac_ypos_pred0"]},
        {"metric": "decision_prob_threshold", "value": float(ORDER_PROB_THRESHOLD)},
        {
            "metric": "intermittent_composite_loss",
            "value": float(iloss.get("composite_intermittent_loss", float("nan"))),
        },
        {
            "metric": "intermittent_freq_count_mae_soft",
            "value": float(iloss.get("freq_count_mae_soft", float("nan"))),
        },
        {
            "metric": "intermittent_segment_volume_wmape",
            "value": float(iloss.get("segment_volume_wmape", float("nan"))),
        },
        {
            "metric": "planning_window_loss",
            "value": float(ploss.get("planning_window_loss", float("nan"))),
        },
        {
            "metric": "planning_composite_loss",
            "value": float(ploss.get("composite_planning_loss", float("nan"))),
        },
        {
            "metric": "planning_strict_daily_wmape",
            "value": float(ploss.get("strict_daily_wmape", float("nan"))),
        },
        {
            "metric": "planning_match_window_days",
            "value": float(PLANNING_MATCH_WINDOW_DAYS),
        },
    ]
    cal_rows += [
        {"metric": "monthly_wmape", "value": wmape_m},
        {"metric": "monthly_bias_ratio", "value": bias_m},
        {"metric": "Q_Jan", "value": q_stats["Q_Jan"]},
        {"metric": "Q_Mar", "value": q_stats["Q_Mar"]},
        {"metric": "Q_target", "value": q_stats["Q_target"]},
        {"metric": "P_order_blend_lambda", "value": P_ORDER_BLEND_LAMBDA},
        {"metric": "P_order_prior_k", "value": P_ORDER_PRIOR_K},
        {"metric": "sliding_lookback_open_days", "value": float(LOOKBACK_OPEN_DAYS)},
        {"metric": "sliding_horizon_days", "value": float(HORIZON_DAYS)},
        {"metric": "sliding_step_train_days", "value": float(SLIDE_STEP_TRAIN_DAYS)},
        {"metric": "sliding_step_april_days", "value": float(SLIDE_STEP_DAYS)},
        {"metric": "training_backend_xgboost_cuda", "value": float(use_gpu)},
    ]
    pd.DataFrame(cal_rows).to_csv(
        HURDLE_VALIDATION_OVERALL, index=False, encoding="utf-8-sig"
    )

    log("Retraining hurdle on sliding samples through March...")
    full_anchors = anchors_with_horizon_in(
        JAN_START, HISTORY_END, step_days=SLIDE_STEP_TRAIN_DAYS
    )
    full_df = build_sliding_samples(
        panel, ID_COLS, CATEGORICAL_ENC, static_enc, full_anchors, closure
    )
    w_full = sample_weights_sliding(full_df)
    clf_full, reg_full, clf_bf, reg_bf = fit_hurdle(
        full_df[feat_cols].fillna(0.0),
        full_df["qty_ea"].astype(float),
        w_full,
        use_gpu=use_gpu,
    )

    def predict_hurdle_batch(batch: pd.DataFrame) -> np.ndarray:
        b = batch.merge(tuple_priors, on=ID_COLS, how="left")
        b["pos_rate_train"] = b["pos_rate_train"].fillna(0.03)
        pred, _, _ = predict_hurdle(
            clf_full,
            reg_full,
            b[feat_cols].fillna(0.0),
            b["pos_rate_train"].values,
            blend_lambda=P_ORDER_BLEND_LAMBDA,
            prior_k=P_ORDER_PRIOR_K,
            clf_backend=clf_bf,
            reg_backend=reg_bf,
        )
        return pred

    log("Sliding April forecast (weekly blocks; predictions extend lookback)...")
    april_anchors = anchors_with_horizon_in(FORECAST_START, FORECAST_END)
    daily_fc = sliding_forecast_period(
        panel,
        ID_COLS,
        CATEGORICAL_ENC,
        static_enc,
        april_anchors,
        predict_hurdle_batch,
        closure,
    )
    q_apr = mean_network_daily_total(daily_fc)
    log(f"  April mean daily network total Q_apr={q_apr:,.1f}  (Q_target={q_stats['Q_target']:,.1f})")

    name_lookup = panel[
        ID_COLS + ["customer_id", "product_name", "temperature_zone"]
    ].drop_duplicates(ID_COLS)
    daily_fc = daily_fc.merge(name_lookup, on=ID_COLS, how="left")
    for path in (
        HURDLE_APRIL_DAILY_NETWORK,
        HURDLE_APRIL_DAILY_STORE,
        HURDLE_APRIL_DAILY_PATTERN,
    ):
        write_hurdle_daily_csv(daily_fc, path, name_lookup)
    log(
        f"  Wrote April daily (sliding, all tiers identical): {HURDLE_DAILY_PATTERN_CSV}, "
        f"{HURDLE_DAILY_STORE_CSV}, {HURDLE_DAILY_NETWORK_CSV}"
    )

    pd.DataFrame(
        [
            {"metric": "Q_Jan", "value": q_stats["Q_Jan"]},
            {"metric": "Q_Mar", "value": q_stats["Q_Mar"]},
            {"metric": "Q_target", "value": q_stats["Q_target"]},
            {"metric": "Q_apr_sliding", "value": q_apr},
            {"metric": "sliding_lookback_open_days", "value": float(LOOKBACK_OPEN_DAYS)},
            {"metric": "sliding_horizon_days", "value": float(HORIZON_DAYS)},
            {"metric": "sliding_step_train_days", "value": float(SLIDE_STEP_TRAIN_DAYS)},
            {"metric": "sliding_step_april_days", "value": float(SLIDE_STEP_DAYS)},
            {"metric": "P_order_blend_lambda", "value": P_ORDER_BLEND_LAMBDA},
            {"metric": "P_order_prior_k", "value": P_ORDER_PRIOR_K},
        ]
    ).to_csv(HURDLE_APRIL_CALIBRATION, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        columns=["warehouse", "customer_id", "store", "store_scale", "store_scale_clipped"]
    ).to_csv(HURDLE_APRIL_STORE_CALIBRATION, index=False, encoding="utf-8-sig")

    monthly_wcp = (
        daily_fc.groupby(["warehouse", "customer_id", "product_id", "product_name", "temperature_zone"])[
            "qty_ea"
        ]
        .sum()
        .reset_index()
        .rename(columns={"qty_ea": "predicted_qty_ea_april"})
    )
    monthly_wcp.to_csv(
        HURDLE_APRIL_MONTHLY_WCP, index=False, encoding="utf-8-sig"
    )

    monthly_store = (
        daily_fc.groupby(ID_COLS + ["customer_id", "product_name", "temperature_zone"])["qty_ea"]
        .sum()
        .reset_index()
        .rename(columns={"qty_ea": "predicted_qty_ea_april"})
    )
    monthly_store.to_csv(HURDLE_APRIL_MONTHLY_STORE, index=False, encoding="utf-8-sig")

    monthly_cp = (
        daily_fc.groupby(["customer_id", "product_id", "product_name", "temperature_zone"])["qty_ea"]
        .sum()
        .reset_index()
        .rename(columns={"qty_ea": "predicted_qty_ea_april"})
    )
    monthly_cp.to_csv(HURDLE_APRIL_MONTHLY_CP, index=False, encoding="utf-8-sig")

    monthly_wh = (
        daily_fc.groupby("warehouse")["qty_ea"]
        .sum()
        .reset_index()
        .rename(columns={"qty_ea": "predicted_qty_ea_april"})
        .sort_values("predicted_qty_ea_april", ascending=False)
    )
    monthly_wh.to_csv(HURDLE_APRIL_MONTHLY_WH, index=False, encoding="utf-8-sig")

    log("Done.")
    log(
        f"April daily rows: {len(daily_fc):,}  total pred qty_ea: {daily_fc['qty_ea'].sum():,.1f}  "
        f"mean daily network total: {q_apr:,.1f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
