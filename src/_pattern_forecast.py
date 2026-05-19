"""Gap + DOW pattern April forecast for intermittent store×product demand.

Shared by hurdle (04b) and store regression (04c). Schedules explicit order days
from historical cadence instead of spreading Poisson expectations across every day.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from _cny import add_cny_features

ID_COLS = ["warehouse", "store", "product_id"]

PATTERN_LOOKBACK_DAYS = 28
PATTERN_MIN_GAP_DAYS = 1
PATTERN_MAX_GAP_DAYS = 28
PATTERN_DOW_MIN_FRAC = 0.35
PATTERN_LONG_HISTORY_BLEND = 0.35
PATTERN_QTY_MEDIAN_WEIGHT = 0.55
PATTERN_QTY_LONG_SHRINK = 0.30


def _mean_interorder_gap(hit_dates: pd.Series) -> float:
    if len(hit_dates) < 2:
        return 7.0
    deltas = np.diff(hit_dates.sort_values().values.astype("datetime64[D]")).astype(np.int64)
    return float(np.clip(np.mean(deltas), PATTERN_MIN_GAP_DAYS, PATTERN_MAX_GAP_DAYS))


def compute_tuple_order_patterns(
    panel: pd.DataFrame,
    *,
    history_end: pd.Timestamp,
    jan_start: pd.Timestamp,
    lookback_days: int = PATTERN_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Per-tuple cadence: short lookback plus blended Jan–history stats."""
    lb_start = history_end - pd.Timedelta(days=lookback_days - 1)
    hist = panel[panel["date"] <= history_end].copy()
    window = hist[(hist["date"] >= lb_start) & (hist["date"] <= history_end)].copy()

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

    long_hist = hist[(hist["date"] >= jan_start) & (hist["date"] <= history_end)]
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
    window = window.copy()
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

    return patterns.drop(
        columns=["pos_rate_short", "median_qty_short", "pos_rate_long", "median_qty_long"],
        errors="ignore",
    )


def _scheduled_order_days(
    last_order: pd.Timestamp,
    gap_days: int,
    forecast_start: pd.Timestamp,
    forecast_end: pd.Timestamp,
    pos_rate: float,
    dow_rates: np.ndarray,
) -> set[pd.Timestamp]:
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
    *,
    forecast_start: pd.Timestamp,
    forecast_end: pd.Timestamp,
) -> pd.DataFrame:
    """April daily forecast: order days from gap phase + robust positive-day qty."""
    order_flags: list[pd.DataFrame] = []
    for _, pat in patterns.iterrows():
        dow_rates = np.array([pat[f"dow_{d}"] for d in range(7)], dtype=np.float64)
        gap = int(round(pat["mean_gap_days"]))
        days = _scheduled_order_days(
            pat["last_order_date"],
            gap,
            forecast_start,
            forecast_end,
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

    wh_days = grid[["warehouse", "date"]].drop_duplicates()
    wh_days = add_cny_features(wh_days.assign(qty_ea=0.0), closure)
    grid = grid.merge(
        wh_days[["warehouse", "date", "is_warehouse_closed"]],
        on=["warehouse", "date"],
        how="left",
    )
    closed = grid["is_warehouse_closed"].fillna(0).astype(int) == 1
    grid.loc[closed, "qty_ea"] = 0.0

    return grid[ID_COLS + ["customer_id", "date", "qty_ea"]]
