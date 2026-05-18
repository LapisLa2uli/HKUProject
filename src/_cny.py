"""Chinese New Year (CNY) handling for demand forecasting.

Per-warehouse closure detection, calendar features, masked qty for lags,
and sample weights. Shared by baseline and hurdle models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CNY_DATE = pd.Timestamp("2026-02-17")
CNY_WINDOW_START = pd.Timestamp("2026-02-10")
CNY_WINDOW_END = pd.Timestamp("2026-02-22")
CNY_WINDOW = (CNY_WINDOW_START, CNY_WINDOW_END)
JAN_BASELINE_START = pd.Timestamp("2026-01-15")
JAN_BASELINE_END = pd.Timestamp("2026-01-31")

# Within CNY_WINDOW, mark (warehouse, date) closed if daily qty < this fraction
# of that warehouse's mean daily qty over JAN_BASELINE.
CLOSURE_QTY_RATIO_THRESHOLD = 0.25


def detect_warehouse_closure(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-(warehouse, date) with ``is_warehouse_closed`` in {0, 1}.

    ``df`` must include ``warehouse``, ``date`` (or ``create_date``), and ``qty_ea``.
    Multiple rows per warehouse-day are summed.

    Rule: within ``CNY_WINDOW``, mark closed iff that day's total qty_ea for the
    warehouse is < ``CLOSURE_QTY_RATIO_THRESHOLD`` * mean daily qty_ea over
    ``JAN_BASELINE_START``..``JAN_BASELINE_END`` for that warehouse.
    Outside CNY_WINDOW, ``is_warehouse_closed`` is 0.
    """
    d = df.copy()
    date_col = "date" if "date" in d.columns else "create_date"
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce").dt.normalize()
    d["qty_ea"] = pd.to_numeric(d["qty_ea"], errors="coerce").fillna(0.0)
    daily = d.groupby(["warehouse", date_col], as_index=False)["qty_ea"].sum()
    daily = daily.rename(columns={date_col: "date"})

    jan_mask = (daily["date"] >= JAN_BASELINE_START) & (daily["date"] <= JAN_BASELINE_END)
    jan_mean = (
        daily.loc[jan_mask]
        .groupby("warehouse", as_index=False)["qty_ea"]
        .mean()
        .rename(columns={"qty_ea": "jan_mean_daily_qty"})
    )
    daily = daily.merge(jan_mean, on="warehouse", how="left")
    daily["jan_mean_daily_qty"] = daily["jan_mean_daily_qty"].fillna(
        daily.groupby("warehouse")["qty_ea"].transform("mean")
    )

    in_window = (daily["date"] >= CNY_WINDOW_START) & (daily["date"] <= CNY_WINDOW_END)
    threshold = CLOSURE_QTY_RATIO_THRESHOLD * daily["jan_mean_daily_qty"].clip(lower=1e-6)
    daily["is_warehouse_closed"] = (
        in_window & (daily["qty_ea"] < threshold)
    ).astype(np.int8)

    return daily[["warehouse", "date", "is_warehouse_closed"]]


def add_cny_features(df: pd.DataFrame, closure: pd.DataFrame) -> pd.DataFrame:
    """Merge closure and add calendar CNY features.

    Adds: ``is_cny_window``, ``is_warehouse_closed``, ``days_to_cny``,
    ``days_from_cny`` (clipped to [-30, 30]).
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    c = closure.copy()
    c["date"] = pd.to_datetime(c["date"], errors="coerce").dt.normalize()
    out = out.merge(c, on=["warehouse", "date"], how="left")
    out["is_warehouse_closed"] = out["is_warehouse_closed"].fillna(0).astype(np.int8)

    d = out["date"]
    out["is_cny_window"] = (
        (d >= CNY_WINDOW_START) & (d <= CNY_WINDOW_END)
    ).astype(np.int8)

    delta_to = (CNY_DATE - d).dt.days.astype(float)
    delta_from = (d - CNY_DATE).dt.days.astype(float)
    out["days_to_cny"] = np.clip(delta_to, -30, 30).astype(np.int16)
    out["days_from_cny"] = np.clip(delta_from, -30, 30).astype(np.int16)
    return out


def mask_qty_for_features(df: pd.DataFrame, qty_col: str = "qty_ea") -> pd.Series:
    """Return ``qty_ea`` with NaN where ``is_warehouse_closed == 1``."""
    q = pd.to_numeric(df[qty_col], errors="coerce").astype(float)
    closed = df["is_warehouse_closed"].fillna(0).astype(int) == 1
    out = q.copy()
    out.loc[closed] = np.nan
    return out


def make_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """Per-row weights: closure -> 0, CNY window (open) -> 0.5, else -> 1."""
    closed = df["is_warehouse_closed"].fillna(0).astype(int) == 1
    in_win = df["is_cny_window"].fillna(0).astype(int) == 1
    w = np.ones(len(df), dtype=np.float64)
    w[closed] = 0.0
    w[in_win & ~closed] = 0.5
    return w
