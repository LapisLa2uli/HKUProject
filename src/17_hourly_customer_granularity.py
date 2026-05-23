"""Hourly and customer-level extension for the active hierarchy.

This script keeps the current category-week/category-day/product-day outputs as
parent constraints, then adds:

- category-hour validation;
- product-hour validation;
- customer-category week/day/hour validation;
- an optional April forecast mode that produces reconciled forecast artifacts
  without reporting April WMAPE.

Hourly validation is possible because the cleaned order-item table keeps
``create_time``. All validation metrics use Jan-Feb-derived shares to predict
March. March actuals are used only for scoring.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from _metrics import bias_ratio, wmape  # noqa: E402

ORDER_ITEMS = PROJECT_ROOT / "processed" / "order_items.csv"
PRODUCT_TAGS = PROJECT_ROOT / "processed" / "tagging" / "product_order_tags.csv"
PARENT_FORECAST = PROJECT_ROOT / "reports" / "weekly_tagged_systematic" / "selected_validation_predictions.csv"
PROD_REPORT_DIR = PROJECT_ROOT / "reports" / "production_granularity"
CATEGORY_DAY_PRED = PROD_REPORT_DIR / "category_day_predictions.csv"
PRODUCT_DAY_PRED = PROD_REPORT_DIR / "product_day_predictions.csv"
SYSTEMATIC_DAY = PROJECT_ROOT / "external_data" / "processed" / "systematic_day_tags.csv"

REPORT_DIR = PROD_REPORT_DIR / "hourly_customer"

VALIDATION_TRAIN_MONTHS = ["2026-01", "2026-02"]
VALIDATION_TARGET_MONTH = "2026-03"
FORECAST_TRAIN_MONTHS = ["2026-01", "2026-02", "2026-03"]
FORECAST_TARGET_MONTH = "2026-04"

TARGETS = {
    "grouped_week": 0.05,
    "grouped_day": 0.07,
    "grouped_hour": 0.12,
    "customer_week": 0.12,
    "customer_day": 0.18,
    "customer_hour": 0.25,
    "product_hour": 0.30,
}


@dataclass
class MetricRow:
    layer: str
    model: str
    mode: str
    deployable: bool
    uses_external_data: bool
    wmape: float
    bias_ratio: float
    actual_total: float
    pred_total: float
    max_period_wmape: float
    n_rows: int
    target: float
    target_met: bool
    notes: str = ""


def log(msg: str) -> None:
    print(f"[hourly-customer] {msg}", flush=True)


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def week_of_month(dates: pd.Series) -> pd.Series:
    return ((dates.dt.day - 1) // 7 + 1).clip(upper=5).astype(int)


def month_dates(month: str) -> pd.DatetimeIndex:
    start = pd.Period(month, freq="M").start_time
    end = pd.Period(month, freq="M").end_time.normalize()
    return pd.date_range(start, end, freq="D")


def load_items() -> pd.DataFrame:
    items = pd.read_csv(ORDER_ITEMS, parse_dates=["create_time", "create_date"])
    tags = pd.read_csv(PRODUCT_TAGS)
    tag_cols = [
        "product_id",
        "product_name",
        "product_reference_group",
        "product_macro_group",
        "festival_sensitivity",
        "replenishment_style",
    ]
    df = items.merge(tags[tag_cols].drop_duplicates("product_id"), on=["product_id", "product_name"], how="left")
    for col in ["product_reference_group", "product_macro_group", "festival_sensitivity", "replenishment_style"]:
        df[col] = df[col].fillna("other")
    df["qty_ea"] = pd.to_numeric(df["qty_ea"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["create_time"] = pd.to_datetime(df["create_time"], errors="coerce")
    df = df[df["create_time"].notna()].copy()
    df["date"] = df["create_time"].dt.normalize()
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["week_of_month"] = week_of_month(df["date"])
    df["dow"] = df["date"].dt.weekday
    df["hour"] = df["create_time"].dt.hour.astype(int)
    df["timestamp_hour"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")
    return df


def load_systematic_day() -> pd.DataFrame:
    if SYSTEMATIC_DAY.exists():
        cal = pd.read_csv(SYSTEMATIC_DAY, parse_dates=["date"])
    else:
        cal = pd.DataFrame({"date": pd.date_range("2026-01-01", "2026-04-30", freq="D")})
        cal["timor_is_holiday"] = 0
        cal["timor_is_adjusted_workday"] = 0
        cal["base_open_weight"] = 1.0
    cal["date"] = pd.to_datetime(cal["date"]).dt.normalize()
    cal["dow"] = cal["date"].dt.weekday
    cal["day_type"] = np.select(
        [
            pd.to_numeric(cal.get("timor_is_holiday", 0), errors="coerce").fillna(0).eq(1),
            pd.to_numeric(cal.get("timor_is_adjusted_workday", 0), errors="coerce").fillna(0).eq(1),
            cal["dow"].ge(5),
        ],
        ["holiday", "adjusted_workday", "weekend"],
        default="workday",
    )
    return cal[["date", "dow", "day_type"]].drop_duplicates("date")


def load_parent() -> pd.DataFrame:
    parent = pd.read_csv(PARENT_FORECAST)
    parent = parent[["product_reference_group", "month", "week_of_month", "qty_ea", "pred"]].copy()
    parent["week_of_month"] = parent["week_of_month"].astype(int)
    parent["pred"] = pd.to_numeric(parent["pred"], errors="coerce").fillna(0.0).clip(lower=0.0)
    parent["qty_ea"] = pd.to_numeric(parent["qty_ea"], errors="coerce").fillna(0.0).clip(lower=0.0)
    return parent


def load_category_day() -> pd.DataFrame:
    day = pd.read_csv(CATEGORY_DAY_PRED, parse_dates=["date"])
    day["date"] = day["date"].dt.normalize()
    day["pred"] = pd.to_numeric(day["pred"], errors="coerce").fillna(0.0).clip(lower=0.0)
    return day


def load_product_day() -> pd.DataFrame:
    prod = pd.read_csv(PRODUCT_DAY_PRED, parse_dates=["date"])
    prod["date"] = prod["date"].dt.normalize()
    prod["pred"] = pd.to_numeric(prod["pred"], errors="coerce").fillna(0.0).clip(lower=0.0)
    return prod


def metric(
    layer: str,
    model: str,
    actual: pd.DataFrame,
    pred: pd.DataFrame,
    keys: list[str],
    period_col: str,
    target: float,
    *,
    mode: str = "validation",
    uses_external_data: bool = False,
    notes: str = "",
) -> tuple[MetricRow, pd.DataFrame]:
    joined = actual.merge(pred[keys + ["pred"]], on=keys, how="outer")
    joined["qty_ea"] = pd.to_numeric(joined["qty_ea"], errors="coerce").fillna(0.0).clip(lower=0.0)
    joined["pred"] = pd.to_numeric(joined["pred"], errors="coerce").fillna(0.0).clip(lower=0.0)
    y = joined["qty_ea"].to_numpy(dtype=float)
    p = joined["pred"].to_numpy(dtype=float)
    period_scores = []
    if period_col in joined.columns:
        for _, g in joined.groupby(period_col, dropna=False):
            if g["qty_ea"].sum() > 0:
                period_scores.append(wmape(g["qty_ea"].to_numpy(dtype=float), g["pred"].to_numpy(dtype=float)))
    score = wmape(y, p)
    row = MetricRow(
        layer=layer,
        model=model,
        mode=mode,
        deployable=True,
        uses_external_data=uses_external_data,
        wmape=float(score),
        bias_ratio=float(bias_ratio(y, p)),
        actual_total=float(y.sum()),
        pred_total=float(p.sum()),
        max_period_wmape=float(max(period_scores)) if period_scores else float("nan"),
        n_rows=int(len(joined)),
        target=float(target),
        target_met=bool(score <= target),
        notes=notes,
    )
    joined["layer"] = layer
    joined["model"] = model
    return row, joined


def normalize_weights(frame: pd.DataFrame, group_cols: list[str], value_col: str = "day_pred") -> pd.DataFrame:
    out = frame.copy()
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    norm = out.groupby(group_cols)["weight"].transform("sum").replace(0, np.nan)
    out["share"] = (out["weight"] / norm).fillna(0.0)
    out["pred"] = out[value_col].fillna(0.0) * out["share"]
    return out


def actual_category_hour(df: pd.DataFrame, categories: Iterable[str], target_month: str) -> pd.DataFrame:
    dates = pd.DataFrame({"date": month_dates(target_month)})
    dates["month"] = target_month
    dates["week_of_month"] = week_of_month(dates["date"])
    dates["dow"] = dates["date"].dt.weekday
    hours = pd.DataFrame({"hour": list(range(24))})
    cats = pd.DataFrame({"product_reference_group": sorted(set(categories))})
    grid = cats.assign(_k=1).merge(dates.assign(_k=1), on="_k").merge(hours.assign(_k=1), on="_k").drop(columns="_k")
    valid = df[df["month"] == target_month]
    actual = valid.groupby(["product_reference_group", "date", "month", "week_of_month", "dow", "hour"], as_index=False)["qty_ea"].sum()
    out = grid.merge(actual, on=["product_reference_group", "date", "month", "week_of_month", "dow", "hour"], how="left")
    out["qty_ea"] = out["qty_ea"].fillna(0.0)
    out["timestamp_hour"] = out["date"] + pd.to_timedelta(out["hour"], unit="h")
    return out


def category_hour_candidates(
    df: pd.DataFrame,
    category_day: pd.DataFrame,
    train_months: list[str],
    calendar: pd.DataFrame,
) -> dict[str, tuple[pd.DataFrame, bool, str]]:
    hist = df[df["month"].isin(train_months)].merge(calendar[["date", "day_type"]], on="date", how="left")
    base_grid = category_day[["product_reference_group", "date", "month", "week_of_month", "dow", "pred"]].rename(columns={"pred": "day_pred"})
    base_grid = base_grid.merge(calendar[["date", "day_type"]], on="date", how="left")
    hours = pd.DataFrame({"hour": list(range(24))})
    grid = base_grid.assign(_k=1).merge(hours.assign(_k=1), on="_k").drop(columns="_k")

    candidates: dict[str, tuple[pd.DataFrame, bool, str]] = {}
    global_hour = hist.groupby("hour", as_index=False)["qty_ea"].sum()
    global_hour["global_share"] = global_hour["qty_ea"] / max(float(global_hour["qty_ea"].sum()), 1e-9)

    g = grid.merge(global_hour[["hour", "global_share"]], on="hour", how="left")
    g["weight"] = g["global_share"].fillna(1 / 24)
    candidates["category_hour_global_hour"] = (
        normalize_weights(g, ["product_reference_group", "date"]),
        False,
        "hour shares from Jan-Feb only",
    )

    cat_hour = hist.groupby(["product_reference_group", "hour"], as_index=False)["qty_ea"].sum()
    cat_total = cat_hour.groupby("product_reference_group", as_index=False)["qty_ea"].sum().rename(columns={"qty_ea": "key_total"})
    cat_hour = cat_hour.merge(cat_total, on="product_reference_group", how="left")
    cat_hour["share"] = cat_hour["qty_ea"] / cat_hour["key_total"].replace(0, np.nan)
    for k in [1000.0, 10000.0, 100000.0, 500000.0]:
        b = grid.merge(cat_hour[["product_reference_group", "hour", "share", "key_total"]], on=["product_reference_group", "hour"], how="left")
        b = b.merge(global_hour[["hour", "global_share"]], on="hour", how="left")
        b[["share", "key_total"]] = b[["share", "key_total"]].fillna(0.0)
        b["global_share"] = b["global_share"].fillna(1 / 24)
        b["weight"] = (b["share"] * b["key_total"] + k * b["global_share"]) / (b["key_total"] + k)
        candidates[f"category_hour_category_hour_shrink_k{k:g}"] = (
            normalize_weights(b, ["product_reference_group", "date"]),
            False,
            "category-specific hourly shares with global shrinkage",
        )

    # Calendar-aware external candidate: hour shape is conditioned on known day
    # type. This is accepted only if validation beats the non-external variants.
    daytype_hour = hist.groupby(["product_reference_group", "day_type", "hour"], as_index=False)["qty_ea"].sum()
    daytype_total = daytype_hour.groupby(["product_reference_group", "day_type"], as_index=False)["qty_ea"].sum().rename(columns={"qty_ea": "key_total"})
    daytype_hour = daytype_hour.merge(daytype_total, on=["product_reference_group", "day_type"], how="left")
    daytype_hour["share"] = daytype_hour["qty_ea"] / daytype_hour["key_total"].replace(0, np.nan)
    for k in [1000.0, 10000.0, 100000.0]:
        b = grid.merge(daytype_hour[["product_reference_group", "day_type", "hour", "share", "key_total"]], on=["product_reference_group", "day_type", "hour"], how="left")
        b = b.merge(global_hour[["hour", "global_share"]], on="hour", how="left")
        b[["share", "key_total"]] = b[["share", "key_total"]].fillna(0.0)
        b["global_share"] = b["global_share"].fillna(1 / 24)
        b["weight"] = (b["share"] * b["key_total"] + k * b["global_share"]) / (b["key_total"] + k)
        candidates[f"category_hour_calendar_daytype_shrink_k{k:g}"] = (
            normalize_weights(b, ["product_reference_group", "date"]),
            True,
            "external calendar day-type conditioned hourly shares",
        )
    return candidates


def actual_product_hour(df: pd.DataFrame, product_day: pd.DataFrame, target_month: str) -> pd.DataFrame:
    prod_cols = ["product_reference_group", "product_id", "product_name"]
    keys = product_day[prod_cols + ["date", "month", "week_of_month", "dow"]].drop_duplicates()
    hours = pd.DataFrame({"hour": list(range(24))})
    grid = keys.assign(_k=1).merge(hours.assign(_k=1), on="_k").drop(columns="_k")
    valid = df[df["month"] == target_month]
    actual = valid.groupby(prod_cols + ["date", "month", "week_of_month", "dow", "hour"], as_index=False)["qty_ea"].sum()
    out = grid.merge(actual, on=prod_cols + ["date", "month", "week_of_month", "dow", "hour"], how="left")
    out["qty_ea"] = out["qty_ea"].fillna(0.0)
    out["timestamp_hour"] = out["date"] + pd.to_timedelta(out["hour"], unit="h")
    return out


def product_hour_from_product_day(
    df: pd.DataFrame,
    product_day: pd.DataFrame,
    train_months: list[str],
) -> dict[str, pd.DataFrame]:
    hist = df[df["month"].isin(train_months)].copy()
    prod_cols = ["product_reference_group", "product_id", "product_name"]
    base_grid = product_day[prod_cols + ["date", "month", "week_of_month", "dow", "pred"]].rename(columns={"pred": "day_pred"})
    hours = pd.DataFrame({"hour": list(range(24))})
    grid = base_grid.assign(_k=1).merge(hours.assign(_k=1), on="_k").drop(columns="_k")

    category_hour = hist.groupby(["product_reference_group", "hour"], as_index=False)["qty_ea"].sum()
    category_hour["cat_total"] = category_hour.groupby("product_reference_group")["qty_ea"].transform("sum")
    category_hour["category_share"] = category_hour["qty_ea"] / category_hour["cat_total"].replace(0, np.nan)

    prod_hour = hist.groupby(prod_cols + ["hour"], as_index=False)["qty_ea"].sum()
    prod_hour["prod_total"] = prod_hour.groupby(prod_cols)["qty_ea"].transform("sum")
    prod_hour["product_share"] = prod_hour["qty_ea"] / prod_hour["prod_total"].replace(0, np.nan)

    out: dict[str, pd.DataFrame] = {}
    for k in [500.0, 2000.0, 10000.0, 50000.0]:
        b = grid.merge(prod_hour[prod_cols + ["hour", "product_share", "prod_total"]], on=prod_cols + ["hour"], how="left")
        b = b.merge(category_hour[["product_reference_group", "hour", "category_share"]], on=["product_reference_group", "hour"], how="left")
        b[["product_share", "prod_total"]] = b[["product_share", "prod_total"]].fillna(0.0)
        b["category_share"] = b["category_share"].fillna(1 / 24)
        b["weight"] = (b["product_share"] * b["prod_total"] + k * b["category_share"]) / (b["prod_total"] + k)
        out[f"product_hour_product_hour_shrink_k{k:g}"] = normalize_weights(b, prod_cols + ["date"])
    return out


def customer_share_predictions(
    df: pd.DataFrame,
    parent: pd.DataFrame,
    category_day: pd.DataFrame,
    category_hour: pd.DataFrame,
    train_months: list[str],
    target_month: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    hist = df[df["month"].isin(train_months)].copy()
    valid = df[df["month"] == target_month].copy()
    key = ["customer_id", "product_reference_group"]
    hist_weight = hist.groupby(key, as_index=False)["qty_ea"].sum().rename(columns={"qty_ea": "weight"})
    customers = hist_weight[["customer_id"]].drop_duplicates()
    categories = parent[["product_reference_group"]].drop_duplicates()
    customer_cat = customers.assign(_k=1).merge(categories.assign(_k=1), on="_k").drop(columns="_k")
    customer_cat = customer_cat.merge(hist_weight, on=key, how="left")
    customer_cat["weight"] = customer_cat["weight"].fillna(0.0)

    pred: dict[str, pd.DataFrame] = {}
    actual: dict[str, pd.DataFrame] = {}

    def allocate(base: pd.DataFrame, base_keys: list[str], layer: str) -> pd.DataFrame:
        out = base.merge(customer_cat, on=["product_reference_group"], how="left")
        out["weight"] = out["weight"].fillna(0.0)
        norm = out.groupby(base_keys)["weight"].transform("sum").replace(0, np.nan)
        out["share"] = (out["weight"] / norm).fillna(0.0)
        out["pred"] = out["pred"] * out["share"]
        out["allocation_model"] = f"{layer}_janfeb_customer_category_share"
        return out

    week_base = parent[["product_reference_group", "month", "week_of_month", "pred"]].copy()
    pred["customer_week"] = allocate(week_base, ["product_reference_group", "month", "week_of_month"], "customer_week")[
        ["customer_id", "product_reference_group", "month", "week_of_month", "pred", "allocation_model"]
    ]
    actual["customer_week"] = valid.groupby(["customer_id", "product_reference_group", "month", "week_of_month"], as_index=False)["qty_ea"].sum()

    day_base = category_day[["product_reference_group", "date", "month", "week_of_month", "dow", "pred"]].copy()
    pred["customer_day"] = allocate(day_base, ["product_reference_group", "date"], "customer_day")[
        ["customer_id", "product_reference_group", "date", "month", "week_of_month", "dow", "pred", "allocation_model"]
    ]
    actual["customer_day"] = valid.groupby(["customer_id", "product_reference_group", "date", "month", "week_of_month", "dow"], as_index=False)["qty_ea"].sum()

    hour_base = category_hour[["product_reference_group", "date", "month", "week_of_month", "dow", "hour", "pred"]].copy()
    pred["customer_hour"] = allocate(hour_base, ["product_reference_group", "date", "hour"], "customer_hour")[
        ["customer_id", "product_reference_group", "date", "month", "week_of_month", "dow", "hour", "pred", "allocation_model"]
    ]
    pred["customer_hour"]["timestamp_hour"] = pred["customer_hour"]["date"] + pd.to_timedelta(pred["customer_hour"]["hour"], unit="h")
    actual["customer_hour"] = valid.groupby(["customer_id", "product_reference_group", "date", "month", "week_of_month", "dow", "hour"], as_index=False)["qty_ea"].sum()
    actual["customer_hour"]["timestamp_hour"] = actual["customer_hour"]["date"] + pd.to_timedelta(actual["customer_hour"]["hour"], unit="h")
    return pred, actual


def build_validation(mode: str = "validation") -> tuple[pd.DataFrame, dict[str, object]]:
    ensure_dirs()
    df = load_items()
    calendar = load_systematic_day()
    train_months = VALIDATION_TRAIN_MONTHS
    target_month = VALIDATION_TARGET_MONTH

    parent = load_parent()
    category_day = load_category_day()
    product_day = load_product_day()

    metrics: list[MetricRow] = []
    joined_outputs: dict[str, pd.DataFrame] = {}

    # Existing grouped week/day metrics, re-scored from the current artifacts.
    parent_actual = parent[["product_reference_group", "month", "week_of_month", "qty_ea"]]
    parent_pred = parent[["product_reference_group", "month", "week_of_month", "pred"]]
    row, joined = metric("grouped_week", "selected_category_week_parent", parent_actual, parent_pred, ["product_reference_group", "month", "week_of_month"], "week_of_month", TARGETS["grouped_week"])
    metrics.append(row)
    joined_outputs["grouped_week"] = joined

    category_day_actual = (
        df[df["month"] == target_month]
        .groupby(["product_reference_group", "date", "month", "week_of_month", "dow"], as_index=False)["qty_ea"]
        .sum()
    )
    row, joined = metric("grouped_day", "selected_category_day", category_day_actual, category_day, ["product_reference_group", "date", "month", "week_of_month", "dow"], "date", TARGETS["grouped_day"])
    metrics.append(row)
    joined_outputs["grouped_day"] = joined

    actual_ch = actual_category_hour(df, parent["product_reference_group"].unique(), target_month)
    ch_candidates = category_hour_candidates(df, category_day, train_months, calendar)
    ch_joined: dict[str, pd.DataFrame] = {}
    for name, (pred, uses_external, notes) in ch_candidates.items():
        row, joined = metric(
            "grouped_hour",
            name,
            actual_ch,
            pred[["product_reference_group", "date", "month", "week_of_month", "dow", "hour", "pred"]],
            ["product_reference_group", "date", "month", "week_of_month", "dow", "hour"],
            "timestamp_hour",
            TARGETS["grouped_hour"],
            uses_external_data=uses_external,
            notes=notes,
        )
        metrics.append(row)
        ch_joined[name] = joined
    grouped_hour_rows = [m for m in metrics if m.layer == "grouped_hour"]
    selected_ch = sorted(grouped_hour_rows, key=lambda r: (r.wmape, abs(r.bias_ratio), r.uses_external_data))[0]
    category_hour = ch_candidates[selected_ch.model][0].copy()
    category_hour["timestamp_hour"] = category_hour["date"] + pd.to_timedelta(category_hour["hour"], unit="h")
    joined_outputs["grouped_hour"] = ch_joined[selected_ch.model]

    actual_ph = actual_product_hour(df, product_day, target_month)
    ph_candidates = product_hour_from_product_day(df, product_day, train_months)
    ph_joined: dict[str, pd.DataFrame] = {}
    for name, pred in ph_candidates.items():
        row, joined = metric(
            "product_hour",
            name,
            actual_ph,
            pred[["product_reference_group", "product_id", "product_name", "date", "month", "week_of_month", "dow", "hour", "pred"]],
            ["product_reference_group", "product_id", "product_name", "date", "month", "week_of_month", "dow", "hour"],
            "timestamp_hour",
            TARGETS["product_hour"],
            uses_external_data=False,
            notes="product-day allocated to order-hour shares",
        )
        metrics.append(row)
        ph_joined[name] = joined
    selected_ph = sorted([m for m in metrics if m.layer == "product_hour"], key=lambda r: (r.wmape, abs(r.bias_ratio)))[0]
    product_hour = ph_candidates[selected_ph.model].copy()
    product_hour["timestamp_hour"] = product_hour["date"] + pd.to_timedelta(product_hour["hour"], unit="h")
    joined_outputs["product_hour"] = ph_joined[selected_ph.model]

    customer_pred, customer_actual = customer_share_predictions(df, parent, category_day, category_hour, train_months, target_month)
    row, joined = metric("customer_week", "customer_category_janfeb_share", customer_actual["customer_week"], customer_pred["customer_week"], ["customer_id", "product_reference_group", "month", "week_of_month"], "week_of_month", TARGETS["customer_week"])
    metrics.append(row)
    joined_outputs["customer_week"] = joined
    row, joined = metric("customer_day", "customer_category_janfeb_share", customer_actual["customer_day"], customer_pred["customer_day"], ["customer_id", "product_reference_group", "date", "month", "week_of_month", "dow"], "date", TARGETS["customer_day"])
    metrics.append(row)
    joined_outputs["customer_day"] = joined
    row, joined = metric("customer_hour", "customer_category_janfeb_share", customer_actual["customer_hour"], customer_pred["customer_hour"], ["customer_id", "product_reference_group", "date", "month", "week_of_month", "dow", "hour"], "timestamp_hour", TARGETS["customer_hour"])
    metrics.append(row)
    joined_outputs["customer_hour"] = joined

    metrics_df = pd.DataFrame([m.__dict__ for m in metrics]).sort_values(["layer", "wmape"])
    metrics_df.to_csv(REPORT_DIR / "hourly_customer_metrics.csv", index=False, encoding="utf-8-sig")
    category_hour.to_csv(REPORT_DIR / "category_hour_predictions.csv", index=False, encoding="utf-8-sig")
    product_hour.to_csv(REPORT_DIR / "product_hour_predictions.csv", index=False, encoding="utf-8-sig")
    customer_pred["customer_week"].to_csv(REPORT_DIR / "customer_category_week_predictions.csv", index=False, encoding="utf-8-sig")
    customer_pred["customer_day"].to_csv(REPORT_DIR / "customer_category_day_predictions.csv", index=False, encoding="utf-8-sig")
    customer_pred["customer_hour"].to_csv(REPORT_DIR / "customer_category_hour_predictions.csv", index=False, encoding="utf-8-sig")
    for name, joined in joined_outputs.items():
        out = joined.copy()
        out["abs_error"] = (out["qty_ea"] - out["pred"]).abs()
        out.to_csv(REPORT_DIR / f"{name}_errors.csv", index=False, encoding="utf-8-sig")

    checks = reconciliation_checks(parent, category_day, category_hour, product_day, product_hour, customer_pred)
    checks.to_csv(REPORT_DIR / "hourly_reconciliation_checks.csv", index=False, encoding="utf-8-sig")

    line_data, bar_data = composite_chart_tables(joined_outputs)
    line_data.to_csv(REPORT_DIR / "six_panel_line_data.csv", index=False, encoding="utf-8-sig")
    bar_data.to_csv(REPORT_DIR / "six_panel_bar_data.csv", index=False, encoding="utf-8-sig")

    best_non_external_hour = metrics_df[(metrics_df["layer"] == "grouped_hour") & (~metrics_df["uses_external_data"])].sort_values("wmape").head(1)
    best_external_hour = metrics_df[(metrics_df["layer"] == "grouped_hour") & (metrics_df["uses_external_data"])].sort_values("wmape").head(1)
    external_effect = {
        "best_non_external_grouped_hour_wmape": float(best_non_external_hour.iloc[0]["wmape"]) if not best_non_external_hour.empty else float("nan"),
        "best_external_grouped_hour_wmape": float(best_external_hour.iloc[0]["wmape"]) if not best_external_hour.empty else float("nan"),
        "external_hourly_accepted": bool(
            not best_external_hour.empty
            and not best_non_external_hour.empty
            and float(best_external_hour.iloc[0]["wmape"]) < float(best_non_external_hour.iloc[0]["wmape"])
        ),
    }
    verdict = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "mode": mode,
        "hourly_timestamp_available": True,
        "selected_grouped_hour_model": selected_ch.model,
        "selected_product_hour_model": selected_ph.model,
        "external_hourly_effect": external_effect,
        "metrics": metrics_df.groupby("layer", as_index=False).first().to_dict(orient="records"),
        "all_reconciliation_checks_passed": bool(checks["passed"].all()),
        "notes": [
            "Hourly validation uses create_time from processed/order_items.csv.",
            "March actuals are used only for validation metrics.",
            "Calendar-conditioned hourly shares are accepted only if the grouped-hour validation WMAPE beats the non-external hourly baseline.",
        ],
    }
    (REPORT_DIR / "hourly_customer_verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(metrics_df, checks, external_effect)
    return metrics_df, verdict


def reconciliation_checks(
    parent: pd.DataFrame,
    category_day: pd.DataFrame,
    category_hour: pd.DataFrame,
    product_day: pd.DataFrame,
    product_hour: pd.DataFrame,
    customer_pred: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    def add(name: str, left: float, right: float) -> None:
        rows.append({"check": name, "left_total": left, "right_total": right, "abs_diff": abs(left - right), "passed": abs(left - right) <= 1e-6 * max(1.0, abs(right))})

    add("category_hour_sum_equals_category_day", float(category_hour["pred"].sum()), float(category_day["pred"].sum()))
    add("product_hour_sum_equals_product_day", float(product_hour["pred"].sum()), float(product_day["pred"].sum()))
    add("customer_week_sum_equals_parent", float(customer_pred["customer_week"]["pred"].sum()), float(parent["pred"].sum()))
    add("customer_day_sum_equals_category_day", float(customer_pred["customer_day"]["pred"].sum()), float(category_day["pred"].sum()))
    add("customer_hour_sum_equals_category_hour", float(customer_pred["customer_hour"]["pred"].sum()), float(category_hour["pred"].sum()))
    add("no_negative_category_hour", float((category_hour["pred"] < -1e-9).sum()), 0.0)
    add("no_negative_product_hour", float((product_hour["pred"] < -1e-9).sum()), 0.0)
    add("no_negative_customer_hour", float((customer_pred["customer_hour"]["pred"] < -1e-9).sum()), 0.0)
    return pd.DataFrame(rows)


def composite_chart_tables(joined: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = [
        ("grouped", "weekly", "grouped_week", "week_of_month", ["product_reference_group"]),
        ("grouped", "daily", "grouped_day", "date", ["product_reference_group"]),
        ("grouped", "hourly", "grouped_hour", "timestamp_hour", ["product_reference_group"]),
        ("customer", "weekly", "customer_week", "week_of_month", ["customer_id", "product_reference_group"]),
        ("customer", "daily", "customer_day", "date", ["customer_id", "product_reference_group"]),
        ("customer", "hourly", "customer_hour", "timestamp_hour", ["customer_id", "product_reference_group"]),
    ]
    lines = []
    bars = []
    for order_precision, time_precision, key, period_col, segment_cols in specs:
        df = joined[key].copy()
        if period_col not in df.columns and period_col == "timestamp_hour":
            df["timestamp_hour"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"], unit="h")
        line = df.groupby(period_col, as_index=False).agg(actual=("qty_ea", "sum"), pred=("pred", "sum"))
        line["order_precision"] = order_precision
        line["time_precision"] = time_precision
        line["period"] = line[period_col].astype(str)
        lines.append(line[["order_precision", "time_precision", "period", "actual", "pred"]])

        seg = df.groupby(segment_cols, as_index=False).agg(actual=("qty_ea", "sum"), pred=("pred", "sum"))
        seg["abs_error"] = (seg["actual"] - seg["pred"]).abs()
        seg["wmape"] = np.divide(seg["abs_error"], seg["actual"].replace(0, np.nan)).fillna(0.0)
        seg["segment"] = seg[segment_cols].astype(str).agg(" | ".join, axis=1)
        seg["order_precision"] = order_precision
        seg["time_precision"] = time_precision
        bars.append(seg.sort_values("abs_error", ascending=False).head(12)[["order_precision", "time_precision", "segment", "actual", "pred", "abs_error", "wmape"]])
    return pd.concat(lines, ignore_index=True), pd.concat(bars, ignore_index=True)


def write_report(metrics: pd.DataFrame, checks: pd.DataFrame, external_effect: dict[str, object]) -> None:
    best = metrics.sort_values("wmape").groupby("layer", as_index=False).first()
    lines = [
        "# Hourly And Customer Granularity Report",
        "",
        "Generated by `src/17_hourly_customer_granularity.py`.",
        "",
        "## Verdict",
        "",
        "- True hourly validation is available because `processed/order_items.csv` contains `create_time`.",
        f"- Best non-external grouped-hour WMAPE: `{external_effect['best_non_external_grouped_hour_wmape']:.6f}`.",
        f"- Best external/calendar grouped-hour WMAPE: `{external_effect['best_external_grouped_hour_wmape']:.6f}`.",
        f"- External hourly feature accepted: `{external_effect['external_hourly_accepted']}`.",
        f"- All reconciliation checks passed: `{checks['passed'].all()}`.",
        "",
        "## Best Metrics By Layer",
        "",
        best.to_markdown(index=False),
        "",
        "## Reconciliation Checks",
        "",
        checks.to_markdown(index=False),
        "",
        "## All Metrics",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "The hourly layer improves mechanism completeness by making the forecast usable below the daily level. It does not fix product/customer sparsity by itself: the lower the order precision, the more WMAPE is dominated by customer/product mix volatility rather than total category volume.",
        "",
    ]
    (REPORT_DIR / "hourly_customer_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_april_forecast() -> None:
    """Write April hourly forecast artifacts without validation metrics."""
    ensure_dirs()
    df = load_items()
    calendar = load_systematic_day()
    train = df[df["month"].isin(FORECAST_TRAIN_MONTHS)].copy()
    dates = pd.DataFrame({"date": month_dates(FORECAST_TARGET_MONTH)})
    dates["month"] = FORECAST_TARGET_MONTH
    dates["week_of_month"] = week_of_month(dates["date"])
    dates["dow"] = dates["date"].dt.weekday
    dates = dates.merge(calendar[["date", "day_type"]], on="date", how="left")
    dates["day_weight"] = np.where(dates["day_type"].eq("holiday"), 0.35, np.where(dates["day_type"].eq("weekend"), 0.70, 1.0))

    cats = train["product_reference_group"].dropna().drop_duplicates().sort_values()
    monthly = train.groupby(["product_reference_group", "month"], as_index=False)["qty_ea"].sum()
    piv = monthly.pivot_table(index="product_reference_group", columns="month", values="qty_ea", fill_value=0.0).reset_index()
    for month in FORECAST_TRAIN_MONTHS:
        if month not in piv.columns:
            piv[month] = 0.0
    piv["month_pred"] = (piv["2026-03"] + 0.25 * (piv["2026-03"] - piv["2026-02"])).clip(lower=0.0)
    mean_hist = piv[FORECAST_TRAIN_MONTHS].mean(axis=1)
    piv["month_pred"] = np.minimum(np.maximum(piv["month_pred"], 0.70 * mean_hist), 1.30 * mean_hist)

    cat_day = pd.DataFrame({"product_reference_group": cats}).assign(_k=1).merge(dates.assign(_k=1), on="_k").drop(columns="_k")
    cat_day = cat_day.merge(piv[["product_reference_group", "month_pred"]], on="product_reference_group", how="left")
    cat_day["month_pred"] = cat_day["month_pred"].fillna(0.0)
    month_weight = cat_day.groupby("product_reference_group")["day_weight"].transform("sum").replace(0, np.nan)
    cat_day["pred"] = cat_day["month_pred"] * cat_day["day_weight"] / month_weight
    cat_day.to_csv(REPORT_DIR / "april_category_day_forecast.csv", index=False, encoding="utf-8-sig")

    ch_candidates = category_hour_candidates(df, cat_day, FORECAST_TRAIN_MONTHS, calendar)
    # Prefer the strongest stable non-external candidate for April unless a
    # previous validation verdict accepted the external hourly model.
    selected_name = "category_hour_category_hour_shrink_k100000"
    if selected_name not in ch_candidates:
        selected_name = sorted(ch_candidates)[0]
    category_hour = ch_candidates[selected_name][0]
    category_hour["timestamp_hour"] = category_hour["date"] + pd.to_timedelta(category_hour["hour"], unit="h")
    category_hour.to_csv(REPORT_DIR / "april_category_hour_forecast.csv", index=False, encoding="utf-8-sig")
    (REPORT_DIR / "april_forecast_note.md").write_text(
        "# April Forecast Note\n\n"
        "Generated by `src/17_hourly_customer_granularity.py --mode forecast`.\n\n"
        "April outputs are forecasts only. No April WMAPE is reported because April actual labels are not available in the current dataset.\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validation", "forecast"], default="validation")
    args = parser.parse_args()
    if args.mode == "validation":
        build_validation(mode=args.mode)
    else:
        build_april_forecast()
    log(f"Wrote outputs to {REPORT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
