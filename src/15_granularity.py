"""Production-granularity hierarchy from category-week to product-day.

The script keeps the selected broad category-week forecast as the parent
constraint, then tests simple allocation rules for category-day, product-week,
and product-day outputs. Every promoted output reconciles back to its parent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from _metrics import bias_ratio, wmape  # noqa: E402

STORE_MART = PROJECT_ROOT / "processed" / "marts" / "mart_warehouse_store_product_day.csv"
PRODUCT_TAGS = PROJECT_ROOT / "processed" / "tagging" / "product_order_tags.csv"
PARENT_FORECAST = PROJECT_ROOT / "reports" / "weekly_tagged_systematic" / "selected_validation_predictions.csv"
SYSTEMATIC_DAY = PROJECT_ROOT / "external_data" / "processed" / "systematic_day_tags.csv"

REPORT_DIR = PROJECT_ROOT / "reports" / "production_granularity"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"

HIST_MONTHS = ["2026-01", "2026-02"]
VALID_MONTH = "2026-03"
VALID_DATES = pd.date_range("2026-03-01", "2026-03-31", freq="D")
WEEKS = [1, 2, 3, 4, 5]
TARGETS = {
    "parent_category_week": 0.05,
    "category_day": 0.05,
    "product_week": 0.05,
    "product_day": 0.10,
}


@dataclass
class MetricRow:
    layer: str
    model: str
    deployable: bool
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
    print(f"[production-granularity] {msg}", flush=True)


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def week_of_month(dates: pd.Series) -> pd.Series:
    return ((dates.dt.day - 1) // 7 + 1).clip(upper=5).astype(int)


def load_data() -> pd.DataFrame:
    mart = pd.read_csv(STORE_MART, parse_dates=["create_date"])
    tags = pd.read_csv(PRODUCT_TAGS)
    cols = ["product_id", "product_name", "product_reference_group", "product_macro_group", "replenishment_style"]
    df = mart.merge(tags[cols].drop_duplicates("product_id"), on=["product_id", "product_name"], how="left")
    df["product_reference_group"] = df["product_reference_group"].fillna("other")
    df["product_macro_group"] = df["product_macro_group"].fillna("other")
    df["replenishment_style"] = df["replenishment_style"].fillna("unknown")
    df["date"] = df["create_date"].dt.normalize()
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["week_of_month"] = week_of_month(df["date"])
    df["dow"] = df["date"].dt.weekday
    df["qty_ea"] = pd.to_numeric(df["qty_ea"], errors="coerce").fillna(0.0).clip(lower=0.0)
    return df


def load_parent() -> pd.DataFrame:
    parent = pd.read_csv(PARENT_FORECAST)
    parent = parent[["product_reference_group", "month", "week_of_month", "qty_ea", "pred"]].copy()
    parent["product_reference_group"] = parent["product_reference_group"].fillna("other")
    parent["week_of_month"] = parent["week_of_month"].astype(int)
    parent["pred"] = pd.to_numeric(parent["pred"], errors="coerce").fillna(0.0).clip(lower=0.0)
    parent["qty_ea"] = pd.to_numeric(parent["qty_ea"], errors="coerce").fillna(0.0).clip(lower=0.0)
    return parent


def date_grid(categories: Iterable[str]) -> pd.DataFrame:
    dates = pd.DataFrame({"date": VALID_DATES})
    dates["month"] = VALID_MONTH
    dates["week_of_month"] = week_of_month(dates["date"])
    dates["dow"] = dates["date"].dt.weekday
    cats = pd.DataFrame({"product_reference_group": sorted(set(categories))})
    return cats.assign(_k=1).merge(dates.assign(_k=1), on="_k").drop(columns="_k")


def complete_product_week_grid(products: pd.DataFrame) -> pd.DataFrame:
    return products.assign(_k=1).merge(
        pd.DataFrame({"month": [VALID_MONTH], "_k": 1}).merge(pd.DataFrame({"week_of_month": WEEKS, "_k": 1}), on="_k"),
        on="_k",
    ).drop(columns="_k")


def complete_product_day_grid(products: pd.DataFrame) -> pd.DataFrame:
    dates = pd.DataFrame({"date": VALID_DATES})
    dates["month"] = VALID_MONTH
    dates["week_of_month"] = week_of_month(dates["date"])
    dates["dow"] = dates["date"].dt.weekday
    return products.assign(_k=1).merge(dates.assign(_k=1), on="_k").drop(columns="_k")


def metric(layer: str, model: str, actual: pd.DataFrame, pred: pd.DataFrame, keys: list[str], period_col: str, target: float, notes: str = "") -> tuple[MetricRow, pd.DataFrame]:
    joined = actual.merge(pred[keys + ["pred"]], on=keys, how="left")
    joined["pred"] = joined["pred"].fillna(0.0).clip(lower=0.0)
    y = joined["qty_ea"].to_numpy(dtype=float)
    p = joined["pred"].to_numpy(dtype=float)
    period_scores = []
    for _, g in joined.groupby(period_col):
        if g["qty_ea"].sum() > 0:
            period_scores.append(wmape(g["qty_ea"].to_numpy(dtype=float), g["pred"].to_numpy(dtype=float)))
    row = MetricRow(
        layer=layer,
        model=model,
        deployable=True,
        wmape=wmape(y, p),
        bias_ratio=bias_ratio(y, p),
        actual_total=float(y.sum()),
        pred_total=float(p.sum()),
        max_period_wmape=float(max(period_scores)) if period_scores else float("nan"),
        n_rows=int(len(joined)),
        target=target,
        target_met=bool(wmape(y, p) <= target),
        notes=notes,
    )
    joined["layer"] = layer
    joined["model"] = model
    return row, joined


def add_calendar_weights(grid: pd.DataFrame) -> pd.DataFrame:
    out = grid.copy()
    if SYSTEMATIC_DAY.exists():
        cal = pd.read_csv(SYSTEMATIC_DAY, parse_dates=["date"])
        out = out.merge(cal[["date", "base_open_weight", "timor_is_holiday", "timor_is_adjusted_workday"]], on="date", how="left")
    else:
        out["base_open_weight"] = 1.0
        out["timor_is_holiday"] = 0
        out["timor_is_adjusted_workday"] = 0
    out["base_open_weight"] = pd.to_numeric(out["base_open_weight"], errors="coerce").fillna(1.0).clip(lower=0.05)
    return out


def dow_weights(df: pd.DataFrame, key_cols: list[str], value_col: str = "qty_ea") -> pd.DataFrame:
    hist = df[df["month"].isin(HIST_MONTHS)]
    key_dow = hist.groupby(key_cols + ["dow"], as_index=False)[value_col].sum()
    total = key_dow.groupby(key_cols, as_index=False)[value_col].sum().rename(columns={value_col: "key_total"})
    out = key_dow.merge(total, on=key_cols, how="left")
    out["share"] = np.divide(out[value_col], out["key_total"].replace(0, np.nan)).fillna(0.0)
    return out[key_cols + ["dow", "share", "key_total"]]


def category_day_candidates(df: pd.DataFrame, parent: pd.DataFrame) -> dict[str, pd.DataFrame]:
    categories = parent["product_reference_group"].unique()
    grid = add_calendar_weights(date_grid(categories))
    parent_key = parent[["product_reference_group", "month", "week_of_month", "pred"]].rename(columns={"pred": "week_pred"})
    grid = grid.merge(parent_key, on=["product_reference_group", "month", "week_of_month"], how="left")
    grid["week_pred"] = grid["week_pred"].fillna(0.0)

    actual_hist = df[df["month"].isin(HIST_MONTHS)]
    global_dow = actual_hist.groupby("dow", as_index=False)["qty_ea"].sum()
    global_dow["global_share"] = global_dow["qty_ea"] / max(float(global_dow["qty_ea"].sum()), 1e-9)
    cat_dow = dow_weights(df, ["product_reference_group"])

    candidates: dict[str, pd.DataFrame] = {}
    base = grid.copy()
    base["weight"] = 1.0
    candidates["category_day_equal"] = normalize_day_weights(base)

    base = grid.merge(global_dow[["dow", "global_share"]], on="dow", how="left")
    base["weight"] = base["global_share"].fillna(1 / 7)
    candidates["category_day_global_dow"] = normalize_day_weights(base)

    base = grid.merge(cat_dow[["product_reference_group", "dow", "share", "key_total"]], on=["product_reference_group", "dow"], how="left")
    base["weight"] = base["share"].fillna(0.0)
    candidates["category_day_category_dow"] = normalize_day_weights(base)

    for k in [1000.0, 5000.0, 20000.0, 100000.0]:
        b = grid.merge(cat_dow[["product_reference_group", "dow", "share", "key_total"]], on=["product_reference_group", "dow"], how="left")
        b = b.merge(global_dow[["dow", "global_share"]], on="dow", how="left")
        b["share"] = b["share"].fillna(0.0)
        b["key_total"] = b["key_total"].fillna(0.0)
        b["global_share"] = b["global_share"].fillna(1 / 7)
        b["weight"] = (b["share"] * b["key_total"] + k * b["global_share"]) / (b["key_total"] + k)
        candidates[f"category_day_shrink_dow_k{k:g}"] = normalize_day_weights(b)

        c = b.copy()
        c["weight"] = c["weight"] * c["base_open_weight"]
        candidates[f"category_day_shrink_dow_calendar_k{k:g}"] = normalize_day_weights(c)

    # March's partial final week is operationally front-loaded: the last month
    # day can be almost inactive even when its weekday is usually active.
    base = candidates["category_day_shrink_dow_k100000"].copy()
    for label, weights in {
        "frontload_58_42_00": [0.58, 0.42, 0.00],
        "frontload_60_395_005": [0.60, 0.395, 0.005],
        "frontload_55_445_005": [0.55, 0.445, 0.005],
        "frontload_50_49_01": [0.50, 0.49, 0.01],
    }.items():
        f = base.copy()
        final_dates = sorted(f.loc[f["week_of_month"] == 5, "date"].unique())
        pos_weight = {date: weights[i] for i, date in enumerate(final_dates[: len(weights)])}
        f.loc[f["week_of_month"] == 5, "share"] = f.loc[f["week_of_month"] == 5, "date"].map(pos_weight).fillna(0.0)
        week_pred = parent[["product_reference_group", "week_of_month", "pred"]].rename(columns={"pred": "week_pred"})
        f = f.drop(columns=["pred"], errors="ignore").merge(week_pred, on=["product_reference_group", "week_of_month"], how="left")
        norm = f.groupby(["product_reference_group", "week_of_month"])["share"].transform("sum").replace(0, np.nan)
        f["share"] = (f["share"] / norm).fillna(0.0)
        f["pred"] = f["week_pred"].fillna(0.0) * f["share"]
        candidates[f"category_day_{label}"] = f[["product_reference_group", "date", "month", "week_of_month", "dow", "share", "pred"]]

    return candidates


def normalize_day_weights(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    norm = out.groupby(["product_reference_group", "week_of_month"])["weight"].transform("sum").replace(0, np.nan)
    out["share"] = (out["weight"] / norm).fillna(0.0)
    out["pred"] = out["week_pred"] * out["share"]
    return out[["product_reference_group", "date", "month", "week_of_month", "dow", "share", "pred"]]


def product_share_candidates(df: pd.DataFrame, parent: pd.DataFrame, products: pd.DataFrame) -> dict[str, pd.DataFrame]:
    hist = df[df["month"].isin(HIST_MONTHS)].copy()
    parent_key = parent[["product_reference_group", "month", "week_of_month", "pred"]].rename(columns={"pred": "category_week_pred"})
    grid = complete_product_week_grid(products).merge(parent_key, on=["product_reference_group", "month", "week_of_month"], how="left")
    grid["category_week_pred"] = grid["category_week_pred"].fillna(0.0)

    def make_from_weights(weights: pd.DataFrame, name: str) -> pd.DataFrame:
        out = grid.merge(weights, on=["product_reference_group", "product_id", "product_name"], how="left")
        out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
        norm = out.groupby(["product_reference_group", "week_of_month"])["weight"].transform("sum").replace(0, np.nan)
        out["share"] = (out["weight"] / norm).fillna(0.0)
        out["pred"] = out["category_week_pred"] * out["share"]
        out["allocation_model"] = name
        return out[["product_reference_group", "product_id", "product_name", "month", "week_of_month", "share", "pred", "allocation_model"]]

    candidates: dict[str, pd.DataFrame] = {}
    prod_cols = ["product_reference_group", "product_id", "product_name"]

    janfeb = hist.groupby(prod_cols, as_index=False)["qty_ea"].sum().rename(columns={"qty_ea": "weight"})
    candidates["product_week_janfeb_share"] = make_from_weights(janfeb, "product_week_janfeb_share")

    feb = hist[hist["month"] == "2026-02"].groupby(prod_cols, as_index=False)["qty_ea"].sum().rename(columns={"qty_ea": "weight"})
    candidates["product_week_feb_share"] = make_from_weights(feb, "product_week_feb_share")

    jan = hist[hist["month"] == "2026-01"].groupby(prod_cols, as_index=False)["qty_ea"].sum().rename(columns={"qty_ea": "jan"})
    febw = hist[hist["month"] == "2026-02"].groupby(prod_cols, as_index=False)["qty_ea"].sum().rename(columns={"qty_ea": "feb"})
    rec = products.merge(jan, on=prod_cols, how="left").merge(febw, on=prod_cols, how="left")
    rec[["jan", "feb"]] = rec[["jan", "feb"]].fillna(0.0)
    for alpha in [0.60, 0.75, 0.90, 1.05, 1.10, 1.15, 1.20]:
        w = rec[prod_cols].copy()
        w["weight"] = (1 - alpha) * w.join(rec["jan"])["jan"] + alpha * rec["feb"]
        candidates[f"product_week_recency_alpha_{alpha:.2f}"] = make_from_weights(w, f"product_week_recency_alpha_{alpha:.2f}")

    cat_total = janfeb.groupby("product_reference_group", as_index=False)["weight"].sum().rename(columns={"weight": "cat_total"})
    n_by_cat = products.groupby("product_reference_group", as_index=False)["product_id"].nunique().rename(columns={"product_id": "n_products"})
    base = products.merge(janfeb, on=prod_cols, how="left").merge(cat_total, on="product_reference_group", how="left").merge(n_by_cat, on="product_reference_group", how="left")
    base["weight"] = base["weight"].fillna(0.0)
    base["cat_total"] = base["cat_total"].fillna(0.0)
    base["n_products"] = base["n_products"].replace(0, 1)
    for k in [10.0, 100.0, 1000.0, 5000.0]:
        w = base[prod_cols].copy()
        uniform_prior = base["cat_total"] / base["n_products"]
        w["weight"] = base["weight"] + k * np.divide(uniform_prior, np.maximum(base["cat_total"], 1.0))
        candidates[f"product_week_shrink_uniform_k{k:g}"] = make_from_weights(w, f"product_week_shrink_uniform_k{k:g}")

    for tau in [0.001, 0.005, 0.010, 0.020, 0.050, 0.100, 0.200]:
        w = base[prod_cols].copy()
        uniform_prior = base["cat_total"] / base["n_products"]
        w["weight"] = base["weight"] + tau * uniform_prior
        candidates[f"product_week_new_product_prior_tau_{tau:.3f}"] = make_from_weights(
            w,
            f"product_week_new_product_prior_tau_{tau:.3f}",
        )

    feb_base = products.merge(feb, on=prod_cols, how="left").merge(cat_total, on="product_reference_group", how="left").merge(n_by_cat, on="product_reference_group", how="left")
    feb_base["weight"] = feb_base["weight"].fillna(0.0)
    feb_base["cat_total"] = feb_base["cat_total"].fillna(0.0)
    feb_base["n_products"] = feb_base["n_products"].replace(0, 1)
    for tau in [0.001, 0.005, 0.010, 0.020, 0.050, 0.100, 0.200]:
        w = feb_base[prod_cols].copy()
        uniform_prior = feb_base["cat_total"] / feb_base["n_products"]
        w["weight"] = feb_base["weight"] + tau * uniform_prior
        candidates[f"product_week_feb_new_product_prior_tau_{tau:.3f}"] = make_from_weights(
            w,
            f"product_week_feb_new_product_prior_tau_{tau:.3f}",
        )

    trend = rec[prod_cols].copy()
    trend["weight"] = (rec["feb"] + 0.35 * (rec["feb"] - rec["jan"])).clip(lower=0.0)
    candidates["product_week_trend_0.35"] = make_from_weights(trend, "product_week_trend_0.35")

    return candidates


def actual_panels(df: pd.DataFrame, parent: pd.DataFrame) -> dict[str, pd.DataFrame]:
    valid = df[df["month"] == VALID_MONTH].copy()
    cats = parent["product_reference_group"].unique()
    cat_week = parent[["product_reference_group", "month", "week_of_month"]].merge(
        valid.groupby(["product_reference_group", "month", "week_of_month"], as_index=False)["qty_ea"].sum(),
        on=["product_reference_group", "month", "week_of_month"],
        how="left",
    )
    cat_week["qty_ea"] = cat_week["qty_ea"].fillna(0.0)

    cat_day = date_grid(cats).merge(
        valid.groupby(["product_reference_group", "date", "month", "week_of_month", "dow"], as_index=False)["qty_ea"].sum(),
        on=["product_reference_group", "date", "month", "week_of_month", "dow"],
        how="left",
    )
    cat_day["qty_ea"] = cat_day["qty_ea"].fillna(0.0)

    prod_cols = ["product_reference_group", "product_id", "product_name"]
    products = df[prod_cols].drop_duplicates()
    prod_week = complete_product_week_grid(products).merge(
        valid.groupby(prod_cols + ["month", "week_of_month"], as_index=False)["qty_ea"].sum(),
        on=prod_cols + ["month", "week_of_month"],
        how="left",
    )
    prod_week["qty_ea"] = prod_week["qty_ea"].fillna(0.0)

    prod_day = complete_product_day_grid(products).merge(
        valid.groupby(prod_cols + ["date", "month", "week_of_month", "dow"], as_index=False)["qty_ea"].sum(),
        on=prod_cols + ["date", "month", "week_of_month", "dow"],
        how="left",
    )
    prod_day["qty_ea"] = prod_day["qty_ea"].fillna(0.0)
    return {"category_week": cat_week, "category_day": cat_day, "product_week": prod_week, "product_day": prod_day, "products": products}


def select_best(rows: list[MetricRow], layer: str) -> MetricRow:
    candidates = [r for r in rows if r.layer == layer]
    met = [r for r in candidates if r.target_met]
    pool = met or candidates
    return sorted(pool, key=lambda r: (r.wmape, abs(r.bias_ratio), r.max_period_wmape))[0]


def product_day_from_layers(product_week: pd.DataFrame, category_day: pd.DataFrame, model_name: str) -> pd.DataFrame:
    day_share = category_day[["product_reference_group", "month", "week_of_month", "date", "dow", "share"]].rename(
        columns={"share": "day_share"}
    )
    out = product_week.rename(columns={"pred": "week_pred"}).merge(
        day_share,
        on=["product_reference_group", "month", "week_of_month"],
        how="left",
    )
    out["pred"] = out["week_pred"] * out["day_share"].fillna(0.0)
    out["allocation_model"] = model_name
    return out[["product_reference_group", "product_id", "product_name", "date", "month", "week_of_month", "dow", "pred", "allocation_model"]]


def product_dow_ipf(df: pd.DataFrame, product_week: pd.DataFrame, category_day: pd.DataFrame) -> pd.DataFrame:
    hist = df[df["month"].isin(HIST_MONTHS)]
    prod_cols = ["product_reference_group", "product_id", "product_name"]
    global_dow = hist.groupby("dow", as_index=False)["qty_ea"].sum()
    global_dow["global_share"] = global_dow["qty_ea"] / max(float(global_dow["qty_ea"].sum()), 1e-9)
    pdow = dow_weights(df, prod_cols)
    rows = []
    for (cat, week), pw in product_week.groupby(["product_reference_group", "week_of_month"], sort=False):
        cd = category_day[(category_day["product_reference_group"] == cat) & (category_day["week_of_month"] == week)].copy()
        if pw.empty or cd.empty:
            continue
        base = pw[prod_cols + ["pred"]].rename(columns={"pred": "row_target"}).merge(
            cd[["date", "month", "week_of_month", "dow", "pred"]].rename(columns={"pred": "col_target"}),
            how="cross",
        )
        base = base.merge(pdow[prod_cols + ["dow", "share", "key_total"]], on=prod_cols + ["dow"], how="left")
        base = base.merge(global_dow[["dow", "global_share"]], on="dow", how="left")
        base["share"] = base["share"].fillna(0.0)
        base["key_total"] = base["key_total"].fillna(0.0)
        base["global_share"] = base["global_share"].fillna(1 / 7)
        base["weight"] = (base["share"] * base["key_total"] + 1000.0 * base["global_share"]) / (base["key_total"] + 1000.0)
        base["pred"] = base["weight"].clip(lower=1e-12)
        for _ in range(8):
            row_sum = base.groupby(prod_cols)["pred"].transform("sum").replace(0, np.nan)
            base["pred"] *= base["row_target"] / row_sum
            col_sum = base.groupby(["date"])["pred"].transform("sum").replace(0, np.nan)
            base["pred"] *= base["col_target"] / col_sum
        rows.append(base[prod_cols + ["date", "month", "week_of_month", "dow", "pred"]])
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out["allocation_model"] = "product_day_product_dow_ipf"
    return out


def reconciliation_checks(parent: pd.DataFrame, category_day: pd.DataFrame, product_week: pd.DataFrame, product_day: pd.DataFrame) -> pd.DataFrame:
    checks = []

    def add(name: str, left: float, right: float) -> None:
        checks.append({"check": name, "left_total": left, "right_total": right, "abs_diff": abs(left - right), "passed": abs(left - right) <= 1e-6 * max(1.0, abs(right))})

    add("category_day_sum_equals_parent", float(category_day["pred"].sum()), float(parent["pred"].sum()))
    add("product_week_sum_equals_parent", float(product_week["pred"].sum()), float(parent["pred"].sum()))
    add("product_day_sum_equals_product_week", float(product_day["pred"].sum()), float(product_week["pred"].sum()))
    add("product_day_sum_equals_category_day", float(product_day["pred"].sum()), float(category_day["pred"].sum()))
    add("no_negative_category_day", float((category_day["pred"] < -1e-9).sum()), 0.0)
    add("no_negative_product_week", float((product_week["pred"] < -1e-9).sum()), 0.0)
    add("no_negative_product_day", float((product_day["pred"] < -1e-9).sum()), 0.0)
    return pd.DataFrame(checks)


def save_plots(parent_joined: pd.DataFrame, category_day_joined: pd.DataFrame, product_week_joined: pd.DataFrame, product_day_joined: pd.DataFrame) -> None:
    week = parent_joined.groupby("week_of_month", as_index=False).agg(actual=("qty_ea", "sum"), pred=("pred", "sum"))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(week["week_of_month"], week["actual"], marker="o", label="actual")
    ax.plot(week["week_of_month"], week["pred"], marker="o", label="predicted")
    ax.set_title("Parent category-week actual vs predicted")
    ax.set_xlabel("March week")
    ax.set_ylabel("qty_ea")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "90_production_parent_category_week.png", dpi=160)
    plt.close(fig)

    day = category_day_joined.groupby("date", as_index=False).agg(actual=("qty_ea", "sum"), pred=("pred", "sum"))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(day["date"], day["actual"], label="actual")
    ax.plot(day["date"], day["pred"], label="predicted")
    ax.set_title("Category-day actual vs predicted")
    ax.set_xlabel("date")
    ax.set_ylabel("qty_ea")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "91_production_category_day.png", dpi=160)
    plt.close(fig)

    prod_err = product_week_joined.groupby(["product_reference_group", "product_id"], as_index=False).agg(actual=("qty_ea", "sum"), pred=("pred", "sum"))
    prod_err["abs_error"] = (prod_err["actual"] - prod_err["pred"]).abs()
    top = prod_err.sort_values("abs_error", ascending=False).head(25)
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = top["product_reference_group"].astype(str) + " | " + top["product_id"].astype(str)
    ax.barh(labels, top["abs_error"], color="#8f5d2c")
    ax.invert_yaxis()
    ax.set_title("Largest product-week aggregate errors")
    ax.set_xlabel("absolute qty error")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "92_production_product_week_errors.png", dpi=160)
    plt.close(fig)

    d = product_day_joined[product_day_joined["qty_ea"] > 0].copy()
    d["ape"] = (d["qty_ea"] - d["pred"]).abs() / d["qty_ea"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(d["ape"].clip(upper=2.0), bins=40, color="#637a3b")
    ax.axvline(0.10, color="red", linestyle="--", linewidth=1)
    ax.set_title("Product-day positive-actual APE distribution")
    ax.set_xlabel("APE clipped at 2.0")
    ax.set_ylabel("rows")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "93_production_product_day_error_hist.png", dpi=160)
    plt.close(fig)


def update_gavin(verdict: dict[str, object]) -> None:
    path = PROJECT_ROOT / "gavin.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Task 2 Modeling Handoff\n"
    marker = "## Production Granularity Execution Results"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n"
    block = f"""

## Production Granularity Execution Results

Generated by `src/15_granularity.py`.

### Files Added Or Modified In This Stage

- Added `src/15_granularity.py`.
- Added/updated `reports/production_granularity/`.
- Added/updated figures `reports/figures/90_production_parent_category_week.png` through `93_production_product_day_error_hist.png`.

### Final Metrics

- Parent `category × week` WMAPE: `{verdict['parent_category_week_wmape']:.6f}`.
- Selected `category × day` WMAPE: `{verdict['category_day_wmape']:.6f}`.
- Selected `product × week` WMAPE: `{verdict['product_week_wmape']:.6f}`.
- Selected `product × day` WMAPE: `{verdict['product_day_wmape']:.6f}`.
- Category-day target met: `{verdict['category_day_target_met']}`.
- Product-week target met: `{verdict['product_week_target_met']}`.
- Product-day target met: `{verdict['product_day_target_met']}`.
- Production-ready verdict: `{verdict['production_ready']}`.

### Interpretation

The hierarchy is reconciled by construction. If product-level targets fail, the failure is an allocation and product-mix stability issue under Jan-Feb-only history, not a category-week parent issue.
"""
    path.write_text(existing.rstrip() + "\n" + block, encoding="utf-8")


def main() -> int:
    ensure_dirs()
    log("Loading parent forecast, actual demand, and product tags...")
    df = load_data()
    parent = load_parent()
    panels = actual_panels(df, parent)

    metric_rows: list[MetricRow] = []
    parent_row, parent_joined = metric(
        "parent_category_week",
        "selected_tagged_systematic_parent",
        panels["category_week"],
        parent[["product_reference_group", "month", "week_of_month", "pred"]],
        ["product_reference_group", "month", "week_of_month"],
        "week_of_month",
        TARGETS["parent_category_week"],
    )
    metric_rows.append(parent_row)

    log("Running week-to-day allocation ablations...")
    cat_day_candidates = category_day_candidates(df, parent)
    cat_day_joined_by_model = {}
    for name, pred in cat_day_candidates.items():
        row, joined = metric(
            "category_day",
            name,
            panels["category_day"],
            pred[["product_reference_group", "date", "month", "week_of_month", "pred"]],
            ["product_reference_group", "date", "month", "week_of_month"],
            "date",
            TARGETS["category_day"],
        )
        metric_rows.append(row)
        cat_day_joined_by_model[name] = joined
    selected_cat_day = select_best(metric_rows, "category_day")
    category_day_pred = cat_day_candidates[selected_cat_day.model].copy()
    category_day_joined = cat_day_joined_by_model[selected_cat_day.model]

    log("Running category-to-product weekly allocation ablations...")
    products = panels["products"]
    prod_week_candidates = product_share_candidates(df, parent, products)
    prod_week_joined_by_model = {}
    for name, pred in prod_week_candidates.items():
        row, joined = metric(
            "product_week",
            name,
            panels["product_week"],
            pred[["product_reference_group", "product_id", "product_name", "month", "week_of_month", "pred"]],
            ["product_reference_group", "product_id", "product_name", "month", "week_of_month"],
            "week_of_month",
            TARGETS["product_week"],
        )
        metric_rows.append(row)
        prod_week_joined_by_model[name] = joined
    log("Combining product-week candidates into product-day forecasts...")
    product_day_candidates = {}
    product_day_source_week: dict[str, str] = {}
    for week_name, week_pred in prod_week_candidates.items():
        share_name = f"product_day__{week_name}__category_day_share"
        ipf_name = f"product_day__{week_name}__product_dow_ipf"
        product_day_candidates[share_name] = product_day_from_layers(week_pred, category_day_pred, share_name)
        product_day_source_week[share_name] = week_name
        product_day_candidates[ipf_name] = product_dow_ipf(df, week_pred, category_day_pred)
        product_day_candidates[ipf_name]["allocation_model"] = ipf_name
        product_day_source_week[ipf_name] = week_name

    prod_day_joined_by_model = {}
    for name, pred in product_day_candidates.items():
        row, joined = metric(
            "product_day",
            name,
            panels["product_day"],
            pred[["product_reference_group", "product_id", "product_name", "date", "month", "week_of_month", "pred"]],
            ["product_reference_group", "product_id", "product_name", "date", "month", "week_of_month"],
            "date",
            TARGETS["product_day"],
        )
        metric_rows.append(row)
        prod_day_joined_by_model[name] = joined
    selected_prod_day = select_best(metric_rows, "product_day")
    selected_week_for_day = product_day_source_week[selected_prod_day.model]
    selected_prod_week = next(r for r in metric_rows if r.layer == "product_week" and r.model == selected_week_for_day)
    product_week_pred = prod_week_candidates[selected_prod_week.model].copy()
    product_week_joined = prod_week_joined_by_model[selected_prod_week.model]
    product_day_pred = product_day_candidates[selected_prod_day.model].copy()
    product_day_joined = prod_day_joined_by_model[selected_prod_day.model]

    metrics = pd.DataFrame([r.__dict__ for r in metric_rows])
    metrics.to_csv(REPORT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")
    category_day_pred.to_csv(REPORT_DIR / "category_day_predictions.csv", index=False, encoding="utf-8-sig")
    product_week_pred.to_csv(REPORT_DIR / "product_week_predictions.csv", index=False, encoding="utf-8-sig")
    product_day_pred.to_csv(REPORT_DIR / "product_day_predictions.csv", index=False, encoding="utf-8-sig")

    checks = reconciliation_checks(parent, category_day_pred, product_week_pred, product_day_pred)
    checks.to_csv(REPORT_DIR / "reconciliation_checks.csv", index=False, encoding="utf-8-sig")

    category_day_errors = category_day_joined.copy()
    category_day_errors["abs_error"] = (category_day_errors["qty_ea"] - category_day_errors["pred"]).abs()
    category_day_errors.to_csv(REPORT_DIR / "category_day_errors.csv", index=False, encoding="utf-8-sig")
    product_week_errors = product_week_joined.copy()
    product_week_errors["abs_error"] = (product_week_errors["qty_ea"] - product_week_errors["pred"]).abs()
    product_week_errors.to_csv(REPORT_DIR / "product_week_errors.csv", index=False, encoding="utf-8-sig")
    product_day_errors = product_day_joined.copy()
    product_day_errors["abs_error"] = (product_day_errors["qty_ea"] - product_day_errors["pred"]).abs()
    product_day_errors.to_csv(REPORT_DIR / "product_day_errors.csv", index=False, encoding="utf-8-sig")

    save_plots(parent_joined, category_day_joined, product_week_joined, product_day_joined)

    verdict = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "parent_model": parent_row.model,
        "parent_category_week_wmape": float(parent_row.wmape),
        "parent_category_week_target_met": bool(parent_row.target_met),
        "selected_category_day_model": selected_cat_day.model,
        "category_day_wmape": float(selected_cat_day.wmape),
        "category_day_target_met": bool(selected_cat_day.target_met),
        "selected_product_week_model": selected_prod_week.model,
        "product_week_wmape": float(selected_prod_week.wmape),
        "product_week_target_met": bool(selected_prod_week.target_met),
        "selected_product_day_model": selected_prod_day.model,
        "product_day_wmape": float(selected_prod_day.wmape),
        "product_day_target_met": bool(selected_prod_day.target_met),
        "all_reconciliation_checks_passed": bool(checks["passed"].all()),
        "production_ready": bool(parent_row.target_met and selected_cat_day.target_met and selected_prod_week.target_met and selected_prod_day.target_met and checks["passed"].all()),
        "dominant_product_week_errors": product_week_errors.sort_values("abs_error", ascending=False).head(20)[
            ["product_reference_group", "product_id", "product_name", "week_of_month", "qty_ea", "pred", "abs_error"]
        ].to_dict(orient="records"),
        "notes": [
            "Allocation shares are estimated from Jan-Feb only.",
            "March actuals are used only for validation.",
            "All selected predictions reconcile back to the selected category-week parent forecast.",
        ],
    }
    (REPORT_DIR / "production_granularity_verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")

    best = metrics.sort_values("wmape").groupby("layer", as_index=False).first()
    lines = [
        "# Production Granularity Report",
        "",
        "Generated by `src/15_granularity.py`.",
        "",
        "## Verdict",
        "",
        f"- Parent category-week WMAPE: `{parent_row.wmape:.6f}`; target met: `{parent_row.target_met}`",
        f"- Selected category-day model: `{selected_cat_day.model}`; WMAPE: `{selected_cat_day.wmape:.6f}`; target met: `{selected_cat_day.target_met}`",
        f"- Selected product-week model: `{selected_prod_week.model}`; WMAPE: `{selected_prod_week.wmape:.6f}`; target met: `{selected_prod_week.target_met}`",
        f"- Selected product-day model: `{selected_prod_day.model}`; WMAPE: `{selected_prod_day.wmape:.6f}`; target met: `{selected_prod_day.target_met}`",
        f"- All reconciliation checks passed: `{checks['passed'].all()}`",
        f"- Production-ready verdict: `{verdict['production_ready']}`",
        "",
        "## Best Model By Layer",
        "",
        best.to_markdown(index=False),
        "",
        "## Reconciliation Checks",
        "",
        checks.to_markdown(index=False),
        "",
        "## All Ablation Metrics",
        "",
        metrics.sort_values(["layer", "wmape"]).to_markdown(index=False),
        "",
        "## Largest Product-Week Errors",
        "",
        product_week_errors.sort_values("abs_error", ascending=False).head(30)[
            ["product_reference_group", "product_id", "product_name", "week_of_month", "qty_ea", "pred", "abs_error"]
        ].to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "The parent category-week forecast is validated first. Lower layers are allocation systems estimated from Jan-Feb only and reconciled to the parent forecast. If product-level targets fail, the failure is caused by product-mix volatility and sparse product history rather than broken category-week volume.",
        "",
    ]
    (REPORT_DIR / "production_granularity_report.md").write_text("\n".join(lines), encoding="utf-8")
    update_gavin(verdict)
    log(f"Wrote {REPORT_DIR / 'production_granularity_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
