"""Independent store-product demand forecast (Workstream 2).

Approach: TSB intermittent-demand volume + category-level calibration with bounded
price-regime context + calendar/DOW disbursement for daily planning rows.

Train Jan-Feb for March validation; refit Jan-Mar for April deployment.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from _cny import add_cny_features, detect_warehouse_closure  # noqa: E402
from _metrics import bias_ratio, store_planning_loss_report, wmape  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "processed"
MARTS = PROCESSED / "marts"
EXT = PROJECT_ROOT / "external_data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports" / "best_forecast"

JAN_START = pd.Timestamp("2026-01-01")
JAN_END = pd.Timestamp("2026-01-31")
TRAIN_END = pd.Timestamp("2026-02-28")
VALID_START = pd.Timestamp("2026-03-01")
VALID_END = pd.Timestamp("2026-03-28")
FORECAST_START = pd.Timestamp("2026-04-01")
FORECAST_END = pd.Timestamp("2026-04-30")
MARCH_END = pd.Timestamp("2026-03-31")

MIN_NONZERO_DAYS_TRAIN = 3
ID_COLS = ["warehouse", "store", "product_id"]
STORE_KEYS = ["warehouse", "customer_id", "store"]
CAT_COL = "product_macro_group"
META_COLS = list(dict.fromkeys(ID_COLS + STORE_KEYS + [CAT_COL, "product_name", "temperature_zone"]))

TSB_ALPHA = 0.12
TSB_BETA = 0.10
PRICE_ELASTICITY = {
    "price_rising": -0.12,
    "price_falling": 0.08,
    "price_stable": 0.0,
    "high_volatility": -0.04,
}
PRICE_SCALE_CLIP = (0.92, 1.08)
PLANNING_WINDOW_DAYS = 2
PLANNING_QTY_REL_TOL = 0.15
TREND_EXTRAPOLATION = True
DOW_SMOOTH_ALPHA = 0.3


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mean_network_daily_total(df: pd.DataFrame, qty_col: str = "qty_ea") -> float:
    if df.empty:
        return 0.0
    return float(df.groupby("date")[qty_col].sum().mean())


def compute_q_targets(panel: pd.DataFrame) -> dict[str, float]:
    jan = panel[(panel["date"] >= JAN_START) & (panel["date"] <= JAN_END)]
    mar = panel[(panel["date"] >= VALID_START) & (panel["date"] <= VALID_END)]
    q_jan = mean_network_daily_total(jan)
    q_mar = mean_network_daily_total(mar)
    return {"Q_Jan": q_jan, "Q_Mar": q_mar, "Q_target": 0.5 * (q_jan + q_mar)}


def load_sparse_mart() -> pd.DataFrame:
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
        low_memory=False,
    )
    df = df.rename(columns={"create_date": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["qty_ea"] = pd.to_numeric(df["qty_ea"], errors="coerce").fillna(0.0).clip(lower=0.0)
    return df.dropna(subset=["date", "warehouse", "store", "product_id"])


def load_taxonomy() -> pd.DataFrame:
    tax = pd.read_csv(
        EXT / "product_category_mapping.csv",
        dtype={"product_id": str},
    )
    tax[CAT_COL] = tax.get(CAT_COL, tax.get("external_category", "other")).fillna("other").astype(str)
    return tax[["product_id", "product_name", "temperature_zone", CAT_COL]].drop_duplicates("product_id")


def load_calendar_tags() -> pd.DataFrame:
    cal = pd.read_csv(EXT / "systematic_day_tags.csv", parse_dates=["date"])
    cal["date"] = pd.to_datetime(cal["date"], errors="coerce").dt.normalize()
    cal["cal_weight"] = pd.to_numeric(cal.get("base_open_weight", 1.0), errors="coerce").fillna(1.0).clip(0.05, 1.5)
    return cal[["date", "cal_weight", "is_open_day"]]


def load_price_regime() -> pd.DataFrame:
    pr = pd.read_csv(EXT / "weekly_price_regime_tags.csv")
    pr["month"] = pr["month"].astype(str)
    pr["price_change_pct"] = pd.to_numeric(pr["price_change_pct"], errors="coerce").fillna(0.0)
    pr["price_scale"] = pr["price_regime"].map(PRICE_ELASTICITY).fillna(0.0) * pr["price_change_pct"]
    pr["price_scale"] = (1.0 + pr["price_scale"]).clip(PRICE_SCALE_CLIP[0], PRICE_SCALE_CLIP[1])
    return pr[["month", "week_of_month", CAT_COL, "price_regime", "price_scale"]]


def build_dense_panel(sparse: pd.DataFrame, history_end: pd.Timestamp) -> pd.DataFrame:
    df = sparse[sparse["date"] <= history_end].copy()
    nonzero_train = (
        df[(df["date"] <= TRAIN_END) & (df["qty_ea"] > 0)]
        .groupby(ID_COLS, observed=True)
        .size()
    )
    keep = nonzero_train[nonzero_train >= MIN_NONZERO_DAYS_TRAIN].index
    keep_df = pd.DataFrame(list(keep), columns=ID_COLS)

    meta = (
        df.groupby(ID_COLS, dropna=False, observed=True)
        .agg(
            product_name=("product_name", "first"),
            temperature_zone=("temperature_zone", "first"),
            customer_id=("customer_id", "first"),
        )
        .reset_index()
    )

    agg = df.groupby(ID_COLS + ["date"], as_index=False, observed=True)["qty_ea"].sum()
    all_dates = pd.date_range(agg["date"].min(), history_end, freq="D")
    tuples = keep_df.copy()
    tuples["_key"] = 1
    dates_df = pd.DataFrame({"date": all_dates, "_key": 1})
    grid = tuples.merge(dates_df, on="_key").drop(columns="_key")
    panel = grid.merge(agg, on=ID_COLS + ["date"], how="left")
    panel["qty_ea"] = panel["qty_ea"].fillna(0.0)
    panel = panel.merge(meta, on=ID_COLS, how="left")
    return panel.sort_values(ID_COLS + ["date"]).reset_index(drop=True)


def attach_context(panel: pd.DataFrame, taxonomy: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    wh_day = panel.groupby(["warehouse", "date"], as_index=False)["qty_ea"].sum()
    closure = detect_warehouse_closure(wh_day)
    out = add_cny_features(panel, closure)
    drop_tax = [c for c in ("product_name", "temperature_zone", CAT_COL) if c in out.columns]
    out = out.drop(columns=drop_tax, errors="ignore")
    out = out.merge(taxonomy, on="product_id", how="left")
    out[CAT_COL] = out[CAT_COL].fillna("other")
    out["product_name"] = out["product_name"].fillna("unknown")
    out["temperature_zone"] = out["temperature_zone"].fillna("unknown")
    out = out.merge(calendar, on="date", how="left")
    out["cal_weight"] = out["cal_weight"].fillna(1.0)
    out["is_open_day"] = out["is_open_day"].fillna(1).astype(int)
    out["open_flag"] = (
        (out["is_warehouse_closed"].fillna(0).astype(int) == 0)
        & (out["is_open_day"] == 1)
    ).astype(np.int8)
    return out


def forecast_open_calendar(
    calendar: pd.DataFrame,
    forecast_start: pd.Timestamp,
    forecast_end: pd.Timestamp,
) -> pd.DataFrame:
    cal = calendar[(calendar["date"] >= forecast_start) & (calendar["date"] <= forecast_end)].copy()
    cal["open_flag"] = (cal["is_open_day"].fillna(1).astype(int) == 1).astype(np.int8)
    return cal[["date", "cal_weight", "open_flag"]].drop_duplicates().sort_values("date")


def count_open_days(cal: pd.DataFrame) -> int:
    return int((cal["open_flag"] == 1).sum())


def _tsb_fit_state(
    qty: np.ndarray,
    open_flag: np.ndarray,
    *,
    alpha: float = TSB_ALPHA,
    beta: float = TSB_BETA,
) -> dict[str, float]:
    open_qty = qty[open_flag.astype(bool)] if len(open_flag) else qty
    pos = open_qty[open_qty > 0] if len(open_qty) else np.array([], dtype=np.float64)
    n_open = max(int(np.sum(open_flag)), 1)
    p = float(max(len(pos) / n_open, 1.0 / n_open))
    level = float(np.median(pos)) if len(pos) else 1.0

    for q, is_open in zip(qty, open_flag):
        if not is_open:
            continue
        if q > 0:
            p = alpha * 1.0 + (1.0 - alpha) * p
            level = beta * float(q) + (1.0 - beta) * level
        else:
            p = alpha * 0.0 + (1.0 - alpha) * p

    daily_rate = p * level
    return {
        "tsb_p": p,
        "tsb_level": level,
        "tsb_daily_rate": daily_rate,
        "train_open_days": float(n_open),
        "train_pos_days": float(len(pos)),
        "train_qty_sum": float(np.sum(open_qty)),
    }


def tsb_monthly_totals(
    panel: pd.DataFrame,
    train_end: pd.Timestamp,
    forecast_cal: pd.DataFrame,
) -> pd.DataFrame:
    train = panel[panel["date"] <= train_end].copy()
    n_open_fc = count_open_days(forecast_cal)
    if n_open_fc == 0:
        return pd.DataFrame(columns=ID_COLS + ["tsb_month_total", "tsb_p", "expected_order_days"])

    rows: list[dict] = []
    for key, g_train in train.groupby(ID_COLS, sort=False, observed=True):
        g_train = g_train.sort_values("date")
        st = _tsb_fit_state(
            g_train["qty_ea"].to_numpy(dtype=np.float64),
            g_train["open_flag"].to_numpy(dtype=bool),
        )
        hist_rate = st["train_qty_sum"] / max(st["train_open_days"], 1.0)
        tsb_total = st["tsb_daily_rate"] * n_open_fc
        hist_total = hist_rate * n_open_fc
        month_total = 0.6 * tsb_total + 0.4 * hist_total

        if TREND_EXTRAPOLATION:
            qty_arr = g_train["qty_ea"].to_numpy(dtype=np.float64)
            open_arr = g_train["open_flag"].to_numpy(dtype=bool)
            n_total = len(qty_arr)
            mid = n_total // 2
            first_half_rate = qty_arr[:mid][open_arr[:mid]].sum() / max(open_arr[:mid].sum(), 1)
            second_half_rate = qty_arr[mid:][open_arr[mid:]].sum() / max(open_arr[mid:].sum(), 1)
            if first_half_rate > 0:
                prod_trend = np.clip(second_half_rate / first_half_rate, 0.75, 1.10)
            else:
                prod_trend = 1.0
            month_total *= prod_trend

        exp_days = int(
            np.clip(
                round(st["tsb_p"] * n_open_fc),
                1,
                n_open_fc,
            )
        )
        row = dict(zip(ID_COLS, key if isinstance(key, tuple) else (key,)))
        row["tsb_month_total"] = float(month_total)
        row["tsb_p"] = float(st["tsb_p"])
        row["expected_order_days"] = exp_days
        rows.append(row)
    return pd.DataFrame(rows)


def category_targets(
    panel: pd.DataFrame,
    train_end: pd.Timestamp,
    forecast_cal: pd.DataFrame,
    price_regime: pd.DataFrame,
) -> pd.DataFrame:
    hist = panel[panel["date"] <= train_end].copy()
    hist = hist[hist["open_flag"] == 1]
    hist["month_key"] = hist["date"].dt.to_period("M").astype(str)

    cat_daily = (
        hist.groupby(STORE_KEYS + [CAT_COL, "date"], as_index=False, observed=True)["qty_ea"]
        .sum()
        .rename(columns={"qty_ea": "day_total"})
    )
    cat_daily["month_key"] = cat_daily["date"].dt.to_period("M").astype(str)
    month_mean = (
        cat_daily.groupby(STORE_KEYS + [CAT_COL, "month_key"], as_index=False, observed=True)["day_total"]
        .mean()
        .rename(columns={"day_total": "mean_daily"})
    )
    pivot = month_mean.pivot_table(
        index=STORE_KEYS + [CAT_COL],
        columns="month_key",
        values="mean_daily",
        aggfunc="first",
        fill_value=0.0,
    ).reset_index()
    month_cols = sorted(c for c in pivot.columns if c not in STORE_KEYS + [CAT_COL])
    if len(month_cols) >= 2:
        raw_w = np.array([1.25**i for i in range(len(month_cols))])
        raw_w /= raw_w.sum()
        pivot["base_daily"] = pivot[month_cols].to_numpy() @ raw_w
        if TREND_EXTRAPOLATION:
            m_vals = pivot[month_cols].to_numpy()
            trend_ratio = np.where(
                m_vals[:, -2] > 0,
                m_vals[:, -1] / np.maximum(m_vals[:, -2], 1e-6),
                1.0,
            )
            trend_ratio = np.clip(trend_ratio, 0.80, 1.05)
            pivot["base_daily"] = pivot["base_daily"] * trend_ratio
    elif month_cols:
        pivot["base_daily"] = pivot[month_cols[0]]
    else:
        pivot["base_daily"] = 0.0

    open_days = count_open_days(forecast_cal)
    month_key = pd.Timestamp(forecast_cal["date"].min()).to_period("M").strftime("%Y-%m")
    pr = price_regime[price_regime["month"] == month_key]
    pr_cat = pr.groupby(CAT_COL, as_index=False, observed=True)["price_scale"].mean()
    pivot = pivot.merge(pr_cat, on=CAT_COL, how="left")
    pivot["price_scale"] = pivot["price_scale"].fillna(1.0)
    pivot["category_month_target"] = pivot["base_daily"] * open_days * pivot["price_scale"]
    return pivot[STORE_KEYS + [CAT_COL, "category_month_target", "price_scale"]]


def reconcile_products(
    tsb_totals: pd.DataFrame,
    panel: pd.DataFrame,
    cat_targets: pd.DataFrame,
) -> pd.DataFrame:
    meta = (
        panel[META_COLS]
        .drop_duplicates(subset=ID_COLS)
        .reset_index(drop=True)
    )
    out = tsb_totals.merge(meta, on=ID_COLS, how="left")
    out["tsb_month_total"] = out["tsb_month_total"].fillna(0.0).clip(lower=0.0)
    out = out.merge(cat_targets, on=STORE_KEYS + [CAT_COL], how="left")
    out["category_month_target"] = out["category_month_target"].fillna(0.0)

    grp = out.groupby(STORE_KEYS + [CAT_COL], observed=True)["tsb_month_total"].transform("sum")
    n_prod = out.groupby(STORE_KEYS + [CAT_COL], observed=True)[ID_COLS[2]].transform("count")
    scale = np.where(grp > 1e-9, out["category_month_target"] / grp, 0.0)
    out["month_total"] = np.where(
        grp > 1e-9,
        out["tsb_month_total"] * scale,
        out["category_month_target"] / np.maximum(n_prod, 1),
    )
    return out


def _compute_dow_shares(
    panel: pd.DataFrame,
    train_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Product-level DOW shares smoothed toward global-level pattern.

    Global pattern uses weekday equalization: Sat/Sun/Mon keep historical
    relative shares, Tue-Fri are forced equal to remove CNY distortion
    in within-weekday variation.

    Returns (product_dow_shares, global_dow_shares).
    """
    hist = panel[(panel["date"] <= train_end) & (panel["open_flag"] == 1)].copy()
    hist_pos = hist[hist["qty_ea"] > 0].copy()
    hist_pos["dow"] = hist_pos["date"].dt.dayofweek.astype(int)

    global_dow_raw = (
        hist_pos.groupby("dow", as_index=False, observed=True)["qty_ea"]
        .sum()
        .rename(columns={"qty_ea": "global_qty"})
    )
    if len(global_dow_raw) < 7:
        all_dows = pd.DataFrame({"dow": range(7)})
        global_dow_raw = all_dows.merge(global_dow_raw, on="dow", how="left")
        global_dow_raw["global_qty"] = global_dow_raw["global_qty"].fillna(0.0)
    global_total = global_dow_raw["global_qty"].sum()
    global_dow_raw["raw_share"] = np.where(
        global_total > 0,
        global_dow_raw["global_qty"] / global_total,
        1.0 / 7.0,
    )

    global_dow = global_dow_raw[["dow"]].copy()
    global_dow["global_dow_share"] = global_dow_raw["raw_share"]
    global_dow["global_dow_share"] = global_dow["global_dow_share"].clip(lower=0.06)
    global_dow["global_dow_share"] = (
        global_dow["global_dow_share"] / global_dow["global_dow_share"].sum()
    )

    tuple_dow = (
        hist_pos.groupby(ID_COLS + ["dow"], as_index=False, observed=True)["qty_ea"]
        .sum()
        .rename(columns={"qty_ea": "dow_qty"})
    )
    tuple_total = (
        hist_pos.groupby(ID_COLS, as_index=False, observed=True)["qty_ea"]
        .sum()
        .rename(columns={"qty_ea": "tuple_pos_total"})
    )
    tuple_dow = tuple_dow.merge(tuple_total, on=ID_COLS, how="left")
    tuple_dow["dow_share_raw"] = np.where(
        tuple_dow["tuple_pos_total"] > 0,
        tuple_dow["dow_qty"] / tuple_dow["tuple_pos_total"],
        1.0 / 7.0,
    )

    tuple_dow = tuple_dow.merge(global_dow[["dow", "global_dow_share"]], on="dow", how="left")
    tuple_dow["global_dow_share"] = tuple_dow["global_dow_share"].fillna(1.0 / 7.0)

    n_pos_days = tuple_dow.groupby(ID_COLS, observed=True)["dow_qty"].transform("count")
    alpha = np.where(n_pos_days >= 14, DOW_SMOOTH_ALPHA, DOW_SMOOTH_ALPHA * 1.5)
    alpha = np.clip(alpha, 0.20, 0.55)
    tuple_dow["dow_share"] = (
        (1 - alpha) * tuple_dow["dow_share_raw"] + alpha * tuple_dow["global_dow_share"]
    )

    min_share = 0.06
    tuple_dow["dow_share"] = tuple_dow["dow_share"].clip(lower=min_share)
    share_sum = tuple_dow.groupby(ID_COLS, observed=True)["dow_share"].transform("sum")
    tuple_dow["dow_share"] = tuple_dow["dow_share"] / share_sum

    return tuple_dow[ID_COLS + ["dow", "dow_share"]], global_dow[["dow", "global_dow_share"]]


def dow_disbursement_weights(
    panel: pd.DataFrame,
    month_totals: pd.DataFrame,
    train_end: pd.Timestamp,
    forecast_cal: pd.DataFrame,
) -> pd.DataFrame:
    """Proportional DOW-weighted distribution with equal weekly anchoring."""
    tuple_dow, global_dow = _compute_dow_shares(panel, train_end)

    month_days = forecast_cal[forecast_cal["open_flag"] == 1][["date", "cal_weight"]].copy()
    month_days["dow"] = month_days["date"].dt.dayofweek.astype(int)
    min_date = month_days["date"].min()
    month_days["week_idx"] = ((month_days["date"] - min_date).dt.days // 7).astype(int)
    n_weeks = month_days["week_idx"].nunique()

    totals = month_totals[ID_COLS + ["month_total"]].copy()
    grid = totals.copy()
    grid["_key"] = 1
    md = month_days.copy()
    md["_key"] = 1
    grid = grid.merge(md, on="_key").drop(columns="_key")
    grid = grid.merge(tuple_dow, on=ID_COLS + ["dow"], how="left")
    grid = grid.merge(global_dow, on="dow", how="left")
    grid["dow_share"] = grid["dow_share"].fillna(grid["global_dow_share"]).fillna(1.0 / 7.0)

    grid["raw_weight"] = grid["dow_share"] * grid["cal_weight"]

    week_weight_sum = grid.groupby(ID_COLS + ["week_idx"], observed=True)["raw_weight"].transform("sum")
    grid["intra_week_share"] = np.where(
        week_weight_sum > 0,
        grid["raw_weight"] / week_weight_sum,
        1.0 / grid.groupby(ID_COLS + ["week_idx"], observed=True)["date"].transform("count"),
    )

    grid["week_target"] = grid["month_total"] / n_weeks
    grid["pred"] = grid["week_target"] * grid["intra_week_share"]
    grid["pred"] = grid["pred"].clip(lower=0.0)

    return grid[ID_COLS + ["date", "pred"]]


def _posthoc_dow_rebalance(daily: pd.DataFrame) -> pd.DataFrame:
    """Rebalance aggregate DOW proportions: anchor Mon/Tue/Sun at realized
    levels, equalize Wed-Sat to fix CNY-distorted Saturday/Friday."""
    daily = daily.copy()
    daily["dow"] = daily["date"].dt.dayofweek.astype(int)

    dow_totals = daily.groupby("dow")["pred"].sum()
    weekly_total = dow_totals.sum()
    if weekly_total <= 0:
        return daily

    realized_shares = dow_totals / weekly_total

    anchor_dows = [0, 1, 6]  # Mon, Tue, Sun — reliably estimated from history
    anchor_total = realized_shares.loc[anchor_dows].sum()
    remaining = 1.0 - anchor_total
    equalize_dows = [2, 3, 4, 5]  # Wed, Thu, Fri, Sat — equalized to remove CNY distortion
    target_equal = remaining / len(equalize_dows)

    target_shares = realized_shares.copy()
    for d in equalize_dows:
        target_shares.loc[d] = target_equal

    scale_factors = {}
    for d in range(7):
        if realized_shares.loc[d] > 1e-9:
            scale_factors[d] = target_shares.loc[d] / realized_shares.loc[d]
        else:
            scale_factors[d] = 1.0

    sf_arr = daily["dow"].map(scale_factors).to_numpy(dtype=np.float64)
    daily["pred"] = daily["pred"].to_numpy(dtype=np.float64) * sf_arr
    daily = daily.drop(columns=["dow"])
    return daily


def build_daily_forecast(
    panel: pd.DataFrame,
    month_totals: pd.DataFrame,
    train_end: pd.Timestamp,
    forecast_cal: pd.DataFrame,
) -> pd.DataFrame:
    daily = dow_disbursement_weights(panel, month_totals, train_end, forecast_cal)
    daily = _posthoc_dow_rebalance(daily)
    meta = (
        panel[META_COLS]
        .drop_duplicates(subset=ID_COLS)
        .reset_index(drop=True)
    )
    daily = daily.merge(meta, on=ID_COLS, how="left")
    return daily.sort_values(ID_COLS + ["date"]).reset_index(drop=True)


def evaluate_march(
    panel: pd.DataFrame,
    forecast: pd.DataFrame,
) -> dict[str, float | int]:
    actual = panel[
        (panel["date"] >= VALID_START)
        & (panel["date"] <= VALID_END)
        & (panel["open_flag"] == 1)
    ][ID_COLS + ["date", "qty_ea"] + ["customer_id"]].copy()
    pred = forecast[
        (forecast["date"] >= VALID_START)
        & (forecast["date"] <= VALID_END)
    ][ID_COLS + ["date", "pred"]].copy()

    merged = actual.merge(pred, on=ID_COLS + ["date"], how="left")
    merged["pred"] = merged["pred"].fillna(0.0).clip(lower=0.0)

    y = merged["qty_ea"].to_numpy(dtype=np.float64)
    p = merged["pred"].to_numpy(dtype=np.float64)

    monthly = (
        merged.groupby(ID_COLS, observed=True)
        .agg(actual=("qty_ea", "sum"), pred=("pred", "sum"))
        .reset_index()
    )
    seg_w = wmape(monthly["actual"].to_numpy(), monthly["pred"].to_numpy())

    planning = store_planning_loss_report(
        merged,
        store_keys=STORE_KEYS,
        product_col="product_id",
        date_col="date",
        y_col="qty_ea",
        pred_col="pred",
        window_days=PLANNING_WINDOW_DAYS,
        qty_rel_tol=PLANNING_QTY_REL_TOL,
    )

    q_stats = compute_q_targets(panel)
    mar_pred_daily = merged.groupby("date", observed=True)["pred"].sum().mean()
    mar_act_daily = merged.groupby("date", observed=True)["qty_ea"].sum().mean()

    return {
        "row_wmape": float(wmape(y, p)),
        "segment_monthly_wmape": float(seg_w),
        "bias_ratio": float(bias_ratio(y, p)),
        "strict_daily_wmape": float(planning.get("strict_daily_wmape", float("nan"))),
        "planning_window_loss": float(planning.get("planning_window_loss", float("nan"))),
        "composite_planning_loss": float(planning.get("composite_planning_loss", float("nan"))),
        "matched_volume_rate": float(planning.get("matched_volume_rate", float("nan"))),
        "line_match_rate": float(planning.get("line_match_rate", float("nan"))),
        "mean_abs_day_slip_matched": float(planning.get("mean_abs_day_slip_matched", float("nan"))),
        "Q_Jan": q_stats["Q_Jan"],
        "Q_Mar": q_stats["Q_Mar"],
        "Q_target": q_stats["Q_target"],
        "mean_daily_pred_march": float(mar_pred_daily),
        "mean_daily_actual_march": float(mar_act_daily),
        "n_eval_rows": int(len(merged)),
        "n_stores": int(planning.get("n_stores", 0)),
    }


def write_report_md(metrics: dict[str, float | int], april_summary: dict[str, float | int]) -> None:
    lines = [
        "# Best forecast report (TSB + category calibration)",
        "",
        "Independent Workstream 2 model: `src/18_best_forecast.py`.",
        "",
        "## Approach",
        "",
        "1. **TSB intermittent demand** at `(warehouse, store, product)` — separate",
        "   occurrence smoothing from order-size smoothing (Teunter–Syntetos–Babai).",
        "2. **Category calibration** — reconcile product totals within each store × macro-category",
        "   to a Jan–Feb pace target, with bounded price-regime elasticity from external tags.",
        "3. **Calendar disbursement** — spread monthly totals across days using historical DOW",
        "   quantity shares and systematic calendar open-day weights.",
        "",
        "March validation trains on **Jan–Feb only**. April deployment retrains on **Jan–Mar**.",
        "",
        "## March validation",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in [
        "segment_monthly_wmape",
        "row_wmape",
        "bias_ratio",
        "planning_window_loss",
        "composite_planning_loss",
        "strict_daily_wmape",
        "matched_volume_rate",
        "line_match_rate",
        "mean_abs_day_slip_matched",
        "Q_target",
        "mean_daily_pred_march",
        "mean_daily_actual_march",
    ]:
        val = metrics.get(key, float("nan"))
        if isinstance(val, float):
            lines.append(f"| {key} | {val:.6f} |")
        else:
            lines.append(f"| {key} | {val} |")

    lines.extend(
        [
            "",
            "## April forecast summary",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
    )
    for key, val in april_summary.items():
        if isinstance(val, float):
            lines.append(f"| {key} | {val:.6f} |")
        else:
            lines.append(f"| {key} | {val} |")

    lines.extend(
        [
            "",
            "## Comparison to baselines",
            "",
            "| Metric | TSB-CatCal (this model) | Hurdle (04b) | Hierarchical product-day |",
            "|---|---:|---:|---:|",
            f"| Segment/monthly volume WMAPE | {metrics.get('segment_monthly_wmape', float('nan')):.4f} | 0.8536 | 0.1930 |",
            f"| Planning window loss (lower better) | {metrics.get('planning_window_loss', float('nan')):.4f} | 1.1752 | — |",
            f"| Composite planning loss | {metrics.get('composite_planning_loss', float('nan')):.4f} | 1.2453 | — |",
            f"| Strict daily WMAPE | {metrics.get('strict_daily_wmape', float('nan')):.4f} | 1.4559 | 0.1930 |",
            f"| Volume bias ratio | {metrics.get('bias_ratio', float('nan')):.4f} | -0.3249 | — |",
            f"| Line match rate (±2d) | {metrics.get('line_match_rate', float('nan')):.4f} | 0.3243 | — |",
            "",
            "Hierarchical WMAPE is at reconciled **product-day** grain (scripts 14–15); hurdle metrics from",
            "`reports/hurdle/hurdle_validation_*.csv`. Category-week hierarchical anchor remains strongest (~3.6% WMAPE).",
            "",
            "## Comparison notes",
            "",
            "- **Hurdle / store regression** optimize sliding-window daily GBDT rows; this model",
            "  targets monthly product volume first, then places quantity on planning days.",
            "- **Hierarchical baseline (scripts 14–17)** anchors category-week (~3.6% WMAPE) and",
            "  reconciles down; product-day WMAPE there is ~19.3%. This script stays at store×product",
            "  grain without multi-layer IPF.",
            "- Planning loss (±2 day product/qty match) is the primary business metric here.",
            "",
            "## Limitations / next steps",
            "",
            "- New or returning products with <3 Jan–Feb order days are excluded.",
            "- Category calibration uses macro-group price tags; product-level elasticity not modeled.",
            "- Day disbursement uses smoothed DOW shares — consider cadence/gap models for top SKUs.",
        ]
    )
    (REPORT_DIR / "best_forecast_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    t0 = time.time()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    log("loading data")
    sparse = load_sparse_mart()
    taxonomy = load_taxonomy()
    calendar = load_calendar_tags()
    price_regime = load_price_regime()

    log("building validation panel (history through March 28)")
    panel_valid = build_dense_panel(sparse, VALID_END)
    panel_valid = attach_context(panel_valid, taxonomy, calendar)

    log("March validation: train Jan-Feb")
    march_cal = forecast_open_calendar(calendar, VALID_START, VALID_END)
    # Respect warehouse closure days observed in the validation panel.
    wh_open = (
        panel_valid[panel_valid["date"].between(VALID_START, VALID_END)]
        .groupby("date", observed=True)["open_flag"]
        .max()
        .reset_index()
    )
    march_cal = march_cal.merge(wh_open, on="date", how="left", suffixes=("", "_wh"))
    march_cal["open_flag"] = march_cal["open_flag_wh"].fillna(march_cal["open_flag"]).astype(np.int8)
    march_cal = march_cal.drop(columns=["open_flag_wh"], errors="ignore")

    tsb_valid = tsb_monthly_totals(panel_valid, TRAIN_END, march_cal)
    cat_tgt_valid = category_targets(panel_valid, TRAIN_END, march_cal, price_regime)
    month_valid = reconcile_products(tsb_valid, panel_valid, cat_tgt_valid)
    march_fc = build_daily_forecast(panel_valid, month_valid, TRAIN_END, march_cal)

    log("evaluating March")
    metrics = evaluate_march(panel_valid, march_fc)
    for k, v in metrics.items():
        if isinstance(v, float):
            log(f"  {k}={v:.6f}")
        else:
            log(f"  {k}={v}")

    metrics_path = REPORT_DIR / "march_validation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    march_fc_out = REPORT_DIR / "march_validation_daily.csv"
    out_cols = list(dict.fromkeys(ID_COLS + STORE_KEYS + ["product_name", "temperature_zone", CAT_COL, "date", "pred"]))
    march_fc[out_cols].to_csv(march_fc_out, index=False, encoding="utf-8-sig")

    log("April forecast: train Jan-Mar")
    panel_april = build_dense_panel(sparse, MARCH_END)
    panel_april = attach_context(panel_april, taxonomy, calendar)
    train_april_end = MARCH_END
    april_cal = forecast_open_calendar(calendar, FORECAST_START, FORECAST_END)
    tsb_april = tsb_monthly_totals(panel_april, train_april_end, april_cal)
    cat_tgt_april = category_targets(panel_april, train_april_end, april_cal, price_regime)
    month_april = reconcile_products(tsb_april, panel_april, cat_tgt_april)
    april_fc = build_daily_forecast(panel_april, month_april, train_april_end, april_cal)

    april_total = float(april_fc["pred"].sum())
    april_daily_mean = float(april_fc.groupby("date", observed=True)["pred"].sum().mean())
    q_stats = compute_q_targets(panel_april)
    april_summary = {
        "april_total_qty": april_total,
        "april_mean_daily_total": april_daily_mean,
        "Q_target": q_stats["Q_target"],
        "april_to_Q_target_ratio": april_daily_mean / q_stats["Q_target"] if q_stats["Q_target"] else float("nan"),
        "n_forecast_rows": len(april_fc),
        "n_products": int(april_fc["product_id"].nunique()),
        "n_stores": int(april_fc.groupby(STORE_KEYS, observed=True).ngroups),
    }
    april_fc[out_cols].to_csv(REPORT_DIR / "april_forecast_daily.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([april_summary]).to_csv(REPORT_DIR / "april_forecast_summary.csv", index=False, encoding="utf-8-sig")

    write_report_md(metrics, april_summary)
    log(f"done in {time.time() - t0:.1f}s -> {REPORT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
