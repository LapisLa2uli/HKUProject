"""Step 5: Visualize baseline forecast results.

Charts produced under ``reports/figures``:
    11_validation_monthly_scatter.png     Monthly (wh,cust,prod) predicted vs actual scatter
    12_april_forecast_daily.png           Apr baseline forecast daily totals
    13_history_plus_forecast.png          Jan-Apr daily totals (history vs baseline forecast)
    14_april_warehouse_share.png          Baseline Apr qty per warehouse
    15_validation_error_hist.png          Monthly validation error histogram

When hurdle April daily exports exist (pattern + store-cal, not network-scaled):
    16_april_daily_baseline_vs_hurdle.png Apr daily total: baseline vs hurdle pattern vs store-cal
    17_april_warehouse_baseline_vs_hurdle.png Top warehouses: baseline vs hurdle (pattern + store-cal)
    18_history_baseline_hurdle_forecast.png History + Apr baseline + hurdle pattern + store-cal
    19_april_customer_baseline_vs_hurdle.png Apr totals by customer: baseline vs hurdle (pattern + store-cal)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 110

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "processed"
MARTS = PROCESSED / "marts"
REPORTS = PROJECT_ROOT / "reports"
FIG = REPORTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)

HURDLE_PATTERN_CSV = "april_forecast_hurdle_daily_pattern.csv"
HURDLE_STORE_CSV = "april_forecast_hurdle_daily_store.csv"
HURDLE_NETWORK_CSV = "april_forecast_hurdle_daily.csv"
STORE_KEYS = ["warehouse", "customer_id", "store"]


def load_hurdle_april_dailies() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Pattern and per-store-calibrated April hurdle daily (excludes network scale)."""
    pattern_path = REPORTS / HURDLE_PATTERN_CSV
    store_path = REPORTS / HURDLE_STORE_CSV
    if pattern_path.exists() and store_path.exists():
        fp = pd.read_csv(pattern_path, parse_dates=["date"])
        fs = pd.read_csv(store_path, parse_dates=["date"])
        return fp, fs

    scaled_path = REPORTS / HURDLE_NETWORK_CSV
    cal_path = REPORTS / "hurdle_april_calibration.csv"
    store_cal_path = REPORTS / "hurdle_april_store_calibration.csv"
    if not (scaled_path.exists() and cal_path.exists() and store_cal_path.exists()):
        return None

    print(
        "[info] Deriving hurdle pattern/store daily from network-scaled export "
        "(re-run 04b_hurdle_model.py to write dedicated CSVs)."
    )
    scaled = pd.read_csv(scaled_path, parse_dates=["date"])
    cal = pd.read_csv(cal_path)
    ns_row = cal.loc[cal["metric"] == "april_network_scale", "value"]
    if ns_row.empty:
        ns_row = cal.loc[cal["metric"] == "april_scale", "value"]
    ns = float(ns_row.iloc[0])
    store_scales = (
        pd.read_csv(store_cal_path)
        .set_index(STORE_KEYS)["store_scale"]
        .astype(np.float64)
    )
    keys = scaled[STORE_KEYS].apply(tuple, axis=1)
    mult = keys.map(store_scales).fillna(1.0).astype(np.float64).values
    qty_scaled = scaled["qty_ea"].astype(np.float64).values
    qty_store = qty_scaled / ns
    qty_pattern = np.where(mult > 1e-12, qty_store / mult, qty_store)
    fs = scaled.copy()
    fs["qty_ea"] = qty_store
    fp = scaled.copy()
    fp["qty_ea"] = qty_pattern
    return fp, fs


def save(fig: plt.Figure, name: str) -> Path:
    out = FIG / name
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out}")
    return out


def main() -> int:
    monthly = pd.read_csv(REPORTS / "baseline_validation_monthly_actual_vs_pred.csv")
    fc = pd.read_csv(REPORTS / "april_forecast_daily.csv", parse_dates=["date"])
    wh_day = pd.read_csv(MARTS / "mart_warehouse_day.csv", parse_dates=["create_date"])
    wh_day = wh_day[wh_day["create_date"] <= pd.Timestamp("2026-03-28")]

    fig, ax = plt.subplots(figsize=(11, 5))
    daily_actual = wh_day.groupby("create_date")["qty_ea"].sum()
    fig.suptitle("Daily total qty_ea: history + April forecast")
    ax.plot(daily_actual.index, daily_actual.values, label="actual (Jan-Mar28)", color="steelblue")
    daily_fc = fc.groupby("date")["qty_ea"].sum()
    ax.plot(daily_fc.index, daily_fc.values, label="forecast (Apr)", color="firebrick")
    ax.axvspan(pd.Timestamp("2026-04-01"), pd.Timestamp("2026-04-30"), color="firebrick", alpha=0.05)
    ax.set_ylabel("qty_ea")
    ax.set_xlabel("date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    save(fig, "13_history_plus_forecast.png")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(monthly["actual"], monthly["pred"], alpha=0.4, s=12)
    lim = max(monthly["actual"].max(), monthly["pred"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color="black", linestyle="--", linewidth=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("March actual monthly qty_ea (per warehouse,customer,product)")
    ax.set_ylabel("March predicted monthly qty_ea")
    ax.set_title("Validation: monthly predicted vs actual")
    ax.grid(True, alpha=0.3)
    save(fig, "11_validation_monthly_scatter.png")

    daily_fc = fc.groupby("date")["qty_ea"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(daily_fc["date"], daily_fc["qty_ea"], color="firebrick", alpha=0.85)
    ax.set_title("April forecast: daily total qty_ea")
    ax.set_ylabel("qty_ea")
    ax.grid(True, axis="y", alpha=0.3)
    fig.autofmt_xdate()
    save(fig, "12_april_forecast_daily.png")

    wh_apr = fc.groupby("warehouse")["qty_ea"].sum().sort_values(ascending=False)
    top = wh_apr.head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top.index[::-1], top.values[::-1], color="darkcyan")
    ax.set_title("Top 15 warehouses by April predicted qty_ea")
    ax.set_xlabel("predicted qty_ea")
    ax.grid(True, axis="x", alpha=0.3)
    save(fig, "14_april_warehouse_share.png")

    monthly = monthly.copy()
    monthly["error"] = monthly["pred"] - monthly["actual"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(monthly["error"], bins=60, color="slategray")
    ax.axvline(0, color="black", linestyle="--")
    ax.set_title("Validation error distribution: predicted - actual (monthly per tuple)")
    ax.set_xlabel("error (qty_ea)")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    save(fig, "15_validation_error_hist.png")

    hurdle = load_hurdle_april_dailies()
    if hurdle is None:
        print("[skip] No hurdle pattern/store daily exports; skipping comparative charts.")
        return 0

    fh_pattern, fh_store = hurdle
    daily_b = fc.groupby("date")["qty_ea"].sum().sort_index()
    daily_hp = fh_pattern.groupby("date")["qty_ea"].sum().sort_index()
    daily_hs = fh_store.groupby("date")["qty_ea"].sum().sort_index()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(daily_b.index, daily_b.values, label="baseline", color="firebrick", linewidth=2)
    ax.plot(
        daily_hp.index,
        daily_hp.values,
        label="hurdle pattern (Apr)",
        color="goldenrod",
        linewidth=2,
    )
    ax.plot(
        daily_hs.index,
        daily_hs.values,
        label="hurdle store-cal (Apr)",
        color="darkorange",
        linewidth=2,
        linestyle="--",
    )
    ax.axvspan(pd.Timestamp("2026-04-01"), pd.Timestamp("2026-04-30"), color="gray", alpha=0.06)
    ax.set_title("April forecast: daily total qty_ea — baseline vs hurdle (pattern + store-cal)")
    ax.set_ylabel("qty_ea")
    ax.set_xlabel("date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    save(fig, "16_april_daily_baseline_vs_hurdle.png")

    wh_b = fc.groupby("warehouse")["qty_ea"].sum()
    wh_p = fh_pattern.groupby("warehouse")["qty_ea"].sum()
    wh_s = fh_store.groupby("warehouse")["qty_ea"].sum()
    wh_cmp = pd.DataFrame({"baseline": wh_b, "hurdle_pattern": wh_p, "hurdle_store": wh_s}).fillna(0)
    wh_cmp["max_pair"] = wh_cmp.max(axis=1)
    top_wh = wh_cmp.nlargest(15, "max_pair").sort_values("baseline")

    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(top_wh))
    h = 0.24
    ax.barh(y - h, top_wh["baseline"], height=h, label="baseline", color="firebrick", alpha=0.85)
    ax.barh(y, top_wh["hurdle_pattern"], height=h, label="hurdle pattern", color="goldenrod", alpha=0.85)
    ax.barh(y + h, top_wh["hurdle_store"], height=h, label="hurdle store-cal", color="darkorange", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(top_wh.index)
    ax.set_xlabel("predicted qty_ea (April)")
    ax.set_title("Top 15 warehouses: April predicted qty — baseline vs hurdle (pattern + store-cal)")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)
    save(fig, "17_april_warehouse_baseline_vs_hurdle.png")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(daily_actual.index, daily_actual.values, label="actual (Jan–Mar 28)", color="steelblue")
    ax.plot(daily_b.index, daily_b.values, label="forecast baseline (Apr)", color="firebrick", linestyle="-")
    ax.plot(
        daily_hp.index,
        daily_hp.values,
        label="forecast hurdle pattern (Apr)",
        color="goldenrod",
        linestyle="-",
    )
    ax.plot(
        daily_hs.index,
        daily_hs.values,
        label="forecast hurdle store-cal (Apr)",
        color="darkorange",
        linestyle="--",
    )
    ax.axvspan(pd.Timestamp("2026-04-01"), pd.Timestamp("2026-04-30"), color="gray", alpha=0.06)
    ax.set_title("Daily total qty_ea: history + April forecasts")
    ax.set_ylabel("qty_ea")
    ax.set_xlabel("date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    save(fig, "18_history_baseline_hurdle_forecast.png")

    cust_b = fc.groupby("customer_id")["qty_ea"].sum()
    cust_p = fh_pattern.groupby("customer_id")["qty_ea"].sum()
    cust_s = fh_store.groupby("customer_id")["qty_ea"].sum()
    cust_cmp = (
        pd.DataFrame({"baseline": cust_b, "hurdle_pattern": cust_p, "hurdle_store": cust_s})
        .fillna(0)
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(cust_cmp))
    w = 0.25
    ax.bar(x - w, cust_cmp["baseline"], width=w, label="baseline", color="firebrick", alpha=0.85)
    ax.bar(x, cust_cmp["hurdle_pattern"], width=w, label="hurdle pattern", color="goldenrod", alpha=0.85)
    ax.bar(x + w, cust_cmp["hurdle_store"], width=w, label="hurdle store-cal", color="darkorange", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(cust_cmp.index)
    ax.set_xlabel("customer_id")
    ax.set_ylabel("predicted qty_ea (April)")
    ax.set_title("April predicted totals by customer — baseline vs hurdle (pattern + store-cal)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    save(fig, "19_april_customer_baseline_vs_hurdle.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
