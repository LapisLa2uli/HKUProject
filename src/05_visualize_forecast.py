"""Step 5: Visualize baseline, hurdle, and store-regression forecast results."""

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

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from _report_paths import (
    BASELINE_APRIL_DAILY,
    BASELINE_VALIDATION_MONTHLY,
    FIGURES,
    HURDLE_APRIL_DAILY_NETWORK,
    HURDLE_APRIL_DAILY_PATTERN,
    HURDLE_APRIL_DAILY_STORE,
    STORE_REG_APRIL_DAILY,
    STORE_REG_VALIDATION_MONTHLY,
    ensure_report_dirs,
)
from _report_paths import PROCESSED, PROJECT_ROOT

MARTS = PROCESSED / "marts"


def save(fig: plt.Figure, name: str) -> Path:
    out = FIGURES / name
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out}")
    return out


def load_hurdle_april_daily() -> pd.DataFrame | None:
    """Sliding-window April daily (pattern / store / network exports are equivalent)."""
    for path in (
        HURDLE_APRIL_DAILY_NETWORK,
        HURDLE_APRIL_DAILY_PATTERN,
        HURDLE_APRIL_DAILY_STORE,
    ):
        if path.exists():
            return pd.read_csv(path, parse_dates=["date"])
    return None


def plot_model_suite(
    monthly: pd.DataFrame,
    fc: pd.DataFrame,
    wh_day: pd.DataFrame,
    *,
    prefix: str,
    label: str,
    color: str,
) -> None:
    _plot_history_plus_forecast(wh_day, fc, f"{label}: history + April", f"{prefix}_history_plus_forecast.png", color)
    _plot_monthly_scatter(monthly, f"{label}: March validation scatter", f"{prefix}_validation_monthly_scatter.png")
    _plot_april_daily_bars(fc, f"{label}: April daily total", f"{prefix}_april_forecast_daily.png", color)
    _plot_april_warehouse_share(fc, f"{label}: top warehouses (April)", f"{prefix}_april_warehouse_share.png")
    _plot_validation_error_hist(monthly, f"{label}: March validation errors", f"{prefix}_validation_error_hist.png")


def _plot_monthly_scatter(monthly: pd.DataFrame, title: str, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(monthly["actual"], monthly["pred"], alpha=0.4, s=12)
    lim = max(monthly["actual"].max(), monthly["pred"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color="black", linestyle="--", linewidth=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("March actual monthly qty_ea")
    ax.set_ylabel("March predicted monthly qty_ea")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    save(fig, fname)


def _plot_april_daily_bars(daily_fc: pd.DataFrame, title: str, fname: str, color: str) -> None:
    daily = daily_fc.groupby("date")["qty_ea"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(daily["date"], daily["qty_ea"], color=color, alpha=0.85)
    ax.set_title(title)
    ax.set_ylabel("qty_ea")
    ax.grid(True, axis="y", alpha=0.3)
    fig.autofmt_xdate()
    save(fig, fname)


def _plot_history_plus_forecast(
    wh_day: pd.DataFrame,
    daily_fc: pd.DataFrame,
    title: str,
    fname: str,
    color: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    daily_actual = wh_day.groupby("create_date")["qty_ea"].sum()
    daily_pred = daily_fc.groupby("date")["qty_ea"].sum()
    ax.plot(daily_actual.index, daily_actual.values, label="actual (Jan–Mar 28)", color="steelblue")
    ax.plot(daily_pred.index, daily_pred.values, label="forecast (Apr)", color=color)
    ax.axvspan(pd.Timestamp("2026-04-01"), pd.Timestamp("2026-04-30"), color=color, alpha=0.05)
    ax.set_ylabel("qty_ea")
    ax.set_xlabel("date")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    save(fig, fname)


def _plot_april_warehouse_share(fc: pd.DataFrame, title: str, fname: str) -> None:
    wh_apr = fc.groupby("warehouse")["qty_ea"].sum().sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(wh_apr.index[::-1], wh_apr.values[::-1], color="darkcyan")
    ax.set_title(title)
    ax.set_xlabel("predicted qty_ea")
    ax.grid(True, axis="x", alpha=0.3)
    save(fig, fname)


def _plot_validation_error_hist(monthly: pd.DataFrame, title: str, fname: str) -> None:
    err = monthly.copy()
    err["error"] = err["pred"] - err["actual"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(err["error"], bins=60, color="slategray")
    ax.axvline(0, color="black", linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("error (qty_ea)")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    save(fig, fname)


def plot_multi_model_comparison(
    fc: pd.DataFrame,
    fh: pd.DataFrame,
    fs_reg: pd.DataFrame,
    wh_day: pd.DataFrame,
) -> None:
    daily_b = fc.groupby("date")["qty_ea"].sum().sort_index()
    daily_h = fh.groupby("date")["qty_ea"].sum().sort_index()
    daily_sr = fs_reg.groupby("date")["qty_ea"].sum().sort_index()
    daily_actual = wh_day.groupby("create_date")["qty_ea"].sum()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily_actual.index, daily_actual.values, label="actual (Jan–Mar 28)", color="steelblue", linewidth=2)
    ax.plot(daily_b.index, daily_b.values, label="baseline sliding (Apr)", color="firebrick", linewidth=2)
    ax.plot(daily_h.index, daily_h.values, label="hurdle sliding (Apr)", color="goldenrod", linewidth=2)
    ax.plot(
        daily_sr.index,
        daily_sr.values,
        label="store regression sliding (Apr)",
        color="seagreen",
        linewidth=2,
    )
    ax.axvspan(pd.Timestamp("2026-04-01"), pd.Timestamp("2026-04-30"), color="gray", alpha=0.06)
    ax.set_title("April daily total qty_ea — all models")
    ax.set_ylabel("qty_ea")
    ax.set_xlabel("date")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    save(fig, "16_april_daily_all_models.png")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(daily_actual.index, daily_actual.values, label="actual (Jan–Mar 28)", color="steelblue")
    ax.plot(daily_b.index, daily_b.values, label="baseline sliding (Apr)", color="firebrick")
    ax.plot(daily_h.index, daily_h.values, label="hurdle sliding (Apr)", color="goldenrod")
    ax.plot(daily_sr.index, daily_sr.values, label="store regression sliding (Apr)", color="seagreen")
    ax.axvspan(pd.Timestamp("2026-04-01"), pd.Timestamp("2026-04-30"), color="gray", alpha=0.06)
    ax.set_title("Daily total qty_ea: history + April forecasts (all models)")
    ax.set_ylabel("qty_ea")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    save(fig, "18_history_all_models_forecast.png")

    wh_b = fc.groupby("warehouse")["qty_ea"].sum()
    wh_h = fh.groupby("warehouse")["qty_ea"].sum()
    wh_r = fs_reg.groupby("warehouse")["qty_ea"].sum()
    wh_cmp = pd.DataFrame(
        {
            "baseline": wh_b,
            "hurdle": wh_h,
            "store_regression": wh_r,
        }
    ).fillna(0)
    wh_cmp["max_pair"] = wh_cmp.max(axis=1)
    top_wh = wh_cmp.nlargest(15, "max_pair").sort_values("baseline")
    y = np.arange(len(top_wh))
    h = 0.22
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(y - h, top_wh["baseline"], height=h, label="baseline sliding", color="firebrick", alpha=0.85)
    ax.barh(y, top_wh["hurdle"], height=h, label="hurdle sliding", color="goldenrod", alpha=0.85)
    ax.barh(y + h, top_wh["store_regression"], height=h, label="store reg sliding", color="seagreen", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(top_wh.index)
    ax.set_xlabel("predicted qty_ea (April)")
    ax.set_title("Top 15 warehouses: April totals — all models")
    ax.legend(fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    save(fig, "17_april_warehouse_all_models.png")

    cust_b = fc.groupby("customer_id")["qty_ea"].sum()
    cust_h = fh.groupby("customer_id")["qty_ea"].sum()
    cust_r = fs_reg.groupby("customer_id")["qty_ea"].sum()
    cust_cmp = (
        pd.DataFrame(
            {
                "baseline": cust_b,
                "hurdle": cust_h,
                "store_regression": cust_r,
            }
        )
        .fillna(0)
        .sort_index()
    )
    x = np.arange(len(cust_cmp))
    w = 0.22
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, cust_cmp["baseline"], width=w, label="baseline sliding", color="firebrick", alpha=0.85)
    ax.bar(x, cust_cmp["hurdle"], width=w, label="hurdle sliding", color="goldenrod", alpha=0.85)
    ax.bar(x + w, cust_cmp["store_regression"], width=w, label="store reg sliding", color="seagreen", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(cust_cmp.index)
    ax.set_ylabel("predicted qty_ea (April)")
    ax.set_title("April totals by customer — all models")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    save(fig, "19_april_customer_all_models.png")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(daily_b.index, daily_b.values, label="baseline sliding", color="firebrick", linewidth=2)
    ax.plot(daily_h.index, daily_h.values, label="hurdle sliding", color="goldenrod", linewidth=2)
    ax.axvspan(pd.Timestamp("2026-04-01"), pd.Timestamp("2026-04-30"), color="gray", alpha=0.06)
    ax.set_title("April forecast: daily total — baseline vs hurdle")
    ax.set_ylabel("qty_ea")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    save(fig, "16_april_daily_baseline_vs_hurdle.png")


def main() -> int:
    ensure_report_dirs()

    if not BASELINE_VALIDATION_MONTHLY.exists() or not BASELINE_APRIL_DAILY.exists():
        print("Missing baseline reports; run 04_baseline_model.py first.", file=sys.stderr)
        return 1

    monthly_b = pd.read_csv(BASELINE_VALIDATION_MONTHLY)
    fc_b = pd.read_csv(BASELINE_APRIL_DAILY, parse_dates=["date"])
    wh_day = pd.read_csv(MARTS / "mart_warehouse_day.csv", parse_dates=["create_date"])
    wh_day = wh_day[wh_day["create_date"] <= pd.Timestamp("2026-03-28")]

    _plot_history_plus_forecast(
        wh_day, fc_b, "Daily total qty_ea: history + April forecast (baseline)",
        "13_history_plus_forecast.png", "firebrick",
    )
    _plot_monthly_scatter(
        monthly_b, "Validation: monthly predicted vs actual (baseline)",
        "11_validation_monthly_scatter.png",
    )
    _plot_april_daily_bars(fc_b, "April forecast: daily total qty_ea (baseline)", "12_april_forecast_daily.png", "firebrick")
    _plot_april_warehouse_share(fc_b, "Top 15 warehouses by April predicted qty_ea (baseline)", "14_april_warehouse_share.png")
    _plot_validation_error_hist(monthly_b, "Validation error distribution (baseline)", "15_validation_error_hist.png")

    if STORE_REG_VALIDATION_MONTHLY.exists() and STORE_REG_APRIL_DAILY.exists():
        monthly_sr = pd.read_csv(STORE_REG_VALIDATION_MONTHLY)
        fc_sr = pd.read_csv(STORE_REG_APRIL_DAILY, parse_dates=["date"])
        _plot_history_plus_forecast(
            wh_day, fc_sr, "Store regression (sliding): history + April",
            "29_store_regression_history_plus_forecast.png", "seagreen",
        )
        _plot_monthly_scatter(
            monthly_sr, "Store regression (sliding): March validation scatter",
            "27_store_regression_validation_monthly_scatter.png",
        )
        _plot_april_daily_bars(
            fc_sr, "Store regression (sliding): April daily total",
            "28_store_regression_april_forecast_daily.png", "seagreen",
        )
        _plot_april_warehouse_share(
            fc_sr, "Store regression (sliding): top warehouses (April)",
            "30_store_regression_april_warehouse_share.png",
        )
        _plot_validation_error_hist(
            monthly_sr, "Store regression (sliding): March validation errors",
            "31_store_regression_validation_error_hist.png",
        )
    else:
        print("[skip] Store regression CSVs missing; run 04c_store_regression_model.py")

    fh = load_hurdle_april_daily()
    if fh is None:
        print("[skip] Hurdle daily exports missing; skipping multi-model charts.")
        return 0

    if STORE_REG_APRIL_DAILY.exists():
        fs_reg = pd.read_csv(STORE_REG_APRIL_DAILY, parse_dates=["date"])
        plot_multi_model_comparison(fc_b, fh, fs_reg, wh_day)
    else:
        print("[skip] Multi-model comparison needs store regression April daily.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
