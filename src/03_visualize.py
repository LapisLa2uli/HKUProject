"""Step 3: Matplotlib visualizations of cleaned demand data."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 110

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "processed"
MARTS = PROCESSED / "marts"
FIG = PROJECT_ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, name: str) -> Path:
    out = FIG / name
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out}")
    return out


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wh_day = pd.read_csv(MARTS / "mart_warehouse_day.csv", parse_dates=["create_date"])
    wh_cust_day = pd.read_csv(MARTS / "mart_warehouse_customer_day.csv", parse_dates=["create_date"])
    wh_prod_day = pd.read_csv(MARTS / "mart_warehouse_product_day.csv", parse_dates=["create_date"])
    wh_summary = pd.read_csv(MARTS / "mart_warehouse_summary.csv", parse_dates=["first_day", "last_day"])
    return wh_day, wh_cust_day, wh_prod_day, wh_summary


def load_orders() -> pd.DataFrame:
    orders = pd.read_csv(
        PROCESSED / "orders.csv",
        dtype={"order_id": str, "customer_id": str},
        parse_dates=["create_time", "create_date"],
        low_memory=False,
    )
    orders["order_total_boxes"] = pd.to_numeric(orders["order_total_boxes"], errors="coerce").fillna(
        0
    )
    orders["create_date"] = pd.to_datetime(orders["create_date"], errors="coerce").dt.normalize()
    return orders


def fig_daily_total(wh_day: pd.DataFrame) -> None:
    daily = wh_day.groupby("create_date")[["orders", "qty_ea"]].sum().reset_index()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.plot(daily["create_date"], daily["orders"], color="steelblue")
    ax1.set_title("Daily total orders (all warehouses)")
    ax1.set_ylabel("orders")
    ax1.grid(True, alpha=0.3)
    ax2.plot(daily["create_date"], daily["qty_ea"], color="darkorange")
    ax2.set_title("Daily total qty_ea (all warehouses)")
    ax2.set_ylabel("qty_ea")
    ax2.set_xlabel("date")
    ax2.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    save(fig, "01_daily_total.png")


def fig_daily_by_customer(wh_cust_day: pd.DataFrame) -> None:
    cust = (
        wh_cust_day.groupby(["create_date", "customer_id"])["qty_ea"].sum().reset_index()
    )
    pivot = cust.pivot(index="create_date", columns="customer_id", values="qty_ea").fillna(0)
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in pivot.columns:
        ax.plot(pivot.index, pivot[col], label=col)
    ax.set_title("Daily qty_ea by customer")
    ax.set_xlabel("date")
    ax.set_ylabel("qty_ea")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    save(fig, "02_daily_by_customer.png")


def fig_warehouse_share(wh_summary: pd.DataFrame) -> None:
    top = wh_summary.head(15).copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["warehouse"][::-1], top["total_qty_ea"][::-1], color="seagreen")
    ax.set_title("Top 15 warehouses by total qty_ea (Jan-Mar)")
    ax.set_xlabel("qty_ea")
    for i, v in enumerate(top["share_qty_pct"][::-1]):
        ax.text(top["total_qty_ea"][::-1].iloc[i], i, f" {v:.1f}%", va="center", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    save(fig, "03_warehouse_share.png")


def fig_warehouse_monthly(wh_day: pd.DataFrame) -> None:
    df = wh_day.copy()
    df["month"] = df["create_date"].dt.to_period("M").astype(str)
    monthly = df.groupby(["warehouse", "month"])["qty_ea"].sum().reset_index()
    pivot = monthly.pivot(index="warehouse", columns="month", values="qty_ea").fillna(0)
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=False).head(15).drop(columns="total")

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", stacked=False, ax=ax, width=0.8)
    ax.set_title("Monthly qty_ea per warehouse (top 15)")
    ax.set_xlabel("warehouse")
    ax.set_ylabel("qty_ea")
    ax.legend(title="month")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    save(fig, "04_warehouse_monthly.png")


def fig_dow_pattern(wh_day: pd.DataFrame) -> None:
    df = wh_day.groupby("create_date")[["qty_ea", "orders"]].sum().reset_index()
    df["dow"] = df["create_date"].dt.dayofweek
    df["dow_name"] = df["create_date"].dt.day_name()
    pat = df.groupby(["dow", "dow_name"])["qty_ea"].mean().reset_index().sort_values("dow")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(pat["dow_name"], pat["qty_ea"], color="slateblue")
    ax.set_title("Average daily qty_ea by day-of-week")
    ax.set_ylabel("avg qty_ea")
    ax.grid(True, axis="y", alpha=0.3)
    save(fig, "05_dow_pattern.png")


def fig_temperature_zone(wh_prod_day: pd.DataFrame) -> None:
    df = wh_prod_day.groupby(["create_date", "temperature_zone"])["qty_ea"].sum().reset_index()
    pivot = df.pivot(index="create_date", columns="temperature_zone", values="qty_ea").fillna(0)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(pivot.index, pivot.T, labels=pivot.columns, alpha=0.85)
    ax.set_title("Daily qty_ea by temperature zone")
    ax.set_ylabel("qty_ea")
    ax.set_xlabel("date")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    save(fig, "06_temperature_zone.png")


def fig_top_products(wh_prod_day: pd.DataFrame) -> None:
    top = (
        wh_prod_day.groupby(["product_id", "product_name"])["qty_ea"]
        .sum()
        .reset_index()
        .sort_values("qty_ea", ascending=False)
        .head(15)
    )
    labels = [
        f"{pid} {pname[:18]}" for pid, pname in zip(top["product_id"], top["product_name"].fillna(""))
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(labels[::-1], top["qty_ea"][::-1], color="indianred")
    ax.set_title("Top 15 products by qty_ea (Jan-Mar)")
    ax.set_xlabel("qty_ea")
    ax.grid(True, axis="x", alpha=0.3)
    save(fig, "07_top_products.png")


def fig_warehouse_heatmap(wh_day: pd.DataFrame) -> None:
    pivot = wh_day.pivot_table(
        index="warehouse", columns="create_date", values="qty_ea", aggfunc="sum"
    ).fillna(0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    pivot = pivot.head(20)

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    n = len(pivot.columns)
    tick_step = max(1, n // 12)
    ax.set_xticks(range(0, n, tick_step))
    ax.set_xticklabels(
        [pd.Timestamp(d).strftime("%m-%d") for d in pivot.columns[::tick_step]], rotation=45
    )
    ax.set_title("Daily qty_ea heatmap (top 20 warehouses)")
    fig.colorbar(im, ax=ax, label="qty_ea")
    save(fig, "08_warehouse_heatmap.png")


def _store_order_boxes_pivot(orders: pd.DataFrame) -> pd.DataFrame:
    df = orders.dropna(subset=["create_date", "store"])
    df = (
        df.groupby(["store", "create_date"], dropna=False)["order_total_boxes"]
        .sum()
        .reset_index()
    )
    pivot = df.pivot(index="store", columns="create_date", values="order_total_boxes").fillna(0)
    return pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]


def fig_store_order_boxes_heatmap(orders: pd.DataFrame) -> None:
    pivot = _store_order_boxes_pivot(orders).head(20)

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(pivot.values, aspect="auto", cmap="magma")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    n = len(pivot.columns)
    tick_step = max(1, n // 12)
    ax.set_xticks(range(0, n, tick_step))
    ax.set_xticklabels(
        [pd.Timestamp(d).strftime("%m-%d") for d in pivot.columns[::tick_step]], rotation=45
    )
    ax.set_title("Daily estimated total order quantity (in boxes) heatmap (top 20 receiving stores, sum per day)")
    fig.colorbar(im, ax=ax, label="estimated total order quantity (boxes)")
    save(fig, "10_store_order_boxes_heatmap.png")


def fig_store_order_boxes_heatmap_all_unlabeled(orders: pd.DataFrame) -> None:
    pivot = _store_order_boxes_pivot(orders)
    n_rows = len(pivot.index)
    fig_h = float(np.clip(0.34 * n_rows, 14.0, 72.0))

    vals = pivot.values.astype(float)
    cmap = plt.colormaps["magma"].copy()
    cmap.set_bad(color="#e8e8e8")
    pos = vals[vals > 0]
    plot_vals = np.where(vals > 0, vals, np.nan)
    if pos.size and float(vals.max()) > 0:
        vmin = max(1.0, float(pos.min()))
        vmax = max(float(vals.max()), vmin * 1.0001)
        norm: LogNorm | None = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None

    fig, ax = plt.subplots(figsize=(14, fig_h))
    im = ax.imshow(plot_vals, aspect="auto", cmap=cmap, norm=norm)
    ax.set_yticks([])
    n = len(pivot.columns)
    tick_step = max(1, n // 12)
    ax.set_xticks(range(0, n, tick_step))
    ax.set_xticklabels(
        [pd.Timestamp(d).strftime("%m-%d") for d in pivot.columns[::tick_step]], rotation=45
    )
    ax.set_title(
        f"Daily 预计总箱数 heatmap (all {n_rows} 收货门店, log color scale, sum per day)"
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("预计总箱数 (sum), log scale")
    save(fig, "11_store_order_boxes_heatmap_all.png")


def fig_customer_warehouse_mix(wh_cust_day: pd.DataFrame) -> None:
    mix = (
        wh_cust_day.groupby(["warehouse", "customer_id"])["qty_ea"].sum().reset_index()
    )
    top_wh = (
        mix.groupby("warehouse")["qty_ea"].sum().sort_values(ascending=False).head(12).index
    )
    mix = mix[mix["warehouse"].isin(top_wh)]
    pivot = mix.pivot(index="warehouse", columns="customer_id", values="qty_ea").fillna(0)
    pivot = pivot.loc[top_wh]
    pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = np.zeros(len(pct))
    for col in pct.columns:
        ax.bar(pct.index, pct[col], bottom=bottom, label=col)
        bottom += pct[col].values
    ax.set_title("Customer mix per warehouse (top 12 warehouses, % of qty_ea)")
    ax.set_ylabel("% of warehouse qty_ea")
    ax.legend(title="customer")
    plt.xticks(rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    save(fig, "09_customer_mix_by_warehouse.png")


def main() -> int:
    wh_day, wh_cust_day, wh_prod_day, wh_summary = load()
    orders = load_orders()

    fig_daily_total(wh_day)
    fig_daily_by_customer(wh_cust_day)
    fig_warehouse_share(wh_summary)
    fig_warehouse_monthly(wh_day)
    fig_dow_pattern(wh_day)
    fig_temperature_zone(wh_prod_day)
    fig_top_products(wh_prod_day)
    fig_warehouse_heatmap(wh_day)
    fig_store_order_boxes_heatmap(orders)
    fig_store_order_boxes_heatmap_all_unlabeled(orders)
    fig_customer_warehouse_mix(wh_cust_day)
    return 0


if __name__ == "__main__":
    sys.exit(main())
