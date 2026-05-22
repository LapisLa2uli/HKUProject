"""Create a focused six-chart error pack for the active hierarchy.

The charts are intentionally limited to:

1. predicted-vs-actual line charts;
2. category error bar charts.

This keeps the handoff readable while still showing the main error sources.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from _metrics import bias_ratio, wmape  # noqa: E402

REPORT_DIR = PROJECT_ROOT / "reports" / "production_granularity"
CHART_DIR = REPORT_DIR / "error_charts"
PARENT_FORECAST = PROJECT_ROOT / "reports" / "weekly_tagged_systematic" / "selected_validation_predictions.csv"
CATEGORY_DAY = REPORT_DIR / "category_day_errors.csv"
PRODUCT_WEEK = REPORT_DIR / "product_week_errors.csv"
PRODUCT_DAY = REPORT_DIR / "product_day_errors.csv"

BLUE = "#2563eb"
INK = "#0f172a"
MUTED = "#64748b"
GRID = "#e5e7eb"
RED = "#dc2626"
AMBER = "#d97706"
GREEN = "#059669"
PURPLE = "#7c3aed"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#cbd5e1",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelcolor": "#334155",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "font.size": 10,
            "legend.frameon": False,
            "lines.linewidth": 2.4,
        }
    )


def reset_chart_dir() -> None:
    if CHART_DIR.exists():
        shutil.rmtree(CHART_DIR)
    CHART_DIR.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, name: str, title: str, note: str, rows: list[dict[str, str]]) -> None:
    path = CHART_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    rows.append({"file": str(path.relative_to(PROJECT_ROOT)), "title": title, "note": note})


def add_value_labels(ax: plt.Axes, values: pd.Series, fmt: str = "{:.1%}") -> None:
    for i, value in enumerate(values):
        if np.isfinite(value):
            ax.text(value, i, " " + fmt.format(value), va="center", ha="left", color=MUTED, fontsize=9)


def weighted_metrics(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    out = df.groupby(keys, as_index=False).agg(actual=("qty_ea", "sum"), pred=("pred", "sum"))
    out["abs_error"] = (out["actual"] - out["pred"]).abs()
    out["wmape"] = np.divide(out["abs_error"], out["actual"].replace(0, np.nan)).fillna(0.0)
    out["bias"] = np.divide(out["pred"] - out["actual"], out["actual"].replace(0, np.nan)).fillna(0.0)
    return out


def line_actual_pred(
    ax: plt.Axes,
    data: pd.DataFrame,
    x: str,
    title: str,
    xlabel: str,
    ylabel: str = "qty_ea",
) -> None:
    ax.plot(data[x], data["actual"], color=INK, marker="o", label="Actual")
    ax.plot(data[x], data["pred"], color=BLUE, marker="o", linestyle="--", label="Predicted")
    ax.fill_between(data[x], data["actual"], data["pred"], color=BLUE, alpha=0.08)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def bar_category_error(ax: plt.Axes, data: pd.DataFrame, title: str, target: float | None, color: str) -> None:
    ordered = data.sort_values("wmape", ascending=True)
    ax.barh(ordered["product_reference_group"], ordered["wmape"], color=color, alpha=0.92)
    if target is not None:
        ax.axvline(target, color=RED, linestyle="--", linewidth=1.3, label=f"Target {target:.0%}")
        ax.legend(loc="lower right")
    ax.set_title(title)
    ax.set_xlabel("WMAPE")
    add_value_labels(ax, ordered["wmape"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main() -> int:
    setup_style()
    reset_chart_dir()

    parent = pd.read_csv(PARENT_FORECAST)
    category_day = pd.read_csv(CATEGORY_DAY, parse_dates=["date"])
    product_week = pd.read_csv(PRODUCT_WEEK)
    product_day = pd.read_csv(PRODUCT_DAY, parse_dates=["date"])

    chart_rows: list[dict[str, str]] = []

    parent_total = weighted_metrics(parent, ["week_of_month"]).sort_values("week_of_month")
    parent_wmape = wmape(parent_total["actual"], parent_total["pred"])
    fig, ax = plt.subplots(figsize=(9, 5))
    line_actual_pred(
        ax,
        parent_total,
        "week_of_month",
        f"Category-week total: actual vs predicted (WMAPE {parent_wmape:.1%})",
        "March week",
    )
    save(fig, "01_category_week_line.png", "Category-week actual vs predicted", "Parent weekly category forecast is the strongest layer.", chart_rows)

    parent_cat = weighted_metrics(parent, ["product_reference_group"])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bar_category_error(ax, parent_cat, "Category-week WMAPE by category", 0.05, GREEN)
    save(fig, "02_category_week_error_bars.png", "Category-week category errors", "Shows which broad categories drive parent forecast error.", chart_rows)

    category_day_total = weighted_metrics(category_day, ["date"]).sort_values("date")
    cat_day_wmape = wmape(category_day_total["actual"], category_day_total["pred"])
    fig, ax = plt.subplots(figsize=(11, 5))
    line_actual_pred(
        ax,
        category_day_total,
        "date",
        f"Category-day total: actual vs predicted (WMAPE {cat_day_wmape:.1%})",
        "March date",
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    save(fig, "03_category_day_line.png", "Category-day actual vs predicted", "Daily timing error is the first major remaining gap.", chart_rows)

    category_day_cat = weighted_metrics(category_day, ["product_reference_group"])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bar_category_error(ax, category_day_cat, "Category-day WMAPE by category", 0.05, AMBER)
    save(fig, "04_category_day_error_bars.png", "Category-day category errors", "All category-day groups remain above the 5% reference target.", chart_rows)

    product_week_total = weighted_metrics(product_week, ["week_of_month"]).sort_values("week_of_month")
    prod_week_wmape = wmape(product_week_total["actual"], product_week_total["pred"])
    fig, ax = plt.subplots(figsize=(9, 5))
    line_actual_pred(
        ax,
        product_week_total,
        "week_of_month",
        f"Product-week total: actual vs predicted (WMAPE {prod_week_wmape:.1%})",
        "March week",
    )
    save(fig, "05_product_week_line.png", "Product-week actual vs predicted", "Total volume reconciles, but product mix is weaker underneath.", chart_rows)

    product_day_cat = weighted_metrics(product_day, ["product_reference_group"])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bar_category_error(ax, product_day_cat, "Product-day WMAPE by category", 0.10, PURPLE)
    save(fig, "06_product_day_error_bars.png", "Product-day category errors", "Product-day is the least mature layer and misses the 10% reference goal.", chart_rows)

    index = pd.DataFrame(chart_rows)
    index.to_csv(CHART_DIR / "chart_index.csv", index=False, encoding="utf-8-sig")

    summary = [
        "# Error Charts",
        "",
        "Generated by `src/16_error_charts.py`.",
        "",
        "This focused chart pack intentionally contains six charts only: three predicted-vs-actual line charts and three category-error bar charts.",
        "",
        "## Current Reading",
        "",
        f"- Category-week parent WMAPE: `{parent_wmape:.6f}`.",
        f"- Category-day total WMAPE: `{cat_day_wmape:.6f}`.",
        f"- Product-week total WMAPE after reconciliation: `{prod_week_wmape:.6f}`.",
        "- Category-week is the reliable anchor; daily timing and product allocation are the main remaining prediction gaps.",
        "",
        "## Chart Index",
        "",
        index.to_markdown(index=False),
        "",
    ]
    (REPORT_DIR / "error_visualization_report.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"Wrote {len(chart_rows)} charts to {CHART_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
