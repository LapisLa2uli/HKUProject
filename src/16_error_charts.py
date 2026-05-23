"""Composite error charts for the active hierarchy.

The chart pack now has exactly two figures:

1. a 2x3 grid of actual-vs-predicted line plots;
2. a 2x3 grid of WMAPE/error bar plots.

Rows are order precision: grouped category vs individual customer-category.
Columns are time precision: weekly, daily, hourly.
"""

from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "production_granularity"
HOURLY_DIR = REPORT_DIR / "hourly_customer"
CHART_DIR = REPORT_DIR / "error_charts"
LINE_DATA = HOURLY_DIR / "six_panel_line_data.csv"
BAR_DATA = HOURLY_DIR / "six_panel_bar_data.csv"
METRICS = HOURLY_DIR / "hourly_customer_metrics.csv"

INK = "#111827"
BLUE = "#2563eb"
MUTED = "#64748b"
GRID = "#e5e7eb"
RED = "#dc2626"
COLORS = {
    "grouped": "#2563eb",
    "customer": "#7c3aed",
}
TIME_ORDER = ["weekly", "daily", "hourly"]
ORDER_ORDER = ["grouped", "customer"]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#cbd5e1",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.75,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.labelcolor": "#334155",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "font.size": 9,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
        }
    )


def reset_chart_dir() -> None:
    if CHART_DIR.exists():
        shutil.rmtree(CHART_DIR)
    CHART_DIR.mkdir(parents=True, exist_ok=True)


def period_to_x(series: pd.Series, time_precision: str) -> pd.Series:
    if time_precision == "weekly":
        return pd.to_numeric(series, errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def layer_metric(metrics: pd.DataFrame, order_precision: str, time_precision: str) -> float:
    layer_map = {
        ("grouped", "weekly"): "grouped_week",
        ("grouped", "daily"): "grouped_day",
        ("grouped", "hourly"): "grouped_hour",
        ("customer", "weekly"): "customer_week",
        ("customer", "daily"): "customer_day",
        ("customer", "hourly"): "customer_hour",
    }
    layer = layer_map[(order_precision, time_precision)]
    rows = metrics[metrics["layer"].eq(layer)].sort_values("wmape")
    return float(rows.iloc[0]["wmape"]) if not rows.empty else float("nan")


def draw_line_grid(line_data: pd.DataFrame, metrics: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), constrained_layout=True)
    for r, order_precision in enumerate(ORDER_ORDER):
        for c, time_precision in enumerate(TIME_ORDER):
            ax = axes[r, c]
            d = line_data[
                line_data["order_precision"].eq(order_precision)
                & line_data["time_precision"].eq(time_precision)
            ].copy()
            wm = layer_metric(metrics, order_precision, time_precision)
            title = f"{order_precision.title()} x {time_precision.title()} | WMAPE {wm:.1%}"
            if d.empty:
                ax.text(0.5, 0.5, "No validation data", ha="center", va="center", color=MUTED)
                ax.set_title(title)
                continue
            d["x"] = period_to_x(d["period"], time_precision)
            d = d.sort_values("x")
            if time_precision == "hourly":
                # Hourly validation has 744 points. Resample to 6-hour windows
                # for a readable line chart without changing the saved metrics.
                dd = d.set_index("x")[["actual", "pred"]].resample("6h").sum().reset_index()
            else:
                dd = d
            ax.plot(dd["x"], dd["actual"], color=INK, marker="o" if len(dd) <= 35 else None, label="Actual")
            ax.plot(dd["x"], dd["pred"], color=COLORS[order_precision], linestyle="--", marker="o" if len(dd) <= 35 else None, label="Predicted")
            ax.fill_between(dd["x"], dd["actual"], dd["pred"], color=COLORS[order_precision], alpha=0.08)
            ax.set_title(title)
            ax.set_ylabel("qty_ea")
            if time_precision == "weekly":
                ax.set_xlabel("week of month")
                ax.set_xticks(sorted(dd["x"].dropna().unique()))
            else:
                ax.set_xlabel("date/time")
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d" if time_precision == "daily" else "%m-%d %Hh"))
                ax.tick_params(axis="x", rotation=35)
            if r == 0 and c == 0:
                ax.legend(loc="best")
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
    fig.suptitle("Actual vs Predicted Across Time and Order Precision", fontsize=18, fontweight="bold")
    path = CHART_DIR / "01_actual_pred_line_grid.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_bar_grid(bar_data: pd.DataFrame, metrics: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    for r, order_precision in enumerate(ORDER_ORDER):
        for c, time_precision in enumerate(TIME_ORDER):
            ax = axes[r, c]
            d = bar_data[
                bar_data["order_precision"].eq(order_precision)
                & bar_data["time_precision"].eq(time_precision)
            ].copy()
            wm = layer_metric(metrics, order_precision, time_precision)
            title = f"{order_precision.title()} x {time_precision.title()} Errors"
            if d.empty:
                ax.text(0.5, 0.5, "No validation data", ha="center", va="center", color=MUTED)
                ax.set_title(title)
                continue
            d = d.sort_values("abs_error", ascending=True).tail(10)
            labels = d["segment"].astype(str).str.slice(0, 36)
            ax.barh(labels, d["wmape"], color=COLORS[order_precision], alpha=0.90)
            ax.axvline(wm, color=RED, linestyle="--", linewidth=1.2, label=f"Layer {wm:.1%}")
            ax.set_title(title)
            ax.set_xlabel("segment WMAPE")
            ax.legend(loc="lower right")
            for i, value in enumerate(d["wmape"]):
                if np.isfinite(value):
                    ax.text(value, i, f" {value:.0%}", va="center", ha="left", color=MUTED, fontsize=8)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
    fig.suptitle("Largest Error Segments Across Time and Order Precision", fontsize=18, fontweight="bold")
    path = CHART_DIR / "02_error_bar_grid.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    setup_style()
    reset_chart_dir()
    if not LINE_DATA.exists() or not BAR_DATA.exists() or not METRICS.exists():
        raise FileNotFoundError(
            "Run `python src/17_hourly_customer_granularity.py` before generating composite charts."
        )
    line_data = pd.read_csv(LINE_DATA)
    bar_data = pd.read_csv(BAR_DATA)
    metrics = pd.read_csv(METRICS)
    line_path = draw_line_grid(line_data, metrics)
    bar_path = draw_bar_grid(bar_data, metrics)
    index = pd.DataFrame(
        [
            {
                "file": str(line_path.relative_to(PROJECT_ROOT)),
                "title": "Composite actual-vs-predicted line grid",
                "note": "2x3 panels: grouped/customer by weekly/daily/hourly.",
            },
            {
                "file": str(bar_path.relative_to(PROJECT_ROOT)),
                "title": "Composite error bar grid",
                "note": "2x3 panels: largest WMAPE/error segments by precision.",
            },
        ]
    )
    index.to_csv(CHART_DIR / "chart_index.csv", index=False, encoding="utf-8-sig")
    summary = [
        "# Error Charts",
        "",
        "Generated by `src/16_error_charts.py`.",
        "",
        "This chart pack contains two composite figures, covering six forecast views:",
        "",
        "- grouped x weekly, daily, hourly;",
        "- customer x weekly, daily, hourly.",
        "",
        "## Chart Index",
        "",
        index.to_markdown(index=False),
        "",
    ]
    (REPORT_DIR / "error_visualization_report.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"Wrote composite charts to {CHART_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
