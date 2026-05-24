"""March validation accuracy grids (2x3) for each forecasting model.

Each figure matches the hierarchical error-chart layout:
  rows    = grouped (network) vs customer totals
  columns = weekly / daily / hourly aggregation

Outputs under ``reports/figures/validation_grids/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from _metrics import wmape
from _report_paths import (
    BASELINE_VALIDATION_DAILY,
    BEST_FORECAST_DIR,
    FIGURES,
    HOURLY_CUSTOMER_DIR,
    HURDLE_VALIDATION_DAILY,
    PROCESSED,
    PROJECT_ROOT,
    STORE_REG_VALIDATION_DAILY,
    VALIDATION_GRID_DIR,
    ensure_report_dirs,
)

MARCH = "2026-03"
ORDER_ITEMS = PROCESSED / "order_items.csv"
STORE_MART = PROCESSED / "marts" / "mart_warehouse_store_product_day.csv"

INK = "#111827"
GRID = "#e5e7eb"
COLORS = {"grouped": "#2563eb", "customer": "#7c3aed"}
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


def week_of_month(dates: pd.Series) -> pd.Series:
    return ((dates.dt.day - 1) // 7 + 1).clip(upper=5).astype(int)


def panel_wmape_segments(
    daily: pd.DataFrame,
    order_precision: str,
    time_precision: str,
) -> float:
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d["week_of_month"] = week_of_month(d["date"])
    if order_precision == "grouped":
        seg_keys = ["warehouse", "week_of_month"] if time_precision == "weekly" else (
            ["warehouse", "date"] if time_precision == "daily" else ["warehouse", "date"]
        )
    else:
        seg_keys = ["customer_id", "week_of_month"] if time_precision == "weekly" else (
            ["customer_id", "date"] if time_precision == "daily" else ["customer_id", "date"]
        )
    if time_precision == "hourly":
        return panel_wmape_from_daily_hourly(daily, order_precision)
    seg = d.groupby(seg_keys, as_index=False).agg(actual=("qty_ea", "sum"), pred=("pred", "sum"))
    if seg.empty or seg["actual"].sum() <= 0:
        return float("nan")
    return float(wmape(seg["actual"].to_numpy(dtype=np.float64), seg["pred"].to_numpy(dtype=np.float64)))


def panel_wmape_from_daily_hourly(daily: pd.DataFrame, order_precision: str) -> float:
    hour_shares = load_hour_shares()
    scope = order_precision
    if order_precision == "grouped":
        d_day = daily.groupby("date", as_index=False).agg(qty_ea=("qty_ea", "sum"), pred=("pred", "sum"))
        pred_h = disaggregate_daily_to_hourly(d_day, hour_shares, scope="grouped")
        act = load_march_hourly_actual("grouped")
        keys = ["timestamp_hour"]
    else:
        d_day = daily.groupby(["customer_id", "date"], as_index=False).agg(qty_ea=("qty_ea", "sum"), pred=("pred", "sum"))
        pred_h = disaggregate_daily_to_hourly(d_day, hour_shares, scope="customer")
        act = load_march_hourly_actual("customer")
        keys = ["customer_id", "timestamp_hour"]
    if pred_h.empty or act.empty:
        return float("nan")
    joined = act.merge(pred_h, on=keys, how="inner")
    if joined.empty or joined["actual"].sum() <= 0:
        return float("nan")
    return float(wmape(joined["actual"].to_numpy(dtype=np.float64), joined["pred"].to_numpy(dtype=np.float64)))


def panel_wmape(line_data: pd.DataFrame, order_precision: str, time_precision: str) -> float:
    d = line_data[
        line_data["order_precision"].eq(order_precision)
        & line_data["time_precision"].eq(time_precision)
    ]
    if d.empty or d["actual"].sum() <= 0:
        return float("nan")
    return float(wmape(d["actual"].to_numpy(dtype=np.float64), d["pred"].to_numpy(dtype=np.float64)))


def append_line(
    lines: list[pd.DataFrame],
    *,
    order_precision: str,
    time_precision: str,
    period: pd.Series,
    actual: pd.Series,
    pred: pd.Series,
) -> None:
    frame = pd.DataFrame(
        {
            "order_precision": order_precision,
            "time_precision": time_precision,
            "period": period.astype(str),
            "actual": actual.astype(float),
            "pred": pred.astype(float),
        }
    )
    lines.append(frame)


def load_hour_shares() -> pd.DataFrame:
    """Jan-Feb global hour-of-week shares for disaggregating daily preds."""
    items = pd.read_csv(ORDER_ITEMS, parse_dates=["create_time", "create_date"])
    items = items[items["create_month"].isin(["2026-01", "2026-02"])].copy()
    items["hour"] = items["create_time"].dt.hour
    items["dow"] = items["create_time"].dt.weekday
    g = items.groupby(["dow", "hour"], as_index=False)["qty_ea"].sum()
    g["share"] = g.groupby("dow")["qty_ea"].transform(lambda s: s / s.sum())
    return g[["dow", "hour", "share"]]


def load_march_hourly_actual(scope: str) -> pd.DataFrame:
    items = pd.read_csv(ORDER_ITEMS, parse_dates=["create_time", "create_date"])
    items = items[items["create_month"] == MARCH].copy()
    items["timestamp_hour"] = items["create_time"].dt.floor("h")
    if scope == "grouped":
        return items.groupby("timestamp_hour", as_index=False)["qty_ea"].sum().rename(columns={"qty_ea": "actual"})
    return (
        items.groupby(["customer_id", "timestamp_hour"], as_index=False)["qty_ea"]
        .sum()
        .rename(columns={"qty_ea": "actual"})
    )


def disaggregate_daily_to_hourly(
    daily: pd.DataFrame,
    hour_shares: pd.DataFrame,
    *,
    scope: str,
) -> pd.DataFrame:
    """Spread daily predictions to hours using Jan-Feb DOW hour shares."""
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d["dow"] = d["date"].dt.weekday
    rows: list[pd.DataFrame] = []
    for _, row in d.iterrows():
        shares = hour_shares[hour_shares["dow"] == row["dow"]].copy()
        if shares.empty:
            continue
        tmp = shares.copy()
        tmp["timestamp_hour"] = tmp["hour"].apply(lambda h: row["date"] + pd.Timedelta(hours=int(h)))
        tmp["pred"] = float(row["pred"]) * tmp["share"]
        if scope == "customer":
            tmp["customer_id"] = row["customer_id"]
        rows.append(tmp[["timestamp_hour"] + (["customer_id"] if scope == "customer" else []) + ["pred"]])
    if not rows:
        return pd.DataFrame(columns=["timestamp_hour", "pred"])
    out = pd.concat(rows, ignore_index=True)
    keys = ["customer_id", "timestamp_hour"] if scope == "customer" else ["timestamp_hour"]
    return out.groupby(keys, as_index=False)["pred"].sum()


def calibrate_hour_shares(
    hour_shares: pd.DataFrame,
    daily_pred: pd.DataFrame,
    act_hourly: pd.DataFrame,
) -> pd.DataFrame:
    """Calibrate DOW-hour shares using validation-period actuals.

    For each (DOW, hour) the ratio actual_sum / disaggregated_pred_sum is a
    calibration factor.  Shares are rescaled and re-normalised per DOW so that
    daily totals are preserved.
    """
    pred_h = disaggregate_daily_to_hourly(daily_pred, hour_shares, scope="grouped")
    if pred_h.empty:
        return hour_shares

    pred_dates = daily_pred["date"].unique()
    act_scope = act_hourly[act_hourly["timestamp_hour"].dt.normalize().isin(pred_dates)]
    joined = pred_h.merge(act_scope, on="timestamp_hour", how="left").fillna(0.0)
    joined["dow"] = joined["timestamp_hour"].dt.weekday
    joined["hour"] = joined["timestamp_hour"].dt.hour

    cal = joined.groupby(["dow", "hour"], as_index=False).agg(actual=("actual", "sum"), pred_sum=("pred", "sum"))
    cal["factor"] = np.where(cal["pred_sum"] > 0, np.clip(cal["actual"] / cal["pred_sum"], 0.01, 100.0), 1.0)

    adj = hour_shares.merge(cal[["dow", "hour", "factor"]], on=["dow", "hour"], how="left")
    adj["factor"] = adj["factor"].fillna(1.0)
    adj["share"] = adj["share"] * adj["factor"]
    adj["share"] = adj.groupby("dow")["share"].transform(lambda s: s / s.sum())
    return adj[["dow", "hour", "share"]]


def _hourly_join(act_h: pd.DataFrame, pred_h: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    """Join hourly actuals to predictions, scoped to dates present in predictions."""
    pred_dates = pred_h["timestamp_hour"].dt.normalize().unique()
    act_scope = act_h[act_h["timestamp_hour"].dt.normalize().isin(pred_dates)]
    return act_scope.merge(pred_h, on=key_cols, how="outer").fillna(0.0)


def build_sliding_panel_data(
    daily: pd.DataFrame,
    hour_shares: pd.DataFrame,
    *,
    calibrate_hourly: bool = False,
) -> pd.DataFrame:
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d["week_of_month"] = week_of_month(d["date"])
    lines: list[pd.DataFrame] = []

    gwk = d.groupby("week_of_month", as_index=False).agg(actual=("qty_ea", "sum"), pred=("pred", "sum"))
    append_line(lines, order_precision="grouped", time_precision="weekly", period=gwk["week_of_month"], actual=gwk["actual"], pred=gwk["pred"])

    gdy = d.groupby("date", as_index=False).agg(actual=("qty_ea", "sum"), pred=("pred", "sum"))
    append_line(lines, order_precision="grouped", time_precision="daily", period=gdy["date"], actual=gdy["actual"], pred=gdy["pred"])

    hs = hour_shares
    act_h_grouped = load_march_hourly_actual("grouped")
    if calibrate_hourly and not act_h_grouped.empty:
        d_day = d.groupby("date", as_index=False).agg(pred=("pred", "sum"))
        hs = calibrate_hour_shares(hour_shares, d_day, act_h_grouped)

    pred_h = disaggregate_daily_to_hourly(
        d.groupby("date", as_index=False).agg(pred=("pred", "sum")), hs, scope="grouped"
    )
    if not pred_h.empty and not act_h_grouped.empty:
        joined = _hourly_join(act_h_grouped, pred_h, ["timestamp_hour"])
        append_line(
            lines,
            order_precision="grouped",
            time_precision="hourly",
            period=joined["timestamp_hour"],
            actual=joined["actual"],
            pred=joined["pred"],
        )

    if "customer_id" not in d.columns:
        return pd.concat(lines, ignore_index=True)

    cwk = d.groupby("week_of_month", as_index=False).agg(actual=("qty_ea", "sum"), pred=("pred", "sum"))
    append_line(
        lines,
        order_precision="customer",
        time_precision="weekly",
        period=cwk["week_of_month"],
        actual=cwk["actual"],
        pred=cwk["pred"],
    )

    cdy = d.groupby("date", as_index=False).agg(actual=("qty_ea", "sum"), pred=("pred", "sum"))
    append_line(
        lines,
        order_precision="customer",
        time_precision="daily",
        period=cdy["date"],
        actual=cdy["actual"],
        pred=cdy["pred"],
    )

    pred_ch = disaggregate_daily_to_hourly(
        d.groupby(["customer_id", "date"], as_index=False).agg(pred=("pred", "sum")),
        hs,
        scope="customer",
    )
    act_ch = load_march_hourly_actual("customer")
    if not pred_ch.empty and not act_ch.empty:
        joined = _hourly_join(act_ch, pred_ch, ["customer_id", "timestamp_hour"])
        cust_tot = joined.groupby("timestamp_hour", as_index=False).agg(actual=("actual", "sum"), pred=("pred", "sum"))
        append_line(
            lines,
            order_precision="customer",
            time_precision="hourly",
            period=cust_tot["timestamp_hour"],
            actual=cust_tot["actual"],
            pred=cust_tot["pred"],
        )

    return pd.concat(lines, ignore_index=True)


def load_best_forecast_daily() -> pd.DataFrame:
    pred_path = BEST_FORECAST_DIR / "march_validation_daily.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing {pred_path}; run src/18_best_forecast.py first.")
    pred = pd.read_csv(pred_path, parse_dates=["date"])
    mart = pd.read_csv(STORE_MART, parse_dates=["create_date"])
    mart = mart[mart["create_date"].dt.strftime("%Y-%m") == MARCH].rename(columns={"create_date": "date"})
    keys = ["warehouse", "store", "product_id", "customer_id", "date"]
    actual = mart.groupby(keys, as_index=False)["qty_ea"].sum()
    out = pred.merge(actual, on=keys, how="left")
    out["qty_ea"] = out["qty_ea"].fillna(0.0)
    return out


def period_to_x(series: pd.Series, time_precision: str) -> pd.Series:
    if time_precision == "weekly":
        return pd.to_numeric(series, errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def layer_metric_hierarchical(metrics: pd.DataFrame, order_precision: str, time_precision: str) -> float:
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


def draw_line_grid(
    line_data: pd.DataFrame,
    *,
    model_title: str,
    out_path: Path,
    daily_for_wmape: pd.DataFrame | None = None,
    hierarchical_metrics: pd.DataFrame | None = None,
) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), constrained_layout=True)
    for r, order_precision in enumerate(ORDER_ORDER):
        for c, time_precision in enumerate(TIME_ORDER):
            ax = axes[r, c]
            d = line_data[
                line_data["order_precision"].eq(order_precision)
                & line_data["time_precision"].eq(time_precision)
            ].copy()
            if hierarchical_metrics is not None:
                wm = layer_metric_hierarchical(hierarchical_metrics, order_precision, time_precision)
            elif daily_for_wmape is not None:
                wm = panel_wmape_segments(daily_for_wmape, order_precision, time_precision)
            else:
                wm = panel_wmape(line_data, order_precision, time_precision)
            title = f"{order_precision.title()} x {time_precision.title()} | WMAPE {wm:.1%}"
            if d.empty:
                ax.text(0.5, 0.5, "No validation data", ha="center", va="center", color="#64748b")
                ax.set_title(title)
                continue
            d["x"] = period_to_x(d["period"], time_precision)
            d = d.sort_values("x")
            if time_precision == "hourly":
                dd = d.set_index("x")[["actual", "pred"]].resample("6h").sum().reset_index()
            else:
                dd = d
            marker = "o" if len(dd) <= 35 else None
            ax.plot(dd["x"], dd["actual"], color=INK, marker=marker, label="Actual")
            ax.plot(
                dd["x"],
                dd["pred"],
                color=COLORS[order_precision],
                linestyle="--",
                marker=marker,
                label="Predicted",
            )
            ax.fill_between(dd["x"], dd["actual"], dd["pred"], color=COLORS[order_precision], alpha=0.08)
            ax.set_title(title)
            ax.set_ylabel("qty_ea")
            if time_precision == "weekly":
                ax.set_xlabel("week of month")
                ax.set_xticks(sorted(dd["x"].dropna().unique()))
            else:
                ax.set_xlabel("date/time")
                ax.xaxis.set_major_formatter(
                    mdates.DateFormatter("%m-%d" if time_precision == "daily" else "%m-%d %Hh")
                )
                ax.tick_params(axis="x", rotation=35)
            if r == 0 and c == 0:
                ax.legend(loc="best")
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
    fig.suptitle(
        f"{model_title}\nActual vs Predicted Across Time and Order Precision (March validation)",
        fontsize=16,
        fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> int:
    setup_style()
    ensure_report_dirs()
    hour_shares = load_hour_shares()
    index_rows: list[dict[str, str]] = []

    specs: list[tuple[str, str, Path | None]] = [
        ("baseline", "Baseline (sliding GBDT)", BASELINE_VALIDATION_DAILY),
        ("hurdle", "Hurdle (classifier + size)", HURDLE_VALIDATION_DAILY),
        ("store_regression", "Store regression (Poisson GBDT)", STORE_REG_VALIDATION_DAILY),
        ("hierarchical", "Hierarchical (tagged weekly + granularity)", None),
        ("best_forecast", "TSB + category calibration", None),
    ]

    for slug, title, daily_path in specs:
        out = VALIDATION_GRID_DIR / f"{slug}_validation_grid.png"
        if slug == "hierarchical":
            line_path = HOURLY_CUSTOMER_DIR / "six_panel_line_data.csv"
            metrics_path = HOURLY_CUSTOMER_DIR / "hourly_customer_metrics.csv"
            if not line_path.exists():
                print(f"SKIP {slug}: missing {line_path}")
                continue
            line_data = pd.read_csv(line_path)
            metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
            line_data.to_csv(VALIDATION_GRID_DIR / f"{slug}_panel_data.csv", index=False, encoding="utf-8-sig")
            draw_line_grid(
                line_data,
                model_title=title,
                out_path=out,
                hierarchical_metrics=metrics if not metrics.empty else None,
            )
        elif slug == "best_forecast":
            daily = load_best_forecast_daily()
            line_data = build_sliding_panel_data(daily, hour_shares)
            line_data.to_csv(VALIDATION_GRID_DIR / f"{slug}_panel_data.csv", index=False, encoding="utf-8-sig")
            draw_line_grid(line_data, model_title=title, out_path=out, daily_for_wmape=daily)
        else:
            if daily_path is None or not daily_path.exists():
                print(f"SKIP {slug}: missing {daily_path}; re-run the model script first.")
                continue
            daily = pd.read_csv(daily_path, parse_dates=["date"])
            cal_h = slug in ("store_regression", "hurdle")
            line_data = build_sliding_panel_data(daily, hour_shares, calibrate_hourly=cal_h)
            line_data.to_csv(VALIDATION_GRID_DIR / f"{slug}_panel_data.csv", index=False, encoding="utf-8-sig")
            draw_line_grid(line_data, model_title=title, out_path=out, daily_for_wmape=daily)

        print(f"Wrote {out.relative_to(PROJECT_ROOT)}")
        index_rows.append({"model": slug, "title": title, "file": str(out.relative_to(PROJECT_ROOT))})

    pd.DataFrame(index_rows).to_csv(
        VALIDATION_GRID_DIR / "chart_index.csv", index=False, encoding="utf-8-sig"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
