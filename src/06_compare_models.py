"""Compare baseline vs hurdle forecasts and March validation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from _metrics import bias_ratio, wmape

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"

# Documented pre-CNY-handling baseline bias ratio (March monthly total)
BASELINE_BIAS_RATIO_PRE_CNY = -0.187


def load_baseline_march() -> tuple[pd.DataFrame, dict[str, float]]:
    monthly = pd.read_csv(REPORTS / "baseline_validation_monthly_actual_vs_pred.csv")
    overall_df = pd.read_csv(REPORTS / "baseline_validation_overall.csv")
    keys = overall_df.set_index("metric")["value"].to_dict()
    return monthly, keys


def load_hurdle_march() -> tuple[pd.DataFrame, dict[str, float]]:
    monthly = pd.read_csv(REPORTS / "hurdle_validation_monthly_actual_vs_pred.csv")
    overall_df = pd.read_csv(REPORTS / "hurdle_validation_overall.csv")
    keys = overall_df.set_index("metric")["value"].to_dict()
    return monthly, keys


def rollup_hurdle_april() -> pd.DataFrame:
    daily = pd.read_csv(REPORTS / "april_forecast_hurdle_daily.csv", parse_dates=["date"])
    return (
        daily.groupby(["warehouse", "customer_id", "product_id"], as_index=False)["qty_ea"]
        .sum()
        .rename(columns={"qty_ea": "hurdle_predicted_qty_ea_april"})
    )


def load_baseline_april_wcp() -> pd.DataFrame:
    df = pd.read_csv(REPORTS / "april_forecast_monthly_by_warehouse_customer_product.csv")
    return df.rename(
        columns={"predicted_qty_ea_april": "baseline_predicted_qty_ea_april"}
    )


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)

    base_m, base_keys = load_baseline_march()
    hur_m, hur_keys = load_hurdle_march()

    base_m = base_m.rename(columns={"actual": "march_actual", "pred": "march_pred_baseline"})
    hur_m = hur_m.rename(columns={"actual": "march_actual_h", "pred": "march_pred_hurdle"})

    post_bias = bias_ratio(
        base_m["march_actual"].values,
        base_m["march_pred_baseline"].values,
    )

    base_daily_mae = base_keys.get("daily_mae", np.nan)
    hur_daily_mae = hur_keys.get("daily_mae", np.nan)

    h_act = hur_m["march_actual_h"].values
    h_pred = hur_m["march_pred_hurdle"].values

    base_apr = load_baseline_april_wcp()
    hur_apr = rollup_hurdle_april()
    cmp_apr = base_apr.merge(
        hur_apr,
        on=["warehouse", "customer_id", "product_id"],
        how="outer",
    ).fillna(0.0)
    cmp_apr["diff"] = (
        cmp_apr["hurdle_predicted_qty_ea_april"] - cmp_apr["baseline_predicted_qty_ea_april"]
    )
    denom = cmp_apr["baseline_predicted_qty_ea_april"].replace(0, np.nan)
    cmp_apr["pct_diff_vs_baseline"] = 100.0 * cmp_apr["diff"] / denom

    wh_apr = (
        cmp_apr.groupby("warehouse", as_index=False)[
            ["baseline_predicted_qty_ea_april", "hurdle_predicted_qty_ea_april"]
        ]
        .sum()
    )
    wh_apr["diff"] = (
        wh_apr["hurdle_predicted_qty_ea_april"] - wh_apr["baseline_predicted_qty_ea_april"]
    )

    cust_apr = (
        cmp_apr.groupby("customer_id", as_index=False)[
            ["baseline_predicted_qty_ea_april", "hurdle_predicted_qty_ea_april"]
        ]
        .sum()
    )
    cust_apr["diff"] = (
        cust_apr["hurdle_predicted_qty_ea_april"] - cust_apr["baseline_predicted_qty_ea_april"]
    )

    cp_apr = (
        cmp_apr.groupby(["customer_id", "product_id"], as_index=False)[
            ["baseline_predicted_qty_ea_april", "hurdle_predicted_qty_ea_april"]
        ]
        .sum()
    )
    cp_apr["diff"] = (
        cp_apr["hurdle_predicted_qty_ea_april"] - cp_apr["baseline_predicted_qty_ea_april"]
    )

    summary_rows = [
        {
            "metric": "baseline_march_bias_ratio_pre_cny_handling",
            "value": BASELINE_BIAS_RATIO_PRE_CNY,
        },
        {
            "metric": "baseline_march_bias_ratio_post_cny_handling",
            "value": post_bias,
        },
        {"metric": "baseline_daily_mae", "value": base_daily_mae},
        {"metric": "hurdle_daily_mae", "value": hur_daily_mae},
        {
            "metric": "baseline_march_monthly_wmape",
            "value": wmape(base_m["march_actual"].values, base_m["march_pred_baseline"].values),
        },
        {
            "metric": "hurdle_march_monthly_wmape",
            "value": wmape(h_act, h_pred),
        },
        {
            "metric": "april_total_baseline",
            "value": float(cmp_apr["baseline_predicted_qty_ea_april"].sum()),
        },
        {
            "metric": "april_total_hurdle",
            "value": float(cmp_apr["hurdle_predicted_qty_ea_april"].sum()),
        },
    ]
    pd.DataFrame(summary_rows).to_csv(
        REPORTS / "model_comparison.csv", index=False, encoding="utf-8-sig"
    )
    cmp_apr.to_csv(
        REPORTS / "model_comparison_april_by_warehouse_customer_product.csv",
        index=False,
        encoding="utf-8-sig",
    )
    wh_apr.to_csv(
        REPORTS / "model_comparison_april_by_warehouse.csv", index=False, encoding="utf-8-sig"
    )
    cust_apr.to_csv(
        REPORTS / "model_comparison_april_by_customer.csv", index=False, encoding="utf-8-sig"
    )
    cp_apr.to_csv(
        REPORTS / "model_comparison_april_by_customer_product.csv",
        index=False,
        encoding="utf-8-sig",
    )

    md = f"""# Model comparison (baseline vs store hurdle)

## March validation — baseline bias (CNY handling)

| Metric | Value |
|--------|------:|
| Baseline bias ratio **before** CNY handling (documented) | {100*BASELINE_BIAS_RATIO_PRE_CNY:.2f}% |
| Baseline bias ratio **after** CNY handling (current run) | {100*post_bias:.2f}% |
| Baseline daily MAE | {base_daily_mae:.4f} |
| Hurdle daily MAE | {hur_daily_mae:.4f} |

## April totals (hurdle rolled up to warehouse–customer–product)

| Model | Total predicted qty_ea |
|-------|------------------------:|
| Baseline | {cmp_apr['baseline_predicted_qty_ea_april'].sum():,.0f} |
| Hurdle (rolled up) | {cmp_apr['hurdle_predicted_qty_ea_april'].sum():,.0f} |

See `model_comparison.csv` and `model_comparison_april_*.csv`.
"""
    (REPORTS / "model_comparison.md").write_text(md, encoding="utf-8")
    print("Wrote reports/model_comparison.md and related CSVs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
