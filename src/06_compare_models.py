"""Compare baseline vs hurdle vs store regression forecasts and March validation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from _metrics import bias_ratio, wmape
from _report_paths import (
    BASELINE_APRIL_MONTHLY_WCP,
    BASELINE_VALIDATION_MONTHLY,
    BASELINE_VALIDATION_OVERALL,
    COMPARISON_APRIL_CP,
    COMPARISON_APRIL_CUST,
    COMPARISON_APRIL_WH,
    COMPARISON_APRIL_WCP,
    COMPARISON_CSV,
    COMPARISON_MD,
    HURDLE_APRIL_DAILY_NETWORK,
    HURDLE_VALIDATION_MONTHLY,
    HURDLE_VALIDATION_OVERALL,
    STORE_REG_APRIL_DAILY,
    STORE_REG_VALIDATION_MONTHLY,
    STORE_REG_VALIDATION_OVERALL,
    ensure_report_dirs,
)

# Documented pre-CNY-handling baseline bias ratio (March monthly total)
BASELINE_BIAS_RATIO_PRE_CNY = -0.187


def load_baseline_march() -> tuple[pd.DataFrame, dict[str, float]]:
    monthly = pd.read_csv(BASELINE_VALIDATION_MONTHLY)
    overall_df = pd.read_csv(BASELINE_VALIDATION_OVERALL)
    keys = overall_df.set_index("metric")["value"].to_dict()
    return monthly, keys


def load_hurdle_march() -> tuple[pd.DataFrame, dict[str, float]]:
    monthly = pd.read_csv(HURDLE_VALIDATION_MONTHLY)
    overall_df = pd.read_csv(HURDLE_VALIDATION_OVERALL)
    keys = overall_df.set_index("metric")["value"].to_dict()
    return monthly, keys


def load_store_regression_march() -> tuple[pd.DataFrame, dict[str, float]]:
    monthly = pd.read_csv(STORE_REG_VALIDATION_MONTHLY)
    overall_df = pd.read_csv(STORE_REG_VALIDATION_OVERALL)
    keys = overall_df.set_index("metric")["value"].to_dict()
    return monthly, keys


def rollup_hurdle_april() -> pd.DataFrame:
    daily = pd.read_csv(HURDLE_APRIL_DAILY_NETWORK, parse_dates=["date"])
    return (
        daily.groupby(["warehouse", "customer_id", "product_id"], as_index=False)["qty_ea"]
        .sum()
        .rename(columns={"qty_ea": "hurdle_predicted_qty_ea_april"})
    )


def rollup_store_regression_april() -> pd.DataFrame:
    daily = pd.read_csv(STORE_REG_APRIL_DAILY, parse_dates=["date"])
    return (
        daily.groupby(["warehouse", "customer_id", "product_id"], as_index=False)["qty_ea"]
        .sum()
        .rename(columns={"qty_ea": "store_regression_predicted_qty_ea_april"})
    )


def load_baseline_april_wcp() -> pd.DataFrame:
    df = pd.read_csv(BASELINE_APRIL_MONTHLY_WCP)
    return df.rename(columns={"predicted_qty_ea_april": "baseline_predicted_qty_ea_april"})


def main() -> int:
    ensure_report_dirs()

    base_m, base_keys = load_baseline_march()
    hur_m, hur_keys = load_hurdle_march()
    store_m, store_keys = load_store_regression_march()

    base_m = base_m.rename(columns={"actual": "march_actual", "pred": "march_pred_baseline"})
    hur_m = hur_m.rename(columns={"actual": "march_actual_h", "pred": "march_pred_hurdle"})
    store_m = store_m.rename(
        columns={"actual": "march_actual_s", "pred": "march_pred_store_regression"}
    )

    post_bias = bias_ratio(
        base_m["march_actual"].values,
        base_m["march_pred_baseline"].values,
    )

    base_daily_mae = base_keys.get("daily_mae", np.nan)
    hur_daily_mae = hur_keys.get("daily_mae", np.nan)
    store_daily_mae = store_keys.get("daily_mae", np.nan)

    h_act = hur_m["march_actual_h"].values
    h_pred = hur_m["march_pred_hurdle"].values
    s_act = store_m["march_actual_s"].values
    s_pred = store_m["march_pred_store_regression"].values

    base_apr = load_baseline_april_wcp()
    hur_apr = rollup_hurdle_april()
    store_apr = rollup_store_regression_april()

    cmp_apr = base_apr.merge(
        hur_apr,
        on=["warehouse", "customer_id", "product_id"],
        how="outer",
    ).merge(
        store_apr,
        on=["warehouse", "customer_id", "product_id"],
        how="outer",
    ).fillna(0.0)

    cmp_apr["diff_hurdle_vs_baseline"] = (
        cmp_apr["hurdle_predicted_qty_ea_april"] - cmp_apr["baseline_predicted_qty_ea_april"]
    )
    cmp_apr["diff_store_vs_hurdle"] = (
        cmp_apr["store_regression_predicted_qty_ea_april"]
        - cmp_apr["hurdle_predicted_qty_ea_april"]
    )
    cmp_apr["diff_store_vs_baseline"] = (
        cmp_apr["store_regression_predicted_qty_ea_april"]
        - cmp_apr["baseline_predicted_qty_ea_april"]
    )
    denom = cmp_apr["baseline_predicted_qty_ea_april"].replace(0, np.nan)
    cmp_apr["pct_diff_hurdle_vs_baseline"] = 100.0 * cmp_apr["diff_hurdle_vs_baseline"] / denom

    model_cols = [
        "baseline_predicted_qty_ea_april",
        "hurdle_predicted_qty_ea_april",
        "store_regression_predicted_qty_ea_april",
    ]

    wh_apr = cmp_apr.groupby("warehouse", as_index=False)[model_cols].sum()
    wh_apr["diff_store_vs_hurdle"] = (
        wh_apr["store_regression_predicted_qty_ea_april"]
        - wh_apr["hurdle_predicted_qty_ea_april"]
    )

    cust_apr = cmp_apr.groupby("customer_id", as_index=False)[model_cols].sum()
    cust_apr["diff_store_vs_hurdle"] = (
        cust_apr["store_regression_predicted_qty_ea_april"]
        - cust_apr["hurdle_predicted_qty_ea_april"]
    )

    cp_apr = cmp_apr.groupby(["customer_id", "product_id"], as_index=False)[model_cols].sum()
    cp_apr["diff_store_vs_hurdle"] = (
        cp_apr["store_regression_predicted_qty_ea_april"]
        - cp_apr["hurdle_predicted_qty_ea_april"]
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
        {"metric": "store_regression_daily_mae", "value": store_daily_mae},
        {
            "metric": "baseline_march_monthly_wmape",
            "value": wmape(base_m["march_actual"].values, base_m["march_pred_baseline"].values),
        },
        {
            "metric": "hurdle_march_monthly_wmape",
            "value": wmape(h_act, h_pred),
        },
        {
            "metric": "store_regression_march_monthly_wmape",
            "value": wmape(s_act, s_pred),
        },
        {
            "metric": "april_total_baseline",
            "value": float(cmp_apr["baseline_predicted_qty_ea_april"].sum()),
        },
        {
            "metric": "april_total_hurdle",
            "value": float(cmp_apr["hurdle_predicted_qty_ea_april"].sum()),
        },
        {
            "metric": "april_total_store_regression",
            "value": float(cmp_apr["store_regression_predicted_qty_ea_april"].sum()),
        },
    ]
    pd.DataFrame(summary_rows).to_csv(COMPARISON_CSV, index=False, encoding="utf-8-sig")
    cmp_apr.to_csv(COMPARISON_APRIL_WCP, index=False, encoding="utf-8-sig")
    wh_apr.to_csv(COMPARISON_APRIL_WH, index=False, encoding="utf-8-sig")
    cust_apr.to_csv(COMPARISON_APRIL_CUST, index=False, encoding="utf-8-sig")
    cp_apr.to_csv(COMPARISON_APRIL_CP, index=False, encoding="utf-8-sig")

    md = f"""# Model comparison (baseline vs hurdle vs store regression)

This file is generated by `src/06_compare_models.py`. Term definitions: [docs/modeling_glossary.md](../docs/modeling_glossary.md).

## How to read the tables

All models use **sliding windows**: 14 warehouse-open lookback days → predict the next 7 days (CNY closure days skipped in lookback; see main README).

- **March validation** compares models to **known** March actuals on sliding samples whose 7-day horizon falls in March. The validation run trains on sliding samples whose horizon falls in January–February only.
- **Daily MAE** (mean absolute error): average \|actual − predicted\| in **units** (`qty_ea`) across open March days. Lower is better.
- **Monthly WMAPE**: error on **March monthly totals** per series, expressed as a fraction of total actual demand. Lower is better. Useful when you care about total volume more than exact day-by-day timing.
- **Bias ratio** (baseline): `sum(predicted) / sum(actual)` on March monthly totals. **1.00** = right total level; **1.05** ≈ 5% over-forecast.
- **April totals**: sum of all April daily predictions (hurdle and store regression are rolled up to warehouse–customer–product so they are comparable to baseline). There are **no April actuals** in the repo for automatic scoring.

## March validation

| Metric | Value |
|--------|------:|
| Baseline bias ratio **before** CNY handling (documented) | {100*BASELINE_BIAS_RATIO_PRE_CNY:.2f}% |
| Baseline bias ratio **after** CNY handling (current run) | {100*post_bias:.2f}% |
| Baseline daily MAE | {base_daily_mae:.4f} |
| Hurdle daily MAE | {hur_daily_mae:.4f} |
| Store regression daily MAE | {store_daily_mae:.4f} |
| Baseline March monthly WMAPE | {wmape(base_m['march_actual'].values, base_m['march_pred_baseline'].values):.4f} |
| Hurdle March monthly WMAPE | {wmape(h_act, h_pred):.4f} |
| Store regression March monthly WMAPE | {wmape(s_act, s_pred):.4f} |

## April totals (store regression and hurdle rolled up to warehouse–customer–product)

| Model | Total predicted qty_ea |
|-------|------------------------:|
| Baseline | {cmp_apr['baseline_predicted_qty_ea_april'].sum():,.0f} |
| Hurdle (rolled up) | {cmp_apr['hurdle_predicted_qty_ea_april'].sum():,.0f} |
| Store regression (rolled up) | {cmp_apr['store_regression_predicted_qty_ea_april'].sum():,.0f} |

## Detail files

| File | Contents |
|------|----------|
| `comparison/model_comparison.csv` | One row per summary metric |
| `comparison/model_comparison_april_by_warehouse_customer_product.csv` | April predictions by series, all models |
| `comparison/model_comparison_april_by_warehouse.csv` | April totals by warehouse |
| `comparison/model_comparison_april_by_customer.csv` | April totals by customer |
| `comparison/model_comparison_april_by_customer_product.csv` | April totals by customer × product |
"""
    COMPARISON_MD.write_text(md, encoding="utf-8")
    print(f"Wrote {COMPARISON_MD.relative_to(COMPARISON_MD.parents[1])} and comparison CSVs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
