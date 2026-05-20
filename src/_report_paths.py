"""Canonical paths for pipeline CSV outputs under ``reports/``."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "processed"
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"

BASELINE_DIR = REPORTS / "baseline"
HURDLE_DIR = REPORTS / "hurdle"
STORE_REG_DIR = REPORTS / "store_regression"
COMPARISON_DIR = REPORTS / "comparison"
HEATMAPS_DIR = REPORTS / "heatmaps"

# Baseline
BASELINE_VALIDATION_MONTHLY = BASELINE_DIR / "baseline_validation_monthly_actual_vs_pred.csv"
BASELINE_VALIDATION_OVERALL = BASELINE_DIR / "baseline_validation_overall.csv"
BASELINE_VALIDATION_PER_WAREHOUSE = BASELINE_DIR / "baseline_validation_per_warehouse.csv"
BASELINE_VALIDATION_PER_CUSTOMER = BASELINE_DIR / "baseline_validation_per_customer.csv"
BASELINE_APRIL_DAILY = BASELINE_DIR / "april_forecast_daily.csv"
BASELINE_APRIL_MONTHLY_WCP = BASELINE_DIR / "april_forecast_monthly_by_warehouse_customer_product.csv"
BASELINE_APRIL_MONTHLY_CP = BASELINE_DIR / "april_forecast_monthly_by_customer_product.csv"
BASELINE_APRIL_MONTHLY_WH = BASELINE_DIR / "april_forecast_monthly_by_warehouse.csv"

# Hurdle
HURDLE_VALIDATION_MONTHLY = HURDLE_DIR / "hurdle_validation_monthly_actual_vs_pred.csv"
HURDLE_VALIDATION_OVERALL = HURDLE_DIR / "hurdle_validation_overall.csv"
HURDLE_VALIDATION_PER_WAREHOUSE = HURDLE_DIR / "hurdle_validation_per_warehouse.csv"
HURDLE_VALIDATION_PER_CUSTOMER = HURDLE_DIR / "hurdle_validation_per_customer.csv"
HURDLE_VALIDATION_CLASSIFIER = HURDLE_DIR / "hurdle_validation_classifier.csv"
HURDLE_VALIDATION_INTERMITTENT = HURDLE_DIR / "hurdle_validation_intermittent_loss.csv"
HURDLE_VALIDATION_PLANNING = HURDLE_DIR / "hurdle_validation_planning_loss.csv"
HURDLE_APRIL_DAILY_PATTERN = HURDLE_DIR / "april_forecast_hurdle_daily_pattern.csv"
HURDLE_APRIL_DAILY_STORE = HURDLE_DIR / "april_forecast_hurdle_daily_store.csv"
HURDLE_APRIL_DAILY_NETWORK = HURDLE_DIR / "april_forecast_hurdle_daily.csv"
HURDLE_APRIL_CALIBRATION = HURDLE_DIR / "hurdle_april_calibration.csv"
HURDLE_APRIL_STORE_CALIBRATION = HURDLE_DIR / "hurdle_april_store_calibration.csv"
HURDLE_APRIL_MONTHLY_WCP = HURDLE_DIR / "april_forecast_hurdle_monthly_by_warehouse_customer_product.csv"
HURDLE_APRIL_MONTHLY_STORE = HURDLE_DIR / "april_forecast_hurdle_monthly_by_store_product.csv"
HURDLE_APRIL_MONTHLY_CP = HURDLE_DIR / "april_forecast_hurdle_monthly_by_customer_product.csv"
HURDLE_APRIL_MONTHLY_WH = HURDLE_DIR / "april_forecast_hurdle_monthly_by_warehouse.csv"

# Store regression
STORE_REG_VALIDATION_MONTHLY = STORE_REG_DIR / "store_regression_validation_monthly_actual_vs_pred.csv"
STORE_REG_VALIDATION_OVERALL = STORE_REG_DIR / "store_regression_validation_overall.csv"
STORE_REG_VALIDATION_PER_WAREHOUSE = STORE_REG_DIR / "store_regression_validation_per_warehouse.csv"
STORE_REG_VALIDATION_PER_CUSTOMER = STORE_REG_DIR / "store_regression_validation_per_customer.csv"
STORE_REG_VALIDATION_PLANNING = STORE_REG_DIR / "store_regression_validation_planning_loss.csv"
STORE_REG_APRIL_DAILY = STORE_REG_DIR / "april_forecast_store_regression_daily.csv"
STORE_REG_APRIL_MONTHLY_STORE = STORE_REG_DIR / "april_forecast_store_regression_monthly_by_store_product.csv"
STORE_REG_APRIL_MONTHLY_WCP = (
    STORE_REG_DIR / "april_forecast_store_regression_monthly_by_warehouse_customer_product.csv"
)

# Model comparison
COMPARISON_CSV = COMPARISON_DIR / "model_comparison.csv"
COMPARISON_MD = REPORTS / "model_comparison.md"
COMPARISON_APRIL_WCP = COMPARISON_DIR / "model_comparison_april_by_warehouse_customer_product.csv"
COMPARISON_APRIL_WH = COMPARISON_DIR / "model_comparison_april_by_warehouse.csv"
COMPARISON_APRIL_CUST = COMPARISON_DIR / "model_comparison_april_by_customer.csv"
COMPARISON_APRIL_CP = COMPARISON_DIR / "model_comparison_april_by_customer_product.csv"

# Heatmap row-order metadata
HEATMAP_ROW_BASELINE = HEATMAPS_DIR / "april_store_heatmap_row_order_baseline.csv"
HEATMAP_ROW_HURDLE_PATTERN = HEATMAPS_DIR / "april_store_heatmap_row_order_hurdle_pattern.csv"
HEATMAP_ROW_HURDLE_STORE = HEATMAPS_DIR / "april_store_heatmap_row_order_hurdle_store.csv"
HEATMAP_ROW_STORE_REG = HEATMAPS_DIR / "april_store_heatmap_row_order_store_regression.csv"
HEATMAP_HIST_HURDLE_PATTERN = HEATMAPS_DIR / "store_history_april_heatmap_row_order_hurdle_pattern.csv"
HEATMAP_HIST_HURDLE_STORE = HEATMAPS_DIR / "store_history_april_heatmap_row_order_hurdle_store.csv"
HEATMAP_HIST_HURDLE_PATTERN_TOP20 = HEATMAPS_DIR / "store_history_april_heatmap_row_order_hurdle_pattern_top20.csv"
HEATMAP_HIST_HURDLE_STORE_TOP20 = HEATMAPS_DIR / "store_history_april_heatmap_row_order_hurdle_store_top20.csv"
HEATMAP_HIST_STORE_REG = HEATMAPS_DIR / "store_history_april_heatmap_row_order_store_regression.csv"
HEATMAP_HIST_STORE_REG_TOP20 = HEATMAPS_DIR / "store_history_april_heatmap_row_order_store_regression_top20.csv"


def ensure_report_dirs() -> None:
    for d in (
        REPORTS,
        FIGURES,
        BASELINE_DIR,
        HURDLE_DIR,
        STORE_REG_DIR,
        COMPARISON_DIR,
        HEATMAPS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def resolve_legacy_csv(name: str) -> Path:
    """Prefer grouped subfolder; fall back to flat ``reports/`` for old layouts."""
    p = REPORTS / name
    if p.exists():
        return p
    if name.startswith("baseline_") or name == "april_forecast_daily.csv":
        return BASELINE_DIR / name
    if name.startswith("hurdle_") or name.startswith("april_forecast_hurdle"):
        return HURDLE_DIR / name
    if name.startswith("store_regression_") or name.startswith("april_forecast_store_regression"):
        return STORE_REG_DIR / name
    if name.startswith("model_comparison"):
        return COMPARISON_DIR / name
    if "heatmap" in name:
        return HEATMAPS_DIR / name
    if name.startswith("april_forecast_monthly"):
        return BASELINE_DIR / name
    return p
