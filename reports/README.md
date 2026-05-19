# Reports layout

Outputs from the forecasting pipeline. For definitions of **qty_ea**, **MAE**, **WMAPE**, **hurdle**, **pattern forecast**, and related terms, see [docs/modeling_glossary.md](../docs/modeling_glossary.md).

## Folders

| Folder | Contents |
|--------|----------|
| `baseline/` | Baseline model validation and April forecast CSVs |
| `hurdle/` | Hurdle model validation, calibration, and April forecast CSVs |
| `store_regression/` | Store-level model validation and April forecast CSVs |
| `comparison/` | Side-by-side tables for all three models |
| `heatmaps/` | Row order for store×day heatmap figures (which store is on which row) |
| `figures/` | PNG charts (exploratory, forecasts, comparisons, heatmaps) |

## Markdown summaries (this directory)

| File | What it is |
|------|------------|
| `baseline_report.md` | Narrative report for the baseline / EDA pass |
| `model_comparison.md` | Auto-generated March metrics and April totals for all models (re-run `06_compare_models.py` to refresh) |

## Key CSV types (by name)

| Name pattern | Plain English |
|--------------|----------------|
| `*_validation_overall.csv` | Summary error metrics for **March** (one row per metric). |
| `*_validation_monthly_actual_vs_pred.csv` | Per-series **March total** actual vs predicted quantity. |
| `*_validation_per_warehouse.csv` / `*_per_customer.csv` | March errors grouped by warehouse or customer. |
| `april_forecast_*_daily.csv` | **One row per day per series** for April predictions. |
| `april_forecast_*_monthly_*.csv` | April predictions **summed over the month** (by store, warehouse, etc.). |
| `hurdle_april_*_calibration.csv` | Multipliers applied in hurdle April **calibration** (level adjustment). |
| `comparison/model_comparison*.csv` | Same April or March view for baseline, hurdle, and store regression together. |

**Rollup:** store-grain models are sometimes summed to **warehouse–customer–product (WCP)** so they can be compared to the baseline grain.

## Figure index (`figures/`)

| Figures | Script | What you see |
|---------|--------|----------------|
| 01–10 | `03_visualize.py` | Exploratory plots (daily totals, warehouse mix, seasonality, top SKUs, …) |
| 11–15 | `05_visualize_forecast.py` | Baseline: March scatter, April daily curve, history + forecast, warehouse share, error histogram |
| 16–19 | `05_visualize_forecast.py` | **All models:** April daily, by warehouse, history + forecast, by customer |
| 20–22 | `05b_store_april_heatmaps.py` | April **store × day** heatmaps (baseline split, hurdle pattern, hurdle store-cal) |
| 23–26 | `05b_store_april_heatmaps.py` | History + April heatmaps (full and top-20 stores) for hurdle |
| 27–31 | `05_visualize_forecast.py` | Store regression: same chart types as 11–15 |
| 32–34 | `05b_store_april_heatmaps.py` | Store regression store×day heatmaps |

**How to read a store×day heatmap:** rows = stores, columns = calendar days, color = total predicted `qty_ea` that day (darker or brighter = more volume). Intermittent stores should show **mostly blank/low days** and **occasional strong columns**; a solid band across many days often means the model spread demand too evenly (see glossary: “Why Poisson + recursion can look wrong on heatmaps”).

## Regenerate or reorganize

Run the pipeline from the project [README](../README.md). To move legacy flat CSVs into subfolders:

```bash
python scripts/reorganize_reports.py
```
