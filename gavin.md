# Task 2 Modeling Handoff

## Current Model

The active model is **not** the hurdle model and **not** the store-regression model. It is a separate deterministic hierarchical forecasting system:

1. forecast broad product-category weekly demand;
2. apply known-ahead calendar and accepted price-regime signals as bounded systematic context;
3. allocate category-week totals into category-day, product-week, product-day, and hour-level outputs;
4. reconcile every child layer back to its parent total.

March is the validation month. April is the real forecast target; April WMAPE is not reported because April actual labels are not present.

## Active Files

- `src/14_tagged_weekly.py`: builds product tags, systematic calendar/price sidecars, and the selected category-week parent forecast.
- `src/15_granularity.py`: allocates category-week into category-day, product-week, and product-day; selects the reconciled product-day system.
- `src/17_hourly_customer_granularity.py`: validates day-to-hour and customer-category layers using real `create_time` order timestamps; also writes April hourly forecast artifacts.
- `src/16_error_charts.py`: creates the two composite 2x3 chart grids for grouped/customer by weekly/daily/hourly views.

Required data artifacts:

- `processed/tagging/product_order_tags.csv`
- `external_data/processed/*.csv`
- `reports/weekly_tagged_systematic/`
- `reports/production_granularity/`

## Latest March Validation Metrics

| Layer | Selected model | WMAPE | Target | Status |
|---|---|---:|---:|---|
| Grouped category-week | selected tagged-systematic parent | `0.035964` | `0.0500` | Pass |
| Grouped category-day | `category_day_frontload_58_42_00` | `0.111988` | `0.0700` | Fail |
| Product-week | `product_week_recency_alpha_1.05` | `0.116388` | `0.0500` | Fail |
| Product-day | `product_day__product_week_recency_alpha_1.05__product_dow_ipf` | `0.193005` | `0.1000` | Fail |
| Grouped category-hour | `category_hour_calendar_daytype_shrink_k1000` | `0.326187` | `0.1200` | Fail |
| Product-hour | `product_hour_product_hour_shrink_k500` | `0.475246` | `0.3000` | Fail |
| Customer-category week | `customer_category_janfeb_share` | `0.087310` | `0.1200` | Pass |
| Customer-category day | `customer_category_janfeb_share` | `0.161023` | `0.1800` | Pass |
| Customer-category hour | `customer_category_janfeb_share` | `0.492230` | `0.2500` | Fail |

All reconciliation checks pass for category-day, product-week, product-day, category-hour, product-hour, and customer-category outputs.

## External Data Result

External data helps only when used as systematic context, not as direct raw regressors.

- Weekly parent forecast uses calendar and price-regime context and remains strong at `0.035964` WMAPE.
- Hourly calendar day-type conditioning improves grouped-hour WMAPE from `0.333997` to `0.326187`, so it is accepted for the hourly layer.
- Weather and other direct external sources are not promoted in the active path.

## April Forecast Status

`src/17_hourly_customer_granularity.py --mode forecast` writes April forecast artifacts under:

- `reports/production_granularity/hourly_customer/april_category_day_forecast.csv`
- `reports/production_granularity/hourly_customer/april_category_hour_forecast.csv`
- `reports/production_granularity/hourly_customer/april_forecast_note.md`

These are forecasts only. They are not validation results.

## Charts

`src/16_error_charts.py` now produces exactly two chart files:

- `reports/production_granularity/error_charts/01_actual_pred_line_grid.png`
- `reports/production_granularity/error_charts/02_error_bar_grid.png`

Each figure contains six panels: grouped/customer rows by weekly/daily/hourly columns.

## Production Readiness

The project is not production-grade yet. The reliable part is the category-week anchor. The remaining prediction gaps are:

- daily timing error below category-week;
- product-mix volatility, especially new or returning March products with little Jan-Feb history;
- hourly concentration around operational order-entry windows;
- customer/category allocation sparsity.

The next improvement should focus on product lifecycle detection, customer-specific cadence, and a better allocation model for sparse products/customers. The hurdle model remains useful as a clean baseline for comparison, but it is not part of the selected active hierarchy.
