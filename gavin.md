# Task 2 Modeling Handoff

## Current Status

The active work is now a separate category/product/week/day forecasting path. The vanilla baseline, hurdle model, store-regression model, shared sliding-window helper, and original requirements were restored to the Git baseline for clean collaboration.

The current model is **neither the hurdle model nor the store-regression model**. It is a deterministic hierarchical forecasting system:

1. predict broad product-category weekly demand;
2. use calendar and price-regime sidecars as bounded systematic adjustments;
3. allocate category-week totals into category-day, product-week, and product-day outputs;
4. reconcile child predictions back to parent totals.

## Most Important Files

### Primary Modeling Code

- `src/14_tagged_weekly.py`
  - Builds product/order tags.
  - Builds systematic calendar and price-regime features.
  - Selects the deployable category-week parent forecast.
  - This is the most important model file.

- `src/15_granularity.py`
  - Takes the selected category-week parent forecast.
  - Allocates it into category-day, product-week, and product-day predictions.
  - Runs reconciliation and production-granularity validation.
  - This is the second most important model file.

- `src/16_error_charts.py`
  - Generates the focused six-chart error pack.
  - It does not change model outputs.

### Required Data Artifacts

- `processed/tagging/product_order_tags.csv`
- `external_data/processed/product_category_mapping.csv`
- `external_data/processed/timor_calendar_daily.csv`
- `external_data/processed/systematic_day_tags.csv`
- `external_data/processed/systematic_week_tags.csv`
- `external_data/processed/xinfadi_category_daily_prices.csv`
- `external_data/processed/mofcom_weekly_prices.csv`
- `external_data/processed/weekly_price_regime_tags.csv`

The three `src` files are enough for code review, but not enough for reproduction. To run the current path, the data artifacts above must also be provided.

## Current Metrics

### Category-Week Parent Forecast

Selected model:

- `monthly_price_regime_elasticity_-0.15_scale_1.0125__share_calendar_systematic_final_shrink_0.25`

Validation metrics:

| Metric | Value |
|---|---:|
| Weekly WMAPE | `0.035964` |
| Bias ratio | `-0.016277` |
| Actual total | `1,253,486.62` |
| Predicted total | `1,233,083.05` |
| Max single-week WMAPE | `0.078085` |
| Prior selected product-category WMAPE | `0.074341` |
| Relative improvement vs prior | `51.62%` |

This layer is the current reliable anchor.

### Production Granularity Allocation

| Layer | Selected model | WMAPE | Target | Target met |
|---|---|---:|---:|---|
| Category-week | selected tagged-systematic parent | `0.035964` | `0.0500` | Yes |
| Category-day | `category_day_frontload_58_42_00` | `0.111988` | `0.0500` | No |
| Product-week | `product_week_feb_share` | `0.117724` | `0.0500` | No |
| Product-day | `product_day_product_dow_ipf` | `0.193312` | `0.1000` | No |

All reconciliation checks pass. The remaining weakness is predictive allocation error, not arithmetic inconsistency.

## Model Basis

The active model is based on:

- product category grouping;
- deterministic product tags;
- known-ahead calendar structure;
- bounded price-regime adjustment;
- historical weekly shares;
- day-of-week allocation;
- product-share allocation;
- exact hierarchical reconciliation.

It is **not** based on:

- the hurdle model;
- the store-regression model;
- direct row-level external-data regression;
- neural sequence models.

The hurdle model remains useful as a clean baseline and as your friend's original modeling path. It is not part of the current selected category/product/week/day system.

## What To Give A Collaborator

For reviewing the new modeling logic:

- `src/14_tagged_weekly.py`
- `src/15_granularity.py`
- `src/16_error_charts.py`
- `gavin.md`

For reproducing the current outputs, also give:

- `processed/tagging/product_order_tags.csv`
- all retained files in `external_data/processed/`
- `reports/weekly_tagged_systematic/`
- `reports/production_granularity/`

For historical audit only:

- `legacy/`

## Production Readiness

The project is closer to production level but not production-grade yet.

What is strong:

- category-week forecasting;
- exact reconciliation;
- clean separation from the vanilla code;
- concise active pipeline.

What is still weak:

- category-day timing;
- product-week product-mix allocation;
- product-day sparse demand;
- proof across more than one validation month.

The next real improvement should focus on allocation quality: product lifecycle detection, new/returning product handling, day-of-week behavior by product family, and customer/store allocation. The category-week parent is no longer the main bottleneck.

## Cleanup Boundary

Quarantined exploratory work is under `legacy/`. Original vanilla files were not moved there. The current active surface is intentionally small: scripts `14`, `15`, `16`, processed sidecars, current reports, and this handoff.
