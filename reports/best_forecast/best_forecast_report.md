# Best forecast report (TSB + category calibration)

Independent Workstream 2 model: `src/18_best_forecast.py`.

## Approach

1. **TSB intermittent demand** at `(warehouse, store, product)` — separate
   occurrence smoothing from order-size smoothing (Teunter–Syntetos–Babai).
2. **Category calibration** — reconcile product totals within each store × macro-category
   to a Jan–Feb pace target, with bounded price-regime elasticity from external tags.
3. **Calendar disbursement** — spread monthly totals across days using historical DOW
   quantity shares and systematic calendar open-day weights.

March validation trains on **Jan–Feb only**. April deployment retrains on **Jan–Mar**.

## March validation

| Metric | Value |
|---|---:|
| segment_monthly_wmape | 0.403507 |
| row_wmape | 1.520905 |
| bias_ratio | -0.006265 |
| planning_window_loss | 1.383297 |
| composite_planning_loss | 1.417699 |
| strict_daily_wmape | 1.520905 |
| matched_volume_rate | 1.000000 |
| line_match_rate | 1.000000 |
| mean_abs_day_slip_matched | nan |
| Q_target | 39928.836982 |
| mean_daily_pred_march | 37401.158780 |
| mean_daily_actual_march | 37636.964286 |

## April forecast summary

| Metric | Value |
|---|---:|
| april_total_qty | 927404.543930 |
| april_mean_daily_total | 34348.316442 |
| Q_target | 39928.836982 |
| april_to_Q_target_ratio | 0.860238 |
| n_forecast_rows | 982071 |
| n_products | 211 |
| n_stores | 1442 |

## Comparison to baselines

| Metric | TSB-CatCal (this model) | Hurdle (04b) | Hierarchical product-day |
|---|---:|---:|---:|
| Segment/monthly volume WMAPE | 0.4035 | 0.8536 | 0.1930 |
| Planning window loss (lower better) | 1.3833 | 1.1752 | — |
| Composite planning loss | 1.4177 | 1.2453 | — |
| Strict daily WMAPE | 1.5209 | 1.4559 | 0.1930 |
| Volume bias ratio | -0.0063 | -0.3249 | — |
| Line match rate (±2d) | 1.0000 | 0.3243 | — |

Hierarchical WMAPE is at reconciled **product-day** grain (scripts 14–15); hurdle metrics from
`reports/hurdle/hurdle_validation_*.csv`. Category-week hierarchical anchor remains strongest (~3.6% WMAPE).

## Comparison notes

- **Hurdle / store regression** optimize sliding-window daily GBDT rows; this model
  targets monthly product volume first, then places quantity on planning days.
- **Hierarchical baseline (scripts 14–17)** anchors category-week (~3.6% WMAPE) and
  reconciles down; product-day WMAPE there is ~19.3%. This script stays at store×product
  grain without multi-layer IPF.
- Planning loss (±2 day product/qty match) is the primary business metric here.

## Limitations / next steps

- New or returning products with <3 Jan–Feb order days are excluded.
- Category calibration uses macro-group price tags; product-level elasticity not modeled.
- Day disbursement uses smoothed DOW shares — consider cadence/gap models for top SKUs.
