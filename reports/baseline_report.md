# Task 2 — Baseline Forecasting Report

This is the first end-to-end pass: data cleaning, warehouse-centric organization,
exploratory visualization, and a baseline demand forecasting model for April
2026. No external signals (weather, traffic, calendar) are used here.

All code runs under the `AI2` conda env at `C:\Users\Hank\.conda\envs\AI2`.

## 1. Data cleaning

Script: [src/01_clean_data.py](../src/01_clean_data.py)

Outputs (under [processed/](../processed)):

- `orders.csv` — order header table (one row per order)
- `order_items.csv` — order line items, joined with order header context
- `logistics_events.csv` — operational events from the `物流信息` sheet
- `clean_summary.csv` — high-level reconciliation metrics

Key normalization steps:

- Renamed all Chinese columns to canonical English IDs (e.g. `订单单号` -> `order_id`,
  `货主编码` -> `customer_id`, `仓库` -> `warehouse`).
- Resolved the duplicated `单位` column and the schema drift between
  `求和项:预计发货数量-EA` (Jan) and `预计发货数量EA` (Feb/Mar) by mapping both
  to canonical `qty` / `qty_ea` / `uom` / `uom_ea` fields.
- Parsed `创建时间` and `操作时间` to proper datetime; preserved Excel-serial
  fallback path in case future exports differ.
- Split `省市区` into `province`, `city`, `district`.
- Deduplicated by `order_id` (no duplicates found) and dropped exact dupes on
  detail/log tables.

Clean summary (Jan + Feb + Mar combined):

| metric | value |
|---|---|
| orders | 30,815 |
| order line items | 437,204 |
| logistics events | 385,596 |
| unique customers | 3 |
| unique warehouses | 24 |
| unique products | 288 |
| order range | 2026-01-01 09:50 -> 2026-03-31 11:02 |
| orphan item / log records | 2 / 2 (negligible) |
| month distribution | Jan 10,928 / Feb 8,130 / Mar 11,757 |

Important caveat surfaced by the profile: the 2.xlsx export ends on
**Mar 31 11:02 AM**, so Mar 29-31 are incomplete (daily totals collapse from
~44k -> 18k -> 13k -> 69). The forecasting pipeline treats the history as
ending on **Mar 28** to avoid feeding incomplete tail data into lag features.

## 2. Warehouse-centric organization

Script: [src/02_build_marts.py](../src/02_build_marts.py)

Outputs (under [processed/marts/](../processed/marts)):

- `mart_warehouse_day.csv` — daily totals per warehouse
- `mart_warehouse_customer_day.csv` — daily totals per warehouse x customer
- `mart_warehouse_product_day.csv` — daily totals per warehouse x product
- `mart_warehouse_customer_product_day.csv` — modeling grain (wh x cust x prod x day)
- `mart_warehouse_summary.csv` — overall ranking of warehouses by total qty_ea
- `mart_top_products_per_warehouse.csv` — top 10 products per warehouse

Hierarchy used for browsing:
`province -> city -> warehouse -> customer -> product -> day`.

Top warehouses by total qty_ea (Jan-Mar):

| warehouse | total qty_ea | share % |
|---|---:|---:|
| 武汉1仓 | 454,632 | 13.0% |
| 西安1仓 | 399,406 | 11.5% |
| 深圳5仓 | 398,529 | 11.4% |
| 广州6仓 | 380,067 | 10.9% |
| 杭州1仓 | 299,842 | 8.6% |
| 苏州3仓 | 286,877 | 8.2% |
| 南京2仓 | 283,011 | 8.1% |
| 上海2仓 | 176,749 | 5.1% |
| 广州3仓 | 148,802 | 4.3% |
| 成都2仓 | 143,533 | 4.1% |

## 3. Visualizations

Script: [src/03_visualize.py](../src/03_visualize.py)

Charts under [reports/figures/](figures):

- `01_daily_total.png` — daily orders and qty_ea over Jan-Mar
- `02_daily_by_customer.png` — daily qty_ea split by customer (C01/C02/C03)
- `03_warehouse_share.png` — top 15 warehouses by total qty_ea
- `04_warehouse_monthly.png` — monthly qty_ea per top 15 warehouses
- `05_dow_pattern.png` — average qty_ea by day-of-week
- `06_temperature_zone.png` — daily qty_ea stacked by `温区`
- `07_top_products.png` — top 15 products by total qty_ea
- `08_warehouse_heatmap.png` — daily qty_ea heatmap (top 20 warehouses)
- `09_customer_mix_by_warehouse.png` — customer mix per top 12 warehouses

Notable patterns:

- Customer C01 dominates volume (~85% of qty_ea), C02/C03 are smaller.
- Cold-zone (`冷冻`) is the largest temperature zone, followed by `常温`,
  with `冷藏` a distant third.
- Weekly seasonality is visible: troughs on certain weekdays.
- Most warehouses serve a single customer; a few mixed warehouses (e.g. `杭州1仓`)
  serve multiple customers and many more SKUs.

## 4. Baseline forecasting model

Script: [src/04_baseline_model.py](../src/04_baseline_model.py)

### Modeling grain

- Target: daily `qty_ea` per `(warehouse, customer_id, product_id)`.
- Active tuples retained: those with `>= 3 non-zero days` in Jan+Feb -> 1,302 tuples.
- Dense daily panel from Jan 1 -> Mar 28 -> 113,286 rows.

### Features (no external data)

- Lags: `lag_1, lag_2, lag_3, lag_7, lag_14, lag_21, lag_28`.
- Rolling stats over windows 7/14/28: mean, std, max, non-zero ratio.
- Calendar: day-of-week, day-of-month, month, week-of-year, weekend flag.
- Entity (ordinal encoded): warehouse, customer_id, product_id, temperature_zone.

### Model

- `sklearn.ensemble.HistGradientBoostingRegressor`
- Loss: Poisson (well-suited for non-negative count-like demand)
- 600 iterations, max 63 leaves, lr=0.06, min_samples_leaf=40, l2=0

### Splits

- Train: Jan 1 -> Feb 28
- Validation: Mar 1 -> Mar 28
- Final fit (for April submission): Jan 1 -> Mar 28
- Forecast: Apr 1 -> Apr 30 via **day-by-day recursive forecast**

### Validation metrics (March 1-28)

| metric | value |
|---|---:|
| daily MAE | 16.88 |
| daily RMSE | 59.88 |
| daily SMAPE | 1.20 |
| daily bias | -6.08 |
| monthly MAE (per wh,cust,prod) | 214.49 |
| monthly RMSE (per wh,cust,prod) | 638.99 |
| monthly SMAPE (per wh,cust,prod) | 0.41 |
| monthly bias | -170.25 |
| March actual total | 1,184,343 |
| March predicted total | 962,678 |
| **monthly WMAPE** | **23.58%** |
| **bias ratio** | **-18.7%** |

Top sources of error (per-warehouse, see
`baseline_validation_per_warehouse.csv`):

- High-volume warehouses systematically under-predicted: `深圳5仓` (-33),
  `上海2仓` (-24), `西安1仓` (-20), `成都2仓` (-18), `北京1仓` (-12).
- Small warehouses are roughly unbiased but have high SMAPE due to low denominators.

The negative bias is driven mostly by the assumption-free baseline failing to
capture upward trend going into March (Mar > Feb). Adding trend features and a
month-over-month growth signal is a natural next improvement.

### April forecast outputs

Files under [reports/](.):

- `april_forecast_daily.csv` — daily predicted qty_ea per (warehouse, customer, product)
- `april_forecast_monthly_by_customer_product.csv` — submission-style aggregate
- `april_forecast_monthly_by_warehouse_customer_product.csv` — finer breakdown
- `april_forecast_monthly_by_warehouse.csv` — warehouse-level totals

April high-level totals:

| metric | value |
|---|---:|
| April predicted total qty_ea | 1,096,710 |
| April average daily qty_ea | ~36,557 |
| C01 share | ~92% (1.01M) |
| C02 share | ~2% (26k) |
| C03 share | ~6% (62k) |
| Top warehouses (Apr) | 深圳5仓, 武汉1仓, 西安1仓, 广州6仓, 南京2仓 |

Charts produced by [src/05_visualize_forecast.py](../src/05_visualize_forecast.py):

- `figures/11_validation_monthly_scatter.png` — predicted vs actual scatter
- `figures/12_april_forecast_daily.png` — April daily forecast bars
- `figures/13_history_plus_forecast.png` — Jan-Apr combined timeline
- `figures/14_april_warehouse_share.png` — April warehouse ranking
- `figures/15_validation_error_hist.png` — March error distribution

### CNY-aware baseline (current run)

The baseline script now uses [src/_cny.py](../src/_cny.py): per-warehouse closure
detection in the CNY window, `qty_ea_masked` for lag/rolling features,
`sample_weight` on closure / ramp days, and March validation on open days only.
See `baseline_validation_overall.csv` for the refreshed headline metrics.

### Store-grain hurdle model (comparison)

Script: [src/04b_hurdle_model.py](../src/04b_hurdle_model.py)  
Mart: [processed/marts/mart_warehouse_store_product_day.csv](../processed/marts/mart_warehouse_store_product_day.csv)

- **Grain:** `(warehouse, store, product_id) × day` (store = 收货门店).
- **Model:** `P(order) × E[qty | order]` with `HistGradientBoostingClassifier`
  (`class_weight="balanced"`) + `HistGradientBoostingRegressor` (Poisson), same
  CNY treatment and extended calendar/cadence features as in the plan.
- **Tuple filter:** `>= 10` non-zero days in Jan–Feb (sparsity control).
- **April forecast:** recursive day-by-day with **seeded** future rows using
  each tuple’s mean `qty_ea` over the last 14 days of history (avoids
  classifier collapse from all-zero cold-start lags).

Headline outputs: `hurdle_validation_*.csv`, `april_forecast_hurdle_*.csv`.  
Side-by-side comparison: [reports/model_comparison.md](model_comparison.md).

> Note: Daily MAE is **not** comparable across baseline vs hurdle (different
> grains and sparsity); use rolled April totals and monthly WMAPE on the store
> grain for hurdle.

## 5. Known limitations and next steps

- Baseline under-predicts heavy warehouses; add **trend** and
  **month-of-year** features, or stack with a per-tuple seasonal-naive model.
- Recursive forecasting can drift on tuples with sparse history; consider a
  **direct multi-horizon** model (one model per horizon) for stability.
- No external data yet. Planned additions (from the plan): **weather**,
  **traffic / mobility**, **holidays and promotional calendar**.
- Mar 29-31 are incomplete in the source export; if a corrected export is
  available, redo the cleaning and re-validate.
- Tuples with `< 3` non-zero days in Jan+Feb are skipped from modeling; for
  full-coverage submission, add a fallback (e.g. expanding the threshold or
  using a zero / popularity prior).

## 6. Reproduction

Run from `D:\stuff\python\HKU project` with the AI2 env:

```powershell
& "C:\Users\Hank\.conda\envs\AI2\python.exe" "src\01_clean_data.py"
& "C:\Users\Hank\.conda\envs\AI2\python.exe" "src\02_build_marts.py"
& "C:\Users\Hank\.conda\envs\AI2\python.exe" "src\03_visualize.py"
& "C:\Users\Hank\.conda\envs\AI2\python.exe" "src\04_baseline_model.py"
& "C:\Users\Hank\.conda\envs\AI2\python.exe" "src\04b_hurdle_model.py"
& "C:\Users\Hank\.conda\envs\AI2\python.exe" "src\06_compare_models.py"
& "C:\Users\Hank\.conda\envs\AI2\python.exe" "src\05_visualize_forecast.py"
```

Dependencies (installed in the AI2 env):

- `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `scipy` (pre-existing)
- `openpyxl` (installed in this session)
