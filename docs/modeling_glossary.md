# Modeling glossary (plain language)

This page defines terms used in the project [README](../README.md) and under `reports/`. You do not need advanced statistics to follow the pipeline; most ideas are “predict future daily quantity from past daily quantity plus calendar effects.”

---

## What we are predicting

| Term | Meaning |
|------|---------|
| **Demand / quantity** | How many units were ordered or shipped. The models predict **`qty_ea`**: quantity in **each** (EA = “each”), the standard unit in the coursework data. |
| **Daily forecast** | One predicted number per **calendar day** for each product series we model. |
| **April forecast** | Predictions for 2026-04-01 … 2026-04-30. The code does **not** use April actuals when training; April is a **future** period. |
| **Validation (March)** | We hold March out while fitting on January–February, then compare predictions to **known** March actuals. This checks whether the model generalizes to a recent month we did not train on. |

---

## How the data is organized

| Term | Meaning |
|------|---------|
| **Grain** | The set of columns that define **one time series** (one line on a chart). Example: one warehouse + one store + one product = one series of daily `qty_ea`. |
| **Tuple** | One specific combination of grain keys (e.g. one store–SKU pair). |
| **Mart** | A pre-aggregated table built in `02_build_marts.py` (sums of `qty_ea` by day and grain). |
| **Dense daily panel** | For every tuple we keep, we create a row for **every calendar day** in the history window. Days with no order get **`qty_ea = 0`**. This makes it easy to compute “yesterday’s quantity” and rolling averages. |
| **Sparse / intermittent demand** | Most days are zero; occasionally there is a large order. Convenience stores often look like this at store×SKU level. |
| **Open warehouse day** | A day when the warehouse was operating (not treated as a Chinese New Year closure day). Validation metrics usually ignore closed days. |

**Grains in this project**

| Model | Grain (one series = …) |
|-------|-------------------------|
| Baseline | warehouse + customer + product |
| Hurdle & store regression | warehouse + **store** + product |

---

## Sliding-window forecasting (this project)

All three models share **`src/_sliding_window.py`**:

| Piece | Setting |
|-------|---------|
| **Lookback** | Last **14 warehouse-open** days of `qty_ea` (CNY closure days skipped, not counted) |
| **Horizon** | Predict the **next 7 calendar days** |
| **Training/validation slide** | Move the anchor forward **1 day** (many overlapping samples) |
| **April slide** | Move forward **7 days** (weekly blocks); use predictions from earlier April weeks in later lookbacks |

Each **training example** is one tuple, one anchor, one day inside the 7-day horizon. Features include the 14 lookback values, lookback mean/std/max, whether lookback days had orders, calendar fields for the target day, and encoded warehouse/store/product IDs.

**Plain language:** “What happened on the last two weeks of real operating days, and what day of the week is it?” → predict that day’s quantity for the coming week.

---

## Train, validate, forecast (timeline)

```text
Jan ───────── Feb ───────── Mar ───────── Apr
[  TRAINING  ]              [VALIDATION]   [FORECAST — no labels in repo]
  fit weights                 compare to      predict only
  on actuals                  actuals
```

| Phase | Dates (default) | Purpose |
|-------|------------------|---------|
| **Training** | Jan 1 – Feb 28 | The model learns patterns from actual history. |
| **Validation** | Mar 1 – Mar 28 | Same model setup, but March was not used to fit the **validation** run; we measure error vs March actuals. |
| **History end** | Mar 28 | Last day of observed data used to build features and pattern statistics. |
| **Forecast** | Apr 1 – Apr 30 | Forward prediction. There is **no separate “test” file** in the repository for April. |

---

## Error metrics (how good is the fit?)

All metrics compare **actual** `qty_ea` to **predicted** `qty_ea`. Lower is usually better, except bias ratio where **1.0** is ideal.

| Metric | Plain English | Formula idea |
|--------|---------------|--------------|
| **MAE** (mean absolute error) | On average, how far off are we in units? | average of \|actual − predicted\| |
| **RMSE** (root mean square error) | Like MAE, but **large mistakes count more**. | square root of average (actual − predicted)² |
| **SMAPE** (symmetric MAPE) | Error as a **percentage**, treating over- and under-prediction similarly. | average of 2\|actual − predicted\| / (\|actual\| + \|predicted\|) |
| **WMAPE** (weighted MAPE) | Total error divided by **total actual** (good for monthly totals). | sum(\|actual − predicted\|) / sum(actual) |
| **Bias** | Are we systematically too high or too low? | sum(predicted − actual) |
| **Bias ratio** | Predicted total as a fraction of actual total. **1.0** = perfect level; **1.1** = 10% too high overall. | sum(predicted) / sum(actual) |

**Caveat for intermittent data:** On a dense panel with many zero days, **SMAPE** can look bad even when predicted values are tiny, because any positive prediction on a zero day counts as a large percentage error. Prefer **MAE on order days**, **monthly WMAPE**, or the hurdle’s **decision-rule** metrics when demand is sparse.

---

## Model building blocks

| Term | Plain English |
|------|----------------|
| **Regression** | Predict a **number** (here, daily quantity) from inputs (lags, calendar, warehouse id, etc.). |
| **Classification** | Predict a **yes/no** label (here: “was there an order this day?”). |
| **Feature** | One input column the model sees (e.g. “quantity 7 days ago”, “day of week”). |
| **Lag** | A feature that uses quantity from **N days ago**. |
| **Rolling mean / std / max** | Summary of quantity over the last 7, 14, or 28 days (computed from past days only, not including “today”). |
| **Categorical encoding** | Turn warehouse / store / product names into integers so tree models can split on them. |
| **GBDT** (gradient boosted decision trees) | A popular machine-learning model: many small decision trees combined. Used in baseline and hurdle (sklearn `HistGradientBoosting*`). |
| **Poisson loss** | A training objective suited to **non-negative counts**. It encourages predictions ≥ 0 and behaves like modeling an **average rate** per day. It does **not** by itself force exact zeros on non-order days. |
| **Sample weight** | How much each row matters during training (0 = ignore, 0.5 = half importance). Used to down-weight Chinese New Year closure days. |
| **Pooling** | One model shared across thousands of store–product series, with IDs as features—instead of a separate model per series. |

---

## The three forecast approaches in this repo

### 1. Baseline — single Poisson GBDT on sliding rows

- One regressor; sliding train/validate/April.
- **Grain:** warehouse × customer × product.

### 2. Hurdle — classifier × size on sliding rows

- **P(order)** from a classifier; **μ** from a Poisson regressor on positive training rows; **pred = P × μ**.
- Same sliding April chaining as baseline.
- **Grain:** warehouse × store × product.

### 3. Store regression — single Poisson GBDT on sliding rows

- Same sliding protocol as baseline at store grain; optional GPU (XGBoost).

---

## Legacy: pattern forecast (not used in `main()` anymore)

Older versions scheduled April order days from **gap + day-of-week** rules (`src/_pattern_forecast.py`). The current pipeline uses **sliding windows** for April on all models. Pattern helpers remain for experiments.

---

## Reports and figures (quick map)

See [reports/README.md](../reports/README.md) for folder layout. Figure numbers in `reports/figures/`:

| Figures | Content |
|---------|---------|
| 01–10 | Exploratory charts (`03_visualize.py`) |
| 11–15 | Baseline validation and April |
| 16–19 | All-model April comparison |
| 20–26 | Hurdle store×day heatmaps |
| 27–31 | Store regression validation and April |
| 32–34 | Store regression store×day heatmaps |

---

## Chinese New Year (CNY) in one paragraph

Around the lunar new year, some warehouses shut down or ship very little. The code marks **closure days**, masks their quantity when building lag features (so the model does not treat closure as “zero demand”), and sets training **weights** to 0 on closed days so the model is not trained to predict shutdown as normal demand.
