# HKU warehouse demand forecasting

Pipeline that cleans monthly Excel extracts, builds warehouse-centric data marts, and fits baseline, hurdle, and store-level regression models using **sliding two-week (14 open-day) lookbacks to predict the next seven days** (CNY closure days skipped). Forecasts and charts are written under `reports/`.

Raw spreadsheets and regenerated CSV outputs are **not** tracked in Git (see `.gitignore`). Clone the repo, add your own files under `Data/`, then run the steps below.

## Prerequisites

- Python 3.10+ recommended  
- Windows, macOS, or Linux — paths in the scripts assume you run commands from the **repository root**

## 1. Environment

From the project root:

```bash
python -m venv .venv
```

Activate the virtual environment (Windows PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Provide raw data

Create a `Data/` folder next to `src/` and place **two workbooks** with the sheet layout expected by the cleaner:

| File       | Meaning (for the bundled pipeline dates) |
|-----------|-------------------------------------------|
| `Data/1.xlsx` | January period                             |
| `Data/2.xlsx` | February + March combined                  |

Required sheets and column naming match the HKU coursework extracts processed by `src/01_clean_data.py`:

- **`订单表`** — order header rows (Chinese column headers renamed internally).  
- **`订单明细`** — line-level SKU quantities.  
- **`物流信息`** — logistics events linked by order id.

If your filenames or calendar periods differ, edit `SOURCE_FILES` near the top of `src/01_clean_data.py` (paths and month labels).

Model scripts assume **2026** calendar splits (train through February, validate March, forecast April). To use another year or cutoff dates, adjust the date constants at the top of `src/04_baseline_model.py`, `src/04b_hurdle_model.py`, and `src/04c_store_regression_model.py` consistently.

## 3. Run the pipeline (from scratch)

Run **from the repository root** so relative paths resolve:

```bash
python src/01_clean_data.py
python src/02_build_marts.py
python src/03_visualize.py
python src/04_baseline_model.py
python src/04b_hurdle_model.py
python src/04c_store_regression_model.py
python src/05_visualize_forecast.py
python src/05b_store_april_heatmaps.py
python src/06_compare_models.py
```

Outputs:

- **`processed/`** — canonical CSVs (`orders.csv`, `order_items.csv`, …) and `processed/marts/` aggregates used by models.  
- **`reports/`** — see [reports/README.md](reports/README.md) for layout:
  - `baseline/`, `hurdle/`, `store_regression/` — model CSV outputs (gitignored)
  - `comparison/` — cross-model tables
  - `heatmaps/` — heatmap row-order metadata
  - `figures/` — PNG charts (validation, April forecasts, comparisons, store heatmaps)
  - `baseline_report.md`, `model_comparison.md` — summaries at `reports/` root

Optional debugging utilities live under `scripts/`.

**Documentation:** [docs/modeling_glossary.md](docs/modeling_glossary.md) explains forecasting terms for readers with basic math background. [reports/README.md](reports/README.md) describes output folders and figure numbers.

---

## Model methodology

This section documents how data are split, how features are built, and how the baseline, hurdle, and store regression models train, validate, and forecast. Shared implementation: `src/_sliding_window.py`, `src/_cny.py`, `src/_metrics.py`; models in `src/04_baseline_model.py`, `src/04b_hurdle_model.py`, `src/04c_store_regression_model.py`.

**Audience:** If you are new to forecasting, read [docs/modeling_glossary.md](docs/modeling_glossary.md) first (*grain*, *sliding window*, *MAE*, *hurdle model*, etc.).

### Concepts in brief

| Idea | What it means here |
|------|---------------------|
| **Target** | Daily **`qty_ea`** per tuple (see grain table below). |
| **Sliding window** | Use the last **14 warehouse-open days** to predict the **next 7 calendar days** (see next section). |
| **Training** | Sliding samples whose 7-day horizon falls in **Jan–Feb**; fit model weights. |
| **Validation** | Sliding samples whose horizon falls in **March**; compare to actuals (model fit on Jan–Feb only for the validation run). |
| **April forecast** | Same sliding protocol, stepping **7 days** at a time; after each week, **predictions** are appended so the next lookback can use them. |
| **Baseline** | One Poisson GBDT; sliding train/validate/forecast. |
| **Hurdle** | Classifier × size regressor on sliding rows; same sliding April path. |
| **Store regression** | One Poisson GBDT (optional GPU) on sliding rows; same sliding April path. |

**Metrics (validation):** **MAE**, **WMAPE**, **bias ratio** — see [glossary](docs/modeling_glossary.md).

### Pipeline overview

```mermaid
flowchart LR
  raw[Raw Excel Data]
  clean[src/01_clean_data.py]
  marts[src/02_build_marts.py]
  mart_cp[mart_warehouse_customer_product_day]
  mart_store[mart_warehouse_store_product_day]
  baseline[src/04_baseline_model.py]
  hurdle[src/04b_hurdle_model.py]
  storeReg[src/04c_store_regression_model.py]
  compare[src/06_compare_models.py]
  reports[reports/ CSV and figures]

  raw --> clean --> marts
  marts --> mart_cp --> baseline --> reports
  marts --> mart_store --> hurdle --> reports
  marts --> mart_store --> storeReg --> reports
  baseline --> compare
  hurdle --> compare
  storeReg --> compare
  compare --> reports
```

### Calendar splits and what counts as “test”

All three models use the **same date constants** (`TRAIN_END`, `VALID_START`, `VALID_END`, `FORECAST_START`, `FORECAST_END`, `HISTORY_END`) at the top of `src/04_baseline_model.py`, `src/04b_hurdle_model.py`, and `src/04c_store_regression_model.py`.

| Role | Dates | Used for |
|------|--------|----------|
| **Training** | 2026-01-01 through **2026-02-28** (`date <= TRAIN_END`) | Fitting model weights |
| **Validation** | **2026-03-01** through **2026-03-28** | March holdout metrics (not in training for the validation pass) |
| **History end** | Same as validation end (`HISTORY_END = VALID_END`) | Building dense panels through the last observed day |
| **Forecast** | **2026-04-01** through **2026-04-30** | Forward April prediction |

**There is no separate labeled “test” split in the repository.** April is a **forecast horizon** (future period): the scripts do not read April actuals for training or in-script evaluation. If you obtain April ground truth later, you can score outputs offline.

Validation metrics are computed on **open warehouse days** only: rows where `is_warehouse_closed != 1` (see Chinese New Year handling below). Closed days are excluded so metrics reflect normal operating demand, not holiday shutdown.

### Tuple universe (which series are modeled)

From the mart through `HISTORY_END`, each model builds a **dense daily panel**: for every kept product series, create a row for **every calendar day**, filling days without shipments as `qty_ea = 0`. That way “quantity yesterday” is always defined.

A tuple is **kept** only if it has at least **`MIN_NONZERO_DAYS_TRAIN = 3`** days with `qty_ea > 0` during **training only** (`date <= TRAIN_END`, i.e. January + February). This drops series that never ordered enough in the train window to learn from.

| Model | Grain (`ID_COLS`) | Input mart |
|-------|-------------------|------------|
| **Baseline** | `(warehouse, customer_id, product_id)` | `processed/marts/mart_warehouse_customer_product_day.csv` |
| **Hurdle** | `(warehouse, store, product_id)` | `processed/marts/mart_warehouse_store_product_day.csv` |
| **Store regression** | `(warehouse, store, product_id)` | same store mart as hurdle |

The hurdle and store-regression models also carry `customer_id` for aggregation and reporting; **store** is the extra key versus baseline.

### Sliding-window protocol (all models)

Implemented in **`src/_sliding_window.py`**. All three models use the **same** windowing for training, March validation, and April forecasting.

| Setting | Value | Meaning |
|---------|--------|---------|
| **Lookback** | 14 days | Last **14 warehouse-open** days strictly before the anchor date (CNY **closure** days do not count toward the 14). |
| **Horizon** | 7 days | Predict `qty_ea` for anchor, anchor+1, …, anchor+6. |
| **Train/valid slide** | 7 days | Weekly anchors (14 open days → next 7). Set `SLIDE_STEP_TRAIN_DAYS=1` in `_sliding_window.py` for denser overlap. |
| **April slide** | 7 days | Non-overlapping weekly blocks across April. |

**One training row** = one tuple × one anchor × one day within the horizon (`horizon_day_index` 0–6). Features for that row:

- **Lookback vector** `lb_14 … lb_1`: daily `qty_ea` on the 14 open days (oldest → newest).
- **Lookback summaries**: mean, std, max, fraction of positive days in the lookback.
- **Horizon calendar**: day-of-week, day-of-month, etc. for the day being predicted.
- **CNY fields** for that horizon day; **categorical encodings** for warehouse / store / product / customer / temperature zone.

**Skipping Chinese New Year**

- **Lookback:** only days with `is_warehouse_closed == 0` are used; you need 14 such days before the anchor or the sample is skipped.
- **Horizon targets:** closure days still appear in the 7-day block but get **weight 0** in training and **prediction forced to 0** in April.

**Training sample windows**

- **Train:** anchors where the full horizon lies in **2026-01-01 … 2026-02-28**.
- **Validation:** anchors where the horizon lies in **2026-03-01 … 2026-03-28** (model trained on train windows only).
- **Refit for April:** anchors with horizon in **Jan–Mar** (all history), then **April** forecast with weekly anchors from **2026-04-01**.

**April chaining (`sliding_forecast_period`)**

For each April anchor (Apr 1, Apr 8, Apr 15, Apr 22, Apr 29 if in range):

1. Build features from history + any **already predicted** days after March 28.
2. Predict 7 days; set closure days to 0.
3. Append predictions to the working panel for the next anchor’s lookback.

```text
  [14 open days lookback]  -->  model  -->  [next 7 days]
         ^                                           |
         |___________ slide 7d (April) _____________|
                    (predictions fill lookback)
```

---

### Shared Chinese New Year handling (`src/_cny.py`)

All three models call the same CNY utilities so holiday shutdowns do not look like normal low demand.

**Plain language:** around Chinese New Year some warehouses stop shipping. We mark those **closure days**, do not use their zeros when building “last week’s average” features, and give them **zero weight** during training so the model is not taught to predict shutdown as real demand.

**`detect_warehouse_closure`** — For each `(warehouse, date)`:

- Outside the CNY window (2026-02-10 … 2026-02-22), `is_warehouse_closed = 0`.
- Inside the window, mark **closed** if that warehouse’s total `qty_ea` that day is below **25%** of its mean daily total over the January baseline window **2026-01-15 … 2026-01-31** (`CLOSURE_QTY_RATIO_THRESHOLD = 0.25`).

**`add_cny_features`** — Merges closure onto the panel and adds:

- `is_cny_window`
- `days_to_cny`, `days_from_cny` (distance to 2026-02-17, clipped to [-30, 30])

**`mask_qty_for_features`** — Sets `qty_ea_masked` to NaN on closed days so lag and rolling features are not driven by artificial zeros during shutdown.

**`make_sample_weights`** — Per row when fitting GBDTs:

| Condition | Weight |
|-----------|--------|
| Warehouse closed | **0** (excluded from loss) |
| In CNY window but open | **0.5** |
| Otherwise | **1** |

---

### Baseline model (`src/04_baseline_model.py`)

**Grain:** `(warehouse, customer_id, product_id)` from `mart_warehouse_customer_product_day.csv`.

**Model:** `HistGradientBoostingRegressor`, `loss="poisson"`, `learning_rate=0.06`, `max_iter=600`, `max_leaf_nodes=63`, `min_samples_leaf=40`.

**Features:** sliding lookback (`lb_*`, `lb_mean`, …), horizon calendar, CNY fields, categorical encodings — see `sliding_feature_columns()` in `src/_sliding_window.py` (not daily lags on the full panel).

#### Train / validate / forecast

1. Build dense panel through `HISTORY_END`; detect CNY closure.
2. **`build_sliding_samples`**: train anchors (Jan–Feb horizons), valid anchors (March horizons).
3. Fit on train samples with `sample_weights_sliding`; predict valid samples; write `reports/baseline/baseline_validation_*.csv`.
4. Retrain on sliding samples with horizons through **March**; **`sliding_forecast_period`** for April (weekly anchors, predictions extend lookback).
5. Write `reports/baseline/april_forecast_daily.csv` and monthly rollups.

---

### Hurdle model (`src/04b_hurdle_model.py`)

**Grain:** `(warehouse, store, product_id)` from `mart_warehouse_store_product_day.csv`.

**Two-part model** on the **same sliding rows** as baseline:

| Part | Target | Model |
|------|--------|--------|
| **Classifier** | `1{qty_ea > 0}` on each horizon day | `HistGradientBoostingClassifier` |
| **Size regressor** | `qty_ea` where actual > 0 | `HistGradientBoostingRegressor`, Poisson loss |

**Combined prediction:** `pred = P(order) × μ` with optional **prior blend** on `P(order)` tuned on March sliding validation (`tuple_recursive_priors`: Jan–Feb order-day rate per tuple).

#### Train / validate / forecast

1. Build dense panel + CNY closure; `build_sliding_samples` for train (Jan–Feb horizons) and valid (March horizons).
2. Fit hurdle; tune prior blend on March; write `reports/hurdle/hurdle_validation_*.csv` (daily, classifier, intermittent composite).
3. Retrain on sliding samples through **March**; **`sliding_forecast_period`** for April with `predict_hurdle` on each batch.
4. Write April dailies to `reports/hurdle/april_forecast_hurdle_daily_*.csv` (pattern / store / network filenames; **same sliding forecast** in current pipeline).

**Note:** Legacy helpers (`pattern_forecast_april`, per-store/network calibration) remain in the file for reference but are **not** used by `main()`.

---

### Store regression model (`src/04c_store_regression_model.py`)

**Grain:** `(warehouse, store, product_id)` — same mart and **same sliding protocol** as hurdle.

**Model:** single Poisson regressor on sliding rows (XGBoost CUDA when available, else sklearn `HistGradientBoostingRegressor`). Same sliding features as baseline.

| Aspect | Hurdle (`04b`) | Store regression (`04c`) |
|--------|----------------|---------------------------|
| Structure | Classifier + size regressor | One regressor on `qty_ea` |
| Features | Sliding lookback + horizon calendar | Same sliding feature set |
| April | `sliding_forecast_period` + `predict_hurdle` | `sliding_forecast_period` + direct Poisson predict |

#### Train / validate / forecast

Same steps as baseline: sliding train (Jan–Feb horizons) → March validation → retrain through March → sliding April forecast.

**Performance:** sample building uses a **vectorized dense-panel** path in `_sliding_window.py` (typically seconds, not hours). Training uses **XGBoost CUDA** when available (`_gbdt_gpu.py`); force CPU with `STORE_REG_CPU=1`, `HURDLE_CPU=1`, or `BASELINE_CPU=1`.

#### Outputs

| File | Purpose |
|------|---------|
| `reports/store_regression/store_regression_validation_*.csv` | March sliding validation |
| `reports/store_regression/april_forecast_store_regression_daily.csv` | April daily (sliding) |
| `reports/store_regression/april_forecast_store_regression_monthly_*.csv` | Monthly rollups |

Charts: `05_visualize_forecast.py` (27–31); heatmaps: `05b_store_april_heatmaps.py` (32–34).

---

### Maintaining date constants and grains

All calendar boundaries and most hyperparameters are **Python constants** in the model scripts. If you change the coursework calendar or raw file layout:

1. Update `SOURCE_FILES` and labels in `src/01_clean_data.py`.
2. Update `TRAIN_END`, `VALID_*`, `FORECAST_*`, and CNY dates in `src/_cny.py` if needed.
3. Update the same split constants in `src/04_baseline_model.py`, `src/04b_hurdle_model.py`, and `src/04c_store_regression_model.py`.

Re-run the full pipeline from step 1 so marts and reports stay consistent.

---

## 4. Troubleshooting

- **`Missing optional dependency 'openpyxl'`** — install requirements again (`pip install -r requirements.txt`).  
- **`[WARN] missing source`** — confirm `Data/1.xlsx` and `Data/2.xlsx` exist (or paths in `SOURCE_FILES`).  
- **Empty or tiny `processed/`** — check sheet names and headers match what `01_clean_data.py` expects.
