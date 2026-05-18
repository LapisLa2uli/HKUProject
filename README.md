# HKU warehouse demand forecasting

Pipeline that cleans monthly Excel extracts, builds warehouse-centric data marts, fits baseline and hurdle (two-part) GBDT models, and writes forecasts and charts under `reports/`.

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

Model scripts assume **2026** calendar splits (train through February, validate March, forecast April). To use another year or cutoff dates, adjust the date constants at the top of `src/04_baseline_model.py` and `src/04b_hurdle_model.py` consistently.

## 3. Run the pipeline (from scratch)

Run **from the repository root** so relative paths resolve:

```bash
python src/01_clean_data.py
python src/02_build_marts.py
python src/03_visualize.py
python src/04_baseline_model.py
python src/04b_hurdle_model.py
python src/05_visualize_forecast.py
python src/05b_store_april_heatmaps.py
python src/06_compare_models.py
```

Outputs:

- **`processed/`** — canonical CSVs (`orders.csv`, `order_items.csv`, …) and `processed/marts/` aggregates used by models.  
- **`reports/`** — Markdown summaries (tracked), CSV metric tables (ignored by Git), and PNG figures under `reports/figures/`.

Optional debugging utilities live under `scripts/`.

## 4. Troubleshooting

- **`Missing optional dependency 'openpyxl'`** — install requirements again (`pip install -r requirements.txt`).  
- **`[WARN] missing source`** — confirm `Data/1.xlsx` and `Data/2.xlsx` exist (or paths in `SOURCE_FILES`).  
- **Empty or tiny `processed/`** — check sheet names and headers match what `01_clean_data.py` expects.
