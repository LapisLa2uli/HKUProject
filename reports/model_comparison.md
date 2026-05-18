# Model comparison (baseline vs store hurdle)

## March validation — baseline bias (CNY handling)

| Metric | Value |
|--------|------:|
| Baseline bias ratio **before** CNY handling (documented) | -18.70% |
| Baseline bias ratio **after** CNY handling (current run) | 2.11% |
| Baseline daily MAE | 15.0228 |
| Hurdle daily MAE | 2.2928 |

## April totals (hurdle rolled up to warehouse–customer–product)

| Model | Total predicted qty_ea |
|-------|------------------------:|
| Baseline | 1,061,637 |
| Hurdle (rolled up) | 1,197,865 |

See `model_comparison.csv` and `model_comparison_april_*.csv`.
