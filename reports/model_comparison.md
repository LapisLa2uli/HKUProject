# Model comparison (baseline vs hurdle vs store regression)

## March validation

| Metric | Value |
|--------|------:|
| Baseline bias ratio **before** CNY handling (documented) | -18.70% |
| Baseline bias ratio **after** CNY handling (current run) | 2.11% |
| Baseline daily MAE | 15.0228 |
| Hurdle daily MAE | 2.2928 |
| Store regression daily MAE | 1.4756 |
| Baseline March monthly WMAPE | 0.1567 |
| Hurdle March monthly WMAPE | 1.2631 |
| Store regression March monthly WMAPE | 0.4118 |

## April totals (store regression and hurdle rolled up to warehouse–customer–product)

| Model | Total predicted qty_ea |
|-------|------------------------:|
| Baseline | 1,061,637 |
| Hurdle (rolled up) | 1,197,865 |
| Store regression (rolled up) | 931,758 |

See `model_comparison.csv` and `model_comparison_april_*.csv`.
