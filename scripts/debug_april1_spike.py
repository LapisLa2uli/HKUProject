"""Fast repro for April 1 qty spike (production recursive path + soften)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import importlib

hm = importlib.import_module("04b_hurdle_model")

SUBSAMPLE_TUPLES = 800
FORECAST_DAYS = 5


def main() -> int:
    log_path = ROOT / "debug-164b5f.log"
    if log_path.exists():
        log_path.unlink()

    raw = hm.load_panel_base()
    closure = hm.detect_warehouse_closure(raw)
    panel = hm.build_dense_panel(raw, hm.HISTORY_END)

    vol = (
        panel[(panel["date"] >= hm.VALID_START) & (panel["date"] <= hm.VALID_END)]
        .groupby(hm.ID_COLS)["qty_ea"]
        .sum()
        .sort_values(ascending=False)
    )
    top = vol.head(SUBSAMPLE_TUPLES).index
    keep = pd.DataFrame(list(top), columns=hm.ID_COLS)
    panel = panel.merge(keep, on=hm.ID_COLS, how="inner")

    panel_cny = hm.add_cny_features(panel.copy(), closure)
    panel_cny["qty_ea_masked"] = hm.mask_qty_for_features(panel_cny)
    panel_cny["order_signal"] = (panel_cny["qty_ea"] > 0).astype(np.float64)
    encoders = hm.build_encoders(panel_cny)
    panel_feat = hm.engineer_features(panel, encoders, closure)

    feat_cols = hm.feature_columns()
    train_df = panel_feat[panel_feat["date"] <= hm.TRAIN_END]
    w_train = hm.make_sample_weights(train_df)
    X_train = train_df[feat_cols].fillna(0.0)
    y_train = train_df["qty_ea"].astype(float)

    clf, reg = hm.fit_hurdle(X_train, y_train, w_train)
    tuple_priors = hm.tuple_recursive_priors(panel)
    dates = pd.date_range(hm.FORECAST_START, periods=FORECAST_DAYS, freq="D")

    fc = hm.recursive_forecast(clf, reg, panel, dates, encoders, closure, tuple_priors)
    by_day = fc.groupby("date")["qty_ea"].sum()
    fc2, soften_s = hm.soften_april_first_day_spike(fc)

    hm._agent_debug_log(
        "D",
        "debug_april1_spike.py:main",
        "post_fix_day_totals",
        {
            "sum_day0_before": float(by_day.iloc[0]),
            "sum_day0_after_soften": float(fc2.groupby("date")["qty_ea"].sum().iloc[0]),
            "sum_day1": float(by_day.iloc[1]),
            "ratio_d0_d1_before": float(by_day.iloc[0] / max(by_day.iloc[1], 1e-9)),
            "ratio_d0_d1_after": float(
                fc2.groupby("date")["qty_ea"].sum().iloc[0] / max(by_day.iloc[1], 1e-9)
            ),
            "apr1_soften_scale": soften_s,
        },
        run_id="post-fix",
    )
    print(f"Wrote debug log: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
