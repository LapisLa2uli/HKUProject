"""Sliding-window forecasting: 14 open days -> next 7 calendar days.

CNY warehouse-closure days are skipped when forming the 14-day lookback (only
``is_warehouse_closed == 0`` days count). Horizon days are every calendar day
in the 7-day block; closure days get target weight 0 and predicted qty forced to 0.

Sample construction uses a **vectorized dense-panel** path when every tuple shares
the same daily calendar (typical after ``build_dense_panel``).
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd

from _cny import CNY_DATE, CNY_WINDOW_END, CNY_WINDOW_START, add_cny_features

LOOKBACK_OPEN_DAYS = 14
HORIZON_DAYS = 7
SLIDE_STEP_DAYS = 7
# Weekly slide for training/validation (14 open days -> next 7); use ``1`` for denser overlap.
SLIDE_STEP_TRAIN_DAYS = 7

LB_COLS = [f"lb_{i}" for i in range(LOOKBACK_OPEN_DAYS, 0, -1)]
ROLL_LB_COLS = ["lb_mean", "lb_std", "lb_max", "lb_nonzero_ratio"]
HORIZON_CAL_COLS = [
    "horizon_dow",
    "horizon_dom",
    "horizon_month",
    "horizon_weekofyear",
    "horizon_is_weekend",
    "horizon_wom",
]
HORIZON_IDX_COL = "horizon_day_index"
CNY_COLS = ["is_cny_window", "is_warehouse_closed", "days_to_cny", "days_from_cny"]


def log(msg: str) -> None:
    print(msg, flush=True)


def is_dense_regular_panel(df: pd.DataFrame, id_cols: list[str]) -> bool:
    counts = df.groupby(id_cols, observed=True).size()
    return len(counts) > 0 and counts.nunique() == 1


def weekly_anchors(
    range_start: pd.Timestamp,
    range_end: pd.Timestamp,
    *,
    step_days: int = SLIDE_STEP_DAYS,
) -> pd.DatetimeIndex:
    start = pd.Timestamp(range_start).normalize()
    end = pd.Timestamp(range_end).normalize()
    anchors = pd.date_range(start, end, freq=f"{step_days}D")
    out = [a for a in anchors if a + pd.Timedelta(days=HORIZON_DAYS - 1) <= end]
    return pd.DatetimeIndex(out)


def anchors_with_horizon_in(
    horizon_start: pd.Timestamp,
    horizon_end: pd.Timestamp,
    *,
    step_days: int = SLIDE_STEP_DAYS,
) -> pd.DatetimeIndex:
    hs = pd.Timestamp(horizon_start).normalize()
    he = pd.Timestamp(horizon_end).normalize()
    first_anchor = hs
    last_anchor = he - pd.Timedelta(days=HORIZON_DAYS - 1)
    if last_anchor < first_anchor:
        return pd.DatetimeIndex([])
    return weekly_anchors(first_anchor, he, step_days=step_days)


def sliding_feature_columns(categorical_enc_cols: list[str]) -> list[str]:
    return (
        list(categorical_enc_cols)
        + LB_COLS
        + ROLL_LB_COLS
        + [HORIZON_IDX_COL]
        + HORIZON_CAL_COLS
        + CNY_COLS
    )


def _last_k_open_rows(open_before: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Last ``k`` non-NaN values per row from ``open_before`` (n_g, width); NaN = closed."""
    n_g, _ = open_before.shape
    out = np.full((n_g, k), np.nan, dtype=np.float64)
    ok = np.zeros(n_g, dtype=bool)
    for g in range(n_g):
        row = open_before[g]
        m = ~np.isnan(row)
        n_open = int(m.sum())
        if n_open >= k:
            ok[g] = True
            out[g] = row[m][-k:]
    return out, ok


def _horizon_calendar_matrix(dates: pd.DatetimeIndex | np.ndarray) -> dict[str, np.ndarray]:
    d = pd.DatetimeIndex(dates).normalize()
    dow = np.asarray(d.dayofweek, dtype=np.int16)
    dom = np.asarray(d.day, dtype=np.int16)
    return {
        "horizon_dow": dow,
        "horizon_dom": dom,
        "horizon_month": np.asarray(d.month, dtype=np.int16),
        "horizon_weekofyear": np.asarray(d.isocalendar().week, dtype=np.int16),
        "horizon_is_weekend": (dow >= 5).astype(np.int8),
        "horizon_wom": np.clip((dom - 1) // 7 + 1, 1, 5).astype(np.int16),
    }


def _cny_from_dates(dates: pd.DatetimeIndex | np.ndarray) -> dict[str, np.ndarray]:
    d = pd.DatetimeIndex(dates).normalize()
    in_win = (d >= CNY_WINDOW_START) & (d <= CNY_WINDOW_END)
    days_to = np.clip((CNY_DATE - d).days, -30, 30).astype(np.int16)
    days_from = np.clip((d - CNY_DATE).days, -30, 30).astype(np.int16)
    return {
        "is_cny_window": np.asarray(in_win, dtype=np.int8),
        "days_to_cny": days_to,
        "days_from_cny": days_from,
    }


def build_sliding_samples_dense(
    panel: pd.DataFrame,
    id_cols: list[str],
    categorical_enc_cols: list[str],
    static_enc: pd.DataFrame,
    anchors: pd.DatetimeIndex,
    *,
    require_target_in_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Vectorized sample builder for a regular warehouse×…×day grid."""
    t0 = time.perf_counter()
    panel = panel.sort_values(id_cols + ["date"]).reset_index(drop=True)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    dates = pd.DatetimeIndex(panel["date"].unique()).sort_values()
    n_d = len(dates)
    n_g = len(panel) // n_d
    if n_g * n_d != len(panel):
        raise ValueError("panel is not a regular dense grid")

    qty = panel["qty_ea"].to_numpy(dtype=np.float64).reshape(n_g, n_d)
    closed = panel["is_warehouse_closed"].fillna(0).to_numpy(dtype=np.int8).reshape(n_g, n_d)
    open_qty = np.where(closed == 0, qty, np.nan)

    if "is_cny_window" in panel.columns:
        cny_win = panel["is_cny_window"].fillna(0).to_numpy(dtype=np.int8).reshape(n_g, n_d)
    else:
        in_win = (dates >= CNY_WINDOW_START) & (dates <= CNY_WINDOW_END)
        cny_win = np.broadcast_to(in_win.astype(np.int8), (n_g, n_d)).copy()

    meta = panel.groupby(id_cols, sort=False, observed=True).head(1).reset_index(drop=True)
    enc = meta[id_cols].merge(static_enc.drop_duplicates(id_cols), on=id_cols, how="inner")
    if len(enc) != n_g:
        raise ValueError("static_enc row count mismatch")
    enc_mat = enc[categorical_enc_cols].to_numpy(dtype=np.float64)

    date_to_idx = {d: i for i, d in enumerate(dates)}
    anchor_idxs = [date_to_idx[pd.Timestamp(a).normalize()] for a in anchors]

    id_blocks: dict[str, list[np.ndarray]] = {c: [] for c in id_cols}
    feat_blocks: dict[str, list[np.ndarray]] = {
        **{c: [] for c in categorical_enc_cols},
        **{c: [] for c in LB_COLS + ROLL_LB_COLS},
        HORIZON_IDX_COL: [],
        **{c: [] for c in HORIZON_CAL_COLS},
        "is_cny_window": [],
        "days_to_cny": [],
        "days_from_cny": [],
    }
    closed_h_list: list[np.ndarray] = []
    anchor_list: list[np.ndarray] = []
    date_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    for a in anchor_idxs:
        if a <= 0:
            continue
        q14, ok = _last_k_open_rows(open_qty[:, :a], LOOKBACK_OPEN_DAYS)
        if not ok.any():
            continue

        h_end = min(a + HORIZON_DAYS, n_d)
        h_idxs = np.arange(a, h_end)
        n_h = len(h_idxs)
        if n_h == 0:
            continue

        g_ix = np.where(ok)[0]
        n_v = len(g_ix)
        q14v = q14[g_ix]
        lb_stack = np.column_stack([q14v[:, -(LOOKBACK_OPEN_DAYS - i)] for i in range(LOOKBACK_OPEN_DAYS)])

        h_dates = dates[h_idxs]
        if require_target_in_range is not None:
            lo, hi = require_target_in_range
            keep_h = (h_dates >= lo) & (h_dates <= hi)
            if not keep_h.any():
                continue
            h_idxs = h_idxs[keep_h]
            h_dates = h_dates[keep_h]
            n_h = len(h_idxs)

        h_qty = qty[g_ix][:, h_idxs]
        h_closed = closed[g_ix][:, h_idxs]
        h_cny = cny_win[g_ix][:, h_idxs]

        rep = n_v * n_h
        for j, c in enumerate(categorical_enc_cols):
            feat_blocks[c].append(np.tile(enc_mat[g_ix, j], n_h))

        for i, col in enumerate(LB_COLS):
            feat_blocks[col].append(np.repeat(lb_stack[:, i], n_h))
        feat_blocks["lb_mean"].append(np.repeat(q14v.mean(axis=1), n_h))
        feat_blocks["lb_std"].append(np.repeat(q14v.std(axis=1), n_h))
        feat_blocks["lb_max"].append(np.repeat(q14v.max(axis=1), n_h))
        feat_blocks["lb_nonzero_ratio"].append(np.repeat((q14v > 0).mean(axis=1), n_h))

        cal = _horizon_calendar_matrix(h_dates)
        h_off = np.tile(np.arange(n_h), n_v)
        feat_blocks[HORIZON_IDX_COL].append(h_off)
        for c in HORIZON_CAL_COLS:
            feat_blocks[c].append(np.repeat(cal[c], n_v))

        cny_d = _cny_from_dates(h_dates)
        feat_blocks["is_cny_window"].append(np.repeat(cny_d["is_cny_window"], n_v))
        feat_blocks["days_to_cny"].append(np.repeat(cny_d["days_to_cny"], n_v))
        feat_blocks["days_from_cny"].append(np.repeat(cny_d["days_from_cny"], n_v))
        closed_h_list.append(h_closed.ravel())

        for i, col in enumerate(id_cols):
            id_blocks[col].append(np.repeat(meta[col].to_numpy()[g_ix], n_h))

        anchor_list.append(np.repeat(dates[a], rep))
        date_list.append(np.tile(h_dates.to_numpy(), n_v))
        y_list.append(h_qty.ravel())

    if not y_list:
        return pd.DataFrame()

    out = pd.DataFrame({c: np.concatenate(id_blocks[c]) for c in id_cols})
    for c in categorical_enc_cols + LB_COLS + ROLL_LB_COLS + [HORIZON_IDX_COL] + HORIZON_CAL_COLS:
        out[c] = np.concatenate(feat_blocks[c])
    out["is_cny_window"] = np.concatenate(feat_blocks["is_cny_window"])
    out["days_to_cny"] = np.concatenate(feat_blocks["days_to_cny"])
    out["days_from_cny"] = np.concatenate(feat_blocks["days_from_cny"])
    out["is_warehouse_closed"] = np.concatenate(closed_h_list)
    out["anchor_date"] = np.concatenate(anchor_list)
    out["date"] = pd.to_datetime(np.concatenate(date_list))
    out["qty_ea"] = np.concatenate(y_list)
    extra_meta = [c for c in meta.columns if c not in out.columns and c not in id_cols]
    if extra_meta:
        out = out.merge(meta[id_cols + extra_meta], on=id_cols, how="left")

    log(f"  build_sliding_samples_dense: {len(out):,} rows in {time.perf_counter() - t0:.1f}s")
    return out


def build_sliding_samples(
    panel: pd.DataFrame,
    id_cols: list[str],
    categorical_enc_cols: list[str],
    static_enc: pd.DataFrame,
    anchors: pd.DatetimeIndex,
    closure: pd.DataFrame | None = None,
    *,
    require_target_in_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    if len(anchors) == 0:
        return pd.DataFrame()

    if is_dense_regular_panel(panel, id_cols):
        if closure is not None and "is_warehouse_closed" not in panel.columns:
            panel = add_cny_features(panel, closure)
        return build_sliding_samples_dense(
            panel,
            id_cols,
            categorical_enc_cols,
            static_enc,
            anchors,
            require_target_in_range=require_target_in_range,
        )

    return _build_sliding_samples_slow(
        panel,
        id_cols,
        categorical_enc_cols,
        static_enc,
        anchors,
        closure,
        require_target_in_range=require_target_in_range,
    )


def _build_sliding_samples_slow(
    panel: pd.DataFrame,
    id_cols: list[str],
    categorical_enc_cols: list[str],
    static_enc: pd.DataFrame,
    anchors: pd.DatetimeIndex,
    closure: pd.DataFrame | None,
    *,
    require_target_in_range: tuple[pd.Timestamp, pd.Timestamp] | None,
) -> pd.DataFrame:
    """Fallback row-by-row builder for irregular panels."""
    t0 = time.perf_counter()
    panel = panel.sort_values(id_cols + ["date"]).copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    enc_merge = static_enc.drop_duplicates(id_cols)
    closure_wh = None
    if closure is not None:
        closure_wh = closure.copy()
        closure_wh["date"] = pd.to_datetime(closure_wh["date"]).dt.normalize()

    rows: list[dict] = []
    for keys, g in panel.groupby(id_cols, sort=False, observed=True):
        g = g.sort_values("date")
        dates = g["date"].values.astype("datetime64[D]")
        qty_a = g["qty_ea"].values.astype(np.float64)
        wh_closed = g["is_warehouse_closed"].fillna(0).astype(np.int8).values
        key_df = (
            pd.DataFrame([keys], columns=id_cols)
            if isinstance(keys, tuple)
            else pd.DataFrame([keys], columns=id_cols)
        )
        enc_row = key_df.merge(enc_merge, on=id_cols, how="inner")
        if enc_row.empty:
            continue
        enc_dict = {c: enc_row[c].iloc[0] for c in categorical_enc_cols}

        for anchor in anchors:
            anchor = pd.Timestamp(anchor).normalize()
            before = np.datetime64(anchor)
            mask = (dates < before) & (wh_closed == 0)
            if int(mask.sum()) < LOOKBACK_OPEN_DAYS:
                continue
            q14 = qty_a[mask][-LOOKBACK_OPEN_DAYS:].astype(np.float64)
            lb_map = {f"lb_{i}": float(q14[-i]) for i in range(1, LOOKBACK_OPEN_DAYS + 1)}
            lb_stats = {
                "lb_mean": float(np.mean(q14)),
                "lb_std": float(np.std(q14)),
                "lb_max": float(np.max(q14)),
                "lb_nonzero_ratio": float(np.mean(q14 > 0)),
            }
            for h in range(HORIZON_DAYS):
                h_date = anchor + pd.Timedelta(days=h)
                if require_target_in_range is not None:
                    lo, hi = require_target_in_range
                    if h_date < lo or h_date > hi:
                        continue
                idx = np.where(dates == np.datetime64(h_date))[0]
                y_actual = float(qty_a[idx[0]]) if len(idx) else np.nan
                is_closed = int(wh_closed[idx[0]]) if len(idx) else 0
                cny_slice = g.loc[g["date"] == h_date]
                if len(cny_slice) and "is_cny_window" in cny_slice.columns:
                    cny_vals = {
                        c: int(cny_slice[c].iloc[0]) for c in CNY_COLS if c in cny_slice.columns
                    }
                else:
                    cny_vals = {c: 0 for c in CNY_COLS}
                row = dict(enc_dict)
                row.update(lb_map)
                row.update(lb_stats)
                row[HORIZON_IDX_COL] = h
                cal = _horizon_calendar_matrix(pd.DatetimeIndex([h_date]))
                row.update({k: int(cal[k][0]) for k in HORIZON_CAL_COLS})
                row.update(cny_vals)
                for i, c in enumerate(id_cols):
                    row[c] = keys[i] if isinstance(keys, tuple) else keys
                row["anchor_date"] = anchor
                row["date"] = h_date
                row["qty_ea"] = y_actual
                row["is_warehouse_closed"] = is_closed
                rows.append(row)

    log(f"  build_sliding_samples_slow: {len(rows):,} rows in {time.perf_counter() - t0:.1f}s")
    return pd.DataFrame(rows)


def sample_weights_sliding(df: pd.DataFrame) -> np.ndarray:
    w = np.ones(len(df), dtype=np.float64)
    closed = df["is_warehouse_closed"].fillna(0).astype(int) == 1
    w[closed] = 0.0
    if "is_cny_window" in df.columns:
        in_win = df["is_cny_window"].fillna(0).astype(int) == 1
        w[in_win & ~closed] = 0.5
    return w


def _extend_panel_through(
    panel: pd.DataFrame,
    id_cols: list[str],
    meta: pd.DataFrame,
    through: pd.Timestamp,
    closure: pd.DataFrame,
) -> pd.DataFrame:
    """Add placeholder rows through ``through`` so April anchors exist on the daily grid."""
    through = pd.Timestamp(through).normalize()
    last = pd.Timestamp(panel["date"].max())
    if through <= last:
        return panel
    new_dates = pd.date_range(last + pd.Timedelta(days=1), through, freq="D")
    if len(new_dates) == 0:
        return panel
    tuples = meta[id_cols].drop_duplicates()
    tuples["_key"] = 1
    dates_df = pd.DataFrame({"date": new_dates, "_key": 1})
    grid = tuples.merge(dates_df, on="_key").drop(columns="_key")
    grid["qty_ea"] = 0.0
    extra_cols = [c for c in meta.columns if c not in grid.columns and c not in id_cols]
    if extra_cols:
        grid = grid.merge(meta[id_cols + extra_cols], on=id_cols, how="left")
    wh_days = grid[["warehouse", "date"]].drop_duplicates().assign(qty_ea=0.0)
    cny = add_cny_features(wh_days, closure)
    grid = grid.merge(
        cny[["warehouse", "date", "is_cny_window", "is_warehouse_closed", "days_to_cny", "days_from_cny"]],
        on=["warehouse", "date"],
        how="left",
    )
    out = pd.concat([panel, grid], ignore_index=True)
    return out.sort_values(id_cols + ["date"]).drop_duplicates(id_cols + ["date"], keep="last")


def sliding_forecast_period(
    panel_history: pd.DataFrame,
    id_cols: list[str],
    categorical_enc_cols: list[str],
    static_enc: pd.DataFrame,
    anchors: pd.DatetimeIndex,
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    closure: pd.DataFrame,
) -> pd.DataFrame:
    """Predict week-by-week; after each anchor, append predictions for the next lookback."""
    if "is_warehouse_closed" not in panel_history.columns:
        panel_history = add_cny_features(panel_history, closure)

    meta_cols = [c for c in panel_history.columns if c not in ("date", "qty_ea")]
    meta = panel_history[meta_cols].drop_duplicates(id_cols)
    work = panel_history.copy()
    fc_end = pd.Timestamp(anchors.max()) + pd.Timedelta(days=HORIZON_DAYS - 1)
    work = _extend_panel_through(work, id_cols, meta, fc_end, closure)
    parts: list[pd.DataFrame] = []

    for anchor in anchors:
        batch = build_sliding_samples(
            work,
            id_cols,
            categorical_enc_cols,
            static_enc,
            pd.DatetimeIndex([anchor]),
            closure,
        )
        if batch.empty:
            continue

        pred = np.clip(predict_fn(batch), 0, None)
        batch["qty_ea"] = pred
        batch.loc[batch["is_warehouse_closed"] == 1, "qty_ea"] = 0.0

        out = batch[id_cols + ["date", "qty_ea"]].copy()
        parts.append(out)

        work = _extend_panel_through(work, id_cols, meta, fc_end, closure)
        aug = out.merge(meta, on=id_cols, how="left")
        aug = add_cny_features(aug, closure)
        work = pd.concat([work, aug], ignore_index=True)
        work = work.sort_values(id_cols + ["date"]).drop_duplicates(
            id_cols + ["date"], keep="last"
        )

    if not parts:
        return pd.DataFrame(columns=id_cols + ["date", "qty_ea"])
    return pd.concat(parts, ignore_index=True).drop_duplicates(id_cols + ["date"], keep="last")
