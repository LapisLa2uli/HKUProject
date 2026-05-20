"""Shared forecast evaluation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def evaluate_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Row-wise diagnostics on ``y_true`` vs ``y_pred`` (non-negative predictions clipped).

    **SMAPE caveat (intermittent demand):** with symmetric MAPE
    ``2|y-ŷ| / (|y|+|ŷ|)``, any row with ``y_true == 0`` and ``ŷ > 0`` contributes
    **exactly 2.0**, regardless of how small ``ŷ`` is. Expected hurdle forecasts
    ``ŷ = p·μ`` are almost always positive on non-order days, so **dense daily
    panels dominated by zeros produce SMAPE ≈ 2 × P(prediction on a zero day)**,
    not “error magnitude.” Use :func:`evaluate_arrays_positive_actual` or
    :func:`evaluate_decision_hurdle_qty` for actionable regression views.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.nan_to_num(np.clip(np.asarray(y_pred, dtype=np.float64), 0, None), nan=0.0, posinf=0.0)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = np.abs(y_true) + np.abs(y_pred)
    safe = denom > 0
    smape_terms = np.zeros_like(y_true, dtype=np.float64)
    np.divide(2 * np.abs(y_true - y_pred), denom, out=smape_terms, where=safe)
    smape = float(np.mean(smape_terms))
    bias = float(np.mean(y_pred - y_true))
    return {"mae": mae, "rmse": rmse, "smape": smape, "bias": bias}


def evaluate_arrays_positive_actual(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict[str, float], int]:
    """Same metrics restricted to rows with ``y_true > 0`` (standard quantity-error slice)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = y_true > 0
    n = int(np.sum(mask))
    if n == 0:
        return {k: float("nan") for k in ["mae", "rmse", "smape", "bias"]}, 0
    return evaluate_arrays(y_true[mask], y_pred[mask]), n


def evaluate_decision_hurdle_qty(
    y_true: np.ndarray,
    pred_qty: np.ndarray,
    p_order: np.ndarray,
    tau: float,
) -> dict[str, float]:
    """Point forecast ``pred_qty`` if ``p_order >= tau``, else 0 (sparse decision rule)."""
    pred_qty = np.clip(np.asarray(pred_qty, dtype=np.float64), 0, None)
    p_order = np.asarray(p_order, dtype=np.float64)
    pred = np.where(p_order >= tau, pred_qty, 0.0)
    return evaluate_arrays(np.asarray(y_true, dtype=np.float64), pred)


def smape_fracs(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Diagnostic fractions driving dense-panel SMAPE toward ~2."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), 0, None)
    both_zero = (y_true == 0) & (y_pred == 0)
    y_zero_yhat_pos = (y_true == 0) & (y_pred > 0)
    y_pos_yhat_zero = (y_true > 0) & (y_pred == 0)
    n = float(len(y_true))
    return {
        "frac_both_zero": float(np.mean(both_zero)),
        "frac_y0_pred_pos": float(np.mean(y_zero_yhat_pos)),
        "frac_ypos_pred0": float(np.mean(y_pos_yhat_zero)),
    }


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted MAPE: sum(|e|) / sum(|y|)."""
    num = float(np.sum(np.abs(y_true - y_pred)))
    den = float(np.sum(np.abs(y_true)))
    return num / den if den > 0 else float("nan")


def bias_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """(sum(pred) - sum(actual)) / sum(actual)."""
    a = float(np.sum(y_true))
    if a == 0:
        return float("nan")
    return float((np.sum(y_pred) - a) / a)


def sliding_horizon_block_wmape(
    df: pd.DataFrame,
    pred: np.ndarray,
    *,
    id_cols: list[str],
    anchor_col: str = "anchor_date",
    y_col: str = "qty_ea",
    open_only: bool = True,
) -> dict[str, float]:
    """WMAPE on 7-day blocks: each (tuple × anchor) sums actual/pred over the horizon."""
    d = df.copy()
    d["_pred"] = np.clip(np.asarray(pred, dtype=np.float64), 0, None)
    if open_only and "is_warehouse_closed" in d.columns:
        d = d[d["is_warehouse_closed"].fillna(0).astype(int) != 1]
    if d.empty:
        return {"block_wmape": float("nan"), "row_wmape": float("nan"), "n_blocks": 0.0}

    keys = id_cols + [anchor_col]
    blk = (
        d.groupby(keys, observed=True)
        .agg(actual=(y_col, "sum"), pred=("_pred", "sum"))
        .reset_index()
    )
    row_w = wmape(d[y_col].to_numpy(dtype=np.float64), d["_pred"].to_numpy(dtype=np.float64))
    blk_w = wmape(blk["actual"].to_numpy(dtype=np.float64), blk["pred"].to_numpy(dtype=np.float64))
    return {
        "block_wmape": float(blk_w),
        "row_wmape": float(row_w),
        "n_blocks": float(len(blk)),
    }


def intermittent_hurdle_loss_report(
    df: pd.DataFrame,
    *,
    tuple_keys: list[str],
    date_col: str = "date",
    y_col: str = "qty_ea",
    pred_qty_col: str = "pred",
    p_order_col: str = "p_order",
    decision_tau: float = 0.5,
    min_days_per_segment: int = 1,
    composite_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Aggregate loss for intermittent / periodic demand **without** dense daily SMAPE.

    Each segment is one ``tuple_keys`` × calendar month (from ``date_col``):

    - **Frequency — soft:** compare realized order-day count ``N_act = #{y>0}`` to
      ``sum_t p_t`` (Bernoulli expected order days from the hurdle head).
    - **Frequency — hard:** compare ``N_act`` to ``#{p_t ≥ τ}`` (scheduled-order days).
    - **Frequency — rates:** same comparisons normalized by segment length ``T``
      (order-day rate vs expected rate).
    - **Volume:** segment total quantity ``sum(y)`` vs ``sum(pred)`` using WMAPE
      and bias_ratio across segments (penalizes mis-sizing when frequency matches).
    - **Quantity on hits:** pooled MAE / RMSE on rows with ``y > 0`` only
      (how wrong is ``pred`` when demand actually occurred).

    The ``composite_intermittent_loss`` scalar mixes normalized frequency error,
    segment-volume WMAPE, and normalized positive-day MAE (default weights
    ``freq=0.38``, ``vol=0.37``, ``qty_pos=0.25``). Tune weights for your planning
    objective (e.g. more weight on volume if totals matter most).
    """
    wdef = {"freq": 0.38, "vol": 0.37, "qty_pos": 0.25}
    w = {**wdef, **(composite_weights or {})}

    d = df.loc[:, tuple_keys + [date_col, y_col, pred_qty_col, p_order_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d["_period"] = d[date_col].dt.to_period("M")

    gcols = tuple_keys + ["_period"]
    seg = (
        d.groupby(gcols, observed=True)
        .agg(
            T=(date_col, "count"),
            n_act=(y_col, lambda s: int(np.sum(np.asarray(s, dtype=np.float64) > 0))),
            n_soft=(p_order_col, "sum"),
            n_hard=(
                p_order_col,
                lambda s: float(np.sum(np.asarray(s, dtype=np.float64) >= decision_tau)),
            ),
            q_act=(y_col, "sum"),
            q_soft=(pred_qty_col, "sum"),
        )
        .reset_index()
    )
    seg = seg[seg["T"] >= min_days_per_segment]
    if seg.empty:
        out = {k: float("nan") for k in ["n_segments", "composite_intermittent_loss"]}
        out["note"] = "no_segments"
        return out

    T = np.maximum(seg["T"].to_numpy(dtype=np.float64), 1.0)
    n_act = seg["n_act"].to_numpy(dtype=np.float64)
    n_soft = seg["n_soft"].to_numpy(dtype=np.float64)
    n_hard = seg["n_hard"].to_numpy(dtype=np.float64)
    rate_act = n_act / T
    rate_soft = n_soft / T
    rate_hard = n_hard / T

    err_n_soft = n_act - n_soft
    err_n_hard = n_act - n_hard
    err_rate_soft = rate_act - rate_soft
    err_rate_hard = rate_act - rate_hard

    q_act = seg["q_act"].to_numpy(dtype=np.float64)
    q_soft = seg["q_soft"].to_numpy(dtype=np.float64)

    y_all = np.asarray(d[y_col].values, dtype=np.float64)
    pred_all = np.nan_to_num(
        np.asarray(d[pred_qty_col].values, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
    )
    pred_all = np.clip(pred_all, 0.0, None)
    pos_mask = y_all > 0

    pos_metrics, n_pos_rows = evaluate_arrays_positive_actual(y_all, pred_all)

    scale_freq = float(np.sqrt(np.maximum(np.mean(n_act), 1.0)))
    norm_freq = float(np.mean(np.abs(err_n_soft)) / max(scale_freq, 1e-9))
    vol_w = wmape(q_act, q_soft)
    mean_y_pos = float(np.mean(y_all[pos_mask])) if np.any(pos_mask) else 1.0
    norm_qty_pos = float(pos_metrics["mae"] / max(mean_y_pos, 1e-9))

    composite = (
        w["freq"] * norm_freq + w["vol"] * (vol_w if np.isfinite(vol_w) else 0.0) + w["qty_pos"] * norm_qty_pos
    )

    out: dict[str, Any] = {
        "n_segments": int(len(seg)),
        "mean_segment_days": float(np.mean(T)),
        "freq_count_mae_soft": float(np.mean(np.abs(err_n_soft))),
        "freq_count_rmse_soft": float(np.sqrt(np.mean(err_n_soft**2))),
        "freq_count_mae_hard": float(np.mean(np.abs(err_n_hard))),
        "freq_count_rmse_hard": float(np.sqrt(np.mean(err_n_hard**2))),
        "freq_rate_mae_soft": float(np.mean(np.abs(err_rate_soft))),
        "freq_rate_mae_hard": float(np.mean(np.abs(err_rate_hard))),
        "segment_volume_wmape": float(vol_w),
        "segment_volume_bias_ratio": float(bias_ratio(q_act, q_soft)),
        "positive_day_n_rows": int(n_pos_rows),
        "positive_day_mae": float(pos_metrics["mae"]),
        "positive_day_rmse": float(pos_metrics["rmse"]),
        "positive_day_bias": float(pos_metrics["bias"]),
        "norm_freq_err_sqrt_mean_n": norm_freq,
        "norm_qty_pos_mae_div_mean_y": norm_qty_pos,
        "composite_intermittent_loss": float(composite),
        "composite_w_freq": float(w["freq"]),
        "composite_w_vol": float(w["vol"]),
        "composite_w_qty_pos": float(w["qty_pos"]),
        "decision_tau": float(decision_tau),
    }
    return out


def _date_to_ordinals(series: pd.Series) -> np.ndarray:
    d = pd.to_datetime(series, errors="coerce").dt.normalize()
    return d.map(lambda x: x.toordinal() if pd.notna(x) else np.nan).to_numpy(dtype=np.float64)


def _match_store_product_orders(
    actual: pd.DataFrame,
    predicted: pd.DataFrame,
    *,
    product_col: str,
    y_col: str,
    pred_col: str,
    window_days: int,
    qty_rel_tol: float,
) -> dict[str, float]:
    """Greedy match actual order lines to predicted lines (same product, within day window)."""
    act_qty_total = float(actual[y_col].sum()) if not actual.empty else 0.0
    pred_qty_total = float(predicted[pred_col].sum()) if not predicted.empty else 0.0

    if act_qty_total <= 0 and pred_qty_total <= 0:
        return {
            "actual_order_qty": 0.0,
            "pred_order_qty": 0.0,
            "matched_actual_qty": 0.0,
            "miss_qty": 0.0,
            "false_qty": 0.0,
            "matched_qty_abs_err": 0.0,
            "n_actual_lines": 0.0,
            "n_pred_lines": 0.0,
            "n_matched_lines": 0.0,
            "mean_abs_day_slip": float("nan"),
            "normalized_loss": 0.0,
        }

    act = actual.sort_values("date_ord").reset_index(drop=True)
    pred = predicted.sort_values("date_ord").reset_index(drop=True)
    pred_used: set[int] = set()

    matched_actual_qty = 0.0
    matched_qty_abs_err = 0.0
    miss_qty = 0.0
    day_slips: list[float] = []
    n_matched = 0

    act_dates = act["date_ord"].to_numpy(dtype=np.float64)
    act_prods = act[product_col].astype(str).to_numpy()
    act_qtys = act[y_col].to_numpy(dtype=np.float64)

    pred_dates = pred["date_ord"].to_numpy(dtype=np.float64)
    pred_prods = pred[product_col].astype(str).to_numpy()
    pred_qtys = pred[pred_col].to_numpy(dtype=np.float64)

    for i in range(len(act)):
        q_a = act_qtys[i]
        best_j: int | None = None
        best_err = float("inf")
        for j in range(len(pred)):
            if j in pred_used:
                continue
            if pred_prods[j] != act_prods[i]:
                continue
            if abs(pred_dates[j] - act_dates[i]) > window_days:
                continue
            err = abs(q_a - pred_qtys[j])
            if err < best_err:
                best_err = err
                best_j = j
        if best_j is None:
            miss_qty += q_a
            continue

        pred_used.add(best_j)
        n_matched += 1
        matched_actual_qty += q_a
        q_p = pred_qtys[best_j]
        rel_err = abs(q_a - q_p) / max(q_a, 1e-9)
        if rel_err > qty_rel_tol:
            matched_qty_abs_err += abs(q_a - q_p)
        day_slips.append(abs(pred_dates[best_j] - act_dates[i]))

    false_qty = float(
        np.sum([pred_qtys[j] for j in range(len(pred)) if j not in pred_used])
    )
    loss_mass = matched_qty_abs_err + miss_qty + false_qty
    normalized_loss = loss_mass / max(act_qty_total, 1e-9)

    return {
        "actual_order_qty": act_qty_total,
        "pred_order_qty": pred_qty_total,
        "matched_actual_qty": matched_actual_qty,
        "miss_qty": miss_qty,
        "false_qty": false_qty,
        "matched_qty_abs_err": matched_qty_abs_err,
        "n_actual_lines": float(len(act)),
        "n_pred_lines": float(len(pred)),
        "n_matched_lines": float(n_matched),
        "mean_abs_day_slip": float(np.mean(day_slips)) if day_slips else float("nan"),
        "normalized_loss": normalized_loss,
    }


def store_product_window_match_loss(
    df: pd.DataFrame,
    *,
    store_keys: list[str],
    product_col: str = "product_id",
    date_col: str = "date",
    y_col: str = "qty_ea",
    pred_col: str = "pred",
    window_days: int = 2,
    min_pred_qty: float = 0.0,
    qty_rel_tol: float = 0.15,
) -> dict[str, Any]:
    """Per-store loss that prioritizes product identity and quantity over exact order day.

    Each store's actual order lines (``y_col > 0``) are matched to predicted order lines
    (``pred_col > min_pred_qty``) with the **same product** and order date within
    ``±window_days``. Unmatched actual volume is penalized (missed product/qty); unmatched
    predicted volume is penalized (wrong product or phantom order). Matched lines incur
    quantity error only when relative error exceeds ``qty_rel_tol`` (approximate amount OK).

    **Lower is better.** Day slip within the window adds no extra penalty.
    """
    cols = store_keys + [date_col, product_col, y_col, pred_col]
    d = df.loc[:, cols].copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce").dt.normalize()
    d["date_ord"] = _date_to_ordinals(d[date_col])
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    d[pred_col] = pd.to_numeric(d[pred_col], errors="coerce").fillna(0.0).clip(lower=0.0)

    store_rows: list[dict[str, float]] = []
    for _, g in d.groupby(store_keys, sort=False, observed=True):
        actual = g.loc[g[y_col] > 0, ["date_ord", product_col, y_col]]
        predicted = g.loc[g[pred_col] > min_pred_qty, ["date_ord", product_col, pred_col]]
        store_rows.append(_match_store_product_orders(
            actual,
            predicted,
            product_col=product_col,
            y_col=y_col,
            pred_col=pred_col,
            window_days=window_days,
            qty_rel_tol=qty_rel_tol,
        ))

    if not store_rows:
        return {"n_stores": 0, "planning_window_loss": float("nan"), "note": "no_stores"}

    act_qty = np.array([r["actual_order_qty"] for r in store_rows], dtype=np.float64)
    weights = np.maximum(act_qty, 0.0)
    w_sum = float(weights.sum())

    def _wmean(key: str) -> float:
        vals = np.array([r[key] for r in store_rows], dtype=np.float64)
        if w_sum <= 0:
            return float(np.mean(vals)) if len(vals) else float("nan")
        return float(np.average(vals, weights=weights))

    matched_qty = float(np.sum([r["matched_actual_qty"] for r in store_rows]))
    act_total = float(np.sum(act_qty))
    pred_total = float(np.sum([r["pred_order_qty"] for r in store_rows]))

    return {
        "n_stores": int(len(store_rows)),
        "window_days": int(window_days),
        "qty_rel_tol": float(qty_rel_tol),
        "actual_order_qty_total": act_total,
        "pred_order_qty_total": pred_total,
        "matched_volume_rate": matched_qty / act_total if act_total > 0 else float("nan"),
        "miss_volume_rate": float(np.sum([r["miss_qty"] for r in store_rows])) / act_total
        if act_total > 0
        else float("nan"),
        "false_volume_rate": float(np.sum([r["false_qty"] for r in store_rows])) / act_total
        if act_total > 0
        else float("nan"),
        "mean_abs_day_slip_matched": _wmean("mean_abs_day_slip"),
        "planning_window_loss": _wmean("normalized_loss"),
        "n_actual_order_lines": float(np.sum([r["n_actual_lines"] for r in store_rows])),
        "n_pred_order_lines": float(np.sum([r["n_pred_lines"] for r in store_rows])),
        "n_matched_order_lines": float(np.sum([r["n_matched_lines"] for r in store_rows])),
        "line_match_rate": float(np.sum([r["n_matched_lines"] for r in store_rows]))
        / float(np.sum([r["n_actual_lines"] for r in store_rows]))
        if float(np.sum([r["n_actual_lines"] for r in store_rows])) > 0
        else float("nan"),
    }


def store_planning_loss_report(
    df: pd.DataFrame,
    *,
    store_keys: list[str],
    product_col: str = "product_id",
    date_col: str = "date",
    y_col: str = "qty_ea",
    pred_col: str = "pred",
    window_days: int = 2,
    min_pred_qty: float = 0.0,
    qty_rel_tol: float = 0.15,
    composite_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Store planning loss vs strict same-day loss (for validation dashboards).

    **Planning window loss** (primary): :func:`store_product_window_match_loss` —
    rewards correct product + approximate quantity even if the order day is off by up to
    ``window_days``.

    **Strict daily WMAPE** (reference): standard ``sum(|y-ŷ|)/sum(y)`` on all rows at
    store×product×day grain — timing errors count fully.

    ``composite_planning_loss`` mixes window loss (default 0.75) and strict daily WMAPE
    (default 0.25) so product/qty accuracy dominates but gross daily misalignment still
    contributes.
    """
    wdef = {"window": 0.75, "strict_daily": 0.25}
    w = {**wdef, **(composite_weights or {})}

    window = store_product_window_match_loss(
        df,
        store_keys=store_keys,
        product_col=product_col,
        date_col=date_col,
        y_col=y_col,
        pred_col=pred_col,
        window_days=window_days,
        min_pred_qty=min_pred_qty,
        qty_rel_tol=qty_rel_tol,
    )

    d = df.loc[:, store_keys + [date_col, product_col, y_col, pred_col]].copy()
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce").fillna(0.0)
    d[pred_col] = pd.to_numeric(d[pred_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    strict = wmape(d[y_col].to_numpy(dtype=np.float64), d[pred_col].to_numpy(dtype=np.float64))

    window_loss = float(window.get("planning_window_loss", float("nan")))
    strict_f = float(strict) if np.isfinite(strict) else 0.0
    composite = w["window"] * window_loss + w["strict_daily"] * strict_f

    out: dict[str, Any] = dict(window)
    out["strict_daily_wmape"] = float(strict)
    out["composite_planning_loss"] = float(composite)
    out["composite_w_window"] = float(w["window"])
    out["composite_w_strict_daily"] = float(w["strict_daily"])
    return out
