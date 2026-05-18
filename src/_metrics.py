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
