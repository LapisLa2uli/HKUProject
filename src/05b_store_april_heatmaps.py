"""April predicted qty_ea heatmaps: store × calendar day (one PNG per model).

Baseline forecasts are at (warehouse, customer_id, product_id); they are split across
stores using Jan–Mar historical qty shares from ``mart_warehouse_store_product_day.csv``.
Hurdle forecasts already include ``store``.

Outputs under ``reports/figures``:
    20_april_store_day_heatmap_baseline.png
    21_april_store_day_heatmap_hurdle_pattern.png
    22_april_store_day_heatmap_hurdle_store.png
    23_store_history_plus_april_heatmap_hurdle_pattern.png
    24_store_history_plus_april_heatmap_hurdle_store.png
    25_store_history_plus_april_heatmap_hurdle_pattern_top20.png
    26_store_history_plus_april_heatmap_hurdle_store_top20.png

Hurdle heatmaps use pattern and per-store-calibrated April daily exports (not network-scaled).
April-only heatmaps use top N stores (``HEATMAP_TOP_N_STORES``); history+April figures include every store.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 110

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from _report_paths import (
    BASELINE_APRIL_DAILY,
    FIGURES,
    HEATMAP_HIST_HURDLE_PATTERN,
    HEATMAP_HIST_HURDLE_PATTERN_TOP20,
    HEATMAP_HIST_HURDLE_STORE,
    HEATMAP_HIST_HURDLE_STORE_TOP20,
    HEATMAP_HIST_STORE_REG,
    HEATMAP_HIST_STORE_REG_TOP20,
    HEATMAP_ROW_BASELINE,
    HEATMAP_ROW_HURDLE_PATTERN,
    HEATMAP_ROW_HURDLE_STORE,
    HEATMAP_ROW_STORE_REG,
    HURDLE_APRIL_CALIBRATION,
    HURDLE_APRIL_DAILY_NETWORK,
    HURDLE_APRIL_DAILY_PATTERN,
    HURDLE_APRIL_DAILY_STORE,
    HURDLE_APRIL_STORE_CALIBRATION,
    STORE_REG_APRIL_DAILY,
    ensure_report_dirs,
)
from _report_paths import PROCESSED

MARTS = PROCESSED / "marts"
FIG = FIGURES

HISTORY_END = pd.Timestamp("2026-03-28")
FORECAST_START = pd.Timestamp("2026-04-01")
TRIPLE_COLS = ["warehouse", "customer_id", "product_id"]
HEATMAP_TOP_N_STORES = 20
STORE_KEYS = ["warehouse", "customer_id", "store"]


def load_hurdle_april_dailies() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Pattern and per-store-calibrated April hurdle daily (excludes network scale)."""
    if HURDLE_APRIL_DAILY_PATTERN.exists() and HURDLE_APRIL_DAILY_STORE.exists():
        fp = pd.read_csv(HURDLE_APRIL_DAILY_PATTERN, parse_dates=["date"])
        fs = pd.read_csv(HURDLE_APRIL_DAILY_STORE, parse_dates=["date"])
        return fp, fs

    if not (
        HURDLE_APRIL_DAILY_NETWORK.exists()
        and HURDLE_APRIL_CALIBRATION.exists()
        and HURDLE_APRIL_STORE_CALIBRATION.exists()
    ):
        return None

    print(
        "[info] Deriving hurdle pattern/store daily from network-scaled export "
        "(re-run 04b_hurdle_model.py to write dedicated CSVs)."
    )
    scaled = pd.read_csv(HURDLE_APRIL_DAILY_NETWORK, parse_dates=["date"])
    cal = pd.read_csv(HURDLE_APRIL_CALIBRATION)
    ns_row = cal.loc[cal["metric"] == "april_network_scale", "value"]
    if ns_row.empty:
        ns_row = cal.loc[cal["metric"] == "april_scale", "value"]
    ns = float(ns_row.iloc[0])
    store_scales = (
        pd.read_csv(HURDLE_APRIL_STORE_CALIBRATION)
        .set_index(STORE_KEYS)["store_scale"]
        .astype(np.float64)
    )
    keys = scaled[STORE_KEYS].apply(tuple, axis=1)
    mult = keys.map(store_scales).fillna(1.0).astype(np.float64).values
    qty_scaled = scaled["qty_ea"].astype(np.float64).values
    qty_store = qty_scaled / ns
    qty_pattern = np.where(mult > 1e-12, qty_store / mult, qty_store)
    fs = scaled.copy()
    fs["qty_ea"] = qty_store
    fp = scaled.copy()
    fp["qty_ea"] = qty_pattern
    return fp, fs


def save(fig: plt.Figure, name: str) -> Path:
    out = FIG / name
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out}")
    return out


def historical_store_shares(mart: pd.DataFrame) -> pd.DataFrame:
    """Per (warehouse, customer, product, store): share of tuple qty in Jan–Mar history."""
    g = (
        mart.groupby(TRIPLE_COLS + ["store"], as_index=False)["qty_ea"]
        .sum()
        .rename(columns={"qty_ea": "hist_qty"})
    )
    tot = g.groupby(TRIPLE_COLS)["hist_qty"].transform("sum")
    g["share"] = np.where(tot > 0, g["hist_qty"] / tot, np.nan)
    return g


def fallback_stores_per_triple(mart: pd.DataFrame) -> pd.DataFrame:
    """Stores observed per triple (for uniform split when hist sum is zero / missing)."""
    return (
        mart.groupby(TRIPLE_COLS)["store"]
        .agg(lambda s: sorted(pd.unique(s)))
        .reset_index()
        .rename(columns={"store": "stores"})
    )


def baseline_forecast_by_store(
    april_base: pd.DataFrame,
    mart_hist: pd.DataFrame,
) -> pd.DataFrame:
    """Explode baseline April preds to store grain using historical shares."""
    shares = historical_store_shares(mart_hist)
    fb = fallback_stores_per_triple(mart_hist)

    # Rows where mart had strictly positive history for this triple → weighted shares.
    hist_part = april_base.merge(
        shares[TRIPLE_COLS + ["store", "share"]].dropna(subset=["share"]),
        on=TRIPLE_COLS,
        how="inner",
    )
    hist_triples = hist_part[TRIPLE_COLS].drop_duplicates()

    missing_keys = april_base[TRIPLE_COLS].drop_duplicates().merge(
        hist_triples.assign(_has_hist=1),
        on=TRIPLE_COLS,
        how="left",
    )
    missing_keys = missing_keys[missing_keys["_has_hist"].isna()][TRIPLE_COLS]

    fb_long = fb.explode("stores", ignore_index=True).rename(columns={"stores": "store"})
    n_per = fb_long.groupby(TRIPLE_COLS)["store"].transform("count")
    fb_long["share"] = np.where(n_per > 0, 1.0 / n_per.astype(np.float64), np.nan)

    fallback_part = april_base.merge(missing_keys, on=TRIPLE_COLS, how="inner").merge(
        fb_long,
        on=TRIPLE_COLS,
        how="inner",
    )

    m = pd.concat([hist_part, fallback_part], ignore_index=True)
    m = m.dropna(subset=["share"])
    m["qty_ea"] = m["qty_ea"].astype(np.float64) * m["share"].astype(np.float64)
    return m[["store", "date", "qty_ea"]]


def store_day_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Daily store-level qty_ea totals."""
    return df.groupby(["store", "date"], as_index=False)["qty_ea"].sum()


def history_plus_april_matrix(
    mart_hist: pd.DataFrame,
    april_store: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """All stores × (history dates + April forecast dates); rows sorted by combined total."""
    hist = store_day_totals(mart_hist)
    apr = store_day_totals(april_store)
    combined = pd.concat([hist, apr], ignore_index=True)
    pt = combined.pivot_table(
        index="store", columns="date", values="qty_ea", aggfunc="sum", fill_value=0.0
    )
    pt = pt.sort_index(axis=1)
    row_tot = pt.sum(axis=1).sort_values(ascending=False)
    pt = pt.loc[row_tot.index]
    # Index of first April column (for vertical marker)
    apr_cols = [c for c in pt.columns if pd.Timestamp(c) >= FORECAST_START]
    apr_ix = pt.columns.get_loc(apr_cols[0]) if apr_cols else len(pt.columns)
    return pt, int(apr_ix)


def plot_history_plus_april_heatmap(
    pt: pd.DataFrame,
    apr_start_ix: int,
    title: str,
    fname: str,
    cmap: str = "YlOrRd",
    *,
    cbar_label: str = "qty_ea (actual Jan–Mar 28 + hurdle forecast Apr)",
) -> None:
    """Heatmap: rows = all stores, cols = history actuals then April forecast."""
    arr = pt.values.astype(np.float64)
    n_store, n_day = arr.shape
    h_in = float(np.clip(8.0 + n_store / 55.0, 12.0, 72.0))
    fig_w = max(14.0, min(22.0, 8.0 + n_day * 0.08))
    fig, ax = plt.subplots(figsize=(fig_w, h_in))

    vmax = float(np.percentile(arr[arr > 0], 99)) if np.any(arr > 0) else 1.0
    vmax = max(vmax, 1.0)
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=vmax)

    im = ax.imshow(arr, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, shrink=min(1.0, 60.0 / h_in), fraction=0.02)
    cbar.set_label(cbar_label)

    if 0 < apr_start_ix < n_day:
        ax.axvline(apr_start_ix - 0.5, color="white", linewidth=1.2, linestyle="--")
        ax.axvline(apr_start_ix - 0.5, color="black", linewidth=0.6, linestyle="--")

    n_hist = apr_start_ix
    n_apr = n_day - apr_start_ix
    ax.set_title(
        title
        + f"\n({n_store} stores × {n_day} days: {n_hist}d actual + {n_apr}d forecast; "
        "rows by total qty, highest at top; dashed line = Apr 1)"
    )
    ax.set_xlabel("date")
    ax.set_ylabel("store (rank)")

    # X ticks: sparse labels
    step_x = max(1, n_day // 20)
    x_ix = np.arange(0, n_day, step_x)
    ax.set_xticks(x_ix)
    x_labels = [
        pt.columns[i].strftime("%m-%d") if hasattr(pt.columns[i], "strftime") else str(pt.columns[i])[:10]
        for i in x_ix
    ]
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)

    if n_store <= 40:
        ax.set_yticks(np.arange(n_store))
        ax.set_yticklabels(pt.index, fontsize=6)
    else:
        step_y = max(1, n_store // 30)
        y_ix = np.arange(0, n_store, step_y)
        ax.set_yticks(y_ix)
        ax.set_yticklabels([f"{i + 1}" for i in y_ix], fontsize=6)

    fig.tight_layout()
    save(fig, fname)


def store_day_matrix(df: pd.DataFrame, top_n: int | None = None) -> pd.DataFrame:
    """Rows = stores (sorted by April total desc), cols = calendar dates."""
    pt = df.pivot_table(index="store", columns="date", values="qty_ea", aggfunc="sum", fill_value=0.0)
    pt = pt.sort_index(axis=1)
    row_tot = pt.sum(axis=1).sort_values(ascending=False)
    pt = pt.loc[row_tot.index]
    if top_n is not None:
        pt = pt.head(int(top_n))
    return pt


def plot_store_day_heatmap(
    pt: pd.DataFrame,
    title: str,
    fname: str,
    cmap: str,
    *,
    top_n_label: int | None = None,
) -> None:
    """Raster heatmap; y-axis is store name (few rows) or rank ticks (many rows)."""
    arr = pt.values.astype(np.float64)
    n_store, n_day = arr.shape

    # Aspect: compact for top-N stores; taller when many rows.
    if n_store <= 40:
        h_in = float(np.clip(5.0 + n_store * 0.35, 7.0, 22.0))
    else:
        h_in = float(np.clip(6.0 + n_store / 70.0, 8.0, 48.0))
    fig_w = 12.0
    fig, ax = plt.subplots(figsize=(fig_w, h_in))

    vmax = float(arr.max())
    if vmax <= 0:
        vmax = 1.0
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=vmax)

    im = ax.imshow(arr, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, shrink=min(1.0, 48.0 / h_in), fraction=0.035)
    cbar.set_label("predicted qty_ea")

    suffix = (
        f"\n(top {top_n_label} stores × {n_day} days by April predicted total; highest at top)"
        if top_n_label is not None
        else f"\n({n_store} stores × {n_day} days; rows sorted by April total, highest at top)"
    )
    ax.set_title(title + suffix)
    ax.set_xlabel("April date")
    ax.set_ylabel("store")

    xticks = np.arange(n_day)
    ax.set_xticks(xticks)
    labels = [d.strftime("%m-%d") if hasattr(d, "strftime") else str(d)[:10] for d in pt.columns]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    # Sparse rank ticks on y (many stores)
    if n_store <= 40:
        ax.set_yticks(np.arange(n_store))
        ax.set_yticklabels(pt.index, fontsize=6)
    else:
        step = max(1, n_store // 25)
        y_ix = np.arange(0, n_store, step)
        ax.set_yticks(y_ix)
        ax.set_yticklabels([f"{i + 1}" for i in y_ix], fontsize=7)

    fig.tight_layout()
    save(fig, fname)


def main() -> int:
    ensure_report_dirs()
    FIG.mkdir(parents=True, exist_ok=True)
    mart_path = MARTS / "mart_warehouse_store_product_day.csv"
    if not mart_path.exists():
        print(f"Missing mart: {mart_path}", file=sys.stderr)
        return 1

    mart = pd.read_csv(
        mart_path,
        parse_dates=["create_date"],
        dtype={"warehouse": str, "customer_id": str, "product_id": str, "store": str},
    ).rename(columns={"create_date": "date"})
    mart["qty_ea"] = pd.to_numeric(mart["qty_ea"], errors="coerce").fillna(0.0)
    mart_hist = mart[mart["date"] <= HISTORY_END].copy()

    if not BASELINE_APRIL_DAILY.exists():
        print(f"Missing {BASELINE_APRIL_DAILY}", file=sys.stderr)
        return 1

    hurdle = load_hurdle_april_dailies()
    if hurdle is None:
        print("Missing hurdle pattern/store daily exports.", file=sys.stderr)
        return 1

    april_base = pd.read_csv(BASELINE_APRIL_DAILY, parse_dates=["date"])
    april_pattern, april_store = hurdle

    base_store = baseline_forecast_by_store(april_base, mart_hist)
    pattern_store = april_pattern[["store", "date", "qty_ea"]].copy()
    store_cal_store = april_store[["store", "date", "qty_ea"]].copy()

    n_top = HEATMAP_TOP_N_STORES
    pt_b = store_day_matrix(base_store, top_n=n_top)
    pt_hp = store_day_matrix(pattern_store, top_n=n_top)
    pt_hs = store_day_matrix(store_cal_store, top_n=n_top)

    plot_store_day_heatmap(
        pt_b,
        "Baseline — April predicted qty_ea by store and day",
        "20_april_store_day_heatmap_baseline.png",
        cmap="viridis",
        top_n_label=n_top,
    )
    plot_store_day_heatmap(
        pt_hp,
        "Hurdle pattern — April predicted qty_ea by store and day",
        "21_april_store_day_heatmap_hurdle_pattern.png",
        cmap="magma",
        top_n_label=n_top,
    )
    plot_store_day_heatmap(
        pt_hs,
        "Hurdle store-cal — April predicted qty_ea by store and day",
        "22_april_store_day_heatmap_hurdle_store.png",
        cmap="magma",
        top_n_label=n_top,
    )

    meta_b = pd.DataFrame({"rank": np.arange(1, len(pt_b) + 1), "store": pt_b.index})
    meta_hp = pd.DataFrame({"rank": np.arange(1, len(pt_hp) + 1), "store": pt_hp.index})
    meta_hs = pd.DataFrame({"rank": np.arange(1, len(pt_hs) + 1), "store": pt_hs.index})
    meta_b.to_csv(HEATMAP_ROW_BASELINE, index=False, encoding="utf-8-sig")
    meta_hp.to_csv(HEATMAP_ROW_HURDLE_PATTERN, index=False, encoding="utf-8-sig")
    meta_hs.to_csv(HEATMAP_ROW_HURDLE_STORE, index=False, encoding="utf-8-sig")
    print(f"Wrote: {HEATMAP_ROW_BASELINE}")
    print(f"Wrote: {HEATMAP_ROW_HURDLE_PATTERN}")
    print(f"Wrote: {HEATMAP_ROW_HURDLE_STORE}")

    pt_all_p, apr_ix = history_plus_april_matrix(mart_hist, pattern_store)
    plot_history_plus_april_heatmap(
        pt_all_p,
        apr_ix,
        "Hurdle pattern — store daily qty_ea: history (actual) + April (forecast)",
        "23_store_history_plus_april_heatmap_hurdle_pattern.png",
        cbar_label="qty_ea (actual Jan–Mar 28 + hurdle pattern Apr)",
    )
    pd.DataFrame({"rank": np.arange(1, len(pt_all_p) + 1), "store": pt_all_p.index}).to_csv(
        HEATMAP_HIST_HURDLE_PATTERN, index=False, encoding="utf-8-sig"
    )
    print(f"Wrote: {HEATMAP_HIST_HURDLE_PATTERN}")

    pt_all_s, _ = history_plus_april_matrix(mart_hist, store_cal_store)
    plot_history_plus_april_heatmap(
        pt_all_s,
        apr_ix,
        "Hurdle store-cal — store daily qty_ea: history (actual) + April (forecast)",
        "24_store_history_plus_april_heatmap_hurdle_store.png",
        cbar_label="qty_ea (actual Jan–Mar 28 + hurdle store-cal Apr)",
    )
    pd.DataFrame({"rank": np.arange(1, len(pt_all_s) + 1), "store": pt_all_s.index}).to_csv(
        HEATMAP_HIST_HURDLE_STORE, index=False, encoding="utf-8-sig"
    )
    print(f"Wrote: {HEATMAP_HIST_HURDLE_STORE}")

    pt_top_p = pt_all_p.head(n_top)
    plot_history_plus_april_heatmap(
        pt_top_p,
        apr_ix,
        f"Hurdle pattern — top {n_top} stores: daily qty_ea history (actual) + April (forecast)",
        "25_store_history_plus_april_heatmap_hurdle_pattern_top20.png",
        cbar_label="qty_ea (actual Jan–Mar 28 + hurdle pattern Apr)",
    )
    pd.DataFrame({"rank": np.arange(1, len(pt_top_p) + 1), "store": pt_top_p.index}).to_csv(
        HEATMAP_HIST_HURDLE_PATTERN_TOP20, index=False, encoding="utf-8-sig"
    )
    print(f"Wrote: {HEATMAP_HIST_HURDLE_PATTERN_TOP20}")

    pt_top_s = pt_all_s.head(n_top)
    plot_history_plus_april_heatmap(
        pt_top_s,
        apr_ix,
        f"Hurdle store-cal — top {n_top} stores: daily qty_ea history (actual) + April (forecast)",
        "26_store_history_plus_april_heatmap_hurdle_store_top20.png",
        cbar_label="qty_ea (actual Jan–Mar 28 + hurdle store-cal Apr)",
    )
    pd.DataFrame({"rank": np.arange(1, len(pt_top_s) + 1), "store": pt_top_s.index}).to_csv(
        HEATMAP_HIST_HURDLE_STORE_TOP20, index=False, encoding="utf-8-sig"
    )
    print(f"Wrote: {HEATMAP_HIST_HURDLE_STORE_TOP20}")

    if STORE_REG_APRIL_DAILY.exists():
        april_sr = pd.read_csv(STORE_REG_APRIL_DAILY, parse_dates=["date"])
        sr_store = april_sr[["store", "date", "qty_ea"]].copy()
        pt_sr = store_day_matrix(sr_store, top_n=n_top)
        plot_store_day_heatmap(
            pt_sr,
            "Store regression — April predicted qty_ea by store and day",
            "32_april_store_day_heatmap_store_regression.png",
            cmap="Greens",
            top_n_label=n_top,
        )
        pd.DataFrame({"rank": np.arange(1, len(pt_sr) + 1), "store": pt_sr.index}).to_csv(
            HEATMAP_ROW_STORE_REG, index=False, encoding="utf-8-sig"
        )
        print(f"Wrote: {HEATMAP_ROW_STORE_REG}")

        pt_all_sr, apr_ix_sr = history_plus_april_matrix(mart_hist, sr_store)
        plot_history_plus_april_heatmap(
            pt_all_sr,
            apr_ix_sr,
            "Store regression — store daily qty_ea: history + April",
            "33_store_history_plus_april_heatmap_store_regression.png",
            cbar_label="qty_ea (actual Jan–Mar 28 + store regression Apr)",
        )
        pd.DataFrame({"rank": np.arange(1, len(pt_all_sr) + 1), "store": pt_all_sr.index}).to_csv(
            HEATMAP_HIST_STORE_REG, index=False, encoding="utf-8-sig"
        )
        print(f"Wrote: {HEATMAP_HIST_STORE_REG}")

        pt_top_sr = pt_all_sr.head(n_top)
        plot_history_plus_april_heatmap(
            pt_top_sr,
            apr_ix_sr,
            f"Store regression — top {n_top} stores: history + April",
            "34_store_history_plus_april_heatmap_store_regression_top20.png",
            cbar_label="qty_ea (actual Jan–Mar 28 + store regression Apr)",
        )
        pd.DataFrame({"rank": np.arange(1, len(pt_top_sr) + 1), "store": pt_top_sr.index}).to_csv(
            HEATMAP_HIST_STORE_REG_TOP20, index=False, encoding="utf-8-sig"
        )
        print(f"Wrote: {HEATMAP_HIST_STORE_REG_TOP20}")
    else:
        print("[skip] Store regression April daily missing; heatmaps 32–34 not generated.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
