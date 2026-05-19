"""Diagnose intermittent-order smearing for a store (history vs model April)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

STORE = sys.argv[1] if len(sys.argv) > 1 else "深圳南太云创谷店"
MARTS = ROOT / "processed/marts/mart_warehouse_store_product_day.csv"
ID_COLS = ["warehouse", "store", "product_id"]


def store_daily(df: pd.DataFrame, label: str) -> None:
    s = df[df["store"] == STORE].copy()
    if s.empty:
        print(f"{label}: no rows for store")
        return
    d = s.groupby("date")["qty_ea"].sum().sort_index()
    pos = d[d > 0]
    gaps = pos.index.to_series().diff().dt.days.dropna()
    print(f"=== {label} ===")
    print(f"days: {len(d)}, order days: {(d > 0).sum()}, pos_rate: {(d > 0).mean():.3f}")
    if len(pos):
        print(
            f"qty on order days: mean={pos.mean():.1f} med={pos.median():.1f} "
            f"min={pos.min():.1f} max={pos.max():.1f}"
        )
    if len(gaps):
        print(f"inter-order gap days: mean={gaps.mean():.2f} med={gaps.median():.1f}")
    print("last order days in range:")
    print(pos.tail(12).to_string())
    print()


def main() -> None:
    hist = pd.read_csv(MARTS, parse_dates=["create_date"])
    hist = hist.rename(columns={"create_date": "date"})
    h = hist[(hist["store"] == STORE) & (hist["date"] <= "2026-03-28")]
    store_daily(h, "History Jan-Mar")

    sr = pd.read_csv(
        ROOT / "reports/store_regression/april_forecast_store_regression_daily.csv",
        parse_dates=["date"],
    )
    store_daily(sr[sr["date"] >= "2026-04-01"], "Store reg April")

    for rel, name in [
        ("reports/hurdle/april_forecast_hurdle_daily_store.csv", "Hurdle store-cal April"),
        ("reports/hurdle/april_forecast_hurdle_daily_pattern.csv", "Hurdle pattern April"),
    ]:
        p = ROOT / rel
        if p.exists():
            hd = pd.read_csv(p, parse_dates=["date"])
            store_daily(hd[hd["store"] == STORE], name)

    # Tuple-level: fraction of April days with tiny positive preds
    april = sr[(sr["store"] == STORE) & (sr["date"] >= "2026-04-01")]
    tiny = april[(april["qty_ea"] > 0) & (april["qty_ea"] < 50)]
    print(
        f"Store-reg product-days: total={len(april):,}  "
        f"pos={(april['qty_ea']>0).sum():,}  "
        f"tiny 0<qty<50={len(tiny):,} ({100*len(tiny)/max(1,len(april)):.1f}%)"
    )
    by_prod = april.groupby("product_id")["qty_ea"].agg(
        pos_rate=lambda s: (s > 0).mean(),
        mean_all="mean",
        mean_pos=lambda s: s[s > 0].mean() if (s > 0).any() else 0.0,
    )
    print("Top products by April pos_rate (smearing):")
    print(by_prod.sort_values("pos_rate", ascending=False).head(8))


if __name__ == "__main__":
    main()
