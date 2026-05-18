"""Step 2: Build warehouse-centric data marts for analysis and modeling.

Inputs (from step 1):
    processed/orders.csv
    processed/order_items.csv

Outputs (CSV in processed/marts/):
    mart_warehouse_day.csv
    mart_warehouse_customer_day.csv
    mart_warehouse_product_day.csv
    mart_warehouse_customer_product_day.csv
    mart_warehouse_store_product_day.csv
    mart_warehouse_summary.csv
    mart_top_products_per_warehouse.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "processed"
MARTS = PROCESSED / "marts"
MARTS.mkdir(parents=True, exist_ok=True)


def load_items() -> pd.DataFrame:
    items = pd.read_csv(
        PROCESSED / "order_items.csv",
        dtype={"order_id": str, "product_id": str, "customer_id": str},
        parse_dates=["create_time"],
        low_memory=False,
    )
    items["create_date"] = pd.to_datetime(items["create_date"], errors="coerce")
    items["qty_ea"] = pd.to_numeric(items["qty_ea"], errors="coerce").fillna(0)
    return items


def write(df: pd.DataFrame, name: str) -> Path:
    out = MARTS / name
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return out


def main() -> int:
    items = load_items()
    items = items.dropna(subset=["create_date", "warehouse"])
    items["create_date"] = items["create_date"].dt.normalize()

    base_keys = ["warehouse", "create_date"]
    wh_day = (
        items.groupby(base_keys, dropna=False)
        .agg(
            orders=("order_id", "nunique"),
            order_lines=("order_id", "count"),
            qty_ea=("qty_ea", "sum"),
            unique_customers=("customer_id", "nunique"),
            unique_products=("product_id", "nunique"),
        )
        .reset_index()
        .sort_values(base_keys)
    )

    wh_cust_day = (
        items.groupby(["warehouse", "customer_id", "customer_name", "create_date"], dropna=False)
        .agg(
            orders=("order_id", "nunique"),
            qty_ea=("qty_ea", "sum"),
            unique_products=("product_id", "nunique"),
        )
        .reset_index()
        .sort_values(["warehouse", "customer_id", "create_date"])
    )

    wh_prod_day = (
        items.groupby(
            ["warehouse", "product_id", "product_name", "temperature_zone", "create_date"],
            dropna=False,
        )
        .agg(
            orders=("order_id", "nunique"),
            qty_ea=("qty_ea", "sum"),
        )
        .reset_index()
        .sort_values(["warehouse", "product_id", "create_date"])
    )

    wh_cust_prod_day = (
        items.groupby(
            [
                "warehouse",
                "customer_id",
                "product_id",
                "product_name",
                "temperature_zone",
                "create_date",
            ],
            dropna=False,
        )
        .agg(qty_ea=("qty_ea", "sum"), orders=("order_id", "nunique"))
        .reset_index()
        .sort_values(["warehouse", "customer_id", "product_id", "create_date"])
    )

    items_store = items.copy()
    items_store["store"] = items_store["store"].fillna("__missing_store__").astype(str)

    wh_store_prod_day = (
        items_store.groupby(
            [
                "warehouse",
                "customer_id",
                "store",
                "product_id",
                "product_name",
                "temperature_zone",
                "create_date",
            ],
            dropna=False,
        )
        .agg(qty_ea=("qty_ea", "sum"), orders=("order_id", "nunique"))
        .reset_index()
        .sort_values(["warehouse", "store", "product_id", "create_date"])
    )

    wh_summary = (
        items.groupby("warehouse", dropna=False)
        .agg(
            total_orders=("order_id", "nunique"),
            total_lines=("order_id", "count"),
            total_qty_ea=("qty_ea", "sum"),
            unique_customers=("customer_id", "nunique"),
            unique_products=("product_id", "nunique"),
            first_day=("create_date", "min"),
            last_day=("create_date", "max"),
        )
        .reset_index()
        .sort_values("total_qty_ea", ascending=False)
    )
    wh_summary["share_qty_pct"] = (
        100 * wh_summary["total_qty_ea"] / wh_summary["total_qty_ea"].sum()
    ).round(2)

    prod_totals = (
        items.groupby(["warehouse", "product_id", "product_name"], dropna=False)["qty_ea"]
        .sum()
        .reset_index()
    )
    prod_totals = prod_totals.sort_values(
        ["warehouse", "qty_ea"], ascending=[True, False]
    )
    prod_totals["rank_in_warehouse"] = prod_totals.groupby("warehouse").cumcount() + 1
    top_products = prod_totals[prod_totals["rank_in_warehouse"] <= 10].copy()

    paths = [
        write(wh_day, "mart_warehouse_day.csv"),
        write(wh_cust_day, "mart_warehouse_customer_day.csv"),
        write(wh_prod_day, "mart_warehouse_product_day.csv"),
        write(wh_cust_prod_day, "mart_warehouse_customer_product_day.csv"),
        write(wh_store_prod_day, "mart_warehouse_store_product_day.csv"),
        write(wh_summary, "mart_warehouse_summary.csv"),
        write(top_products, "mart_top_products_per_warehouse.csv"),
    ]
    for p in paths:
        print(f"Wrote: {p}")

    print("\n=== Warehouse summary (top 10 by qty_ea) ===")
    print(wh_summary.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
