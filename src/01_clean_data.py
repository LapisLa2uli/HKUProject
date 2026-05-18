"""Step 1: Load, clean and normalize the raw monthly Excel files.

Reads ``Data/1.xlsx`` (Jan) and ``Data/2.xlsx`` (Feb + Mar) and writes canonical
CSV tables to ``processed/`` covering orders, order lines, and logistics events.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
OUT_DIR = PROJECT_ROOT / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_FILES = [
    ("2026-01", DATA_DIR / "1.xlsx"),
    ("2026-02-03", DATA_DIR / "2.xlsx"),
]

ORDER_RENAME = {
    "订单单号": "order_id",
    "订单类型": "order_type",
    "货主编码": "customer_id",
    "货主": "customer_name",
    "仓库": "warehouse",
    "收货门店": "store",
    "省市区": "province_city_district",
    "求和项:预计发货数量EA": "order_qty_ea",
    "预计总箱数": "order_total_boxes",
    "创建人": "creator",
    "创建时间": "create_time_raw",
}

ITEM_BASE_RENAME = {
    "订单单号": "order_id",
    "商品编码": "product_id",
    "商品名称": "product_name",
    "温区": "temperature_zone",
}

LOG_RENAME = {
    "订单号": "order_id",
    "操作时间": "event_time",
    "操作记录": "event_record",
    "操作人": "operator",
}


def parse_datetime(series: pd.Series) -> pd.Series:
    """Parse a column of mixed datetime strings / Excel serials into datetime."""
    parsed = pd.to_datetime(series, errors="coerce")
    missing = parsed.isna()
    if missing.any():
        as_num = pd.to_numeric(series[missing], errors="coerce")
        parsed.loc[missing] = pd.to_datetime(
            as_num, origin="1899-12-30", unit="D", errors="coerce"
        )
    return parsed


def read_order_sheet(path: Path, source_label: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="订单表", dtype=str)
    df = df.rename(columns=ORDER_RENAME)
    df["source_file"] = path.name
    df["source_label"] = source_label

    df["create_time"] = parse_datetime(df["create_time_raw"])
    df["create_date"] = df["create_time"].dt.date
    df["create_month"] = df["create_time"].dt.to_period("M").astype(str)

    df["order_qty_ea"] = pd.to_numeric(df["order_qty_ea"], errors="coerce")
    df["order_total_boxes"] = pd.to_numeric(df["order_total_boxes"], errors="coerce")

    parts = df["province_city_district"].fillna("").str.split("-", n=2, expand=True)
    df["province"] = parts[0].replace("", np.nan)
    df["city"] = parts[1].replace("", np.nan) if parts.shape[1] > 1 else np.nan
    df["district"] = parts[2].replace("", np.nan) if parts.shape[1] > 2 else np.nan

    return df.drop(columns=["create_time_raw"])


def read_item_sheet(path: Path, source_label: str) -> pd.DataFrame:
    """Read ``订单明细`` and resolve the duplicated ``单位`` / quantity column variants."""
    df = pd.read_excel(path, sheet_name="订单明细", dtype=str, header=None)

    header_row = df.iloc[0].tolist()
    df = df.iloc[1:].reset_index(drop=True)

    new_names: list[str] = []
    qty_seen = 0
    uom_seen = 0
    for raw in header_row:
        name = str(raw).strip() if raw is not None else ""
        if name in ITEM_BASE_RENAME:
            new_names.append(ITEM_BASE_RENAME[name])
        elif "单位" in name:
            new_names.append("uom" if uom_seen == 0 else "uom_ea")
            uom_seen += 1
        elif "预计发货数量" in name:
            if "EA" in name or "ea" in name:
                new_names.append("qty_ea")
            else:
                new_names.append("qty" if qty_seen == 0 else f"qty_extra_{qty_seen}")
                qty_seen += 1
        else:
            new_names.append(name or f"col_{len(new_names)}")
    df.columns = new_names

    if "qty_ea" not in df.columns and "qty" in df.columns:
        df["qty_ea"] = df["qty"]
    if "uom_ea" not in df.columns:
        df["uom_ea"] = df.get("uom")

    df["qty"] = pd.to_numeric(df.get("qty"), errors="coerce")
    df["qty_ea"] = pd.to_numeric(df["qty_ea"], errors="coerce")

    df["source_file"] = path.name
    df["source_label"] = source_label
    return df


def read_log_sheet(path: Path, source_label: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="物流信息", dtype=str)
    df = df.rename(columns=LOG_RENAME)
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df["source_file"] = path.name
    df["source_label"] = source_label
    return df


def main() -> int:
    orders_all: list[pd.DataFrame] = []
    items_all: list[pd.DataFrame] = []
    logs_all: list[pd.DataFrame] = []

    for label, path in SOURCE_FILES:
        if not path.exists():
            print(f"[WARN] missing source: {path}")
            continue
        print(f"[load] {path.name} ({label})")
        orders_all.append(read_order_sheet(path, label))
        items_all.append(read_item_sheet(path, label))
        logs_all.append(read_log_sheet(path, label))

    orders = pd.concat(orders_all, ignore_index=True)
    items = pd.concat(items_all, ignore_index=True)
    logs = pd.concat(logs_all, ignore_index=True)

    before_orders = len(orders)
    orders = orders.drop_duplicates(subset=["order_id"])
    items = items.drop_duplicates()
    logs = logs.drop_duplicates()

    item_orders = set(items["order_id"].dropna())
    head_orders = set(orders["order_id"].dropna())
    orphan_items = len(item_orders - head_orders)
    orphan_logs = len(set(logs["order_id"].dropna()) - head_orders)

    orders_lookup = orders[[
        "order_id",
        "customer_id",
        "customer_name",
        "warehouse",
        "store",
        "province",
        "city",
        "district",
        "create_time",
        "create_date",
        "create_month",
    ]]
    items_enriched = items.merge(orders_lookup, on="order_id", how="left")

    orders.to_csv(OUT_DIR / "orders.csv", index=False, encoding="utf-8-sig")
    items_enriched.to_csv(OUT_DIR / "order_items.csv", index=False, encoding="utf-8-sig")
    logs.to_csv(OUT_DIR / "logistics_events.csv", index=False, encoding="utf-8-sig")

    summary = {
        "orders_rows_input": before_orders,
        "orders_rows_dedup": len(orders),
        "items_rows": len(items_enriched),
        "logs_rows": len(logs),
        "unique_customers": orders["customer_id"].nunique(),
        "unique_warehouses": orders["warehouse"].nunique(),
        "unique_products": items["product_id"].nunique(),
        "orphan_item_orders": orphan_items,
        "orphan_log_orders": orphan_logs,
        "order_create_min": str(orders["create_time"].min()),
        "order_create_max": str(orders["create_time"].max()),
        "month_distribution": orders.groupby("create_month")["order_id"].count().to_dict(),
    }
    summary_df = pd.DataFrame(list(summary.items()), columns=["metric", "value"])
    summary_df.to_csv(OUT_DIR / "clean_summary.csv", index=False, encoding="utf-8-sig")

    print("\n=== Clean summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nWrote: {OUT_DIR/'orders.csv'}")
    print(f"Wrote: {OUT_DIR/'order_items.csv'}")
    print(f"Wrote: {OUT_DIR/'logistics_events.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
