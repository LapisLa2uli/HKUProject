"""Move flat ``reports/*.csv`` into grouped subfolders (idempotent)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
sys.path.insert(0, str(ROOT / "src"))

from _report_paths import (  # noqa: E402
    BASELINE_DIR,
    COMPARISON_DIR,
    HEATMAPS_DIR,
    HURDLE_DIR,
    STORE_REG_DIR,
    ensure_report_dirs,
)


def target_dir(name: str) -> Path | None:
    if name.startswith("baseline_"):
        return BASELINE_DIR
    if name in (
        "april_forecast_daily.csv",
        "april_forecast_monthly_by_warehouse.csv",
        "april_forecast_monthly_by_customer_product.csv",
        "april_forecast_monthly_by_warehouse_customer_product.csv",
    ):
        return BASELINE_DIR
    if name.startswith("hurdle_") or name.startswith("april_forecast_hurdle"):
        return HURDLE_DIR
    if name.startswith("store_regression_") or name.startswith("april_forecast_store_regression"):
        return STORE_REG_DIR
    if name.startswith("model_comparison"):
        return COMPARISON_DIR
    if "heatmap" in name and name.endswith(".csv"):
        return HEATMAPS_DIR
    return None


def main() -> int:
    ensure_report_dirs()
    moved = 0
    for src in sorted(REPORTS.glob("*.csv")):
        dest_dir = target_dir(src.name)
        if dest_dir is None:
            print(f"[skip] {src.name}")
            continue
        dest = dest_dir / src.name
        if dest.exists():
            src.unlink()
            print(f"[exists] {dest.relative_to(ROOT)}")
            continue
        shutil.move(str(src), str(dest))
        print(f"[move] {src.name} -> {dest.relative_to(ROOT)}")
        moved += 1
    print(f"Done. Moved {moved} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
