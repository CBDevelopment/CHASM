from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


BASE = Path("D:/projects/research/CHASM/download_data/aia_wavelengths")
CSV_PATH = BASE / "rsun_obs_by_wavelength.csv"
OUTPUT = BASE / "rsun_obs_193_by_year.png"
TARGET_WL = "193"


def _load_rows(csv_path: Path) -> List[dict]:
    rows: List[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main() -> int:
    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}")
        return 1

    rows = _load_rows(CSV_PATH)
    filtered = []
    for row in rows:
        if row.get("wavelength") != TARGET_WL:
            continue
        try:
            dt = datetime.fromisoformat(row["date"])
            rsun_obs = float(row["rsun_obs"])
        except Exception:
            continue
        filtered.append((dt, rsun_obs))

    if not filtered:
        print(f"No rows for wavelength {TARGET_WL}A")
        return 1

    # Group by year and month, average values per month
    by_year: Dict[int, Dict[int, List[float]]] = {}
    for dt, rsun_obs in filtered:
        by_year.setdefault(dt.year, {}).setdefault(dt.month, []).append(rsun_obs)

    years = sorted(by_year.keys())
    n_years = len(years)
    ncols = 2 if n_years > 1 else 1
    nrows = (n_years + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows), sharey=True
    )
    if n_years == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    month_labels = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    month_ticks = list(range(1, 13))

    for ax, year in zip(axes, years):
        monthly = by_year[year]
        xs = []
        ys = []
        for m in month_ticks:
            vals = monthly.get(m, [])
            if not vals:
                continue
            xs.append(m)
            ys.append(sum(vals) / len(vals))

        ax.plot(xs, ys, marker="o", linewidth=1.5)
        ax.set_title(str(year))
        ax.set_xticks(month_ticks)
        ax.set_xticklabels(month_labels, rotation=0)
        ax.set_xlabel("Month")
        ax.set_ylabel("RSUN_OBS (arcsec)")
        ax.grid(True, alpha=0.2)

    # Hide unused axes if any
    for ax in axes[n_years:]:
        ax.axis("off")

    fig.suptitle("RSUN_OBS for 193A by Year")
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150)
    print(f"Saved plot to: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
