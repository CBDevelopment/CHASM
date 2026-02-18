from pathlib import Path
from datetime import datetime

BASE_DIR = Path("CHASM_data")
DRAWINGS_DIR = BASE_DIR / "swpc_synoptic_drawings" / "2022"
SELECTIONS_DIR = BASE_DIR / "chasm_selections" / "2022"


def extract_date_from_drawing(filename: str) -> str:
    """Extract date from e.g. boul_neutl_fd_20220101_0630.jpg -> 2022-01-01"""
    stem = Path(filename).stem  # boul_neutl_fd_20220101_0630
    date_part = stem.split("_")[-2]  # 20220101
    return f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"


def extract_date_from_selection(filename: str) -> str:
    """Extract date from e.g. DATA-boul_neutl_fd_20170101_0330.npz -> 2017-01-01"""
    stem = Path(filename).stem  # DATA-boul_neutl_fd_20170101_0330
    date_part = stem.split("_")[-2]  # 20170101
    return f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"


def get_dates_from_dir(directory: Path, extract_fn) -> dict[str, Path]:
    """Returns a dict of date_str -> file path"""
    dates = {}
    for f in sorted(directory.iterdir()):
        if f.is_file():
            try:
                date = extract_fn(f.name)
                dates[date] = f
            except (IndexError, ValueError) as e:
                print(f"  [WARN] Could not parse date from: {f.name} ({e})")
    return dates


print(f"Checking:\n  Drawings:   {DRAWINGS_DIR}\n  Selections: {SELECTIONS_DIR}\n")

drawing_dates = get_dates_from_dir(DRAWINGS_DIR, extract_date_from_drawing)
selection_dates = get_dates_from_dir(SELECTIONS_DIR, extract_date_from_selection)

print(f"Drawings found:   {len(drawing_dates)} unique dates")
print(f"Selections found: {len(selection_dates)} unique dates\n")

in_drawings_not_selections = sorted(set(drawing_dates) - set(selection_dates))
in_selections_not_drawings = sorted(set(selection_dates) - set(drawing_dates))

print(f"Dates in DRAWINGS but not in SELECTIONS ({len(in_drawings_not_selections)}):")
for d in in_drawings_not_selections:
    print(f"  {d}  <-  {drawing_dates[d].name}")

print(f"\nDates in SELECTIONS but not in DRAWINGS ({len(in_selections_not_drawings)}):")
for d in in_selections_not_drawings:
    print(f"  {d}  <-  {selection_dates[d].name}")
