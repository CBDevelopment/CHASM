from pathlib import Path
from datetime import date, timedelta

BASE_DIR = (
    Path(__file__).parent.parent / "CHASM_data" / "sdo_imagery" / "full_size_images"
)

YEAR = 2022
WAVELENGTH = "193"
TARGET_DIR = BASE_DIR / str(YEAR) / WAVELENGTH

# Build full set of expected dates for the year
start = date(YEAR, 1, 1)
end = date(YEAR, 12, 31)
all_dates = set()
d = start
while d <= end:
    all_dates.add(d.strftime("%Y-%m-%d"))
    d += timedelta(days=1)

# Collect dates present on disk
present_dates = set()
missing_stems = []
for f in TARGET_DIR.iterdir():
    if f.is_file():
        present_dates.add(f.stem)  # e.g. "2022-01-01"

missing = sorted(all_dates - present_dates)
extra = sorted(present_dates - all_dates)

print(f"Directory: {TARGET_DIR}")
print(f"Expected dates : {len(all_dates)}")
print(f"Present on disk: {len(present_dates)}")
print(f"Missing ({len(missing)}):")
for d in missing:
    print(f"  {d}")

if extra:
    print(f"\nUnexpected files ({len(extra)}):")
    for d in extra:
        print(f"  {d}")
