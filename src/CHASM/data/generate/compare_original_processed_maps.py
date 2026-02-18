"""Compare two SDO wavelengths for the same date side by side.

Directory layout expected:
    <base>/{year}/{wavelength}/{date}.fits

Usage:
    python compare_original_processed_maps.py [date] [wl1] [wl2]

    date : YYYY-MM-DD  (default: first available date)
    wl1  : wavelength  (default: 193)
    wl2  : wavelength  (default: 171)

Examples:
    python compare_original_processed_maps.py 2017-01-01 193 171
    python compare_original_processed_maps.py 2017-01-01
    python compare_original_processed_maps.py
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import (
    AsinhStretch,
    ImageNormalize,
    PercentileInterval,
    LinearStretch,
)
from sunpy.map import Map as SunpyMap
from sunpy.visualization.colormaps import cm

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path("D:/projects/research/CHASM/CHASM_data/sdo_imagery")

# ── SDO display settings ───────────────────────────────────────────────────────
_WLS = [94, 131, 171, 193, 211, 304, 335, 6173]

_NORMS = [
    ImageNormalize(vmin=0, vmax=445.5, stretch=AsinhStretch(0.005), clip=True),
    ImageNormalize(vmin=0, vmax=981.3, stretch=AsinhStretch(0.005), clip=True),
    ImageNormalize(vmin=0, vmax=6457.5, stretch=AsinhStretch(0.005), clip=True),
    ImageNormalize(vmin=0, vmax=7757.31, stretch=AsinhStretch(0.005), clip=True),
    ImageNormalize(vmin=0, vmax=6539.8, stretch=AsinhStretch(0.005), clip=True),
    ImageNormalize(vmin=0, vmax=3756.0, stretch=AsinhStretch(0.005), clip=True),
    ImageNormalize(vmin=0, vmax=915.0, stretch=AsinhStretch(0.005), clip=True),
    ImageNormalize(vmin=-100, vmax=100, stretch=LinearStretch(), clip=True),
]

_CMAPS = [
    cm.sdoaia94,
    cm.sdoaia131,
    cm.sdoaia171,
    cm.sdoaia193,
    cm.sdoaia211,
    cm.sdoaia304,
    cm.sdoaia335,
    "gray",
]

NORMS = dict(zip(_WLS, _NORMS))
CMAPS = dict(zip(_WLS, _CMAPS))


# ── Helpers ────────────────────────────────────────────────────────────────────


def load_fits(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        hdul.verify("silentfix")
        for hdu in hdul:
            if getattr(hdu, "data", None) is not None:
                return hdu.data.astype(float), hdu.header
    raise RuntimeError(f"No image HDU found in {path}")


def header_info(header: fits.Header) -> dict:
    """Extract pertinent display fields from a FITS header."""

    def _get(*keys, default="—"):
        for k in keys:
            v = header.get(k)
            if v is not None:
                return v
        return default

    return {
        "DATE-OBS": _get("DATE-OBS", "DATE_OBS", "DATE__OBS"),
        "TELESCOP": _get("TELESCOP"),
        "INSTRUME": _get("INSTRUME"),
        "WAVELNTH": _get("WAVELNTH"),
        "EXPTIME": _get("EXPTIME"),
        "RSUN_OBS": _get("RSUN_OBS", "rsun_obs"),
        "CDELT1": _get("CDELT1", "cdelt1"),
        "CDELT2": _get("CDELT2", "cdelt2"),
        "CRPIX1": _get("CRPIX1", "crpix1"),
        "CRPIX2": _get("CRPIX2", "crpix2"),
    }


def build_sunpy_map(data: np.ndarray, header: fits.Header) -> SunpyMap:
    meta = {}
    for card in header.cards:
        try:
            meta[card.keyword] = card.value
        except Exception:
            pass
    meta.setdefault("CUNIT1", "arcsec")
    meta.setdefault("CUNIT2", "arcsec")
    meta.setdefault("CTYPE1", "HPLN-TAN")
    meta.setdefault("CTYPE2", "HPLT-TAN")
    return SunpyMap(data, meta)


def list_available(directory: Path) -> None:
    year_dirs = sorted(p for p in directory.iterdir() if p.is_dir())
    if not year_dirs:
        print(f"  (empty: {directory})")
        return
    for year_dir in year_dirs:
        wl_dirs = sorted(p for p in year_dir.iterdir() if p.is_dir())
        for wl_dir in wl_dirs:
            dates = sorted(p.stem for p in wl_dir.glob("*.fits"))
            if dates:
                print(
                    f"  {year_dir.name}/{wl_dir.name}: {len(dates)} files  "
                    f"[{dates[0]} … {dates[-1]}]"
                )


def first_available_date(wl: str) -> tuple[str, str] | tuple[None, None]:
    """Return (date, year) of the first FITS file found for the given wavelength."""
    for year_dir in sorted(p for p in BASE.iterdir() if p.is_dir()):
        for p in sorted((year_dir / wl).glob("*.fits")):
            return p.stem, year_dir.name
    return None, None


# ── Main ───────────────────────────────────────────────────────────────────────


def main(date: str | None = None, wl1: str = "193", wl2: str = "171") -> None:
    if not BASE.exists():
        print(f"Base directory not found: {BASE}")
        return

    # Auto-pick date using wl1 as reference
    if date is None:
        date, year = first_available_date(wl1)
        if date is None:
            print(f"No FITS files found for wavelength {wl1} under {BASE}")
            print("\nAvailable:")
            list_available(BASE)
            return
        print(f"Auto-selected date: {date}")
    else:
        year = date.split("-")[0]

    paths = {
        wl1: BASE / year / wl1 / f"{date}.fits",
        wl2: BASE / year / wl2 / f"{date}.fits",
    }

    missing = [(wl, p) for wl, p in paths.items() if not p.exists()]
    if missing:
        for wl, p in missing:
            print(f"File not found for wavelength {wl}: {p}")
        print("\nAvailable:")
        list_available(BASE)
        return

    # Load both
    panels = {}
    for wl, path in paths.items():
        data, hdr = load_fits(path)
        panels[wl] = (data, hdr, header_info(hdr), path)

    # ── Print summary ──────────────────────────────────────────────────────────
    sep = "=" * 72
    print(sep)
    for wl, (data, hdr, info, path) in panels.items():
        print(f"WAVELENGTH {wl} Å")
        print(f"  Path      : {path}")
        print(f"  Shape     : {data.shape[1]} × {data.shape[0]} px")
        for k, v in info.items():
            if v != "—":
                print(f"  {k:<10}: {v}")
        finite = data[np.isfinite(data)]
        print(
            f"  Data      : min={finite.min():.2f}  max={finite.max():.2f}"
            f"  mean={finite.mean():.2f}  std={finite.std():.2f}"
        )
        print()
    print(sep)

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(f"{date}", fontsize=14, fontweight="bold")

    for ax, (wl, (data, hdr, info, path)) in zip(axes, panels.items()):
        wl_int = int(wl)
        norm = NORMS.get(wl_int) or ImageNormalize(
            interval=PercentileInterval(99.5), stretch=AsinhStretch(0.005), clip=True
        )
        cmap = CMAPS.get(wl_int, "gray")

        try:
            smap = build_sunpy_map(data, hdr)
            smap.plot_settings["cmap"] = cmap
            smap.plot_settings["norm"] = norm
            smap.plot(axes=ax)
        except Exception:
            ax.imshow(data, cmap=cmap, norm=norm, origin="lower")

        ax.set_title(
            f"{wl_int} Å   {data.shape[1]}×{data.shape[0]} px\n"
            f"CDELT: ({info['CDELT1']}, {info['CDELT2']})   RSUN_OBS: {info['RSUN_OBS']}\n"
            f"DATE-OBS: {info['DATE-OBS']}",
            fontsize=9,
            loc="left",
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    _date = sys.argv[1] if len(sys.argv) > 1 else None
    _wl1 = sys.argv[2] if len(sys.argv) > 2 else "193"
    _wl2 = sys.argv[3] if len(sys.argv) > 3 else "171"
    main(date=_date, wl1=_wl1, wl2=_wl2)
