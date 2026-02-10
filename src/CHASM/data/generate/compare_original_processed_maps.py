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
from astropy.wcs import WCS
from sunpy.visualization.colormaps import cm


BASE = Path("D:/projects/research/CHASM/download_data/aia_imagery")

# SDO AIA normalization and colormap settings
sdo_norms = [
    ImageNormalize(vmin=0, vmax=445.5, stretch=AsinhStretch(0.005), clip=True),  # 94
    ImageNormalize(vmin=0, vmax=981.3, stretch=AsinhStretch(0.005), clip=True),  # 131
    ImageNormalize(vmin=0, vmax=6457.5, stretch=AsinhStretch(0.005), clip=True),  # 171
    ImageNormalize(vmin=0, vmax=7757.31, stretch=AsinhStretch(0.005), clip=True),  # 193
    ImageNormalize(vmin=0, vmax=6539.8, stretch=AsinhStretch(0.005), clip=True),  # 211
    ImageNormalize(vmin=0, vmax=3756, stretch=AsinhStretch(0.005), clip=True),  # 304
    ImageNormalize(vmin=0, vmax=915, stretch=AsinhStretch(0.005), clip=True),  # 335
    ImageNormalize(vmin=-100, vmax=100, stretch=LinearStretch(), clip=True),  # mag
]

sdo_norms_dict = {
    k: v for k, v in zip([94, 131, 171, 193, 211, 304, 335, 6173], sdo_norms)
}

sdo_cmaps = [
    cm.sdoaia94,
    cm.sdoaia131,
    cm.sdoaia171,
    cm.sdoaia193,
    cm.sdoaia211,
    cm.sdoaia304,
    cm.sdoaia335,
    "gray",
]

sdo_cmaps_dict = {
    k: v for k, v in zip([94, 131, 171, 193, 211, 304, 335, 6173], sdo_cmaps)
}


def load_fits(path: Path):
    hdul = fits.open(path)
    # find first HDU with data
    for h in hdul:
        if getattr(h, "data", None) is not None:
            data = h.data.astype(float)
            header = h.header
            break
    else:
        hdul.close()
        raise RuntimeError(f"No image HDU in {path}")
    hdul.close()
    return data, header


def print_header_snippet(name, header, keys=5):
    print(f"{name}: shape header entries (first {keys}):")
    cnt = 0
    for card in header.cards:
        if cnt >= keys:
            break
        key = getattr(card, "keyword", None)
        try:
            val = card.value
            print(f"  {key}: {val}")
        except Exception:
            # fall back to raw card string if value can't be parsed
            print(f"  {key}: <unparsable>  raw={str(card)}")
        cnt += 1


def main(date="2017-01-01", wavelength=None, vmin=None, vmax=None):
    year = date.split("-")[0]
    full_dir = BASE / f"{year}_FullSize"
    if wavelength is None:
        wls = [p.name for p in full_dir.iterdir() if p.is_dir()]
        if not wls:
            print("No wavelength folders found in", full_dir)
            return
        wavelength = wls[0]

    full_path = full_dir / str(wavelength) / f"{date}.fits"
    res_path = BASE / str(year) / str(wavelength) / f"{date}.fits"

    if not full_path.exists():
        print("Full-size not found:", full_path)
        return
    if not res_path.exists():
        print("Resampled not found:", res_path)
        return

    full, hdr_full = load_fits(full_path)
    res, hdr_res = load_fits(res_path)

    print(f"Full: {full_path} -> shape={full.shape}")
    print_header_snippet("Full header", hdr_full)
    print("")
    print(f"Resampled: {res_path} -> shape={res.shape}")
    print_header_snippet("Resampled header", hdr_res)

    # Determine wavelength value from input or headers
    wl_val = None
    try:
        wl_val = int(wavelength) if wavelength is not None else None
    except Exception:
        wl_val = None
    if wl_val is None:
        try:
            wl_val = int(
                hdr_full.get(
                    "WAVELNTH", hdr_full.get("WAVELN", hdr_full.get("WAVE", 0))
                )
            )
        except Exception:
            wl_val = None

    # Get appropriate colormap and normalization from dictionaries
    if wl_val in sdo_cmaps_dict:
        cmap_full = cmap_res = sdo_cmaps_dict[wl_val]
        norm_full = norm_res = sdo_norms_dict[wl_val]
        print(f"Using SDO standard for wavelength {wl_val}")
    else:
        # Fallback for unknown wavelengths
        cmap_full = cmap_res = "gray"
        if vmin is None:
            vmin = 100
        if vmax is None:
            vmax = 5000
        norm_full = norm_res = ImageNormalize(
            vmin=vmin, vmax=vmax, stretch=AsinhStretch()
        )
        print(f"Using fallback normalization for unknown wavelength: {wl_val}")

    def cmap_name(c):
        if isinstance(c, str):
            return c
        try:
            return getattr(c, "name", str(c))
        except Exception:
            return str(c)

    print(f"Using colormap: {cmap_name(cmap_full)}")
    print(f"Using normalization: vmin={norm_full.vmin}, vmax={norm_full.vmax}")

    # Display side-by-side using SunPy Map for proper WCS projection
    is_mag = wl_val == 6173
    if wl_val in sdo_norms_dict:
        # Use SunPy Map plotting for correct WCS-aware display
        # Build SunPy maps from loaded data+header with minimal sanitization
        def _ensure_meta(hdr):
            # Preserve all existing metadata, only add missing coordinate units
            # Handle unparsable cards by iterating carefully
            new_hdr = {}
            for card in hdr.cards:
                try:
                    key = card.keyword
                    val = card.value
                    new_hdr[key] = val
                except Exception:
                    # Skip unparsable cards
                    pass
            new_hdr.setdefault("CUNIT1", "arcsec")
            new_hdr.setdefault("CUNIT2", "arcsec")
            new_hdr.setdefault("CTYPE1", "HPLN-TAN")
            new_hdr.setdefault("CTYPE2", "HPLT-TAN")
            return new_hdr

        m_full = SunpyMap(full, _ensure_meta(hdr_full))
        m_res = SunpyMap(res, _ensure_meta(hdr_res))

        m_full.plot_settings["cmap"] = cmap_full
        m_full.plot_settings["norm"] = norm_full
        m_res.plot_settings["cmap"] = cmap_res
        m_res.plot_settings["norm"] = norm_res

        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(1, 2, 1, projection=m_full.wcs)
        m_full.plot(axes=ax1)
        ax1.set_title(f"Full: {full.shape} - {wl_val}Å")

        ax2 = fig.add_subplot(1, 2, 2, projection=m_res.wcs)
        m_res.plot(axes=ax2)
        ax2.set_title(f"Resampled: {res.shape} - {wl_val}Å")

        plt.tight_layout()
        plt.show()
    else:
        # Fallback to simple matplotlib display for unknown wavelengths
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(full, cmap=cmap_full, origin="lower", norm=norm_full)
        axes[0].set_title(f"Full: {full.shape}")
        axes[1].imshow(res, cmap=cmap_res, origin="lower", norm=norm_res)
        axes[1].set_title(f"Resampled: {res.shape}")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    date = sys.argv[1] if len(sys.argv) > 1 else "2017-01-01"
    wl = sys.argv[2] if len(sys.argv) > 2 else None
    vmin = float(sys.argv[3]) if len(sys.argv) > 3 else None
    vmax = float(sys.argv[4]) if len(sys.argv) > 4 else None
    main(date=date, wavelength=wl, vmin=vmin, vmax=vmax)
