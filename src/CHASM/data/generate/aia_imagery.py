# Standard library
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

# Third-party - data & astronomy
import astropy.units as u
import drms
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.fits import Header, HDUList, PrimaryHDU
from astropy.io.fits.hdu.compressed import CompImageHDU
from sunpy.map import Map
import sunpy.map

# Module constants
DEFAULT_RESOLUTION = 512
DEFAULT_PADDING_FACTOR = 0.1
FITS_COMPRESSION_TYPE = "HCOMPRESS_1"
FITS_QUANTIZE_LEVEL = 16.0
INVALID_FITS_VALUES = {".", "", "MISSING", "-nan", "nan"}


def _is_valid_fits_value(key: Any, value: Any) -> bool:
    """Check if a key-value pair is valid for FITS headers."""
    if not key or not str(key).strip():
        return False
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str) and value.strip() in INVALID_FITS_VALUES:
        return False
    return True


def _normalize_jsoc_timestamp(timestamp_str: str) -> str:
    """Convert JSOC timestamp formats to ISO-compatible strings.

    Examples: '2017.01.01_03:36:00_TAI' -> '2017-01-01T03:36:00'
    """
    ts = str(timestamp_str)
    # Replace dots with dashes in date: 2017.01.01 -> 2017-01-01
    ts = ts.replace(".", "-", 2)  # Only first 2 dots
    # Replace underscore between date and time with T
    ts = ts.replace("_", "T", 1)
    # Remove timezone suffix
    ts = ts.split("_")[0]
    return ts


def _find_image_hdu(hdul: HDUList):
    """Find the first HDU with image data."""
    for hdu in hdul:
        if getattr(hdu, "data", None) is not None:
            return hdu
    return None


def _sanitize_header(header):
    """Return a sanitized header dict suitable for SunPy Map creation.

    Preserves all critical WCS/coordinate metadata from the original FITS header,
    while adding sensible defaults for any missing required keys.
    """
    if header is None:
        return {}

    # Filter valid values only using helper
    raw_header = dict(header)
    hdr = {k: v for k, v in raw_header.items() if _is_valid_fits_value(k, v)}

    # Normalize DATE_OBS variants
    if "DATE__OBS" in hdr and "DATE_OBS" not in hdr:
        hdr["DATE_OBS"] = hdr["DATE__OBS"]
    if "DATE-OBS" in hdr and "DATE_OBS" not in hdr:
        hdr["DATE_OBS"] = hdr["DATE-OBS"]
    if "DATE_OBS" in hdr and "DATE-OBS" not in hdr:
        hdr["DATE-OBS"] = hdr["DATE_OBS"]

    # Preserve critical WCS keys from original header (try lowercase variants)
    for key in ("CDELT1", "CDELT2", "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2"):
        if key not in hdr and key.lower() in header:
            hdr[key] = header[key.lower()]
        # Create lowercase variants for compatibility
        if key in hdr:
            hdr[key.lower()] = hdr[key]

    # Normalize RSUN_OBS
    if "RSUN_OBS" not in hdr and "rsun_obs" in hdr:
        hdr["RSUN_OBS"] = hdr["rsun_obs"]
    if "rsun_obs" not in hdr and "RSUN_OBS" in hdr:
        hdr["rsun_obs"] = hdr["RSUN_OBS"]

    # Normalize DSUN_OBS
    if "DSUN_OBS" not in hdr and "dsun_obs" in hdr:
        hdr["DSUN_OBS"] = hdr["dsun_obs"]
    if "dsun_obs" not in hdr and "DSUN_OBS" in hdr:
        hdr["dsun_obs"] = hdr["DSUN_OBS"]

    return hdr


class JSOCQuery:
    """Query builder for JSOC data retrieval."""

    # Series constants
    AIA_SERIES = "aia.lev1_euv_12s"
    HMI_SERIES = "hmi.M_720s"
    VALID_SERIES = {AIA_SERIES, HMI_SERIES}

    # Valid AIA wavelengths and HMI magnetogram
    VALID_WAVELENGTHS = {94, 131, 171, 193, 211, 304, 335, 6173}

    def __init__(
        self,
        series: str,
        datetime_iso: datetime,
        wavelength: int | None = None,
        segment: str | None = None,
    ):
        if series not in self.VALID_SERIES:
            raise ValueError(f"Unsupported series: {series}")
        if not isinstance(datetime_iso, datetime):
            raise ValueError("datetime_iso must be a datetime object")

        self.series = series
        self.datetime_iso = datetime_iso
        self.segment = segment

        # For AIA series a wavelength is required; for HMI series wavelength is ignored
        if self.series == self.AIA_SERIES:
            if wavelength is None:
                raise ValueError("Wavelength must be provided for AIA series")
            if wavelength not in self.VALID_WAVELENGTHS:
                raise ValueError(f"Unsupported wavelength: {wavelength}")
            self.wavelength = wavelength
        else:
            # HMI: no wavelength field in JSOC dataset selector
            self.wavelength = None

    def to_query_string(self) -> str:
        time_str = self.datetime_iso.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
        if self.series.startswith("hmi."):
            base = f"{self.series}[{time_str}]"
        else:
            base = f"{self.series}[{time_str}][{self.wavelength}]"
        if self.segment:
            return base + "{" + self.segment + "}"
        return base


class AIAImageDownloader:
    """Downloader for AIA and HMI images from JSOC."""

    # These are the 8 wavelengths utilized by the CHRONNOS model (6173 is the LoS magnetogram)
    DEFAULT_WAVELENGTHS = ["94", "131", "171", "193", "211", "304", "335", "6173"]

    # WCS metadata keys to fetch from JSOC
    WCS_KEYS = [
        "CDELT1",
        "CDELT2",
        "CRPIX1",
        "CRPIX2",
        "CRVAL1",
        "CRVAL2",
        "CTYPE1",
        "CTYPE2",
        "CUNIT1",
        "CUNIT2",
        "RSUN_OBS",
        "DSUN_OBS",
        "DATE__OBS",
        "DATE-OBS",
        "T_OBS",
        "TELESCOP",
        "INSTRUME",
    ]

    def __init__(self, save_path: Path, email: str = None):
        self.save_path = save_path
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.drms_client = drms.Client(email=email) if email else drms.Client()
        self.wavelengths = self.DEFAULT_WAVELENGTHS
        self.aia_series = JSOCQuery.AIA_SERIES
        self.hmi_series = JSOCQuery.HMI_SERIES

    def get_jsoc_query(self, datetime_iso: datetime, wavelength: int) -> JSOCQuery:
        series = self.aia_series if wavelength != 6173 else self.hmi_series
        segment = "magnetogram" if series == self.hmi_series else "image"
        wl = wavelength if series != self.hmi_series else None
        return JSOCQuery(
            series=series, datetime_iso=datetime_iso, wavelength=wl, segment=segment
        )

    def _query_is_available(self, query: JSOCQuery, threshold_minutes: int = 0) -> bool:
        """Check if JSOC data is available for the query within threshold."""
        try:
            available_df = self.drms_client.query(
                query.to_query_string(), key="T_REC, WAVELNTH"
            )
        except Exception:
            return False

        if available_df is None or available_df.empty:
            return False

        # Choose timestamp column (prefer T_REC)
        tcol = next(
            (col for col in ["T_REC", "DATE__OBS"] if col in available_df.columns),
            available_df.columns[0],
        )

        # Normalize and parse timestamps
        timestamps = available_df[tcol].astype(str).apply(_normalize_jsoc_timestamp)
        parsed = pd.to_datetime(timestamps, errors="coerce").dt.tz_localize(None)

        if parsed.isna().all():
            return False

        # Check if any observation is within threshold
        time_diff = (parsed - query.datetime_iso).abs().dt.total_seconds() / 60
        return (time_diff <= threshold_minutes).any()

    def _paths_for_time_and_wavelength(
        self, dt: datetime, wavelength: int | None
    ) -> tuple[Path, Path]:
        """Return (full_path, resampled_path) for a given datetime and wavelength."""
        year_dir = str(dt.year)
        wl_dir = str(wavelength or 6173)  # Use 6173 for HMI magnetograms
        filename = f"{dt.date().isoformat()}.fits"

        full_dir = self.save_path / f"{year_dir}_FullSize" / wl_dir
        resampled_dir = self.save_path / year_dir / wl_dir
        return full_dir / filename, resampled_dir / filename

    def _fetch_query_metadata(self, query: JSOCQuery) -> dict:
        """Fetch WCS metadata from JSOC for a query."""
        try:
            metadata_df = self.drms_client.query(
                query.to_query_string(), key=",".join(self.WCS_KEYS)
            )
            if metadata_df is None or metadata_df.empty:
                return {}

            # Filter out NaN values
            metadata = metadata_df.iloc[0].to_dict()
            return {k: v for k, v in metadata.items() if not pd.isna(v)}
        except Exception as e:
            self.logger.warning(f"Failed to query metadata: {e}")
            return {}

    def _export_fits_for_query(
        self, query: JSOCQuery, download_dir: Path
    ) -> Path | None:
        """Export and download FITS file from JSOC, injecting WCS metadata."""
        # Fetch WCS metadata first
        metadata = self._fetch_query_metadata(query)

        export_request: drms.ExportRequest = self.drms_client.export(
            query.to_query_string(),
            n=1,
        )

        try:
            df = export_request.download(download_dir)
        except Exception:
            self.logger.exception(
                f"Export/download failed for query {query.to_query_string()}"
            )
            return None

        # Extract actual downloaded path from df['download'] and rename to YYYY-MM-DD.fits
        try:
            if df is None or df.empty:
                self.logger.warning(
                    f"Empty export response for {query.to_query_string()}"
                )
                return None
            if "download" in df.columns:
                dl = df["download"].dropna().astype(str).iloc[0]
                p = Path(dl)
                if not p.is_absolute():
                    p = download_dir / p.name

            # Inject WCS metadata into the FITS header
            if metadata:
                self._inject_wcs_metadata(p, metadata)

            target = download_dir / (query.datetime_iso.date().isoformat() + ".fits")
            if p.resolve() != target.resolve():
                p.rename(target)
            return target
        except Exception:
            self.logger.exception(
                f"Error processing export response for query {query.to_query_string()}"
            )
            return None

    def _inject_wcs_metadata(self, fits_path: Path, metadata: dict) -> None:
        """Inject WCS metadata from JSOC query into a FITS file header."""
        try:
            with fits.open(fits_path, mode="update") as hdul:
                img_hdu = _find_image_hdu(hdul)
                if img_hdu is not None:
                    # Inject valid WCS keys into header
                    for key, value in metadata.items():
                        if _is_valid_fits_value(key, value):
                            try:
                                img_hdu.header[key] = value
                            except Exception:
                                pass
        except Exception as e:
            self.logger.error(f"Failed to inject WCS metadata into {fits_path}: {e}")

    def download_image_for_query(
        self, query: JSOCQuery, threshold_minutes: int = 60
    ) -> Path | None:
        full_path, _resampled_path = self._paths_for_time_and_wavelength(
            query.datetime_iso, query.wavelength
        )
        # If full-size file already exists, just return it
        if full_path.exists():
            return full_path

        # Ensure the query is available and download the full-size FITS
        if not self._query_is_available(query, threshold_minutes):
            self.logger.warning(
                f"Query not available within threshold: {query.to_query_string()}"
            )
            return None

        full_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Downloading: {query.to_query_string()}")
        downloaded_file = self._export_fits_for_query(query, full_path.parent)
        if downloaded_file is None:
            self.logger.error(
                f"Failed to download full image for query: {query.to_query_string()}"
            )
            return None
        return downloaded_file

    def _parallel_process(
        self,
        items: list,
        worker_func: Callable,
        max_workers: int = 4,
        progress_callback: Callable[[int, int], None] = None,
    ) -> list:
        """Generic parallel processing with progress tracking and interrupt handling."""
        results = []
        total = len(items)
        completed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            try:
                future_to_item = {
                    executor.submit(worker_func, item): item for item in items
                }

                for future in as_completed(future_to_item):
                    result = future.result()
                    results.append(result)
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total)

            except KeyboardInterrupt:
                self.logger.warning("Processing interrupted by user (Ctrl+C)")
                print("\n\nInterrupted! Cancelling pending tasks...")
                for future in future_to_item:
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise

        return results

    def download_images_parallel(
        self,
        queries: list[JSOCQuery],
        threshold_minutes: int = 60,
        max_workers: int = 4,
        progress_callback: Callable[[int, int], None] = None,
    ) -> list[tuple[JSOCQuery, Path | None]]:
        """Download multiple images in parallel."""

        def worker(query: JSOCQuery) -> tuple[JSOCQuery, Path | None]:
            try:
                path = self.download_image_for_query(query, threshold_minutes)
                return (query, path)
            except Exception as e:
                self.logger.error(
                    f"Exception downloading {query.to_query_string()}: {e}"
                )
                return (query, None)

        return self._parallel_process(queries, worker, max_workers, progress_callback)


class CHASM_AIADownloader(AIAImageDownloader):
    def __init__(self, save_path: Path, email: str = None):
        super().__init__(save_path, email)

    def _parse_swpc_drawing_timestamp(self, drawing_filename: str) -> datetime:
        base_name = Path(drawing_filename).stem
        timestamp_str = base_name.split("boul_neutl_fd_")[-1]
        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M")
        return timestamp

    def get_dates_from_swpc_drawing_dir(self, swpc_dir: Path) -> list[datetime]:
        drawing_files = list(swpc_dir.glob("boul_neutl_fd_*.jpg"))
        dates = []
        for file in drawing_files:
            dt = self._parse_swpc_drawing_timestamp(file.name)
            dates.append(dt)
        return dates

    def _prepMap(
        self,
        s_map: Map,
        resolution: int,
        padding_factor: float = DEFAULT_PADDING_FACTOR,
    ):
        """Returns the adjusted map. The solar disk is centered to the image,
        the north pole axis aligned with the y-axis and the scale adjusted to (1 + padding) * R_Sun[arcsec] / resolution

        Directly from Jarolim et al.

        :param s_map: '~sunpy.map.Map' object
        :param resolution: pixels along x- and y-axis
        :param padding_factor: distance between solar limb and image border given in R_Sun
        :return: adjusted sunpy map
        """
        warnings.simplefilter("ignore")  # ignore warnings
        r_obs_pix = s_map.rsun_obs / s_map.scale[0]  # normalize solar radius
        r_obs_pix = (1 + padding_factor) * r_obs_pix
        scale_factor = resolution / (2 * r_obs_pix.value)
        s_map = Map(np.nan_to_num(s_map.data).astype(np.float32), s_map.meta)
        s_map = s_map.rotate(recenter=True, scale=scale_factor, missing=0, order=3)
        arcs_frame = (resolution / 2) * s_map.scale[0].value
        s_map = s_map.submap(
            SkyCoord(
                -arcs_frame * u.arcsec,
                -arcs_frame * u.arcsec,
                frame=s_map.coordinate_frame,
            ),
            top_right=SkyCoord(
                arcs_frame * u.arcsec,
                arcs_frame * u.arcsec,
                frame=s_map.coordinate_frame,
            ),
        )
        # remove overlap after submap
        pad_x = s_map.data.shape[0] - resolution
        pad_y = s_map.data.shape[1] - resolution
        s_map = s_map.submap(
            bottom_left=[pad_x // 2, pad_y // 2] * u.pix,
            top_right=[pad_x // 2 + resolution - 1, pad_y // 2 + resolution - 1]
            * u.pix,
        )

        # Ensure exact resolution (handles any rounding errors from rotation/submap)
        if s_map.data.shape[0] != resolution or s_map.data.shape[1] != resolution:
            from scipy.ndimage import zoom

            scale_x = resolution / s_map.data.shape[0]
            scale_y = resolution / s_map.data.shape[1]
            resized_data = zoom(s_map.data, (scale_x, scale_y), order=1)
            # Update CDELT to reflect the resize
            s_map = Map(resized_data, s_map.meta)
            s_map.meta["cdelt1"] = s_map.meta["cdelt1"] / scale_x
            s_map.meta["cdelt2"] = s_map.meta["cdelt2"] / scale_y
            s_map.meta["CDELT1"] = s_map.meta["cdelt1"]
            s_map.meta["CDELT2"] = s_map.meta["cdelt2"]
            # Update reference pixel to center
            s_map.meta["crpix1"] = resolution / 2 + 0.5
            s_map.meta["crpix2"] = resolution / 2 + 0.5
            s_map.meta["CRPIX1"] = s_map.meta["crpix1"]
            s_map.meta["CRPIX2"] = s_map.meta["crpix2"]

        #
        s_map.meta["r_sun"] = s_map.rsun_obs.value / s_map.meta["cdelt1"]
        return s_map

    def _create_resampled_map(
        self, data: np.ndarray, header: dict, resolution: int
    ) -> Map:
        """Create and resample a SunPy map."""
        sanitized_header = _sanitize_header(header)
        s_map = sunpy.map.Map(data, sanitized_header)
        return self._prepMap(s_map, resolution)

    def _save_compressed_fits(
        self, map_data: np.ndarray, meta: dict, output_path: Path
    ) -> None:
        """Save map data as compressed FITS with cleaned metadata."""
        # Clean metadata for FITS compatibility
        clean_meta = {k: v for k, v in meta.items() if _is_valid_fits_value(k, v)}

        try:
            primary = PrimaryHDU()
            comp = CompImageHDU(
                data=map_data,
                header=Header(clean_meta),
                compression_type=FITS_COMPRESSION_TYPE,
                quantize_level=FITS_QUANTIZE_LEVEL,
            )
            HDUList([primary, comp]).writeto(output_path, overwrite=True)
        except Exception as e:
            # If header creation fails, log the problematic keys
            self.logger.error(f"Failed to create FITS header: {e}")
            for k, v in clean_meta.items():
                try:
                    Header([(k, v)])
                except Exception:
                    self.logger.error(f"  Invalid key: {k}={v}")
            raise

    def _post_process_FITS(
        self,
        full_path: Path,
        resampled_path: Path,
        resolution: int = DEFAULT_RESOLUTION,
    ) -> Path | None:
        """Create resampled FITS from full-size FITS."""
        try:
            with fits.open(full_path, mode="readonly", memmap=False) as hdul:
                hdul.verify("silentfix")

                img_hdu = _find_image_hdu(hdul)
                if img_hdu is None:
                    self.logger.error(f"No image HDU found in {full_path}")
                    return None

                # Create resampled map
                res_map = self._create_resampled_map(
                    img_hdu.data, dict(img_hdu.header), resolution
                )

                # Preserve original metadata
                for key in (
                    "DATE-OBS",
                    "DATE_OBS",
                    "DATE__OBS",
                    "TELESCOP",
                    "INSTRUME",
                    "RSUN_OBS",
                    "rsun_obs",
                    "DSUN_OBS",
                    "dsun_obs",
                ):
                    if key in img_hdu.header and key not in res_map.meta:
                        res_map.meta[key] = img_hdu.header[key]

                # Ensure coordinate system
                res_map.meta.setdefault("CTYPE1", "HPLN-TAN")
                res_map.meta.setdefault("CTYPE2", "HPLT-TAN")
                res_map.meta.setdefault("CUNIT1", "arcsec")
                res_map.meta.setdefault("CUNIT2", "arcsec")

            # Save compressed FITS
            self._save_compressed_fits(res_map.data, res_map.meta, resampled_path)
            self.logger.info(f"Wrote: {resampled_path}")
            return resampled_path

        except Exception as e:
            self.logger.error(f"Failed to resample {full_path}: {e}")
            return None

    def post_process_parallel(
        self,
        full_resampled_pairs: list[tuple[Path, Path]],
        resolution: int = DEFAULT_RESOLUTION,
        max_workers: int = 4,
        progress_callback: Callable[[int, int], None] = None,
    ) -> list[tuple[Path, Path | None]]:
        """Post-process multiple full-size FITS files in parallel."""

        def worker(paths: tuple[Path, Path]) -> tuple[Path, Path | None]:
            full_path, resampled_path = paths
            try:
                result = self._post_process_FITS(full_path, resampled_path, resolution)
                return (full_path, result)
            except Exception as e:
                self.logger.error(f"Exception post-processing {full_path}: {e}")
                return (full_path, None)

        return self._parallel_process(
            full_resampled_pairs, worker, max_workers, progress_callback
        )


if __name__ == "__main__":
    SAVE_DIR = Path("D:/projects/research/CHASM/download_data/aia_imagery")
    downloader = CHASM_AIADownloader(
        save_path=SAVE_DIR, email="cbeckdevelopment@gmail.com"
    )

    # Test with a small set of dates and wavelengths
    years = [2017]
    test_dates = []
    for year in years:
        swpc_dir = Path(f"D:/projects/research/CHASM/download_data/drawings/{year}")
        dates = downloader.get_dates_from_swpc_drawing_dir(swpc_dir)
        print(f"Found {len(dates)} drawing dates for year {year}")
        test_dates.extend(dates[:10])
    test_wavelengths = JSOCQuery.VALID_WAVELENGTHS

    # Build test queries
    test_queries = [
        downloader.get_jsoc_query(d, int(wl))
        for d in test_dates
        for wl in test_wavelengths
    ]
    print(f"Test queries: {len(test_queries)}")

    def download_progress(completed, total):
        print(f"  Download progress: {completed}/{total}")

    try:
        download_results = downloader.download_images_parallel(
            test_queries,
            threshold_minutes=60,
            max_workers=6,
            progress_callback=download_progress,
        )

        # Report download results
        successful_downloads = sum(
            1 for _, path in download_results if path is not None
        )
        print(
            f"\nDownload complete: {successful_downloads}/{len(test_queries)} successful"
        )

        # Post-process any downloaded full-size FITS into resampled versions in parallel
        print("\nPost-processing full-size images in parallel...")
        pairs_to_process = []

        for year in years:
            full_root = SAVE_DIR / f"{year}_FullSize"
            if not full_root.exists():
                continue
            for wl_dir in full_root.iterdir():
                if not wl_dir.is_dir():
                    continue
                try:
                    wavelength = int(wl_dir.name)
                except Exception:
                    continue
                for full_file in wl_dir.glob("*.fits"):
                    # expect filename like YYYY-MM-DD.fits
                    try:
                        dt = datetime.fromisoformat(full_file.stem)
                    except Exception:
                        # skip files that don't match the date pattern
                        continue
                    _, resampled_path = downloader._paths_for_time_and_wavelength(
                        dt, wavelength
                    )
                    if resampled_path.exists():
                        continue
                    resampled_path.parent.mkdir(parents=True, exist_ok=True)
                    pairs_to_process.append((full_file, resampled_path))

        if pairs_to_process:
            print(f"Found {len(pairs_to_process)} files to post-process")

            def postprocess_progress(completed, total):
                print(f"  Post-process progress: {completed}/{total}")

            postprocess_results = downloader.post_process_parallel(
                pairs_to_process,
                resolution=512,
                max_workers=6,
                progress_callback=postprocess_progress,
            )

            # Report post-processing results
            successful_postprocess = sum(
                1 for _, path in postprocess_results if path is not None
            )
            print(
                f"\nPost-processing complete: {successful_postprocess}/{len(pairs_to_process)} successful"
            )
        else:
            print("No files need post-processing")

    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user. Exiting gracefully...")
        import sys

        sys.exit(1)
