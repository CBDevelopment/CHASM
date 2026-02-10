from datetime import datetime
from pathlib import Path
import numpy as np
import logging

from chasm.data import DrawingsDataset, AIADataset, SAMMaskDataset, CHASMDataset

from chasm.data.generate import (
    SWPCDrawingsScraper,
    SAMSegmentationMasksGenerator,
    CHASM_AIADownloader,
    JSOCQuery,
)
from chasm.gui.app import run_app

logger = logging.getLogger("Pipeline Tests")

BASE_DIR = Path(__file__).parent / "pipeline_tests"
logger.info(f"Base directory for pipeline tests: {BASE_DIR}")

DRAWINGS_DIR = BASE_DIR / "drawings"

# SWPC Synoptic Drawings
# Get your own drawings for a specific date or dates
logger.info("SWPC Synoptic Drawings")
swpc_scraper = SWPCDrawingsScraper(save_path=DRAWINGS_DIR)
drawing_file = swpc_scraper.get_drawing_for_date("2026-01-19")
# drawing_files = swpc_scraper.get_drawings_for_date_range("2026-01-01", "2026-01-05")

drawings_dataset = DrawingsDataset(root=DRAWINGS_DIR)
# Can do the following to get our drawings dataset from Google Drive
# drawings_dataset = DrawingsDataset(root=DRAWINGS_DIR, fetch_online=True)
logger.info(f"Number of drawings in dataset: {len(drawings_dataset)}")

# SAM Masks
logger.info("SAM Masks")
# sam_mask_generator = SAMSegmentationMasksGenerator(
#     swpc_drawing_dir=DRAWINGS_DIR,
#     save_dir=BASE_DIR / "masks",
#     model_type="vit_b",
#     sam_checkpoint_filepath=BASE_DIR / "checkpoints" / "sam_vit_b_01ec64.pth",
# )
# sam_mask_generator.process_all_images()

masks_dataset = SAMMaskDataset(root=BASE_DIR / "masks")
print(f"Number of SAM masks in dataset: {len(masks_dataset)}")

# logger.info("Displaying segmentations for first drawing...")
# mask_dict = np.load(masks_dataset[0], allow_pickle=True)
# anns = [mask_dict[key].item() for key in mask_dict.keys()]
# SAMMaskDataset.show_segmentations(anns, drawing_path=drawings_dataset[0])

# Use CHASM
run_app(
    drawings_dataset=drawings_dataset,
    sam_dataset=masks_dataset,
    save_dir=BASE_DIR / "chasm_selections",
    max_width=800,
    max_height=600,
    scale_factor=0.75,
)

# AIA/HMI Imagery
# Download AIA/HMI imagery for the same date(s) as the SWPC drawings using JSOC queries
aia_downloader = CHASM_AIADownloader(
    save_dir=BASE_DIR / "aia_imagery", email="cbeckdevelopment@gmail.com"
)
swpc_dates = aia_downloader.get_dates_from_swpc_drawing_dir(DRAWINGS_DIR)
jsoc_queries = [
    aia_downloader.get_jsoc_query(date, int(wl))
    for date in swpc_dates
    for wl in JSOCQuery.VALID_WAVELENGTHS
]


def download_progress(completed, total):
    print(f"  Download progress: {completed}/{total}")


download_results = aia_downloader.download_images_parallel(
    jsoc_queries,
    threshold_minutes=60,
    max_workers=6,
    progress_callback=download_progress,
)

# Post-Process the downloaded AIA/HMI imagery (e.g. resampling, cropping, aligning)
print("\nPost-processing full-size images in parallel...")
pairs_to_process = []
years = [2017]
for year in years:
    full_root = aia_downloader.save_path / f"{year}_FullSize"
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
            _, resampled_path = aia_downloader._paths_for_time_and_wavelength(
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

    postprocess_results = aia_downloader.post_process_parallel(
        pairs_to_process,
        resolution=512,
        max_workers=6,
        progress_callback=postprocess_progress,
    )

# Split CHASM data into 1407, 1111, 970

# Crop and Align CHASM Tool Selections

# Train/Test CHRONNOS
