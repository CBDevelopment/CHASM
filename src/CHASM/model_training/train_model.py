import numpy as np
import cv2
from astropy.io import fits
import re
from dataclasses import dataclass, field
from tqdm import tqdm
from pathlib import Path

from chronnos.train.model import Trainer
from chasm.data.folder_utils import *
from chronnos.data.convert import (
    get_intersecting_files,
    convertMaps,
    convertMasks,
    sdo_norms_dict,
    get_local_correction_table,
    _MapConverter,
)

from chasm.data.sdo_dataset import SDODataset
from chasm.data.chasm_dataset import (
    CHASMDataset,
    CHASM1407,
    CHASM1111,
    CHASM960,
)
from chasm.data.fits_dataset import FITSDataset
from chasm.data.chronnos_dataset import CHRONNOSDataset, DefaultChronnosConfig


# modified from CHRONNOS get_intersecting_files in convert.py
def get_intersecting_files_date(path, dirs, extensions=None):
    """Find intersecting files in directory.

    :param path: base directory
    :param dirs: directories to scan
    :param months: filter for months
    :param years: filter for years
    :param extensions: file extension for each directory
    :return: list of grouped files
    """
    extensions = extensions if extensions is not None else [".fits"] * len(dirs)
    base_path = Path(path)
    basenames = [f.name for f in base_path.rglob("*") if f.is_file()]
    basename_dates_to_rest = {}
    num_date_occurs = {}
    for basename in tqdm(basenames, desc="Getting intersecting files"):
        date_pattern = r"^(\d{4}-\d{2}-\d{2})(.*)$"
        match = re.match(date_pattern, basename)
        if match:
            date_part, rest = match.groups()
            basename_dates_to_rest[date_part] = rest
            if date_part in num_date_occurs:
                num_date_occurs[date_part] += 1
            else:
                num_date_occurs[date_part] = 1
        else:
            print(
                f"No match for: {basename}, (there should be a match for all files except logs)"
            )

    intersecting_basename_dates = [
        key for key, value in num_date_occurs.items() if value >= len(dirs)
    ]
    intersecting_basenames = [
        date_part + basename_dates_to_rest[date_part]
        for date_part in intersecting_basename_dates
    ]
    intersecting_basenames = sorted(list(intersecting_basenames))
    return [
        [str(Path(path) / str(directory) / b) for b in intersecting_basenames]
        for directory in dirs
    ]


# TODO add more useful verbosity
def remove_unmatched_files_recursive(folder_a, folder_b):
    """
    Remove files in subfolders of folder_a that do not have a matching filename
    in the corresponding subfolder of folder_b.

    Assumes folder structure like:
        folder_a/
            subfolder1/
                file1, file2
            subfolder2/
        folder_b/
            subfolder1/
            subfolder2/

    :param folder_a: path to folder where unmatched files will be deleted
    :param folder_b: path to reference folder for matching files
    """
    # Iterate over all subfolders in folder_a
    folder_a_path = Path(folder_a)
    folder_b_path = Path(folder_b)

    for subfolder in folder_a_path.iterdir():
        if not subfolder.is_dir():
            continue

        subfolder_b = folder_b_path / subfolder.name

        # Only process if both subfolders exist
        if not subfolder_b.is_dir():
            print(f"Skipping {subfolder}, no matching subfolder in {folder_b}")
            continue

        # Get filenames in both subfolders
        files_a = {f.name for f in subfolder.iterdir()}
        files_b = {f.name for f in subfolder_b.iterdir()}

        unmatched = files_a - files_b

        for filename in unmatched:
            path_to_remove = subfolder / filename
            if path_to_remove.is_file():
                print(f"Removing {path_to_remove}")
                path_to_remove.unlink()


def main():
    chronnos_ds = CHRONNOSDataset(
        root=r"D:\CHASM_TABLES\PREDICTION_PIPELINE_TEST_2\chronnos_training",
        config=DefaultChronnosConfig(),
    )
    aia_ds = SDODataset(
        r"D:\CHASM_TABLES\PREDICTION_PIPELINE_TEST_2\aia", fetch_online=False
    )
    fits_ds = FITSDataset(root=r"D:\CHASM_TABLES\PREDICTION_PIPELINE_TEST_2\fits")
    chronnos_ds.build(aia_ds, fits_ds, verbose=True)

    chronnos_ds.train_models_swpc(
        results_path=r"D:\CHASM_TABLES\PREDICTION_PIPELINE_TEST_2\training_results"
    )


if __name__ == "__main__":
    main()
