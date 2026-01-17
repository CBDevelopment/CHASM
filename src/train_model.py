import os
import numpy as np
import cv2
from astropy.io import fits
import re
from dataclasses import dataclass, field
from tqdm import tqdm
from pathlib import Path

# NOTE: Don't use CHRONNOS package for now, being tinkered to improve things
from MultiChannelCHDetection.chronnos.train.model import Trainer
from datasets.folder_utils import *
from MultiChannelCHDetection.chronnos.data.convert import get_intersecting_files, convertMaps, convertMasks, sdo_norms_dict, get_local_correction_table, _MapConverter

from datasets.aia_dataset import AIADataset
from datasets.chasm_dataset import CHASMDataset, CHASM1407, CHASM1111, CHASM960
from datasets.fits_dataset import FITSDataset
from datasets.chronnos_dataset import CHRONNOSDataset, DefaultChronnosConfig

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
    extensions = extensions if extensions is not None else ['.fits'] * len(dirs)
    basenames = [os.path.basename(file) for root, dirs, files in os.walk(path) for file in files]
    basename_dates_to_rest = {}
    num_date_occurs = {}
    for basename in tqdm(basenames, desc="Getting intersecting files"):
        date_pattern = r"^(\d{4}-\d{2}-\d{2})(.*)$" 
        match = re.match(date_pattern, basename)
        if match:
            date_part, rest = match.groups()
            basename_dates_to_rest[date_part] = rest
            if (date_part in num_date_occurs):
                num_date_occurs[date_part] += 1
            else:
                num_date_occurs[date_part] = 1
        else:
            print(f"No match for: {basename}, (there should be a match for all files except logs)")
    
    intersecting_basename_dates = [key for key, value in num_date_occurs.items() if value >= len(dirs)]
    intersecting_basenames = [date_part + basename_dates_to_rest[date_part] for date_part in intersecting_basename_dates]
    intersecting_basenames = sorted(list(intersecting_basenames))
    return [[os.path.join(path, str(directory), b) for b in intersecting_basenames] for directory in dirs]

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
    for subfolder_name in os.listdir(folder_a):
        subfolder_a_path = os.path.join(folder_a, subfolder_name)
        subfolder_b_path = os.path.join(folder_b, subfolder_name)

        # Only process if both subfolders exist
        if not os.path.isdir(subfolder_a_path):
            continue
        if not os.path.isdir(subfolder_b_path):
            print(f"Skipping {subfolder_a_path}, no matching subfolder in {folder_b}")
            continue

        # Get filenames in both subfolders
        files_a = set(os.listdir(subfolder_a_path))
        files_b = set(os.listdir(subfolder_b_path))

        unmatched = files_a - files_b

        for filename in unmatched:
            path_to_remove = os.path.join(subfolder_a_path, filename)
            if os.path.isfile(path_to_remove):
                print(f"Removing {path_to_remove}")
                os.remove(path_to_remove)

if __name__=="__main__":
    chronnos_ds = CHRONNOSDataset(root=r"D:\CHASM_TABLES\PREDICTION_PIPELINE_TEST_2\chronnos_training", config=DefaultChronnosConfig()) 
    aia_ds = AIADataset(r"D:\CHASM_TABLES\PREDICTION_PIPELINE_TEST_2\aia", fetch_online=False)
    fits_ds = FITSDataset(root=r"D:\CHASM_TABLES\PREDICTION_PIPELINE_TEST_2\fits")
    chronnos_ds.build(aia_ds, fits_ds, verbose=True)

    chronnos_ds.train_models_swpc(results_path=r"D:\CHASM_TABLES\PREDICTION_PIPELINE_TEST_2\training_results")