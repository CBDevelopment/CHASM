import re 
import os 
import shutil
from tqdm import tqdm
from pathlib import Path

def extract_date(filename):
    # Ensure we’re always working with a string
    filename = str(filename)

    # Search for 'YYYY-MM-DD' or 'YYYYMMDD' anywhere in the filename
    match2 = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    match1 = re.search(r'(\d{8})', filename)

    if match2:
        date = match2.group(1)
    elif match1:
        date_raw = match1.group(1)
        date = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:]}"
    else:
        return None

    return date


def get_file_by_date(filename, folder: str | Path | list[str]):
    filename_date = extract_date(filename)
    if not filename_date:
        return None
    
    if isinstance(folder, list):
        for f in folder:
            if extract_date(f) == filename_date:
                return f
            
    elif isinstance(folder, (str, Path)):
        for f in os.listdir(folder):
            if extract_date(f) == filename_date:
                return os.path.join(folder, f)

    else:
        print("Unknown type for folder, should be str | Path | list[str]")
        return None
    
    return None


def rename_files_in_folder(root_folder):
    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            new_name = extract_date(fname)
            if new_name and fname != new_name:
                src = os.path.join(dirpath, fname)
                dst = os.path.join(dirpath, new_name)
                print(f"Renaming: {src} -> {dst}")
                os.rename(src, dst)

# TODO make sure this works (unit tests)
def merge_directories(aia_path, fits_path, output_path):

    # Output base
    os.makedirs(output_path, exist_ok=True)

    # ---- Merge AIA ----
    # Get years if present (folder has years has wavelengths) or just wavelengths (folder has wavelengths)
    years = [d for d in os.listdir(aia_path)
             if os.path.isdir(os.path.join(aia_path, d)) and
                any(os.path.isdir(os.path.join(aia_path, d, sub))
                    for sub in os.listdir(os.path.join(aia_path, d)))]
    if not years:
        years = [None]

    for year in tqdm(years, desc="Merging directories AIA [year]"):
        # Handle sorted by year vs flat directory structure
        if year!=None:
            year_path = os.path.join(aia_path, year)
            if not os.path.isdir(year_path):
                continue

        else:
            year_path = aia_path # Flat directory just use outer directory

        for wavelength in tqdm(os.listdir(year_path), desc=f"Merging Directories [Wavelength]"):
            wavelength_path = os.path.join(year_path, wavelength)
            if not os.path.isdir(wavelength_path):
                continue

            target_folder = os.path.join(output_path, wavelength)
            os.makedirs(target_folder, exist_ok=True)

            for file_name in tqdm(os.listdir(wavelength_path), desc="Merging Directories [File Names]"):
                ext = os.path.splitext(file_name)[-1]
                date_file_name = extract_date(file_name) # Use consistent naming convention to work for CHRONNOS preprocessing
                src_file = os.path.join(wavelength_path, file_name)
                dst_file = os.path.join(target_folder, date_file_name + ext)
                shutil.copy2(src_file, dst_file)

    # ---- Merge FITS ----
    cropped_folder = os.path.join(output_path, "cropped_centered_masks")
    os.makedirs(cropped_folder, exist_ok=True)

    for year in tqdm(years, desc="Merging Directory FITS [year]"):
        # Handle sorted by year vs flat directory structure
        if year!=None:
            year_path = os.path.join(fits_path, year)
            if not os.path.isdir(year_path):
                continue

        else:
            year_path = fits_path # Flat directory just use outer directory

        for file_name in os.listdir(year_path):
            ext = os.path.splitext(file_name)[-1]
            date_file_name = extract_date(file_name) # Use consistent naming convention to work for CHRONNOS preprocessing
            if date_file_name==None:
                print(f"Could not find date for {file_name}")
                continue

            src_file = os.path.join(year_path, file_name)
            dst_file = os.path.join(cropped_folder, date_file_name + ext)
            shutil.copy2(src_file, dst_file)