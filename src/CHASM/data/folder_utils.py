import re
import shutil
from tqdm import tqdm
from pathlib import Path


def extract_date(filename):
    # Ensure we’re always working with a string
    filename = str(filename)

    # Search for 'YYYY-MM-DD' or 'YYYYMMDD' anywhere in the filename
    match2 = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    match1 = re.search(r"(\d{8})", filename)

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
        folder_path = Path(folder)
        for f in folder_path.iterdir():
            if extract_date(f.name) == filename_date:
                return str(f)

    else:
        print("Unknown type for folder, should be str | Path | list[str]")
        return None

    return None


def rename_files_in_folder(root_folder):
    root_path = Path(root_folder)
    for dir_path in root_path.rglob("*"):
        if dir_path.is_dir():
            for file_path in dir_path.iterdir():
                if file_path.is_file():
                    new_name = extract_date(file_path.name)
                    if new_name and file_path.name != new_name:
                        dst = file_path.parent / new_name
                        print(f"Renaming: {file_path} -> {dst}")
                        file_path.rename(dst)


# TODO make sure this works (unit tests)
def merge_directories(aia_path, fits_path, output_path):
    # Output base
    output_path_obj = Path(output_path)
    output_path_obj.mkdir(parents=True, exist_ok=True)

    # ---- Merge AIA ----
    # Get years if present (folder has years has wavelengths) or just wavelengths (folder has wavelengths)
    aia_path_obj = Path(aia_path)
    years = [
        d.name
        for d in aia_path_obj.iterdir()
        if d.is_dir() and any(sub.is_dir() for sub in d.iterdir())
    ]
    if not years:
        years = [None]

    for year in tqdm(years, desc="Merging directories AIA [year]"):
        # Handle sorted by year vs flat directory structure
        if year != None:
            year_path = aia_path_obj / year
            if not year_path.is_dir():
                continue
        else:
            year_path = aia_path_obj  # Flat directory just use outer directory

        for wavelength_dir in tqdm(
            list(year_path.iterdir()), desc=f"Merging Directories [Wavelength]"
        ):
            if not wavelength_dir.is_dir():
                continue

            target_folder = output_path_obj / wavelength_dir.name
            target_folder.mkdir(parents=True, exist_ok=True)

            for file_path in tqdm(
                list(wavelength_dir.iterdir()), desc="Merging Directories [File Names]"
            ):
                if not file_path.is_file():
                    continue
                ext = file_path.suffix
                date_file_name = extract_date(
                    file_path.name
                )  # Use consistent naming convention to work for CHRONNOS preprocessing
                dst_file = target_folder / f"{date_file_name}{ext}"
                shutil.copy2(str(file_path), str(dst_file))

    # ---- Merge FITS ----
    cropped_folder = output_path_obj / "cropped_centered_masks"
    cropped_folder.mkdir(parents=True, exist_ok=True)

    fits_path_obj = Path(fits_path)
    for year in tqdm(years, desc="Merging Directory FITS [year]"):
        # Handle sorted by year vs flat directory structure
        if year != None:
            year_path = fits_path_obj / year
            if not year_path.is_dir():
                continue
        else:
            year_path = fits_path_obj  # Flat directory just use outer directory

        for file_path in year_path.iterdir():
            if not file_path.is_file():
                continue
            ext = file_path.suffix
            date_file_name = extract_date(
                file_path.name
            )  # Use consistent naming convention to work for CHRONNOS preprocessing
            if date_file_name == None:
                print(f"Could not find date for {file_path.name}")
                continue

            dst_file = cropped_folder / f"{date_file_name}{ext}"
            shutil.copy2(str(file_path), str(dst_file))
