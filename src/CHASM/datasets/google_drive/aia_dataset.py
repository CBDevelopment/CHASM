import numpy as np
from pathlib import Path
import shutil

from .auto_download_dataset import AutoDownloadDataset
from ..mag_resizer import MagResizer


# TODO fix issue where if fetch_online and resize it resizes even if directory exists
class AIADataset(AutoDownloadDataset):
    def __init__(
        self, root="download_data/aia_wavelengths", fetch_online=True, resize_mag=True
    ):
        if fetch_online:
            file_id = "1yUAaiS-YGf7r-VRsFZfwsCDHN36uJnZ4"
            url = f"https://drive.google.com/uc?id={file_id}"
            super().__init__(
                root=root,
                filename="aia.tar.gz",
                url=url,
            )
        else:
            super().__init__(root)

        # Resize mag so it can be used for models, sizing up to same size as other wavelengths
        if fetch_online and resize_mag:
            self._prepare_and_resize_magnetograms()

    def _prepare_and_resize_magnetograms(self):
        years = self.get_years()
        for year in years:
            year_path = Path(self.root) / year
            for folder_path in year_path.iterdir():
                if not folder_path.is_dir():
                    continue
                if (
                    folder_path.name == "6173"
                ):  # specifically target the magnetograms and move to pre_resized
                    dst = year_path / f"pre_resized_{folder_path.name}"
                    if not dst.exists():
                        print(f"[INFO] Moving {folder_path} → {dst}")
                        shutil.move(str(folder_path), str(dst))
                    else:
                        print(f"[WARN] {dst} already exists, skipping move")

                    # now resample into a fresh 6173/ folder
                    new_6173 = year_path / "6173"
                    new_6173.mkdir(parents=True, exist_ok=True)
                    print(f"[INFO] Resizing magnetograms in {year}/6173 ...")
                    resizer = MagResizer(
                        root=self.root,
                        year=year,
                        source_folder=f"pre_resized_{folder_path.name}",
                        target_folder=folder_path.name,
                    )
                    resizer.process_folder()

    def get_years(self):
        root_path = Path(self.root)
        return [d.name for d in root_path.iterdir() if d.is_dir()]

    def get_wavelength_folders(self, year):
        year_path = Path(self.root) / year
        folders = [str(d) for d in year_path.iterdir() if d.is_dir()]
        return folders

    # TODO need to make converter in this or another dataset for running through CHRONNOS?
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        data = np.load(file_path, allow_pickle=True)
        return data
