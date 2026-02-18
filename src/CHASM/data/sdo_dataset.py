import numpy as np
from pathlib import Path

from .auto_download_dataset import AutoDownloadDataset


class SDODataset(AutoDownloadDataset):
    def __init__(self, root="download_data/aia_wavelengths", fetch_online=False):
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
