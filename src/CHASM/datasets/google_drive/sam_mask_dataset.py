from .auto_download_dataset import AutoDownloadDataset
from pathlib import Path


class SAMMaskDataset(AutoDownloadDataset):
    def __init__(self, root="download_data/masks", fetch_online=True):
        self.root = root
        if fetch_online:
            file_id = "1Gf5X-o6KPX4IRyWcAqcWBVbMdj4lYc7b"
            url = f"https://drive.google.com/uc?id={file_id}"
            super().__init__(
                root=root,
                filename="masks.tar.gz",
                url=url,
            )
        else:
            super().__init__(root=root)

    def filenames(self) -> list[str]:
        return sorted((str(p) for p in Path(self.root).rglob("*") if p.is_file()))
