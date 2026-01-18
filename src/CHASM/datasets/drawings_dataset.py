from .auto_download_dataset import AutoDownloadDataset
from pathlib import Path


class DrawingsDataset(AutoDownloadDataset):
    def __init__(self, root="download_data/drawings", fetch_online=True):
        self.root = root
        if fetch_online:
            file_id = "1HakI-i6iXuqWDFywPx7Ae6RvqIx5tu-0"
            url = f"https://drive.google.com/uc?id={file_id}"
            super().__init__(
                root=root,
                filename="drawings.tar.gz",
                url=url,
            )
        else:
            super().__init__(root=root)

    def filenames(self) -> list[str]:
        return sorted((str(p) for p in Path(self.root).rglob("*") if p.is_file()))
