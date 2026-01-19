from pathlib import Path
import tarfile
import zipfile
import gdown
from torch.utils.data import Dataset


class AutoDownloadDataset(Dataset):
    def __init__(self, root=None, url=None, filename=None, transform=None):
        """
        Args:
            dataset_name (str): Name for local cache directory
            root (str or Path, optional): Where to store dataset
            url (str, optional): URL to download dataset from
            foldername (str, optional): Extracted folder name
            transform (callable, optional): Transform applied to each sample
        """
        self.root = Path(root)
        self.filename = filename
        self.url = url
        self.transform = transform

        # Download if missing
        self.prepare()

        # Index all files (you can adjust this filter)

        # TODO figure out issue with file paths here, giving errors
        self.file_paths = sorted(
            [
                p
                for p in self.root.rglob("*")
                if p.suffix in [".npz", ".npy", ".png", ".jpg"]
            ]
        )
        if not self.file_paths:
            raise RuntimeError(f"No data files found in {self.root}")

    def get_file_paths(self):
        return self.file_paths

    def get_data_path(self):
        return self.root

    def _check_exists(self):
        return self.root.exists() and any(self.root.iterdir())

    def _download(self):
        self.root.mkdir(parents=True, exist_ok=True)
        archive_path = self.root / self.filename
        print(f"Downloading from {self.url} to {archive_path}...")
        gdown.download(self.url, str(archive_path), quiet=False)
        return archive_path

    def _extract(self, archive_path):
        print(f"Extracting {archive_path} to {self.root}...")
        if (
            archive_path.suffixes[-2:] == [".tar", ".gz"]
            or archive_path.suffix == ".tgz"
        ):
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=self.root)
        elif archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(path=self.root)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path}")
        archive_path.unlink()

    def prepare(self):
        if self._check_exists():
            print(f"Dataset already exists at {self.root}, skipping download.")
        elif self.url and self.filename:
            archive_path = self._download()
            self._extract(archive_path)
        elif not self.url:
            # No URL provided (fetch_online=False), just check if data exists
            if not self.root.exists() or not any(self.root.iterdir()):
                raise RuntimeError(
                    f"Dataset not found at {self.root}. "
                    "Set fetch_online=True to download, or ensure the data exists at the specified path."
                )
        print("Dataset is ready.")
