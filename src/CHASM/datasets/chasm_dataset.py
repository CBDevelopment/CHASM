from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import json
from .auto_download_dataset import AutoDownloadDataset
from .folder_utils import extract_date


class CHASMDataset(AutoDownloadDataset):
    def __init__(self, root="download_data/chasm", fetch_online=True):
        self.root = root
        self.mode = "all"
        self.all_file_paths = []
        if fetch_online:
            # TODO fix it so it downloads the CHASM with new CSV format
            file_id = "1DeKFXMQ39jh2-5Q5gVK7z_7sPoVQf6Eh"
            url = f"https://drive.google.com/uc?id={file_id}"
            super().__init__(
                root=root,
                filename="tool_selections.tar.gz",
                url=url,
            )
        else:
            super().__init__(root=root)

    # TODO add set mode for aia (or general dataset object?)
    def get_mode(self):
        return self.mode

    def set_mode_all(self):
        self.mode = "all"
        self.file_paths = self.all_file_paths

    def set_mode_train(self):
        self.mode = "train"
        self.file_paths = [
            item
            for item in self.all_file_paths
            if extract_date(str(item)).split("-")[1] not in ["11", "12"]
        ]

    def set_mode_test(self):
        self.mode = "test"
        self.file_paths = [
            item
            for item in self.all_file_paths
            if extract_date(str(item)).split("-")[1] in ["11", "12"]
        ]

    def filenames(self) -> list[str]:
        return sorted((str(p) for p in Path(self.root).rglob("*") if p.is_file()))

    def get_aia_days(self):
        j = json.load(open(Path(self.root) / "missing_wavelength_dates.json"))[
            "MISSING_WAVELENGTHS"
        ]
        return j

    def get_pre_timeshift_days(self):
        j = json.load(open(Path(self.root) / "missing_wavelength_dates.json"))[
            "PRE_TIMESHIFT"
        ]
        return j

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        data = np.load(file_path, allow_pickle=True)
        return data


class CHASM1407(CHASMDataset, Dataset):
    # TODO figure out where the extra ~30 days are coming from
    def __init__(
        self, root="download_data/chasm_as_npz", fetch_online=True, needs_aia=True
    ):
        super().__init__(root, fetch_online)
        self.root = root
        self.needs_aia = needs_aia
        # TODO get aia days properly from an aia path
        if needs_aia:
            self.aia_days = (
                self.get_aia_days()
            )  # TODO fix this is broken right now! (have it link to actual AIA folder)
        else:
            self.aia_days = []
        self.prune_to_1407()

    def prune_to_1407(self):
        new_file_paths = []
        for idx, file_path in tqdm(
            enumerate(self.file_paths),
            total=len(self.file_paths),  # so tqdm knows the max
            desc="Pruning to 1407",
        ):
            # TODO could make this less brittle by using dictionary instead of directly from file name?
            filename = Path(file_path).stem
            date_str = extract_date(filename)  # '2017-05-10'

            data = np.load(file_path, allow_pickle=True)
            # Drop paths that are "bad" (incorrect # of CHs or flagged)

            if (date_str not in self.aia_days) or not self.needs_aia:
                new_file_paths.append(self.file_paths[idx])

        self.all_file_paths = new_file_paths  # TODO figure out why not down to 1111
        self.file_paths = self.all_file_paths

    def filenames(self):
        return self.file_paths

    # def __len__(self):
    #     return len(self.file_paths)

    # def __getitem__(self, idx):
    #     file_path = self.file_paths[idx]

    #     data = np.load(file_path, allow_pickle=True)
    #     sample = data[list(data.keys())[0]]

    #     return sample


class CHASM1111(CHASMDataset, Dataset):
    def __init__(
        self, root="download_data/chasm_as_npz", fetch_online=True, needs_aia=True
    ):
        super().__init__(root, fetch_online)
        self.root = root
        self.needs_aia = needs_aia
        if needs_aia:
            self.aia_days = (
                self.get_aia_days()
            )  # TODO fix this is broken right now! (have it link to actual AIA folder)
        else:
            self.aia_days = None
        self.prune_to_1111()

    def prune_to_1111(self):
        new_file_paths = []
        for idx, file_path in tqdm(
            enumerate(self.file_paths),
            total=len(self.file_paths),  # so tqdm knows the max
            desc="Pruning to 1111",
        ):
            # TODO could make this less brittle by using dictionary instead of directly from file name?
            filename = Path(file_path).stem
            date_str = extract_date(filename)  # '2017-05-10'

            # file_path_date =
            data = np.load(file_path, allow_pickle=True)

            # TODO need to fix the CHASM download from Google Drive otherwise this won't work (needs key "quality")
            # Drop paths that are "bad" (incorrect # of CHs or flagged)

            if "Quality" in data.keys():
                if data["All Detected"] and data["Quality"] == "Good":
                    if self.needs_aia:
                        if date_str in self.aia_days:
                            new_file_paths.append(self.file_paths[idx])
                    else:
                        new_file_paths.append(self.file_paths[idx])

            # new_file_paths.append(self.file_paths[idx])

        # self.all_file_paths = new_file_paths # TODO figure out why not down to 1111
        self.file_paths = new_file_paths

    def filenames(self):
        return self.file_paths

    # def __len__(self):
    #     return len(self.file_paths)

    # def __getitem__(self, idx):
    #     file_path = self.file_paths[idx]

    #     data = np.load(file_path, allow_pickle=True)
    #     return data


# TODO add CHASM 960 (somehow get the AIA times, maybe manually through that list?)


class CHASM960(CHASMDataset, Dataset):
    def __init__(self, fetch_online=True, root="download_data/chasm_as_npz"):
        super().__init__(root, fetch_online)
        self.root = root
        self.aia_days = self.get_aia_days()
        self.pre_timeshift_days = self.get_pre_timeshift_days()
        self.prune_to_960()

    def filenames(self):
        return self.file_paths

    def prune_to_960(self):
        new_file_paths = []
        for idx, file_path in tqdm(
            enumerate(self.file_paths),
            total=len(self.file_paths),  # so tqdm knows the max
            desc="Pruning to 967",
        ):
            # TODO could make this less brittle by using dictionary instead of directly from file name?
            filename = Path(file_path).stem
            date_str = filename.split("T")[0]  # '2017-05-10'

            # file_path_date =
            data = np.load(file_path, allow_pickle=True)
            # Drop paths that are "bad" (incorrect # of CHs or flagged)

            if data["All Detected"] and np.all(data["Quality"] == "Good"):
                if date_str not in self.pre_timeshift_days:
                    new_file_paths.append(self.file_paths[idx])

        self.all_file_paths = new_file_paths  # TODO figure out why not down to 1111
        self.file_paths = self.all_file_paths

    # def __len__(self):
    #     return len(self.file_paths)

    # def __getitem__(self, idx):
    #     file_path = self.file_paths[idx]

    #     data = np.load(file_path, allow_pickle=True)
    #     return data
