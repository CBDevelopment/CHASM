from pathlib import Path
import numpy as np
from tqdm import tqdm
import json

try:
    from .auto_download_dataset import AutoDownloadDataset
except ImportError:
    from auto_download_dataset import AutoDownloadDataset


# TODO: Also return AIA imagery in the datasets
class CHASMDataset(AutoDownloadDataset):
    def __init__(self, root: str, fetch_online=False):
        self.root = root
        self.mode = "all"
        if fetch_online:
            file_id = "17XR3TII5onfWo67E3uyBjpYXPYzB-Tna"
            url = f"https://drive.google.com/uc?id={file_id}"
            super().__init__(
                root=root,
                filename="tool_selections.tar.gz",
                url=url,
            )
        else:
            super().__init__(root=root)

    def filenames(self) -> list[str]:
        return sorted((str(p) for p in Path(self.root).rglob("*") if p.is_file()))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        data = np.load(file_path, allow_pickle=True)
        return data


class CHASM1407(CHASMDataset):
    def __init__(self, root: str, sdo_imagery_dir: str, fetch_online=False):
        super().__init__(root, fetch_online)
        self.root = root
        self.sdo_imagery_dir = sdo_imagery_dir
        self.fetch_online = fetch_online
        self.files = super().filenames()
        self.json_file = [f for f in self.files if "missing_wavelength_dates" in f][0]
        self.files_1407 = self.prune_to_1407()

    def filenames(self):
        return self.files_1407

    def prune_to_1407(self) -> list[Path]:
        missing_wavelength_dates = json.load(open(self.json_file, "r"))[
            "MISSING_WAVELENGTHS"
        ]
        missing_wavelength_dates = [
            d.replace("-", "") for d in missing_wavelength_dates
        ]
        print("Missing Wavelength Dates: ", len(missing_wavelength_dates))

        remaining_files = [
            Path(f) for f in self.files if "missing_wavelength_dates" not in f
        ]
        # 2022-03-23 synoptic map could not be segmented, not included in CHASM Selections
        # 2022-08-16 synoptic map was rotated 90 degrees, not included in CHASM Selections

        not_present_193A_dates = {"20171108", "20220720"}
        # These two dates are missing 193A images so they could not be used to create properly scaled CHASM masks

        remaining_dates = (
            set([f.stem.split("_")[-2] for f in remaining_files])
            - not_present_193A_dates
            - set(missing_wavelength_dates)
        )
        print("CHASM Selections: ", len(remaining_dates))

        return [f for f in remaining_files if f.stem.split("_")[-2] in remaining_dates]

    def __len__(self):
        return len(self.files_1407)

    def __getitem__(self, idx):
        file_path = self.files_1407[idx]

        data = np.load(file_path, allow_pickle=True)
        return data


class CHASM1111(CHASM1407):
    def __init__(self, root: str, sdo_imagery_dir: str, fetch_online=False):
        super().__init__(root, fetch_online)
        self.root = root
        self.sdo_imagery_dir = sdo_imagery_dir
        self.fetch_online = fetch_online
        self.files_1111 = self.prune_to_1111()

    def filenames(self):
        return self.files_1111

    def prune_to_1111(self) -> list[Path]:
        new_file_paths = []
        for idx, file_path in tqdm(
            enumerate(self.files_1407),
            total=len(self.files_1407),  # so tqdm knows the max
            desc="Pruning to 1111",
        ):
            data = np.load(file_path, allow_pickle=True)
            if data["All Detected"] and np.all(data["Quality"] == "Good"):
                new_file_paths.append(self.files_1407[idx])

        print("CHASM-1111 Selections: ", len(new_file_paths))
        return new_file_paths


class CHASM970(CHASM1111):
    def __init__(self, root: str, sdo_imagery_dir: str, fetch_online=False):
        super().__init__(root, fetch_online)
        self.root = root
        self.sdo_imagery_dir = sdo_imagery_dir
        self.fetch_online = fetch_online
        self.files_970 = self.prune_to_970()

    def filenames(self):
        return self.files_970

    def prune_to_970(self) -> list[Path]:
        timeshifted_dates = json.load(open(self.json_file, "r"))["PRE_TIMESHIFT_DATES"]
        timeshifted_dates = [d.replace("-", "") for d in timeshifted_dates]

        new_file_paths = []
        for idx, file_path in tqdm(
            enumerate(self.files_1111),
            total=len(self.files_1111),  # so tqdm knows the max
            desc="Pruning to 970",
        ):
            if file_path.stem.split("_")[-2] not in timeshifted_dates:
                new_file_paths.append(self.files_1111[idx])

        print("CHASM-970 Selections: ", len(new_file_paths))
        return new_file_paths


if __name__ == "__main__":
    BASE_DIR = Path("../../../CHASM_data")
    CHASM_SELECTIONS_DIR = BASE_DIR / "chasm_selections"

    chasm_selections = CHASMDataset(root=CHASM_SELECTIONS_DIR, fetch_online=False)
    print(len(chasm_selections.filenames()))

    print("CHASM 1407")
    chasm_1407 = CHASM1407(
        root=CHASM_SELECTIONS_DIR,
        sdo_imagery_dir=BASE_DIR / "sdo_imagery" / "full_size_images",
    )

    chasm_1111 = CHASM1111(
        root=CHASM_SELECTIONS_DIR,
        sdo_imagery_dir=BASE_DIR / "sdo_imagery" / "full_size_images",
    )

    chasm_970 = CHASM970(
        root=CHASM_SELECTIONS_DIR,
        sdo_imagery_dir=BASE_DIR / "sdo_imagery" / "full_size_images",
    )
