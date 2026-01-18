import numpy as np
from pathlib import Path
import csv
from typing import Optional, List, Dict, Any
from chasm.datasets.folder_utils import get_file_by_date


class DataPersistenceManager:
    """Manages saving and loading annotation data to/from disk."""

    def __init__(
        self,
        save_dir: str,
        load_dir: Optional[Any] = None,
        masks_dir: Optional[Any] = None,
    ) -> None:
        self.save_dir = save_dir
        self.load_dir = load_dir
        self.masks_dir = masks_dir

    def get_matching_sam_mask(self, current_filename: str) -> str:
        """Find the corresponding SAM mask file for the current image."""
        if self.masks_dir is None:
            raise ValueError("masks_dir not set")
        mask_file = get_file_by_date(current_filename, self.masks_dir.filenames())
        return mask_file

    def save_current_drawing(
        self,
        filename: str,
        saved_masks: List[Dict[str, Any]],
        detected_chs: str,
        true_chs: str,
    ) -> None:
        """Save the current drawing's annotation data."""
        print("Saving Information")
        flags = [f["flagged"] for f in saved_masks]
        data = {
            "info": saved_masks,
            "SAM Detected": detected_chs,
            "True CHs": true_chs,
            "All Detected": detected_chs == true_chs,
            "Good Quality": True if all(flag is False for flag in flags) else False,
        }

        basename = Path(filename).name
        save_path = Path(self.save_dir) / f"{basename.split('.')[0][:-4]}.npz"
        print(f"SAVING DRAWING TO {save_path}")
        np.savez_compressed(save_path, **data)

    def save_selection_time(self, filename: str, selection_time: float) -> None:
        """Save the time taken to select annotations for the current image."""
        csv_file = Path(self.save_dir) / "selection_times.csv"

        existing_data = []
        if csv_file.exists():
            with open(csv_file, mode="r", newline="") as file:
                reader = csv.reader(file)
                existing_data = list(reader)

        updated = False
        for row in existing_data:
            if row[0] == filename:
                row[1] = selection_time
                updated = True
                break

        if not updated:
            existing_data.append([filename, selection_time])

        with open(csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(existing_data)
        print(f"Time to select: {selection_time:0.2f}")

    def file_exists(self, filename: str) -> bool:
        """Check if annotation file already exists for the given filename."""
        save_path = Path(self.save_dir) / f"{filename.split('.')[0][:-4]}.npz"
        return save_path.exists()

    def load_annotation(self, filename: str) -> Optional[Any]:
        """Load existing annotation data for a file."""
        save_path = Path(self.save_dir) / f"{filename.split('.')[0][:-4]}.npz"
        if save_path.exists():
            return np.load(save_path, allow_pickle=True)
        return None
