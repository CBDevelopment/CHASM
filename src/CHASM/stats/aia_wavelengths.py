from pathlib import Path
import numpy as np
from chasm.datasets.google_drive.aia_dataset import AIADataset

if __name__ == "__main__":
    path = r"/home/egsmith/scratch/CoronalHoles/src/polished_pipeline/PREDICTIONS/CHASM1111/aia"
    aia = AIADataset(
        path,
    )
    years = aia.get_years()

    for year in years:
        wavelength_folders = aia.get_wavelength_folders(year)
        num_per_wavelength = [
            len(list(Path(folder).iterdir())) for folder in wavelength_folders
        ]

        print(
            f"Year: {year}, Mean: {np.mean(num_per_wavelength)}, Standard Deviation: {np.std(num_per_wavelength)}"
        )

    # TODO get this by year as well
