from datasets.auto_download_dataset import AutoDownloadDataset
import numpy as np
import os

# TODO clean up imports
import sunpy.map
import matplotlib.pyplot as plt
import os
from astropy.visualization import AsinhStretch, LinearStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.fits import HDUList, PrimaryHDU, Header
from astropy.io.fits.hdu.compressed import CompImageHDU
from multiprocessing import Pool
import numpy as np
import shutil

# TODO need to add filtering by the range of time for each of them
# TODO need to add the ability to resize mags when downloaded automatically
class MagResizer():
    def __init__(self, root, year,  source_folder="6173", target_folder="6173_resampled"):
        self.root = root
        self.year = year
        self.source_folder = source_folder
        self.target_folder = target_folder

    def process_single_file(self, fits_file):
        """Process a single FITS file to create a resampled version."""
        try:
            # Create and resample the map
            mag_map = sunpy.map.Map(f"{self.root}/{self.year}/{self.source_folder}/{fits_file}")
            resampled_map = mag_map.resample([512, 512]*u.pix)

            # Create a new header with proper ordering
            new_header = fits.Header()

            # Add required keywords in the correct order
            new_header['SIMPLE'] = True
            new_header['BITPIX'] = -32
            new_header['NAXIS'] = 2
            new_header['NAXIS1'] = 512
            new_header['NAXIS2'] = 512

            # Get original header and copy keywords
            with fits.open(f"{self.root}/{self.year}/{self.source_folder}/{fits_file}") as hdul:
                original_header = hdul[1].header

                # Copy essential keywords
                essential_keys = [
                    'TELESCOP', 'INSTRUME', 'WAVELNTH', 'WCSNAME',
                    'CTYPE1', 'CTYPE2', 'CRPIX1', 'CRPIX2',
                    'CRVAL1', 'CRVAL2', 'CDELT1', 'CDELT2',
                    'CUNIT1', 'CUNIT2', 'RSUN_OBS', 'RSUN_REF', 'R_SUN',
                    'DATE-OBS', 'T_OBS', 'T_REC', 'BUNIT'
                ]

                for key in essential_keys:
                    if key in original_header:
                        try:
                            new_header[key] = original_header[key]
                        except Exception as e:
                            print(
                                f"Warning: Could not copy keyword {key} for {fits_file}: {str(e)}")

                # Copy HIERARCH keywords
                for key in original_header:
                    if key.startswith('HIERARCH'):
                        try:
                            new_header[key] = original_header[key]
                        except Exception as e:
                            print(
                                f"Warning: Could not copy HIERARCH keyword {key} for {fits_file}: {str(e)}")

            # Save resampled file
            os.makedirs(f"{self.root}/{self.year}/{self.target_folder}", exist_ok=True)
            output_path = f"{self.root}/{self.year}/{self.target_folder}/{fits_file}"

            compressed_resampled_map = CompImageHDU(
                data=resampled_map.data,
                header=new_header,
                compression_type='HCOMPRESS_1',
                quantize_level=16.0
            )
            compressed_hdul = HDUList([PrimaryHDU(), compressed_resampled_map])
            compressed_hdul.writeto(output_path, overwrite=False)
            return f"Successfully processed {fits_file}"

        except Exception as e:
            return f"Error processing {fits_file}: {str(e)}"

    def process_folder(self):
        print("PROCESSING FOLDER! ")
        fits_dir = f"{self.root}/{self.year}/{self.source_folder}"
        print("DIR ", fits_dir)
        fits_files = sorted([f for f in os.listdir(fits_dir) if f.endswith(".fits")])
        print("RUNNING POOL. ")
        with Pool(processes=8) as pool: 
            results = pool.map(self.process_single_file, fits_files)
        print("POOL COMPLETE ", results)

    def view_fits(self, fits_file):
        """View a FITS file."""
        resampled = f"{self.root}/{self.year}/{self.source_folder}/{fits_file}"
        mag_map = sunpy.map.Map(resampled)
        print(mag_map.data.shape)
        norm = ImageNormalize(vmin=-100, vmax=100, stretch=LinearStretch())
        fig = plt.figure()
        ax = plt.subplot(projection=mag_map.wcs)
        ax.imshow(mag_map.data, cmap='gray', norm=norm)
        ax.grid(False)
        plt.show()


# TODO fix issue where if fetch_online and resize it resizes even if directory exists
class AIADataset(AutoDownloadDataset):
    def __init__(self, root="download_data/aia_wavelengths", fetch_online=True, resize_mag=True):
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
            year_path = os.path.join(self.root, year)
            for folder in os.listdir(year_path):
                if folder == "6173":  # specifically target the magnetograms and move to pre_resized
                    src = os.path.join(year_path, folder)
                    dst = os.path.join(year_path, f"pre_resized_{folder}")
                    if not os.path.exists(dst):
                        print(f"[INFO] Moving {src} → {dst}")
                        shutil.move(src, dst)
                    else:
                        print(f"[WARN] {dst} already exists, skipping move")

                    # now resample into a fresh 6173/ folder
                    os.makedirs(os.path.join(year_path, "6173"), exist_ok=True)
                    print(f"[INFO] Resizing magnetograms in {year}/6173 ...")
                    resizer = MagResizer(
                        root=self.root,
                        year=year,
                        source_folder=f"pre_resized_{folder}",
                        target_folder=folder
                    )
                    resizer.process_folder()

    def get_years(self):
        return [
            name for name in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, name))
        ]
    
    def get_wavelength_folders(self, year):
        year_path = os.path.join(self.root, year)
        folders = [
            os.path.join(year_path, name)
            for name in os.listdir(year_path)
            if os.path.isdir(os.path.join(year_path, name))
        ]
        return folders
    
    # TODO need to make converter in this or another dataset for running through CHRONNOS?
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        data = np.load(file_path, allow_pickle=True)
        return data

    # TODO here add the AIA loading script (that queries the website directly) for usage to make a dataset from scratch instead of our drive