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

class MagResizer():
    def __init__(self, root, wavelength):
        self.root = root
        #self.year = year
        self.wavelength = wavelength

    def process_single_file(self, fits_file):
        """Process a single FITS file to create a resampled version."""
        try:
            # Create and resample the map
            mag_map = sunpy.map.Map(f"{self.root}/{self.wavelength}/{fits_file}")
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
            with fits.open(f"{self.root}/{self.wavelength}/{fits_file}") as hdul:
                original_header = hdul[1].header
                print("ORIGINAL HEADER: ", original_header)

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
            output_path = f"{self.root}/resampled_mag/{fits_file}"
            compressed_resampled_map = CompImageHDU(
                data=resampled_map.data,
                header=new_header,
                compression_type='HCOMPRESS_1',
                quantize_level=16.0
            )
            compressed_hdul = HDUList([PrimaryHDU(), compressed_resampled_map])
            compressed_hdul.writeto(output_path, overwrite=True)
            return f"Successfully processed {fits_file}"

        except ZeroDivisionError as e:
            return f"Error processing {fits_file}: {str(e)}"


    def main(self):
        # Get list of FITS files
        fits_files = sorted([f for f in os.listdir(f"{self.root}/{self.wavelength}")
                            if f.endswith(".fits")])

        # Create output directory if it doesn't exist
        os.makedirs(f"{self.root}/resampled_mag", exist_ok=True)

        # Process files in parallel
        num_processes = 8
        with Pool(processes=num_processes) as pool:
            results = pool.map(self.process_single_file, fits_files)

        # Print results
        for result in results:
            print(result)

    def view_fits(self, fits_file):
        """View a FITS file."""
        resampled = f"{self.root}/resampled_mag/{fits_file}"
        mag_map = sunpy.map.Map(resampled)
        print(mag_map.data.shape)
        norm = ImageNormalize(vmin=-100, vmax=100, stretch=LinearStretch())
        fig = plt.figure()
        ax = plt.subplot(projection=mag_map.wcs)
        ax.imshow(mag_map.data, cmap='gray', norm=norm)
        ax.grid(False)
        plt.show()


if __name__ == '__main__':
    root = r"data\2016_SPoCA_CH_comparision\aia"
    #year = r"2024_aia_fits"
    wavelength = 6173

    resizer = MagResizer(root, wavelength)
    resizer.main()

    resampled_files = sorted([f for f in os.listdir(f"{root}/resampled_mag")
                              if f.endswith(".fits")])
    resizer.view_fits(resampled_files[2])
