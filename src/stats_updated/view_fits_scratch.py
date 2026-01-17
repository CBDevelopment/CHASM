import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

def view_fits(fname):
    # open file
    with fits.open(fname) as hdul:
        data = hdul[0].data
    
    # if there are extra axes, collapse them
    if data.ndim > 2:
        data = data[0]
    
    plt.imshow(data, origin="lower", cmap="gray")
    plt.title(f"Shape: {data.shape}")
    plt.colorbar(label="Pixel value")
    plt.show()

if __name__ == "__main__":
    view_fits(r"manual_downloaded_data_for_table\FITS\CHASM1111_small\boul_neutl_fd_20170107_0658.fits")
    view_fits(r"manual_downloaded_data_for_table\chronnos\chasm1111\chronnos_merged_inputs\cropped_centered_masks\2017-01-07.fits")