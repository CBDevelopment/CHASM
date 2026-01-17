import os
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sunpy.visualization.colormaps as smaps


# Define AIA 193 Å colormap (from SunPy)
def get_aia193_cmap():
    try:
        cmap = matplotlib.colormaps['sdoaia193']
    except KeyError:
        from astropy import units as u
        cmap = smaps.color_tables.aia_color_table(193 * u.AA)
    return cmap


def _load_data(input_):
    """
    Load data from FITS filename, NPY filename, or numpy array.
    Returns (data, label).
    """
    if isinstance(input_, str):
        ext = os.path.splitext(input_)[1].lower()
        if ext in [".fits", ".fit", ".fts", ".gz"]:
            with fits.open(input_) as hdul:
                if ext==".gz":
                    data = hdul[0].data.astype(float)
                else:
                    data = hdul[1].data.astype(float)
            label = os.path.basename(input_)
            type = "fits"
        elif ext == ".npy":
            data = np.load(input_).astype(float)
            label = os.path.basename(input_)
            if data.shape==(512, 512):
                type = "mask"
            else:
                type = "npy"
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    elif isinstance(input_, np.ndarray):
        data = input_.astype(float)
        label = "NumPy array"
        type = "npy"
    else:
        raise TypeError("Input must be a FITS file path, .npy file path, or numpy.ndarray")

    return data, label, type


def display_two_fits(file1, file2):
    data1, label1, type1, = _load_data(file1)
    data2, label2, type2 = _load_data(file2)

    if data1 is None or data2 is None:
        print("One of the inputs has no image data.")
        return

    print(f"{label1} size: {data1.shape}")
    print(f"{label2} size: {data2.shape}")

    # Normalize both datasets to the same scale
    vmin = min(np.nanmin(data1), np.nanmin(data2))
    vmax = max(np.nanmax(data1), np.nanmax(data2))
    print("VMIN & VMAV ", vmin, vmax)
    cmap_aia = get_aia193_cmap()


    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    if type1=="fits" or type1=="mask":
        im1 = axes[0].imshow(data1, cmap=cmap_aia, origin='lower', vmin=vmin, vmax=vmax)
        axes[0].set_title(label1)
        
    elif type1=="npy":
        im1 = axes[0].imshow(data1[:, :, 3], cmap=cmap_aia, origin='lower', vmin=vmin, vmax=vmax)
        axes[0].set_title(label1)

    if type2=="fits" or type2=="mask":
        im2 = axes[1].imshow(data2, cmap=cmap_aia, origin='lower', vmin=vmin, vmax=vmax)
        axes[1].set_title(label2)

    elif type2=="npy":
        im2 = axes[1].imshow(data2[:, :, 3], cmap=cmap_aia, origin='lower', vmin=vmin, vmax=vmax)
        axes[1].set_title(label2)

    im3 = None
    if type1=="fits" or type1=="mask" and type2=="fits" or type2=="mask":
        # Compute difference
        diff = data1 - data2
        diff_abs = np.nanmax(np.abs(diff))
        im3 = axes[2].imshow(diff, cmap='RdBu_r', origin='lower', vmin=-diff_abs, vmax=diff_abs)
        axes[2].set_title("Difference (File1 - File2)")

    if type1=="npy" and type2=="npy":
        diff = data1 - data2
        diff_abs = np.nanmax(np.abs(diff))
        im3 = axes[2].imshow(diff[:, :, 3], cmap='RdBu_r', origin='lower', vmin=-diff_abs, vmax=diff_abs)
        axes[2].set_title("Difference (File1 - File2)")

    # Colorbars
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    if im3:
        fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle("AIA 193 Å Comparison + Difference")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # NOTE: TO debug mismatch in new pipelien inputs for the maps. It's quite slight but causes a slightly worse result
    date = "2017-01-12"
    file1 = rf"D:\CHASM_TABLES\CHASM-1110-PredictionsFromDrive\{date}.npy"
    file2 = rf"D:\CHASM_TABLES\TESTING_NEW_PIPELINE\chasm1111_predictions\predictions{date}.npy"

    display_two_fits(file1, file2)
