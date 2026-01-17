import argparse
import numpy as np
import matplotlib.pyplot as plt
import sunpy.map
from sunpy.visualization.colormaps import color_tables as ct
import matplotlib


def load_data(aia_path, blue_mask_path, red_mask_path):
    # Load the AIA FITS image
    aia_map = sunpy.map.Map(aia_path)
    aia_image = aia_map.data

    # Load masks
    mask_blue = np.load(blue_mask_path, allow_pickle=True).squeeze()
    mask_red = np.load(red_mask_path, allow_pickle=True).squeeze()

    # Check compatibility
    for m, name in [(mask_blue, "Blue mask"), (mask_red, "Red mask")]:
        if aia_image.shape != m.shape:
            raise ValueError(
                f"Shape mismatch: AIA image {aia_image.shape} vs {name} {m.shape}"
            )

    return aia_image, mask_blue, mask_red


def plot_overlays(aia_image, mask_blue, mask_red):
    cmap_193 = matplotlib.colormaps["sdoaia193"]

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(aia_image, cmap=cmap_193, origin="lower")

    masked_blue = np.ma.masked_where(mask_blue == 0, mask_blue)
    ax.imshow(masked_blue, cmap="Blues", alpha=0.4, origin="lower")

    masked_red = np.ma.masked_where(mask_red == 0, mask_red)
    ax.imshow(masked_red, cmap="Reds", alpha=0.4, origin="lower")

    fig.colorbar(im, ax=ax, label="AIA 193 Å Intensity")
    ax.set_title("AIA 193 Å with Blue + Red Masks")
    ax.set_xlabel("X pixels")
    ax.set_ylabel("Y pixels")

    plt.show()


def plot_masks(mask_blue, mask_red):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Blue mask
    axes[0].imshow(mask_blue, cmap="gray", origin="lower")
    axes[0].set_title("Blue Mask (Grayscale)")
    axes[0].axis("off")

    # Red mask
    axes[1].imshow(mask_red, cmap="gray", origin="lower")
    axes[1].set_title("Red Mask (Grayscale)")
    axes[1].axis("off")

    # Subtraction
    subtraction = mask_blue.astype(float) - mask_red.astype(float)
    axes[2].imshow(subtraction, cmap="coolwarm", origin="lower")
    axes[2].set_title("Blue - Red Mask (Coolwarm)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("overlay.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Overlay AIA image with prediction masks"
    )
    parser.add_argument("--aia", required=True, help="Path to AIA FITS file")
    parser.add_argument("--blue", required=True, help="Path to blue mask .npy file")
    parser.add_argument("--red", required=True, help="Path to red mask .npy file")
    args = parser.parse_args()

    aia_image, mask_blue, mask_red = load_data(args.aia, args.blue, args.red)
    plot_overlays(aia_image, mask_blue, mask_red)
    plot_masks(mask_blue, mask_red)


if __name__ == "__main__":
    main()
