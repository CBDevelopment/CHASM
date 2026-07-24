import argparse
from pathlib import Path


from .sdo_dataset import SDODataset
from .drawings_dataset import DrawingsDataset
from .sam_mask_dataset import SAMMaskDataset
from .chasm_dataset import CHASMDataset


def download_aia_imagery(root="download_data"):
    """
    Download AIA imagery to the specified root directory.

    Args:
        root (str): Root directory for downloads
    """
    download_path = Path(root) / "aia_wavelengths"
    print(f"Downloading AIA imagery to {download_path}")

    aia_dataset = SDODataset(root=download_path, fetch_online=True, resize_mag=True)


def download_swpc_synoptic_drawings(root="download_data"):
    """
    Download SWPC synoptic drawings to the specified root directory.

    Args:
        root (str): Root directory for downloads
    """
    download_path = Path(root) / "drawings"
    print(f"Downloading SWPC synoptic drawings to {download_path}")

    drawings_dataset = DrawingsDataset(root=download_path, fetch_online=True)


def download_SAM_masks(root="download_data"):
    """
    Download SAM masks to the specified root directory.

    Args:
        root (str): Root directory for downloads
    """
    download_path = Path(root) / "masks"
    print(f"Downloading SAM masks to {download_path}")

    sam_dataset = SAMMaskDataset(root=download_path, fetch_online=True)


def download_CHASM_tool_selections(root="download_data"):
    """
    Download CHASM tool selections to the specified root directory.

    Args:
        root (str): Root directory for downloads
    """
    download_path = Path(root) / "chasm"
    print(f"Downloading CHASM tool selections to {download_path}")

    chasm_dataset = CHASMDataset(root=download_path, fetch_online=True)


def main():
    parser = argparse.ArgumentParser(
        description="Download CHASM datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--root",
        type=str,
        default="download_data",
        help="Root directory for downloads (default: download_data)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["aia", "drawings", "sam", "chasm", "all"],
        default="all",
        help="Which dataset to download (default: all)",
    )

    args = parser.parse_args()

    if args.dataset == "all" or args.dataset == "aia":
        download_aia_imagery(args.root)

    if args.dataset == "all" or args.dataset == "drawings":
        download_swpc_synoptic_drawings(args.root)

    if args.dataset == "all" or args.dataset == "sam":
        download_SAM_masks(args.root)

    if args.dataset == "all" or args.dataset == "chasm":
        download_CHASM_tool_selections(args.root)


if __name__ == "__main__":
    main()
