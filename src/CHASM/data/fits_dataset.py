import cv2
from pathlib import Path
from sunpy.map import Map as sunpyMap
from astropy.io import fits
import numpy as np
from tqdm import tqdm
import torch
from .chasm_dataset import CHASMDataset
from .drawings_dataset import DrawingsDataset
from .folder_utils import extract_date, get_file_by_date


class Circle:
    def __init__(self, center_x, center_y, radius):
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius

    def __str__(self):
        return (
            f"Circle: Center=({self.center_x}, {self.center_y}), Radius={self.radius}"
        )


class FITSDataset:
    def __init__(self, root="download_data/fits", min_radius=900, max_radius=1100):
        self.root = root
        self.min_radius = min_radius
        self.max_radius = max_radius

    def hough_circles(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply a median blur to reduce noise
        blurred = cv2.medianBlur(gray, 5)

        # Use HoughCircles to detect circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=self.min_radius,
            param1=40,
            param2=30,
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )

        return circles

    def select_best_hough_circle(self, circles, image):
        # Calculate the image center (width / 2, height / 2)
        image_center = (image.shape[1] / 2, image.shape[0] / 2)
        best_circle = None
        min_distance = float("inf")

        # Ensure circles are valid and loop over them
        for circle in circles[0, :]:
            circle_center = (circle[0], circle[1])  # (x, y)
            distance_to_center = np.sqrt(
                (circle_center[0] - image_center[0]) ** 2
                + (circle_center[1] - image_center[1]) ** 2
            )
            # Check for the circle closest to the image center
            if distance_to_center < min_distance:
                min_distance = distance_to_center
                best_circle = circle

        best_circle = Circle(
            int(best_circle[0]), int(best_circle[1]), int(best_circle[2])
        )
        return best_circle

    def crop_masks_to_circle(self, mask_file, circle: Circle):
        cropped_masks = []

        for mask in mask_file["info"]:
            if mask["mask_region"] is None:
                # Create blank mask with same dimensions as circle region
                blank_mask = np.zeros(
                    (2 * circle.radius, 2 * circle.radius), dtype=np.uint8
                )
                cropped_masks.append(blank_mask)
            else:
                segmentation_mask = mask["mask_region"]["segmentation"]
                cropped_mask = segmentation_mask[
                    circle.center_y - circle.radius : circle.center_y + circle.radius,
                    circle.center_x - circle.radius : circle.center_x + circle.radius,
                ]
                cropped_masks.append(cropped_mask)

        return cropped_masks

    def _to_fits(
        self, image_filename: str, chasm: CHASMDataset, solar_r=959.78, resolution=4096
    ):
        image_file = cv2.imread(image_filename)
        circles = self.hough_circles(image_file)
        best_circle = self.select_best_hough_circle(circles, image_file)

        # TODO make sure this works with new naming convention, and new dataset add get files names as signature
        formatted_date = extract_date(image_filename)

        # mask_file = np.load(os.path.join(self.mask_dir, mask_filename), allow_pickle=True)
        # NOTE: kind of slow, but is more flexible in matching files by dates, no exact name match needed

        # TODO make this work with a list of file names from the dataset objects (add that to abstract type)
        mask_filename = get_file_by_date(image_filename, chasm.filenames())
        if mask_filename == None:
            print(f"Was not able to find a matching mask filename for {image_filename}")
            return

        mask_file = np.load(mask_filename, allow_pickle=True)

        cropped_masks = self.crop_masks_to_circle(mask_file, best_circle)
        cropped_masks = np.array(cropped_masks, dtype=np.uint8)
        # Merge all masks into one
        merged_mask = np.zeros_like(cropped_masks[0], dtype=np.uint8)
        for mask in cropped_masks:
            merged_mask = cv2.bitwise_or(merged_mask, mask)
        cropped_masks = merged_mask
        # Flip the cropped masks vertically
        cropped_masks = np.flipud(cropped_masks)

        # calculate mask pixel scale to solar disk pixel scale
        # sdo_pixel_scale = 0.6  # arcseconds / pixel, each pixel in an SDO image represents 0.6 arcseconds
        sdo_pixel_scale = (2.2 * solar_r) / resolution
        solar_radius = (
            solar_r  # arcseconds, this is variable based on the oribit of the SDO
        )

        # Calculate solar radius in pixels in an SDO image
        sdo_solar_radius_pixels = solar_radius / sdo_pixel_scale  # pixels

        # Scale the cropped masks to the expected disk size in an SDO image
        cropped_masks = cv2.resize(
            cropped_masks,
            (int(sdo_solar_radius_pixels) * 2, int(sdo_solar_radius_pixels) * 2),
            interpolation=cv2.INTER_NEAREST,
        )

        blank_image = np.zeros((resolution, resolution), dtype=np.uint8)
        blank_image_centerx, blank_image_centery = (
            blank_image.shape[1] // 2,
            blank_image.shape[0] // 2,
        )
        masks_centerx, masks_centery = (
            cropped_masks.shape[1] // 2,
            cropped_masks.shape[0] // 2,
        )

        blank_image[
            blank_image_centery - masks_centery : blank_image_centery + masks_centery,
            blank_image_centerx - masks_centerx : blank_image_centerx + masks_centerx,
        ] = cropped_masks
        cropped_masks = blank_image

        # Create a header for the FITS file
        header = {
            "SIMPLE": True,
            "BITPIX": -32,  # Floating point
            "NAXIS": 2,
            "NAXIS1": cropped_masks.shape[1],
            "NAXIS2": cropped_masks.shape[0],
            "TELESCOP": "CHASM",
            "INSTRUME": "CHASM",
            "DATE-OBS": formatted_date,
            "WAVELNTH": 193,
            "WAVEUNIT": "Angstrom",
            # Helioprojective coordinates
            "CRPIX1": cropped_masks.shape[1] / 2,  # Reference pixel X
            "CRPIX2": cropped_masks.shape[0] / 2,  # Reference pixel Y
            "CRVAL1": 0.0,  # Solar X coordinate
            "CRVAL2": 0.0,  # Solar Y coordinate
            "CDELT1": sdo_pixel_scale,
            "CDELT2": sdo_pixel_scale,
            "CUNIT1": "arcsec",
            "CUNIT2": "arcsec",
        }

        # Create a Sunpy map
        sunpy_map = sunpyMap((cropped_masks, header))

        # Save as FITS file
        output_filename = f"{Path(image_filename).stem}.fits"
        output_filepath = Path(self.root) / output_filename
        sunpy_map.save(str(output_filepath), overwrite=True)

        return sunpy_map

    def build(self, drawings: DrawingsDataset, chasm: CHASMDataset):
        def crop_masks(drawings: DrawingsDataset, chasm: CHASMDataset, year=None):
            # TODO implement filenames in aia
            image_files = drawings.filenames()

            for image_file in tqdm(image_files, desc="Cropping Image Files in Folder"):
                if image_file.endswith(
                    ".jpg"
                ):  # Ignores README.md selection_times.csv, etc.
                    try:
                        self._to_fits(image_file, chasm)
                    except Exception as e:
                        print(f"Failed to process {image_file}, {e}")

        # TODO implement me
        root_path = Path(self.root)
        if not root_path.exists():
            root_path.mkdir(parents=True, exist_ok=True)
            crop_masks(drawings, chasm)

        else:
            print(
                f"Root {self.root} already exists. Cannot build to existing directory. Exiting"
            )

    def filenames(self) -> list[str]:
        return sorted((str(p) for p in Path(self.root).rglob("*") if p.is_file()))

    def __len__(self):
        return len(self.filenames())

    def __getitem__(self, idx):
        file_path = self.filenames()[idx]

        data = fits.open(file_path, allow_pickle=True)
        return data
