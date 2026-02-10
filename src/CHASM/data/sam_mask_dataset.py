import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from PIL import Image

from .auto_download_dataset import AutoDownloadDataset

logger = logging.getLogger(__name__)


class SAMMaskDataset(AutoDownloadDataset):
    def __init__(self, root="download_data/masks", fetch_online=False):
        self.root = root
        if fetch_online:
            file_id = "1Gf5X-o6KPX4IRyWcAqcWBVbMdj4lYc7b"
            url = f"https://drive.google.com/uc?id={file_id}"
            super().__init__(
                root=root,
                filename="masks.tar.gz",
                url=url,
            )
        else:
            super().__init__(root=root)

    def filenames(self) -> list[str]:
        return sorted((str(p) for p in Path(self.root).rglob("*") if p.is_file()))

    @staticmethod
    def show_segmentations(
        segmentations, drawing_path: Path, area_min: int = 5000, area_max: int = 1000000
    ):
        image = Image.open(str(drawing_path))
        logger.info(f"Showing segmentations with area range: {area_min} - {area_max}")

        if len(segmentations) == 0:
            return

        filtered_segmentations = [
            segmentation
            for segmentation in segmentations
            if segmentation["area"] > area_min and segmentation["area"] < area_max
        ]
        sorted_segmentations = sorted(
            filtered_segmentations, key=(lambda x: x["area"]), reverse=True
        )

        if len(sorted_segmentations) == 0:
            logger.info("No segmentations found in the specified area range.")
            return

        logger.info(f"Number of filtered segmentations: {len(sorted_segmentations)}")

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.set_autoscale_on(False)

        # Show the original image
        ax.imshow(image)

        # Draw masks using contours
        import cv2

        for i, seg in enumerate(sorted_segmentations):
            segmentation = seg["segmentation"]

            # Find contours using OpenCV
            contours, _ = cv2.findContours(
                segmentation.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            # Generate random color
            color = (
                np.random.randint(0, 255, size=3) / 255.0
            )  # Normalize to 0-1 for matplotlib

            # Draw each contour as a polygon
            for contour in contours:
                # Reshape contour points from (n, 1, 2) to (n, 2)
                contour_points = contour.squeeze()

                if len(contour_points) > 2:  # Need at least 3 points for a polygon
                    # Note: contour points are (x, y) which is (col, row)
                    polygon = plt.Polygon(
                        contour_points,
                        fill=True,
                        facecolor=(*color, 0.4),  # 40% opacity
                        edgecolor=color,
                        linewidth=2,
                    )
                    ax.add_patch(polygon)

        ax.set_xlim(0, image.size[0])
        ax.set_ylim(image.size[1], 0)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        return file_path
