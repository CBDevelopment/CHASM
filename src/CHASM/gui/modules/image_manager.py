import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image, ImageTk
from .threaded_image_loader import ThreadedImageLoader


class ImageManager:
    """Manages image loading, scaling, and preprocessing."""

    def __init__(
        self,
        load_dir: Any,
        masks_dir: Any,
        save_dir: str,
        max_width: int = 800,
        max_height: int = 600,
    ) -> None:
        self.load_dir = load_dir
        self.masks_dir = masks_dir
        self.save_dir = save_dir
        self.max_width = max_width
        self.max_height = max_height
        self.min_width = 600

        self.idx: Optional[int] = None
        self.image: Optional[np.ndarray] = None
        self.scale_factor: Optional[float] = None
        self.scaled_image: Optional[np.ndarray] = None
        self.tk_image: Optional[ImageTk.PhotoImage] = None

        # Add the threaded loader
        self.image_loader = ThreadedImageLoader(
            self.load_dir, self.masks_dir, self.save_dir, queue_size=1
        )

    def get_next_image(self) -> Image.Image:
        """Load the next image from the directory."""
        file_path = self.load_dir.filenames()[self.idx]
        if file_path.endswith("README.md"):
            self.idx += 1
            file_path = self.load_dir.filenames()[self.idx]

        img = Image.open(file_path)
        return img

    def load_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Load and scale an image for display."""
        self.image = image
        self.scale_factor = min(
            self.max_width / image.shape[1], self.max_height / image.shape[0], 1.0
        )
        scaled_width = int(self.image.shape[1] * self.scale_factor)
        scaled_height = int(self.image.shape[0] * self.scale_factor)
        self.scaled_image = cv2.resize(
            image, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA
        )
        self.tk_image = ImageTk.PhotoImage(Image.fromarray(self.scaled_image))

        return {
            "scaled_image": self.scaled_image,
            "tk_image": self.tk_image,
            "width": scaled_width,
            "height": scaled_height,
            "scale_factor": self.scale_factor,
        }

    def get_next_preloaded_package(self) -> Optional[Dict[str, Any]]:
        """Get the next preloaded image package from the threaded loader."""
        return self.image_loader.get_next_package()

    def start_loading(self, scale_factor: float, idx: int) -> None:
        """Start the background image loading thread."""
        self.image_loader.start_loading(scale_factor, idx)

    def stop_loading(self) -> None:
        """Stop the background image loading thread."""
        self.image_loader.stop_loading()

    def inc_idx(self) -> None:
        """Increment the image index."""
        self.idx += 1

    def dec_idx(self) -> None:
        """Decrement the image index."""
        self.idx -= 1

    def get_current_filename(self) -> Optional[str]:
        """Get the current image filename."""
        if self.idx is not None:
            return self.load_dir.filenames()[self.idx]
        return None

    def should_skip_current_file(self) -> bool:
        """Check if the current file should be skipped (already processed)."""
        if self.idx is None:
            return False
        file_path = Path(self.load_dir.filenames()[self.idx])
        basename = file_path.name
        save_path = Path(self.save_dir) / f"{basename.split('.')[0][:-4]}.npz"
        return save_path.exists()
