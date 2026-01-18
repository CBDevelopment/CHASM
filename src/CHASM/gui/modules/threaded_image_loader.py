import numpy as np
import cv2
from pathlib import Path

# Requires the Pillow library for handling image files
from PIL import Image, ImageTk
from queue import Queue
from threading import Thread, Lock
import time
from chasm.datasets.folder_utils import get_file_by_date


# TODO make this take datasets
class ThreadedImageLoader:
    def __init__(self, load_dir, masks_dir, save_dir, queue_size=3):
        self.load_dir = load_dir
        self.masks_dir = masks_dir
        self.save_dir = save_dir
        self.preload_queue = Queue(maxsize=queue_size)
        self.current_idx = None
        self.loading_lock = Lock()
        self.should_stop = False
        self.loader_thread = None

    def start_loading(self, scale_factor, start_idx=0):
        """Start the background loading thread"""
        self.should_stop = False
        self.current_idx = start_idx
        self.loader_thread = Thread(
            target=self._background_loader, args=(scale_factor,), daemon=True
        )
        self.loader_thread.start()

    def stop_loading(self):
        """Stop the background loading thread"""
        self.should_stop = True
        if self.loader_thread:
            self.loader_thread.join()
        with self.loading_lock:
            while not self.preload_queue.empty():
                try:
                    self.preload_queue.get_nowait()
                except Queue.Empty:
                    break

    def get_matching_sam_match(self, next_idx):
        print(self.load_dir.filenames()[next_idx])
        print(self.masks_dir.filenames())
        mask_file = get_file_by_date(
            self.load_dir.filenames()[next_idx], self.masks_dir.filenames()
        )
        return np.load(mask_file, allow_pickle=True)

    def _background_loader(self, scale_factor):
        """Background thread that continuously loads and processes images"""
        while not self.should_stop:
            if self.preload_queue.full():
                time.sleep(0.1)  # Don't busy-wait if queue is full
                continue

            try:
                with self.loading_lock:
                    next_idx = self.current_idx + 1

                    # TODO make this less brittle
                    while True:
                        file_path = Path(self.load_dir.filenames()[next_idx])
                        save_path = Path(self.save_dir) / f"DATA-{file_path.stem}.npz"
                        if not save_path.exists():
                            break
                        next_idx += 1
                    self.current_idx = next_idx

                    # Skip README.md
                    while self.load_dir.filenames()[next_idx].endswith("README.md"):
                        next_idx += 1

                    # Load and process image
                    file_path = self.load_dir.filenames()[next_idx]
                    img = Image.open(file_path)
                    img_array = np.array(img)

                    # Scale image
                    scaled_width = int(img_array.shape[1] * scale_factor)
                    scaled_height = int(img_array.shape[0] * scale_factor)
                    scaled_image = cv2.resize(
                        img_array,
                        (scaled_width, scaled_height),
                        interpolation=cv2.INTER_AREA,
                    )
                    tk_image = ImageTk.PhotoImage(Image.fromarray(scaled_image))

                    # Load and process masks
                    masks = self.get_matching_sam_match(next_idx)
                    processed_masks = [
                        masks[arr_str].item()
                        for arr_str in masks
                        if 5000 <= masks[arr_str].item()["area"] <= 1000000
                    ]

                    # Pre-process polygon points for each mask
                    polygon_data = []
                    for mask in processed_masks:
                        segmentation = mask["segmentation"]
                        scaled_segmentation = cv2.resize(
                            segmentation.astype(np.uint8),
                            (scaled_image.shape[1], scaled_image.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )

                        contours, _ = cv2.findContours(
                            scaled_segmentation,
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )

                        mask_polygons = []
                        for contour in contours:
                            points = [(point[0][0], point[0][1]) for point in contour]
                            mask_polygons.append(points)

                        polygon_data.append(
                            {"original_mask": mask, "polygons": mask_polygons}
                        )

                    # Package everything needed for display
                    preload_package = {
                        "idx": next_idx,
                        "file_name": file_name,
                        "tk_image": tk_image,
                        "scaled_image": scaled_image,
                        "polygon_data": polygon_data,
                        "width": scaled_width,
                        "height": scaled_height,
                    }

                    self.preload_queue.put(preload_package)

            except ZeroDivisionError as e:
                print(f"Error in background loader: {e}")
                time.sleep(0.1)

    def get_next_package(self):
        """Get the next preloaded package, returns None if none available"""
        try:
            return self.preload_queue.get_nowait()
        except:
            return None
