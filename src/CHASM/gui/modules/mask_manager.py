import numpy as np
import cv2
from tkinter import Canvas
from typing import List, Dict, Tuple, Optional, Any


class MaskManager:
    """Manages mask operations including loading, drawing, merging, and selection."""

    def __init__(self) -> None:
        self.masks: List[Dict[str, Any]] = []
        self.selected_mask: Optional[Dict[str, Any]] = None
        self.mask_map: Dict[int, int] = {}  # Maps polygon IDs to mask indices
        self.clicked_masks: List[int] = []
        self.clicked_masks_idx: int = 0
        self.masks_to_merge: List[int] = []

    def load_masks(
        self, mask_file: str, min_area: int = 5000, max_area: int = 1000000
    ) -> List[Dict[str, Any]]:
        """Load masks from an npz file and filter by area."""
        loaded_masks = np.load(mask_file, allow_pickle=True)
        self.masks = [
            loaded_masks[arr_str].item()
            for arr_str in loaded_masks
            if min_area <= loaded_masks[arr_str].item()["area"] <= max_area
        ]
        return self.masks

    def draw_masks(
        self, canvas: Canvas, scaled_image_shape: Tuple[int, int, int]
    ) -> None:
        """Draw all masks on the canvas with unique colors."""
        self.mask_map = {}
        for i, ann in enumerate(self.masks):
            segmentation = ann["segmentation"]
            scaled_segmentation = cv2.resize(
                segmentation.astype(np.uint8),
                (scaled_image_shape[1], scaled_image_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

            contours, _ = cv2.findContours(
                scaled_segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            color = "#{:02x}{:02x}{:02x}".format(*np.random.randint(0, 255, size=3))

            for contour in contours:
                contour_points = [(point[0][0], point[0][1]) for point in contour]
                poly_id = canvas.create_polygon(
                    *[coord for point in contour_points for coord in point],
                    fill=color,
                    outline=color,
                )
                self.mask_map[poly_id] = i

    def get_clicked_masks(self, canvas: Canvas, x: int, y: int) -> List[int]:
        """Get all masks at the clicked position."""
        self.clicked_masks_idx = 0
        self.clicked_masks = []
        clicked_items = canvas.find_overlapping(x, y, x, y)

        for item in clicked_items:
            if len(clicked_items) == 1:
                self.reset_selection(canvas)
            if item in self.mask_map:
                self.clicked_masks.append(item)

        return self.clicked_masks

    def show_mask_item(self, canvas: Canvas) -> Optional[Tuple[int, Dict[str, Any]]]:
        """Show the currently selected mask based on clicked_masks_idx."""
        if len(self.clicked_masks) == 0:
            return None

        self.reset_selection(canvas)
        item = self.clicked_masks[self.clicked_masks_idx]
        mask_index = self.mask_map[item]
        self.selected_mask = self.masks[mask_index]
        self.highlight_selected_mask(canvas, item)
        return mask_index, self.selected_mask

    def cycle_masks(self, canvas: Canvas) -> Optional[Tuple[int, Dict[str, Any]]]:
        """Cycle to the next overlapping mask."""
        self.reset_selection(canvas)
        self.clicked_masks_idx = (self.clicked_masks_idx + 1) % len(self.clicked_masks)
        return self.show_mask_item(canvas)

    def highlight_selected_mask(self, canvas: Canvas, selected_id: int) -> None:
        """Highlight the selected mask with a red outline."""
        canvas.itemconfig(selected_id, outline="red", width=2)

    def reset_selection(self, canvas: Canvas) -> None:
        """Reset all mask selections."""
        self.selected_mask = None
        for item in self.mask_map:
            canvas.itemconfig(item, outline="", fill="", width=1)

    def add_mask_to_merge(self, mask_id: int) -> bool:
        """Add a mask to the merge list."""
        if mask_id not in self.masks_to_merge:
            self.masks_to_merge.append(mask_id)
            return True
        return False

    def remove_mask_from_merge(self, mask_id: int) -> bool:
        """Remove a mask from the merge list."""
        if mask_id in self.masks_to_merge:
            self.masks_to_merge.remove(mask_id)
            return True
        return False

    def merge_masks(self, mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
        """Merge two masks using logical OR."""
        return np.logical_or(mask1, mask2)

    def subtract_masks(self, mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
        """Subtract mask2 from mask1 (mask1 - mask2)."""
        return np.bitwise_and(mask1, np.bitwise_not(mask2))

    def merge_selected_masks(self, canvas: Canvas) -> bool:
        """Merge all selected masks into one mask."""
        if len(self.masks_to_merge) < 2:
            print("Need at least two masks to merge.")
            return False

        # Merge selected masks
        merged_mask = self.masks[self.mask_map[self.masks_to_merge[0]]]["segmentation"]
        for mask_id in self.masks_to_merge[1:]:
            mask_index = self.mask_map[mask_id]
            merged_mask = self.merge_masks(
                merged_mask, self.masks[mask_index]["segmentation"]
            )

        # Track items to remove
        polygons_to_remove = list(self.masks_to_merge)
        masks_to_remove = sorted(
            [self.mask_map[mask_id] for mask_id in self.masks_to_merge],
            reverse=True,
        )

        # Remove polygons from canvas and mask_map
        for item in polygons_to_remove:
            canvas.delete(item)
            self.mask_map.pop(item)

        # Remove masks from the list
        for mask_index in masks_to_remove:
            self.masks.pop(mask_index)

        # Add the merged mask
        self.masks.append(
            {"segmentation": np.array(merged_mask), "area": np.sum(merged_mask)}
        )

        # Clear the masks_to_merge list
        self.masks_to_merge = []
        return True

    def subtract_selected_masks(self, canvas: Canvas) -> bool:
        """Subtract the second selected mask from the first."""
        if len(self.masks_to_merge) != 2:
            print("Can only subtract two masks.")
            return False

        # Get the first mask (the one from which we'll subtract)
        subtract_mask = self.masks[self.mask_map[self.masks_to_merge[0]]][
            "segmentation"
        ]

        # Subtract the remaining masks from the first mask
        for mask_id in self.masks_to_merge[1:]:
            mask_index = self.mask_map[mask_id]
            subtract_mask = self.subtract_masks(
                subtract_mask, self.masks[mask_index]["segmentation"]
            )

        # Track items to remove
        polygons_to_remove = list(self.masks_to_merge)
        masks_to_remove = sorted(
            [self.mask_map[mask_id] for mask_id in self.masks_to_merge],
            reverse=True,
        )

        # Remove polygons from canvas and mask_map
        for item in polygons_to_remove:
            canvas.delete(item)
            self.mask_map.pop(item)

        # Remove masks from the list
        for mask_index in masks_to_remove:
            self.masks.pop(mask_index)

        # Add the subtracted mask
        self.masks.append(
            {"segmentation": np.array(subtract_mask), "area": np.sum(subtract_mask)}
        )

        # Clear the masks_to_merge list
        self.masks_to_merge = []
        return True
