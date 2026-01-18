import numpy as np
import cv2
from tkinter import Canvas
from typing import List, Dict, Any, Optional


class AnnotationManager:
    """Manages coronal hole annotations including saving, removing, and validation."""

    def __init__(self) -> None:
        self.saved_masks: List[Dict[str, Any]] = []

    def add_annotation(
        self,
        mask_region: Dict[str, Any],
        coronal_hole_id: str,
        confidence: str,
        polarity: str,
        flagged: bool,
    ) -> None:
        """Add a new coronal hole annotation."""
        self.saved_masks.append(
            {
                "mask_region": mask_region,
                "id": coronal_hole_id,
                "confidence": confidence,
                "polarity": polarity,
                "flagged": flagged,
            }
        )
        print(
            f"Added Coronal Hole: ID={coronal_hole_id}, Confidence={confidence}, Polarity={polarity}"
        )

    def remove_annotation(self, idx: int) -> Optional[Dict[str, Any]]:
        """Remove an annotation by index."""
        if 0 <= idx < len(self.saved_masks):
            removed_ch = self.saved_masks.pop(idx)
            print(f"Removed Coronal Hole {removed_ch}")
            return removed_ch
        return None

    def get_annotation_count(self) -> int:
        """Get the number of saved annotations."""
        return len(self.saved_masks)

    def clear_annotations(self) -> None:
        """Clear all saved annotations."""
        self.saved_masks = []

    def get_annotations(self) -> List[Dict[str, Any]]:
        """Get all saved annotations."""
        return self.saved_masks

    def create_CH_polygon(self, mask: np.ndarray, canvas: Canvas) -> None:
        """Create a polygon representation of a coronal hole mask on a canvas."""
        # Convert mask to binary
        binary_mask = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            print("No contours found.")
            return

        # Ensure canvas dimensions are up-to-date
        canvas.update_idletasks()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)

        # Get bounding box for the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Calculate scaling factors for the canvas
        scale_x = canvas_width / w
        scale_y = canvas_height / h
        scale = min(scale_x, scale_y)  # Uniform scaling to fit the canvas

        # Scale and translate contour points to fit the canvas
        scaled_contour_points = [
            (
                int((point[0][0] - x) * scale + (canvas_width - w * scale) / 2),
                int((point[0][1] - y) * scale + (canvas_height - h * scale) / 2),
            )
            for point in largest_contour
        ]

        # Flatten points for canvas.create_polygon
        flattened_points = [coord for point in scaled_contour_points for coord in point]

        # Draw the polygon on the canvas
        canvas.create_polygon(flattened_points, outline="blue", fill="blue", width=2)

    def has_annotations(self) -> bool:
        """Check if there are any saved annotations."""
        return len(self.saved_masks) > 0

    def validate_annotations(self) -> bool:
        """Validate that all annotations have required fields."""
        for idx, annotation in enumerate(self.saved_masks):
            required_fields = ["mask_region", "id", "confidence", "polarity", "flagged"]
            for field in required_fields:
                if field not in annotation:
                    print(f"Warning: Annotation {idx} missing field '{field}'")
                    return False
        return True
