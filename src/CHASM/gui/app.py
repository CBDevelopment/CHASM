import tkinter as tk
import numpy as np
from time import perf_counter
from pathlib import Path

import argparse

from chasm.gui.modules.mask_manager import MaskManager
from chasm.gui.modules.image_manager import ImageManager
from chasm.gui.modules.data_persistence import DataPersistenceManager
from chasm.gui.modules.annotation_manager import AnnotationManager
from chasm.gui.modules.gui_builder import GUIComponentBuilder
from chasm.gui.modules.event_handler import EventHandler


# TODO have app save in JUST .npz format, having the .csv is kind of confusing
class CHASM_GUI:
    """Main coordinator for the Coronal Hole Annotation application."""

    def __init__(
        self,
        load_dir,
        masks_dir,
        save_dir,
        max_width=800,
        max_height=600,
        scale_factor=1,
    ):
        # Timing for performance tracking
        self.selection_start_time = None
        self.selection_end_time = None

        # Initialize managers
        self.image_manager = ImageManager(
            load_dir, masks_dir, save_dir, max_width, max_height
        )
        self.mask_manager = MaskManager()
        self.data_manager = DataPersistenceManager(save_dir, load_dir, masks_dir)
        self.annotation_manager = AnnotationManager()

        # Create root window
        self.root = tk.Tk()

        # Build GUI
        self.gui_builder = GUIComponentBuilder(
            self.root, max_width, max_height, scale_factor
        )
        self.gui_builder.build_gui(
            on_previous_drawing=self.previous_drawing,
            on_next_drawing=self.next_drawing,
            on_save_coronal_hole=self.save_coronal_hole,
            on_merge_select=self.merge_select,
            on_merge_selected_masks=self.merge_selected_masks,
        )

        # Setup event handling
        self.event_handler = EventHandler(
            self.root,
            self.gui_builder.image_canvas,
            self.mask_manager,
            self.annotation_manager,
            self.gui_builder,
        )
        self.event_handler.set_callbacks(
            on_next_drawing=self.next_drawing,
            on_previous_drawing=self.previous_drawing,
            on_save_coronal_hole=self.save_coronal_hole,
            on_redraw_masks=self.redraw_masks,
            on_merge_select=self.merge_select,
            on_subtract_selected=self.subtract_selected_masks,
        )
        self.event_handler.bind_keys()

        # Add window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.destroy)

        # Update the initial display
        self.gui_builder.update_coronal_hole_display(
            self.annotation_manager.get_annotations(),
            self.remove_coronal_hole,
            self.annotation_manager.create_CH_polygon,
        )

        # Run the Tkinter event loop
        self.root.mainloop()

    def load_image(self, image):
        """Load and display an image with masks."""
        result = self.image_manager.load_image(image)

        # Update canvas
        self.gui_builder.image_canvas.config(
            width=result["width"], height=result["height"]
        )
        self.gui_builder.image_canvas.delete("all")
        self.gui_builder.image_canvas.create_image(
            0, 0, anchor="nw", image=result["tk_image"]
        )

        # Load and draw masks
        print("LOADING MASKS")
        current_filename = self.image_manager.get_current_filename()
        mask_file = self.data_manager.get_matching_sam_mask(current_filename)
        self.mask_manager.load_masks(mask_file)
        self.mask_manager.draw_masks(
            self.gui_builder.image_canvas, self.image_manager.scaled_image.shape
        )

    def redraw_masks(self):
        """Redraw all masks on the canvas."""
        self.mask_manager.draw_masks(
            self.gui_builder.image_canvas, self.image_manager.scaled_image.shape
        )

    def save_CH_masks(self):
        """Save coronal hole masks if there are annotations or 'no coronal holes' is checked."""
        if self.image_manager.idx is not None:
            if (
                self.gui_builder.no_coronal_holes.get()
                or self.annotation_manager.has_annotations()
            ):
                current_filename = self.image_manager.get_current_filename()
                detected_chs = self.gui_builder.detected_chs_entry.get()
                true_chs = self.gui_builder.true_chs_entry.get()
                self.data_manager.save_current_drawing(
                    current_filename,
                    self.annotation_manager.get_annotations(),
                    detected_chs,
                    true_chs,
                )

                # Save selection time
                if self.selection_start_time is not None:
                    self.selection_end_time = perf_counter()
                    selection_time = self.selection_end_time - self.selection_start_time
                    self.data_manager.save_selection_time(
                        current_filename, selection_time
                    )

    def update_coronal_hole_display(self):
        """Update the display of saved coronal holes."""
        self.gui_builder.update_coronal_hole_display(
            self.annotation_manager.get_annotations(),
            self.remove_coronal_hole,
            self.annotation_manager.create_CH_polygon,
        )

    def remove_coronal_hole(self, idx):
        """Remove a coronal hole annotation by index."""
        self.annotation_manager.remove_annotation(idx)
        self.update_coronal_hole_display()

    def next_drawing(self):
        """Load the next drawing, using preloaded data when available."""
        self.save_CH_masks()

        # Get next preloaded package
        self.selection_start_time = perf_counter()
        next_package = self.image_manager.get_next_preloaded_package()

        if next_package is None:
            print("No preloaded data available, loading synchronously...")
            self.image_manager.idx = (
                self.image_manager.idx + 1 if self.image_manager.idx is not None else 0
            )

            # Skip already processed files
            while self.image_manager.should_skip_current_file():
                self.image_manager.inc_idx()

            self.load_image(np.array(self.image_manager.get_next_image()))
            # Start the background loader
            self.image_manager.start_loading(
                self.image_manager.scale_factor, self.image_manager.idx
            )
        else:
            # Update current index and apply preloaded data
            self.image_manager.idx = next_package["idx"]
            self.image_manager.scaled_image = next_package["scaled_image"]
            self.image_manager.tk_image = next_package["tk_image"]

            # Update canvas size and draw image
            self.gui_builder.image_canvas.config(
                width=next_package["width"], height=next_package["height"]
            )
            self.gui_builder.image_canvas.delete("all")
            self.gui_builder.image_canvas.create_image(
                0, 0, anchor="nw", image=self.image_manager.tk_image
            )

            # Update masks and draw using pre-calculated polygons
            self.mask_manager.masks = [
                data["original_mask"] for data in next_package["polygon_data"]
            ]
            self.mask_manager.mask_map = {}

            for i, mask_data in enumerate(next_package["polygon_data"]):
                for polygon in mask_data["polygons"]:
                    color = "#{:02x}{:02x}{:02x}".format(
                        *np.random.randint(0, 255, size=3)
                    )
                    poly_id = self.gui_builder.image_canvas.create_polygon(
                        *[coord for point in polygon for coord in point],
                        fill=color,
                        outline=color,
                    )
                    self.mask_manager.mask_map[poly_id] = i

        self.annotation_manager.clear_annotations()
        self.update_coronal_hole_display()

    def previous_drawing(self):
        """Load the previous drawing."""
        print("Stopping loader thread")
        self.image_manager.stop_loading()
        self.image_manager.dec_idx()
        # Start the background loader
        self.image_manager.start_loading(
            self.image_manager.scale_factor, self.image_manager.idx
        )
        print("Loading previous drawing")
        self.load_image(np.array(self.image_manager.get_next_image()))
        self.update_coronal_hole_display()

    def remove_merge_item(self, current_mask, merge_frame):
        """Remove a mask from the merge selection."""
        merge_frame.destroy()
        self.mask_manager.remove_mask_from_merge(current_mask)

    def create_merge_item_row(self, current_mask):
        """Create a row in the merge UI for a selected mask."""
        mask_index = self.mask_manager.mask_map[current_mask]
        segmentation = self.mask_manager.masks[mask_index]["segmentation"]

        self.gui_builder.create_merge_item_row(
            current_mask,
            lambda mask_id, canvas: self.annotation_manager.create_CH_polygon(
                segmentation, canvas
            ),
            self.remove_merge_item,
        )

    def merge_select(self):
        """Select a mask for merging."""
        if not self.mask_manager.clicked_masks:
            return

        current_mask = self.mask_manager.clicked_masks[
            self.mask_manager.clicked_masks_idx
        ]
        if self.mask_manager.add_mask_to_merge(current_mask):
            self.create_merge_item_row(current_mask)

    def merge_selected_masks(self):
        """Merge all selected masks into one."""
        success = self.mask_manager.merge_selected_masks(self.gui_builder.image_canvas)
        if success:
            self.mask_manager.reset_selection(self.gui_builder.image_canvas)
            self.redraw_masks()
            self.gui_builder.clear_merge_rows()

    def subtract_selected_masks(self):
        """Subtract one mask from another."""
        success = self.mask_manager.subtract_selected_masks(
            self.gui_builder.image_canvas
        )
        if success:
            self.mask_manager.reset_selection(self.gui_builder.image_canvas)
            self.redraw_masks()
            self.gui_builder.clear_merge_rows()

    def save_coronal_hole(self):
        """Save the currently selected mask as a coronal hole annotation."""
        if self.mask_manager.selected_mask is None:
            print("No mask selected")
            return

        print("Saving Coronal Hole")
        coronal_hole_id = self.gui_builder.coronal_hole_id_dropdown.get()
        confidence = self.gui_builder.confidence_dropdown.get()
        polarity = self.gui_builder.polarity_dropdown.get()
        flag_button_value = bool(self.gui_builder.flag_button_value.get())

        self.annotation_manager.add_annotation(
            self.mask_manager.selected_mask,
            coronal_hole_id,
            confidence,
            polarity,
            flag_button_value,
        )
        self.update_coronal_hole_display()

    def destroy(self):
        """Clean up resources and destroy the window."""
        # Stop the background loader thread
        self.image_manager.stop_loading()
        # Save current drawing if there is one
        self.save_CH_masks()
        # Destroy the window
        self.root.destroy()


# Entry point for the GUI application
def run_app(
    drawings_dataset=None,
    sam_dataset=None,
    save_dir=None,
    max_width=1000,
    max_height=800,
    scale_factor=1,
):
    """
    Launch the CHASM GUI application.

    Can be called programmatically with dataset objects, or will parse command-line
    arguments if datasets are not provided.

    Args:
        drawings_dataset: DrawingsDataset object (optional, will parse from CLI if None)
        sam_dataset: SAMDataset object for masks (optional, will parse from CLI if None)
        save_dir: Directory to save tool selections (optional, will parse from CLI if None)
        max_width: Maximum canvas width (default: 1000)
        max_height: Maximum canvas height (default: 800)
    """
    # If datasets not provided, parse command-line arguments
    if drawings_dataset is None or sam_dataset is None or save_dir is None:
        from chasm.data import DrawingsDataset
        from chasm.data import SAMMaskDataset

        parser = argparse.ArgumentParser(
            description="CHASM GUI - Coronal Hole Annotation Tool"
        )
        parser.add_argument(
            "--drawings",
            type=str,
            default="download_data/drawings",
            help="Path to drawings dataset directory (default: download_data/drawings)",
        )
        parser.add_argument(
            "--masks",
            type=str,
            default="download_data/masks",
            help="Path to SAM masks dataset directory (default: download_data/masks)",
        )
        parser.add_argument(
            "--save-dir",
            type=str,
            default="tool_selections",
            help="Directory to save tool selections (default: tool_selections)",
        )
        parser.add_argument(
            "--download",
            action="store_true",
            help="Download datasets from online if not present (sets fetch_online=True)",
        )
        parser.add_argument(
            "--max-width",
            type=int,
            default=1000,
            help="Maximum canvas width (default: 1000)",
        )
        parser.add_argument(
            "--max-height",
            type=int,
            default=800,
            help="Maximum canvas height (default: 800)",
        )
        parser.add_argument(
            "--scale",
            type=float,
            default=1.0,
            help="GUI scale factor for buttons and text (default: 1.0)",
        )

        args = parser.parse_args()

        # Create datasets from parsed arguments
        drawings_dataset = DrawingsDataset(
            root=args.drawings, fetch_online=args.download
        )
        sam_dataset = SAMMaskDataset(root=args.masks, fetch_online=args.download)
        save_dir = args.save_dir
        max_width = args.max_width
        max_height = args.max_height
        scale_factor = args.scale

    app = CHASM_GUI(
        drawings_dataset,
        sam_dataset,
        save_dir,
        max_width=max_width,
        max_height=max_height,
        scale_factor=scale_factor,
    )


if __name__ == "__main__":
    run_app()
