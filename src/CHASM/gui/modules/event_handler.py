import tkinter as tk
from tkinter import Canvas
from typing import Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .mask_manager import MaskManager
    from .annotation_manager import AnnotationManager
    from .gui_builder import GUIComponentBuilder


class EventHandler:
    """Handles keyboard and mouse events for the Coronal Hole Classifier."""

    def __init__(
        self,
        root: tk.Tk,
        canvas: Canvas,
        mask_manager: "MaskManager",
        annotation_manager: "AnnotationManager",
        gui_builder: "GUIComponentBuilder",
    ) -> None:
        self.root = root
        self.canvas = canvas
        self.mask_manager = mask_manager
        self.annotation_manager = annotation_manager
        self.gui_builder = gui_builder

        # Callback functions to be set by the main app
        self.on_next_drawing: Optional[Callable[[], None]] = None
        self.on_previous_drawing: Optional[Callable[[], None]] = None
        self.on_save_coronal_hole: Optional[Callable[[], None]] = None
        self.on_redraw_masks: Optional[Callable[[], None]] = None
        self.on_merge_select: Optional[Callable[[], None]] = None
        self.on_subtract_selected: Optional[Callable[[], None]] = None

    def set_callbacks(
        self,
        on_next_drawing: Optional[Callable[[], None]] = None,
        on_previous_drawing: Optional[Callable[[], None]] = None,
        on_save_coronal_hole: Optional[Callable[[], None]] = None,
        on_redraw_masks: Optional[Callable[[], None]] = None,
        on_merge_select: Optional[Callable[[], None]] = None,
        on_subtract_selected: Optional[Callable[[], None]] = None,
    ) -> None:
        """Set callback functions for event actions."""
        if on_next_drawing:
            self.on_next_drawing = on_next_drawing
        if on_previous_drawing:
            self.on_previous_drawing = on_previous_drawing
        if on_save_coronal_hole:
            self.on_save_coronal_hole = on_save_coronal_hole
        if on_redraw_masks:
            self.on_redraw_masks = on_redraw_masks
        if on_merge_select:
            self.on_merge_select = on_merge_select
        if on_subtract_selected:
            self.on_subtract_selected = on_subtract_selected

    def bind_keys(self) -> None:
        """Bind keyboard shortcuts."""
        self.root.bind("<space>", self.handle_space_bar)
        self.root.bind("<d>", self.handle_d_key)
        self.root.bind("<m>", self.handle_m_key)
        self.root.bind("<s>", self.handle_s_key)
        self.canvas.bind("<Button-1>", self.handle_mask_click)

    def handle_space_bar(self, event: tk.Event) -> None:
        """Handle space bar press - redraw all masks."""
        if self.on_redraw_masks:
            self.mask_manager.reset_selection(self.canvas)
            self.on_redraw_masks()

    def handle_d_key(self, event: tk.Event) -> None:
        """Handle D key press - cycle through overlapping masks."""
        print("D Pressed")
        result = self.mask_manager.cycle_masks(self.canvas)
        if result:
            mask_index, selected_mask = result
            self.gui_builder.update_info_label(
                f"Mask {mask_index} selected: Area={selected_mask['area']}"
            )
        else:
            self.gui_builder.update_info_label("None")

    def handle_m_key(self, event: tk.Event) -> None:
        """Handle M key press - select mask for merging."""
        print("M Pressed")
        if self.on_merge_select:
            self.on_merge_select()

    def handle_s_key(self, event: tk.Event) -> None:
        """Handle S key press - subtract selected masks."""
        if self.on_subtract_selected:
            self.on_subtract_selected()

    def handle_mask_click(self, event: tk.Event) -> None:
        """Handle mouse click on a mask."""
        self.mask_manager.reset_selection(self.canvas)
        self.mask_manager.get_clicked_masks(self.canvas, event.x, event.y)
        result = self.mask_manager.show_mask_item(self.canvas)

        if result:
            mask_index, selected_mask = result
            self.gui_builder.update_info_label(
                f"Mask {mask_index} selected: Area={selected_mask['area']}"
            )
        else:
            self.gui_builder.update_info_label("Click on a mask to select it.")
