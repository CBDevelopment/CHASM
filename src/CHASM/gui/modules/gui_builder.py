import tkinter as tk
from tkinter import Canvas, Button, Label
from tkinter import ttk
import tkinter.font as tkFont
from typing import List, Optional, Callable, Any, Dict


class GUIComponentBuilder:
    """Builds and manages GUI components for the Coronal Hole Classifier."""

    def __init__(
        self, root: tk.Tk, max_width: int = 800, max_height: int = 600
    ) -> None:
        self.root = root
        self.max_width = max_width
        self.max_height = max_height
        self.min_width = 600

        # GUI widgets that will be accessed by other components
        self.image_canvas: Optional[Canvas] = None
        self.detected_chs_entry: Optional[ttk.Entry] = None
        self.true_chs_entry: Optional[ttk.Entry] = None
        self.coronal_hole_id_dropdown: Optional[ttk.Entry] = None
        self.confidence_dropdown: Optional[ttk.Combobox] = None
        self.polarity_dropdown: Optional[ttk.Combobox] = None
        self.flag_button_value: Optional[tk.IntVar] = None
        self.flag_button: Optional[ttk.Checkbutton] = None
        self.no_coronal_holes: Optional[tk.BooleanVar] = None
        self.no_coronal_holes_checkbox: Optional[tk.Checkbutton] = None
        self.info_label: Optional[Label] = None
        self.ch_list: Optional[tk.Frame] = None
        self.merge_canvas: Optional[Canvas] = None
        self.merge_items: Optional[tk.Frame] = None
        self.merge_rows: List[tk.Frame] = []

        self.labelFont = tkFont.Font(family="Helvetica", size=16)

    def build_gui(
        self,
        on_previous_drawing: Callable[[], None],
        on_next_drawing: Callable[[], None],
        on_save_coronal_hole: Callable[[], None],
        on_merge_select: Callable[[], None],
        on_merge_selected_masks: Callable[[], None],
    ) -> None:
        """Build all GUI components."""
        self.root.title("Synoptic Map Viewer")
        self.root.geometry("1400x900")

        # Validation function for numeric entry
        def on_validate_input(P):
            if P == "" or P.isdigit():
                return True
            else:
                return False

        validate_input = self.root.register(on_validate_input)

        # Left panel (map display)
        left_frame = tk.Frame(self.root)
        left_frame.pack(side=tk.LEFT, padx=20, pady=10)

        # Image canvas
        self.image_canvas = Canvas(
            left_frame, width=self.min_width, height=self.max_height, bg="white"
        )
        self.image_canvas.pack()

        # Right panel
        right_frame = tk.Frame(self.root)
        right_frame.pack(side=tk.LEFT, padx=20, pady=10)

        # Title
        Label(
            right_frame, text="Coronal Hole Masks", font=("Helvetica", 20, "bold")
        ).pack(anchor="w", pady=5)

        # Drawing navigation buttons
        drawing_buttons_frame = tk.Frame(right_frame)
        drawing_buttons_frame.pack(side="top")

        previous_drawing_button = Button(
            drawing_buttons_frame,
            font=self.labelFont,
            text="Previous Drawing",
            command=on_previous_drawing,
            width=20,
        )
        previous_drawing_button.pack(side="left", padx=10, pady=20, fill="x")

        next_drawing_button = Button(
            drawing_buttons_frame,
            font=self.labelFont,
            text="Next Drawing",
            command=on_next_drawing,
            width=20,
        )
        next_drawing_button.pack(side="left", padx=10, pady=20, fill="x")

        # No coronal holes checkbox
        self.no_coronal_holes = tk.BooleanVar()
        self.no_coronal_holes_checkbox = tk.Checkbutton(
            drawing_buttons_frame,
            text="No Coronal Holes",
            variable=self.no_coronal_holes,
            font=self.labelFont,
        )
        self.no_coronal_holes_checkbox.pack(side="left", padx=10, pady=20, fill="x")

        # CH annotation controls
        ch_buttons = tk.Frame(right_frame)
        ch_buttons.pack(pady=10, anchor="w")

        # Detected CHs entry
        Label(ch_buttons, font=self.labelFont, text="Saved CHs:").grid(
            row=0, column=3, padx=5
        )
        self.detected_chs_entry = ttk.Entry(
            ch_buttons, width=5, validate="key", state="readonly"
        )
        self.detected_chs_entry.grid(row=0, column=4, padx=5)

        # True CHs entry
        Label(ch_buttons, font=self.labelFont, text="True CHs:").grid(
            row=0, column=1, padx=5
        )
        self.true_chs_entry = ttk.Entry(
            ch_buttons,
            width=5,
            validate="key",
            validatecommand=(validate_input, "%P"),
        )
        self.true_chs_entry.grid(row=0, column=2, padx=5)

        # Mask placeholder
        mask_canvas = Canvas(ch_buttons, width=100, height=50, bg="lightgray")
        mask_canvas.grid(row=1, column=0, padx=10)
        mask_canvas.create_oval(25, 10, 75, 40, outline="blue", fill="blue")

        # Coronal hole ID dropdown
        Label(ch_buttons, font=self.labelFont, text="ID:").grid(row=1, column=1, padx=5)
        self.coronal_hole_id_dropdown = ttk.Entry(
            ch_buttons,
            width=5,
            validate="key",
            validatecommand=(validate_input, "%P"),
        )
        self.coronal_hole_id_dropdown.grid(row=1, column=2, padx=5)

        # Confidence dropdown
        Label(ch_buttons, font=self.labelFont, text="Confidence:").grid(
            row=1, column=3, padx=5
        )
        confidence_values = ["1", "2", "3", "4"]
        self.confidence_dropdown = ttk.Combobox(
            ch_buttons, values=confidence_values, state="readonly", width=5
        )
        self.confidence_dropdown.set("3")
        self.confidence_dropdown.grid(row=1, column=4, padx=5)

        # Polarity dropdown
        Label(ch_buttons, font=self.labelFont, text="Polarity:").grid(
            row=1, column=5, padx=5
        )
        polarity_values = ["+", "-"]
        self.polarity_dropdown = ttk.Combobox(
            ch_buttons, values=polarity_values, state="readonly", width=5
        )
        self.polarity_dropdown.set("+")
        self.polarity_dropdown.grid(row=1, column=6, padx=5)

        # Flag button
        labelFontStyle = ttk.Style()
        labelFontStyle.configure("BigFont.TCheckbutton", font=self.labelFont)
        self.flag_button_value = tk.IntVar(value=0)
        self.flag_button = ttk.Checkbutton(
            ch_buttons,
            state="readonly",
            text="Flag Bad",
            style="BigFont.TCheckbutton",
            variable=self.flag_button_value,
            width=10,
        )
        self.flag_button.grid(row=3, column=3, padx=5)

        # Save coronal hole button
        Button(
            right_frame,
            font=self.labelFont,
            text="Save Coronal Hole",
            command=on_save_coronal_hole,
            width=20,
        ).pack(pady=20)

        # Info label
        self.info_label = Label(
            self.root, text="Click on a mask to select it.", wraplength=200
        )

        # CH list frame
        self.ch_list = tk.Frame(right_frame, width=500, bg="lightblue")
        self.ch_list.pack(fill=tk.BOTH, expand=True, anchor="s")

        # Merge frame setup
        bottom_frame = tk.Frame(left_frame)
        bottom_frame.pack(side=tk.BOTTOM, padx=20, pady=10)

        self.merge_canvas = tk.Canvas(
            bottom_frame, width=260, height=180, bg="lightgray"
        )
        self.merge_items = tk.Frame(self.merge_canvas)
        scrollbar = tk.Scrollbar(
            self.merge_items, orient="vertical", command=self.merge_canvas.yview
        )
        scrollbar.pack(side=tk.RIGHT, fill="y")
        self.merge_canvas.create_window((0, 0), window=self.merge_items, anchor="nw")
        self.merge_canvas.configure(yscrollcommand=scrollbar.set)
        self.merge_canvas.pack(side=tk.TOP)

        merge_actions = tk.Frame(bottom_frame)
        merge_actions.pack(side=tk.BOTTOM, padx=20, pady=10)

        merge_selection = Button(
            merge_actions, text="Select for Merge", command=on_merge_select
        )
        merge_selection.pack(side=tk.LEFT, padx=20, pady=10)

        merge_button = Button(
            merge_actions,
            text="Merge Selected Masks",
            command=on_merge_selected_masks,
        )
        merge_button.pack(side=tk.RIGHT, padx=20, pady=10)

    def update_detected_chs_count(self, count: int) -> None:
        """Update the detected CHs count display."""
        self.detected_chs_entry.config(state="normal")
        self.detected_chs_entry.delete(0, tk.END)
        self.detected_chs_entry.insert(0, str(count))
        self.detected_chs_entry.config(state="readonly")

    def clear_frame(self, frame: tk.Frame) -> None:
        """Clear all widgets from a frame."""
        for widget in frame.winfo_children():
            widget.destroy()
        frame.update_idletasks()

    def update_coronal_hole_display(
        self,
        saved_masks: List[Dict[str, Any]],
        on_remove_callback: Callable[[int], None],
        create_polygon_callback: Callable[[Any, Canvas], None],
    ) -> None:
        """Update the display of saved coronal holes."""
        self.clear_frame(self.ch_list)
        self.update_detected_chs_count(len(saved_masks))

        Label(self.ch_list, text="Saved Coronal Holes", font=self.labelFont).pack(
            pady=20
        )

        for saved_mask_idx, saved_coronal_hole in enumerate(saved_masks):
            coronal_hole_id = saved_coronal_hole["id"]
            polarity = saved_coronal_hole["polarity"]
            confidence = saved_coronal_hole["confidence"]
            flagged_bad = saved_coronal_hole["flagged"]
            segmentation = saved_coronal_hole["mask_region"]["segmentation"]

            # New frame for this CH
            new_ch_frame = tk.Frame(self.ch_list, bg="lightblue")
            new_ch_frame.pack(fill="x")

            # Draw the CH on a canvas
            ch_canvas = Canvas(new_ch_frame, width=75, height=75)
            ch_canvas.pack(side="left", padx=4, pady=4)
            create_polygon_callback(segmentation, ch_canvas)

            # Add label
            ch_label = Label(
                new_ch_frame,
                font=self.labelFont,
                text=f"ID: {coronal_hole_id}, Polarity: {polarity}, Confidence: {confidence}, Flagged Bad: {flagged_bad}",
            )
            ch_label.pack(side="left", pady=20, fill="x")

            # Remove button
            ch_remove_button = Button(
                new_ch_frame,
                text="-",
                bg="gray",
                borderwidth=5,
                anchor="w",
                command=lambda idx=saved_mask_idx: on_remove_callback(idx),
            )
            ch_remove_button.pack(side="right", pady=20, fill="x")

    def create_merge_item_row(
        self,
        current_mask: Any,
        create_polygon_callback: Callable[[Any, Canvas], None],
        on_remove_callback: Callable[[Any, tk.Frame], None],
    ) -> None:
        """Create a row in the merge items list."""
        merge_frame = tk.Frame(self.merge_items, bg="lightblue")
        merge_frame.pack(fill="both", expand=True)
        self.merge_rows.append(merge_frame)

        ch_canvas = Canvas(merge_frame, width=50, height=50)
        ch_canvas.pack(side="left", padx=4, pady=4)
        create_polygon_callback(current_mask, ch_canvas)

        merge_label = Label(merge_frame, text=f"Mask {current_mask} selected to merge.")
        merge_label.pack(side="left")

        merge_remove_button = Button(
            merge_frame,
            text="-",
            bg="gray",
            borderwidth=5,
            anchor="w",
            command=lambda: on_remove_callback(current_mask, merge_frame),
        )
        merge_remove_button.pack(side="right")

        self.merge_items.update_idletasks()
        self.merge_canvas.config(scrollregion=self.merge_canvas.bbox("all"))

    def clear_merge_rows(self) -> None:
        """Clear all merge item rows."""
        for row in self.merge_rows:
            row.destroy()
        self.merge_rows = []

    def update_info_label(self, text: str) -> None:
        """Update the info label text."""
        self.info_label.config(text=text)
