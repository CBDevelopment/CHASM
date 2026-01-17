import tkinter as tk
from tkinter import Canvas, Button, Label
from tkinter import ttk
import tkinter.font as tkFont
import numpy as np
import cv2
import os
from PIL import Image, ImageTk  # Requires the Pillow library for handling image files
from time import perf_counter
import csv
from datasets.folder_utils import get_file_by_date
from chasm.app.threaded_image_loader import *


# TODO have app save in JUST .npz format, having the .csv is kind of confusing
class CoronalHoleClassifier:
    def __init__(self, load_dir, masks_dir, save_dir, max_width=800, max_height=600):
        self.idx = None
        self.load_dir = load_dir
        self.masks_dir = masks_dir
        self.save_dir = save_dir
        self.original_img = None

        self.selection_start_time = None
        self.selection_end_time = None

        # Sizes
        self.max_width = max_width
        self.max_height = max_height

        # Add the threaded loader
        self.image_loader = ThreadedImageLoader(
            self.load_dir, self.masks_dir, self.save_dir, queue_size=1
        )

        # Stuff (modified) from manual process for masks
        self.masks = None  # No mask with no image yet
        self.selected_mask = None

        # Used for scaling and storing image
        # All of thes are loaded with load_image()
        self.image = None
        self.scale_factor = None
        self.scaled_image = None
        self.tk_image = None
        self.masks = None
        self.max_width = 800
        self.min_width = 600

        # Storing masks to save
        self.saved_masks = []

        # Used to iterate through overlapping masks
        self.clicked_masks = []
        self.clicked_masks_idx = 0

        # GUI Stuff
        self.root = tk.Tk()
        self.root.title("Synoptic Map Viewer")
        self.root.geometry("1400x900")

        def on_validate_input(P):
            # This function is called when the user types something in the Entry widget
            # P is the current content of the Entry
            if P == "" or P.isdigit():  # Allow empty or digits only
                return True
            else:
                return False

        validate_input = self.root.register(on_validate_input)

        # Left panel (map display)
        self.left_frame = tk.Frame(self.root)
        self.left_frame.pack(side=tk.LEFT, padx=20, pady=10)

        # Width and height will change dynamically later as image loaded
        self.image_canvas = Canvas(
            self.left_frame, width=self.min_width, height=self.max_height, bg="white"
        )
        self.image_canvas.pack()

        # Right panel
        self.right_frame = tk.Frame(self.root)
        self.right_frame.pack(side=tk.LEFT, padx=20, pady=10)

        # Title
        Label(
            self.right_frame, text="Coronal Hole Masks", font=("Helvetica", 20, "bold")
        ).pack(anchor="w", pady=5)

        # Next Drawing / Previous Drawing buttons at top (with its own frame)
        self.drawing_buttons_frame = tk.Frame(
            self.right_frame
        )  # Used to get the buttons side by side at the top
        self.drawing_buttons_frame.pack(side="top")  # fill="x" to pack horizontally

        self.labelFont = tkFont.Font(family="Helvetica", size=16)

        previous_drawing_button = Button(
            self.drawing_buttons_frame,
            font=self.labelFont,
            text="Previous Drawing",
            command=self.previous_drawing,
            width=20,
        )
        previous_drawing_button.pack(side="left", padx=10, pady=20, fill="x")

        next_drawing_button = Button(
            self.drawing_buttons_frame,
            font=self.labelFont,
            text="Next Drawing",
            command=self.next_drawing,
            width=20,
        )
        next_drawing_button.pack(side="left", padx=10, pady=20, fill="x")

        # Checkbox for no coronal holes
        self.no_coronal_holes = tk.BooleanVar()
        self.no_coronal_holes_checkbox = tk.Checkbutton(
            self.drawing_buttons_frame,
            text="No Coronal Holes",
            variable=self.no_coronal_holes,
            font=self.labelFont,
        )
        self.no_coronal_holes_checkbox.pack(side="left", padx=10, pady=20, fill="x")

        # Row for CH Mask, Confidence, and Polarity (back to right frame)
        self.ch_buttons = tk.Frame(self.right_frame)
        self.ch_buttons.pack(pady=10, anchor="w")

        # Entry for number of CHs detected by SAM
        Label(self.ch_buttons, font=self.labelFont, text="Saved CHs:").grid(
            row=0, column=3, padx=5
        )
        self.detected_chs_entry = ttk.Entry(
            self.ch_buttons, width=5, validate="key", state="readonly"
        )
        self.detected_chs_entry.grid(row=0, column=4, padx=5)
        self.update_detected_chs_count()

        # Entry for true number of CHs in the image
        Label(self.ch_buttons, font=self.labelFont, text="True CHs:").grid(
            row=0, column=1, padx=5
        )
        self.true_chs_entry = ttk.Entry(
            self.ch_buttons,
            width=5,
            validate="key",
            validatecommand=(validate_input, "%P"),
        )
        self.true_chs_entry.grid(row=0, column=2, padx=5)

        # Mask placeholder
        self.mask_canvas = Canvas(self.ch_buttons, width=100, height=50, bg="lightgray")
        self.mask_canvas.grid(row=1, column=0, padx=10)
        self.mask_canvas.create_oval(25, 10, 75, 40, outline="blue", fill="blue")

        # Coronal hole id dropdown
        Label(self.ch_buttons, font=self.labelFont, text="ID:").grid(
            row=1, column=1, padx=5
        )
        self.coronal_hole_id_dropdown = ttk.Entry(
            self.ch_buttons,
            width=5,
            validate="key",
            validatecommand=(validate_input, "%P"),
        )
        self.coronal_hole_id_dropdown.grid(row=1, column=2, padx=5)

        # Confidence dropdown
        Label(self.ch_buttons, font=self.labelFont, text="Confidence:").grid(
            row=1, column=3, padx=5
        )
        confidence_values = ["1", "2", "3", "4"]
        self.confidence_dropdown = ttk.Combobox(
            self.ch_buttons, values=confidence_values, state="readonly", width=5
        )
        self.confidence_dropdown.set("3")  # Default value
        self.confidence_dropdown.grid(row=1, column=4, padx=5)

        # Polarity dropdown
        Label(self.ch_buttons, font=self.labelFont, text="Polarity:").grid(
            row=1, column=5, padx=5
        )
        polarity_values = ["+", "-"]
        self.polarity_dropdown = ttk.Combobox(
            self.ch_buttons, values=polarity_values, state="readonly", width=5
        )
        self.polarity_dropdown.set("+")  # Default value
        self.polarity_dropdown.grid(row=1, column=6, padx=5)

        # Create a style for the font to work in ttk style checkbutton
        self.labelFontStyle = ttk.Style()
        self.labelFontStyle.configure("BigFont.TCheckbutton", font=self.labelFont)

        # Create a variable to hold the state of the Checkbutton
        self.flag_button_value = tk.IntVar(
            value=0
        )  # Set the default value to 0 (unchecked)
        # NOTE: Got rid of ttk for font option
        self.flag_button = ttk.Checkbutton(
            self.ch_buttons,
            state="readonly",
            text="Flag Bad",
            style="BigFont.TCheckbutton",
            variable=self.flag_button_value,
            width=10,
        )
        self.flag_button.grid(row=3, column=3, padx=5)

        # Save coronal hole button below fields
        Button(
            self.right_frame,
            font=self.labelFont,
            text="Save Coronal Hole",
            command=self.save_coronal_hole,
            width=20,
        ).pack(pady=20)

        # Add a label to display selected mask info
        self.info_label = Label(
            self.root, text="Click on a mask to select it.", wraplength=200
        )

        self.ch_list = tk.Frame(
            self.right_frame, width=500, bg="lightblue"
        )  # Frame to store saved coronal holes
        self.ch_list.pack(fill=tk.BOTH, expand=True, anchor="s")

        self.update_coronal_hole_display()  # Set up list (blank at first)

        # Merge frame setup
        self.bottom_frame = tk.Frame(self.left_frame)
        self.bottom_frame.pack(side=tk.BOTTOM, padx=20, pady=10)

        self.merge_rows = []
        self.merge_canvas = tk.Canvas(
            self.bottom_frame, width=260, height=180, bg="lightgray"
        )
        self.merge_items = tk.Frame(self.merge_canvas)
        scrollbar = tk.Scrollbar(
            self.merge_items, orient="vertical", command=self.merge_canvas.yview
        )
        scrollbar.pack(side=tk.RIGHT, fill="y")
        self.merge_canvas.create_window((0, 0), window=self.merge_items, anchor="nw")
        self.merge_canvas.configure(yscrollcommand=scrollbar.set)
        self.merge_canvas.pack(side=tk.TOP)

        self.merge_actions = tk.Frame(self.bottom_frame)
        self.merge_actions.pack(side=tk.BOTTOM, padx=20, pady=10)

        self.merge_selection = Button(
            self.merge_actions, text="Select for Merge", command=self.merge_select
        )
        self.merge_selection.pack(side=tk.LEFT, padx=20, pady=10)

        self.merge_button = Button(
            self.merge_actions,
            text="Merge Selected Masks",
            command=self.merge_selected_masks,
        )
        self.merge_button.pack(side=tk.RIGHT, padx=20, pady=10)

        self.masks_to_merge = []

        # Key Binds
        self.bind_keys()

        # Add window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.destroy)

        # Run the Tkinter event loop
        self.root.mainloop()

    def get_matching_sam_match(self):
        mask_file = get_file_by_date(
            self.load_dir.filenames()[self.idx], self.masks_dir.filenames()
        )
        self.masks = np.load(mask_file, allow_pickle=True)

    def update_detected_chs_count(self):
        detected_chs = len(self.saved_masks)
        self.detected_chs_entry.config(state="normal")  # Enable editing
        self.detected_chs_entry.delete(0, tk.END)  # Clear the current value
        self.detected_chs_entry.insert(0, str(detected_chs))  # Insert the new value
        self.detected_chs_entry.config(state="readonly")

    def bind_keys(self):
        """Binds keys to functions"""
        self.root.bind("<space>", self.draw_masks_space_bar)
        self.root.bind("<d>", self.cycle_masks_d_key)
        self.root.bind("<m>", self.merge_select_hotkey)
        self.root.bind("<s>", self.subtract_selected_masks)

        # Bind click event for selecting a mask
        self.image_canvas.bind("<Button-1>", self.on_mask_click)

    def clear_frame(self, frame):
        # Iterate over all widgets in the frame and destroy them
        for widget in frame.winfo_children():
            widget.destroy()
        frame.update_idletasks()

    def inc_idx(self):
        """Increments index corresponding to which file in the folder corresponds to the image, to go forwards."""
        self.idx += 1

    def dec_idx(self):
        """Decrements index corresponding to which file in the folder corresponds to the image, to go backwards."""
        self.idx -= 1

    def load_masks(self):
        # TODO take self.masks_dir.filenames()[self.idx]
        # self.masks = np.load(self.masks_dir.filenames()[self.idx], allow_pickle=True)s
        self.get_matching_sam_match()
        self.masks = [
            self.masks[arr_str].item()
            for arr_str in self.masks
            if 5000 <= self.masks[arr_str].item()["area"] <= 1000000
        ]

    def load_image(self, image):
        """Does through the process of loading a specific image into the GUI, including the clickable masks."""
        # Set up image
        self.image = image  # Keep original image for processing
        self.scale_factor = min(
            self.max_width / image.shape[1], self.max_height / image.shape[0], 1.0
        )
        scaled_width = int(self.image.shape[1] * self.scale_factor)
        scaled_height = int(self.image.shape[0] * self.scale_factor)
        self.scaled_image = cv2.resize(
            image, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA
        )
        self.tk_image = ImageTk.PhotoImage(Image.fromarray(self.scaled_image))
        self.image_canvas.config(
            width=scaled_width, height=scaled_height
        )  # Change canvas size to fit image

        # Display image
        self.image_canvas.delete("all")  # Clear canvas
        self.image_canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

        print("LOADING MASKS")
        self.load_masks()

        # Draw all masks on the canvas with unique colors
        self.draw_masks()

    def create_CH_polygon(self, mask, canvas):
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
        canvas_height = canvas.canvas_height = canvas.winfo_height()

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

    def save_selection_time(self):
        if self.selection_start_time is not None:
            selection_time = self.selection_end_time - self.selection_start_time

            csv_file = os.path.join(self.save_dir, "selection_times.csv")

            existing_data = []
            if os.path.exists(csv_file):
                with open(csv_file, mode="r", newline="") as file:
                    reader = csv.reader(file)
                    existing_data = list(reader)

            updated = False
            for row in existing_data:
                if row[0] == self.load_dir.filenames()[self.idx]:
                    row[1] = selection_time
                    updated = True
                    break

            if not updated:
                existing_data.append(
                    [self.load_dir.filenames()[self.idx], selection_time]
                )

            with open(csv_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(existing_data)
            print(f"Time to select: {selection_time:0.2f}")

    # TODO remove this after got whole app working
    # def save_detection_fraction(self):
    #     csv_file = os.path.join(self.save_dir, "detection_fractions.csv") # TODO fit this into the .npz structure.
    #     detected_by_sam = self.detected_chs_entry.get()
    #     true_chs = self.true_chs_entry.get()
    #     all_detected = detected_by_sam == true_chs

    #     existing_data = []
    #     file_exists = os.path.exists(csv_file)
    #     if file_exists:
    #         with open(csv_file, mode='r', newline='') as file:
    #             reader = csv.reader(file)
    #             existing_data = list(reader)

    #     updated = False
    #     for row in existing_data:
    #         if row[0] == os.listdir(self.load_dir)[self.idx]:
    #             row[1] = detected_by_sam
    #             row[2] = true_chs
    #             row[3] = all_detected
    #             updated = True
    #             break

    #     if not updated:
    #         existing_data.append(
    #             [os.listdir(self.load_dir)[self.idx], detected_by_sam, true_chs, all_detected])

    #      # TODO remove this, replacing this information into .npz file instead
    #     with open(csv_file, mode='w', newline='') as file:
    #         writer = csv.writer(file)
    #         if not file_exists:
    #             writer.writerow(["Filename", "SAM Detected", "True CHs", "All Detected"])
    #         writer.writerows(existing_data)
    #     print(f"Detected by SAM: {detected_by_sam}, True CHs: {true_chs}")
    #     print(f"All Detected?: {all_detected}")

    def save_current_drawing(self):
        """Save the current drawing's data"""
        print("Saving Information")
        flags = [f["flagged"] for f in self.saved_masks]
        data = {
            "info": self.saved_masks,
            "SAM Detected": self.detected_chs_entry.get(),  # TODO
            "True CHs": self.true_chs_entry.get(),  # TODO
            "All Detected": self.detected_chs_entry.get() == self.true_chs_entry.get(),
            "Good Quality": True
            if all(flag is False for flag in flags)
            else False,  # TODO handle "Good Quality" vs "good_quality" in tests
        }

        print("SAVING DRAWING TO ")
        np.savez_compressed(
            # NOTE: Saving it without the data now
            os.path.join(
                self.save_dir,
                str(
                    os.path.basename(self.load_dir.filenames()[self.idx]).split(".")[0][
                        :-4
                    ]
                )
                + ".npz",
            ),
            **data,
        )
        self.selection_end_time = perf_counter()
        self.save_selection_time()  # NOTE: Might want to keep this in .csv still since it is sort of seperate
        # self.save_detection_fraction()

    def save_CH_masks(self):
        # Don't save if idx is None
        # Don't save if no_coronal_holes is not checked and there are no saved masks
        # Save if no_coronal_holes is checked
        # print("SAVE CHECKS ---------")
        # print(self.idx)
        # print(self.no_coronal_holes.get())
        # print(len(self.saved_masks))
        if self.idx is not None:
            if self.no_coronal_holes.get() or len(self.saved_masks) > 0:
                self.save_current_drawing()

    def update_coronal_hole_display(self):
        self.clear_frame(self.ch_list)
        self.update_detected_chs_count()

        Label(self.ch_list, text="Saved Coronal Holes", font=self.labelFont).pack(
            pady=20
        )  # Add title for list back
        for saved_mask_idx, saved_coronal_hole in enumerate(
            self.saved_masks
        ):  # Add each saved CH information to list to display
            coronal_hole_id = saved_coronal_hole["id"]
            polarity = saved_coronal_hole["polarity"]
            confidence = saved_coronal_hole["confidence"]
            flagged_bad = saved_coronal_hole["flagged"]
            segmentation = saved_coronal_hole["mask_region"]["segmentation"]

            # New frame within list to have buttons side by side
            new_ch_frame = tk.Frame(self.ch_list, bg="lightblue")
            new_ch_frame.pack(fill="x")

            # Draw the CH on a canvas in the new frame
            ch_canvas = Canvas(new_ch_frame, width=75, height=75)
            ch_canvas.pack(side="left", padx=4, pady=4)
            self.create_CH_polygon(segmentation, ch_canvas)

            # Add label and corresponding remove button
            ch_label = Label(
                new_ch_frame,
                font=self.labelFont,
                text=f"ID: {coronal_hole_id}, Polarity: {polarity}, Confidence: {confidence}, Flagged Bad: {flagged_bad}",
            )
            ch_label.pack(side="left", pady=20, fill="x")

            # Command removes this specific coronal hole from list, button next to appropriate label
            ch_remove_button = Button(
                new_ch_frame,
                text="-",
                bg="gray",
                borderwidth=5,
                anchor="w",
                command=lambda: self.remove_coronal_hole(saved_mask_idx),
            )

            ch_remove_button.pack(side="right", pady=20, fill="x")

    def remove_coronal_hole(self, idx):
        removed_ch = self.saved_masks.pop(idx)
        print("Removed Coronal Hole " + str(removed_ch))
        self.update_coronal_hole_display()

    def draw_masks(self):
        """Draws all masks on the disk"""
        self.mask_map = {}
        for i, ann in enumerate(self.masks):
            segmentation = ann["segmentation"]
            scaled_segmentation = cv2.resize(
                segmentation.astype(np.uint8),
                (self.scaled_image.shape[1], self.scaled_image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

            contours, _ = cv2.findContours(
                scaled_segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            color = "#{:02x}{:02x}{:02x}".format(*np.random.randint(0, 255, size=3))

            for contour in contours:
                contour_points = [(point[0][0], point[0][1]) for point in contour]
                poly_id = self.image_canvas.create_polygon(
                    *[coord for point in contour_points for coord in point],
                    fill=color,
                    outline=color,
                )
                self.mask_map[poly_id] = i

    def next_drawing(self):
        """Modified to use preloaded data"""
        self.save_CH_masks()

        # Get next preloaded package
        self.selection_start_time = perf_counter()
        next_package = self.image_loader.get_next_package()
        if next_package is None:
            print("No preloaded data available, loading synchronously...")
            self.idx = self.idx + 1 if self.idx is not None else 0
            while os.path.exists(
                os.path.join(
                    self.save_dir,
                    # TODO make load_dir sequential
                    str(self.load_dir.filenames()[self.idx][:-4] + ".npz"),
                )
            ):
                self.inc_idx()

            self.load_image(np.array(self.get_next_image()))
            # Start the background loader
            self.image_loader.start_loading(self.scale_factor, self.idx)
        else:
            # Update current index and apply preloaded data
            self.idx = next_package["idx"]
            self.scaled_image = next_package["scaled_image"]
            self.tk_image = next_package["tk_image"]

            # Update canvas size and draw image
            self.image_canvas.config(
                width=next_package["width"], height=next_package["height"]
            )
            self.image_canvas.delete("all")
            self.image_canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

            # Update masks and draw using pre-calculated polygons
            self.masks = [
                data["original_mask"] for data in next_package["polygon_data"]
            ]
            self.mask_map = {}

            for i, mask_data in enumerate(next_package["polygon_data"]):
                for polygon in mask_data["polygons"]:
                    color = "#{:02x}{:02x}{:02x}".format(
                        *np.random.randint(0, 255, size=3)
                    )
                    poly_id = self.image_canvas.create_polygon(
                        *[coord for point in polygon for coord in point],
                        fill=color,
                        outline=color,
                    )
                    self.mask_map[poly_id] = i

        self.saved_masks = []
        self.update_coronal_hole_display()

    def previous_drawing(self):
        # Kill the loader thread
        print("Stopping loader thread")
        self.image_loader.stop_loading()
        self.dec_idx()
        # Start the background loader
        self.image_loader.start_loading(self.scale_factor, self.idx)
        print("Loading previous drawing")
        self.load_image(np.array(self.get_next_image()))
        self.update_coronal_hole_display()

    def get_next_image(self):
        file_name = self.load_dir.filenames()[self.idx]
        if file_name == "README.md":
            self.idx += 1
            file_name = self.load_dir.filenames()[self.idx]

        img = Image.open(os.path.join(self.load_dir.root, file_name))
        return img

    def merge_masks(self, mask1, mask2):
        combined_map = np.logical_or(mask1, mask2)
        return combined_map

    def subtract_masks(self, mask1, mask2):
        """Subtract mask2 from mask1 (mask1 - mask2)."""
        result_mask = np.bitwise_and(mask1, np.bitwise_not(mask2))
        return result_mask

    def remove_merge_item(self, current_mask, merge_frame):
        merge_frame.destroy()
        self.masks_to_merge.remove(current_mask)

    def create_merge_item_row(self, current_mask):
        merge_frame = tk.Frame(self.merge_items, bg="lightblue")
        merge_frame.pack(fill="both", expand=True)
        self.merge_rows.append(merge_frame)

        ch_canvas = Canvas(merge_frame, width=50, height=50)
        ch_canvas.pack(side="left", padx=4, pady=4)
        self.create_CH_polygon(
            self.masks[self.mask_map[current_mask]]["segmentation"], ch_canvas
        )

        merge_label = Label(merge_frame, text=f"Mask {current_mask} selected to merge.")
        merge_label.pack(side="left")

        merge_remove_button = Button(
            merge_frame,
            text="-",
            bg="gray",
            borderwidth=5,
            anchor="w",
            command=lambda: self.remove_merge_item(current_mask, merge_frame),
        )
        merge_remove_button.pack(side="right")

        self.merge_items.update_idletasks()
        self.merge_canvas.config(scrollregion=self.merge_canvas.bbox("all"))

    def merge_select_hotkey(self, event):
        """Binds the M key to merge_select"""
        self.merge_select()

    def merge_select(self):
        """Selects a mask for merging."""
        print("M Pressed")
        current_mask = self.clicked_masks[self.clicked_masks_idx]
        if current_mask not in self.masks_to_merge:
            self.masks_to_merge.append(current_mask)
            self.create_merge_item_row(current_mask)

    def merge_selected_masks(self):
        """Merges selected masks into one mask."""
        if len(self.masks_to_merge) < 2:
            print("Need at least two masks to merge.")
            return

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
            reverse=True,  # Sort in descending order to prevent index shifting
        )

        print("Polygons to remove: ", polygons_to_remove)
        print("Masks to remove: ", masks_to_remove)

        # Remove polygons from canvas and mask_map
        for item in polygons_to_remove:
            self.image_canvas.delete(item)
            self.mask_map.pop(item)

        # Remove masks from the list
        for mask_index in masks_to_remove:
            self.masks.pop(mask_index)

        # Add the merged mask
        self.masks.append(
            {"segmentation": np.array(merged_mask), "area": np.sum(merged_mask)}
        )

        # Reset selection and redraw
        self.reset_selection()
        self.draw_masks()

        # Clear the masks_to_merge list
        self.masks_to_merge = []
        for row in self.merge_rows:
            row.destroy()

    def subtract_selected_masks(self, event):
        if len(self.masks_to_merge) != 2:
            print("Can only subtract two masks.")
            return

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
            reverse=True,  # Sort in descending order to prevent index shifting
        )

        print("Polygons to remove: ", polygons_to_remove)
        print("Masks to remove: ", masks_to_remove)

        # Remove polygons from canvas and mask_map
        for item in polygons_to_remove:
            self.image_canvas.delete(item)
            self.mask_map.pop(item)

        # Remove masks from the list
        for mask_index in masks_to_remove:
            self.masks.pop(mask_index)

        # Add the subtracted mask
        self.masks.append(
            {"segmentation": np.array(subtract_mask), "area": np.sum(subtract_mask)}
        )

        # Reset selection and redraw
        self.reset_selection()
        self.draw_masks()

        # Clear the masks_to_merge list
        self.masks_to_merge = []
        for row in self.merge_rows:
            row.destroy()

    def reset_selection(self):
        self.selected_mask = None
        for item in self.mask_map:
            self.image_canvas.itemconfig(item, outline="", fill="", width=1)
        self.info_label.config(text="None")

    def draw_masks_space_bar(self, event):
        self.reset_selection()
        self.draw_masks()

    def cycle_masks_d_key(self, event):
        print("D Pressed")
        self.reset_selection()
        self.clicked_masks_idx = (self.clicked_masks_idx + 1) % len(self.clicked_masks)
        self.show_mask_item()

    def get_clicked_masks(self, event):
        self.clicked_masks_idx = 0
        self.clicked_masks = []
        x, y = event.x, event.y
        clicked_items = self.image_canvas.find_overlapping(x, y, x, y)

        for item in clicked_items:
            if len(clicked_items) == 1:
                self.reset_selection()
            if item in self.mask_map:
                self.clicked_masks.append(item)

        return self.clicked_masks

    def show_mask_item(self):
        """Shows mask based on what has been clicked and the index"""
        if len(self.clicked_masks) == 0:  # If no masks, nothing to show
            return

        self.reset_selection()
        item = self.clicked_masks[self.clicked_masks_idx]
        mask_index = self.mask_map[item]
        self.selected_mask = self.masks[mask_index]
        self.highlight_selected_mask(item)
        self.info_label.config(
            text=f"Mask {mask_index} selected: Area={self.selected_mask['area']}"
        )

    def on_mask_click(self, event):
        """Handle mask selection by clicking on it."""
        self.reset_selection()
        self.get_clicked_masks(event)  # Now have self.clicked_masks
        self.show_mask_item()

    def highlight_selected_mask(self, selected_id):
        """Highlight the selected mask with a red outline. Does not reset other masks."""
        self.image_canvas.itemconfig(selected_id, outline="red", width=2)

    def save_coronal_hole(self):
        print("Saving Coronal Hole")
        coronal_hole_id = self.coronal_hole_id_dropdown.get()
        confidence = self.confidence_dropdown.get()
        polarity = self.polarity_dropdown.get()
        flag_button_value = bool(self.flag_button_value.get())

        # TODO add the .csv type stuff here
        self.saved_masks.append(
            {
                "mask_region": self.selected_mask,
                "id": coronal_hole_id,
                "confidence": confidence,
                "polarity": polarity,
                "flagged": flag_button_value,
            }
        )
        self.update_coronal_hole_display()  # Update list of saved holes

    def destroy(self):
        """Clean up resources and destroy the window"""
        # Stop the background loader thread
        self.image_loader.stop_loading()
        # Save current drawing if there is one
        if self.idx is not None:
            self.save_CH_masks()
        # Destroy the window
        self.root.destroy()


def run_app(base_path=None, year=None):
    """
    Launch the CHASM GUI application.

    Args:
        base_path: Root directory containing drawings, sam_masks, and tool_selections folders
        year: Optional year subdirectory
    """
    import argparse

    # If called from command line without arguments, parse them
    if base_path is None:
        parser = argparse.ArgumentParser(
            description="CHASM - Coronal Hole Annotation Tool"
        )
        parser.add_argument("base_path", help="Root directory containing data folders")
        parser.add_argument("--year", type=int, help="Year subdirectory (optional)")
        args = parser.parse_args()
        base_path = args.base_path
        year = args.year

    # Global variables
    # TODO make this work with downloaded_dataset objects instead of folders

    if year != None:
        load_dir = f"{base_path}/drawings/{year}"
        masks_dir = f"{base_path}/sam_masks/{year}"
        save_dir = f"{base_path}/tool_selections/{year}"

    else:
        load_dir = f"{base_path}/drawings"
        masks_dir = f"{base_path}/sam_masks"
        save_dir = f"{base_path}/tool_selections"

    app = CoronalHoleClassifier(
        load_dir, masks_dir, save_dir, max_width=800, max_height=800
    )


# Create main window
if __name__ == "__main__":
    # Global variables
    # TODO make this work with downloaded_dataset objects instead of folders
    ROOT = "data_test_set"
    year = 2017
    load_dir = f"{ROOT}/drawings/{year}"
    masks_dir = f"{ROOT}/masks/{year}"
    save_dir = f"{ROOT}/tool_selections/{year}"

    # TODO make handle directory or DatasetObjects passed in (either a single dataset object or multiple)
    app = CoronalHoleClassifier(
        load_dir, masks_dir, save_dir, max_width=1000, max_height=800
    )
