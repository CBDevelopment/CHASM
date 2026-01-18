# CHASM GUI Application

Interactive Tkinter-based application for coronal hole annotation using semi-automatic methods.

## Features

- **Visual Annotation**: Click-based mask selection from SAM-generated regions
- **Mask Operations**:
    - Merge multiple masks together
    - Subtract overlapping regions
    - Cycle through overlapping masks
- **Quality Control**:
    - Confidence ratings (1-4)
    - Polarity assignment (+/-)
    - Flag problematic regions
- **Performance**: Background image loading for smooth navigation

## Usage

### Python API

```python
from chasm.gui.app import run_app
from chasm.datasets import DrawingsDataset, SAMDataset

# Create dataset objects pointing to your directories
drawings = DrawingsDataset(root="data/drawings", fetch_online=False)
sam_masks = SAMDataset(root="data/sam_masks")

# Launch GUI
run_app(
    drawings_dataset=drawings,
    sam_dataset=sam_masks,
    save_dir="data/tool_selections"
)
```

### Command Line

```bash
# Specify all three directories
chasm --drawings data/drawings \
      --sam-masks data/sam_masks \
      --save-dir data/tool_selections

# With custom canvas size
chasm --drawings data/drawings \
      --sam-masks data/sam_masks \
      --save-dir data/tool_selections \
      --max-width 1200 \
      --max-height 900

# With auto-download enabled
chasm --drawings data/drawings \
      --sam-masks data/sam_masks \
      --save-dir data/tool_selections
```

### Working with Year Subdirectories

```python
# If your data is organized by year
drawings = DrawingsDataset(root="data/drawings/2024", fetch_online=False)
sam_masks = SAMDataset(root="data/sam_masks/2024")

run_app(drawings, sam_masks, "data/tool_selections/2024")
```

## Keyboard Shortcuts

- **Space** - Redraw masks
- **D** - Cycle through overlapping masks at cursor
- **M** - Select current mask for merge
- **S** - Subtract selected masks

## Data Structure

**Input directories:**

- `drawings/` - Source images
- `sam_masks/` - Pre-computed segmentation masks (`.npz` format)

**Output:**

- `tool_selections/` - Annotated coronal hole data (`.npz` format)

Each saved annotation contains:

- Mask regions and metadata
- Confidence and polarity
- Quality flags
- Selection timing information
