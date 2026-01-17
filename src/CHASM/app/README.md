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

```bash
chasm <base_path> [--year YEAR]
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
