# CHASM Package

The main CHASM package for coronal hole annotation and analysis.

## Modules

### app/

Tkinter-based GUI application for manual annotation and semi-automatic coronal hole detection.

**Key Components:**

- `app.py` - Main application class and entry point
- `chasm.py` - Core annotation logic
- `main.py` - Application launcher
- `threaded_image_loader.py` - Background image loading for improved performance

### preprocessing/

Data preprocessing utilities for solar imagery.

**Key Components:**

- `load_fits.py` - FITS file loading and visualization
- `pdf2jpg.py` - PDF to JPEG conversion for drawings
- `resize_mag.py` - Magnetogram resizing and processing
- `SWPC.py` - SWPC data downloader
- `sam/` - Segment Anything Model preprocessing

## Usage

Install the package and run:

```bash
chasm <base_path> [--year YEAR]
```

Where `base_path` contains:

- `drawings/` - Solar drawings or imagery
- `sam_masks/` - Pre-computed SAM segmentation masks
- `tool_selections/` - Saved annotations
