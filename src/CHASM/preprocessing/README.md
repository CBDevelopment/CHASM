# CHASM Preprocessing

Utilities for preparing solar imagery data for annotation and analysis.

## Modules

### load_fits.py

Load and visualize FITS files from solar observatories.

**Features:**

- Reads SDO/AIA and other solar FITS files
- Applies proper normalization and color mapping
- Supports command-line usage

### pdf2jpg.py

Convert NOAA/SWPC hand-drawn solar maps from PDF to JPEG format.

**Usage:**

```python
from chasm.preprocessing.pdf2jpg import convert_pdf_to_images
convert_pdf_to_images('input.pdf', 'output_dir/')
```

### resize_mag.py

Resize and process magnetogram data for consistent analysis.

**Features:**

- Handles SDO/HMI magnetograms
- Maintains proper coordinate transformations
- Batch processing support

### SWPC.py

Download coronal hole data and drawings from NOAA Space Weather Prediction Center.

**Usage:**

```python
from chasm.preprocessing.SWPC import download_data
download_data(start_date='2023-01-01', end_date='2023-12-31')
```

### sam/

Segment Anything Model integration for automatic mask generation.

**Purpose:** Pre-compute segmentation masks that serve as candidates for coronal hole annotation in the gui.modules.

See `sam/README.md` for detailed usage.
