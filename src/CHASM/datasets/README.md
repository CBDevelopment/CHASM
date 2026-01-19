# Dataset Utilities

Data loading and management utilities for various solar physics datasets.

## Dataset Classes

### aia_dataset.py

Load SDO/AIA (Atmospheric Imaging Assembly) EUV imagery.

### chasm_dataset.py

Load CHASM-annotated coronal hole data.

### chronnos_dataset.py

Interface for Chronnos multi-channel detection data.

### combined_dataset.py

Combine multiple datasets for training/evaluation.

### drawings_dataset.py

Load NOAA hand-drawn solar maps.

### fits_dataset.py

Generic FITS file dataset loader.

### prediction_dataset.py

Prepare data for model inference.

### sam_dataset.py

Load SAM-generated mask data.

## Utilities

### folder_utils.py

Helper functions for file organization and date-based file matching.

**Key Functions:**

- `get_file_by_date()` - Match files across directories by timestamp
- Directory traversal and filtering

## Usage

```python
from datasets.google_drive.chasm_dataset import CHASMDataset
from datasets.folder_utils import get_file_by_date

# Load annotated data
dataset = CHASMDataset('path/to/annotations')

# Match files by date
mask_file = get_file_by_date(image_filename, mask_dir_files)
```

## Integration

These utilities are used throughout CHASM for:

- Training machine learning models
- Loading data into the GUI
- Batch processing and evaluation
