# Statistical Analysis

Tools for analyzing coronal hole detection results and generating visualizations.

## Modules

### aia_wavelengths.py

Analysis of AIA multi-wavelength data for coronal hole studies.

### overlay_view_scratch.py

Experimental overlay visualization development.

### stats_helper.py

Common statistical utilities and helper functions.

### tables_from_raw_data.py

Generate summary tables from raw detection/annotation data.

**Features:**

- Aggregate results across datasets
- Calculate detection metrics
- Format publication-ready tables

### view_fits_scratch.py

Interactive FITS file viewing and analysis.

## Usage

```python
from stats_updated.tables_from_raw_data import generate_summary_tables

# Generate result tables
generate_summary_tables(
    annotations_dir='tool_selections/',
    predictions_dir='predictions/',
    output_file='results_table.csv'
)
```

## Purpose

These tools support:

- Research paper preparation
- Model performance evaluation
- Data quality assessment
- Visualization for presentations

## Output

Generated files may include:

- CSV tables with detection statistics
- Performance comparison charts
- Overlay images showing predictions vs ground truth
