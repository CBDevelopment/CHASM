# Scratch / Development Scripts

Experimental and utility scripts for data processing, format conversion, and testing.

**Note:** These are development-only scripts and are not included in the distributed package.

## Scripts

### Data Conversion

- `convert_csv_style_to_npz.py` - Migrate old CSV format to NPZ
- `convert_npz_to_npy.py` - Convert between numpy formats
- `open_npz.py` / `read_npz.py` - Inspect NPZ file contents
- `add_quality_to_npz.py` - Add quality metadata to existing files

### File Management

- `csv_rename.py` - Batch rename CSV files
- `rename_aia_to_dates.py` - Standardize AIA filenames to date format
- `rename_model_for_new_path.py` - Update model paths
- `rename_quality.py` - Update quality field names

### Display & Testing

- `display_fits.py` - Quick FITS file viewer
- `fits_convert_hdul0_to1.py` - Convert FITS HDU formats
- `test_prediction_dataset.py` - Test prediction data loading
- `check_path.py` - Verify file paths

### Processing

- `resize_mag.py` - Magnetogram resizing utility
- `train_finetuned.py` - Fine-tuning experiments

### General

- `scratch.py` - General experimentation file

## Usage

These scripts are typically run directly:

```bash
python scratch/script_name.py
```

Not intended for production use or package distribution.
