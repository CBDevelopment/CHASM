# CHASM Source Directory

This directory contains the source code for the CHASM (Coronal Hole Annotation using Semi-automatic Methods) project.

## Structure

### Core Package

- **chasm/** - Main CHASM package
    - **app/** - Tkinter-based GUI application for coronal hole annotation
    - **preprocessing/** - Data preprocessing utilities for solar imagery

### Supporting Modules

- **datasets/** - Dataset loaders and utilities for various solar data formats
- **MultiChannelCHDetection/** - Multi-channel coronal hole detection using Chronnos

### Utility Scripts

- **commands.py** - Command-line interface definitions
- **predict.py** - Model prediction script
- **train_model.py** - Model training script
- **move\_\*.py** - File and model relocation utilities

### Development

- **scratch/** - Experimental and development scripts
- **scripts/** - Shell scripts for batch processing
- **stats_updated/** - Statistical analysis and visualization tools

## Usage

The main package can be installed and used as:

```bash
pip install .
chasm <data_directory> [--year YEAR]
```

For development work, use the scripts and utilities in the respective directories.
