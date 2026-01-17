# Shell Scripts

Batch processing and automation scripts for CHASM workflows.

## Scripts

### Prediction Scripts

- `predict_chasm1111.sh` - Run predictions on CHASM model 1111
- `predict_chasm1407.sh` - Run predictions on CHASM model 1407
- `predict_chasm967.sh` - Run predictions on CHASM model 967
- `predict_finetune.sh` - Run fine-tuned model predictions
- `predict_spocach.sh` - Run SPOCA-CH predictions

### Data Processing

- `resize_mag.sh` - Batch magnetogram resizing
- `move_model.sh` - Model file relocation

### Analysis

- `generate_tables.sh` - Generate result tables from predictions
- `overlay_view.sh` - Create overlay visualizations

## Usage

Make scripts executable and run:

```bash
chmod +x scripts/script_name.sh
./scripts/script_name.sh
```

Or use bash directly:

```bash
bash scripts/script_name.sh
```

## Configuration

Most scripts expect specific directory structures and may need path adjustments for your environment. Check the script contents before running.

## Platform

These are shell scripts designed for Unix-like systems (Linux, macOS, WSL on Windows). Windows users should use WSL, Git Bash, or convert to PowerShell equivalents.
