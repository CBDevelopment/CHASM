# SAM Preprocessing

Integration with Meta's Segment Anything Model (SAM) for automatic mask generation on solar imagery.

## Purpose

Pre-compute segmentation masks that serve as candidates for coronal hole regions. These masks are then loaded into the CHASM GUI where users can select, merge, and annotate the relevant regions.

## Requirements

- `torch` - PyTorch framework
- `segment-anything` - Meta's SAM model
- Pre-trained SAM checkpoint file

## Usage

```python
from chasm.preprocessing.sam.preprocess import generate_masks

# Generate masks for a directory of images
generate_masks(
    input_dir='drawings/',
    output_dir='sam_masks/',
    model_type='vit_h',
    checkpoint='sam_vit_h.pth'
)
```

## Output Format

Masks are saved as `.npz` files containing:

- Segmentation arrays
- Bounding boxes
- Confidence scores
- Area measurements

## Filtering

Masks are typically filtered by area (5,000 - 1,000,000 pixels) to focus on coronal hole-sized regions and exclude noise or full-disk selections.
