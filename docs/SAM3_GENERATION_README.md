# SAM3 Boundary Mask Generation

Enhanced preprocessing pipeline for generating boundary masks using SAM3 with multiple inference strategies to handle resolution loss and improve thin-object detection.

## Problem

The original approach processes full images directly, causing SAM3 to miss small/distant thin objects (poles, traffic signs, etc.) due to resolution loss during resizing.

## Solution

Three inference strategies with different trade-offs:

| Strategy | Speed | Quality | Use Case |
|----------|-------|---------|----------|
| **Baseline** | ⚡⚡⚡ Fast | ⭐ Basic | Quick prototyping |
| **Multi-Crop** | ⚡⚡ Medium | ⭐⭐ Good | Balanced production |
| **Tiled** | ⚡ Slow | ⭐⭐⭐ Best | Maximum quality |

## Quick Start

### 1. Visualize Strategies on Sample Image

Compare all three strategies visually:

```bash
cd notebooks
jupyter notebook visualize_strategies.ipynb
```

This will show side-by-side comparisons of what each strategy detects.

### 2. Generate Full Dataset

Choose your strategy and run:

```bash
# Baseline: Fast, may miss small distant objects
python generate_sam3_masks.py \
    --data_root ../data/cityscapes \
    --strategy baseline

# Multi-Crop: Good balance (recommended)
python generate_sam3_masks.py \
    --data_root ../data/cityscapes \
    --strategy multi_crop \
    --grid_size 2 2

# Tiled: Highest quality, captures boundary objects
python generate_sam3_masks.py \
    --data_root ../data/cityscapes \
    --strategy tiled \
    --window_size 1024 \
    --stride 512
```

## Strategy Details

### 1. Baseline (No Crop)
```bash
python generate_sam3_masks.py --data_root ../data/cityscapes --strategy baseline
```

- Processes full image directly (2048x1024 → SAM3 input size)
- ✅ Fastest (1 inference per image)
- ❌ May miss small/distant objects due to resolution loss
- **Output:** `sam3_boundary_baseline/`

### 2. Multi-Crop (Fixed Grid)
```bash
python generate_sam3_masks.py \
    --data_root ../data/cityscapes \
    --strategy multi_crop \
    --grid_size 2 2
```

- Splits image into fixed grid (e.g., 2x2 = 4 crops)
- Each crop processed at higher effective resolution
- ✅ Better thin-object detection
- ✅ Reasonable speed (4 inferences for 2x2)
- ⚠️ May miss objects on crop boundaries
- **Output:** `sam3_boundary_multi_crop/`

**Grid Options:**
- `--grid_size 2 2`: Four quadrants (top-left, top-right, bottom-left, bottom-right)
- `--grid_size 1 2`: Left/right split
- `--grid_size 2 1`: Top/bottom split

### 3. Tiled (Sliding Window)
```bash
python generate_sam3_masks.py \
    --data_root ../data/cityscapes \
    --strategy tiled \
    --window_size 1024 \
    --stride 512
```

- Overlapping sliding windows capture everything
- ✅ Best quality - captures boundary objects
- ✅ Most consistent detection
- ❌ Slowest (9-12 inferences per image depending on stride)
- **Output:** `sam3_boundary_tiled/`

**Window Options:**
- `--window_size 1024`: Size of sliding window (square)
- `--stride 512`: Step size (smaller = more overlap = better quality but slower)

## Command-Line Arguments

### Required
- `--data_root`: Path to Cityscapes dataset root
- `--strategy`: One of `baseline`, `multi_crop`, `tiled`

### Optional
- `--output_dir`: Custom output directory name (default: `sam3_boundary_{strategy}`)
- `--splits`: Which splits to process (default: `train val`)
- `--overwrite`: Regenerate existing masks
- `--device`: Device to use (default: `cuda` if available)

### Strategy-Specific

**Multi-Crop:**
- `--grid_size ROWS COLS`: Grid dimensions (default: `2 2`)

**Tiled:**
- `--window_size N`: Window size in pixels (default: `1024`)
- `--stride N`: Stride in pixels (default: `512`)

## Examples

### Generate Only Validation Set (Fast Testing)
```bash
python generate_sam3_masks.py \
    --data_root ../data/cityscapes \
    --strategy multi_crop \
    --splits val \
    --grid_size 2 2
```

### High Quality for Training Set Only
```bash
python generate_sam3_masks.py \
    --data_root ../data/cityscapes \
    --strategy tiled \
    --splits train \
    --window_size 1024 \
    --stride 512
```

### Compare Multiple Strategies
```bash
# Generate all three strategies
python generate_sam3_masks.py --data_root ../data/cityscapes --strategy baseline
python generate_sam3_masks.py --data_root ../data/cityscapes --strategy multi_crop --grid_size 2 2
python generate_sam3_masks.py --data_root ../data/cityscapes --strategy tiled --window_size 1024 --stride 512

# Then compare in training...
```

## Output Structure

```
cityscapes/
├── sam3_boundary_baseline/      # From baseline strategy
│   ├── train/
│   │   ├── aachen/
│   │   │   └── aachen_000000_000019_leftImg8bit.npy
│   │   └── ...
│   └── val/
├── sam3_boundary_multi_crop/    # From multi-crop strategy
│   ├── train/
│   └── val/
└── sam3_boundary_tiled/         # From tiled strategy
    ├── train/
    └── val/
```

## Using Generated Masks in Training

Update your dataset class to point to the desired boundary directory:

```python
class CityscapesSAM3Dataset(Dataset):
    def __init__(self, root_dir, split="train", boundary_strategy="multi_crop"):
        # ...
        self.bnd_dir = self.root_dir / f"sam3_boundary_{boundary_strategy}" / split
        # ...
```

Or in the training script:

```bash
python train_segformer_boundary.py \
    --data_root ../data/cityscapes \
    --boundary_dir sam3_boundary_tiled \
    ...
```

## Performance Benchmarks

Approximate times on A100 GPU for full Cityscapes (2975 train + 500 val images):

| Strategy | Train Time | Val Time | Total | Boundary Pixels (avg) |
|----------|------------|----------|-------|----------------------|
| Baseline | ~30 min | ~5 min | ~35 min | ~50K |
| Multi-Crop 2x2 | ~2 hours | ~20 min | ~2.3 hours | ~70K |
| Tiled (1024/512) | ~6 hours | ~1 hour | ~7 hours | ~85K |

*Times are approximate and depend on GPU, I/O speed, and image content.*

## Recommendations

### For Experimentation
Use **Baseline** or **Multi-Crop 2x2** on `val` split only:
```bash
python generate_sam3_masks.py --data_root ../data/cityscapes --strategy multi_crop --splits val --grid_size 2 2
```

### For Production Training
Use **Multi-Crop 2x2** for full dataset:
```bash
python generate_sam3_masks.py --data_root ../data/cityscapes --strategy multi_crop --grid_size 2 2
```

### For Paper/Best Results
Use **Tiled** with 50% overlap:
```bash
python generate_sam3_masks.py --data_root ../data/cityscapes --strategy tiled --window_size 1024 --stride 512
```

## Troubleshooting

### Out of Memory
```bash
# Reduce window size for tiled
python generate_sam3_masks.py --strategy tiled --window_size 768 --stride 384

# Or use multi-crop instead
python generate_sam3_masks.py --strategy multi_crop --grid_size 2 2
```

### Too Slow
```bash
# Use baseline or increase stride
python generate_sam3_masks.py --strategy tiled --window_size 1024 --stride 768
```

### Missing Boundaries
Ensure you're using multi-crop or tiled strategy, and verify with the visualization notebook first.

## Next Steps

1. **Visualize first**: Run `visualize_strategies.ipynb` on a sample image
2. **Choose strategy**: Based on quality/speed trade-off
3. **Generate masks**: Run full dataset generation
4. **Update training**: Point training script to new boundary directory
5. **Compare results**: Train models with different strategies and compare performance

## Files

- `generate_sam3_masks.py` - Main generation script
- `notebooks/visualize_strategies.ipynb` - Visual comparison tool
- `SAM3_SegFormer_Combined.ipynb` - Original notebook with all strategies (deprecated)
