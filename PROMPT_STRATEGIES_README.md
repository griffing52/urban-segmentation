# SAM3 Prompt Strategy Testing

Guide for testing different text prompt strategies to improve SAM3 boundary detection for thin objects.

## Overview

Different levels of prompt specificity can significantly affect SAM3's detection quality. We provide 4 prompt strategy levels to experiment with:

| Level | Description | Example | Use Case |
|-------|-------------|---------|----------|
| **L1_Baseline** | Simple, generic terms | "fence", "pole" | Fast baseline, general detection |
| **L2_Descriptive** | More descriptive | "metal fence", "street pole" | Better specificity, still concise |
| **L3_Physical** | Physical attributes | "thin vertical metal fence" | When shape/size matters |
| **L4_Specific** | Context + location | "street fence along road edge" | Urban scene-specific |

## Quick Start

### 1. Visual Comparison (Recommended First Step)

Test all 4 prompt strategies on a single image:

```bash
cd notebooks
jupyter notebook visualize_prompts.ipynb
```

**What you'll see:**
- Side-by-side boundary detection for all 4 strategies
- Difference maps showing what each strategy captures
- Per-class breakdown of detection quality
- Zoomed regions to see fine details

### 2. Run Full Dataset Generation

After choosing your preferred prompt level:

```bash
# Example: Use L2_Descriptive prompts with multi-crop strategy
python generate_sam3_masks.py \
    --data_root ../data/cityscapes \
    --strategy multi_crop \
    --prompt_level L2_Descriptive \
    --grid_size 2 2
```

## Prompt Strategy Details

### L1_Baseline (Default)
```python
{
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    11: "person",
    12: "rider",
    17: "motorcycle",
    18: "bicycle"
}
```
- ✅ Fast, simple, general
- ✅ Works across different contexts
- ❌ May be too generic (e.g., "pole" could match many things)

### L2_Descriptive
```python
{
    4: "metal fence",
    5: "street pole",
    6: "traffic signal light",
    7: "road traffic sign",
    11: "pedestrian person",
    12: "cyclist rider",
    17: "motorcycle vehicle",
    18: "bicycle bike"
}
```
- ✅ Better specificity without being verbose
- ✅ Adds material/function context
- ⚠️ May miss non-metal fences, non-street poles

### L3_Physical
```python
{
    4: "thin vertical metal fence barrier",
    5: "tall narrow cylindrical pole post",
    6: "small traffic light signal hanging",
    7: "flat rectangular traffic sign board",
    11: "standing upright human person",
    12: "person riding bicycle or motorcycle",
    17: "two-wheeled motorcycle with engine",
    18: "two-wheeled pedal bicycle"
}
```
- ✅ Emphasizes physical characteristics (shape, size, orientation)
- ✅ Helps distinguish similar objects
- ❌ Longer prompts = slower inference
- ⚠️ May be too restrictive

### L4_Specific
```python
{
    4: "street fence barrier along road edge",
    5: "utility pole or street lamp post",
    6: "traffic light mounted on pole or wire",
    7: "traffic sign mounted on pole showing information",
    11: "person pedestrian walking on street or sidewalk",
    12: "person riding on bicycle or motorcycle",
    17: "motorized motorcycle on road",
    18: "bicycle on road or bike lane"
}
```
- ✅ Urban scene-specific context
- ✅ Includes typical locations/positions
- ❌ Most restrictive - may miss edge cases
- ❌ Slowest due to longer prompts

## Usage Examples

### Test on Validation Set First
```bash
# Quick test with different prompt levels
python generate_sam3_masks.py \
    --data_root ../data/cityscapes \
    --strategy baseline \
    --prompt_level L2_Descriptive \
    --splits val
```

### Combine with Cropping Strategies
```bash
# Multi-crop with descriptive prompts
python generate_sam3_masks.py \
    --data_root ../data/cityscapes \
    --strategy multi_crop \
    --prompt_level L2_Descriptive \
    --grid_size 2 2

# Tiled with physical prompts
python generate_sam3_masks.py \
    --data_root ../data/cityscapes \
    --strategy tiled \
    --prompt_level L3_Physical \
    --window_size 1024 \
    --stride 512
```

### Generate Multiple Variants for Comparison
```bash
# Generate all prompt levels with baseline strategy
for level in L1_Baseline L2_Descriptive L3_Physical L4_Specific; do
    python generate_sam3_masks.py \
        --data_root ../data/cityscapes \
        --strategy baseline \
        --prompt_level $level \
        --splits val
done
```

## Expected Output Structure

```
cityscapes/
├── sam3_boundary_baseline_L1_Baseline/
│   └── val/
├── sam3_boundary_baseline_L2_Descriptive/
│   └── val/
├── sam3_boundary_multicrop_L2_Descriptive/
│   └── train/
│   └── val/
└── sam3_boundary_tiled_L3_Physical/
    └── train/
    └── val/
```

## Recommendations

### For Experimentation
1. **Start with visualization**: Run `visualize_prompts.ipynb` on 3-5 sample images
2. **Test on val set**: Try all 4 levels with baseline strategy on validation set
3. **Analyze results**: Look at per-class performance in the notebook

### For Production
Based on preliminary tests, we recommend:

**General Use:**
```bash
python generate_sam3_masks.py \
    --data_root ../data/cityscapes \
    --strategy multi_crop \
    --prompt_level L2_Descriptive \
    --grid_size 2 2
```

**Maximum Quality:**
```bash
python generate_sam3_masks.py \
    --data_root ../data/cityscapes \
    --strategy tiled \
    --prompt_level L3_Physical \
    --window_size 1024 \
    --stride 512
```

**Speed Priority:**
```bash
python generate_sam3_masks.py \
    --data_root ../data/cityscapes \
    --strategy baseline \
    --prompt_level L1_Baseline
```

## Performance Considerations

| Prompt Level | Inference Time | Detection Quality | Best For |
|--------------|----------------|-------------------|----------|
| L1_Baseline | Fastest | Good | Prototyping, general use |
| L2_Descriptive | Fast | Better | Production, balanced |
| L3_Physical | Medium | Best (specific) | When attributes matter |
| L4_Specific | Slowest | Variable | Urban scenes only |

*Inference time increases with prompt length due to text encoding.*

## Hybrid Strategy (Advanced)

For best results, consider using different prompt levels for different classes:

```python
# In generate_sam3_masks.py, you could modify:
HYBRID_PROMPTS = {
    4: PROMPTS_L3_PHYSICAL[4],      # fence - physical attributes help
    5: PROMPTS_L2_DESCRIPTIVE[5],   # pole - descriptive enough
    6: PROMPTS_L4_SPECIFIC[6],      # traffic light - context helps
    7: PROMPTS_L4_SPECIFIC[7],      # traffic sign - context helps
    11: PROMPTS_L1_BASELINE[11],    # person - simple is better
    12: PROMPTS_L2_DESCRIPTIVE[12], # rider - slight description
    17: PROMPTS_L3_PHYSICAL[17],    # motorcycle - shape matters
    18: PROMPTS_L3_PHYSICAL[18],    # bicycle - shape matters
}
```

## Evaluation Pipeline

To scientifically determine the best prompt strategy:

1. **Generate boundaries** with all strategies on val set
2. **Train models** with each boundary variant
3. **Compare mIoU** on thin classes (4, 5, 6, 7, 11, 12, 17, 18)
4. **Analyze per-class** improvement

```bash
# Generate all variants
for level in L1_Baseline L2_Descriptive L3_Physical L4_Specific; do
    python generate_sam3_masks.py \
        --data_root ../data/cityscapes \
        --strategy multi_crop \
        --prompt_level $level \
        --grid_size 2 2
done

# Train with each variant
for level in L1_Baseline L2_Descriptive L3_Physical L4_Specific; do
    python train_segformer_boundary.py \
        --data_root ../data/cityscapes \
        --boundary_dir sam3_boundary_multicrop_${level} \
        --experiment_name segformer_sam3_${level} \
        --epochs 80
done

# Compare results in your analysis notebook
```

## Troubleshooting

### Prompts too long / OOM
- Use L1_Baseline or L2_Descriptive
- Reduce batch size during inference

### Missing detections
- Try L3_Physical or L4_Specific
- Verify with visualization notebook first

### Too many false positives
- Use L4_Specific to add constraints
- Or revert to L1_Baseline for simplicity

## Files

- `generate_sam3_masks.py` - Main script with prompt level support
- `notebooks/visualize_prompts.ipynb` - Visual comparison tool
- `notebooks/visualize_strategies.ipynb` - Cropping strategy comparison

## Next Steps

1. Run `visualize_prompts.ipynb` on sample images
2. Choose 1-2 promising prompt levels
3. Generate full dataset
4. Train and evaluate downstream task performance
5. Report findings!

Good luck! 🚀
