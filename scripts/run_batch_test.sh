#!/bin/bash
# Quick test of batch generation pipeline
# Only runs 5 images with 3 configurations (baseline + multi_crop × 2 prompts)

set -e

DATA_ROOT="../data/cityscapes"
OUTPUT_DIR="../outputs/test_batch_masks"

echo "========================================"
echo "SAM3 Batch Generation TEST"
echo "Testing 3 configurations on 5 images"
echo "========================================"
echo ""

python generate_sam3_masks_from_configs.py \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --method baseline multi_crop \
    --prompts L1_Baseline L2_Descriptive \
    --grid_sizes 2 2 \
    --max_images 5

echo ""
echo "✓ Test complete! Check: $OUTPUT_DIR"
