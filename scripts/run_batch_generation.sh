#!/bin/bash
# Batch SAM3 Mask Generation Runner
# Generates masks for multiple strategies and prompt levels

set -e  # Exit on error

DATA_ROOT="../data/cityscapes"
OUTPUT_DIR="../generated_masks"
MAX_IMAGES=""  # Set to "--max_images 50" for testing

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}SAM3 Batch Mask Generation${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Run generation for all combinations
# This will create:
# - 4 baseline runs (one per prompt level)
# - 8 multi_crop runs (2 grids × 4 prompts)
# - 8 tiled runs (2 strides × 4 prompts)
# Total: 20 different configurations

python generate_sam3_masks_from_configs.py \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --method baseline multi_crop tiled \
    --prompts L1_Baseline L2_Descriptive L3_Physical L4_Specific \
    --grid_sizes 2 2 1 2 \
    --strides 256 512 \
    $MAX_IMAGES

echo ""
echo -e "${GREEN}✓ Batch generation complete!${NC}"
echo -e "Output directory: $OUTPUT_DIR"
