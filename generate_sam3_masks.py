"""
SAM3 Boundary Mask Generation with Multiple Inference Strategies

Generates boundary masks for thin objects (poles, signs, fences, etc.) using SAM3
with three different inference strategies to handle resolution loss issues.

Strategies:
1. Baseline: Process full image (may miss small distant objects)
2. Multi-Crop: Split image into fixed grid (2x2 or 2x1)
3. Tiled: Sliding window with overlap (best for objects on boundaries)

Usage:
    # Generate using baseline strategy
    python generate_sam3_masks.py --data_root ../data/cityscapes --strategy baseline

    # Generate using multi-crop strategy (2x2 grid)
    python generate_sam3_masks.py --data_root ../data/cityscapes --strategy multi_crop --grid_size 2 2

    # Generate using tiled sliding window
    python generate_sam3_masks.py --data_root ../data/cityscapes --strategy tiled --window_size 1024 --stride 512
"""

import os
import glob
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm

import numpy as np
import torch
import cv2
from PIL import Image

from transformers import Sam3Processor, Sam3Model


# ============================================================================
# Configuration
# ============================================================================

# Prompt Strategy Levels - Different levels of specificity for SAM3 text prompts

# Level 1: Baseline - Simple, generic terms
PROMPTS_L1_BASELINE = {
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    11: "person",
    12: "rider",
    17: "motorcycle",
    18: "bicycle",
}

# Level 2: Descriptive - More descriptive terms
PROMPTS_L2_DESCRIPTIVE = {
    4: "metal fence",
    5: "street pole",
    6: "traffic signal light",
    7: "road traffic sign",
    11: "pedestrian person",
    12: "cyclist rider",
    17: "motorcycle vehicle",
    18: "bicycle bike",
}

# Level 3: Physical - Physical attributes emphasized
PROMPTS_L3_PHYSICAL = {
    4: "thin vertical metal fence barrier",
    5: "tall narrow cylindrical pole post",
    6: "small traffic light signal hanging",
    7: "flat rectangular traffic sign board",
    11: "standing upright human person",
    12: "person riding bicycle or motorcycle",
    17: "two-wheeled motorcycle with engine",
    18: "two-wheeled pedal bicycle",
}

# Level 4: Specific - Context and location details
PROMPTS_L4_SPECIFIC = {
    4: "street fence barrier along road edge",
    5: "utility pole or street lamp post",
    6: "traffic light mounted on pole or wire",
    7: "traffic sign mounted on pole showing information",
    11: "person pedestrian walking on street or sidewalk",
    12: "person riding on bicycle or motorcycle",
    17: "motorized motorcycle on road",
    18: "bicycle on road or bike lane",
}

# Map prompt strategy names to their dictionaries
PROMPT_STRATEGIES = {
    "L1_Baseline": PROMPTS_L1_BASELINE,
    "L2_Descriptive": PROMPTS_L2_DESCRIPTIVE,
    "L3_Physical": PROMPTS_L3_PHYSICAL,
    "L4_Specific": PROMPTS_L4_SPECIFIC
}

# Default prompts (for backward compatibility)
THIN_CLASS_PROMPTS = PROMPTS_L1_BASELINE

REFINED_THIN_CLASS_PROMPTS = {
    4: "vertical fence barrier",          # "fence" can sometimes be vague lines
    5: "vertical street utility pole",    # "pole" is too generic (could be a stick)
    6: "traffic light housing device",    # Focuses on the physical box, not just the light
    7: "traffic sign on post",            # Helps separate sign from billboard ads
    11: "pedestrian walking",             # "person" can match billboard photos
    12: "person riding vehicle",          # "rider" is vague (horse? bike?)
    17: "motorcycle vehicle",             # Adds "vehicle" to imply 3D object
    18: "physical bicycle vehicle",       # Explicitly excludes 2D markings
}

# ============================================================================
# SAM3 Model Loading
# ============================================================================

def load_sam3_model(device: str = "cuda"):
    """Load SAM3 model and processor"""
    print(f"Loading SAM3 model on {device}...")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model.eval()
    print("SAM3 model loaded successfully")
    return model, processor


# ============================================================================
# Core SAM3 Inference
# ============================================================================

@torch.no_grad()
def generate_boundary_for_crop(
    image: Image.Image, 
    model: Sam3Model, 
    processor: Sam3Processor, 
    device: str,
    prompts: Dict[int, str] = None
) -> np.ndarray:
    """
    Generate boundary map for a single image/crop.
    
    This is the core SAM3 inference logic extracted from the notebook.
    
    Args:
        image: PIL Image (RGB)
        model: SAM3 model
        processor: SAM3 processor
        device: Device to run on
        prompts: Dictionary of {train_id: text_prompt}. If None, uses THIN_CLASS_PROMPTS
        
    Returns:
        Binary boundary map (H, W) as bool numpy array
    """
    if prompts is None:
        prompts = THIN_CLASS_PROMPTS
    
    w, h = image.size
    combined_boundary = np.zeros((h, w), dtype=np.uint8)
    
    # Pre-compute vision embeddings once for efficiency
    img_inputs = processor(images=image, return_tensors="pt").to(device)
    vision_embeds = model.get_vision_features(pixel_values=img_inputs.pixel_values)
    
    # Process each thin class
    for train_id, text_prompt in prompts.items():
        # Prepare text inputs
        text_inputs = processor(text=text_prompt, return_tensors="pt").to(device)
        
        # Run SAM3 Decoder with pre-computed vision features
        outputs = model(
            vision_embeds=vision_embeds,
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
        )
        
        # Post-process to get binary masks
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.4,       # Confidence threshold
            mask_threshold=0.5,  # Pixel probability threshold
            target_sizes=img_inputs.get("original_sizes").tolist()
        )[0]
        
        masks = results["masks"]
        
        # Skip if nothing found
        if len(masks) == 0:
            continue
        
        # Convert to numpy if needed
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        
        # Combine all instances of this class (logical OR)
        class_mask = np.any(masks, axis=0).astype(np.uint8)
        
        # Extract boundaries using Canny edge detection
        edges = cv2.Canny(class_mask * 255, 100, 200)
        
        # Accumulate
        combined_boundary |= (edges > 0)
    
    # Dilate slightly to make boundaries more learnable (3px thick)
    kernel = np.ones((3, 3), np.uint8)
    dilated_boundary = cv2.dilate(combined_boundary, kernel, iterations=1)
    
    return dilated_boundary.astype(bool)


# ============================================================================
# Strategy 1: Baseline (No Crop)
# ============================================================================

def strategy_baseline(
    image: Image.Image,
    model: Sam3Model,
    processor: Sam3Processor,
    device: str
) -> np.ndarray:
    """
    Baseline strategy: Process full image directly.
    
    This may miss small/distant objects due to resolution loss when
    SAM3 resizes the image internally.
    """
    return generate_boundary_for_crop(image, model, processor, device)


# ============================================================================
# Strategy 2: Multi-Crop (Fixed Grid)
# ============================================================================

def strategy_multi_crop(
    image: Image.Image,
    model: Sam3Model,
    processor: Sam3Processor,
    device: str,
    grid_size: Tuple[int, int] = (2, 2)
) -> np.ndarray:
    """
    Multi-crop strategy: Split image into fixed grid and process each crop.
    
    Args:
        image: Full resolution image
        grid_size: (rows, cols) - e.g., (2, 2) for 4 crops, (2, 1) for 2 crops
        
    Returns:
        Stitched boundary map at full resolution
    """
    w, h = image.size
    rows, cols = grid_size
    
    crop_height = h // rows
    crop_width = w // cols
    
    # Initialize full-size output
    full_boundary = np.zeros((h, w), dtype=bool)
    
    # Process each crop
    for i in range(rows):
        for j in range(cols):
            # Calculate crop boundaries
            y1 = i * crop_height
            y2 = (i + 1) * crop_height if i < rows - 1 else h
            x1 = j * crop_width
            x2 = (j + 1) * crop_width if j < cols - 1 else w
            
            # Extract crop
            crop = image.crop((x1, y1, x2, y2))
            
            # Process crop
            crop_boundary = generate_boundary_for_crop(crop, model, processor, device)
            
            # Place back into full image (logical OR for overlaps)
            full_boundary[y1:y2, x1:x2] |= crop_boundary
    
    return full_boundary


# ============================================================================
# Strategy 3: Tiled (Sliding Window)
# ============================================================================

def strategy_tiled(
    image: Image.Image,
    model: Sam3Model,
    processor: Sam3Processor,
    device: str,
    window_size: int = 1024,
    stride: int = 512
) -> np.ndarray:
    """
    Tiled sliding window strategy: Process overlapping crops.
    
    This captures objects that might be on crop boundaries in multi-crop.
    Uses logical OR for overlapping regions.
    
    Args:
        image: Full resolution image
        window_size: Size of sliding window (square)
        stride: Step size for sliding window
        
    Returns:
        Stitched boundary map at full resolution
    """
    w, h = image.size
    full_boundary = np.zeros((h, w), dtype=bool)
    
    # Generate window positions
    y_positions = list(range(0, h - window_size + 1, stride))
    x_positions = list(range(0, w - window_size + 1, stride))
    
    # Ensure we cover the right and bottom edges
    if y_positions[-1] + window_size < h:
        y_positions.append(h - window_size)
    if x_positions[-1] + window_size < w:
        x_positions.append(w - window_size)
    
    total_windows = len(y_positions) * len(x_positions)
    
    # Process each window
    with tqdm(total=total_windows, desc="Processing tiles", leave=False) as pbar:
        for y in y_positions:
            for x in x_positions:
                # Extract window
                window = image.crop((x, y, x + window_size, y + window_size))
                
                # Process window
                window_boundary = generate_boundary_for_crop(window, model, processor, device)
                
                # Merge into full image (logical OR)
                full_boundary[y:y+window_size, x:x+window_size] |= window_boundary
                
                pbar.update(1)
    
    return full_boundary


# ============================================================================
# Strategy Dispatcher
# ============================================================================

def generate_boundary_map(
    image: Image.Image,
    model: Sam3Model,
    processor: Sam3Processor,
    device: str,
    strategy: str,
    **strategy_kwargs
) -> np.ndarray:
    """
    Generate boundary map using specified strategy.
    
    Args:
        image: Input image
        model: SAM3 model
        processor: SAM3 processor
        device: Device
        strategy: One of ['baseline', 'multi_crop', 'tiled']
        **strategy_kwargs: Strategy-specific arguments
        
    Returns:
        Binary boundary map
    """
    if strategy == "baseline":
        return strategy_baseline(image, model, processor, device)
    
    elif strategy == "multi_crop":
        grid_size = strategy_kwargs.get("grid_size", (2, 2))
        return strategy_multi_crop(image, model, processor, device, grid_size)
    
    elif strategy == "tiled":
        window_size = strategy_kwargs.get("window_size", 1024)
        stride = strategy_kwargs.get("stride", 512)
        return strategy_tiled(image, model, processor, device, window_size, stride)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ============================================================================
# Dataset Processing
# ============================================================================

def find_cityscapes_images(root: Path, splits: List[str] = ["train", "val"]) -> List[Path]:
    """Find all Cityscapes images"""
    # Handle potential nested structure
    if (root / "leftImg8bit_trainvaltest" / "leftImg8bit").exists():
        img_dir = root / "leftImg8bit_trainvaltest" / "leftImg8bit"
    else:
        img_dir = root / "leftImg8bit"
    
    image_paths = []
    for split in splits:
        search_pattern = str(img_dir / split / "*" / "*_leftImg8bit.png")
        found = glob.glob(search_pattern)
        image_paths.extend([Path(p) for p in found])
    
    return sorted(image_paths), img_dir


def get_output_path(img_path: Path, img_dir: Path, output_dir: Path) -> Path:
    """Get output path preserving directory structure"""
    try:
        relative_path = img_path.relative_to(img_dir)
    except ValueError:
        return None
    
    # Change extension to .npy
    output_path = output_dir / relative_path.parent / (img_path.stem + ".npy")
    return output_path


def process_dataset(
    data_root: Path,
    output_dir_name: str,
    strategy: str,
    splits: List[str],
    overwrite: bool,
    device: str,
    **strategy_kwargs
):
    """Process entire dataset with specified strategy"""
    
    # Load model
    model, processor = load_sam3_model(device)
    
    # Find images
    image_paths, img_dir = find_cityscapes_images(data_root, splits)
    print(f"Found {len(image_paths)} images across splits: {splits}")
    
    # Setup output directory
    output_dir = data_root / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Process images
    skipped = 0
    processed = 0
    errors = 0
    
    for img_path in tqdm(image_paths, desc=f"SAM3 Boundaries ({strategy})"):
        # Get output path
        output_path = get_output_path(img_path, img_dir, output_dir)
        if output_path is None:
            continue
        
        # Skip if exists and not overwriting
        if output_path.exists() and not overwrite:
            skipped += 1
            continue
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load image
            image = Image.open(img_path).convert("RGB")
            
            # Generate boundary map
            boundary_map = generate_boundary_map(
                image, model, processor, device, strategy, **strategy_kwargs
            )
            
            # Save
            np.save(output_path, boundary_map)
            processed += 1
            
        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")
            errors += 1
            continue
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Processed: {processed}")
    print(f"  Skipped (already exist): {skipped}")
    print(f"  Errors: {errors}")
    print(f"{'='*60}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate SAM3 boundary masks with multiple strategies"
    )
    
    # Required arguments
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to Cityscapes dataset root"
    )
    
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=["baseline", "multi_crop", "tiled"],
        help="Inference strategy to use"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory name (default: sam3_boundary_{strategy})"
    )
    
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val"],
        help="Dataset splits to process (default: train val)"
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing boundary masks"
    )
    
    # Prompt strategy selection
    parser.add_argument(
        "--prompt_level",
        type=str,
        default="L1_Baseline",
        choices=["L1_Baseline", "L2_Descriptive", "L3_Physical", "L4_Specific"],
        help="Prompt strategy level (default: L1_Baseline)"
    )
    
    # Multi-crop strategy arguments
    parser.add_argument(
        "--grid_size",
        type=int,
        nargs=2,
        default=[2, 2],
        metavar=("ROWS", "COLS"),
        help="Grid size for multi_crop strategy (default: 2 2)"
    )
    
    # Tiled strategy arguments
    parser.add_argument(
        "--window_size",
        type=int,
        default=1024,
        help="Window size for tiled strategy (default: 1024)"
    )
    
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Stride for tiled strategy (default: 512)"
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available)"
    )
    
    args = parser.parse_args()
    
    # Set output directory name
    if args.output_dir is None:
        args.output_dir = f"sam3_boundary_{args.strategy}_{args.prompt_level}"
    
    # Set global prompts based on selected level
    global THIN_CLASS_PROMPTS
    THIN_CLASS_PROMPTS = PROMPT_STRATEGIES[args.prompt_level]
    
    # Print configuration
    print("\n" + "="*60)
    print("SAM3 Boundary Generation Configuration")
    print("="*60)
    print(f"Data root: {args.data_root}")
    print(f"Strategy: {args.strategy}")
    print(f"Prompt level: {args.prompt_level}")
    print(f"Output directory: {args.output_dir}")
    print(f"Splits: {args.splits}")
    print(f"Device: {args.device}")
    
    if args.strategy == "multi_crop":
        print(f"Grid size: {args.grid_size[0]}x{args.grid_size[1]}")
    elif args.strategy == "tiled":
        print(f"Window size: {args.window_size}")
        print(f"Stride: {args.stride}")
    
    print("="*60 + "\n")
    
    # Prepare strategy kwargs
    strategy_kwargs = {}
    if args.strategy == "multi_crop":
        strategy_kwargs["grid_size"] = tuple(args.grid_size)
    elif args.strategy == "tiled":
        strategy_kwargs["window_size"] = args.window_size
        strategy_kwargs["stride"] = args.stride
    
    # Process dataset
    process_dataset(
        data_root=Path(args.data_root),
        output_dir_name=args.output_dir,
        strategy=args.strategy,
        splits=args.splits,
        overwrite=args.overwrite,
        device=args.device,
        **strategy_kwargs
    )


if __name__ == "__main__":
    main()
