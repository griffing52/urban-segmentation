#!/usr/bin/env python3
"""
SAM3 Boundary Mask Generation from Configuration Pairs

Generates SAM3 boundary masks for different method and prompt level combinations,
saving organized results for later comparison and analysis.

Uses the same direct SAM3 strategies as visualize_prompts.ipynb and visualize_strategies.ipynb

Usage:
    # Generate baseline + all prompts
    python generate_sam3_masks_from_configs.py --data_root ./data/cityscapes --method baseline --prompts L1_Baseline L2_Descriptive

    # Generate multi_crop with specific grid sizes
    python generate_sam3_masks_from_configs.py --data_root ./data/cityscapes --method multi_crop --prompts L2_Descriptive L3_Physical \
        --grid_sizes 2 2

    # Generate all combinations
    python generate_sam3_masks_from_configs.py --data_root ./data/cityscapes --method baseline multi_crop tiled \
        --prompts L1_Baseline L2_Descriptive L3_Physical L4_Specific
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from PIL import Image
import torch

# Import SAM3 functions directly
from generate_sam3_masks import (
    load_sam3_model,
    strategy_baseline,
    strategy_multi_crop,
    strategy_tiled,
    generate_boundary_for_crop,
    PROMPT_STRATEGIES,
    find_cityscapes_images
)


@dataclass
class ConfigPair:
    """Single configuration: method + prompt + params"""
    method: str
    prompt_level: str
    params: Dict = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}
    
    def __str__(self) -> str:
        """Readable config name"""
        param_str = "_".join([f"{k}={v}" for k, v in self.params.items()])
        if param_str:
            return f"{self.method}_{self.prompt_level}_{param_str}"
        return f"{self.method}_{self.prompt_level}"
    
    def get_output_dir(self) -> str:
        """Directory name for this config"""
        return f"sam3_{str(self)}"


class SAM3ConfigGenerator:
    """Generate SAM3 masks for multiple configurations using direct SAM3 strategies"""
    
    def __init__(self, data_root: str, output_base_dir: str = "./generated_masks"):
        self.data_root = Path(data_root)
        self.output_base = Path(output_base_dir)
        self.output_base.mkdir(parents=True, exist_ok=True)
        
        # Load SAM3 model once
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✓ Using device: {device}")
        self.model, self.processor = load_sam3_model(device)
        self.device = device
    
    def generate_config_pairs(
        self,
        methods: List[str],
        prompts: List[str],
        grid_sizes: List[Tuple[int, int]] = None,
        strides: List[int] = None
    ) -> List[ConfigPair]:
        """
        Generate all config pairs from methods and prompts.
        
        Args:
            methods: List of methods (baseline, multi_crop, tiled)
            prompts: List of prompt levels (L1_Baseline, L2_Descriptive, etc.)
            grid_sizes: Grid sizes for multi_crop [[2,2], [1,2], etc.]
            strides: Strides for tiled [256, 512, 768, 1024]
            
        Returns:
            List of ConfigPair objects
        """
        configs = []
        
        for prompt in prompts:
            for method in methods:
                if method == "baseline":
                    configs.append(ConfigPair(method="baseline", prompt_level=prompt))
                
                elif method == "multi_crop":
                    grid_list = grid_sizes if grid_sizes else [[2, 2]]
                    for grid in grid_list:
                        configs.append(ConfigPair(
                            method="multi_crop",
                            prompt_level=prompt,
                            params={"grid_size": tuple(grid) if isinstance(grid, list) else grid}
                        ))
                
                elif method == "tiled":
                    stride_list = strides if strides else [512]
                    for stride in stride_list:
                        configs.append(ConfigPair(
                            method="tiled",
                            prompt_level=prompt,
                            params={"window_size": 1024, "stride": stride}
                        ))
        
        return configs
    
    def _generate_for_config(self, config: ConfigPair, image: Image.Image) -> np.ndarray:
        """Generate boundary mask for single image using specified config"""
        
        # Get prompts for this level
        if config.prompt_level not in PROMPT_STRATEGIES:
            raise ValueError(f"Unknown prompt level: {config.prompt_level}")
        
        prompts = PROMPT_STRATEGIES[config.prompt_level]
        
        # Run appropriate strategy
        if config.method == "baseline":
            boundary = strategy_baseline(image, self.model, self.processor, self.device)
        
        elif config.method == "multi_crop":
            grid_size = config.params.get("grid_size", (2, 2))
            boundary = strategy_multi_crop(
                image, self.model, self.processor, self.device,
                grid_size=grid_size
            )
        
        elif config.method == "tiled":
            window_size = config.params.get("window_size", 1024)
            stride = config.params.get("stride", 512)
            boundary = strategy_tiled(
                image, self.model, self.processor, self.device,
                window_size=window_size,
                stride=stride
            )
        
        else:
            raise ValueError(f"Unknown method: {config.method}")
        
        return boundary
    
    def run_generation(
        self,
        configs: List[ConfigPair],
        splits: List[str] = None,
        dry_run: bool = False,
        verbose: bool = True,
        max_images: int = None,
        cities: List[str] = None
    ) -> Dict:
        """
        Generate masks for all configurations on all images.
        
        Args:
            configs: List of ConfigPair objects
            splits: Dataset splits to process (default: ["val"])
            dry_run: Print what would be done without executing
            verbose: Print progress
            max_images: Limit number of images per split to process
            cities: Specific cities to process (if None, processes all)
            
        Returns:
            Summary dict with success/failure counts
        """
        if splits is None:
            splits = ["val"]
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "data_root": str(self.data_root),
            "splits": splits,
            "total_configs": len(configs),
            "max_images": max_images,
            "cities": cities,
            "completed": 0,
            "failed": 0,
            "skipped": 0,
            "configs": []
        }
        
        print("\n" + "="*80)
        print(f"SAM3 Mask Generation: {len(configs)} configurations")
        print(f"Splits: {', '.join(splits)}")
        if max_images:
            print(f"Max images per split: {max_images}")
        if cities:
            print(f"Cities: {', '.join(cities)}")
        print("="*80 + "\n")
        
        for config_idx, config in enumerate(configs, 1):
            config_name = str(config)
            config_output_base = self.output_base / config.get_output_dir()
            
            print(f"[{config_idx}/{len(configs)}] {config_name}")
            
            if dry_run:
                print(f"  [DRY RUN]\n")
                summary["skipped"] += 1
                summary["configs"].append({
                    "config": config_name,
                    "status": "dry_run",
                    "output_dir": str(config_output_base)
                })
                continue
            
            try:
                images_processed = 0
                images_failed = 0
                
                # Process each split
                for split in splits:
                    split_output = config_output_base / split
                    split_output.mkdir(parents=True, exist_ok=True)
                    
                    # Get all images for this split
                    image_list, _ = find_cityscapes_images(self.data_root, splits=[split])
                    
                    if not image_list:
                        raise RuntimeError(f"No images found in {split} split")
                    
                    # Filter by cities if specified
                    if cities:
                        image_list = [img for img in image_list if img.parent.name in cities]
                    
                    # Limit number of images if specified
                    if max_images:
                        image_list = image_list[:max_images]
                    
                    print(f"  Processing {split} split: {len(image_list)} images")
                    
                    # Process each image
                    for img_idx, img_path in enumerate(image_list, 1):
                        try:
                            # Load image
                            image = Image.open(img_path).convert("RGB")
                            
                            # Generate boundary
                            boundary = self._generate_for_config(config, image)
                            
                            # Save as .npy in split-specific subdirectory
                            img_stem = img_path.stem
                            city_name = img_path.parent.name
                            city_output = split_output / city_name
                            city_output.mkdir(parents=True, exist_ok=True)
                            
                            output_path = city_output / f"{img_stem}_boundary.npy"
                            np.save(output_path, boundary.astype(np.uint8))
                            
                            images_processed += 1
                            
                            if verbose and img_idx % 10 == 0:
                                print(f"    [{img_idx}/{len(image_list)}]")
                        
                        except Exception as e:
                            images_failed += 1
                            if verbose:
                                print(f"    ✗ Failed to process {img_path.name}: {str(e)[:50]}")
                
                print(f"  ✓ Completed: {images_processed} saved, {images_failed} failed")
                print(f"  Output: {config_output_base}")
                
                summary["completed"] += 1
                summary["configs"].append({
                    "config": config_name,
                    "status": "completed",
                    "output_dir": str(config_output_base),
                    "images_processed": images_processed,
                    "images_failed": images_failed
                })
            
            except Exception as e:
                print(f"  ✗ ERROR: {str(e)[:100]}")
                summary["failed"] += 1
                summary["configs"].append({
                    "config": config_name,
                    "status": "error",
                    "output_dir": str(config_output_base),
                    "error": str(e)[:200]
                })
            
            print()
        
        # Print summary
        print("="*80)
        print("Generation Summary")
        print("="*80)
        print(f"Completed: {summary['completed']}/{len(configs)}")
        print(f"Failed:    {summary['failed']}/{len(configs)}")
        print(f"Dry Run:   {summary['skipped']}/{len(configs)}")
        print(f"Output:    {self.output_base}/")
        print("="*80 + "\n")
        
        # Save metadata
        results_file = self.output_base / "generation_metadata.json"
        with open(results_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Metadata saved: {results_file}")
        
        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Generate SAM3 masks from configuration pairs using direct SAM3 strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:

  Generate baseline with different prompts:
    python generate_sam3_masks_from_configs.py --data_root ./data/cityscapes \\
      --method baseline --prompts L1_Baseline L2_Descriptive L3_Physical

  Generate multi_crop with different prompts and grid sizes:
    python generate_sam3_masks_from_configs.py --data_root ./data/cityscapes \\
      --method multi_crop --prompts L2_Descriptive L3_Physical \\
      --grid_sizes 2 2 --grid_sizes 1 2

  Generate all three methods:
    python generate_sam3_masks_from_configs.py --data_root ./data/cityscapes \\
      --method baseline multi_crop tiled \\
      --prompts L1_Baseline L2_Descriptive L3_Physical L4_Specific
        """
    )
    
    # Input arguments
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to Cityscapes dataset root"
    )
    
    parser.add_argument(
        "--method",
        type=str,
        nargs="+",
        default=["baseline"],
        choices=["baseline", "multi_crop", "tiled"],
        help="Generation method(s) to use"
    )
    
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=["L1_Baseline", "L2_Descriptive"],
        choices=["L1_Baseline", "L2_Descriptive", "L3_Physical", "L4_Specific"],
        help="Prompt level(s) to use"
    )
    
    parser.add_argument(
        "--grid_sizes",
        type=int,
        nargs="+",
        action="append",
        help="Grid sizes for multi_crop (e.g., --grid_sizes 2 2 --grid_sizes 1 2)"
    )
    
    parser.add_argument(
        "--strides",
        type=int,
        nargs="+",
        default=[512],
        help="Strides for tiled method (default: 512)"
    )
    
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["val"],
        choices=["train", "val", "test"],
        help="Dataset splits to process (default: val)"
    )
    
    parser.add_argument(
        "--max_configs",
        type=int,
        default=None,
        help="Limit number of configs to generate (useful for testing)"
    )
    
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Limit number of images per split to process (useful for quick testing)"
    )
    
    parser.add_argument(
        "--cities",
        type=str,
        nargs="+",
        default=None,
        help="Specific cities to process (e.g., frankfurt lindau munster). If not specified, all cities in split are used."
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_masks",
        help="Base output directory for generated masks"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be done without executing"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress"
    )
    
    args = parser.parse_args()
    
    # Parse grid sizes
    grid_sizes = None
    if args.grid_sizes:
        grid_sizes = [args.grid_sizes[i:i+2] for i in range(0, len(args.grid_sizes), 2)]
        grid_sizes = [g for g in grid_sizes if len(g) == 2]
    
    # Create generator (loads model once)
    gen = SAM3ConfigGenerator(args.data_root, args.output_dir)
    
    # Generate configs
    configs = gen.generate_config_pairs(
        methods=args.method,
        prompts=args.prompts,
        grid_sizes=grid_sizes,
        strides=args.strides
    )
    
    # Limit configs if requested
    if args.max_configs:
        configs = configs[:args.max_configs]
    
    print(f"\n{len(configs)} configuration pairs to generate:")
    for config in configs:
        print(f"  - {config}")
    print()
    
    # Run generation
    gen.run_generation(
        configs, 
        splits=args.splits, 
        dry_run=args.dry_run, 
        verbose=args.verbose,
        max_images=args.max_images,
        cities=args.cities
    )


if __name__ == "__main__":
    main()
