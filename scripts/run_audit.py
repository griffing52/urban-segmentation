"""Main script to run the audit pipeline on Cityscapes dataset.

Audits boundary adherence of generated SAM3 masks by comparing them with
human annotations from the Cityscapes dataset. Supports comparing multiple
strategies and prompt levels.
"""

import csv
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from auditor.config import AuditConfig
from auditor.engine import Auditor, AuditResult
from src.dataset_utils import list_cityscapes_split, identify_thin_objects


class CityscapesAuditDataset:
    """Cityscapes dataset for auditing with generated SAM3 masks.
    
    Loads real Cityscapes validation images and matches them with:
    - Human annotations (ground truth)
    - Generated SAM3 masks from specified strategy/prompt
    """

    def __init__(
        self,
        data_root: Path,
        generated_masks_root: Path,
        strategy: str = "baseline",
        prompt_level: str = "L1_Baseline",
        split: str = "val",
        target_classes: Optional[List[int]] = None
    ):
        """Initialize Cityscapes audit dataset.
        
        Args:
            data_root: Path to Cityscapes dataset root.
            generated_masks_root: Path to generated_masks directory.
            strategy: SAM3 strategy used (baseline, multi_crop, tiled).
            prompt_level: Prompt level (L1_Baseline, L2_Descriptive, etc.).
            split: Dataset split ('val', 'train', 'test').
            target_classes: Filter to specific Cityscapes class IDs.
                Defaults: [17 (pole), 18 (traffic light), 19 (traffic sign)].
        """
        self.data_root = Path(data_root)
        self.generated_masks_root = Path(generated_masks_root)
        self.strategy = strategy
        self.prompt_level = prompt_level
        self.split = split
        self.target_classes = target_classes or [17, 18, 19]

        # Build mask directory name (handles multi_crop and tiled parameters)
        # Examples: sam3_baseline_L1_Baseline, sam3_multi_crop_L2_Descriptive_grid_size=(2, 2)
        self.mask_dir = self._build_mask_dir_name()
        
        # Check if masks are organized by split (val/train/test subdirectory)
        potential_mask_root = self.generated_masks_root / self.mask_dir / split
        if not potential_mask_root.exists():
            # Try without split subdirectory
            potential_mask_root = self.generated_masks_root / self.mask_dir
        
        self.mask_root = potential_mask_root

        # Get list of images
        self.image_paths, self.gt_root = list_cityscapes_split(self.data_root, split)

    def _build_mask_dir_name(self) -> str:
        """Build mask directory name based on strategy and prompt level.
        
        The prompt_level parameter now includes any additional parameters
        (e.g., 'L1_Baseline' or 'L2_Descriptive_grid_size=(2, 2)').
        """
        return f"sam3_{self.strategy}_{self.prompt_level}"

    def __len__(self) -> int:
        """Return number of valid image samples."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Optional[Dict]:
        """Load image, masks, and metadata.
        
        Returns:
            Dict with 'image_id', 'image', 'mask_human', 'mask_sam', 'class_id'
            or None if data cannot be loaded.
        """
        image_path = self.image_paths[idx]
        city = image_path.parent.name
        stem = image_path.stem.replace("_leftImg8bit", "")
        
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Load ground truth (labelIds)
            gt_path = self.gt_root / city / f"{stem}_gtFine_labelIds.png"
            if not gt_path.exists():
                return None
            gt_image = Image.open(gt_path)
            gt_labels = np.array(gt_image, dtype=np.int32)

            # Load SAM3 mask
            mask_path = self.mask_root / city / f"{stem}_leftImg8bit_boundary.npy"
            if not mask_path.exists():
                return None
            sam_mask = np.load(mask_path)

            # For thin object classes, use instance-based filtering
            # Get human mask by combining all target classes
            human_mask = np.zeros_like(gt_labels, dtype=np.uint8)
            for class_id in self.target_classes:
                human_mask[gt_labels == class_id] = 255

            # Skip if no target classes present
            if human_mask.max() == 0:
                return None

            # Determine primary class from majority
            primary_class = None
            for class_id in self.target_classes:
                class_pixels = (gt_labels == class_id).sum()
                if class_pixels > 0:
                    primary_class = class_id
                    break

            return {
                "image_id": stem,
                "city": city,
                "image": image,
                "mask_human": human_mask,
                "mask_sam": sam_mask,
                "class_id": primary_class if primary_class is not None else self.target_classes[0],
                "image_path": str(image_path)
            }

        except Exception as e:
            return None


def load_config(
    data_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    target_classes: Optional[List[int]] = None
) -> AuditConfig:
    """Load or create audit configuration.
    
    Args:
        data_root: Path to Cityscapes dataset. If None, uses default.
        output_dir: Path to save results. If None, uses default.
        target_classes: List of class IDs to audit. Default: [17, 18, 19].
    
    Returns:
        AuditConfig instance.
    """
    # Default paths relative to project root
    project_root = Path(__file__).parent.parent
    
    if data_root is None:
        data_root = project_root / "data" / "cityscapes"
    else:
        data_root = Path(data_root)
    
    if output_dir is None:
        output_dir = project_root / "outputs" / "audit_results"
    else:
        output_dir = Path(output_dir)
    
    if target_classes is None:
        target_classes = [17, 18, 19]

    sam_checkpoint = project_root / "models" / "sam_vit_h.pth"

    config = AuditConfig(
        data_root=data_root,
        target_classes=target_classes,
        sobel_ksize=3,
        boundary_dilation=2,
        sam_checkpoint=sam_checkpoint,
        output_dir=output_dir
    )

    return config


def save_results_to_csv(results: List[AuditResult], output_path: Path) -> None:
    """Save audit results to CSV file.
    
    Args:
        results: List of AuditResult objects.
        output_path: Path to output CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as csvfile:
        if not results:
            return

        fieldnames = list(results[0].to_dict().keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result.to_dict())

    print(f"Results saved to {output_path}")


def run_audit(
    config: AuditConfig,
    generated_masks_root: Optional[Path] = None,
    strategies: Optional[List[Tuple[str, str]]] = None,
    num_samples: Optional[int] = None,
    split: str = "val"
) -> Dict[str, List[AuditResult]]:
    """Run the audit pipeline comparing multiple SAM3 strategies.
    
    Args:
        config: AuditConfig with hyperparameters.
        generated_masks_root: Path to generated_masks directory.
            If None, assumes it's in project root.
        strategies: List of (method, prompt_level) tuples to compare.
            Examples: [("baseline", "L1_Baseline"), ("baseline", "L2_Descriptive")]
            If None, uses single baseline with L1 prompt.
        num_samples: Max number of samples to process. If None, processes all.
        split: Dataset split to use ('val', 'train', 'test').
    
    Returns:
        Dictionary mapping strategy names to list of AuditResult objects.
    """
    # Create output directories
    config.make_dirs()

    # Set default paths (generated_masks is at project root, not under data)
    if generated_masks_root is None:
        # Navigate up from data_root to project root, then to generated_masks
        # data_root = /path/to/project/data/cityscapes
        # project_root = /path/to/project
        project_root = config.data_root.parent.parent
        generated_masks_root = project_root / "generated_masks"
    else:
        generated_masks_root = Path(generated_masks_root)

    if strategies is None:
        strategies = [("baseline", "L1_Baseline")]

    # Initialize auditor (no real model, uses MockSAM for now)
    auditor = Auditor(config)

    # Dictionary to collect results per strategy
    all_results: Dict[str, List[AuditResult]] = {}
    errors_by_strategy: Dict[str, List[Tuple[int, str]]] = {}

    # Run audit for each strategy
    for method, prompt_level in strategies:
        strategy_name = f"sam3_{method}_{prompt_level}"
        print(f"\n{'='*70}")
        print(f"Auditing strategy: {strategy_name}")
        print(f"{'='*70}")

        try:
            # Initialize dataset for this strategy
            dataset = CityscapesAuditDataset(
                data_root=config.data_root,
                generated_masks_root=generated_masks_root,
                strategy=method,
                prompt_level=prompt_level,
                split=split,
                target_classes=config.target_classes
            )

            # Check if mask directory exists
            if not dataset.mask_root.exists():
                print(f"⚠ Mask directory not found: {dataset.mask_root}")
                print(f"  Skipping strategy {strategy_name}")
                continue

            results: List[AuditResult] = []
            errors: List[Tuple[int, str]] = []

            # Process dataset
            total_samples = len(dataset) if num_samples is None else min(num_samples, len(dataset))
            sample_indices = range(min(num_samples if num_samples else len(dataset), len(dataset)))

            with tqdm(sample_indices, desc=f"{strategy_name}", unit="sample") as pbar:
                processed = 0
                for idx in pbar:
                    try:
                        sample = dataset[idx]
                        
                        # Skip if sample couldn't be loaded
                        if sample is None:
                            continue

                        # Run audit
                        result = auditor.audit_sample(
                            image=sample["image"],
                            human_mask=sample["mask_human"],
                            class_id=sample["class_id"],
                            image_id=sample["image_id"]
                        )

                        results.append(result)
                        processed += 1
                        pbar.set_postfix({"processed": processed})

                    except Exception as e:
                        errors.append((idx, str(e)))
                        tqdm.write(f"⚠ Error processing sample {idx}: {e}")

            all_results[strategy_name] = results
            errors_by_strategy[strategy_name] = errors

            # Save results for this strategy
            output_csv = config.output_dir / f"{strategy_name}_results.csv"
            save_results_to_csv(results, output_csv)

            # Print summary for this strategy
            print_strategy_summary(strategy_name, results, errors)

        except Exception as e:
            print(f"✗ Error initializing dataset for {strategy_name}: {e}")
            continue

    # Print overall comparison if multiple strategies
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("COMPARATIVE ANALYSIS ACROSS STRATEGIES")
        print(f"{'='*70}")
        print_comparative_summary(all_results)

    return all_results


def print_strategy_summary(strategy_name: str, results: List[AuditResult], errors: List[Tuple]) -> None:
    """Print audit summary for a single strategy.
    
    Args:
        strategy_name: Name of the strategy.
        results: List of audit results.
        errors: List of errors encountered.
    """
    if not results:
        print(f"✗ No results for {strategy_name}")
        return

    boundary_human = np.array([r.boundary_score_human for r in results])
    boundary_sam = np.array([r.boundary_score_sam for r in results])
    ious = np.array([r.iou for r in results])
    dices = np.array([r.dice for r in results])

    print(f"\n✓ Processed: {len(results)} samples, Errors: {len(errors)}")
    
    print(f"\nBoundary Adherence - Human Annotations:")
    print(f"  Mean: {boundary_human.mean():.4f}  |  Std: {boundary_human.std():.4f}")
    print(f"  Min:  {boundary_human.min():.4f}  |  Max: {boundary_human.max():.4f}")

    print(f"\nBoundary Adherence - SAM Predictions:")
    print(f"  Mean: {boundary_sam.mean():.4f}  |  Std: {boundary_sam.std():.4f}")
    print(f"  Min:  {boundary_sam.min():.4f}  |  Max: {boundary_sam.max():.4f}")

    print(f"\nOverlap Metrics:")
    print(f"  IoU   Mean: {ious.mean():.4f}  |  Std: {ious.std():.4f}")
    print(f"  Dice  Mean: {dices.mean():.4f}  |  Std: {dices.std():.4f}")


def print_comparative_summary(all_results: Dict[str, List[AuditResult]]) -> None:
    """Print comparison across multiple strategies.
    
    Args:
        all_results: Dictionary mapping strategy names to results.
    """
    metrics_by_strategy = {}

    for strategy_name, results in all_results.items():
        if not results:
            continue

        boundary_human = np.array([r.boundary_score_human for r in results])
        boundary_sam = np.array([r.boundary_score_sam for r in results])
        ious = np.array([r.iou for r in results])

        metrics_by_strategy[strategy_name] = {
            "n_samples": len(results),
            "boundary_human_mean": boundary_human.mean(),
            "boundary_sam_mean": boundary_sam.mean(),
            "iou_mean": ious.mean(),
        }

    # Print summary table
    print(f"\n{'Strategy':<50} {'Samples':<10} {'BND_Human':<12} {'BND_SAM':<12} {'IoU':<10}")
    print("-" * 94)
    
    for strategy_name, metrics in sorted(metrics_by_strategy.items()):
        print(
            f"{strategy_name:<50} {metrics['n_samples']:<10} "
            f"{metrics['boundary_human_mean']:<12.4f} {metrics['boundary_sam_mean']:<12.4f} "
            f"{metrics['iou_mean']:<10.4f}"
        )


def discover_strategies(generated_masks_root: Path) -> List[str]:
    """Discover all available mask generation strategies.
    
    Scans the generated_masks directory for sam3_* subdirectories
    and returns their full names (including parameters).
    
    Args:
        generated_masks_root: Path to generated_masks directory.
    
    Returns:
        List of strategy directory names (e.g., 'sam3_baseline_L1_Baseline').
    """
    if not generated_masks_root.exists():
        print(f"⚠ Generated masks directory not found: {generated_masks_root}")
        return []
    
    strategies = []
    for item in sorted(generated_masks_root.iterdir()):
        if item.is_dir() and item.name.startswith('sam3_'):
            # Skip analysis and metadata directories
            if item.name in ['analysis', 'metadata']:
                continue
            strategies.append(item.name)
    
    return strategies


def parse_strategy_name(strategy_dir_name: str) -> Optional[Tuple[str, str]]:
    """Parse a strategy directory name into (method, full_prompt_params).
    
    Examples:
        'sam3_baseline_L1_Baseline' -> ('baseline', 'L1_Baseline')
        'sam3_multi_crop_L2_Descriptive_grid_size=(2, 2)' -> ('multi_crop', 'L2_Descriptive_grid_size=(2, 2)')
        'sam3_tiled_L3_Physical_window_size=1024_stride=512' -> ('tiled', 'L3_Physical_window_size=1024_stride=512')
    
    Args:
        strategy_dir_name: Directory name from generated_masks.
    
    Returns:
        Tuple of (method, prompt_with_params) or None if parsing fails.
    """
    # Remove 'sam3_' prefix
    if not strategy_dir_name.startswith('sam3_'):
        return None
    
    name = strategy_dir_name.replace('sam3_', '')
    parts = name.split('_')
    
    if len(parts) < 2:
        return None
    
    # Extract method (baseline, multi_crop, tiled)
    method = parts[0]
    
    # Handle multi_crop as two-word method
    if method == 'multi' and len(parts) > 1 and parts[1] == 'crop':
        method = 'multi_crop'
        prompt_with_params = '_'.join(parts[2:])
    else:
        prompt_with_params = '_'.join(parts[1:])
    
    return (method, prompt_with_params)


if __name__ == "__main__":
    # Load configuration
    config = load_config()

    # Get project root and generated_masks path
    project_root = config.data_root.parent.parent
    generated_masks_root = project_root / "generated_masks"

    # Discover all available strategies from generated_masks directory
    print(f"\n{'='*70}")
    print("Discovering available strategies...")
    print(f"{'='*70}")
    
    strategy_dirs = discover_strategies(generated_masks_root)
    
    if not strategy_dirs:
        print("⚠ No strategies found in generated_masks directory.")
        print(f"  Looking in: {generated_masks_root}")
        sys.exit(1)
    
    print(f"\nFound {len(strategy_dirs)} strategies:")
    for strategy_dir in strategy_dirs:
        print(f"  - {strategy_dir}")
    
    # Parse strategy names to extract method and prompt level
    strategies_to_audit = []
    for strategy_dir in strategy_dirs:
        parsed = parse_strategy_name(strategy_dir)
        if parsed:
            strategies_to_audit.append(parsed)
        else:
            print(f"  ✗ Could not parse: {strategy_dir}")
    
    if not strategies_to_audit:
        print("\n⚠ No valid strategies to audit.")
        sys.exit(1)
    
    print(f"\nParsed {len(strategies_to_audit)} strategies:")
    for method, prompt in strategies_to_audit:
        full_name = f"sam3_{method}_{prompt}"
        print(f"  ✓ {full_name}")
    
    print(f"\nReady to audit {len(strategies_to_audit)} strategies")

    # Run audit pipeline comparing strategies
    all_results = run_audit(
        config=config,
        generated_masks_root=generated_masks_root,
        strategies=strategies_to_audit,
        num_samples=50,  # Process first 50 samples per strategy
        split="val"
    )

    print(f"\n{'='*70}")
    print("Audit pipeline complete!")
    print(f"Results saved to: {config.output_dir}")
    print(f"{'='*70}")
