"""
Dataset utilities for Cityscapes segmentation analysis and benchmarking.

Provides functions for:
- Loading and managing Cityscapes validation splits
- Dataset filtering (hard/easy subsets based on model performance or pixel statistics)
- IoU calculation and evaluation metrics
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from PIL import Image


def list_cityscapes_split(root: Path, split: str = "val") -> Tuple[List[Path], Path]:
    """
    List all images in a Cityscapes split.

    Args:
        root: Path to Cityscapes root directory
        split: 'train', 'val', or 'test'

    Returns:
        Tuple of (sorted list of image paths, ground truth directory path)
    """
    img_dir = root / "leftImg8bit_trainvaltest" / "leftImg8bit" / split
    gt_dir = root / "gtFine_trainvaltest" / "gtFine" / split

    image_paths = sorted(img_dir.rglob("*_leftImg8bit.png"))
    print(f"Found {len(image_paths)} images in split='{split}'")
    return image_paths, gt_dir


def make_cityscapes_dataframe(root: Path, split: str = "val") -> pd.DataFrame:
    """
    Build a DataFrame with all metadata for a Cityscapes split.

    Args:
        root: Path to Cityscapes root directory
        split: 'train', 'val', or 'test'

    Returns:
        DataFrame with columns: city, image_id, image_path, labelIds_path, etc.
    """
    img_dir = root / "leftImg8bit_trainvaltest" / "leftImg8bit" / split
    gt_dir = root / "gtFine_trainvaltest" / "gtFine" / split

    records = []
    for img_path in sorted(img_dir.rglob("*_leftImg8bit.png")):
        city = img_path.parent.name
        stem = img_path.name.replace("_leftImg8bit.png", "")

        label_ids = gt_dir / city / f"{stem}_gtFine_labelIds.png"
        instance_ids = gt_dir / city / f"{stem}_gtFine_instanceIds.png"
        color_png = gt_dir / city / f"{stem}_gtFine_color.png"
        polygons_json = gt_dir / city / f"{stem}_gtFine_polygons.json"

        records.append(
            {
                "city": city,
                "image_id": stem,
                "image_path": str(img_path),
                "labelIds_path": str(label_ids),
                "instanceIds_path": str(instance_ids),
                "color_path": str(color_png),
                "polygons_path": str(polygons_json),
            }
        )

    df = pd.DataFrame.from_records(records)
    return df


def identify_thin_objects(
    gt_path: Path,
    thin_threshold: int = 20,
    target_classes: Optional[List[int]] = None
) -> bool:
    """
    Detect if an image contains thin objects (e.g., thin poles, traffic signs, persons).

    Uses connected component analysis to identify small but elongated regions
    (high aspect ratio or small total pixels).

    Args:
        gt_path: Path to ground truth label image (labelIds.png)
        thin_threshold: Maximum pixel count to consider "thin"
        target_classes: List of trainIds to check. If None, checks all non-background.
                       Example: [11] for person, [13] for traffic sign, [20] for pole

    Returns:
        True if image contains thin objects, False otherwise
    """
    from scipy import ndimage
    
    try:
        gt_image = Image.open(gt_path)
        gt_array = np.array(gt_image, dtype=np.int32)
    except Exception as e:
        print(f"Error loading {gt_path}: {e}")
        return False

    if target_classes is None:
        # Check all non-zero pixels
        target_mask = gt_array > 0
    else:
        target_mask = np.isin(gt_array, target_classes)

    if not target_mask.any():
        return False

    # Label connected components
    labeled, num_features = ndimage.label(target_mask)

    # Check each component
    for component_id in range(1, num_features + 1):
        component_mask = labeled == component_id
        num_pixels = component_mask.sum()

        # Thin object: very small component
        if num_pixels < thin_threshold:
            return True

        # Alternative: check aspect ratio for thin elongated objects
        positions = np.argwhere(component_mask)
        if len(positions) > 0:
            height = positions[:, 0].max() - positions[:, 0].min() + 1
            width = positions[:, 1].max() - positions[:, 1].min() + 1
            aspect_ratio = max(height, width) / (min(height, width) + 1e-6)
            
            # High aspect ratio (thin and tall/long)
            if aspect_ratio > 5 and num_pixels < thin_threshold * 2:
                return True

    return False


def create_hard_subset(
    df: pd.DataFrame,
    thin_threshold: int = 20,
    target_classes: Optional[List[int]] = None,
    subset_name: str = "hard"
) -> pd.DataFrame:
    """
    Filter dataset to create a "hard" subset containing only images with thin objects.

    Args:
        df: Cityscapes DataFrame from make_cityscapes_dataframe()
        thin_threshold: Maximum pixel count to consider "thin"
        target_classes: List of trainIds to check. If None, all thin objects.
        subset_name: Name for the subset column

    Returns:
        Filtered DataFrame containing only hard images
    """
    from tqdm.auto import tqdm
    
    hard_mask = []
    print(f"Identifying thin objects (threshold={thin_threshold})...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        gt_path = Path(row["labelIds_path"])
        has_thin = identify_thin_objects(gt_path, thin_threshold, target_classes)
        hard_mask.append(has_thin)

    hard_df = df[hard_mask].copy()
    hard_df[f"{subset_name}_set"] = True
    print(f"Found {len(hard_df)} hard images out of {len(df)} total")
    
    return hard_df


# ============================================================================
# IoU Metrics & Evaluation
# ============================================================================

def compute_iou(pred: np.ndarray, gt: np.ndarray, num_classes: int = 19, 
                ignore_index: int = 255) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-class IoU between prediction and ground truth.

    Args:
        pred: Predicted segmentation (H, W) with integer class IDs
        gt: Ground truth segmentation (H, W) with integer class IDs
        num_classes: Number of classes to evaluate
        ignore_index: Class ID to ignore (usually 255)

    Returns:
        Tuple of (per_class_iou array, valid_classes_mask)
    """
    pred = pred.astype(np.int64)
    gt = gt.astype(np.int64)

    iou_array = np.full(num_classes, np.nan, dtype=np.float32)

    for class_id in range(num_classes):
        if class_id == ignore_index:
            continue

        pred_mask = pred == class_id
        gt_mask = gt == class_id

        intersection = (pred_mask & gt_mask).sum()
        union = (pred_mask | gt_mask).sum()

        if union == 0:
            # Class not present in GT
            iou_array[class_id] = np.nan
        else:
            iou_array[class_id] = intersection / union

    return iou_array


def load_benchmark_results(results_dir: Path) -> pd.DataFrame:
    """
    Load all per-image IoU results from benchmark CSV files.

    Expected filename format: {ModelName}_per_image_iou.csv

    Args:
        results_dir: Directory containing _per_image_iou.csv files

    Returns:
        Combined DataFrame with 'model' column
    """
    all_files = list(results_dir.glob("*_per_image_iou.csv"))

    if not all_files:
        print("No result files found! Make sure you ran the benchmark notebooks.")
        return pd.DataFrame()

    dfs = []
    for f in all_files:
        # Filename format: {ModelName}_per_image_iou.csv
        model_name = f.name.replace("_per_image_iou.csv", "").replace("Wrapper", "")

        df = pd.read_csv(f)
        df["model"] = model_name
        dfs.append(df)
        print(f"Loaded {len(df)} rows for model: {model_name}")

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


# ============================================================================
# Subset-based Evaluation
# ============================================================================

def evaluate_on_subset(
    results_df: pd.DataFrame,
    subset_image_ids: List[str],
    subset_name: str = "subset"
) -> pd.DataFrame:
    """
    Filter benchmark results to a specific subset of images.

    Args:
        results_df: DataFrame from load_benchmark_results()
        subset_image_ids: List of image_id values to include
        subset_name: Display name for the subset

    Returns:
        Filtered DataFrame
    """
    subset_mask = results_df["image_id"].isin(subset_image_ids)
    subset_results = results_df[subset_mask].copy()
    subset_results[f"evaluated_on_{subset_name}"] = True
    
    print(f"Evaluating on {subset_name}: {len(subset_results)} results")
    return subset_results


def compare_subsets(
    all_results: pd.DataFrame,
    hard_subset_image_ids: List[str],
    class_columns: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Generate side-by-side comparison of model performance on All vs Hard subsets.

    Args:
        all_results: Full benchmark results DataFrame
        hard_subset_image_ids: List of "hard" image IDs
        class_columns: List of class column names. If None, infers from data.

    Returns:
        Dict with keys:
        - 'all_cityscapes': Results on full validation set
        - 'hard_cityscapes': Results on hard subset
        - 'comparison': Aggregated comparison table
    """
    if class_columns is None:
        # Infer class columns (exclude metadata)
        exclude = {'image_id', 'city', 'model', 'image_mIoU'}
        class_columns = [c for c in all_results.columns if c not in exclude]

    # Full dataset results
    all_cityscapes = all_results.copy()
    if 'image_mIoU' not in all_cityscapes.columns:
        all_cityscapes['image_mIoU'] = all_cityscapes[class_columns].mean(axis=1)

    # Hard subset results
    hard_mask = all_results['image_id'].isin(hard_subset_image_ids)
    hard_cityscapes = all_results[hard_mask].copy()
    if 'image_mIoU' not in hard_cityscapes.columns:
        hard_cityscapes['image_mIoU'] = hard_cityscapes[class_columns].mean(axis=1)

    # Aggregated comparison
    all_stats = all_cityscapes.groupby('model')[['image_mIoU'] + class_columns].mean()
    all_stats.columns = [f'{c}_all' for c in all_stats.columns]

    hard_stats = hard_cityscapes.groupby('model')[['image_mIoU'] + class_columns].mean()
    hard_stats.columns = [f'{c}_hard' for c in hard_stats.columns]

    comparison = pd.concat([all_stats, hard_stats], axis=1)
    
    # Calculate degradation
    comparison['miou_degradation'] = (
        comparison['image_mIoU_all'] - comparison['image_mIoU_hard']
    )
    comparison = comparison.sort_values('miou_degradation', ascending=False)

    return {
        'all_cityscapes': all_cityscapes,
        'hard_cityscapes': hard_cityscapes,
        'comparison': comparison
    }
