"""
Model benchmarking utilities for semantic segmentation on Cityscapes.

Provides:
- Abstract base class for model wrappers
- Generic inference loop
- IoU calculation and result saving
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from PIL import Image
import torch


# ============================================================================
# Abstract Model Interface
# ============================================================================

class BaseSegmentationModel(ABC):
    """
    Abstract base class for segmentation models.
    
    All models must implement the predict() method which takes a PIL Image
    and returns a segmentation mask.
    """

    @abstractmethod
    def predict(self, image: Image.Image) -> np.ndarray:
        """
        Run inference on a single image.

        Args:
            image (PIL.Image.Image): Input image in RGB mode.

        Returns:
            np.ndarray: Predicted segmentation mask of shape (H, W).
                        Values should be integer trainIds (0-18 for Cityscapes).
        """
        pass


# ============================================================================
# Inference & Evaluation
# ============================================================================

def run_inference_over_df(
    df: pd.DataFrame,
    model: BaseSegmentationModel,
    pred_root: Path,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Generic inference loop using BaseSegmentationModel.

    Saves predictions as .npy files and returns updated DataFrame with paths.

    Args:
        df: DataFrame with 'image_path' column
        model: Instance of BaseSegmentationModel
        pred_root: Directory to save .npy predictions
        overwrite: If True, re-run inference even if file exists

    Returns:
        Copy of df with 'pred_trainIds_path' column added
    """
    from tqdm.auto import tqdm

    df = df.copy()
    pred_paths = []

    print(f"Running inference with {model.__class__.__name__}...")
    print(f"Saving results to: {pred_root}")

    pred_root.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = Path(row["image_path"])
        image_id = row["image_id"]

        # Save as {image_id}_trainIds.npy
        out_path = pred_root / f"{image_id}_trainIds.npy"

        if out_path.exists() and not overwrite:
            pred_paths.append(str(out_path))
            continue

        # Load and predict
        img = Image.open(img_path).convert("RGB")
        pred_trainids = model.predict(img)

        # Save prediction
        np.save(out_path, pred_trainids)
        pred_paths.append(str(out_path))

    df["pred_trainIds_path"] = pred_paths
    return df


def compute_per_image_iou(
    pred_path: Path,
    gt_path: Path,
    num_classes: int = 19,
    ignore_index: int = 255,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute IoU for a single image across all classes.

    Args:
        pred_path: Path to prediction .npy or image file
        gt_path: Path to ground truth labelIds.png
        num_classes: Number of classes
        ignore_index: Index to ignore in evaluation
        class_names: Optional list of class names for output dict keys

    Returns:
        Dict mapping class names (or indices) to IoU values
    """
    # Load prediction
    if str(pred_path).endswith('.npy'):
        pred = np.load(pred_path).astype(np.int32)
    else:
        pred = np.array(Image.open(pred_path), dtype=np.int32)

    # Load ground truth
    gt = np.array(Image.open(gt_path), dtype=np.int32)

    # Ensure same shape
    if pred.shape != gt.shape:
        pred = Image.fromarray(pred.astype(np.uint8))
        pred = np.array(pred.resize(gt.shape[::-1], Image.NEAREST))

    iou_dict = {}

    for class_id in range(num_classes):
        if class_id == ignore_index:
            continue

        pred_mask = (pred == class_id).astype(bool)
        gt_mask = (gt == class_id).astype(bool)

        intersection = (pred_mask & gt_mask).sum()
        union = (pred_mask | gt_mask).sum()

        if union == 0:
            iou = np.nan
        else:
            iou = float(intersection) / float(union)

        # Use class name if provided, otherwise use numeric ID
        key = class_names[class_id] if class_names else f"class_{class_id:02d}"
        iou_dict[key] = iou

    return iou_dict


def evaluate_model_on_split(
    pred_dir: Path,
    gt_dir: Path,
    split_df: pd.DataFrame,
    model_name: str,
    num_classes: int = 19,
    ignore_index: int = 255,
    class_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Evaluate a model's predictions on a validation split.

    Computes per-image and per-class IoU metrics.

    Args:
        pred_dir: Directory containing prediction .npy files
        gt_dir: Root ground truth directory
        split_df: DataFrame with 'image_id' and city info
        model_name: Name of the model (for output)
        num_classes: Number of classes
        ignore_index: Index to ignore
        class_names: Optional class names

    Returns:
        DataFrame with per-image IoU scores
    """
    from tqdm.auto import tqdm

    records = []

    print(f"Evaluating {model_name}...")
    for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
        image_id = row["image_id"]
        city = row["city"]

        pred_path = pred_dir / f"{image_id}_trainIds.npy"
        gt_path = gt_dir / city / f"{image_id}_gtFine_labelIds.png"

        if not pred_path.exists():
            print(f"Warning: Missing prediction for {image_id}")
            continue
        if not gt_path.exists():
            print(f"Warning: Missing GT for {image_id}")
            continue

        iou_dict = compute_per_image_iou(
            pred_path, gt_path, num_classes, ignore_index, class_names
        )
        iou_dict["image_id"] = image_id
        iou_dict["city"] = city

        records.append(iou_dict)

    results_df = pd.DataFrame.from_records(records)
    results_df["model"] = model_name

    # Calculate mIoU for each image
    class_cols = [c for c in results_df.columns if c not in ["image_id", "city", "model"]]
    results_df["image_mIoU"] = results_df[class_cols].mean(axis=1)

    return results_df


def save_results_csv(
    results_df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
) -> Path:
    """
    Save evaluation results to CSV.

    Args:
        results_df: DataFrame from evaluate_model_on_split()
        output_dir: Where to save the CSV
        model_name: Used in filename

    Returns:
        Path to saved CSV file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / f"{model_name}_per_image_iou.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")
    
    return csv_path
