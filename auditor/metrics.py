"""Metrics for evaluating mask boundary adherence."""

import cv2
import numpy as np
from typing import Tuple

from .config import AuditConfig


def compute_boundary_adherence(
    image: np.ndarray,
    mask: np.ndarray,
    config: AuditConfig
) -> float:
    """Compute boundary adherence score measuring edge sharpness of a mask.
    
    This metric evaluates how well a mask aligns with image edges by measuring
    the intensity of gradients at the mask boundary. High scores indicate sharp,
    well-defined edges; low scores indicate blurry or poorly-defined boundaries.
    
    Algorithm:
        1. Convert image to grayscale if needed.
        2. Compute gradient magnitude using Sobel operators (x and y directions).
        3. Extract mask boundary using morphological gradient.
        4. Dilate boundary for robust gradient sampling.
        5. Mask the gradient image with the boundary mask.
        6. Return mean gradient intensity at boundary.
    
    Args:
        image: Input image (H, W, 3) or (H, W).
        mask: Binary mask (H, W) where True/1 indicates object region.
        config: AuditConfig containing Sobel kernel size and dilation parameters.
    
    Returns:
        Float in range [0, 255] representing mean gradient magnitude at mask boundary.
        Higher values indicate sharper edges.
    
    Example:
        >>> image = cv2.imread("city.png")
        >>> mask = cv2.imread("mask.png", 0) > 128
        >>> config = AuditConfig(data_root=Path("data"))
        >>> score = compute_boundary_adherence(image, mask, config)
    """
    # Convert image to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.astype(np.uint8)

    # Ensure mask is uint8 binary
    mask_binary = (mask > 0).astype(np.uint8) * 255

    # Compute gradient magnitude using Sobel
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=config.sobel_ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=config.sobel_ksize)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

    # Extract boundary using morphological gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    boundary_mask = cv2.morphologyEx(mask_binary, cv2.MORPH_GRADIENT, kernel)

    # Dilate boundary for better gradient sampling
    if config.boundary_dilation > 0:
        kernel_dilate = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.boundary_dilation * 2 + 1, config.boundary_dilation * 2 + 1)
        )
        boundary_mask = cv2.dilate(boundary_mask, kernel_dilate, iterations=1)

    # Mask the gradient image with boundary
    boundary_gradients = gradient_magnitude[boundary_mask > 0]

    # Return mean intensity (handle empty boundary)
    if len(boundary_gradients) == 0:
        return 0.0

    return float(np.mean(boundary_gradients))


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Intersection over Union (IoU) between two binary masks.
    
    Standard metric for binary segmentation evaluation.
    
    Args:
        mask1: First binary mask (H, W).
        mask2: Second binary mask (H, W).
    
    Returns:
        Float in range [0, 1]. Returns 0.0 if both masks are empty.
    
    Example:
        >>> mask_a = np.random.rand(256, 256) > 0.5
        >>> mask_b = np.random.rand(256, 256) > 0.5
        >>> iou = compute_iou(mask_a, mask_b)
    """
    mask1_bin = (mask1 > 0).astype(bool)
    mask2_bin = (mask2 > 0).astype(bool)

    intersection = np.logical_and(mask1_bin, mask2_bin).sum()
    union = np.logical_or(mask1_bin, mask2_bin).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def compute_mask_metrics(
    mask1: np.ndarray,
    mask2: np.ndarray
) -> Tuple[float, float]:
    """Compute multiple mask comparison metrics.
    
    Args:
        mask1: First binary mask (H, W).
        mask2: Second binary mask (H, W).
    
    Returns:
        Tuple of (iou, dice_coefficient).
    """
    mask1_bin = (mask1 > 0).astype(bool)
    mask2_bin = (mask2 > 0).astype(bool)

    intersection = np.logical_and(mask1_bin, mask2_bin).sum()
    union = np.logical_or(mask1_bin, mask2_bin).sum()

    # IOU
    iou = float(intersection / union) if union > 0 else 0.0

    # Dice coefficient
    dice = float(2 * intersection / (mask1_bin.sum() + mask2_bin.sum())) \
        if (mask1_bin.sum() + mask2_bin.sum()) > 0 else 0.0

    return iou, dice
