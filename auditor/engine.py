"""Auditor engine for evaluating mask quality."""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np

from .config import AuditConfig
from .metrics import compute_boundary_adherence, compute_iou, compute_mask_metrics


class MockSAM:
    """Mock SAM predictor for testing without actual model.
    
    Returns a slightly eroded version of the input mask to simulate
    a "tighter" prediction compared to the input mask.
    """

    def __init__(self):
        """Initialize MockSAM."""
        pass

    def predict(
        self,
        image: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Generate a mock prediction by eroding the provided mask.
        
        Args:
            image: Input image (unused in mock).
            box: Bounding box (unused in mock).
            mask: Reference mask to erode.
        
        Returns:
            Eroded version of input mask.
        """
        if mask is None:
            raise ValueError("MockSAM requires a mask input")

        mask_binary = (mask > 0).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = cv2.erode(mask_binary, kernel, iterations=1)
        return eroded


@dataclass
class AuditResult:
    """Result from auditing a single sample.
    
    Attributes:
        image_id: Identifier of the image being audited.
        class_id: Cityscapes class ID of the object.
        boundary_score_human: Boundary adherence score for human annotation.
        boundary_score_sam: Boundary adherence score for SAM prediction.
        iou: Intersection over Union between human and SAM masks.
        dice: Dice coefficient between human and SAM masks.
        human_area: Area (pixel count) of human annotation.
        sam_area: Area (pixel count) of SAM prediction.
    """

    image_id: str
    class_id: int
    boundary_score_human: float
    boundary_score_sam: float
    iou: float
    dice: float
    human_area: int
    sam_area: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return asdict(self)


class Auditor:
    """Auditor for evaluating mask boundary adherence and quality.
    
    This class orchestrates the audit pipeline: it takes human annotations
    and SAM predictions, computes boundary adherence scores, and collects
    detailed metrics for analysis.
    
    Args:
        config: AuditConfig containing hyperparameters.
        model: SAM predictor or MockSAM for generating predictions.
    """

    def __init__(self, config: AuditConfig, model: Optional[Any] = None):
        """Initialize the Auditor.
        
        Args:
            config: AuditConfig with hyperparameters.
            model: SAM predictor. If None, uses MockSAM.
        """
        self.config = config
        self.model = model if model is not None else MockSAM()

    def get_bbox_from_mask(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Extract bounding box from binary mask.
        
        Args:
            mask: Binary mask (H, W).
        
        Returns:
            Bounding box as [x_min, y_min, x_max, y_max] or None if mask is empty.
        """
        mask_binary = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        x_min, y_min = np.inf, np.inf
        x_max, y_max = -np.inf, -np.inf

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

    def audit_sample(
        self,
        image: np.ndarray,
        human_mask: np.ndarray,
        class_id: int,
        image_id: str = "unknown"
    ) -> AuditResult:
        """Audit a single sample by comparing human and SAM masks.
        
        Pipeline:
            1. Extract bounding box from human mask.
            2. Generate SAM prediction using the bounding box.
            3. Compute boundary adherence scores for both masks.
            4. Calculate overlap metrics (IoU, Dice).
            5. Return comprehensive audit result.
        
        Args:
            image: Input image (H, W, 3).
            human_mask: Human-annotated binary mask (H, W).
            class_id: Cityscapes class ID of the object.
            image_id: Identifier for the image (default: "unknown").
        
        Returns:
            AuditResult containing all computed metrics.
        
        Raises:
            ValueError: If bounding box cannot be extracted or SAM prediction fails.
        """
        # Step 1: Extract bounding box from human mask
        bbox = self.get_bbox_from_mask(human_mask)
        if bbox is None:
            raise ValueError(f"Could not extract bounding box from mask for image {image_id}")

        # Step 2: Get SAM prediction
        sam_mask = self.model.predict(image=image, box=bbox, mask=human_mask)

        # Step 3: Compute boundary adherence scores
        boundary_score_human = compute_boundary_adherence(image, human_mask, self.config)
        boundary_score_sam = compute_boundary_adherence(image, sam_mask, self.config)

        # Step 4: Compute overlap metrics
        iou, dice = compute_mask_metrics(human_mask, sam_mask)

        # Calculate areas
        human_area = int(np.sum(human_mask > 0))
        sam_area = int(np.sum(sam_mask > 0))

        # Step 5: Return result
        result = AuditResult(
            image_id=image_id,
            class_id=class_id,
            boundary_score_human=boundary_score_human,
            boundary_score_sam=boundary_score_sam,
            iou=iou,
            dice=dice,
            human_area=human_area,
            sam_area=sam_area
        )

        return result
