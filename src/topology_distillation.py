"""
Topological distillation utilities for thin-structure segmentation.

This module provides:
- Differentiable soft skeletonization for batched masks
- SAM3-guided soft-clDice loss
- Thin-class filtering helpers to avoid over-regularizing large classes
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


CITYSCAPES_TRAIN_ID_TO_NAME: Dict[int, str] = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic_light",
    7: "traffic_sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
}

CITYSCAPES_NAME_TO_TRAIN_ID: Dict[str, int] = {
    name: class_id for class_id, name in CITYSCAPES_TRAIN_ID_TO_NAME.items()
}

DEFAULT_THIN_CLASS_IDS: Tuple[int, ...] = (4, 5, 6, 7, 11, 12, 17, 18)


@dataclass(frozen=True)
class ThinClassFilter:
    """
    Utility for restricting topology loss to thin classes only.

    This prevents applying clDice regularization to large texture-dominant classes
    (e.g., road/building), where topology constraints can be counterproductive.
    """

    thin_class_ids: Tuple[int, ...] = DEFAULT_THIN_CLASS_IDS

    @classmethod
    def from_class_names(
        cls,
        class_names: Sequence[str],
        name_to_id: Optional[Mapping[str, int]] = None,
    ) -> "ThinClassFilter":
        lookup = name_to_id or CITYSCAPES_NAME_TO_TRAIN_ID
        thin_class_ids = tuple(lookup[name] for name in class_names)
        return cls(thin_class_ids=thin_class_ids)

    def select(self, class_ids: Iterable[int]) -> List[int]:
        thin_set = set(self.thin_class_ids)
        return [class_id for class_id in class_ids if class_id in thin_set]

    def classes_present_in_labels(self, labels: torch.Tensor) -> List[int]:
        unique = torch.unique(labels)
        thin_set = set(self.thin_class_ids)
        return [int(class_id.item()) for class_id in unique if int(class_id.item()) in thin_set]


def _ensure_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        return x
    if x.ndim == 3:
        return x.unsqueeze(1)
    raise ValueError(f"Expected tensor with 3 or 4 dims, got shape={tuple(x.shape)}")


class SoftSkeletonization(nn.Module):
    """
    Differentiable soft skeletonization with iterative min/max pooling.

    Input is expected to be a probability map in [0, 1] with shape:
    - (B, 1, H, W) or
    - (B, H, W)
    """

    def __init__(self, iterations: int = 25, kernel_size: int = 3):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        if iterations < 1:
            raise ValueError("iterations must be >= 1.")
        self.iterations = iterations
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def _soft_erode(self, x: torch.Tensor) -> torch.Tensor:
        return -F.max_pool2d(-x, kernel_size=self.kernel_size, stride=1, padding=self.padding)

    def _soft_dilate(self, x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(x, kernel_size=self.kernel_size, stride=1, padding=self.padding)

    def _soft_open(self, x: torch.Tensor) -> torch.Tensor:
        return self._soft_dilate(self._soft_erode(x))

    def forward(self, prob_map: torch.Tensor) -> torch.Tensor:
        img = _ensure_bchw(prob_map).clamp(0.0, 1.0)

        opened = self._soft_open(img)
        skeleton = F.relu(img - opened)

        for _ in range(self.iterations - 1):
            img = self._soft_erode(img)
            opened = self._soft_open(img)
            delta = F.relu(img - opened)
            skeleton = skeleton + F.relu(delta - skeleton * delta)

        return skeleton


class SAM3GuidedCLDiceLoss(nn.Module):
    """
    SAM3-guided soft-clDice between student prediction and SAM3 pseudo-labels.

    For a class c, with student probability P_c and teacher/SAM3 mask T_c:
      - Topology Precision (tprec): overlap of student skeleton with teacher mask
      - Topology Sensitivity (tsens): overlap of teacher skeleton with student mask
      - clDice = 2 * tprec * tsens / (tprec + tsens)
      - Loss  = 1 - clDice
    """

    def __init__(self, skeleton_iterations: int = 25, eps: float = 1e-6):
        super().__init__()
        self.skeletonizer = SoftSkeletonization(iterations=skeleton_iterations)
        self.eps = eps

    def forward(
        self,
        student_logits: torch.Tensor,
        sam3_pseudo_mask: torch.Tensor,
        class_index: int,
        return_details: bool = False,
    ):
        if student_logits.ndim != 4:
            raise ValueError("student_logits must have shape (B, C, H, W)")
        if class_index < 0 or class_index >= student_logits.shape[1]:
            raise ValueError(f"class_index {class_index} is out of bounds for C={student_logits.shape[1]}")

        student_prob = torch.softmax(student_logits, dim=1)[:, class_index : class_index + 1]
        teacher_prob = _ensure_bchw(sam3_pseudo_mask).float().clamp(0.0, 1.0)

        if teacher_prob.shape[-2:] != student_prob.shape[-2:]:
            teacher_prob = F.interpolate(
                teacher_prob,
                size=student_prob.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        student_skeleton = self.skeletonizer(student_prob)
        teacher_skeleton = self.skeletonizer(teacher_prob)

        reduce_dims = (1, 2, 3)

        # Topology precision: how much of student's predicted skeleton is
        # supported by SAM3 teacher mask.
        tprec_num = torch.sum(student_skeleton * teacher_prob, dim=reduce_dims)
        tprec_den = torch.sum(student_skeleton, dim=reduce_dims)
        topology_precision = (tprec_num + self.eps) / (tprec_den + self.eps)

        # Topology sensitivity: how much of SAM3 teacher skeleton is recovered
        # by the student's predicted probability map.
        tsens_num = torch.sum(teacher_skeleton * student_prob, dim=reduce_dims)
        tsens_den = torch.sum(teacher_skeleton, dim=reduce_dims)
        topology_sensitivity = (tsens_num + self.eps) / (tsens_den + self.eps)

        cldice = (2.0 * topology_precision * topology_sensitivity + self.eps) / (
            topology_precision + topology_sensitivity + self.eps
        )
        loss = 1.0 - cldice
        mean_loss = loss.mean()

        if not return_details:
            return mean_loss

        details = {
            "cldice": cldice.mean().detach(),
            "topology_precision": topology_precision.mean().detach(),
            "topology_sensitivity": topology_sensitivity.mean().detach(),
        }
        return mean_loss, details


def compute_thin_class_topology_loss(
    student_logits: torch.Tensor,
    sam3_masks_by_class: Mapping[int, torch.Tensor],
    thin_class_filter: ThinClassFilter,
    cldice_loss: Optional[SAM3GuidedCLDiceLoss] = None,
    candidate_class_ids: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute mean SAM3-guided clDice loss over selected thin classes only.

    Args:
        student_logits: (B, C, H, W) student logits.
        sam3_masks_by_class: Mapping {class_id: pseudo-mask tensor (B,H,W) or (B,1,H,W)}.
        thin_class_filter: Thin class filter utility.
        cldice_loss: Optional instantiated loss module.
        candidate_class_ids: Optional class IDs to consider. If None, use sam3 mapping keys.

    Returns:
        (mean_topology_loss, details)
    """
    device = student_logits.device
    criterion = cldice_loss or SAM3GuidedCLDiceLoss()

    available = list(candidate_class_ids) if candidate_class_ids is not None else list(sam3_masks_by_class.keys())
    selected = thin_class_filter.select(available)

    per_class_losses: List[torch.Tensor] = []
    per_class_cldice: List[torch.Tensor] = []

    for class_id in selected:
        if class_id not in sam3_masks_by_class:
            continue
        class_loss, details = criterion(
            student_logits=student_logits,
            sam3_pseudo_mask=sam3_masks_by_class[class_id],
            class_index=class_id,
            return_details=True,
        )
        per_class_losses.append(class_loss)
        per_class_cldice.append(details["cldice"])

    if not per_class_losses:
        zero = torch.zeros((), device=device, dtype=student_logits.dtype)
        return zero, {"selected_thin_classes": torch.tensor(0, device=device), "mean_cldice": zero}

    stacked_losses = torch.stack(per_class_losses)
    stacked_cldice = torch.stack(per_class_cldice)
    mean_loss = stacked_losses.mean()

    return mean_loss, {
        "selected_thin_classes": torch.tensor(len(per_class_losses), device=device),
        "mean_cldice": stacked_cldice.mean(),
    }
