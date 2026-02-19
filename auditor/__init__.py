"""Auditor module for evaluating mask boundary adherence.

This package provides tools for auditing segmentation mask quality,
particularly focused on evaluating boundary adherence for thin objects
like poles, traffic lights, and traffic signs.

Main components:
    - config: AuditConfig dataclass for hyperparameters
    - metrics: Boundary adherence and overlap metrics
    - engine: Auditor class orchestrating the audit pipeline
"""

from .config import AuditConfig
from .engine import Auditor, MockSAM, AuditResult
from .metrics import compute_boundary_adherence, compute_iou, compute_mask_metrics

__all__ = [
    "AuditConfig",
    "Auditor",
    "MockSAM",
    "AuditResult",
    "compute_boundary_adherence",
    "compute_iou",
    "compute_mask_metrics",
]
