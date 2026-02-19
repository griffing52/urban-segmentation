"""Configuration module for the auditor pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class AuditConfig:
    """Configuration for the boundary adherence auditor.
    
    This dataclass encapsulates all hyperparameters needed for analyzing
    mask boundary quality, particularly for thin objects like poles,
    traffic lights, and traffic signs.
    
    Attributes:
        data_root: Path to the Cityscapes dataset root directory.
        target_classes: List of Cityscapes class IDs to audit. Default: [17 (pole),
            18 (traffic light), 19 (traffic sign)].
        sobel_ksize: Kernel size for Sobel edge detection. Should be odd.
        boundary_dilation: Number of pixels to dilate the mask boundary for better
            gradient analysis.
        sam_checkpoint: Path to the SAM (Segment Anything Model) checkpoint file.
        output_dir: Path to directory where audit results will be saved.
    """

    data_root: Path
    """Path to Cityscapes dataset root."""

    target_classes: List[int] = field(default_factory=lambda: [17, 18, 19])
    """Cityscapes class IDs: 17=pole, 18=traffic light, 19=traffic sign."""

    sobel_ksize: int = 3
    """Kernel size for Sobel edge detection (must be odd)."""

    boundary_dilation: int = 2
    """Pixels to dilate mask boundary for boundary analysis."""

    sam_checkpoint: Path = Path("models/sam_vit_h.pth")
    """Path to SAM model checkpoint."""

    output_dir: Path = Path("outputs/audit_results")
    """Path to directory for saving audit results."""

    def make_dirs(self) -> None:
        """Create output directories if they don't exist.
        
        Raises:
            OSError: If directory creation fails due to permission issues.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
