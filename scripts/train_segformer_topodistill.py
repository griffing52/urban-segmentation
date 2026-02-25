"""
SegFormer + SAM3 Topological Distillation Training Script.

This experiment combines:
  L = L_ce + lambda_topology * L_clDice

where L_clDice is only applied to configurable thin classes to preserve topology
on poles/riders/traffic assets without over-regularizing large classes.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from src.dataset_utils import list_cityscapes_split
from src.topology_distillation import (
    DEFAULT_THIN_CLASS_IDS,
    SAM3GuidedCLDiceLoss,
    ThinClassFilter,
    compute_thin_class_topology_loss,
)


class CityscapesTopologyDistillDataset(Dataset):
    """Cityscapes dataset that returns labels plus SAM3 pseudo-label masks per class."""

    def __init__(
        self,
        root_dir: str,
        sam3_root: str,
        split: str,
        feature_extractor: SegformerImageProcessor,
        thin_class_ids: Tuple[int, ...],
        sam3_file_suffix: str = "_leftImg8bit_topology.npz",
    ):
        self.root_dir = Path(root_dir)
        self.sam3_root = Path(sam3_root)
        self.split = split
        self.feature_extractor = feature_extractor
        self.thin_class_ids = tuple(thin_class_ids)
        self.sam3_file_suffix = sam3_file_suffix

        self.image_paths, self.gt_root = list_cityscapes_split(self.root_dir, split)

        self.id_to_trainid = {
            7: 0,
            8: 1,
            11: 2,
            12: 3,
            13: 4,
            17: 5,
            19: 6,
            20: 7,
            21: 8,
            22: 9,
            23: 10,
            24: 11,
            25: 12,
            26: 13,
            27: 14,
            28: 15,
            31: 16,
            32: 17,
            33: 18,
        }

    def __len__(self) -> int:
        return len(self.image_paths)

    def encode_target(self, mask: Image.Image) -> np.ndarray:
        raw = np.array(mask, dtype=np.int32)
        mapped = np.full_like(raw, 255)
        for raw_id, train_id in self.id_to_trainid.items():
            mapped[raw == raw_id] = train_id
        return mapped

    def _sam3_path(self, city: str, stem: str) -> Path:
        return self.sam3_root / self.split / city / f"{stem}{self.sam3_file_suffix}"

    def _load_sam3_class_masks(self, sam3_path: Path, shape_hw: Tuple[int, int]) -> np.ndarray:
        """
        Returns a float array of shape [K, H, W] where K=len(thin_class_ids).

        Expected npz keys:
          - class IDs as strings (e.g., "5", "6", "12"), OR
          - class names like "class_5".
        """
        h, w = shape_hw
        out = np.zeros((len(self.thin_class_ids), h, w), dtype=np.float32)

        if not sam3_path.exists():
            return out

        with np.load(sam3_path) as data:
            for i, class_id in enumerate(self.thin_class_ids):
                arr = None
                if str(class_id) in data:
                    arr = data[str(class_id)]
                elif f"class_{class_id}" in data:
                    arr = data[f"class_{class_id}"]

                if arr is None:
                    continue

                arr = arr.astype(np.float32)
                if arr.ndim > 2:
                    arr = arr.squeeze()
                if arr.shape != (h, w):
                    arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)

                out[i] = np.clip(arr, 0.0, 1.0)

        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_path = self.image_paths[idx]
        city = image_path.parent.name
        stem = image_path.name.replace("_leftImg8bit.png", "")

        image = Image.open(image_path).convert("RGB")
        gt_path = self.gt_root / city / f"{stem}_gtFine_labelIds.png"
        label = Image.open(gt_path)
        mapped = self.encode_target(label)

        encoded = self.feature_extractor(
            images=image,
            segmentation_maps=Image.fromarray(mapped),
            return_tensors="pt",
        )
        pixel_values = encoded.pixel_values.squeeze(0)
        labels = encoded.labels.squeeze(0)

        sam3_path = self._sam3_path(city=city, stem=stem)
        sam3_masks = self._load_sam3_class_masks(sam3_path=sam3_path, shape_hw=labels.shape)

        return {
            "pixel_values": pixel_values,
            "labels": labels.long(),
            "sam3_masks": torch.from_numpy(sam3_masks).float(),
            "thin_class_ids": torch.tensor(self.thin_class_ids, dtype=torch.long),
        }


def build_dataset(
    dataset_name: str,
    data_root: str,
    sam3_root: str,
    split: str,
    processor: SegformerImageProcessor,
    thin_class_ids: Tuple[int, ...],
    sam3_file_suffix: str,
) -> Dataset:
    if dataset_name == "cityscapes":
        return CityscapesTopologyDistillDataset(
            root_dir=data_root,
            sam3_root=sam3_root,
            split=split,
            feature_extractor=processor,
            thin_class_ids=thin_class_ids,
            sam3_file_suffix=sam3_file_suffix,
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def train_one_epoch(
    model: SegformerForSemanticSegmentation,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_topology: float,
    thin_filter: ThinClassFilter,
    cldice_loss: SAM3GuidedCLDiceLoss,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_ce = 0.0
    total_topology = 0.0

    pbar = tqdm(loader, desc="Train")
    for batch in pbar:
        optimizer.zero_grad()

        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        sam3_masks = batch["sam3_masks"].to(device)
        thin_class_ids = batch["thin_class_ids"][0].tolist()

        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits
        logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)

        ce_loss = F.cross_entropy(logits, labels, ignore_index=255)

        sam3_masks_by_class = {
            class_id: sam3_masks[:, idx]
            for idx, class_id in enumerate(thin_class_ids)
        }
        topology_loss, topology_stats = compute_thin_class_topology_loss(
            student_logits=logits,
            sam3_masks_by_class=sam3_masks_by_class,
            thin_class_filter=thin_filter,
            cldice_loss=cldice_loss,
            candidate_class_ids=thin_class_ids,
        )

        # Combined objective: L = L_ce + lambda * L_clDice
        loss = ce_loss + lambda_topology * topology_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_topology += topology_loss.item()

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "ce": f"{ce_loss.item():.4f}",
                "cld": f"{topology_loss.item():.4f}",
                "k": int(topology_stats["selected_thin_classes"].item()),
            }
        )

    n = max(1, len(loader))
    return {
        "train_loss": total_loss / n,
        "train_ce_loss": total_ce / n,
        "train_cldice_loss": total_topology / n,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SegFormer with SAM3 topological distillation")

    parser.add_argument("--dataset", type=str, default="cityscapes", help="Dataset backend name")
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root path")
    parser.add_argument("--sam3_root", type=str, required=True, help="Root directory for SAM3 class pseudo-label files")
    parser.add_argument(
        "--sam3_file_suffix",
        type=str,
        default="_leftImg8bit_topology.npz",
        help="Per-image SAM3 pseudo-label suffix",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
        help="SegFormer model name",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=6e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lambda_topology", type=float, default=0.5)
    parser.add_argument("--skeleton_iterations", type=int, default=25)
    parser.add_argument(
        "--thin_class_ids",
        type=int,
        nargs="+",
        default=list(DEFAULT_THIN_CLASS_IDS),
        help="Train IDs used for topology distillation",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processor = SegformerImageProcessor.from_pretrained(args.model_name)
    processor.do_reduce_labels = False

    thin_class_ids = tuple(args.thin_class_ids)
    thin_filter = ThinClassFilter(thin_class_ids=thin_class_ids)
    cldice_loss = SAM3GuidedCLDiceLoss(skeleton_iterations=args.skeleton_iterations)

    train_ds = build_dataset(
        dataset_name=args.dataset,
        data_root=args.data_root,
        sam3_root=args.sam3_root,
        split="train",
        processor=processor,
        thin_class_ids=thin_class_ids,
        sam3_file_suffix=args.sam3_file_suffix,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model_name,
        num_labels=19,
        ignore_mismatched_sizes=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            lambda_topology=args.lambda_topology,
            thin_filter=thin_filter,
            cldice_loss=cldice_loss,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"loss={metrics['train_loss']:.4f} | "
            f"ce={metrics['train_ce_loss']:.4f} | "
            f"clDice={metrics['train_cldice_loss']:.4f}"
        )


if __name__ == "__main__":
    main()
