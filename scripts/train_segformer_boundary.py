"""
SegFormer + SAM3 Boundary Training Script

Trains SegFormer with auxiliary boundary loss using SAM3-generated boundary maps.
Supports checkpointing, resumption, and detailed training metrics logging.
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from tqdm import tqdm
import evaluate

from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


# ============================================================================
# Dataset Definition
# ============================================================================

class CityscapesSAM3Dataset(Dataset):
    def __init__(self, root_dir, split="train", feature_extractor=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.feature_extractor = feature_extractor

        # Paths
        self.img_dir = self.root_dir / "leftImg8bit_trainvaltest" / "leftImg8bit" / split
        self.lbl_dir = self.root_dir / "gtFine_trainvaltest" / "gtFine" / split
        self.bnd_dir = self.root_dir / "sam3_boundary" / split

        self.items = []

        # Walk directories
        if not self.img_dir.exists():
            raise FileNotFoundError(f"❌ Path not found: {self.img_dir}")

        for city_folder in self.img_dir.glob("*"):
            if not city_folder.is_dir():
                continue
            for img_file in city_folder.glob("*_leftImg8bit.png"):
                base_name = img_file.name.replace("_leftImg8bit.png", "")
                self.items.append({
                    "img": img_file,
                    "lbl": self.lbl_dir / city_folder.name / f"{base_name}_gtFine_labelIds.png",
                    "bnd": self.bnd_dir / city_folder.name / f"{base_name}_leftImg8bit.npy"
                })

        # Define Mapping: Raw Cityscapes ID -> Train ID (0-18)
        self.id_to_trainid = {
            7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
            19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
            26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18
        }

    def __len__(self):
        return len(self.items)

    def encode_target(self, mask):
        """Map raw labelIds to trainIds"""
        mask = np.array(mask)
        label_mask = np.full_like(mask, 255)  # Fill with ignore index
        for k, v in self.id_to_trainid.items():
            label_mask[mask == k] = v
        return label_mask

    def __getitem__(self, idx):
        item = self.items[idx]
        image = Image.open(item["img"]).convert("RGB")
        label = Image.open(item["lbl"])

        # Load Boundary
        if item["bnd"].exists():
            boundary = np.load(item["bnd"])
        else:
            boundary = np.zeros(label.size[::-1], dtype=np.uint8)

        # Map labels to trainIds
        label_mapped = self.encode_target(label)

        encoded = self.feature_extractor(
            images=image,
            segmentation_maps=Image.fromarray(label_mapped),
            return_tensors="pt"
        )

        pixel_values = encoded.pixel_values.squeeze(0)
        labels = encoded.labels.squeeze(0)

        # Resize boundary to match labels
        h, w = labels.shape
        boundary_resized = cv2.resize(
            boundary.astype(np.uint8), (w, h), 
            interpolation=cv2.INTER_NEAREST
        )

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "sam3_boundary": torch.from_numpy(boundary_resized).long()
        }


# ============================================================================
# Model Definition
# ============================================================================

class SegFormerWithBoundary(nn.Module):
    def __init__(self, num_classes=19, pretrained="nvidia/segformer-b1-finetuned-cityscapes-1024-1024"):
        super().__init__()
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            pretrained, num_labels=num_classes,
            ignore_mismatched_sizes=True, output_hidden_states=True
        )
        self.boundary_head = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, pixel_values, labels=None, boundary_targets=None):
        outputs = self.segformer(pixel_values, labels=labels, output_hidden_states=True)

        decoder_feats = outputs.hidden_states[-1]
        bnd_logits = self.boundary_head(decoder_feats)
        bnd_logits = F.interpolate(
            bnd_logits, size=pixel_values.shape[-2:], 
            mode='bilinear', align_corners=False
        ).squeeze(1)

        loss = None
        if labels is not None and boundary_targets is not None:
            seg_loss = outputs.loss
            bnd_loss = F.binary_cross_entropy_with_logits(
                bnd_logits, boundary_targets.float(), 
                pos_weight=torch.tensor(5.0).to(pixel_values.device)
            )
            loss = seg_loss + 2.0 * bnd_loss
            return {
                "loss": loss, 
                "seg_loss": seg_loss, 
                "bnd_loss": bnd_loss, 
                "logits": outputs.logits
            }

        return {"logits": outputs.logits}


# ============================================================================
# Training Logger
# ============================================================================

class TrainingLogger:
    """Logs training metrics to JSON for later analysis"""
    
    def __init__(self, log_dir: Path, experiment_name: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.json"
        
        self.history = {
            "experiment_name": experiment_name,
            "start_time": timestamp,
            "epochs": [],
            "config": {}
        }
    
    def log_config(self, config: Dict):
        """Log training configuration"""
        self.history["config"] = config
        self._save()
    
    def log_epoch(self, epoch: int, metrics: Dict):
        """Log metrics for an epoch"""
        entry = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        self.history["epochs"].append(entry)
        self._save()
    
    def _save(self):
        """Save history to JSON"""
        with open(self.log_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def get_log_file(self) -> Path:
        return self.log_file


# ============================================================================
# Checkpoint Manager
# ============================================================================

class CheckpointManager:
    """Manages model checkpoints during training"""
    
    def __init__(self, checkpoint_dir: Path, keep_last_n: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
    
    def save_checkpoint(
        self, 
        model: nn.Module, 
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict,
        is_best: bool = False
    ):
        """Save a training checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics
        }
        
        # Save regular checkpoint
        ckpt_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, ckpt_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"✅ Saved best model to {best_path}")
        
        # Clean old checkpoints
        self._cleanup_old_checkpoints()
        
        return ckpt_path
    
    def _cleanup_old_checkpoints(self):
        """Keep only the last N checkpoints"""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pth"),
            key=lambda p: p.stat().st_mtime
        )
        
        # Remove old checkpoints
        if len(checkpoints) > self.keep_last_n:
            for ckpt in checkpoints[:-self.keep_last_n]:
                ckpt.unlink()
    
    def load_checkpoint(self, checkpoint_path: Path, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
        """Load a checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        return checkpoint.get("epoch", 0), checkpoint.get("metrics", {})
    
    def find_latest_checkpoint(self) -> Optional[Path]:
        """Find the most recent checkpoint"""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pth"),
            key=lambda p: p.stat().st_mtime
        )
        return checkpoints[-1] if checkpoints else None


# ============================================================================
# Training Function
# ============================================================================

def train_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_seg_loss = 0
    total_bnd_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        optimizer.zero_grad()
        
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        boundary_targets = batch["sam3_boundary"].to(device)
        
        outputs = model(pixel_values, labels=labels, boundary_targets=boundary_targets)
        
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_seg_loss += outputs["seg_loss"].item()
        total_bnd_loss += outputs["bnd_loss"].item()
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "seg": f"{outputs['seg_loss'].item():.4f}",
            "bnd": f"{outputs['bnd_loss'].item():.4f}"
        })
    
    n_batches = len(train_loader)
    return {
        "train_loss": total_loss / n_batches,
        "train_seg_loss": total_seg_loss / n_batches,
        "train_bnd_loss": total_bnd_loss / n_batches
    }


def validate(model, val_loader, device, metric, num_batches=None):
    """Validate the model"""
    model.eval()
    
    if num_batches is None:
        num_batches = len(val_loader)
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Validation", total=num_batches)):
            if i >= num_batches:
                break
            
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(pixel_values)
            logits = outputs["logits"]
            
            upsampled = F.interpolate(
                logits, size=labels.shape[-2:], 
                mode="bilinear", align_corners=False
            )
            preds = upsampled.argmax(dim=1)
            
            metric.add_batch(
                predictions=preds.cpu().numpy(),
                references=labels.cpu().numpy()
            )
    
    results = metric.compute(num_labels=19, ignore_index=255)
    return {
        "val_mean_iou": results["mean_iou"],
        "val_mean_accuracy": results["mean_accuracy"],
        "val_overall_accuracy": results["overall_accuracy"]
    }


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train SegFormer with Boundary Loss")
    
    # Paths
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to Cityscapes dataset root")
    parser.add_argument("--output_dir", type=str, default="./experiments",
                        help="Directory for checkpoints and logs")
    parser.add_argument("--experiment_name", type=str, default="segformer_sam3_boundary",
                        help="Name for this experiment")
    
    # Training
    parser.add_argument("--epochs", type=int, default=80,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=6e-5,
                        help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    
    # Model
    parser.add_argument("--model_name", type=str, 
                        default="nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
                        help="Pretrained SegFormer model name")
    
    # Checkpointing
    parser.add_argument("--save_every", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--keep_last_n", type=int, default=3,
                        help="Keep only last N checkpoints")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    # Validation
    parser.add_argument("--val_every", type=int, default=1,
                        help="Validate every N epochs")
    parser.add_argument("--val_batches", type=int, default=None,
                        help="Number of validation batches (None = full validation)")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory structure
    output_dir = Path(args.output_dir) / args.experiment_name
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    
    # Initialize logger
    logger = TrainingLogger(log_dir, args.experiment_name)
    logger.log_config(vars(args))
    print(f"Logging to: {logger.get_log_file()}")
    
    # Initialize checkpoint manager
    ckpt_manager = CheckpointManager(checkpoint_dir, keep_last_n=args.keep_last_n)
    
    # Load data
    print("Loading datasets...")
    processor = SegformerImageProcessor.from_pretrained(args.model_name)
    processor.do_reduce_labels = False
    
    train_ds = CityscapesSAM3Dataset(args.data_root, "train", processor)
    val_ds = CityscapesSAM3Dataset(args.data_root, "val", processor)
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, 
        shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
    
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    
    # Initialize model
    print("Initializing model...")
    model = SegFormerWithBoundary(num_classes=19, pretrained=args.model_name)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    metric = evaluate.load("mean_iou")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_miou = 0.0
    
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        start_epoch, prev_metrics = ckpt_manager.load_checkpoint(
            Path(args.resume_from), model, optimizer
        )
        start_epoch += 1  # Start from next epoch
        best_miou = prev_metrics.get("val_mean_iou", 0.0)
        print(f"Resuming from epoch {start_epoch}, best mIoU: {best_miou:.4f}")
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    for epoch in range(start_epoch, args.epochs):
        epoch_num = epoch + 1
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch_num)
        
        print(f"\nEpoch {epoch_num}/{args.epochs}")
        print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        print(f"  Seg Loss: {train_metrics['train_seg_loss']:.4f}")
        print(f"  Bnd Loss: {train_metrics['train_bnd_loss']:.4f}")
        
        # Validate
        val_metrics = {}
        if epoch_num % args.val_every == 0:
            val_metrics = validate(model, val_loader, device, metric, args.val_batches)
            print(f"  Val mIoU: {val_metrics['val_mean_iou']:.4f}")
            print(f"  Val Accuracy: {val_metrics['val_mean_accuracy']:.4f}")
        
        # Combine metrics
        all_metrics = {**train_metrics, **val_metrics}
        logger.log_epoch(epoch_num, all_metrics)
        
        # Save checkpoint
        is_best = False
        if val_metrics and val_metrics["val_mean_iou"] > best_miou:
            best_miou = val_metrics["val_mean_iou"]
            is_best = True
        
        if epoch_num % args.save_every == 0 or is_best:
            ckpt_path = ckpt_manager.save_checkpoint(
                model, optimizer, epoch_num, all_metrics, is_best
            )
            print(f"  Checkpoint saved: {ckpt_path}")
    
    print("\n✅ Training complete!")
    print(f"Best mIoU: {best_miou:.4f}")
    print(f"Logs saved to: {logger.get_log_file()}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
