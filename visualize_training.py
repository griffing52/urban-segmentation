"""
Visualize training metrics from JSON logs

Usage:
    python visualize_training.py --log_file experiments/segformer_sam3_boundary/logs/training_log.json
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_training_log(log_file):
    """Load training log from JSON file"""
    with open(log_file, 'r') as f:
        return json.load(f)


def plot_training_metrics(history, save_dir=None):
    """Create comprehensive training visualizations"""
    epochs_data = history["epochs"]
    
    if not epochs_data:
        print("No training data found in log file")
        return
    
    epochs = [e["epoch"] for e in epochs_data]
    
    # Extract metrics
    train_loss = [e.get("train_loss") for e in epochs_data]
    train_seg_loss = [e.get("train_seg_loss") for e in epochs_data]
    train_bnd_loss = [e.get("train_bnd_loss") for e in epochs_data]
    val_miou = [e.get("val_mean_iou") for e in epochs_data if e.get("val_mean_iou") is not None]
    val_acc = [e.get("val_mean_accuracy") for e in epochs_data if e.get("val_mean_accuracy") is not None]
    
    # Filter out None values for validation metrics
    val_epochs = [e["epoch"] for e in epochs_data if e.get("val_mean_iou") is not None]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Training Progress: {history['experiment_name']}", fontsize=16)
    
    # Plot 1: Overall Training Loss
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, 'b-', linewidth=2, label='Total Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Loss Components
    ax = axes[0, 1]
    if train_seg_loss and any(x is not None for x in train_seg_loss):
        ax.plot(epochs, train_seg_loss, 'g-', linewidth=2, label='Segmentation Loss')
    if train_bnd_loss and any(x is not None for x in train_bnd_loss):
        ax.plot(epochs, train_bnd_loss, 'r-', linewidth=2, label='Boundary Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Components')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Validation mIoU
    ax = axes[1, 0]
    if val_miou:
        ax.plot(val_epochs, val_miou, 'purple', marker='o', linewidth=2, markersize=6)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mIoU')
        ax.set_title('Validation Mean IoU')
        ax.grid(True, alpha=0.3)
        
        # Add best mIoU annotation
        best_miou = max(val_miou)
        best_epoch = val_epochs[val_miou.index(best_miou)]
        ax.axhline(y=best_miou, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_miou:.4f}')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No validation data', ha='center', va='center', transform=ax.transAxes)
    
    # Plot 4: Validation Accuracy
    ax = axes[1, 1]
    if val_acc:
        ax.plot(val_epochs, val_acc, 'orange', marker='s', linewidth=2, markersize=6)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Validation Mean Accuracy')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No validation data', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    # Save figure if directory provided
    if save_dir:
        save_path = Path(save_dir) / "training_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    plt.show()


def print_summary(history):
    """Print training summary statistics"""
    print("\n" + "="*60)
    print(f"Experiment: {history['experiment_name']}")
    print(f"Started: {history['start_time']}")
    print("="*60)
    
    if "config" in history and history["config"]:
        print("\nConfiguration:")
        for key, value in history["config"].items():
            print(f"  {key}: {value}")
    
    epochs_data = history["epochs"]
    if not epochs_data:
        print("\nNo training data found")
        return
    
    print(f"\nTotal Epochs Trained: {len(epochs_data)}")
    
    # Training loss stats
    train_losses = [e["train_loss"] for e in epochs_data if "train_loss" in e]
    if train_losses:
        print(f"\nTraining Loss:")
        print(f"  Initial: {train_losses[0]:.4f}")
        print(f"  Final: {train_losses[-1]:.4f}")
        print(f"  Best: {min(train_losses):.4f}")
    
    # Validation mIoU stats
    val_mious = [e["val_mean_iou"] for e in epochs_data if "val_mean_iou" in e]
    if val_mious:
        best_idx = val_mious.index(max(val_mious))
        best_epoch = [e["epoch"] for e in epochs_data if "val_mean_iou" in e][best_idx]
        
        print(f"\nValidation mIoU:")
        print(f"  Best: {max(val_mious):.4f} (epoch {best_epoch})")
        print(f"  Final: {val_mious[-1]:.4f}")
    
    # Validation accuracy stats
    val_accs = [e["val_mean_accuracy"] for e in epochs_data if "val_mean_accuracy" in e]
    if val_accs:
        print(f"\nValidation Accuracy:")
        print(f"  Best: {max(val_accs):.4f}")
        print(f"  Final: {val_accs[-1]:.4f}")
    
    print("="*60 + "\n")


def compare_experiments(log_files):
    """Compare multiple training runs"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Experiment Comparison", fontsize=16)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(log_files)))
    
    for log_file, color in zip(log_files, colors):
        history = load_training_log(log_file)
        epochs_data = history["epochs"]
        
        if not epochs_data:
            continue
        
        label = history["experiment_name"]
        
        # Plot training loss
        epochs = [e["epoch"] for e in epochs_data]
        train_loss = [e.get("train_loss") for e in epochs_data]
        axes[0].plot(epochs, train_loss, color=color, linewidth=2, label=label)
        
        # Plot validation mIoU
        val_epochs = [e["epoch"] for e in epochs_data if e.get("val_mean_iou") is not None]
        val_miou = [e["val_mean_iou"] for e in epochs_data if e.get("val_mean_iou") is not None]
        if val_miou:
            axes[1].plot(val_epochs, val_miou, color=color, marker='o', linewidth=2, label=label)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('mIoU')
    axes[1].set_title('Validation mIoU')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize training progress")
    parser.add_argument("--log_file", type=str, required=True,
                        help="Path to training log JSON file")
    parser.add_argument("--compare", type=str, nargs='+',
                        help="Compare multiple log files")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save plots")
    parser.add_argument("--no_plot", action="store_true",
                        help="Don't show plots, just print summary")
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare multiple experiments
        compare_experiments(args.compare)
    else:
        # Single experiment
        history = load_training_log(args.log_file)
        
        # Print summary
        print_summary(history)
        
        # Plot metrics
        if not args.no_plot:
            save_dir = args.save_dir or Path(args.log_file).parent
            plot_training_metrics(history, save_dir)


if __name__ == "__main__":
    main()
