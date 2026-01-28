# SegFormer + SAM3 Boundary Training

Standalone training script for SegFormer with auxiliary boundary loss using SAM3-generated boundary maps.

## Features

- ✅ **Checkpoint Management**: Automatic checkpoint saving with configurable frequency
- ✅ **Resume Training**: Continue from previous checkpoints
- ✅ **Progress Logging**: All metrics saved to JSON for later analysis
- ✅ **Validation**: Regular validation with mIoU tracking
- ✅ **Best Model Saving**: Automatically saves best performing model
- ✅ **Visualization Tools**: Plot training curves and compare experiments

## Quick Start

### 1. Basic Training

```bash
python train_segformer_boundary.py \
    --data_root ../data/cityscapes \
    --output_dir ./experiments \
    --experiment_name my_first_run \
    --epochs 80 \
    --batch_size 4
```

### 2. Resume from Checkpoint

```bash
python train_segformer_boundary.py \
    --data_root ../data/cityscapes \
    --output_dir ./experiments \
    --experiment_name my_first_run \
    --resume_from experiments/my_first_run/checkpoints/checkpoint_epoch_040.pth
```

### 3. Visualize Training Progress

```bash
# View single experiment
python visualize_training.py \
    --log_file experiments/my_first_run/logs/my_first_run_20260128_120000.json

# Compare multiple experiments
python visualize_training.py \
    --compare experiments/run1/logs/*.json experiments/run2/logs/*.json
```

## Command-Line Arguments

### Required Arguments

- `--data_root`: Path to Cityscapes dataset root directory

### Training Configuration

- `--epochs`: Number of training epochs (default: 80)
- `--batch_size`: Batch size for training (default: 4)
- `--learning_rate`: Learning rate (default: 6e-5)
- `--num_workers`: Number of data loading workers (default: 4)

### Model Configuration

- `--model_name`: Pretrained SegFormer model (default: nvidia/segformer-b1-finetuned-cityscapes-1024-1024)

### Output & Logging

- `--output_dir`: Directory for checkpoints and logs (default: ./experiments)
- `--experiment_name`: Name for this experiment (default: segformer_sam3_boundary)

### Checkpointing

- `--save_every`: Save checkpoint every N epochs (default: 5)
- `--keep_last_n`: Keep only last N checkpoints (default: 3)
- `--resume_from`: Path to checkpoint to resume from

### Validation

- `--val_every`: Validate every N epochs (default: 1)
- `--val_batches`: Number of validation batches (None = full validation)

## Output Structure

```
experiments/
└── my_experiment/
    ├── checkpoints/
    │   ├── checkpoint_epoch_005.pth
    │   ├── checkpoint_epoch_010.pth
    │   ├── checkpoint_epoch_015.pth
    │   └── best_model.pth
    └── logs/
        ├── my_experiment_20260128_120000.json
        └── training_curves.png
```

## Training Log Format

Logs are saved as JSON files with the following structure:

```json
{
  "experiment_name": "my_experiment",
  "start_time": "20260128_120000",
  "config": {
    "epochs": 80,
    "batch_size": 4,
    "learning_rate": 6e-5,
    ...
  },
  "epochs": [
    {
      "epoch": 1,
      "timestamp": "2026-01-28T12:15:30",
      "train_loss": 0.5234,
      "train_seg_loss": 0.4123,
      "train_bnd_loss": 0.0555,
      "val_mean_iou": 0.7234,
      "val_mean_accuracy": 0.8456
    },
    ...
  ]
}
```

## Example: Overnight Training

For long training runs, use `nohup` or `screen`:

```bash
# Using nohup
nohup python train_segformer_boundary.py \
    --data_root ../data/cityscapes \
    --experiment_name overnight_run \
    --epochs 80 \
    --batch_size 4 \
    --save_every 5 \
    > training.log 2>&1 &

# Check progress
tail -f training.log

# Or use screen
screen -S training
python train_segformer_boundary.py --data_root ../data/cityscapes --epochs 80
# Ctrl+A, D to detach
# screen -r training to reattach
```

## Hyperparameter Tuning

Example configurations for different scenarios:

### Fast Prototyping (Small GPU)
```bash
python train_segformer_boundary.py \
    --batch_size 2 \
    --val_batches 50 \
    --save_every 10
```

### Full Training (Large GPU - A100)
```bash
python train_segformer_boundary.py \
    --batch_size 8 \
    --epochs 100 \
    --save_every 5 \
    --val_every 2
```

### Different Model Backbone
```bash
python train_segformer_boundary.py \
    --model_name nvidia/segformer-b3-finetuned-cityscapes-1024-1024 \
    --batch_size 2 \
    --learning_rate 5e-5
```

## Monitoring Training

### Real-time Monitoring

```python
# In a separate Python session or Jupyter notebook
import json
from pathlib import Path

log_file = "experiments/my_run/logs/my_run_20260128_120000.json"

with open(log_file) as f:
    data = json.load(f)
    
# Get latest metrics
latest = data["epochs"][-1]
print(f"Epoch {latest['epoch']}: Loss={latest['train_loss']:.4f}, mIoU={latest.get('val_mean_iou', 'N/A')}")
```

### Generate Plots Automatically

Add this to your cron or run periodically:

```bash
#!/bin/bash
# auto_plot.sh
LOG_FILE="experiments/my_run/logs/*.json"
python visualize_training.py --log_file $LOG_FILE --save_dir ./plots
```

## Tips

1. **GPU Memory**: If you run out of memory, reduce `batch_size` or use gradient accumulation
2. **Checkpoint Frequency**: Save more frequently early on, less frequently later
3. **Validation Cost**: Use `--val_batches` to speed up validation during development
4. **Best Model**: Always saved to `best_model.pth` based on validation mIoU
5. **Resume Training**: Can resume from any checkpoint, optimizer state is preserved

## Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Reduce `num_workers`
- Use a smaller model variant (b0 or b1 instead of b3/b4)

### Slow Training
- Increase `num_workers` (4-8 typically optimal)
- Use `--val_batches` for faster validation
- Check disk I/O (SAM3 boundary maps should be pre-generated)

### Checkpoint Issues
- Ensure sufficient disk space
- Check write permissions in `output_dir`
- Old checkpoints auto-deleted based on `--keep_last_n`

## Next Steps

To make this more generalizable for different models/configurations:

1. Create a YAML config file system
2. Add support for different loss functions
3. Add learning rate scheduling
4. Add data augmentation options
5. Support for mixed precision training (AMP)

Let me know if you'd like help implementing any of these!
