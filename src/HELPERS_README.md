## Urban Segmentation Analysis Helpers

A modular set of utilities to simplify benchmarking and analysis of semantic segmentation models on Cityscapes.

### Overview

The project is organized into three helper modules:

1. **`dataset_utils.py`** - Dataset loading, filtering, and hard subset creation
2. **`model_utils.py`** - Abstract model interface and evaluation pipeline
3. **`analysis_utils.py`** - Performance analysis and visualization

### Quick Start

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("src")))

from dataset_utils import make_cityscapes_dataframe, create_hard_subset
from model_utils import BaseSegmentationModel, run_inference_over_df

# Load dataset
val_df = make_cityscapes_dataframe(Path("data/cityscapes"), split="val")

# Create hard subset (images with thin objects)
hard_df = create_hard_subset(val_df, thin_threshold=20)

# Run inference with your model (must inherit BaseSegmentationModel)
val_df_pred = run_inference_over_df(val_df, model=your_model, pred_root=pred_dir)
```

### Module Details

#### `dataset_utils.py`

Core dataset management and hard subset creation.

**Main Functions:**

- `list_cityscapes_split(root, split)` - List all images in a split
- `make_cityscapes_dataframe(root, split)` - Create structured DataFrame with metadata
- `identify_thin_objects(gt_path, thin_threshold, target_classes)` - Detect thin/small objects
- `create_hard_subset(df, thin_threshold, target_classes)` - Filter to hard images
- `load_benchmark_results(results_dir)` - Load all model results from CSVs
- `evaluate_on_subset(results_df, subset_image_ids)` - Filter results to subset
- `compare_subsets(all_results, hard_subset_image_ids)` - Compare All vs Hard performance

**The "Hard Subset":**

The hard subset identifies images containing small, elongated objects that are challenging to segment:
- Thin poles
- Thin persons/pedestrians
- Traffic signs
- Narrow boundaries

Uses connected component analysis with configurable thresholds:
- `thin_threshold=20`: Components with <20 pixels are "thin"
- `aspect_ratio > 5`: High aspect ratio regions are considered elongated

#### `model_utils.py`

Model interface and evaluation framework.

**Main Classes & Functions:**

- `BaseSegmentationModel` (abstract) - Base class for all model wrappers
  - Requires `predict(image: Image.Image) -> np.ndarray` implementation
  - Returns integer array with trainIds (0-18 for Cityscapes)

- `run_inference_over_df(df, model, pred_root, overwrite)` - Generic inference loop
  - Saves predictions as `.npy` files
  - Efficient: skips existing predictions unless `overwrite=True`

- `compute_per_image_iou(pred_path, gt_path, num_classes, class_names)` - Single-image IoU
  - Handles both `.npy` and `.png` predictions
  - Returns per-class IoU dict

- `evaluate_model_on_split(pred_dir, gt_dir, split_df, model_name)` - Full evaluation
  - Computes per-image, per-class IoU for entire split
  - Returns DataFrame ready for analysis

- `save_results_csv(results_df, output_dir, model_name)` - Save results
  - Standard filename format: `{model_name}_per_image_iou.csv`

#### `analysis_utils.py`

Performance analysis and visualization.

**Main Functions:**

- `compute_image_statistics(df)` - Per-image stats across models
  - `mean_performance`: Average mIoU across models
  - `difficulty`: 1 - mean (higher = harder)
  - `moe_gain`: Max - mean (MoE potential)

- `identify_easy_vs_hard(stats_df, n_images)` - Top N easiest/hardest images

- `analyze_per_class_difficulty(df, class_name)` - Per-class analysis
  - Identifies hardest images for specific class
  - Shows model disagreement potential

- `model_comparison_scatter(df, model1, model2, class_name)` - Scatter plot comparison

- `compute_consensus(df, good_threshold, bad_threshold)` - Multi-model consensus stats
  - Images where all models succeed
  - Images where at least one model fails

- `plot_distribution_ridgeline(df, class_name)` - Ridgeline plot of IoU distributions

- `plot_all_vs_hard_comparison(comparison_df)` - Side-by-side mIoU comparison

- `plot_degradation(comparison_df)` - Degradation visualization

### Usage Examples

#### Example 1: Create Hard Subset & Evaluate

```python
from dataset_utils import make_cityscapes_dataframe, create_hard_subset, compare_subsets
from dataset_utils import load_benchmark_results

# Load validation set
val_df = make_cityscapes_dataframe(CITYSCAPES_ROOT, split="val")

# Create hard subset
hard_df = create_hard_subset(val_df, thin_threshold=20)
hard_image_ids = set(hard_df["image_id"])

# Load all benchmark results
all_results = load_benchmark_results(RESULTS_DIR)

# Compare performance
comparison = compare_subsets(all_results, hard_image_ids)

# Access results
all_performance = comparison['all_cityscapes']  # Full validation set
hard_performance = comparison['hard_cityscapes']  # Hard subset only
comparison_table = comparison['comparison']      # Aggregated comparison
```

#### Example 2: Analyze Image Difficulty

```python
from analysis_utils import compute_image_statistics, identify_easy_vs_hard

# Load benchmark results
results_df = load_benchmark_results(RESULTS_DIR)

# Compute per-image statistics
stats = compute_image_statistics(results_df)

# Find easiest and hardest images
easy, hard = identify_easy_vs_hard(stats, n_images=10)

print(f"Hardest image: {hard.index[0]}")
print(f"Mean performance: {hard['mean_performance'].iloc[0]:.4f}")
print(f"MoE potential: {hard['moe_gain'].iloc[0]:.4f}")
```

#### Example 3: Implement Custom Model

```python
from model_utils import BaseSegmentationModel, run_inference_over_df
import torch
from PIL import Image
import numpy as np

class MyModelWrapper(BaseSegmentationModel):
    def __init__(self, checkpoint_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = load_my_model(checkpoint_path).to(self.device)
        self.model.eval()
    
    def predict(self, image: Image.Image) -> np.ndarray:
        # Convert to tensor
        inputs = preprocess_image(image)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(inputs)
        
        # Convert to trainIds
        pred_mask = postprocess_output(outputs)
        return pred_mask.astype(np.uint8)

# Use in pipeline
model = MyModelWrapper("path/to/checkpoint")
val_df_pred = run_inference_over_df(
    val_df, 
    model=model, 
    pred_root=CITYSCAPES_ROOT / "MyModel_preds" / "val"
)
```

### File Organization

```
urban-segmentation/
├── src/
│   ├── __init__.py
│   ├── dataset_utils.py      # Dataset & hard subset logic
│   ├── model_utils.py        # Model interface & evaluation
│   └── analysis_utils.py     # Analysis & visualization
├── notebooks/
│   ├── All_vs_Hard_Cityscapes.ipynb           # Hard subset evaluation
│   ├── Template_Using_Helpers.ipynb           # Integration template
│   ├── Cross_Model_Analysis.ipynb             # Original analysis notebook
│   ├── SegFormer-BM-Project8.ipynb            # Model-specific notebooks
│   └── ... (other model notebooks)
└── README.md
```

### Key Concepts

#### Hard Subset Creation

The "hard subset" is created by identifying images with thin objects (small, elongated regions):

1. Load ground truth segmentation masks
2. Use connected component analysis
3. Find components with:
   - `<20 pixels` (thin threshold), OR
   - `aspect_ratio > 5` (high aspect ratio)
4. Filter images that contain at least one thin object

This creates a challenging test set for evaluating model robustness on complex urban scenes.

#### Model Performance Degradation

The "pitch" metric shows how much models degrade when evaluated on the hard subset:

- **Degradation = mIoU(All Cityscapes) - mIoU(Hard Cityscapes)**
- Positive values: Performance drops on hard subset
- Negative values: Actually performs better on hard subset (rare)

**Key Finding:** While standard models degrade on complex scenes, our method maintains performance on the 'Hard' subset.

### Running the Examples

1. **Setup paths** in notebook cells to match your environment
2. **Load data**: `val_df = make_cityscapes_dataframe(CITYSCAPES_ROOT)`
3. **Create hard subset**: `hard_df = create_hard_subset(val_df)`
4. **Run analysis**: See `All_vs_Hard_Cityscapes.ipynb`

### Dependencies

```python
pandas
numpy
Pillow
torch
torchvision
transformers
matplotlib
seaborn
scipy  # For connected component analysis
cityscapesscripts  # For Cityscapes label definitions
tqdm  # For progress bars
```

### Notes

- All predictions must be saved as `.npy` files with trainIds (integer 0-18, 255 for ignore)
- Ground truth must be in Cityscapes standard format (labelIds.png)
- Results are aggregated by image_id, so multiple model predictions per image are averaged
- The hard subset threshold (20 pixels) can be tuned for different difficulty levels

### Future Enhancements

- [ ] Per-class hard subset variants (thin persons only, thin poles only, etc.)
- [ ] Confidence-based subset filtering
- [ ] Interactive visualization dashboard
- [ ] Automatic report generation
