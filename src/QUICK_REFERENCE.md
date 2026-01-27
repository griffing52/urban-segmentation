# Quick Reference: Helper Modules

## Import Pattern
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("../src")))

from dataset_utils import *
from model_utils import *
from analysis_utils import *
```

## Dataset Operations

| Function | Purpose | Example |
|----------|---------|---------|
| `make_cityscapes_dataframe()` | Load val split metadata | `val_df = make_cityscapes_dataframe(root)` |
| `create_hard_subset()` | Filter to hard images | `hard_df = create_hard_subset(val_df)` |
| `load_benchmark_results()` | Load all model results | `results = load_benchmark_results(dir)` |
| `compare_subsets()` | All vs Hard comparison | `comp = compare_subsets(results, hard_ids)` |

## Model Interface

**Must implement:**
```python
class MyModel(BaseSegmentationModel):
    def predict(self, image: Image.Image) -> np.ndarray:
        # Return (H, W) array with trainIds (0-18)
        pass
```

**Common operations:**
```python
# Run inference
run_inference_over_df(val_df, model, pred_root)

# Evaluate
evaluate_model_on_split(pred_dir, gt_dir, val_df, "ModelName")

# Save results
save_results_csv(results_df, output_dir, "ModelName")
```

## Analysis

| Function | Returns | Use Case |
|----------|---------|----------|
| `compute_image_statistics()` | Stats per image | Find easy/hard images |
| `identify_easy_vs_hard()` | Top N easiest/hardest | Quick filtering |
| `analyze_per_class_difficulty()` | Per-class stats | Class-specific analysis |
| `compute_consensus()` | Multi-model agreement | MoE potential |

## Visualization

```python
from analysis_utils import (
    plot_all_vs_hard_comparison,
    plot_degradation,
    plot_distribution_ridgeline,
)

# Side-by-side mIoU
plot_all_vs_hard_comparison(comparison_table)

# Performance drop visualization
plot_degradation(comparison_table)

# Distribution by model
plot_distribution_ridgeline(results_df)
```

## Typical Workflow

### 1. Prepare Dataset
```python
val_df = make_cityscapes_dataframe(CITYSCAPES_ROOT, split="val")
hard_df = create_hard_subset(val_df, thin_threshold=20)
hard_ids = set(hard_df["image_id"])
```

### 2. Evaluate Model
```python
model = MyModelWrapper()
run_inference_over_df(val_df, model, pred_root)
evaluate_model_on_split(pred_dir, gt_dir, val_df, "MyModel")
save_results_csv(results_df, RESULTS_DIR, "MyModel")
```

### 3. Compare Performance
```python
all_results = load_benchmark_results(RESULTS_DIR)
comp = compare_subsets(all_results, hard_ids)

# comp['all_cityscapes'] - results on full set
# comp['hard_cityscapes'] - results on hard set
# comp['comparison'] - aggregated metrics + degradation
```

### 4. Analyze & Visualize
```python
stats = compute_image_statistics(all_results)
easy, hard = identify_easy_vs_hard(stats, n_images=10)

plot_all_vs_hard_comparison(comp['comparison'])
plot_degradation(comp['comparison'])
```

## Hard Subset Logic

**What is "hard"?**
- Images containing thin/small objects
- Detection threshold: <20 pixels OR aspect_ratio>5
- Examples: poles, thin persons, traffic signs

**Why it matters:**
- Tests robustness on complex urban scenes
- Real-world relevance (actual streets have thin objects)
- Shows model performance degradation

**Key metric: Degradation**
```
Degradation = mIoU_All - mIoU_Hard
```
- Positive = Performance drops on hard (expected)
- Negative = Better on hard (unusual, means better at detail)

## Column Reference

### After `make_cityscapes_dataframe()`
- `image_id`: Unique image ID
- `city`: City name (e.g., 'aachen')
- `image_path`: Path to RGB image
- `labelIds_path`: Path to ground truth
- `instanceIds_path`: Instance segmentation
- etc.

### After `load_benchmark_results()`
- `image_id`: Image identifier
- `city`: City name
- `model`: Model name
- `image_mIoU`: Mean IoU for image
- Class columns: Per-class IoU (0.0-1.0 or NaN if class absent)

### After `compare_subsets()`
- `comparison_table` columns:
  - `image_mIoU_all`: mIoU on all Cityscapes
  - `image_mIoU_hard`: mIoU on hard subset
  - `miou_degradation`: Difference (all - hard)
  - Per-class variants with `_all` and `_hard` suffixes

## Troubleshooting

**"Class not found" error**
- Check column names match class names in ground truth
- Use standard Cityscapes class names

**"Missing prediction" warnings**
- Ensure `run_inference_over_df()` completed successfully
- Check `pred_root` directory exists and contains `.npy` files

**Hard subset too small/large**
- Adjust `thin_threshold` parameter (default: 20)
- Lower threshold = more images included = less "hard"
- Higher threshold = fewer images = more "hard"

**Results mismatch**
- Ensure predictions and ground truth are same resolution
- Verify trainIds are 0-18 (255 for ignore)
- Check image IDs match between prediction and GT filenames

## Notebooks Reference

- **`All_vs_Hard_Cityscapes.ipynb`** - Complete hard subset evaluation with visualization
- **`Template_Using_Helpers.ipynb`** - Integration example for your models
- **`Cross_Model_Analysis.ipynb`** - Original multi-model comparison (pre-helpers)

---

**For detailed documentation**, see `HELPERS_README.md`
