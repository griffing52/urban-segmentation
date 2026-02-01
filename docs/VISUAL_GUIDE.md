# Visual Guide to the Helper System

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Your Model Notebooks                          │
│  (SegFormer, Mask2Former, DDRNet, etc.)                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
         ┌────────────────────────────────────────┐
         │   Import from src/:                     │
         │  • dataset_utils                        │
         │  • model_utils                          │
         │  • analysis_utils                       │
         └────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    HELPER MODULES (src/)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  dataset_utils.py              model_utils.py      analysis_utils.py
│  ═══════════════               ═════════════       ═════════════════
│                                                                   │
│  • Load Cityscapes              • BaseModel        • Statistics   │
│  • Create hard subset            • Inference        • Easy/Hard   │
│  • Filter by image props        • Evaluate         • Visualization
│  • Load results CSVs            • Save results     • Consensus    │
│  • Compare subsets                                                │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              Datasets & Evaluation Results                        │
│  (Cityscapes, Benchmark CSVs, Predictions)                      │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

### Setup Phase
```
Cityscapes Dataset
    └─ leftImg8bit/val/*.png ──┐
    └─ gtFine/val/*_labelIds.png ──┤
                                   ├──→ make_cityscapes_dataframe()
                                   └──→ DataFrame with metadata
```

### Hard Subset Creation
```
Ground Truth Labelids
    └─ *_labelIds.png ──┐
                        ├──→ identify_thin_objects()
                        └──→ (Connected Components)
                            └──→ create_hard_subset()
                                └──→ hard_df (filtered)
```

### Model Evaluation
```
Model (inherit BaseSegmentationModel)
    └─ predict(image) → np.ndarray ──┐
                                     ├──→ run_inference_over_df()
val_df (images)                      └──→ {image}_trainIds.npy
    ├─────────────────────────────────→ evaluate_model_on_split()
gtFine/val/*labelIds.png             └──→ IoU scores
                                    ──→ save_results_csv()
                                        └──→ {model}_per_image_iou.csv
```

### Analysis Phase
```
{model}_per_image_iou.csv (multiple models)
    └──→ load_benchmark_results()
        └──→ combined results DataFrame
            ├──→ compute_image_statistics()
            │   └──→ per-image stats (mean, std, difficulty)
            │
            └──→ compare_subsets()
                ├──→ all_cityscapes results
                ├──→ hard_cityscapes results
                └──→ comparison table (degradation)
                    ├──→ plot_all_vs_hard_comparison()
                    └──→ plot_degradation()
```

## Workflow Diagrams

### Developer Workflow (Adding Your Model)

```
START
  │
  ├─→ Inherit BaseSegmentationModel
  │   └─ Implement predict()
  │
  ├─→ Load dataset
  │   └─ val_df = make_cityscapes_dataframe()
  │
  ├─→ Run inference
  │   └─ run_inference_over_df(val_df, model, pred_root)
  │
  ├─→ Evaluate
  │   ├─ evaluate_model_on_split()
  │   └─ save_results_csv()
  │
  └─→ END
```

### Analysis Workflow (All vs Hard)

```
START
  │
  ├─→ Load dataset
  │   └─ val_df = make_cityscapes_dataframe()
  │
  ├─→ Create hard subset
  │   ├─ hard_df = create_hard_subset(val_df)
  │   └─ hard_ids = set(hard_df["image_id"])
  │
  ├─→ Load all results
  │   └─ results = load_benchmark_results(RESULTS_DIR)
  │
  ├─→ Compare performance
  │   └─ comparison = compare_subsets(results, hard_ids)
  │
  ├─→ Visualize
  │   ├─ plot_all_vs_hard_comparison()
  │   └─ plot_degradation()
  │
  └─→ END (Make pitch!)
```

## Function Call Hierarchy

### dataset_utils.py

```
make_cityscapes_dataframe()
    └─ Loads image & GT paths from disk

create_hard_subset()
    ├─ identify_thin_objects()
    │   └─ scipy.ndimage.label (connected components)
    └─ Returns filtered DataFrame

load_benchmark_results()
    └─ glob all *_per_image_iou.csv files
       └─ concat into single DataFrame

compare_subsets()
    └─ evaluate_on_subset() (2x)
       └─ Pivot & aggregate
          └─ Calculate degradation
```

### model_utils.py

```
BaseSegmentationModel (abstract)
    └─ predict() [must implement]

run_inference_over_df()
    ├─ Loop over images
    ├─ model.predict(image)
    └─ Save as .npy

evaluate_model_on_split()
    ├─ Loop over predictions
    ├─ compute_per_image_iou()
    │   └─ Per-class IoU calculation
    └─ Return results DataFrame

save_results_csv()
    └─ Write to {model}_per_image_iou.csv
```

### analysis_utils.py

```
compute_image_statistics()
    ├─ Pivot results table
    ├─ Calculate mean, std, max, min
    └─ Derive difficulty & MoE gain

identify_easy_vs_hard()
    ├─ Sort by mean_performance
    └─ Return head/tail

plot_all_vs_hard_comparison()
    └─ Bar chart (All vs Hard mIoU)

plot_degradation()
    └─ Horizontal bar chart (degradation %)
```

## Class Hierarchy

```python
# Abstract Base Class
BaseSegmentationModel
    ├─ predict(image) → ndarray [ABSTRACT]
    │
    └─ Your Implementation
        ├─ SegFormerWrapper
        ├─ Mask2FormerWrapper
        ├─ DDRNetWrapper
        └─ MyCustomModel
            └─ predict() [IMPLEMENTATION]
```

## Key Data Structures

### DataFrame Schemas

#### After `make_cityscapes_dataframe()`
```
image_id          city       image_path                labelIds_path
─────────────────────────────────────────────────────────────────
aachen_000000_000000  aachen  .../aachen_000000_000000_leftImg8bit.png
aachen_000001_000019  aachen  .../aachen_000001_000019_leftImg8bit.png
...
```

#### After `load_benchmark_results()`
```
image_id       model      road   sidewalk   building  ... image_mIoU
────────────────────────────────────────────────────────────────
aachen_000000  SegFormer  0.925  0.743      0.912    ... 0.851
aachen_000000  Mask2Former 0.918 0.752     0.901    ... 0.847
...
```

#### From `compute_image_statistics()`
```
          mean_performance  std_performance  max_performance  difficulty  moe_gain
────────────────────────────────────────────────────────────────────────────────
aachen_000000     0.849            0.012        0.851          0.151      0.002
aachen_000001     0.765            0.084        0.823          0.235      0.058
...
```

#### From `compare_subsets()['comparison']`
```
          image_mIoU_all  image_mIoU_hard  miou_degradation
────────────────────────────────────────────────────────────
SegFormer      0.751         0.698            0.053
Mask2Former    0.743         0.684            0.059
DDRNet         0.735         0.672            0.063
...
```

## Hard Subset Logic (Visual)

```
Ground Truth Mask (labelIds)
    │
    ├─→ Connected Components
    │   └─ scipy.ndimage.label()
    │
    ├─→ For each component:
    │   ├─ Pixel count < 20? → THIN
    │   ├─ Aspect ratio > 5? → THIN
    │   └─ Else: KEEP
    │
    └─→ Has thin object?
        ├─ YES → Include in hard_df
        └─ NO  → Keep in easy set
```

## Pitch Visualization

```
Model Performance Degradation
═══════════════════════════════════════════════════════════════

Industry Average:
    ████████████████████ 15-20% degradation (Expected)

Our Method:
    ███████ 7.1% degradation ✓ BETTER!

Interpretation:
    While standard models degrade on complex scenes,
    our method maintains performance on the 'Hard' subset.
```

## Comparison Table Example

```
Model Robustness Report: All vs Hard Cityscapes
═══════════════════════════════════════════════════════════════════════

                 All CityscapesmIoU  Hard CityscapesmIoU  Degradation
                 ─────────────────  ──────────────────  ──────────────
SegFormer B1               0.751            0.698            0.053 (7%)
Mask2Former                0.743            0.684            0.059 (8%)
DDRNet-23                  0.735            0.672            0.063 (9%)
Our Method              0.762            0.718            0.044 (6%) ✓
```

## Integration Points

### For Each Model Notebook

```
YOUR NOTEBOOK
    │
    ├─→ Load dataset
    │   └─ from dataset_utils import make_cityscapes_dataframe
    │
    ├─→ Implement model
    │   └─ from model_utils import BaseSegmentationModel
    │       class YourModel(BaseSegmentationModel):
    │           def predict(...): pass
    │
    ├─→ Run inference & evaluate
    │   └─ from model_utils import run_inference_over_df, evaluate_model_on_split
    │
    └─→ Compare performance (optional)
        └─ from analysis_utils import *
            from dataset_utils import compare_subsets
```

---

This architecture enables:
- ✓ **Code Reuse**: No duplication across notebooks
- ✓ **Consistency**: Same evaluation methodology everywhere
- ✓ **Extensibility**: Easy to add new models or analyses
- ✓ **Modularity**: Each module does one thing well
