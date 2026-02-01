# SAM3 Grid Search Analysis - Quick Start

## What to Do

This notebook analyzes SAM3 boundary detection results from grid search experiments.

### Step 1: Generate Masks (if not already done)

Choose one of these based on your needs:

#### Quick Test (30 minutes, validation only)
```bash
python quick_launch_sam3.py fast --data_root ./data/cityscapes
```

#### Balanced (2-3 hours, validation only)  
```bash
python quick_launch_sam3.py balanced --data_root ./data/cityscapes
```

#### Maximum Quality (6-8 hours, validation only)
```bash
python quick_launch_sam3.py quality --data_root ./data/cityscapes
```

#### Production (full training set, 12+ hours)
```bash
python quick_launch_sam3.py production --data_root ./data/cityscapes
```

### Step 2: Run This Notebook

Once masks are generated:

1. Open `notebooks/SAM3_Grid_Search_Comparison.ipynb`
2. Run all cells (Ctrl+Shift+Enter in VS Code)
3. Results will be saved to `grid_search_results/analysis/`

### What You'll Get

- **configuration_summary.csv** - Ranked configurations with metrics
- **detailed_results.csv** - Per-image evaluation results
- **RECOMMENDATIONS.txt** - Summary with top performers
- **method_performance.png** - Boxplot comparison
- **method_prompt_heatmap.png** - IoU by method & prompt
- **top_configurations.png** - Rankings visualization

## Expected Output

The notebook will:
1. ✓ Discover all generated mask configurations
2. ✓ Extract ground truth boundaries from Cityscapes
3. ✓ Compute 5 metrics per image-config pair (IoU, Dice, F1, Precision, Recall)
4. ✓ Create summary rankings
5. ✓ Generate 3 visualization charts
6. ✓ Save all results to CSV and text files

## Performance

- **Time**: 2-10 minutes depending on number of configurations
- **Memory**: ~2-4 GB
- **Disk**: ~100 MB for results

All operations are optimized for speed and memory efficiency.

## Key Metrics Explained

- **IoU** (Intersection over Union): Main metric, 0-1 scale
- **Dice**: Similar to F1, measures overlap  
- **Precision**: Fewer false positives = higher
- **Recall**: Fewer false negatives = higher
- **F1**: Balanced score between precision & recall

## Tips

- Start with `fast` scenario to test the workflow
- Use `balanced` for most production cases
- Use `quality` or `production-hq` if you need maximum accuracy
- Run notebook cells once masks are generated
- Check RECOMMENDATIONS.txt for best configurations

## Troubleshooting

**No results found**: Make sure masks are generated first (Step 1)
**Slow execution**: This is normal for large configurations. Grab a coffee!
**Memory errors**: Reduce number of configs or use a machine with more RAM

---

**Ready?** Run a quick test:
```bash
python quick_launch_sam3.py fast --data_root ./data/cityscapes
```

Then open the notebook!
