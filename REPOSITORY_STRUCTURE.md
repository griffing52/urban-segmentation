# Repository Structure Guide

## Overview
This project has been reorganized into a modular, professional structure with clear separation of concerns.

## Directory Structure

```
urban-segmentation/
├── README.md                          # Main project documentation
├── LICENSE                            # License file
├── .gitignore                         # Git ignore rules
│
├── src/                               # Core utilities and modules
│   ├── __init__.py
│   ├── analysis_utils.py              # Analysis helpers
│   ├── dataset_utils.py               # Dataset loading and processing
│   ├── model_utils.py                 # Model utilities
│   └── QUICK_REFERENCE.md             # API documentation
│
├── scripts/                           # Executable Python scripts
│   ├── generate_sam3_masks.py         # Main SAM3 mask generation
│   ├── generate_sam3_masks_from_configs.py  # Batch generation from configs
│   ├── train_segformer_boundary.py    # Training script
│   └── visualize_training.py          # Training visualization
│
├── configs/                           # Configuration files
│   ├── sam3_generation.yaml           # SAM3 generation settings
│   ├── training.yaml                  # Training hyperparameters
│   └── analysis.yaml                  # Analysis settings
│
├── notebooks/                         # Jupyter notebooks for analysis
│   ├── SAM3_Grid_Search_Comparison.ipynb
│   ├── SAM3_SegFormer_Combined.ipynb
│   ├── Cross_Model_Analysis.ipynb
│   └── ... (other analysis notebooks)
│
├── data/                              # Data directory (read-only)
│   ├── cityscapes/
│   │   ├── leftImg8bit_trainvaltest/  # Input images
│   │   ├── gtFine_trainvaltest/       # Ground truth labels
│   │   ├── sam3_boundary/             # SAM3 boundary masks
│   │   └── benchmark_results/         # Evaluation results
│   └── ... (other datasets)
│
├── outputs/                           # Generated outputs (git-ignored)
│   ├── generated_masks/               # SAM3-generated masks
│   │   ├── sam3_baseline_L1_Baseline/
│   │   ├── sam3_multi_crop_L2_Descriptive_grid_size=([2, 2], [1, 2])/
│   │   ├── sam3_tiled_L3_Physical_window_size=1024_stride=256/
│   │   └── generation_metadata.json
│   ├── experiments/                   # Training experiments
│   │   ├── checkpoints/               # Model checkpoints
│   │   ├── logs/                      # Training logs
│   │   └── my_first_run/
│   ├── analysis/                      # Analysis results
│   │   ├── configuration_summary.csv
│   │   ├── detailed_results.csv
│   │   ├── RECOMMENDATIONS.txt
│   │   ├── method_performance.png
│   │   └── method_prompt_heatmap.png
│   └── grid_search_results/           # Grid search outputs
│       ├── analysis/
│       └── masks/
│
├── logs/                              # Application logs
│   └── (runtime logs)
│
├── docs/                              # Documentation
│   ├── GETTING_STARTED.md             # Getting started guide
│   ├── SAM3_GENERATION_README.md      # SAM3 documentation
│   ├── TRAINING_README.md             # Training documentation
│   ├── PROMPT_STRATEGIES_README.md    # Prompt strategies guide
│   ├── ANALYSIS_NOTEBOOK_QUICKSTART.md # Analysis guide
│   ├── VISUAL_GUIDE.md                # Visual explanations
│   └── ... (other documentation)
│
└── experiments/                       # Legacy experiments (legacy)
    └── my_first_run/
```

## Key Changes

### Before
- Root directory cluttered with scripts, configs, and documentation
- No clear separation between different types of files
- Documentation mixed with code
- Outputs scattered in various locations

### After
- **src/**: Core reusable modules and utilities
- **scripts/**: Standalone executable scripts
- **configs/**: YAML configuration files for different tasks
- **notebooks/**: Analysis and exploration notebooks
- **data/**: Read-only input data
- **outputs/**: All generated outputs (organized by type)
- **logs/**: Runtime logs
- **docs/**: All documentation

## Usage Guide

### Running Scripts

```bash
# Generate SAM3 masks (baseline strategy)
python scripts/generate_sam3_masks.py --config configs/sam3_generation.yaml

# Train SegFormer with boundary loss
python scripts/train_segformer_boundary.py --config configs/training.yaml

# Visualize training metrics
python scripts/visualize_training.py --log_file outputs/experiments/logs/training_log.json

# Generate masks from batch configs
python scripts/generate_sam3_masks_from_configs.py --data_root data/cityscapes
```

### Analysis Workflows

1. **Quick Analysis**: Open notebooks in `notebooks/` and run cells sequentially
2. **Grid Search**: Use `SAM3_Grid_Search_Comparison.ipynb` to compare configurations
3. **Model Evaluation**: Use `Cross_Model_Analysis.ipynb` for model comparisons

### Configuration Files

Edit YAML files in `configs/` to customize:
- SAM3 generation strategies and settings
- Training hyperparameters
- Analysis parameters

## Output Organization

All generated outputs are organized by type:
- **generated_masks/**: SAM3 output masks organized by strategy
- **experiments/**: Training experiments with checkpoints and logs
- **analysis/**: Analysis results and visualizations
- **grid_search_results/**: Grid search masks and analysis

## Next Steps

1. Update script imports to use absolute paths from root
2. Consider adding a `pyproject.toml` or `setup.py` for package installation
3. Add pre-commit hooks for code quality
4. Create a Makefile for common commands (optional)

## Notes

- The `experiments/` directory now primarily contains past experiments (can be archived)
- The `outputs/` directory should be in `.gitignore`
- Configuration files provide defaults; command-line arguments can override them
- Logs directory can be used for runtime application logs
