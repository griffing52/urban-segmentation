# Quick Start Guide - New Repository Structure

## 🎯 What's Changed?

Your repository is now organized professionally with clear separation of concerns.

## 📁 Key Directories

| Directory | Purpose |
|-----------|---------|
| **scripts/** | Executable Python scripts (training, generation, visualization) |
| **configs/** | Configuration files (.yaml) for different tasks |
| **src/** | Reusable utility modules and helpers |
| **notebooks/** | Jupyter notebooks for analysis and exploration |
| **docs/** | All documentation files |
| **data/** | Read-only input datasets |
| **outputs/** | Generated outputs (git-ignored) |
| **logs/** | Runtime logs |

## 🚀 Quick Commands

### Generate SAM3 Masks
```bash
python scripts/generate_sam3_masks.py
# Or with custom config:
python scripts/generate_sam3_masks.py --config configs/sam3_generation.yaml
```

### Train Model
```bash
python scripts/train_segformer_boundary.py
# Or with custom config:
python scripts/train_segformer_boundary.py --config configs/training.yaml
```

### Visualize Training
```bash
python scripts/visualize_training.py --log_file outputs/experiments/logs/training_log.json
```

### Run Analysis
```bash
jupyter notebook notebooks/SAM3_Grid_Search_Comparison.ipynb
```

## 📚 Documentation

All documentation is organized in `docs/`:
- **GETTING_STARTED.md** - Getting started guide
- **SAM3_GENERATION_README.md** - SAM3 documentation
- **TRAINING_README.md** - Training guide
- **ANALYSIS_NOTEBOOK_QUICKSTART.md** - Analysis notebooks guide

## 🔧 Configuration Files

Located in `configs/`:
- **sam3_generation.yaml** - SAM3 generation settings
- **training.yaml** - Training hyperparameters
- **analysis.yaml** - Analysis configuration

Edit these files to customize behavior.

## 📝 Project Files

- **REPOSITORY_STRUCTURE.md** - Detailed directory structure and file descriptions
- **REORGANIZATION_SUMMARY.md** - What changed and why
- **README.md** - Main project documentation (update this with your project info)

## ⚡ Common Workflows

### 1. Generate Masks and Train Model
```bash
python scripts/generate_sam3_masks.py --config configs/sam3_generation.yaml
python scripts/train_segformer_boundary.py --config configs/training.yaml
```

### 2. Analyze Grid Search Results
```bash
jupyter notebook notebooks/SAM3_Grid_Search_Comparison.ipynb
```

### 3. Visualize Training Progress
```bash
python scripts/visualize_training.py --log_file outputs/experiments/logs/training_log.json
```

## 🗂️ Output Organization

All outputs are organized by type in `outputs/`:
```
outputs/
├── generated_masks/    # SAM3-generated masks
├── experiments/        # Training experiments
│   ├── checkpoints/   # Model checkpoints
│   └── logs/          # Training logs
├── analysis/          # Analysis results
└── grid_search_results/  # Grid search outputs
```

## 🔑 Key Points

✅ **Scripts**: All executable code in `scripts/`  
✅ **Config**: All settings in YAML files under `configs/`  
✅ **Docs**: All documentation in `docs/`  
✅ **Outputs**: All generated files in `outputs/`  
✅ **Code**: Reusable utilities in `src/`  

## 💡 Tips

- Check script docstrings for detailed usage: `python scripts/generate_sam3_masks.py --help`
- Modify config files to experiment with different settings
- Outputs are automatically organized and ready for analysis
- Use notebooks for interactive exploration

## 📖 Learn More

1. Read **REPOSITORY_STRUCTURE.md** for detailed descriptions
2. Check **docs/GETTING_STARTED.md** for comprehensive setup
3. Review individual script docstrings for implementation details

## 🆘 Need Help?

- **Can't find something?** → Check REPOSITORY_STRUCTURE.md
- **How do I run X?** → Check the script's `--help` or docs/
- **Where are outputs?** → Check `outputs/` directory

---

**Happy researching!** 🚀
