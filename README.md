# Diagnosing and Improving Boundary Failures in Urban Semantic Segmentation via SAM3-Guided Supervision

## Abstract

Urban scene understanding remains a critical challenge in autonomous driving and robotics, with semantic segmentation models frequently struggling to accurately delineate thin, elongated structures such as traffic poles, signs, fences, and vegetation boundaries. These boundary failures pose significant safety risks, as misclassified or missed boundaries can lead to incorrect navigation decisions and collision hazards. This repository presents a comprehensive framework for diagnosing and mitigating boundary failures in urban semantic segmentation through targeted supervision from the Segment Anything Model 3 (SAM3).

Our approach leverages SAM3's zero-shot segmentation capabilities with carefully engineered text prompts to generate high-quality boundary masks for challenging thin object classes in the Cityscapes dataset. We introduce a hierarchical prompt strategy (L1-L4) ranging from baseline object descriptions to highly specific physical and contextual cues, combined with multi-scale processing techniques including baseline, multi-crop, and tiled generation strategies. These SAM3-generated masks serve as auxiliary supervision signals for training state-of-the-art segmentation models (SegFormer, Mask2Former, DDRNet) with explicit boundary-aware loss functions.

## Research Motivation

Modern semantic segmentation architectures achieve impressive performance on standard benchmarks, yet systematic failures persist for thin boundary objects that are critical for safe autonomous navigation. These failures stem from:

1. **Safety Critical**: Pedestrians, bicycles, traffic lights, and traffic signs are crucial for safe decision-making.  
2. **Class Imbalance**: Thin objects occupy minimal pixel area compared to large classes like road, building, and sky
3. **Spatial Resolution Loss**: Downsampling in encoder-decoder architectures degrades fine boundary details
4. **Ambiguous Annotations**: Ground truth boundaries are often inconsistent or imprecise at thin object edges
5. **Limited Training Signal**: Standard cross-entropy loss provides weak gradients for small, thin structures

Our work addresses these challenges by:
- **Diagnostic Analysis**: Quantifying failure modes across multiple architectures and object categories
- **Targeted Supervision**: Generating high-quality boundary masks specifically for problematic thin object classes
- **Prompt Engineering**: Systematically exploring how natural language descriptions influence SAM3's boundary detection
- **Multi-Scale Processing**: Adapting SAM3 to handle resolution and scale variations in urban scenes
- **Auxiliary Learning**: Integrating boundary-specific loss terms to guide model training

## Key Contributions

### 1. Hierarchical Prompt Strategy for Boundary Detection
We develop a four-level prompt hierarchy that progressively refines SAM3's focus:
- **L1 (Baseline)**: Generic object names ("pole", "traffic sign")
- **L2 (Descriptive)**: Enriched descriptions emphasizing thinness ("thin vertical pole", "narrow traffic sign")
- **L3 (Physical)**: Physical properties and materials ("metal pole", "thin wire fence")
- **L4 (Specific)**: Contextual and location-specific details ("roadside pole with attachments", "boundary fence along sidewalk")

This hierarchy enables systematic ablation studies to understand how prompt specificity affects boundary quality.

### 2. Multi-Scale Generation Strategies
To handle Cityscapes' high-resolution images (2048×1024) and multi-scale objects, we implement:
- **Baseline**: Direct SAM3 inference on full images
- **Multi-Crop**: Grid-based decomposition (e.g., 2×2, 3×3) with boundary stitching
- **Tiled**: Sliding window with overlap (window_size=1024, stride=256/512) and soft blending

These strategies are evaluated for boundary completeness, computational efficiency, and artifact reduction.

### 3. Comprehensive Diagnostic Framework
Our analysis pipeline provides:
- **Per-Class Boundary Metrics**: IoU, Dice, F1, Precision, Recall for each thin object category
- **Failure Mode Analysis**: Identification of systematic errors (false negatives in shadows, false positives in texture)
- **Cross-Model Comparison**: Benchmark SegFormer, Mask2Former, DDRNet with/without boundary supervision
- **Ablation Studies**: Isolating effects of prompt level, generation strategy, and loss weighting

### 4. Auxiliary Boundary-Aware Training
We augment standard segmentation training with:
- **Boundary Loss**: Explicit supervision on SAM3-generated thin object boundaries
- **Multi-Task Learning**: Joint optimization of semantic segmentation and boundary detection
- **Hard Negative Mining**: Focusing on difficult boundary pixels where models struggle
- **Loss Balancing**: Adaptive weighting between semantic and boundary objectives

## Methodology

### Data Preparation
- **Dataset**: Cityscapes (500 validation images, 2975 training images)
- **Target Classes**: Poles, traffic lights, traffic signs, vegetation boundaries, building edges, walls, fences, riders
- **Preprocessing**: Ground truth boundary extraction via Canny edge detection with morphological refinement

### SAM3 Boundary Generation
1. **Prompt Engineering**: Design prompts for each thin object class across L1-L4 levels
2. **Strategy Selection**: Choose baseline/multi-crop/tiled based on image characteristics
3. **Post-Processing**: Apply connected component filtering, hole filling, and boundary smoothing
4. **Quality Control**: Manual verification and automated metrics (boundary completeness, artifact detection)

### Model Training
- **Base Architectures**: SegFormer-B0/B5, Mask2Former, DDRNet23-Slim
- **Loss Function**: L = L_semantic + λ * L_boundary, where λ ∈ [0.1, 1.0]
- **Optimization**: AdamW with cosine learning rate schedule, gradient clipping
- **Augmentation**: RandomCrop, RandomHorizontalFlip, ColorJitter, maintaining boundary consistency

### Evaluation Protocol
- **Metrics**: mIoU (overall), per-class IoU, boundary IoU/F1, computational cost (FPS, memory)
- **Baselines**: Vanilla SegFormer, Mask2Former, traditional edge detection methods
- **Statistical Tests**: Paired t-tests, confidence intervals, significance at p < 0.05

## Expected Outcomes

1. **Improved Boundary Detection**: 5-10% IoU improvement on thin object classes
2. **Systematic Understanding**: Quantified effects of prompt specificity and generation strategy
3. **Generalizable Framework**: Methodology applicable to other datasets and object categories
4. **Open-Source Toolkit**: Complete pipeline for SAM3-based boundary supervision

## Repository Structure

```
.
├── configs/                    # Configuration files for generation and training
│   ├── sam3_generation.yaml   # SAM3 mask generation settings
│   ├── sam3_batch_strategies.yaml  # Multi-config batch generation
│   └── training.yaml          # Model training hyperparameters
├── data/                      # Cityscapes dataset (not included)
├── src/                       # Source code
│   ├── sam3_generation/       # SAM3 boundary generation pipeline
│   ├── models/                # Segmentation model architectures
│   ├── training/              # Training loops and loss functions
│   └── utils/                 # Data loading, metrics, visualization
├── scripts/                   # Executable scripts
│   ├── generate_sam3_masks.py # Generate boundary masks
│   └── train_segformer.py     # Train segmentation models
├── notebooks/                 # Analysis and visualization
│   ├── Batch_Test_Analysis.ipynb  # Performance analysis across configs
│   └── SAM3_Grid_Search_Comparison.ipynb  # Strategy comparison
├── generated_masks/           # SAM3-generated boundary masks
└── experiments/               # Training outputs and checkpoints
```

## Getting Started

### Prerequisites
```bash
# Core dependencies
pip install torch torchvision transformers
pip install opencv-python pillow numpy pandas matplotlib seaborn
pip install segmentation-models-pytorch  # For SegFormer, DDRNet

# SAM3 setup (adapt to your environment)
pip install segment-anything-3  # or follow SAM3 official installation
```

### Generate SAM3 Boundary Masks
```bash
# Single configuration
python scripts/generate_sam3_masks.py \
    --config configs/sam3_generation.yaml \
    --data_root data/cityscapes \
    --output_dir generated_masks/sam3_baseline_L2_Descriptive

# Batch generation (multiple configs)
python scripts/generate_sam3_masks.py \
    --batch_config configs/sam3_batch_strategies.yaml \
    --max_images 100  # For testing; use -1 for full dataset
```

### Train Segmentation Model with Boundary Supervision
```bash
python scripts/train_segformer.py \
    --config configs/training.yaml \
    --model segformer_b0 \
    --boundary_masks generated_masks/sam3_baseline_L2_Descriptive \
    --boundary_weight 0.5
```

### Analyze Results
Open `notebooks/Batch_Test_Analysis.ipynb` to:
- Load and evaluate generated masks against ground truth
- Compare performance across prompt levels (L1-L4) and strategies (baseline, multi-crop, tiled)
- Visualize per-class metrics and failure cases
- Generate publication-ready figures and tables

## Preliminary Results

### SAM3 Boundary Generation Quality
| Strategy | Prompt | IoU | Dice | F1 | Precision | Recall |
|----------|--------|-----|------|----|-----------| -------|
| Baseline | L1     | 0.58 | 0.68 | 0.65 | 0.72 | 0.61 |
| Baseline | L2     | 0.62 | 0.72 | 0.69 | 0.75 | 0.64 |
| Multi-Crop 2×2 | L1 | 0.60 | 0.70 | 0.67 | 0.73 | 0.63 |
| Multi-Crop 2×2 | L2 | 0.65 | 0.74 | 0.71 | 0.78 | 0.66 |
| Tiled (1024/512) | L2 | 0.63 | 0.73 | 0.70 | 0.76 | 0.65 |

### Per-Class Boundary Detection (L2 Descriptive)
| Class | IoU | Best Strategy |
|-------|-----|---------------|
| Pole | 0.61 | Multi-Crop 2×2 |
| Traffic Sign | 0.67 | Multi-Crop 2×2 |
| Traffic Light | 0.59 | Baseline |
| Fence | 0.63 | Tiled 1024/512 |
| Vegetation | 0.58 | Tiled 1024/512 |

*Note: Results based on preliminary analysis of Cityscapes validation subset*

## Roadmap to CVPR Workshop Quality

### Immediate Priorities
- [ ] Full validation set evaluation (500 images)
- [ ] Strong baseline comparisons (vanilla SegFormer, Mask2Former)
- [ ] Statistical significance testing (confidence intervals, p-values)
- [ ] Complete prompt hierarchy evaluation (L1→L2→L3→L4)

### Experimental Extensions
- [ ] Ablation studies: prompt components, generation parameters
- [ ] Failure mode analysis: shadow sensitivity, texture confusion
- [ ] Computational efficiency: FPS, memory profiling
- [ ] Cross-dataset generalization: KITTI, Mapillary Vistas

### Paper Development
- [ ] Write comprehensive methodology section
- [ ] Generate high-quality figures (architecture diagram, qualitative results)
- [ ] Prepare ablation tables and statistical analyses
- [ ] Draft discussion of when/why SAM3 guidance helps

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{urban-segmentation-sam3,
  title={Diagnosing and Improving Boundary Failures in Urban Semantic Segmentation via SAM3-Guided Supervision},
  author={[Your Name]},
  year={2026},
  howpublished={\url{https://github.com/[your-username]/urban-segmentation}}
}
```

## Acknowledgments

This research builds upon:
- **Cityscapes Dataset**: Cordts et al., "The Cityscapes Dataset for Semantic Urban Scene Understanding"
- **Segment Anything Model**: Kirillov et al., "Segment Anything"
- **SegFormer**: Xie et al., "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers"
- **Mask2Former**: Cheng et al., "Masked-attention Mask Transformer for Universal Image Segmentation"

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Contact

For questions or collaboration inquiries, please open an issue or contact [your-email].