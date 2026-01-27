"""
Analysis and visualization utilities for segmentation model comparison.

Provides functions for:
- Easy vs Hard image analysis
- Multi-model consensus evaluation
- Per-class difficulty analysis
- Result visualization (ridgeline plots, scatter plots, etc.)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# Core Analysis Functions
# ============================================================================

def compute_image_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-image statistics across all models.

    Args:
        df: DataFrame from load_benchmark_results() with 'image_mIoU' or class columns

    Returns:
        DataFrame indexed by image_id with columns:
        - mean_performance: Average mIoU across models
        - std_performance: Std dev across models
        - max_performance: Best model performance (oracle)
        - min_performance: Worst model performance
        - difficulty: 1.0 - mean_performance (higher = harder)
        - moe_gain: max - mean (potential for mixture of experts)
    """
    class_cols = [
        c for c in df.columns 
        if c not in ['image_id', 'city', 'model', 'image_mIoU']
    ]

    if 'image_mIoU' not in df.columns:
        df = df.copy()
        df['image_mIoU'] = df[class_cols].mean(axis=1)

    # Pivot: rows=images, cols=models, values=mIoU
    pivot = df.pivot(index='image_id', columns='model', values='image_mIoU')

    stats_df = pd.DataFrame(index=pivot.index)
    stats_df['mean_performance'] = pivot.mean(axis=1)
    stats_df['std_performance'] = pivot.std(axis=1)
    stats_df['max_performance'] = pivot.max(axis=1)
    stats_df['min_performance'] = pivot.min(axis=1)
    stats_df['difficulty'] = 1.0 - stats_df['mean_performance']
    stats_df['moe_gain'] = stats_df['max_performance'] - stats_df['mean_performance']

    return stats_df


def identify_easy_vs_hard(
    stats_df: pd.DataFrame,
    n_images: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify easiest and hardest images by mean model performance.

    Args:
        stats_df: Output from compute_image_statistics()
        n_images: How many top/bottom images to return

    Returns:
        Tuple of (easy_df, hard_df)
    """
    easy_df = stats_df.sort_values('mean_performance', ascending=False).head(n_images)
    hard_df = stats_df.sort_values('mean_performance', ascending=True).head(n_images)

    return easy_df, hard_df


def analyze_per_class_difficulty(
    df: pd.DataFrame,
    class_name: str,
) -> Dict[str, any]:
    """
    Analyze difficulty of a specific class across all images.

    Args:
        df: DataFrame from load_benchmark_results()
        class_name: Column name for the class (e.g., 'person', 'road')

    Returns:
        Dict with analysis results:
        - 'pivot': Pivot table (images x models)
        - 'mean_iou': Per-image mean IoU
        - 'hardest_images': Top 5 hardest images for this class
        - 'high_potential_images': Top 5 with model disagreement
        - 'overall_mean': Average IoU across all images/models
    """
    if class_name not in df.columns:
        raise ValueError(f"Class '{class_name}' not found in data columns")

    # Filter to images where class is present
    class_df = df[['image_id', 'model', class_name]].dropna()

    if class_df.empty:
        raise ValueError(f"No valid data for class '{class_name}'")

    # Pivot: rows=images, cols=models
    pivot = class_df.pivot(index='image_id', columns='model', values=class_name)

    # Calculate per-image stats
    pivot['mean_iou'] = pivot.mean(axis=1)
    pivot['max_iou'] = pivot.max(axis=1)
    pivot['gain'] = pivot['max_iou'] - pivot['mean_iou']

    return {
        'pivot': pivot,
        'mean_iou': pivot['mean_iou'],
        'hardest_images': pivot.sort_values('mean_iou').head(5),
        'high_potential_images': pivot.sort_values('gain', ascending=False).head(5),
        'overall_mean': pivot['mean_iou'].mean(),
    }


def model_comparison_scatter(
    df: pd.DataFrame,
    model1: str,
    model2: str,
    class_name: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Create scatter plot comparing two models.

    Args:
        df: Benchmark results DataFrame
        model1: First model name
        model2: Second model name
        class_name: Specific class to compare. If None, uses 'image_mIoU'
        ax: Matplotlib axis to draw on. If None, creates new figure.

    Returns:
        Matplotlib axis with plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    metric = class_name if class_name else 'image_mIoU'
    
    if metric not in df.columns:
        class_cols = [c for c in df.columns if c not in ['image_id', 'city', 'model']]
        if metric == 'image_mIoU' and metric not in df.columns:
            df = df.copy()
            df['image_mIoU'] = df[class_cols].mean(axis=1)

    pivot = df.pivot(index='image_id', columns='model', values=metric)

    if model1 not in pivot.columns or model2 not in pivot.columns:
        raise ValueError(f"Model {model1} or {model2} not found in data")

    x = pivot[model1]
    y = pivot[model2]

    ax.scatter(x, y, alpha=0.6, s=50)
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='x=y')

    ax.set_xlabel(f'{model1} IoU', fontsize=11)
    ax.set_ylabel(f'{model2} IoU', fontsize=11)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    title = f'{model1} vs {model2}'
    if class_name:
        title += f' ({class_name})'
    ax.set_title(title, fontsize=12)

    return ax


def compute_consensus(
    df: pd.DataFrame,
    good_threshold: float = 0.75,
    bad_threshold: float = 0.4,
    class_name: Optional[str] = None,
) -> Dict[str, int]:
    """
    Compute multi-model consensus statistics.

    Args:
        df: Benchmark results DataFrame
        good_threshold: IoU threshold for "success"
        bad_threshold: IoU threshold for "failure"
        class_name: Specific class to analyze. If None, uses 'image_mIoU'

    Returns:
        Dict with consensus statistics:
        - 'all_models_succeed': Images where ALL models score > good_threshold
        - 'any_model_succeeds': Images where AT LEAST ONE model scores > good_threshold
        - 'all_models_fail': Images where ALL models score < bad_threshold
        - 'any_model_fails': Images where AT LEAST ONE model scores < bad_threshold
    """
    metric = class_name if class_name else 'image_mIoU'
    
    if metric not in df.columns:
        class_cols = [c for c in df.columns if c not in ['image_id', 'city', 'model']]
        if metric == 'image_mIoU':
            df = df.copy()
            df['image_mIoU'] = df[class_cols].mean(axis=1)

    pivot = df.pivot(index='image_id', columns='model', values=metric)

    results = {
        'all_models_succeed': (pivot > good_threshold).all(axis=1).sum(),
        'any_model_succeeds': (pivot > good_threshold).any(axis=1).sum(),
        'all_models_fail': (pivot < bad_threshold).all(axis=1).sum(),
        'any_model_fails': (pivot < bad_threshold).any(axis=1).sum(),
    }

    return results


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_distribution_ridgeline(
    df: pd.DataFrame,
    class_name: str = 'image_mIoU',
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Create ridgeline plot of IoU distributions per model.

    Args:
        df: Benchmark results DataFrame
        class_name: Class to visualize. Default is overall 'image_mIoU'
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure
    """
    if class_name not in df.columns:
        class_cols = [c for c in df.columns if c not in ['image_id', 'city', 'model']]
        if class_name == 'image_mIoU':
            df = df.copy()
            df['image_mIoU'] = df[class_cols].mean(axis=1)

    plot_df = df[['model', class_name]].dropna().copy()
    plot_df.columns = ['Model', 'IoU']

    models = sorted(plot_df['Model'].unique())

    # Create ridgeline
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    g = sns.FacetGrid(
        plot_df,
        row="Model",
        hue="Model",
        aspect=10,
        height=0.8,
        palette="viridis",
        row_order=models,
        hue_order=models,
    )

    # Draw histograms
    g.map(
        sns.histplot,
        "IoU",
        bins=30,
        element="poly",
        fill=True,
        alpha=0.7,
        stat="density",
    )

    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Label each ridge
    def label_ridge(x, color, label):
        ax = plt.gca()
        ax.text(
            0, 0.2, label,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
            transform=ax.transAxes,
        )

    g.map(label_ridge, "IoU")

    g.figure.subplots_adjust(hspace=-0.5)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    plt.xlim(0, 1)
    plt.xlabel("IoU Score", fontsize=11)
    plt.suptitle(f"IoU Distribution by Model: {class_name}", y=0.98, fontsize=12)

    # Reset theme
    sns.set_theme(style="whitegrid")

    return g.figure


def plot_all_vs_hard_comparison(
    comparison_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 6),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot side-by-side comparison of mIoU on All vs Hard Cityscapes.

    Args:
        comparison_df: Output from compare_subsets()['comparison']
        figsize: Figure size

    Returns:
        Tuple of (figure, axes)
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(comparison_df))
    width = 0.35

    all_scores = comparison_df['image_mIoU_all']
    hard_scores = comparison_df['image_mIoU_hard']

    ax.bar(x - width/2, all_scores, width, label='All Cityscapes', alpha=0.8)
    ax.bar(x + width/2, hard_scores, width, label='Hard Cityscapes', alpha=0.8)

    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('mIoU', fontsize=11)
    ax.set_title('Model Performance: All vs Hard Subsets', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df.index, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    return fig, ax


def plot_degradation(
    comparison_df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 6),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot mIoU degradation from All to Hard subset per model.

    Args:
        comparison_df: Output from compare_subsets()['comparison']
        figsize: Figure size

    Returns:
        Tuple of (figure, axes)
    """
    fig, ax = plt.subplots(figsize=figsize)

    degradation = comparison_df['miou_degradation'].sort_values(ascending=False)

    colors = ['red' if x > 0.05 else 'green' for x in degradation.values]
    ax.barh(range(len(degradation)), degradation.values, color=colors, alpha=0.7)

    ax.set_yticks(range(len(degradation)))
    ax.set_yticklabels(degradation.index)
    ax.set_xlabel('mIoU Degradation (All → Hard)', fontsize=11)
    ax.set_title('Performance Drop on Hard Subset', fontsize=12)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, axis='x', alpha=0.3)

    fig.tight_layout()
    return fig, ax
