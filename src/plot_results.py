"""
Plot Results Module

This module generates analysis plots from the marker identification results.
It calculates and plots the histogram of specificity score distribution for best marker(s) of all cell types:
- Highest on-target/off-target ratio among top 10 markers
- Highest ratio of two markers combined among top 10 markers (taking the product of on-target/off-target ratios)
"""

import ast
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Use non-interactive backend for headless environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .io_utils import load_tissue_cells, load_gene_list, load_expression_matrices


def across_cell_analysis_top10(
    tissue_cells: List,
    genes: List,
    nTPM_matrix: np.ndarray,
    markers_file: str,
    output_dir: str,
    method_name: str = "high"
) -> List[float]:
    """
    Analyze best target/off-target ratio among top 10 markers.
    Returns the ratios for each cell type.
    
    Args:
        tissue_cells: List of [tissue, cell] pairs
        genes: List of gene names
        nTPM_matrix: Expression matrix
        markers_file: Path to recommended markers CSV
        output_dir: Directory to save results
        method_name: Name for output files
        
    Returns:
        List of ratios for each cell type
    """
    m = len(tissue_cells)
    cell_strs = [f"{t} {c}" for t, c in tissue_cells]
    
    rec_df = pd.read_csv(markers_file)
    rec_df['markers'] = rec_df['markers'].apply(lambda x: ast.literal_eval(x))
    
    # Build mapping
    top_markers = {}
    for i in range(min(m, len(rec_df))):
        cell = rec_df.iloc[i, 0]
        if cell not in cell_strs:
            continue
        cell_id = cell_strs.index(cell)
        markers = []
        for j in range(min(10, len(rec_df.iloc[i, 1]))):
            marker = rec_df.iloc[i, 1][j]
            if [marker] in genes:
                marker_id = genes.index([marker])
                markers.append(marker_id)
        if markers:
            top_markers[cell_id] = markers
    
    # Calculate best ratio for each cell
    ratios = []
    for i in range(m):
        if i not in top_markers:
            ratios.append(1.0)
            continue
            
        markers = top_markers[i]
        cell_type = tissue_cells[i][1]
        delete_idxs = [idx for idx in range(m) if tissue_cells[idx][1] == cell_type]
        
        ratio_list = []
        for marker_id in markers:
            column_data = nTPM_matrix[:, marker_id].copy()
            column_data = np.delete(column_data, delete_idxs)
            
            off_exp = np.max(column_data) if len(column_data) > 0 else 0
            targ_exp = nTPM_matrix[i, marker_id]
            
            if off_exp == 0 and targ_exp == 0:
                ratio_list.append(1.0)
            elif off_exp == 0:
                ratio_list.append(float('inf'))
            else:
                ratio_list.append(targ_exp / off_exp)
        
        ratios.append(max(ratio_list) if ratio_list else 1.0)
    
    return ratios


def across_cell_analysis_top2_combination(
    tissue_cells: List,
    genes: List,
    nTPM_matrix: np.ndarray,
    markers_file: str,
    output_dir: str,
    method_name: str = "high"
) -> List[float]:
    """
    Analyze best 2-marker combination from top 10 markers.
    
    Finds the pair of markers with highest product of target/off-target ratios.
    Returns the ratios for each cell type.
    
    Args:
        tissue_cells: List of [tissue, cell] pairs
        genes: List of gene names
        nTPM_matrix: Expression matrix
        markers_file: Path to recommended markers CSV
        output_dir: Directory to save results
        method_name: Name for output files
        
    Returns:
        List of ratio products for each cell type
    """
    m = len(tissue_cells)
    cell_strs = [f"{t} {c}" for t, c in tissue_cells]
    
    rec_df = pd.read_csv(markers_file)
    rec_df['markers'] = rec_df['markers'].apply(lambda x: ast.literal_eval(x))
    
    # Build mapping
    top_markers = {}
    for i in range(min(m, len(rec_df))):
        cell = rec_df.iloc[i, 0]
        if cell not in cell_strs:
            continue
        cell_id = cell_strs.index(cell)
        markers = []
        for j in range(min(10, len(rec_df.iloc[i, 1]))):
            marker = rec_df.iloc[i, 1][j]
            if [marker] in genes:
                marker_id = genes.index([marker])
                markers.append(marker_id)
        if markers:
            top_markers[cell_id] = markers
    
    # Calculate best 2-marker combination for each cell
    ratios = []
    for i in range(m):
        if i not in top_markers or len(top_markers[i]) < 2:
            ratios.append(1.0)
            continue
            
        markers = top_markers[i]
        cell_type = tissue_cells[i][1]
        delete_idxs = [idx for idx in range(m) if tissue_cells[idx][1] == cell_type]
        
        # Pre-compute ratios for all markers
        marker_ratios = []
        for marker_id in markers:
            column_data = nTPM_matrix[:, marker_id].copy()
            column_data = np.delete(column_data, delete_idxs)
            
            off_exp = np.max(column_data) if len(column_data) > 0 else 0
            targ_exp = nTPM_matrix[i, marker_id]
            
            if off_exp == 0 and targ_exp == 0:
                marker_ratios.append(1.0)
            elif off_exp == 0:
                marker_ratios.append(float('inf'))
            else:
                marker_ratios.append(targ_exp / off_exp)
        
        # Find best pair
        best_product = 0
        for k in range(len(markers)):
            for l in range(k + 1, len(markers)):
                product = marker_ratios[k] * marker_ratios[l]
                if product > best_product:
                    best_product = product
        
        ratios.append(best_product if best_product > 0 else 1.0)
    
    return ratios


def plot_ratio_distribution(
    single_marker_ratios: List[float],
    two_marker_ratios: List[float],
    output_dir: str,
    method_name: str = "high"
) -> None:
    """
    Create a grouped bar chart showing the percentage of cell types 
    with ratios >= different thresholds (2, 3, 4, 5, 6, >6).
    
    Args:
        single_marker_ratios: List of ratios from single marker analysis
        two_marker_ratios: List of ratios from two-marker combination analysis
        output_dir: Directory to save the plot
        method_name: Name for output file
    """
    # Define thresholds
    thresholds = [2, 3, 4, 5, 6]
    threshold_labels = ['2', '3', '4', '5', '6', '>6']
    
    # Calculate percentages for single marker
    single_percentages = []
    total_cells = len(single_marker_ratios)
    
    for threshold in thresholds:
        count = sum(1 for r in single_marker_ratios if (np.isfinite(r) and r >= threshold) or (not np.isfinite(r)))
        single_percentages.append((count / total_cells) * 100)
    
    # For >6 threshold
    count_gt6 = sum(1 for r in single_marker_ratios if (np.isfinite(r) and r > 6) or (not np.isfinite(r)))
    single_percentages.append((count_gt6 / total_cells) * 100)
    
    # Calculate percentages for two markers combined
    two_percentages = []
    
    for threshold in thresholds:
        count = sum(1 for r in two_marker_ratios if (np.isfinite(r) and r >= threshold) or (not np.isfinite(r)))
        two_percentages.append((count / total_cells) * 100)
    
    # For >6 threshold
    count_gt6 = sum(1 for r in two_marker_ratios if (np.isfinite(r) and r > 6) or (not np.isfinite(r)))
    two_percentages.append((count_gt6 / total_cells) * 100)
    
    # Create the plot
    output_path = Path(output_dir)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(threshold_labels))
    width = 0.35  # Width of bars
    
    # Create bars - using specific colors to match the reference image
    bars1 = ax.bar(x - width/2, single_percentages, width, label='Single marker', color='#FF0000')  # Red
    bars2 = ax.bar(x + width/2, two_percentages, width, label='Two markers combined (multiplied)', color='#FF69B4')  # Hot pink/magenta
    
    # Customize the plot
    ax.set_xlabel('Fold difference in the expression level to distinguish different cell types', fontsize=12)
    ax.set_ylabel('Percentage of cell types', fontsize=12)
    ax.set_title('Number of cell types distinguishable with surface marker(s)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(threshold_labels)
    ax.set_ylim(0, 80)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save as both PNG and SVG
    fig.savefig(output_path / f'ratio_distribution_{method_name}.png', 
                format='png', dpi=150, bbox_inches='tight')
    fig.savefig(output_path / f'ratio_distribution_{method_name}.svg', 
                format='svg', bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: ratio_distribution_{method_name}.png and ratio_distribution_{method_name}.svg")


def run_plotting(
    data_dir: str,
    output_dir: str,
    plots_dir: str,
    verbose: bool = True
) -> None:
    """
    Run all plotting and analysis tasks.
    
    Args:
        data_dir: Directory containing processed data files
        output_dir: Directory containing marker recommendation files
        plots_dir: Directory to save plots
        verbose: Whether to print progress
    """
    if verbose:
        print("\n" + "=" * 60)
        print("GENERATING PLOTS AND ANALYSIS")
        print("=" * 60)
    
    # Create plots directory
    plots_path = Path(plots_dir)
    plots_path.mkdir(parents=True, exist_ok=True)
    
    # Load data using shared utilities
    if verbose:
        print("\nLoading data...")
    
    data_path = Path(data_dir)
    tissue_cells = load_tissue_cells(data_path / "tissue_cell_pairs.tsv")
    genes = load_gene_list(data_path / "gene_list.csv")
    nTPM_high, nTPM_median, _, _ = load_expression_matrices(data_dir)
    
    if verbose:
        print(f"  Loaded {len(tissue_cells)} tissue-cell pairs")
        print(f"  Loaded {len(genes)} genes")
    
    output_path = Path(output_dir)
    
    # Run analyses for HIGH method
    if verbose:
        print("\nAnalyzing HIGH method results...")
    
    markers_high = output_path / 'recommended_whole_body_markers_high.csv'
    if markers_high.exists():
        single_ratios_high = across_cell_analysis_top10(tissue_cells, genes, nTPM_high, str(markers_high),
                                                        plots_dir, "high")
        two_ratios_high = across_cell_analysis_top2_combination(tissue_cells, genes, nTPM_high, 
                                                                 str(markers_high), plots_dir, "high")
        plot_ratio_distribution(single_ratios_high, two_ratios_high, plots_dir, "high")
    else:
        print(f"  Warning: {markers_high} not found, skipping HIGH analysis")
    
    # Run analyses for MEDIAN method
    if verbose:
        print("\nAnalyzing MEDIAN method results...")
    
    markers_median = output_path / 'recommended_whole_body_markers_median.csv'
    if markers_median.exists():
        single_ratios_median = across_cell_analysis_top10(tissue_cells, genes, nTPM_median, str(markers_median),
                                               plots_dir, "median")
        two_ratios_median = across_cell_analysis_top2_combination(tissue_cells, genes, nTPM_median,
                                                                str(markers_median), plots_dir, "median")
        plot_ratio_distribution(single_ratios_median, two_ratios_median, plots_dir, "median")
    else:
        print(f"  Warning: {markers_median} not found, skipping MEDIAN analysis")
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"Plots and analysis saved to: {plots_dir}")
        print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate plots and analysis from marker prediction results"
    )
    parser.add_argument(
        "--data-dir", 
        default="Results",
        help="Directory containing processed data files (default: Results)"
    )
    parser.add_argument(
        "--output-dir",
        default="Results",
        help="Directory containing marker recommendation files (default: Results)"
    )
    parser.add_argument(
        "--plots-dir",
        default="Plots",
        help="Directory to save plots (default: Plots)"
    )
    
    args = parser.parse_args()
    
    run_plotting(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        plots_dir=args.plots_dir
    )
