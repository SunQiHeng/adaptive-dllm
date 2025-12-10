#!/usr/bin/env python3
"""
Analyze stability of cross-category averaged attribution.
Average all categories within each run, then compare across runs.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cosine
import json
from typing import Dict, List


def load_attribution_results(base_dir: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Load attribution results from multiple runs."""
    results = {}
    run_dirs = [d for d in os.listdir(base_dir) if d.startswith('run_')]
    run_dirs = sorted(run_dirs)
    
    print(f"Found {len(run_dirs)} runs: {run_dirs}")
    
    for run_dir in run_dirs:
        run_path = os.path.join(base_dir, run_dir)
        results[run_dir] = {}
        npy_files = [f for f in os.listdir(run_path) if f.endswith('.npy')]
        
        for npy_file in npy_files:
            category = npy_file.replace('attribution_', '').replace('.npy', '')
            file_path = os.path.join(run_path, npy_file)
            results[run_dir][category] = np.load(file_path)
            print(f"  Loaded {run_dir}/{category}: shape {results[run_dir][category].shape}")
    
    return results


def compute_run_average(run_data: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Compute average attribution across all categories for a single run.
    
    Args:
        run_data: Dict[category_name -> attribution_array]
    
    Returns:
        Average attribution array (n_layers, n_heads)
    """
    categories = sorted(run_data.keys())
    all_attributions = [run_data[cat] for cat in categories]
    
    # Stack and average
    stacked = np.stack(all_attributions, axis=0)  # (n_categories, n_layers, n_heads)
    averaged = stacked.mean(axis=0)  # (n_layers, n_heads)
    
    return averaged


def analyze_cross_run_similarity(run_averages: Dict[str, np.ndarray]):
    """
    Analyze similarity between run-averaged attributions.
    """
    run_names = sorted(run_averages.keys())
    n_runs = len(run_names)
    
    print(f"\n{'='*80}")
    print(f"Cross-Run Similarity Analysis (Category-Averaged)")
    print(f"{'='*80}\n")
    
    # Basic statistics for each run
    print("Statistics for each run (averaged across categories):")
    print(f"{'Run':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 68)
    
    for run_name in run_names:
        data = run_averages[run_name]
        print(f"{run_name:<20} {data.mean():<12.6f} {data.std():<12.6f} "
              f"{data.min():<12.6f} {data.max():<12.6f}")
    
    # Pairwise comparisons
    print(f"\n{'='*80}")
    print("Pairwise Similarity Metrics:")
    print(f"{'Run Pair':<40} {'Pearson':<10} {'Spearman':<10} {'Cosine':<10}")
    print("-" * 80)
    
    all_metrics = {
        'pearson': [],
        'spearman': [],
        'cosine': []
    }
    
    for i, run1 in enumerate(run_names):
        for j, run2 in enumerate(run_names):
            if i < j:
                arr1 = run_averages[run1]
                arr2 = run_averages[run2]
                
                # Pearson
                flat1 = arr1.flatten()
                flat2 = arr2.flatten()
                pearson, _ = stats.pearsonr(flat1, flat2)
                
                # Spearman
                spearman, _ = stats.spearmanr(flat1, flat2)
                
                # Cosine similarity
                cosine_sim = 1 - cosine(flat1, flat2)
                
                all_metrics['pearson'].append(pearson)
                all_metrics['spearman'].append(spearman)
                all_metrics['cosine'].append(cosine_sim)
                
                print(f"{run1} vs {run2:<25} {pearson:<10.4f} {spearman:<10.4f} {cosine_sim:<10.4f}")
    
    # Summary
    print(f"\n{'='*80}")
    print("Summary Statistics:")
    print("-" * 80)
    print(f"Mean Pearson Correlation:  {np.mean(all_metrics['pearson']):.4f} ± {np.std(all_metrics['pearson']):.4f}")
    print(f"Mean Spearman Correlation: {np.mean(all_metrics['spearman']):.4f} ± {np.std(all_metrics['spearman']):.4f}")
    print(f"Mean Cosine Similarity:    {np.mean(all_metrics['cosine']):.4f} ± {np.std(all_metrics['cosine']):.4f}")
    
    # Stability assessment
    mean_pearson = np.mean(all_metrics['pearson'])
    
    print(f"\n{'='*80}")
    print("Stability Assessment:")
    print("-" * 80)
    
    if mean_pearson > 0.95:
        stability = "VERY HIGH ✅"
        interpretation = "Category-averaged attributions are extremely consistent across runs"
    elif mean_pearson > 0.90:
        stability = "HIGH 🟢"
        interpretation = "Category-averaged attributions are highly consistent across runs"
    elif mean_pearson > 0.80:
        stability = "MODERATE-HIGH 🟡"
        interpretation = "Category-averaged attributions show good consistency across runs"
    elif mean_pearson > 0.70:
        stability = "MODERATE 🟡"
        interpretation = "Category-averaged attributions show moderate consistency across runs"
    else:
        stability = "LOW ⚠️"
        interpretation = "Category-averaged attributions show low consistency across runs"
    
    print(f"Stability Level: {stability}")
    print(f"Interpretation: {interpretation}")
    print("="*80)
    
    return all_metrics


def plot_averaged_attributions(run_averages: Dict[str, np.ndarray], output_dir: str):
    """Plot the averaged attributions for each run."""
    run_names = sorted(run_averages.keys())
    n_runs = len(run_names)
    
    # Find global min/max for consistent color scale
    all_data = list(run_averages.values())
    vmin = min(data.min() for data in all_data)
    vmax = max(data.max() for data in all_data)
    
    # Plot 1: Side-by-side heatmaps
    fig, axes = plt.subplots(1, n_runs, figsize=(6*n_runs, 6))
    if n_runs == 1:
        axes = [axes]
    
    fig.suptitle('Category-Averaged Attribution Across Runs', fontsize=16)
    
    for idx, run in enumerate(run_names):
        data = run_averages[run]
        ax = axes[idx]
        
        im = ax.imshow(data, aspect='auto', cmap='RdBu_r', 
                      vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_xlabel('Head Index', fontsize=10)
        ax.set_ylabel('Layer Index', fontsize=10)
        ax.set_title(f'{run}', fontsize=12)
        
        if idx == n_runs - 1:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Attribution Score', fontsize=10)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'averaged_attribution_comparison.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()
    
    # Plot 2: Difference maps
    if n_runs >= 2:
        fig, axes = plt.subplots(1, n_runs-1, figsize=(6*(n_runs-1), 6))
        if n_runs == 2:
            axes = [axes]
        
        fig.suptitle('Attribution Differences Between Runs', fontsize=16)
        
        baseline_run = run_names[0]
        baseline_data = run_averages[baseline_run]
        
        for idx, run in enumerate(run_names[1:]):
            data = run_averages[run]
            diff = data - baseline_data
            
            ax = axes[idx]
            im = ax.imshow(diff, aspect='auto', cmap='RdBu', 
                          vmin=-np.abs(diff).max(), vmax=np.abs(diff).max(),
                          interpolation='nearest')
            ax.set_xlabel('Head Index', fontsize=10)
            ax.set_ylabel('Layer Index', fontsize=10)
            ax.set_title(f'{run} - {baseline_run}', fontsize=12)
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Attribution Difference', fontsize=10)
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, 'averaged_attribution_differences.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()
    
    # Plot 3: Scatter plots between runs
    if n_runs >= 2:
        n_pairs = (n_runs * (n_runs - 1)) // 2
        fig, axes = plt.subplots(1, n_pairs, figsize=(6*n_pairs, 5))
        if n_pairs == 1:
            axes = [axes]
        
        fig.suptitle('Attribution Correlation Between Runs', fontsize=16)
        
        pair_idx = 0
        for i, run1 in enumerate(run_names):
            for j, run2 in enumerate(run_names):
                if i < j:
                    data1 = run_averages[run1].flatten()
                    data2 = run_averages[run2].flatten()
                    
                    ax = axes[pair_idx]
                    ax.scatter(data1, data2, alpha=0.3, s=10)
                    
                    # Add regression line
                    z = np.polyfit(data1, data2, 1)
                    p = np.poly1d(z)
                    x_line = np.array([data1.min(), data1.max()])
                    ax.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'y={z[0]:.2f}x+{z[1]:.2f}')
                    
                    # Add diagonal line
                    ax.plot([data1.min(), data1.max()], [data1.min(), data1.max()], 
                           'k--', alpha=0.5, label='y=x')
                    
                    # Compute correlation
                    corr, _ = stats.pearsonr(data1, data2)
                    
                    ax.set_xlabel(f'{run1}', fontsize=10)
                    ax.set_ylabel(f'{run2}', fontsize=10)
                    ax.set_title(f'r = {corr:.4f}', fontsize=12)
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                    
                    pair_idx += 1
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, 'averaged_attribution_scatter.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()


def plot_per_layer_analysis(run_averages: Dict[str, np.ndarray], output_dir: str):
    """Analyze attribution per layer across runs."""
    run_names = sorted(run_averages.keys())
    n_layers = list(run_averages.values())[0].shape[0]
    
    # Compute per-layer mean
    layer_means = {run: [] for run in run_names}
    for run_name in run_names:
        data = run_averages[run_name]
        for layer_idx in range(n_layers):
            layer_means[run_name].append(data[layer_idx].mean())
    
    # Plot per-layer consistency
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for run_name in run_names:
        ax.plot(range(n_layers), layer_means[run_name], marker='o', label=run_name, linewidth=2)
    
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Mean Attribution', fontsize=12)
    ax.set_title('Per-Layer Attribution Across Runs (Category-Averaged)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'per_layer_attribution.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze cross-category averaged attribution stability')
    parser.add_argument('--results_dir', type=str,
                       default='/home/qiheng/Projects/adaptive-dllm/models/LLaDA/attribution/attribution_results_20251123_044549',
                       help='Directory containing run results')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for analysis results')
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, 'cross_category_average_analysis')
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("Cross-Category Average Attribution Stability Analysis")
    print("="*80)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    
    # Load results
    print("\nLoading attribution results...")
    results = load_attribution_results(args.results_dir)
    
    # Compute average for each run
    print("\n" + "="*80)
    print("Computing category-averaged attribution for each run...")
    print("-" * 80)
    
    run_averages = {}
    for run_name, run_data in results.items():
        averaged = compute_run_average(run_data)
        run_averages[run_name] = averaged
        print(f"{run_name}: {len(run_data)} categories averaged → shape {averaged.shape}")
    
    # Save averaged data
    for run_name, averaged in run_averages.items():
        output_file = os.path.join(args.output_dir, f'{run_name}_averaged.npy')
        np.save(output_file, averaged)
        print(f"  Saved: {output_file}")
    
    # Analyze similarity
    metrics = analyze_cross_run_similarity(run_averages)
    
    # Generate plots
    print(f"\n{'='*80}")
    print("Generating plots...")
    print("-" * 80)
    plot_averaged_attributions(run_averages, args.output_dir)
    plot_per_layer_analysis(run_averages, args.output_dir)
    
    # Save metrics
    output_file = os.path.join(args.output_dir, 'similarity_metrics.json')
    with open(output_file, 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in metrics.items()}, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Analysis complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

