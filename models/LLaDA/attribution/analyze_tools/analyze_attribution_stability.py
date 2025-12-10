#!/usr/bin/env python3
"""
Improved stability analysis focusing on important heads.
Filter out near-zero attribution heads to get more meaningful stability metrics.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cosine
import json
from typing import Dict, List, Tuple


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


def get_important_heads_mask(arrays: List[np.ndarray], threshold: float = 0.01) -> np.ndarray:
    """
    Create a mask for important heads based on mean absolute attribution across runs.
    
    Args:
        arrays: List of attribution arrays from different runs
        threshold: Minimum absolute attribution to consider a head important
    
    Returns:
        Boolean mask of shape (n_layers, n_heads)
    """
    stacked = np.stack(arrays, axis=0)  # (n_runs, n_layers, n_heads)
    mean_abs_attr = np.abs(stacked).mean(axis=0)  # (n_layers, n_heads)
    mask = mean_abs_attr >= threshold
    return mask


def compute_masked_metrics(arr1: np.ndarray, arr2: np.ndarray, mask: np.ndarray) -> Dict:
    """
    Compute similarity metrics only for masked (important) positions.
    """
    # Apply mask
    flat1 = arr1[mask].flatten()
    flat2 = arr2[mask].flatten()
    
    if len(flat1) == 0:
        return None
    
    # Pearson correlation
    pearson_corr, pearson_pval = stats.pearsonr(flat1, flat2)
    
    # Spearman correlation
    spearman_corr, spearman_pval = stats.spearmanr(flat1, flat2)
    
    # Cosine similarity
    cosine_sim = 1 - cosine(flat1, flat2)
    
    # Normalized MSE
    mse = np.mean((flat1 - flat2) ** 2)
    variance = (np.var(flat1) + np.var(flat2)) / 2
    normalized_mse = mse / variance if variance > 0 else mse
    
    return {
        'pearson': pearson_corr,
        'spearman': spearman_corr,
        'cosine': cosine_sim,
        'normalized_mse': normalized_mse,
        'n_elements': len(flat1)
    }


def compute_masked_cv(arrays: List[np.ndarray], mask: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute coefficient of variation only for important heads.
    
    Returns:
        cv: CV array (same shape as input, with inf for unimportant heads)
        mean_cv: Mean CV over important heads
    """
    stacked = np.stack(arrays, axis=0)  # (n_runs, n_layers, n_heads)
    mean = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0)
    
    # Compute CV
    cv = np.full_like(mean, np.inf)
    valid = (mean != 0) & mask
    cv[valid] = std[valid] / np.abs(mean[valid])
    
    # Mean CV over important heads
    mean_cv = cv[mask].mean() if mask.sum() > 0 else np.inf
    
    return cv, mean_cv


def analyze_category_stability(results: Dict[str, Dict[str, np.ndarray]], 
                               category: str,
                               threshold: float = 0.01) -> Dict:
    """
    Analyze stability for a category, focusing on important heads.
    """
    run_names = sorted(results.keys())
    arrays = [results[run][category] for run in run_names]
    
    # Get important heads mask
    mask = get_important_heads_mask(arrays, threshold=threshold)
    n_important = mask.sum()
    n_total = mask.size
    
    print(f"\n{'='*80}")
    print(f"Category: {category}")
    print(f"{'='*80}")
    print(f"Important heads (|attr| >= {threshold}): {n_important}/{n_total} ({100*n_important/n_total:.1f}%)")
    
    # Compute pairwise metrics
    n_runs = len(run_names)
    pairwise_metrics = {
        'pearson': [],
        'spearman': [],
        'cosine': [],
        'normalized_mse': []
    }
    
    for i in range(n_runs):
        for j in range(i+1, n_runs):
            metrics = compute_masked_metrics(arrays[i], arrays[j], mask)
            if metrics is not None:
                pairwise_metrics['pearson'].append(metrics['pearson'])
                pairwise_metrics['spearman'].append(metrics['spearman'])
                pairwise_metrics['cosine'].append(metrics['cosine'])
                pairwise_metrics['normalized_mse'].append(metrics['normalized_mse'])
    
    # Compute CV for important heads
    cv, mean_cv = compute_masked_cv(arrays, mask)
    
    # Print statistics
    print(f"\nPairwise Similarity (Important Heads Only):")
    for metric_name, values in pairwise_metrics.items():
        values = np.array(values)
        print(f"  {metric_name}:")
        print(f"    Mean: {values.mean():.4f}")
        print(f"    Std:  {values.std():.4f}")
        print(f"    Min:  {values.min():.4f}")
        print(f"    Max:  {values.max():.4f}")
    
    print(f"\nCoefficient of Variation (Important Heads):")
    print(f"  Mean CV: {mean_cv:.4f}")
    print(f"  Max CV:  {cv[mask].max():.4f}")
    print(f"  Min CV:  {cv[mask].min():.4f}")
    
    # Assess stability
    mean_pearson = np.mean(pairwise_metrics['pearson'])
    
    if mean_pearson > 0.90 and mean_cv < 0.5:
        stability = "HIGH"
        emoji = "✅"
    elif mean_pearson > 0.80 and mean_cv < 1.0:
        stability = "MODERATE-HIGH"
        emoji = "🟢"
    elif mean_pearson > 0.70:
        stability = "MODERATE"
        emoji = "🟡"
    else:
        stability = "LOW"
        emoji = "⚠️"
    
    print(f"\n{emoji} Stability Assessment: {stability}")
    print(f"{'='*80}")
    
    return {
        'category': category,
        'n_important_heads': int(n_important),
        'n_total_heads': int(n_total),
        'percent_important': float(100 * n_important / n_total),
        'threshold': threshold,
        'pairwise_metrics': {k: [float(v) for v in vals] for k, vals in pairwise_metrics.items()},
        'mean_pearson': float(mean_pearson),
        'mean_spearman': float(np.mean(pairwise_metrics['spearman'])),
        'mean_cosine': float(np.mean(pairwise_metrics['cosine'])),
        'mean_cv': float(mean_cv),
        'max_cv': float(cv[mask].max()),
        'min_cv': float(cv[mask].min()),
        'stability': stability
    }


def plot_important_heads_mask(mask: np.ndarray, category: str, output_dir: str):
    """Plot which heads are considered important."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(mask.astype(float), aspect='auto', cmap='RdYlGn', interpolation='nearest')
    ax.set_xlabel('Head Index', fontsize=12)
    ax.set_ylabel('Layer Index', fontsize=12)
    ax.set_title(f'Important Heads Mask - Category: {category}', fontsize=14)
    
    # Add text annotations
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j]:
                ax.text(j, i, '✓', ha='center', va='center', color='white', fontsize=6)
    
    plt.colorbar(im, ax=ax, label='Important (1) / Unimportant (0)')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f'important_heads_mask_{category}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved mask: {output_file}")
    plt.close()


def plot_top_k_heads(results: Dict[str, Dict[str, np.ndarray]], 
                     category: str, 
                     k: int = 20,
                     output_dir: str = None):
    """
    Plot the top-K most important heads and their stability across runs.
    """
    run_names = sorted(results.keys())
    arrays = [results[run][category] for run in run_names]
    
    # Compute mean absolute attribution
    stacked = np.stack(arrays, axis=0)
    mean_abs_attr = np.abs(stacked).mean(axis=0)
    
    # Get top-K positions
    flat_mean = mean_abs_attr.flatten()
    top_k_indices = np.argsort(flat_mean)[-k:][::-1]
    
    # Convert to (layer, head) coordinates
    n_heads = arrays[0].shape[1]
    top_k_coords = [(idx // n_heads, idx % n_heads) for idx in top_k_indices]
    
    # Extract values for top-K heads across runs
    top_k_values = np.zeros((k, len(run_names)))
    for run_idx, arr in enumerate(arrays):
        for head_idx, (layer, head) in enumerate(top_k_coords):
            top_k_values[head_idx, run_idx] = arr[layer, head]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(k)
    width = 0.25
    
    for run_idx, run_name in enumerate(run_names):
        offset = (run_idx - 1) * width
        ax.bar(x + offset, top_k_values[:, run_idx], width, 
               label=run_name, alpha=0.8)
    
    ax.set_xlabel('Top-K Head Rank', fontsize=12)
    ax.set_ylabel('Attribution Score', fontsize=12)
    ax.set_title(f'Top-{k} Most Important Heads - Category: {category}', fontsize=14)
    ax.legend()
    
    # Set x-tick labels as (layer, head)
    labels = [f'L{l}H{h}' for l, h in top_k_coords]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    
    if output_dir:
        output_file = os.path.join(output_dir, f'top_{k}_heads_{category}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  Saved top-K plot: {output_file}")
    
    plt.close()
    
    return top_k_coords, top_k_values


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Improved attribution stability analysis')
    parser.add_argument('--results_dir', type=str,
                       default='/home/qiheng/Projects/adaptive-dllm/models/LLaDA/attribution/attribution_results_20251123_044549',
                       help='Directory containing run results')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for analysis results')
    parser.add_argument('--threshold', type=float, default=0.01,
                       help='Threshold for considering a head important (default: 0.01)')
    parser.add_argument('--top_k', type=int, default=20,
                       help='Number of top heads to visualize (default: 20)')
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, 'stability_analysis_improved')
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("Improved Attribution Stability Analysis")
    print("="*80)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Importance threshold: {args.threshold}")
    print(f"Top-K heads: {args.top_k}")
    print("="*80)
    
    # Load results
    print("\nLoading attribution results...")
    results = load_attribution_results(args.results_dir)
    
    # Get categories
    categories = sorted(list(results[list(results.keys())[0]].keys()))
    print(f"\nCategories to analyze: {categories}")
    
    # Analyze each category
    all_results = {}
    
    for category in categories:
        # Run analysis
        category_result = analyze_category_stability(
            results, category, threshold=args.threshold
        )
        all_results[category] = category_result
        
        # Generate plots
        run_names = sorted(results.keys())
        arrays = [results[run][category] for run in run_names]
        mask = get_important_heads_mask(arrays, threshold=args.threshold)
        
        print(f"\nGenerating plots for {category}...")
        plot_important_heads_mask(mask, category, args.output_dir)
        plot_top_k_heads(results, category, k=args.top_k, output_dir=args.output_dir)
    
    # Save results
    output_file = os.path.join(args.output_dir, 'stability_analysis_improved.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")
    
    # Print summary table
    print("\n" + "="*80)
    print("FINAL STABILITY ASSESSMENT (Focusing on Important Heads)")
    print("="*80)
    print(f"\n{'Category':<12} {'Pearson':<8} {'Spearman':<9} {'Cosine':<8} {'Mean CV':<9} {'Stability':<15}")
    print("-" * 80)
    
    for category in categories:
        res = all_results[category]
        stability_display = f"{res['stability']}"
        
        if res['stability'] == 'HIGH':
            emoji = '✅'
        elif res['stability'] == 'MODERATE-HIGH':
            emoji = '🟢'
        elif res['stability'] == 'MODERATE':
            emoji = '🟡'
        else:
            emoji = '⚠️'
        
        print(f"{category:<12} {res['mean_pearson']:<8.4f} {res['mean_spearman']:<9.4f} "
              f"{res['mean_cosine']:<8.4f} {res['mean_cv']:<9.4f} {emoji} {stability_display}")
    
    print("="*80)
    print("\nKey Findings:")
    print("- Analysis focuses only on important heads (|attribution| >= threshold)")
    print("- This provides more meaningful stability metrics")
    print("- High Pearson correlation indicates consistent attribution patterns")
    print("- Low CV indicates stable attribution values across runs")
    print("="*80)


if __name__ == '__main__':
    main()


