#!/usr/bin/env python3
"""
Compare consistency of head attribution across multiple runs for Dream model
"""
import argparse
import json
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kendalltau
from collections import defaultdict


def load_attribution_results(run_dir):
    """Load attribution results from a run directory"""
    summary_file = os.path.join(run_dir, "attribution_summary.json")
    
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"Summary file not found: {summary_file}")
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    # Load attribution arrays for Dream categories
    results = {}
    for category in ['code', 'math', 'science', 'chat', 'safety']:
        # Try both _avg and regular naming
        npy_file_avg = os.path.join(run_dir, f"attribution_{category}_avg.npy")
        npy_file = os.path.join(run_dir, f"attribution_{category}.npy")
        
        if os.path.exists(npy_file_avg):
            results[category] = np.load(npy_file_avg)
        elif os.path.exists(npy_file):
            results[category] = np.load(npy_file)
    
    return summary, results


def compute_ranking_correlation(scores1, scores2):
    """Compute ranking correlation between two score arrays"""
    # Flatten arrays if they are 2D
    if len(scores1.shape) > 1:
        scores1 = scores1.flatten()
    if len(scores2.shape) > 1:
        scores2 = scores2.flatten()
    
    # Compute Spearman and Kendall correlations
    spearman_corr, spearman_p = spearmanr(scores1, scores2)
    kendall_corr, kendall_p = kendalltau(scores1, scores2)
    
    # Compute top-k overlap
    top_k_values = [10, 20, 50, 100]
    top_k_overlaps = {}
    
    for k in top_k_values:
        if len(scores1) < k:
            continue
        top1 = set(np.argsort(scores1)[-k:])
        top2 = set(np.argsort(scores2)[-k:])
        overlap = len(top1.intersection(top2)) / k
        top_k_overlaps[k] = overlap
    
    return {
        'spearman': spearman_corr,
        'spearman_p': spearman_p,
        'kendall': kendall_corr,
        'kendall_p': kendall_p,
        'top_k_overlap': top_k_overlaps
    }


def compare_runs(run_dirs):
    """Compare attribution results across multiple runs"""
    print("Loading results from all runs...")
    all_summaries = []
    all_results = []
    
    for i, run_dir in enumerate(run_dirs):
        print(f"  Loading run {i+1}: {run_dir}")
        summary, results = load_attribution_results(run_dir)
        all_summaries.append(summary)
        all_results.append(results)
    
    print("\n" + "="*70)
    print("CONSISTENCY ANALYSIS")
    print("="*70)
    
    # Compare each pair of runs
    n_runs = len(run_dirs)
    comparisons = defaultdict(list)
    
    for i in range(n_runs):
        for j in range(i+1, n_runs):
            print(f"\nComparing Run {i+1} vs Run {j+1}")
            print("-" * 50)
            
            # Get categories present in both runs
            categories = set(all_results[i].keys()).intersection(set(all_results[j].keys()))
            
            for category in sorted(categories):
                scores1 = all_results[i][category]
                scores2 = all_results[j][category]
                
                if scores1.shape != scores2.shape:
                    print(f"  Warning: Shape mismatch for {category}")
                    continue
                
                corr_results = compute_ranking_correlation(scores1, scores2)
                comparisons[category].append(corr_results)
                
                print(f"\n  {category.upper()}:")
                print(f"    Spearman correlation: {corr_results['spearman']:.4f} (p={corr_results['spearman_p']:.4e})")
                print(f"    Kendall correlation:  {corr_results['kendall']:.4f} (p={corr_results['kendall_p']:.4e})")
                print(f"    Top-K overlap:")
                for k, overlap in corr_results['top_k_overlap'].items():
                    print(f"      Top-{k}: {overlap:.2%}")
    
    # Compute average correlations across categories
    print("\n" + "="*70)
    print("AVERAGE CORRELATIONS ACROSS CATEGORIES")
    print("="*70)
    
    for category in sorted(comparisons.keys()):
        corr_list = comparisons[category]
        
        avg_spearman = np.mean([c['spearman'] for c in corr_list])
        avg_kendall = np.mean([c['kendall'] for c in corr_list])
        
        print(f"\n{category.upper()}:")
        print(f"  Average Spearman: {avg_spearman:.4f}")
        print(f"  Average Kendall:  {avg_kendall:.4f}")
        
        if 'top_k_overlap' in corr_list[0]:
            print(f"  Average Top-K overlap:")
            for k in corr_list[0]['top_k_overlap'].keys():
                avg_overlap = np.mean([c['top_k_overlap'][k] for c in corr_list])
                print(f"    Top-{k}: {avg_overlap:.2%}")
    
    return comparisons, all_summaries, all_results


def plot_consistency(comparisons, output_dir):
    """Generate visualization plots for consistency analysis"""
    print("\nGenerating plots...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Correlation heatmap
    categories = sorted(comparisons.keys())
    spearman_scores = []
    kendall_scores = []
    
    for category in categories:
        spearman_scores.append(np.mean([c['spearman'] for c in comparisons[category]]))
        kendall_scores.append(np.mean([c['kendall'] for c in comparisons[category]]))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax1.bar(x - width/2, spearman_scores, width, label='Spearman', alpha=0.8)
    ax1.bar(x + width/2, kendall_scores, width, label='Kendall', alpha=0.8)
    ax1.set_xlabel('Category')
    ax1.set_ylabel('Correlation')
    ax1.set_title('Average Rank Correlation Across Runs (Dream)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Plot 2: Top-K overlap
    if comparisons[categories[0]][0]['top_k_overlap']:
        k_values = sorted(comparisons[categories[0]][0]['top_k_overlap'].keys())
        
        for category in categories:
            overlaps = []
            for k in k_values:
                avg_overlap = np.mean([c['top_k_overlap'][k] for c in comparisons[category]])
                overlaps.append(avg_overlap)
            ax2.plot(k_values, overlaps, marker='o', label=category.capitalize())
        
        ax2.set_xlabel('Top-K')
        ax2.set_ylabel('Overlap Ratio')
        ax2.set_title('Top-K Head Overlap Across Runs (Dream)')
        ax2.legend()
        ax2.grid(alpha=0.3)
        ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'consistency_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir}/consistency_analysis.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare attribution consistency across runs for Dream')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Base directory containing run_1, run_2, run_3 subdirectories')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for comparison results (default: base_dir/consistency_analysis)')
    
    args = parser.parse_args()
    
    # Find run directories
    base_dir = Path(args.base_dir)
    run_dirs = []
    
    for run_subdir in sorted(base_dir.glob("run_*")):
        if run_subdir.is_dir():
            run_dirs.append(str(run_subdir))
    
    if len(run_dirs) < 2:
        print(f"Error: Found only {len(run_dirs)} run directories. Need at least 2 for comparison.")
        print(f"Looking in: {base_dir}")
        return
    
    print(f"Found {len(run_dirs)} runs to compare:")
    for run_dir in run_dirs:
        print(f"  - {run_dir}")
    
    # Compare runs
    comparisons, summaries, results = compare_runs(run_dirs)
    
    # Set output directory
    if args.output_dir is None:
        output_dir = os.path.join(args.base_dir, 'consistency_analysis')
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots
    plot_consistency(comparisons, output_dir)
    
    # Save detailed results
    output_file = os.path.join(output_dir, 'consistency_results.json')
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    results_dict = {
        'run_dirs': run_dirs,
        'comparisons': convert_to_serializable(dict(comparisons)),
        'summaries': summaries
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")
    print("\n" + "="*70)
    print("CONSISTENCY ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

