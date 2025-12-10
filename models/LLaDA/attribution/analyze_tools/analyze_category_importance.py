#!/usr/bin/env python3
"""
Analyze the average importance of different categories across runs.
Check if the category importance ranking is consistent across runs.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
from typing import Dict, List
import pandas as pd


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
    
    return results


def compute_category_statistics(results: Dict[str, Dict[str, np.ndarray]]) -> pd.DataFrame:
    """
    Compute statistics for each category across runs.
    
    Returns:
        DataFrame with columns: [run, category, mean_abs, mean, std, max_abs, sum_abs]
    """
    data = []
    
    for run_name, categories in results.items():
        for category, attribution in categories.items():
            stats_dict = {
                'run': run_name,
                'category': category,
                'mean_abs': np.abs(attribution).mean(),  # 平均绝对重要性
                'mean': attribution.mean(),               # 平均值（带符号）
                'std': attribution.std(),                 # 标准差
                'max_abs': np.abs(attribution).max(),     # 最大绝对值
                'sum_abs': np.abs(attribution).sum(),     # 总绝对重要性
                'median_abs': np.median(np.abs(attribution)),  # 中位数
            }
            data.append(stats_dict)
    
    return pd.DataFrame(data)


def analyze_category_consistency(df: pd.DataFrame) -> Dict:
    """
    Analyze consistency of category importance across runs.
    """
    categories = sorted(df['category'].unique())
    runs = sorted(df['run'].unique())
    
    print(f"\n{'='*80}")
    print(f"Category Importance Analysis Across Runs")
    print(f"{'='*80}\n")
    
    # 1. Average importance per category per run
    print("Average Absolute Importance by Category and Run:")
    print(f"{'Category':<12}", end='')
    for run in runs:
        print(f"{run:<20}", end='')
    print("Mean    Std     CV")
    print("-" * 80)
    
    category_means = {}
    for category in categories:
        cat_data = df[df['category'] == category]
        values = cat_data['mean_abs'].values
        category_means[category] = values
        
        print(f"{category:<12}", end='')
        for val in values:
            print(f"{val:<20.6f}", end='')
        print(f"{values.mean():<8.6f}{values.std():<8.6f}{values.std()/values.mean():<8.4f}")
    
    # 2. Category ranking per run
    print(f"\n{'='*80}")
    print("Category Ranking by Importance (1=most important):")
    print(f"{'Category':<12}", end='')
    for run in runs:
        print(f"{run:<20}", end='')
    print("Rank Variance")
    print("-" * 80)
    
    rankings = {cat: [] for cat in categories}
    for run in runs:
        run_data = df[df['run'] == run].sort_values('mean_abs', ascending=False)
        for rank, (_, row) in enumerate(run_data.iterrows(), 1):
            rankings[row['category']].append(rank)
    
    for category in categories:
        ranks = rankings[category]
        print(f"{category:<12}", end='')
        for rank in ranks:
            print(f"{rank:<20}", end='')
        print(f"{np.var(ranks):.4f}")
    
    # 3. Spearman rank correlation between runs
    print(f"\n{'='*80}")
    print("Spearman Rank Correlation Between Runs:")
    print(f"{'Run Pair':<40} {'Correlation':<12} {'P-value'}")
    print("-" * 80)
    
    correlations = []
    for i, run1 in enumerate(runs):
        for j, run2 in enumerate(runs):
            if i < j:
                data1 = df[df['run'] == run1].sort_values('category')['mean_abs'].values
                data2 = df[df['run'] == run2].sort_values('category')['mean_abs'].values
                corr, pval = stats.spearmanr(data1, data2)
                correlations.append(corr)
                print(f"{run1} vs {run2:<25} {corr:<12.4f} {pval:.6f}")
    
    mean_corr = np.mean(correlations)
    print(f"\nMean Spearman Correlation: {mean_corr:.4f}")
    
    # 4. Pearson correlation
    print(f"\n{'='*80}")
    print("Pearson Correlation Between Runs:")
    print(f"{'Run Pair':<40} {'Correlation':<12} {'P-value'}")
    print("-" * 80)
    
    pearson_corrs = []
    for i, run1 in enumerate(runs):
        for j, run2 in enumerate(runs):
            if i < j:
                data1 = df[df['run'] == run1].sort_values('category')['mean_abs'].values
                data2 = df[df['run'] == run2].sort_values('category')['mean_abs'].values
                corr, pval = stats.pearsonr(data1, data2)
                pearson_corrs.append(corr)
                print(f"{run1} vs {run2:<25} {corr:<12.4f} {pval:.6f}")
    
    mean_pearson = np.mean(pearson_corrs)
    print(f"\nMean Pearson Correlation: {mean_pearson:.4f}")
    
    # 5. Statistical test: repeated measures
    print(f"\n{'='*80}")
    print("Statistical Test (Friedman Test for Related Samples):")
    print("-" * 80)
    
    # Reshape data for Friedman test
    category_arrays = [category_means[cat] for cat in categories]
    stat, pval = stats.friedmanchisquare(*category_arrays)
    print(f"Friedman statistic: {stat:.4f}")
    print(f"P-value: {pval:.6f}")
    if pval < 0.05:
        print("Result: Categories have SIGNIFICANTLY different importance (p < 0.05)")
    else:
        print("Result: No significant difference in category importance (p >= 0.05)")
    
    return {
        'categories': categories,
        'runs': runs,
        'category_means': {k: v.tolist() for k, v in category_means.items()},
        'rankings': rankings,
        'mean_spearman': float(mean_corr),
        'mean_pearson': float(mean_pearson),
        'friedman_stat': float(stat),
        'friedman_pval': float(pval)
    }


def plot_category_importance(df: pd.DataFrame, output_dir: str):
    """Plot category importance across runs."""
    categories = sorted(df['category'].unique())
    runs = sorted(df['run'].unique())
    
    # Plot 1: Bar chart with error bars
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(categories))
    width = 0.25
    
    for i, run in enumerate(runs):
        run_data = df[df['run'] == run].sort_values('category')
        offset = (i - 1) * width
        ax.bar(x + offset, run_data['mean_abs'], width, label=run, alpha=0.8)
    
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Mean Absolute Attribution', fontsize=12)
    ax.set_title('Category Importance Across Runs', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'category_importance_bars.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()
    
    # Plot 2: Line plot showing consistency
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for run in runs:
        run_data = df[df['run'] == run].sort_values('category')
        ax.plot(categories, run_data['mean_abs'], marker='o', label=run, linewidth=2)
    
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Mean Absolute Attribution', fontsize=12)
    ax.set_title('Category Importance Consistency Across Runs', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'category_importance_lines.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()
    
    # Plot 3: Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pivot_data = df.pivot(index='run', columns='category', values='mean_abs')
    pivot_data = pivot_data[sorted(pivot_data.columns)]
    
    sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax)
    ax.set_title('Category Importance Heatmap', fontsize=14)
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Run', fontsize=12)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'category_importance_heatmap.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()
    
    # Plot 4: Box plot showing distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data_for_box = [df[df['category'] == cat]['mean_abs'].values for cat in categories]
    bp = ax.boxplot(data_for_box, labels=categories, patch_artist=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Mean Absolute Attribution', fontsize=12)
    ax.set_title('Category Importance Distribution Across Runs', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'category_importance_boxplot.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_ranking_consistency(rankings: Dict, output_dir: str):
    """Plot ranking consistency across runs."""
    categories = sorted(rankings.keys())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create position for each category
    x = np.arange(len(categories))
    
    for i, (cat, ranks) in enumerate(sorted(rankings.items())):
        ax.plot([i]*len(ranks), ranks, 'o', markersize=10, alpha=0.6)
        ax.plot([i, i], [min(ranks), max(ranks)], 'k-', linewidth=2, alpha=0.3)
    
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Rank (1=Most Important)', fontsize=12)
    ax.set_title('Category Ranking Consistency Across Runs', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, len(categories) + 1)
    ax.invert_yaxis()  # 1 at top
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'category_ranking_consistency.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze category importance consistency')
    parser.add_argument('--results_dir', type=str,
                       default='/home/qiheng/Projects/adaptive-dllm/models/LLaDA/attribution/attribution_results_20251123_044549',
                       help='Directory containing run results')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for analysis results')
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, 'category_importance_analysis')
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("Category Importance Consistency Analysis")
    print("="*80)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    
    # Load results
    print("\nLoading attribution results...")
    results = load_attribution_results(args.results_dir)
    
    # Compute statistics
    print("\nComputing category statistics...")
    df = compute_category_statistics(results)
    
    # Save raw data
    csv_file = os.path.join(args.output_dir, 'category_statistics.csv')
    df.to_csv(csv_file, index=False)
    print(f"Saved statistics to: {csv_file}")
    
    # Analyze consistency
    analysis_results = analyze_category_consistency(df)
    
    # Generate plots
    print(f"\n{'='*80}")
    print("Generating plots...")
    print("-" * 80)
    plot_category_importance(df, args.output_dir)
    plot_ranking_consistency(analysis_results['rankings'], args.output_dir)
    
    # Save analysis results
    output_file = os.path.join(args.output_dir, 'category_analysis_results.json')
    with open(output_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    print(f"\n{'='*80}")
    print(f"Analysis complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*80}")
    
    # Final summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)
    print(f"Mean Spearman Rank Correlation: {analysis_results['mean_spearman']:.4f}")
    print(f"Mean Pearson Correlation:       {analysis_results['mean_pearson']:.4f}")
    
    if analysis_results['mean_spearman'] > 0.9:
        consistency = "VERY HIGH ✅"
    elif analysis_results['mean_spearman'] > 0.7:
        consistency = "HIGH 🟢"
    elif analysis_results['mean_spearman'] > 0.5:
        consistency = "MODERATE 🟡"
    else:
        consistency = "LOW ⚠️"
    
    print(f"\nCategory Importance Consistency: {consistency}")
    print(f"\nInterpretation:")
    print(f"  - High correlation means category importance ranking is consistent")
    print(f"  - This validates that different task types have stable attribution patterns")
    print("="*80)


if __name__ == '__main__':
    main()

