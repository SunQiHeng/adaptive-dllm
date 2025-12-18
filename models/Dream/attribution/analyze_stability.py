#!/usr/bin/env python3
"""
分析多次运行的head归因稳定性
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cosine
import pandas as pd

def load_attribution_data(base_dir, run_folders):
    """加载所有运行的归因数据"""
    categories = ['chat', 'code', 'math', 'safety', 'science']
    
    all_data = {}
    for run_name in run_folders:
        run_dir = Path(base_dir) / run_name
        all_data[run_name] = {}
        
        for category in categories:
            avg_file = run_dir / f'attribution_{category}_avg.npy'
            if avg_file.exists():
                all_data[run_name][category] = np.load(avg_file)
                print(f"Loaded {run_name}/{category}: shape {all_data[run_name][category].shape}")
        
        # Load summary if exists
        summary_file = run_dir / 'attribution_summary.json'
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                all_data[run_name]['summary'] = json.load(f)
    
    return all_data

def calculate_correlation_matrix(data_list):
    """计算多个运行之间的相关性"""
    n_runs = len(data_list)
    correlation_matrix = np.zeros((n_runs, n_runs))
    
    for i in range(n_runs):
        for j in range(n_runs):
            # Flatten the attribution data
            data_i = data_list[i].flatten()
            data_j = data_list[j].flatten()
            
            # Calculate Pearson correlation
            corr, _ = pearsonr(data_i, data_j)
            correlation_matrix[i, j] = corr
    
    return correlation_matrix

def calculate_stability_metrics(all_data, categories):
    """计算稳定性指标"""
    run_names = list(all_data.keys())
    n_runs = len(run_names)
    
    stability_results = {}
    
    for category in categories:
        # Collect data for this category from all runs
        category_data = []
        for run_name in run_names:
            if category in all_data[run_name]:
                category_data.append(all_data[run_name][category])
        
        if len(category_data) != n_runs:
            print(f"Warning: {category} not available in all runs")
            continue
        
        # Stack data: shape (n_runs, n_layers, n_heads)
        stacked_data = np.stack(category_data, axis=0)
        
        # Calculate statistics across runs
        mean_attr = np.mean(stacked_data, axis=0)
        std_attr = np.std(stacked_data, axis=0)
        cv_attr = std_attr / (mean_attr + 1e-10)  # coefficient of variation
        
        # Calculate pairwise correlations
        correlations = []
        spearman_correlations = []
        cosine_similarities = []
        
        for i in range(n_runs):
            for j in range(i+1, n_runs):
                data_i = category_data[i].flatten()
                data_j = category_data[j].flatten()
                
                pearson_corr, _ = pearsonr(data_i, data_j)
                spearman_corr, _ = spearmanr(data_i, data_j)
                cosine_sim = 1 - cosine(data_i, data_j)
                
                correlations.append(pearson_corr)
                spearman_correlations.append(spearman_corr)
                cosine_similarities.append(cosine_sim)
        
        # Top-k head consistency
        k_values = [5, 10, 20]
        top_k_consistency = {}
        
        for k in k_values:
            # For each run, get top-k heads
            top_k_sets = []
            for run_data in category_data:
                flat_indices = np.argsort(run_data.flatten())[-k:]
                top_k_sets.append(set(flat_indices))
            
            # Calculate Jaccard similarity between all pairs
            jaccard_scores = []
            for i in range(n_runs):
                for j in range(i+1, n_runs):
                    intersection = len(top_k_sets[i] & top_k_sets[j])
                    union = len(top_k_sets[i] | top_k_sets[j])
                    jaccard = intersection / union if union > 0 else 0
                    jaccard_scores.append(jaccard)
            
            top_k_consistency[k] = {
                'mean': np.mean(jaccard_scores),
                'std': np.std(jaccard_scores),
                'scores': jaccard_scores
            }
        
        stability_results[category] = {
            'mean_attribution': mean_attr,
            'std_attribution': std_attr,
            'cv_attribution': cv_attr,
            'pearson_correlations': {
                'mean': np.mean(correlations),
                'std': np.std(correlations),
                'values': correlations
            },
            'spearman_correlations': {
                'mean': np.mean(spearman_correlations),
                'std': np.std(spearman_correlations),
                'values': spearman_correlations
            },
            'cosine_similarities': {
                'mean': np.mean(cosine_similarities),
                'std': np.std(cosine_similarities),
                'values': cosine_similarities
            },
            'top_k_consistency': top_k_consistency,
            'raw_data': stacked_data
        }
    
    return stability_results

def plot_correlation_heatmaps(all_data, categories, output_dir):
    """绘制不同运行之间的相关性热图"""
    run_names = list(all_data.keys())
    n_categories = len(categories)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, category in enumerate(categories):
        # Collect data for this category
        category_data = []
        for run_name in run_names:
            if category in all_data[run_name]:
                category_data.append(all_data[run_name][category])
        
        if len(category_data) < 2:
            continue
        
        # Calculate correlation matrix
        corr_matrix = calculate_correlation_matrix(category_data)
        
        # Plot heatmap
        ax = axes[idx]
        im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=0.5, vmax=1.0)
        ax.set_xticks(range(len(run_names)))
        ax.set_yticks(range(len(run_names)))
        ax.set_xticklabels([f'Run {i+1}' for i in range(len(run_names))], rotation=45)
        ax.set_yticklabels([f'Run {i+1}' for i in range(len(run_names))])
        ax.set_title(f'{category.capitalize()} - Pearson Correlation', fontsize=12, fontweight='bold')
        
        # Add correlation values
        for i in range(len(run_names)):
            for j in range(len(run_names)):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=10)
        
        plt.colorbar(im, ax=ax)
    
    # Remove extra subplot
    if n_categories < 6:
        fig.delaxes(axes[5])
    
    plt.suptitle('Cross-Run Correlation Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_run_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation heatmaps to {output_dir / 'cross_run_correlations.png'}")

def plot_cv_heatmaps(stability_results, output_dir):
    """绘制变异系数(CV)热图"""
    categories = list(stability_results.keys())
    n_categories = len(categories)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for idx, category in enumerate(categories):
        cv_data = stability_results[category]['cv_attribution']
        
        ax = axes[idx]
        im = ax.imshow(cv_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.5)
        ax.set_xlabel('Head Index', fontsize=10)
        ax.set_ylabel('Layer Index', fontsize=10)
        ax.set_title(f'{category.capitalize()} - Coefficient of Variation', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label='CV')
        
        # Add mean CV
        mean_cv = np.mean(cv_data)
        ax.text(0.02, 0.98, f'Mean CV: {mean_cv:.4f}', 
                transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Remove extra subplot
    if n_categories < 6:
        fig.delaxes(axes[5])
    
    plt.suptitle('Coefficient of Variation Across Runs', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'cv_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved CV heatmaps to {output_dir / 'cv_heatmaps.png'}")

def plot_stability_summary(stability_results, output_dir):
    """绘制稳定性指标汇总"""
    categories = list(stability_results.keys())
    
    # Prepare data for plotting
    pearson_means = [stability_results[cat]['pearson_correlations']['mean'] for cat in categories]
    pearson_stds = [stability_results[cat]['pearson_correlations']['std'] for cat in categories]
    
    spearman_means = [stability_results[cat]['spearman_correlations']['mean'] for cat in categories]
    spearman_stds = [stability_results[cat]['spearman_correlations']['std'] for cat in categories]
    
    cosine_means = [stability_results[cat]['cosine_similarities']['mean'] for cat in categories]
    cosine_stds = [stability_results[cat]['cosine_similarities']['std'] for cat in categories]
    
    mean_cvs = [np.mean(stability_results[cat]['cv_attribution']) for cat in categories]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Pearson correlations
    ax = axes[0, 0]
    x = np.arange(len(categories))
    ax.bar(x, pearson_means, yerr=pearson_stds, capsize=5, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([cat.capitalize() for cat in categories], rotation=45, ha='right')
    ax.set_ylabel('Pearson Correlation', fontsize=12)
    ax.set_title('Average Pearson Correlation Across Runs', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='0.9 threshold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    
    # Add values on bars
    for i, (mean, std) in enumerate(zip(pearson_means, pearson_stds)):
        ax.text(i, mean + std + 0.02, f'{mean:.3f}', ha='center', fontsize=9)
    
    # Plot 2: Spearman correlations
    ax = axes[0, 1]
    ax.bar(x, spearman_means, yerr=spearman_stds, capsize=5, alpha=0.7, color='lightcoral', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([cat.capitalize() for cat in categories], rotation=45, ha='right')
    ax.set_ylabel('Spearman Correlation', fontsize=12)
    ax.set_title('Average Spearman Correlation Across Runs', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='0.9 threshold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    
    for i, (mean, std) in enumerate(zip(spearman_means, spearman_stds)):
        ax.text(i, mean + std + 0.02, f'{mean:.3f}', ha='center', fontsize=9)
    
    # Plot 3: Cosine similarities
    ax = axes[1, 0]
    ax.bar(x, cosine_means, yerr=cosine_stds, capsize=5, alpha=0.7, color='lightgreen', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([cat.capitalize() for cat in categories], rotation=45, ha='right')
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title('Average Cosine Similarity Across Runs', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='0.9 threshold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    
    for i, (mean, std) in enumerate(zip(cosine_means, cosine_stds)):
        ax.text(i, mean + std + 0.02, f'{mean:.3f}', ha='center', fontsize=9)
    
    # Plot 4: Coefficient of Variation
    ax = axes[1, 1]
    ax.bar(x, mean_cvs, alpha=0.7, color='plum', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([cat.capitalize() for cat in categories], rotation=45, ha='right')
    ax.set_ylabel('Mean CV', fontsize=12)
    ax.set_title('Average Coefficient of Variation', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for i, cv in enumerate(mean_cvs):
        ax.text(i, cv + 0.002, f'{cv:.4f}', ha='center', fontsize=9)
    
    plt.suptitle('Stability Metrics Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'stability_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved stability summary to {output_dir / 'stability_summary.png'}")

def plot_top_k_consistency(stability_results, output_dir):
    """绘制Top-K一致性分析"""
    categories = list(stability_results.keys())
    k_values = [5, 10, 20]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, k in enumerate(k_values):
        ax = axes[idx]
        
        means = [stability_results[cat]['top_k_consistency'][k]['mean'] for cat in categories]
        stds = [stability_results[cat]['top_k_consistency'][k]['std'] for cat in categories]
        
        x = np.arange(len(categories))
        ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels([cat.capitalize() for cat in categories], rotation=45, ha='right')
        ax.set_ylabel('Jaccard Similarity', fontsize=12)
        ax.set_title(f'Top-{k} Head Consistency', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.02, f'{mean:.3f}', ha='center', fontsize=9)
    
    plt.suptitle('Top-K Head Consistency Across Runs (Jaccard Similarity)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'top_k_consistency.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved top-k consistency to {output_dir / 'top_k_consistency.png'}")

def plot_attribution_comparison(all_data, categories, output_dir):
    """绘制不同运行的归因值对比"""
    run_names = list(all_data.keys())
    
    for category in categories:
        # Collect data
        category_data = []
        for run_name in run_names:
            if category in all_data[run_name]:
                category_data.append(all_data[run_name][category])
        
        if len(category_data) < 2:
            continue
        
        n_runs = len(category_data)
        fig, axes = plt.subplots(1, n_runs, figsize=(6*n_runs, 5))
        
        if n_runs == 1:
            axes = [axes]
        
        # Find global min/max for consistent color scale
        vmin = min([data.min() for data in category_data])
        vmax = max([data.max() for data in category_data])
        
        for idx, (data, run_name) in enumerate(zip(category_data, run_names)):
            ax = axes[idx]
            im = ax.imshow(data, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
            ax.set_xlabel('Head Index', fontsize=10)
            ax.set_ylabel('Layer Index', fontsize=10)
            ax.set_title(f'{run_name}', fontsize=12, fontweight='bold')
            plt.colorbar(im, ax=ax)
        
        plt.suptitle(f'{category.capitalize()} - Attribution Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'attribution_comparison_{category}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved attribution comparison for {category}")

def generate_stability_report(stability_results, all_data, output_dir):
    """生成稳定性分析报告"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("HEAD ATTRIBUTION STABILITY ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    run_names = list(all_data.keys())
    report_lines.append(f"Number of runs analyzed: {len(run_names)}")
    report_lines.append(f"Runs: {', '.join(run_names)}")
    report_lines.append("")
    
    categories = list(stability_results.keys())
    
    for category in categories:
        report_lines.append("-" * 80)
        report_lines.append(f"Category: {category.upper()}")
        report_lines.append("-" * 80)
        
        results = stability_results[category]
        
        # Correlation metrics
        report_lines.append(f"\n1. Correlation Metrics:")
        report_lines.append(f"   Pearson Correlation:")
        report_lines.append(f"      Mean: {results['pearson_correlations']['mean']:.4f}")
        report_lines.append(f"      Std:  {results['pearson_correlations']['std']:.4f}")
        report_lines.append(f"      Values: {', '.join([f'{v:.4f}' for v in results['pearson_correlations']['values']])}")
        
        report_lines.append(f"\n   Spearman Correlation:")
        report_lines.append(f"      Mean: {results['spearman_correlations']['mean']:.4f}")
        report_lines.append(f"      Std:  {results['spearman_correlations']['std']:.4f}")
        report_lines.append(f"      Values: {', '.join([f'{v:.4f}' for v in results['spearman_correlations']['values']])}")
        
        report_lines.append(f"\n   Cosine Similarity:")
        report_lines.append(f"      Mean: {results['cosine_similarities']['mean']:.4f}")
        report_lines.append(f"      Std:  {results['cosine_similarities']['std']:.4f}")
        report_lines.append(f"      Values: {', '.join([f'{v:.4f}' for v in results['cosine_similarities']['values']])}")
        
        # Coefficient of Variation
        mean_cv = np.mean(results['cv_attribution'])
        median_cv = np.median(results['cv_attribution'])
        max_cv = np.max(results['cv_attribution'])
        
        report_lines.append(f"\n2. Coefficient of Variation (CV):")
        report_lines.append(f"   Mean CV:   {mean_cv:.4f}")
        report_lines.append(f"   Median CV: {median_cv:.4f}")
        report_lines.append(f"   Max CV:    {max_cv:.4f}")
        
        # Top-K consistency
        report_lines.append(f"\n3. Top-K Head Consistency (Jaccard Similarity):")
        for k in [5, 10, 20]:
            topk = results['top_k_consistency'][k]
            report_lines.append(f"   Top-{k}:")
            report_lines.append(f"      Mean: {topk['mean']:.4f}")
            report_lines.append(f"      Std:  {topk['std']:.4f}")
        
        report_lines.append("")
    
    # Overall assessment
    report_lines.append("=" * 80)
    report_lines.append("OVERALL STABILITY ASSESSMENT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    avg_pearson = np.mean([stability_results[cat]['pearson_correlations']['mean'] for cat in categories])
    avg_spearman = np.mean([stability_results[cat]['spearman_correlations']['mean'] for cat in categories])
    avg_cosine = np.mean([stability_results[cat]['cosine_similarities']['mean'] for cat in categories])
    avg_cv = np.mean([np.mean(stability_results[cat]['cv_attribution']) for cat in categories])
    
    report_lines.append(f"Average Pearson Correlation: {avg_pearson:.4f}")
    report_lines.append(f"Average Spearman Correlation: {avg_spearman:.4f}")
    report_lines.append(f"Average Cosine Similarity: {avg_cosine:.4f}")
    report_lines.append(f"Average Coefficient of Variation: {avg_cv:.4f}")
    report_lines.append("")
    
    # Interpretation
    report_lines.append("Interpretation:")
    report_lines.append("")
    
    if avg_pearson >= 0.95:
        report_lines.append("✓ EXCELLENT stability (Pearson ≥ 0.95)")
    elif avg_pearson >= 0.90:
        report_lines.append("✓ GOOD stability (Pearson ≥ 0.90)")
    elif avg_pearson >= 0.80:
        report_lines.append("⚠ MODERATE stability (Pearson ≥ 0.80)")
    else:
        report_lines.append("✗ POOR stability (Pearson < 0.80)")
    
    if avg_cv <= 0.1:
        report_lines.append("✓ LOW variability (CV ≤ 0.1)")
    elif avg_cv <= 0.2:
        report_lines.append("⚠ MODERATE variability (CV ≤ 0.2)")
    else:
        report_lines.append("✗ HIGH variability (CV > 0.2)")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # Save report
    report_text = '\n'.join(report_lines)
    with open(output_dir / 'stability_report.txt', 'w') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    print(f"\nReport saved to {output_dir / 'stability_report.txt'}")

def main():
    # Configuration
    base_dir = '/home/qiheng/Projects/adaptive-dllm/models/Dream/attribution/attribution_results_20251211_155653'
    run_folders = ['run_1_seed_42', 'run_2_seed_123', 'run_3_seed_2024']
    categories = ['chat', 'code', 'math', 'safety', 'science']
    
    # Create output directory
    output_dir = Path(base_dir) / 'stability_analysis'
    output_dir.mkdir(exist_ok=True)
    
    print("Loading attribution data...")
    all_data = load_attribution_data(base_dir, run_folders)
    
    print("\nCalculating stability metrics...")
    stability_results = calculate_stability_metrics(all_data, categories)
    
    print("\nGenerating visualizations...")
    plot_correlation_heatmaps(all_data, categories, output_dir)
    plot_cv_heatmaps(stability_results, output_dir)
    plot_stability_summary(stability_results, output_dir)
    plot_top_k_consistency(stability_results, output_dir)
    plot_attribution_comparison(all_data, categories, output_dir)
    
    print("\nGenerating stability report...")
    generate_stability_report(stability_results, all_data, output_dir)
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"All results saved to: {output_dir}")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()

