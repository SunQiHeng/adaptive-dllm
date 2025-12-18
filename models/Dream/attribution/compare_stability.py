#!/usr/bin/env python3
"""
对比分析两组归因结果的稳定性
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
    
    return all_data

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
            continue
        
        # Stack data: shape (n_runs, n_layers, n_heads)
        stacked_data = np.stack(category_data, axis=0)
        
        # Calculate statistics across runs
        mean_attr = np.mean(stacked_data, axis=0)
        std_attr = np.std(stacked_data, axis=0)
        cv_attr = std_attr / (np.abs(mean_attr) + 1e-10)  # coefficient of variation
        
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

def analyze_experiment(base_dir, run_folders, exp_name):
    """分析单个实验的稳定性"""
    print(f"\n{'='*80}")
    print(f"Analyzing: {exp_name}")
    print(f"Path: {base_dir}")
    print(f"{'='*80}")
    
    categories = ['chat', 'code', 'math', 'safety', 'science']
    
    all_data = load_attribution_data(base_dir, run_folders)
    stability_results = calculate_stability_metrics(all_data, categories)
    
    # Calculate summary metrics
    summary = {
        'exp_name': exp_name,
        'base_dir': base_dir
    }
    
    for category in categories:
        if category not in stability_results:
            continue
        
        results = stability_results[category]
        summary[category] = {
            'pearson_mean': results['pearson_correlations']['mean'],
            'pearson_std': results['pearson_correlations']['std'],
            'spearman_mean': results['spearman_correlations']['mean'],
            'spearman_std': results['spearman_correlations']['std'],
            'cosine_mean': results['cosine_similarities']['mean'],
            'cosine_std': results['cosine_similarities']['std'],
            'cv_mean': np.mean(results['cv_attribution']),
            'cv_median': np.median(results['cv_attribution']),
            'top5_consistency': results['top_k_consistency'][5]['mean'],
            'top10_consistency': results['top_k_consistency'][10]['mean'],
            'top20_consistency': results['top_k_consistency'][20]['mean'],
        }
    
    # Overall averages
    summary['overall'] = {
        'pearson_mean': np.mean([summary[cat]['pearson_mean'] for cat in categories if cat in summary]),
        'spearman_mean': np.mean([summary[cat]['spearman_mean'] for cat in categories if cat in summary]),
        'cosine_mean': np.mean([summary[cat]['cosine_mean'] for cat in categories if cat in summary]),
        'cv_mean': np.mean([summary[cat]['cv_mean'] for cat in categories if cat in summary]),
        'top20_consistency': np.mean([summary[cat]['top20_consistency'] for cat in categories if cat in summary]),
    }
    
    return summary, stability_results

def print_comparison_report(summary1, summary2):
    """打印对比报告"""
    exp1_name = summary1['exp_name']
    exp2_name = summary2['exp_name']
    
    print(f"\n{'='*100}")
    print(f"STABILITY COMPARISON REPORT")
    print(f"{'='*100}")
    print(f"\nExperiment 1: {exp1_name}")
    print(f"Experiment 2: {exp2_name}")
    print(f"\n{'='*100}")
    
    categories = ['chat', 'code', 'math', 'safety', 'science']
    
    # Create comparison table
    print(f"\n{'Category':<12} {'Metric':<20} {exp1_name:<25} {exp2_name:<25} {'Improvement':<15}")
    print(f"{'-'*100}")
    
    for category in categories:
        if category not in summary1 or category not in summary2:
            continue
        
        cat_display = category.capitalize()
        
        # Pearson correlation
        pearson1 = summary1[category]['pearson_mean']
        pearson2 = summary2[category]['pearson_mean']
        improvement = ((pearson2 - pearson1) / abs(pearson1)) * 100 if pearson1 != 0 else 0
        print(f"{cat_display:<12} {'Pearson Corr':<20} {pearson1:>7.4f} ± {summary1[category]['pearson_std']:>5.4f}   {pearson2:>7.4f} ± {summary2[category]['pearson_std']:>5.4f}   {improvement:>+7.1f}%")
        
        # Spearman correlation
        spearman1 = summary1[category]['spearman_mean']
        spearman2 = summary2[category]['spearman_mean']
        improvement = ((spearman2 - spearman1) / abs(spearman1)) * 100 if spearman1 != 0 else 0
        print(f"{'':12} {'Spearman Corr':<20} {spearman1:>7.4f} ± {summary1[category]['spearman_std']:>5.4f}   {spearman2:>7.4f} ± {summary2[category]['spearman_std']:>5.4f}   {improvement:>+7.1f}%")
        
        # Top-20 consistency
        top20_1 = summary1[category]['top20_consistency']
        top20_2 = summary2[category]['top20_consistency']
        improvement = ((top20_2 - top20_1) / abs(top20_1)) * 100 if top20_1 != 0 else 0
        print(f"{'':12} {'Top-20 Jaccard':<20} {top20_1:>7.4f}             {top20_2:>7.4f}             {improvement:>+7.1f}%")
        
        # CV
        cv1 = summary1[category]['cv_mean']
        cv2 = summary2[category]['cv_mean']
        improvement = ((cv1 - cv2) / abs(cv1)) * 100 if cv1 != 0 else 0  # Lower is better
        print(f"{'':12} {'Mean CV':<20} {cv1:>7.4f}             {cv2:>7.4f}             {improvement:>+7.1f}%")
        
        print(f"{'-'*100}")
    
    # Overall comparison
    print(f"\n{'OVERALL':<12} {'Metric':<20} {exp1_name:<25} {exp2_name:<25} {'Improvement':<15}")
    print(f"{'-'*100}")
    
    overall1 = summary1['overall']
    overall2 = summary2['overall']
    
    pearson_imp = ((overall2['pearson_mean'] - overall1['pearson_mean']) / abs(overall1['pearson_mean'])) * 100
    print(f"{'':12} {'Avg Pearson':<20} {overall1['pearson_mean']:>7.4f}             {overall2['pearson_mean']:>7.4f}             {pearson_imp:>+7.1f}%")
    
    spearman_imp = ((overall2['spearman_mean'] - overall1['spearman_mean']) / abs(overall1['spearman_mean'])) * 100
    print(f"{'':12} {'Avg Spearman':<20} {overall1['spearman_mean']:>7.4f}             {overall2['spearman_mean']:>7.4f}             {spearman_imp:>+7.1f}%")
    
    top20_imp = ((overall2['top20_consistency'] - overall1['top20_consistency']) / abs(overall1['top20_consistency'])) * 100
    print(f"{'':12} {'Avg Top-20':<20} {overall1['top20_consistency']:>7.4f}             {overall2['top20_consistency']:>7.4f}             {top20_imp:>+7.1f}%")
    
    cv_imp = ((overall1['cv_mean'] - overall2['cv_mean']) / abs(overall1['cv_mean'])) * 100  # Lower is better
    print(f"{'':12} {'Avg CV':<20} {overall1['cv_mean']:>7.4f}             {overall2['cv_mean']:>7.4f}             {cv_imp:>+7.1f}%")
    
    print(f"\n{'='*100}")
    print("CONCLUSION:")
    print(f"{'='*100}")
    
    # Determine which is better
    better_count = 0
    total_metrics = 4
    
    if overall2['pearson_mean'] > overall1['pearson_mean']:
        better_count += 1
        print(f"✓ Pearson correlation: {exp2_name} is BETTER ({pearson_imp:+.1f}%)")
    else:
        print(f"✗ Pearson correlation: {exp1_name} is better ({-pearson_imp:+.1f}%)")
    
    if overall2['spearman_mean'] > overall1['spearman_mean']:
        better_count += 1
        print(f"✓ Spearman correlation: {exp2_name} is BETTER ({spearman_imp:+.1f}%)")
    else:
        print(f"✗ Spearman correlation: {exp1_name} is better ({-spearman_imp:+.1f}%)")
    
    if overall2['top20_consistency'] > overall1['top20_consistency']:
        better_count += 1
        print(f"✓ Top-20 consistency: {exp2_name} is BETTER ({top20_imp:+.1f}%)")
    else:
        print(f"✗ Top-20 consistency: {exp1_name} is better ({-top20_imp:+.1f}%)")
    
    if overall2['cv_mean'] < overall1['cv_mean']:
        better_count += 1
        print(f"✓ Coefficient of Variation: {exp2_name} is BETTER ({cv_imp:+.1f}% reduction)")
    else:
        print(f"✗ Coefficient of Variation: {exp1_name} is better ({-cv_imp:+.1f}% reduction)")
    
    print(f"\n{'-'*100}")
    if better_count >= 3:
        print(f"🏆 WINNER: {exp2_name} is SIGNIFICANTLY BETTER ({better_count}/{total_metrics} metrics improved)")
    elif better_count == 2:
        print(f"📊 RESULT: {exp2_name} is SLIGHTLY BETTER ({better_count}/{total_metrics} metrics improved)")
    elif better_count == 1:
        print(f"📊 RESULT: {exp1_name} is SLIGHTLY BETTER ({total_metrics-better_count}/{total_metrics} metrics)")
    else:
        print(f"🏆 WINNER: {exp1_name} is SIGNIFICANTLY BETTER ({total_metrics-better_count}/{total_metrics} metrics)")
    print(f"{'='*100}\n")

def plot_comparison(summary1, summary2, output_dir):
    """绘制对比图"""
    categories = ['chat', 'code', 'math', 'safety', 'science']
    exp1_name = summary1['exp_name']
    exp2_name = summary2['exp_name']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Pearson correlations
    ax = axes[0, 0]
    x = np.arange(len(categories))
    width = 0.35
    pearson1 = [summary1[cat]['pearson_mean'] for cat in categories if cat in summary1]
    pearson2 = [summary2[cat]['pearson_mean'] for cat in categories if cat in summary2]
    
    ax.bar(x - width/2, pearson1, width, label=exp1_name, alpha=0.8)
    ax.bar(x + width/2, pearson2, width, label=exp2_name, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([cat.capitalize() for cat in categories], rotation=45, ha='right')
    ax.set_ylabel('Pearson Correlation', fontsize=12)
    ax.set_title('Pearson Correlation Comparison', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='0.9 threshold')
    
    # Spearman correlations
    ax = axes[0, 1]
    spearman1 = [summary1[cat]['spearman_mean'] for cat in categories if cat in summary1]
    spearman2 = [summary2[cat]['spearman_mean'] for cat in categories if cat in summary2]
    
    ax.bar(x - width/2, spearman1, width, label=exp1_name, alpha=0.8)
    ax.bar(x + width/2, spearman2, width, label=exp2_name, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([cat.capitalize() for cat in categories], rotation=45, ha='right')
    ax.set_ylabel('Spearman Correlation', fontsize=12)
    ax.set_title('Spearman Correlation Comparison', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='0.9 threshold')
    
    # Top-20 consistency
    ax = axes[1, 0]
    top20_1 = [summary1[cat]['top20_consistency'] for cat in categories if cat in summary1]
    top20_2 = [summary2[cat]['top20_consistency'] for cat in categories if cat in summary2]
    
    ax.bar(x - width/2, top20_1, width, label=exp1_name, alpha=0.8)
    ax.bar(x + width/2, top20_2, width, label=exp2_name, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([cat.capitalize() for cat in categories], rotation=45, ha='right')
    ax.set_ylabel('Top-20 Jaccard Similarity', fontsize=12)
    ax.set_title('Top-20 Head Consistency Comparison', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Coefficient of Variation
    ax = axes[1, 1]
    cv1 = [summary1[cat]['cv_mean'] for cat in categories if cat in summary1]
    cv2 = [summary2[cat]['cv_mean'] for cat in categories if cat in summary2]
    
    ax.bar(x - width/2, cv1, width, label=exp1_name, alpha=0.8)
    ax.bar(x + width/2, cv2, width, label=exp2_name, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([cat.capitalize() for cat in categories], rotation=45, ha='right')
    ax.set_ylabel('Mean CV', fontsize=12)
    ax.set_title('Coefficient of Variation Comparison (Lower is Better)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Stability Comparison Between Experiments', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'experiment_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {output_dir / 'experiment_comparison.png'}")

def main():
    # Configuration for both experiments
    exp1_dir = '/home/qiheng/Projects/adaptive-dllm/models/Dream/attribution/attribution_results_20251210_231629'
    exp2_dir = '/home/qiheng/Projects/adaptive-dllm/models/Dream/attribution/attribution_results_20251211_155653'
    run_folders = ['run_1_seed_42', 'run_2_seed_123', 'run_3_seed_2024']
    
    # Analyze both experiments
    summary1, results1 = analyze_experiment(exp1_dir, run_folders, "Exp1_20251210")
    summary2, results2 = analyze_experiment(exp2_dir, run_folders, "Exp2_20251211")
    
    # Print comparison report
    print_comparison_report(summary1, summary2)
    
    # Create comparison visualizations
    output_dir = Path(exp2_dir) / 'comparison_with_previous'
    output_dir.mkdir(exist_ok=True)
    
    plot_comparison(summary1, summary2, output_dir)
    
    print(f"\n{'='*100}")
    print(f"Comparison complete! Results saved to: {output_dir}")
    print(f"{'='*100}")

if __name__ == '__main__':
    main()


