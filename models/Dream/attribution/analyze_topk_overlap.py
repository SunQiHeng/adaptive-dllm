#!/usr/bin/env python3
"""
详细分析Top-K heads的具体重叠情况
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_topk_overlap_detailed(base_dir, run_folders, categories, k_values=[5, 10, 20, 50]):
    """详细分析Top-K heads的重叠情况"""
    
    for category in categories:
        print(f"\n{'='*80}")
        print(f"Category: {category.upper()}")
        print(f"{'='*80}")
        
        # Load data from all runs
        run_data = []
        for run_name in run_folders:
            file_path = Path(base_dir) / run_name / f'attribution_{category}_avg.npy'
            if file_path.exists():
                data = np.load(file_path)
                run_data.append(data)
        
        if len(run_data) < 2:
            continue
        
        n_layers, n_heads = run_data[0].shape
        total_heads = n_layers * n_heads
        print(f"\nModel structure: {n_layers} layers × {n_heads} heads = {total_heads} total heads")
        
        # Analyze each K value
        for k in k_values:
            print(f"\n{'-'*80}")
            print(f"Top-{k} Analysis (top {k/total_heads*100:.2f}% of all heads)")
            print(f"{'-'*80}")
            
            # Get top-k heads for each run
            top_k_indices = []
            top_k_coords = []
            
            for i, data in enumerate(run_data):
                flat_data = data.flatten()
                flat_indices = np.argsort(flat_data)[-k:]
                top_k_indices.append(set(flat_indices))
                
                # Convert to (layer, head) coordinates
                coords = [(idx // n_heads, idx % n_heads) for idx in flat_indices]
                top_k_coords.append(coords)
                
                print(f"\nRun {i+1} ({run_folders[i]}):")
                print(f"  Top-{k} head indices (flat): {sorted(flat_indices)[:10]}... (showing first 10)")
                
                # Show top-5 heads with their attribution values
                top_5_flat = np.argsort(flat_data)[-5:][::-1]
                print(f"  Top-5 heads with values:")
                for rank, idx in enumerate(top_5_flat, 1):
                    layer = idx // n_heads
                    head = idx % n_heads
                    value = flat_data[idx]
                    print(f"    #{rank}: Layer {layer}, Head {head} = {value:.6f}")
            
            # Calculate pairwise overlaps
            print(f"\nPairwise overlaps:")
            for i in range(len(run_data)):
                for j in range(i+1, len(run_data)):
                    intersection = top_k_indices[i] & top_k_indices[j]
                    union = top_k_indices[i] | top_k_indices[j]
                    jaccard = len(intersection) / len(union)
                    
                    print(f"  Run {i+1} vs Run {j+1}:")
                    print(f"    Intersection: {len(intersection)}/{k} heads ({len(intersection)/k*100:.1f}%)")
                    print(f"    Union: {len(union)} heads")
                    print(f"    Jaccard similarity: {jaccard:.4f}")
            
            # Find common heads across all runs
            common_heads = top_k_indices[0].intersection(*top_k_indices[1:])
            print(f"\n  Common heads across ALL {len(run_data)} runs: {len(common_heads)}/{k} ({len(common_heads)/k*100:.1f}%)")
            
            if len(common_heads) > 0:
                print(f"  Common head indices: {sorted(common_heads)}")
                print(f"  Common head coordinates (layer, head):")
                for idx in sorted(common_heads):
                    layer = idx // n_heads
                    head = idx % n_heads
                    values = [data.flatten()[idx] for data in run_data]
                    print(f"    Layer {layer:2d}, Head {head:2d}: values = {[f'{v:.6f}' for v in values]}")
            
            # Find heads that appear in at least 2 runs
            all_heads_count = {}
            for indices in top_k_indices:
                for idx in indices:
                    all_heads_count[idx] = all_heads_count.get(idx, 0) + 1
            
            heads_in_at_least_2 = [idx for idx, count in all_heads_count.items() if count >= 2]
            print(f"\n  Heads appearing in at least 2 runs: {len(heads_in_at_least_2)}")
            
            # Visualize overlap for Top-20
            if k == 20:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Create a matrix showing which heads appear in which runs
                matrix = np.zeros((len(heads_in_at_least_2), len(run_data)))
                head_labels = []
                
                for i, idx in enumerate(sorted(heads_in_at_least_2)):
                    layer = idx // n_heads
                    head = idx % n_heads
                    head_labels.append(f"L{layer}H{head}")
                    
                    for j, indices in enumerate(top_k_indices):
                        if idx in indices:
                            matrix[i, j] = 1
                
                if len(heads_in_at_least_2) > 0:
                    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
                    ax.set_xticks(range(len(run_data)))
                    ax.set_xticklabels([f'Run {i+1}' for i in range(len(run_data))])
                    ax.set_yticks(range(len(heads_in_at_least_2)))
                    ax.set_yticklabels(head_labels, fontsize=8)
                    ax.set_xlabel('Run', fontsize=12)
                    ax.set_ylabel('Head (Layer-Head)', fontsize=12)
                    ax.set_title(f'{category.capitalize()} - Top-20 Heads Across Runs\n(Heads appearing in ≥2 runs)', 
                               fontsize=14, fontweight='bold')
                    plt.colorbar(im, ax=ax, label='Present in Top-20')
                    plt.tight_layout()
                    
                    output_dir = Path(base_dir) / 'stability_analysis'
                    plt.savefig(output_dir / f'top20_overlap_{category}.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"\n  Saved visualization to top20_overlap_{category}.png")

def main():
    base_dir = '/home/qiheng/Projects/adaptive-dllm/models/Dream/attribution/attribution_results_20251210_231629'
    run_folders = ['run_1_seed_42', 'run_2_seed_123', 'run_3_seed_2024']
    categories = ['chat', 'code', 'math', 'safety', 'science']
    
    analyze_topk_overlap_detailed(base_dir, run_folders, categories)
    
    print(f"\n{'='*80}")
    print("Detailed Top-K overlap analysis complete!")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()

