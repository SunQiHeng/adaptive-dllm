#!/usr/bin/env python3
"""
Generate average head importance from all runs and categories.
"""
import os
import numpy as np
import torch
import json
from pathlib import Path


def load_all_attributions(results_dir: str):
    """
    Load all attribution results from all runs and categories.
    
    Returns:
        Dict[run_name][category] = attribution_array
    """
    results = {}
    run_dirs = [d for d in os.listdir(results_dir) if d.startswith('run_')]
    run_dirs = sorted(run_dirs)
    
    print(f"Found {len(run_dirs)} runs: {run_dirs}")
    
    for run_dir in run_dirs:
        run_path = os.path.join(results_dir, run_dir)
        results[run_dir] = {}
        npy_files = [f for f in os.listdir(run_path) if f.endswith('.npy')]
        
        for npy_file in npy_files:
            category = npy_file.replace('attribution_', '').replace('.npy', '')
            file_path = os.path.join(run_path, npy_file)
            results[run_dir][category] = np.load(file_path)
            print(f"  Loaded {run_dir}/{category}: shape {results[run_dir][category].shape}")
    
    return results


def compute_global_average(results):
    """
    Compute average across all runs and all categories.
    
    Returns:
        average_attribution: (n_layers, n_heads) array
        metadata: dict with statistics
    """
    all_attributions = []
    
    for run_name, categories in results.items():
        for category, attribution in categories.items():
            all_attributions.append(attribution)
    
    # Stack and compute mean
    stacked = np.stack(all_attributions, axis=0)  # (n_runs * n_categories, n_layers, n_heads)
    averaged = stacked.mean(axis=0)  # (n_layers, n_heads)
    
    print(f"\n{'='*80}")
    print(f"Global Average Attribution Statistics")
    print(f"{'='*80}")
    print(f"Total samples averaged: {len(all_attributions)} (runs={len(results)}, categories={len(results[list(results.keys())[0]])})")
    print(f"Shape: {averaged.shape}")
    print(f"Mean: {averaged.mean():.6f}")
    print(f"Std: {averaged.std():.6f}")
    print(f"Min: {averaged.min():.6f}")
    print(f"Max: {averaged.max():.6f}")
    
    # Convert to absolute values for importance scores
    importance = np.abs(averaged)
    
    print(f"\nAbsolute Attribution (Importance) Statistics:")
    print(f"Mean: {importance.mean():.6f}")
    print(f"Std: {importance.std():.6f}")
    print(f"Min: {importance.min():.6f}")
    print(f"Max: {importance.max():.6f}")
    
    metadata = {
        'n_samples': len(all_attributions),
        'n_runs': len(results),
        'n_categories': len(results[list(results.keys())[0]]),
        'shape': list(averaged.shape),
        'mean_attribution': float(averaged.mean()),
        'std_attribution': float(averaged.std()),
        'mean_importance': float(importance.mean()),
        'std_importance': float(importance.std())
    }
    
    return averaged, importance, metadata


def save_importance_config(importance, metadata, output_dir):
    """
    Save importance scores in multiple formats for different use cases.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save raw attribution (with signs)
    np.save(os.path.join(output_dir, 'average_attribution.npy'), importance)
    print(f"\nSaved raw attribution to: {output_dir}/average_attribution.npy")
    
    # 2. Save absolute importance scores (no signs)
    importance_abs = np.abs(importance)
    np.save(os.path.join(output_dir, 'average_importance.npy'), importance_abs)
    print(f"Saved importance scores to: {output_dir}/average_importance.npy")
    
    # 3. Save as PyTorch format (for adaptive_utils.py)
    n_layers, n_heads = importance.shape
    importance_dict = {layer_idx: torch.tensor(importance_abs[layer_idx]) 
                      for layer_idx in range(n_layers)}
    
    torch_save_path = os.path.join(output_dir, 'head_importance.pt')
    torch.save({
        'importance_scores': importance_dict,
        'metadata': metadata
    }, torch_save_path)
    print(f"Saved PyTorch format to: {torch_save_path}")
    
    # 4. Save normalized importance (for easy use)
    # Normalize per layer to [0, 1]
    normalized_importance = np.zeros_like(importance_abs)
    for layer_idx in range(n_layers):
        layer_scores = importance_abs[layer_idx]
        min_score = layer_scores.min()
        max_score = layer_scores.max()
        if max_score > min_score:
            normalized_importance[layer_idx] = (layer_scores - min_score) / (max_score - min_score)
        else:
            normalized_importance[layer_idx] = 0.5  # All equal
    
    np.save(os.path.join(output_dir, 'normalized_importance.npy'), normalized_importance)
    print(f"Saved normalized importance to: {output_dir}/normalized_importance.npy")
    
    # 5. Save metadata as JSON
    metadata_path = os.path.join(output_dir, 'importance_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_path}")
    
    # 6. Generate different sparsity configs (20%-80%, step 10%)
    print(f"\n{'='*80}")
    print(f"Generating Sparsity Configurations")
    print(f"{'='*80}")
    
    sparsity_configs_dir = os.path.join(output_dir, 'sparsity_configs')
    os.makedirs(sparsity_configs_dir, exist_ok=True)
    
    for keep_ratio in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        # Calculate threshold to keep top keep_ratio heads
        flat_importance = importance_abs.flatten()
        sorted_importance = np.sort(flat_importance)[::-1]
        threshold_idx = int(len(sorted_importance) * keep_ratio)
        threshold = sorted_importance[threshold_idx]
        
        # Create mask
        keep_mask = importance_abs >= threshold
        
        config = {
            'keep_ratio': keep_ratio,
            'threshold': float(threshold),
            'n_heads_kept': int(keep_mask.sum()),
            'n_heads_total': int(keep_mask.size),
            'keep_mask': keep_mask.tolist(),
            'importance_scores': importance_abs.tolist()
        }
        
        config_path = os.path.join(sparsity_configs_dir, f'config_keep_{int(keep_ratio*100)}.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"  Keep {int(keep_ratio*100):2d}%: threshold={threshold:.6f}, n_heads={keep_mask.sum()}/{keep_mask.size}")
    
    print(f"\nAll sparsity configs saved to: {sparsity_configs_dir}/")
    
    return importance_dict


def visualize_importance(importance, output_dir):
    """Create visualization of importance scores."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Set non-interactive backend for servers
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("\nWarning: matplotlib not available, skipping visualization")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Heatmap
    ax = axes[0]
    im = ax.imshow(importance, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_xlabel('Head Index', fontsize=12)
    ax.set_ylabel('Layer Index', fontsize=12)
    ax.set_title('Average Head Importance (All Runs & Categories)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Importance Score')
    
    # Plot 2: Distribution
    ax = axes[1]
    ax.hist(importance.flatten(), bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Importance Scores', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Save figure (avoid tight_layout which can cause RecursionError in some environments)
    output_file = os.path.join(output_dir, 'importance_visualization.png')
    try:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nSaved visualization to: {output_file}")
    except RecursionError:
        print("\nWarning: RecursionError during visualization, skipping plot generation")
    finally:
        plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate average head importance')
    parser.add_argument('--results_dir', type=str,
                       default='/home/qiheng/Projects/adaptive-dllm/attribution_results_20251124_005811',
                       help='Directory containing attribution results')
    parser.add_argument('--output_dir', type=str,
                       default='/home/qiheng/Projects/adaptive-dllm/configs/head_importance',
                       help='Output directory for importance configs')
    args = parser.parse_args()
    
    print("="*80)
    print("Generate Average Head Importance from Attribution Results")
    print("="*80)
    print(f"Input: {args.results_dir}")
    print(f"Output: {args.output_dir}")
    print("="*80)
    
    # Load all attributions
    print("\nLoading attribution results...")
    results = load_all_attributions(args.results_dir)
    
    # Compute global average
    print("\nComputing global average...")
    averaged, importance, metadata = compute_global_average(results)
    
    # Save in multiple formats
    print("\nSaving importance configurations...")
    importance_dict = save_importance_config(importance, metadata, args.output_dir)
    
    # Visualize
    print("\nGenerating visualization...")
    visualize_importance(np.abs(importance), args.output_dir)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nGenerated files in: {args.output_dir}/")
    print("  - average_attribution.npy: Raw attribution scores (with signs)")
    print("  - average_importance.npy: Absolute importance scores")
    print("  - normalized_importance.npy: Layer-normalized importance [0, 1]")
    print("  - head_importance.pt: PyTorch format for adaptive_utils")
    print("  - importance_metadata.json: Statistics and metadata")
    print("  - sparsity_configs/: Pre-configured sparsity levels (20%-80%)")
    print("  - importance_visualization.png: Heatmap and distribution")
    print("="*80)


if __name__ == '__main__':
    main()

