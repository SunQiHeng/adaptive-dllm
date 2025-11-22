#!/usr/bin/env python3
"""
Analyze and visualize head attribution results
"""
import os
import sys
import numpy as np
import json
from pathlib import Path
import argparse


def load_attribution_results(results_dir: str):
    """Load all attribution results from directory"""
    results_dir = Path(results_dir)
    
    results = {}
    for npy_file in results_dir.glob('attribution_*.npy'):
        category = npy_file.stem.replace('attribution_', '')
        scores = np.load(npy_file)
        results[category] = scores
    
    return results


def analyze_category_attribution(category: str, scores: np.ndarray):
    """Analyze attribution for a single category"""
    n_layers, n_heads = scores.shape
    
    print(f"\n{'='*80}")
    print(f"Category: {category}")
    print(f"{'='*80}")
    print(f"Shape: {scores.shape} (layers x heads)")
    print(f"Overall mean: {scores.mean():.6f}")
    print(f"Overall std:  {scores.std():.6f}")
    print(f"Min score:    {scores.min():.6f}")
    print(f"Max score:    {scores.max():.6f}")
    
    # Per-layer statistics
    print(f"\nPer-layer statistics:")
    print(f"{'Layer':<8} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print(f"{'-'*56}")
    
    for layer_idx in range(n_layers):
        layer_scores = scores[layer_idx]
        print(f"{layer_idx:<8} {layer_scores.mean():<12.6f} {layer_scores.std():<12.6f} "
              f"{layer_scores.min():<12.6f} {layer_scores.max():<12.6f}")
    
    # Top heads per layer
    print(f"\nTop 5 heads per layer:")
    for layer_idx in range(min(5, n_layers)):  # Show first 5 layers
        layer_scores = scores[layer_idx]
        top_heads = np.argsort(layer_scores)[-5:][::-1]
        top_scores = layer_scores[top_heads]
        print(f"  Layer {layer_idx}: heads {top_heads.tolist()} "
              f"(scores: {[f'{s:.4f}' for s in top_scores]})")
    
    if n_layers > 5:
        print(f"  ... (showing first 5 of {n_layers} layers)")
    
    # Overall top heads
    flat_indices = np.argsort(scores.flatten())[-10:][::-1]
    top_layers = flat_indices // n_heads
    top_heads_overall = flat_indices % n_heads
    top_scores_overall = scores.flatten()[flat_indices]
    
    print(f"\nTop 10 heads overall:")
    for i, (layer, head, score) in enumerate(zip(top_layers, top_heads_overall, top_scores_overall)):
        print(f"  {i+1}. Layer {layer:2d}, Head {head:2d}: {score:.6f}")


def compare_categories(results: dict):
    """Compare attribution patterns across categories"""
    if len(results) < 2:
        return
    
    print(f"\n{'='*80}")
    print(f"Cross-Category Comparison")
    print(f"{'='*80}")
    
    categories = list(results.keys())
    
    # Average importance per category
    print(f"\nAverage head importance by category:")
    print(f"{'Category':<15} {'Mean':<12} {'Std':<12}")
    print(f"{'-'*39}")
    
    category_means = {}
    for category in sorted(categories):
        scores = results[category]
        mean_score = scores.mean()
        std_score = scores.std()
        category_means[category] = mean_score
        print(f"{category:<15} {mean_score:<12.6f} {std_score:<12.6f}")
    
    # Find category-specific important heads
    print(f"\nCategory-specific head importance:")
    
    for category in categories:
        scores = results[category]
        n_layers, n_heads = scores.shape
        
        # Get top heads for this category
        flat_indices = np.argsort(scores.flatten())[-5:][::-1]
        top_layers = flat_indices // n_heads
        top_heads = flat_indices % n_heads
        
        print(f"\n  {category}:")
        for layer, head in zip(top_layers, top_heads):
            print(f"    Layer {layer:2d}, Head {head:2d}", end="")
            
            # Compare with other categories
            this_score = scores[layer, head]
            other_scores = [results[other_cat][layer, head] 
                          for other_cat in categories if other_cat != category]
            
            if len(other_scores) > 0:
                avg_other = np.mean(other_scores)
                ratio = this_score / (avg_other + 1e-10)
                print(f"  (this: {this_score:.4f}, others: {avg_other:.4f}, ratio: {ratio:.2f}x)")
            else:
                print(f"  (score: {this_score:.4f})")


def export_for_adaptive_config(results: dict, output_file: str, 
                               top_k_percent: float = 0.5):
    """
    Export head importance for adaptive sparse attention configuration
    
    Args:
        results: Attribution results dict
        output_file: Output file path
        top_k_percent: Keep top k% heads with higher capacity
    """
    print(f"\n{'='*80}")
    print(f"Exporting Adaptive Configuration")
    print(f"{'='*80}")
    
    # Average across all categories
    all_scores = np.stack(list(results.values()))
    avg_scores = all_scores.mean(axis=0)  # (n_layers, n_heads)
    
    n_layers, n_heads = avg_scores.shape
    
    # Normalize scores to [0, 1] range per layer
    normalized_scores = np.zeros_like(avg_scores)
    for layer_idx in range(n_layers):
        layer_scores = avg_scores[layer_idx]
        min_score = layer_scores.min()
        max_score = layer_scores.max()
        if max_score > min_score:
            normalized_scores[layer_idx] = (layer_scores - min_score) / (max_score - min_score)
        else:
            normalized_scores[layer_idx] = 0.5
    
    # Create adaptive config
    config = {
        'importance_scores': normalized_scores.tolist(),
        'shape': [n_layers, n_heads],
        'source': 'nemotron_attribution',
        'categories': list(results.keys())
    }
    
    # Save config
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Saved adaptive config to: {output_file}")
    print(f"  Layers: {n_layers}")
    print(f"  Heads per layer: {n_heads}")
    print(f"  Score range: [{normalized_scores.min():.4f}, {normalized_scores.max():.4f}]")
    
    # Show per-layer statistics
    print(f"\nPer-layer importance statistics:")
    print(f"{'Layer':<8} {'Mean':<12} {'Std':<12}")
    print(f"{'-'*32}")
    for layer_idx in range(n_layers):
        layer_scores = normalized_scores[layer_idx]
        print(f"{layer_idx:<8} {layer_scores.mean():<12.4f} {layer_scores.std():<12.4f}")


def main():
    parser = argparse.ArgumentParser(description='Analyze head attribution results')
    parser.add_argument('results_dir', type=str, 
                       help='Directory containing attribution results')
    parser.add_argument('--export_config', type=str, default=None,
                       help='Export adaptive config to file')
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.results_dir}")
    results = load_attribution_results(args.results_dir)
    
    if len(results) == 0:
        print("No attribution results found!")
        return
    
    print(f"Found {len(results)} categories: {list(results.keys())}")
    
    # Analyze each category
    for category in sorted(results.keys()):
        analyze_category_attribution(category, results[category])
    
    # Compare categories
    compare_categories(results)
    
    # Export config if requested
    if args.export_config:
        export_for_adaptive_config(results, args.export_config)
    
    print(f"\n{'='*80}")
    print(f"Analysis complete!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

