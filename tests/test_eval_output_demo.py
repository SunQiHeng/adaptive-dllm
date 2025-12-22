#!/usr/bin/env python3
"""
Demo script to show the adaptive config statistics output
不实际运行evaluation，只展示adaptive配置的统计信息
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.LLaDA.sparse.adaptive_utils import create_adaptive_sparsity_config
from models.Dream.sparse.adaptive_utils_dream import create_adaptive_sparsity_config as create_adaptive_sparsity_config_dream

# Import the print function from eval scripts
# We'll just copy it here for demonstration
def print_adaptive_config_stats(adaptive_config, select, n_layers, n_heads, model_name="Model"):
    """
    Print statistics about adaptive sparsity configuration.
    """
    print("\n" + "=" * 80)
    print(f"📊 {model_name} Adaptive Sparsity Configuration Statistics")
    print("=" * 80)
    
    sparsity_levels = adaptive_config['sparsity_levels']
    metadata = adaptive_config.get('metadata', {})
    
    # Check if we're using relative weights or absolute keep_ratios
    output_relative_weights = metadata.get('output_relative_weights', True)
    
    if output_relative_weights:
        print(f"Mode: Relative Weights (mean=1.0, multiply by select at inference)")
    else:
        print(f"Mode: Absolute Keep Ratios (pre-computed)")
    
    print(f"Target select: {select:.3f} ({select*100:.1f}%)")
    print(f"Layers: {n_layers}, KV Heads per layer: {n_heads}")
    print(f"Total heads: {n_layers * n_heads}")
    
    # Collect all weights/keep_ratios
    all_values = []
    layer_means = []
    
    print("\n" + "-" * 80)
    print("Per-Layer Statistics (showing first 5 and last 5 layers):")
    print("-" * 80)
    
    for layer_idx in range(n_layers):
        values = sparsity_levels[layer_idx]
        layer_mean = values.mean().item()
        layer_min = values.min().item()
        layer_max = values.max().item()
        
        all_values.append(values)
        layer_means.append(layer_mean)
        
        # Only print first 5 and last 5 layers for brevity
        if layer_idx < 5 or layer_idx >= n_layers - 5:
            # Calculate actual keep_ratio if using relative weights
            if output_relative_weights:
                actual_mean = layer_mean * select
                actual_min = layer_min * select
                actual_max = layer_max * select
                # Clamp to [0, 1]
                actual_mean = min(actual_mean, 1.0)
                actual_max = min(actual_max, 1.0)
                
                print(f"Layer {layer_idx:2d}: weight_mean={layer_mean:.4f} "
                      f"→ keep_ratio={actual_mean:.4f} ({actual_mean*100:.1f}%), "
                      f"range=[{actual_min:.3f}, {actual_max:.3f}]")
            else:
                print(f"Layer {layer_idx:2d}: keep_ratio_mean={layer_mean:.4f} ({layer_mean*100:.1f}%), "
                      f"range=[{layer_min:.3f}, {layer_max:.3f}]")
        elif layer_idx == 5:
            print("  ...")
    
    # Global statistics
    all_values_tensor = torch.cat(all_values)
    global_mean = all_values_tensor.mean().item()
    global_min = all_values_tensor.min().item()
    global_max = all_values_tensor.max().item()
    global_std = all_values_tensor.std().item()
    
    print("\n" + "-" * 80)
    print("Global Statistics:")
    print("-" * 80)
    
    if output_relative_weights:
        actual_global_mean = global_mean * select
        actual_global_min = global_min * select
        actual_global_max = global_max * select
        # Clamp
        actual_global_mean = min(actual_global_mean, 1.0)
        actual_global_max = min(actual_global_max, 1.0)
        
        print(f"Relative weights:")
        print(f"  Mean:   {global_mean:.4f} (should be ≈1.0)")
        print(f"  Std:    {global_std:.4f}")
        print(f"  Range:  [{global_min:.3f}, {global_max:.3f}]")
        print(f"\nActual keep_ratios (weights × select={select}):")
        print(f"  Mean:   {actual_global_mean:.4f} ({actual_global_mean*100:.1f}%)")
        print(f"  Target: {select:.4f} ({select*100:.1f}%)")
        print(f"  Deviation: {abs(actual_global_mean - select):.4f} ({abs(actual_global_mean - select)*100:.2f}%)")
        print(f"  Range:  [{actual_global_min:.3f}, {actual_global_max:.3f}]")
        
        # Count heads that will hit upper limit (keep_ratio > 1.0 after scaling)
        clamped_count = (all_values_tensor * select > 1.0).sum().item()
        total_heads = n_layers * n_heads
        print(f"  Heads hitting upper limit (>1.0): {clamped_count}/{total_heads} ({clamped_count/total_heads*100:.1f}%)")
        
        if abs(actual_global_mean - select) < 0.01:
            print(f"  ✅ Mean matches target (deviation < 1%)")
        elif abs(actual_global_mean - select) < 0.05:
            print(f"  ⚠️  Mean has slight deviation (1-5%)")
        else:
            print(f"  ❌ Mean deviates significantly (>5%)")
    else:
        print(f"Keep ratios:")
        print(f"  Mean:   {global_mean:.4f} ({global_mean*100:.1f}%)")
        print(f"  Std:    {global_std:.4f}")
        print(f"  Range:  [{global_min:.3f}, {global_max:.3f}]")
    
    # Layer-wise variation
    layer_means_tensor = torch.tensor(layer_means)
    layer_mean_std = layer_means_tensor.std().item()
    layer_mean_min = layer_means_tensor.min().item()
    layer_mean_max = layer_means_tensor.max().item()
    
    print("\n" + "-" * 80)
    print("Layer-wise Variation:")
    print("-" * 80)
    
    if output_relative_weights:
        actual_layer_min = layer_mean_min * select
        actual_layer_max = layer_mean_max * select
        print(f"Layer means range: [{layer_mean_min:.4f}, {layer_mean_max:.4f}]")
        print(f"Layer means std:   {layer_mean_std:.4f}")
        print(f"Actual keep_ratio range across layers: [{actual_layer_min:.3f}, {actual_layer_max:.3f}]")
        print(f"Variation: {(actual_layer_max - actual_layer_min)*100:.1f}% spread")
    else:
        print(f"Layer means range: [{layer_mean_min:.4f}, {layer_mean_max:.4f}]")
        print(f"Layer means std:   {layer_mean_std:.4f}")
        print(f"Variation: {(layer_mean_max - layer_mean_min)*100:.1f}% spread")
    
    print("=" * 80 + "\n")


def test_llada():
    """Test LLaDA adaptive config output"""
    print("\n" + "🔵" * 40)
    print("Testing LLaDA Adaptive Config Output")
    print("🔵" * 40)
    
    # Load LLaDA pre-computed importance
    llada_importance_path = '/home/qiheng/Projects/adaptive-dllm/configs/head_importance_llada_base/head_importance.pt'
    
    if not os.path.exists(llada_importance_path):
        print(f"❌ File not found: {llada_importance_path}")
        print("Using random importance instead...")
        importance_scores = None
        strategy = 'normal'
    else:
        print(f"✓ Loading pre-computed importance from: {llada_importance_path}")
        importance_data = torch.load(llada_importance_path, weights_only=False)
        importance_scores = importance_data['importance_scores']
        strategy = None
    
    # LLaDA-8B config
    n_layers = 32
    n_heads = 8  # KV heads
    select = 0.3
    
    print(f"\nCreating adaptive config for LLaDA-8B...")
    print(f"  Layers: {n_layers}")
    print(f"  KV Heads: {n_heads}")
    print(f"  Select: {select}")
    
    adaptive_config = create_adaptive_sparsity_config(
        n_layers=n_layers,
        n_heads=n_heads,
        importance_scores=importance_scores,
        strategy=strategy,
        min_sparsity=0.15,
        max_sparsity=0.85,
        normalize_strategy='global_percentile',
        output_relative_weights=True,
        seed=42
    )
    
    print_adaptive_config_stats(adaptive_config, select, n_layers, n_heads, "LLaDA-8B")


def test_dream():
    """Test Dream adaptive config output"""
    print("\n" + "🟣" * 40)
    print("Testing Dream Adaptive Config Output")
    print("🟣" * 40)
    
    # Load Dream pre-computed importance
    dream_importance_path = '/home/qiheng/Projects/adaptive-dllm/configs/head_importance_dream/head_importance.pt'
    
    if not os.path.exists(dream_importance_path):
        print(f"❌ File not found: {dream_importance_path}")
        print("Using random importance instead...")
        importance_scores = None
        strategy = 'normal'
    else:
        print(f"✓ Loading pre-computed importance from: {dream_importance_path}")
        importance_data = torch.load(dream_importance_path, weights_only=False)
        importance_scores = importance_data['importance_scores']
        strategy = None
    
    # Dream-7B config
    n_layers = 28
    n_heads = 4  # KV heads
    select = 0.3
    
    print(f"\nCreating adaptive config for Dream-7B...")
    print(f"  Layers: {n_layers}")
    print(f"  KV Heads: {n_heads}")
    print(f"  Select: {select}")
    
    adaptive_config = create_adaptive_sparsity_config_dream(
        n_layers=n_layers,
        n_heads=n_heads,
        importance_scores=importance_scores,
        strategy=strategy,
        min_sparsity=0.1,
        max_sparsity=0.9,
        normalize_strategy='global_percentile',
        output_relative_weights=True,
        seed=42
    )
    
    print_adaptive_config_stats(adaptive_config, select, n_layers, n_heads, "Dream-7B")


if __name__ == '__main__':
    test_llada()
    test_dream()
    
    print("\n" + "✅" * 40)
    print("Demo completed! This output will appear when running evaluation scripts.")
    print("✅" * 40 + "\n")

