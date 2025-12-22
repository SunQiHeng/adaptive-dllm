#!/usr/bin/env python3
"""
测试Dream adaptive模式下实际的attention keep ratio

运行此脚本查看在不同select值下，每层和全局的平均keep_ratio
"""

import sys
import os
sys.path.insert(0, '/home/qiheng/Projects/adaptive-dllm')

import torch
from models.Dream.sparse.adaptive_utils_dream import create_adaptive_sparsity_config

def test_dream_keep_ratios():
    """测试Dream模型在不同select值下的keep ratio"""
    
    print("=" * 80)
    print("Dream Adaptive Keep Ratio 测试")
    print("=" * 80)
    
    # Dream模型配置 (需要根据实际模型调整)
    # Dream-v0-Instruct-7B 使用的配置
    n_layers = 28
    n_heads = 4  # KV heads (Dream uses GQA: 32 query heads, 4 KV heads)
    
    # 加载预计算的importance scores
    importance_path = '/home/qiheng/Projects/adaptive-dllm/configs/head_importance_dream/head_importance.pt'
    
    if os.path.exists(importance_path):
        print(f"\n✓ 加载预计算的importance scores: {importance_path}")
        importance_data = torch.load(importance_path, weights_only=False)
        importance_scores = importance_data['importance_scores']
    else:
        print(f"\n⚠ 未找到预计算文件，使用随机importance scores")
        importance_scores = None
    
    # 创建adaptive配置
    print(f"\n创建adaptive配置...")
    print(f"  Layers: {n_layers}")
    print(f"  KV Heads: {n_heads}")
    print(f"  Output mode: 相对权重 (mean=1.0)")
    
    adaptive_config = create_adaptive_sparsity_config(
        n_layers=n_layers,
        n_heads=n_heads,
        importance_scores=importance_scores,
        min_sparsity=0.1,
        max_sparsity=0.9,
        normalize_strategy='global_percentile',
        output_relative_weights=True,
        seed=42
    )
    
    sparsity_levels = adaptive_config['sparsity_levels']
    
    # 打印相对权重统计
    print("\n" + "=" * 80)
    print("相对权重统计 (这些是归一化的重要性权重，mean≈1.0)")
    print("=" * 80)
    
    all_weights = []
    for layer_idx in range(n_layers):
        weights = sparsity_levels[layer_idx]
        all_weights.append(weights)
        print(f"Layer {layer_idx:2d}: mean={weights.mean():.4f}, "
              f"min={weights.min():.4f}, max={weights.max():.4f}, "
              f"range=[{weights.min():.3f}, {weights.max():.3f}]")
    
    # 全局统计
    all_weights_tensor = torch.cat(all_weights)
    print("\n" + "-" * 80)
    print(f"全局权重统计: mean={all_weights_tensor.mean():.4f}, "
          f"min={all_weights_tensor.min():.4f}, max={all_weights_tensor.max():.4f}")
    
    # 测试不同的select值
    select_values = [0.2, 0.3, 0.5, 0.8]
    
    print("\n" + "=" * 80)
    print("在不同select值下的实际keep_ratio（推理时）")
    print("=" * 80)
    
    for select in select_values:
        print(f"\n{'=' * 80}")
        print(f"SELECT = {select:.1f} (目标: 平均保留{select*100:.0f}%的attention块)")
        print(f"{'=' * 80}")
        
        layer_keep_ratios = []
        
        for layer_idx in range(n_layers):
            weights = sparsity_levels[layer_idx]
            
            # 模拟推理时的计算: keep_ratio = weight * select
            keep_ratios = torch.clamp(weights * select, 0.0, 1.0)
            
            mean_keep = keep_ratios.mean().item()
            min_keep = keep_ratios.min().item()
            max_keep = keep_ratios.max().item()
            
            layer_keep_ratios.append(keep_ratios)
            
            # 计算有多少heads触及上限
            clamped_heads = (weights * select > 1.0).sum().item()
            
            print(f"Layer {layer_idx:2d}: "
                  f"平均={mean_keep:.4f} ({mean_keep*100:.1f}%), "
                  f"范围=[{min_keep:.3f}, {max_keep:.3f}], "
                  f"触及上限: {clamped_heads}/{n_heads} heads")
        
        # 全局统计
        all_keep_ratios = torch.cat(layer_keep_ratios)
        global_mean = all_keep_ratios.mean().item()
        global_min = all_keep_ratios.min().item()
        global_max = all_keep_ratios.max().item()
        
        # 计算总的触及上限的heads数量
        total_clamped = sum((layer_keep_ratios[i] >= 1.0).sum().item() 
                           for i in range(n_layers))
        total_heads = n_layers * n_heads
        
        print(f"\n{'─' * 80}")
        print(f"📊 全局统计:")
        print(f"   平均keep_ratio: {global_mean:.4f} ({global_mean*100:.1f}%)")
        print(f"   目标select:     {select:.4f} ({select*100:.1f}%)")
        print(f"   偏差:          {abs(global_mean - select):.4f} ({abs(global_mean - select)*100:.1f}%)")
        print(f"   范围:          [{global_min:.3f}, {global_max:.3f}]")
        print(f"   触及上限:       {total_clamped}/{total_heads} heads ({total_clamped/total_heads*100:.1f}%)")
        
        # 分析
        if abs(global_mean - select) < 0.01:
            print(f"   ✅ 实际平均值与目标非常接近！")
        elif abs(global_mean - select) < 0.05:
            print(f"   ⚠️  实际平均值与目标有轻微偏差")
        else:
            print(f"   ❌ 实际平均值与目标偏差较大（可能由于大量heads触及上限）")

if __name__ == "__main__":
    test_dream_keep_ratios()

