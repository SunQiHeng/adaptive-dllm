#!/usr/bin/env python3
"""
分析两组不同种子的归因结果的一致性
"""

import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# 加载两组结果
file1 = "/home/qiheng/Projects/adaptive-dllm/configs/head_importance_dream_base_loss_gateIG_zero_maskp0.15-0.3-0.5-0.7-0.9_mcs2_mean_masked_seed47_n50_k8_L2048_dseed47_mseed47_ts20251227_191418/head_importance.pt"
file2 = "/home/qiheng/Projects/adaptive-dllm/configs/head_importance_dream_base_loss_gateIG_zero_maskp0.15-0.3-0.5-0.7-0.9_mcs2_mean_masked_seed1234_n50_k8_L2048_dseed1234_mseed1234_ts20251227_191323/head_importance.pt"

data1 = torch.load(file1, map_location='cpu')
data2 = torch.load(file2, map_location='cpu')

print("=" * 80)
print("Dream 归因结果一致性分析")
print("=" * 80)
print(f"\n配置 1: seed=47")
print(f"配置 2: seed=1234")
print(f"\n元数据:")
print(f"  方法: {data1['metadata'].get('method', 'N/A')}")
print(f"  样本数: {data1['metadata'].get('max_samples', 'N/A')}")
print(f"  IG steps: {data1['metadata'].get('ig_steps', 'N/A')}")
print(f"  Mask probs: {data1['metadata'].get('mask_probs', 'N/A')}")

# 获取重要性分数
scores1 = data1['importance_scores']
scores2 = data2['importance_scores']

layers = sorted(scores1.keys())
print(f"\n层数: {len(layers)}")
print(f"层索引: {min(layers)} ~ {max(layers)}")

# 逐层分析
print("\n" + "=" * 80)
print("逐层一致性分析")
print("=" * 80)

all_pearson = []
all_spearman = []
all_top10_overlap = []
all_top20_overlap = []

for layer_idx in layers:
    s1 = scores1[layer_idx].numpy()
    s2 = scores2[layer_idx].numpy()
    
    # 相关系数
    pearson_r, _ = pearsonr(s1, s2)
    spearman_r, _ = spearmanr(s1, s2)
    
    all_pearson.append(pearson_r)
    all_spearman.append(spearman_r)
    
    # Top-K overlap
    n_heads = len(s1)
    top10_1 = set(np.argsort(s1)[-10:])
    top10_2 = set(np.argsort(s2)[-10:])
    overlap10 = len(top10_1 & top10_2) / 10.0
    all_top10_overlap.append(overlap10)
    
    top20_1 = set(np.argsort(s1)[-20:])
    top20_2 = set(np.argsort(s2)[-20:])
    overlap20 = len(top20_1 & top20_2) / 20.0
    all_top20_overlap.append(overlap20)
    
    if layer_idx % 5 == 0:  # 每5层打印一次
        print(f"Layer {layer_idx:2d}: Pearson={pearson_r:.4f}, Spearman={spearman_r:.4f}, "
              f"Top10-overlap={overlap10:.2%}, Top20-overlap={overlap20:.2%}")

# 总体统计
print("\n" + "=" * 80)
print("总体一致性统计")
print("=" * 80)
print(f"\nPearson 相关系数:")
print(f"  平均: {np.mean(all_pearson):.4f}")
print(f"  中位数: {np.median(all_pearson):.4f}")
print(f"  最小: {np.min(all_pearson):.4f}")
print(f"  最大: {np.max(all_pearson):.4f}")
print(f"  标准差: {np.std(all_pearson):.4f}")

print(f"\nSpearman 秩相关系数:")
print(f"  平均: {np.mean(all_spearman):.4f}")
print(f"  中位数: {np.median(all_spearman):.4f}")
print(f"  最小: {np.min(all_spearman):.4f}")
print(f"  最大: {np.max(all_spearman):.4f}")
print(f"  标准差: {np.std(all_spearman):.4f}")

print(f"\nTop-10 头重叠率:")
print(f"  平均: {np.mean(all_top10_overlap):.2%}")
print(f"  中位数: {np.median(all_top10_overlap):.2%}")
print(f"  最小: {np.min(all_top10_overlap):.2%}")
print(f"  最大: {np.max(all_top10_overlap):.2%}")

print(f"\nTop-20 头重叠率:")
print(f"  平均: {np.mean(all_top20_overlap):.2%}")
print(f"  中位数: {np.median(all_top20_overlap):.2%}")
print(f"  最小: {np.min(all_top20_overlap):.2%}")
print(f"  最大: {np.max(all_top20_overlap):.2%}")

# 全局相关性（所有层的分数拼接）
print("\n" + "=" * 80)
print("全局一致性（跨所有层）")
print("=" * 80)

all_scores1 = np.concatenate([scores1[l].numpy() for l in layers])
all_scores2 = np.concatenate([scores2[l].numpy() for l in layers])

global_pearson, _ = pearsonr(all_scores1, all_scores2)
global_spearman, _ = spearmanr(all_scores1, all_scores2)

print(f"\n全局 Pearson 相关系数: {global_pearson:.4f}")
print(f"全局 Spearman 相关系数: {global_spearman:.4f}")

# 分数分布对比
print("\n" + "=" * 80)
print("分数分布统计")
print("=" * 80)

print(f"\nSeed 47:")
print(f"  均值: {all_scores1.mean():.6f}")
print(f"  中位数: {np.median(all_scores1):.6f}")
print(f"  标准差: {all_scores1.std():.6f}")
print(f"  最小值: {all_scores1.min():.6f}")
print(f"  最大值: {all_scores1.max():.6f}")

print(f"\nSeed 1234:")
print(f"  均值: {all_scores2.mean():.6f}")
print(f"  中位数: {np.median(all_scores2):.6f}")
print(f"  标准差: {all_scores2.std():.6f}")
print(f"  最小值: {all_scores2.min():.6f}")
print(f"  最大值: {all_scores2.max():.6f}")

# 判断一致性水平
print("\n" + "=" * 80)
print("一致性评估")
print("=" * 80)

mean_pearson = np.mean(all_pearson)
mean_spearman = np.mean(all_spearman)
mean_top10_overlap = np.mean(all_top10_overlap)

if mean_pearson > 0.9 and mean_spearman > 0.9:
    consistency = "非常高"
    color = "🟢"
elif mean_pearson > 0.8 and mean_spearman > 0.8:
    consistency = "高"
    color = "🟡"
elif mean_pearson > 0.7 and mean_spearman > 0.7:
    consistency = "中等"
    color = "🟠"
else:
    consistency = "较低"
    color = "🔴"

print(f"\n{color} 总体一致性水平: {consistency}")
print(f"\n解释:")
print(f"  - Pearson 相关 {mean_pearson:.3f} ({'优秀' if mean_pearson > 0.9 else '良好' if mean_pearson > 0.8 else '一般' if mean_pearson > 0.7 else '较差'})")
print(f"  - Spearman 相关 {mean_spearman:.3f} ({'优秀' if mean_spearman > 0.9 else '良好' if mean_spearman > 0.8 else '一般' if mean_spearman > 0.7 else '较差'})")
print(f"  - Top-10 重叠 {mean_top10_overlap:.1%} ({'优秀' if mean_top10_overlap > 0.7 else '良好' if mean_top10_overlap > 0.6 else '一般' if mean_top10_overlap > 0.5 else '较差'})")

if mean_pearson > 0.85:
    print(f"\n✅ 结论: 两组结果高度一致，归因方法稳定可靠。")
elif mean_pearson > 0.7:
    print(f"\n⚠️  结论: 两组结果基本一致，但存在一定差异，可能受随机性影响。")
else:
    print(f"\n❌ 结论: 两组结果一致性较低，归因方法可能不够稳定。")

print("\n" + "=" * 80)

