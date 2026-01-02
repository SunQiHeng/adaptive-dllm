#!/usr/bin/env python3
"""
可视化两组归因结果的一致性
"""

import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 加载两组结果
file1 = "/home/qiheng/Projects/adaptive-dllm/configs/head_importance_dream_base_loss_gateIG_zero_maskp0.15-0.3-0.5-0.7-0.9_mcs2_mean_masked_seed47_n50_k8_L2048_dseed47_mseed47_ts20251227_191418/head_importance.pt"
file2 = "/home/qiheng/Projects/adaptive-dllm/configs/head_importance_dream_base_loss_gateIG_zero_maskp0.15-0.3-0.5-0.7-0.9_mcs2_mean_masked_seed1234_n50_k8_L2048_dseed1234_mseed1234_ts20251227_191323/head_importance.pt"

data1 = torch.load(file1, map_location='cpu')
data2 = torch.load(file2, map_location='cpu')

scores1 = data1['importance_scores']
scores2 = data2['importance_scores']
layers = sorted(scores1.keys())

# 创建图表
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Dream Attribution Consistency Analysis (seed=47 vs seed=1234)', fontsize=16, fontweight='bold')

# 1. 逐层相关系数
ax = axes[0, 0]
pearson_scores = []
spearman_scores = []
for l in layers:
    s1 = scores1[l].numpy()
    s2 = scores2[l].numpy()
    p, _ = pearsonr(s1, s2)
    s, _ = spearmanr(s1, s2)
    pearson_scores.append(p)
    spearman_scores.append(s)

ax.plot(layers, pearson_scores, 'o-', label='Pearson', linewidth=2, markersize=6)
ax.plot(layers, spearman_scores, 's-', label='Spearman', linewidth=2, markersize=6)
ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='Excellent (>0.9)')
ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='Good (>0.8)')
ax.set_xlabel('Layer Index', fontsize=12)
ax.set_ylabel('Correlation Coefficient', fontsize=12)
ax.set_title('Per-Layer Correlation', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

# 2. 全局散点图
ax = axes[0, 1]
all_scores1 = np.concatenate([scores1[l].numpy() for l in layers])
all_scores2 = np.concatenate([scores2[l].numpy() for l in layers])

# 采样显示（太多点会很慢）
sample_idx = np.random.choice(len(all_scores1), min(2000, len(all_scores1)), replace=False)
ax.scatter(all_scores1[sample_idx], all_scores2[sample_idx], alpha=0.3, s=10)

# 拟合线
z = np.polyfit(all_scores1, all_scores2, 1)
p = np.poly1d(z)
x_line = np.linspace(all_scores1.min(), all_scores1.max(), 100)
ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'Fit: y={z[0]:.3f}x+{z[1]:.3f}')

# 对角线
ax.plot([all_scores1.min(), all_scores1.max()], [all_scores1.min(), all_scores1.max()], 
        'k-', alpha=0.5, linewidth=1, label='Perfect correlation')

global_pearson, _ = pearsonr(all_scores1, all_scores2)
ax.set_xlabel('Seed 47 Scores', fontsize=12)
ax.set_ylabel('Seed 1234 Scores', fontsize=12)
ax.set_title(f'Global Correlation (Pearson={global_pearson:.3f})', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Top-K 重叠率
ax = axes[0, 2]
top_k_values = [5, 10, 15, 20, 25, 30]
overlaps = []

for k in top_k_values:
    overlap_per_layer = []
    for l in layers:
        s1 = scores1[l].numpy()
        s2 = scores2[l].numpy()
        top_k1 = set(np.argsort(s1)[-k:])
        top_k2 = set(np.argsort(s2)[-k:])
        overlap = len(top_k1 & top_k2) / k
        overlap_per_layer.append(overlap)
    overlaps.append(np.mean(overlap_per_layer))

ax.plot(top_k_values, overlaps, 'o-', linewidth=2, markersize=8, color='green')
ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='Good (>80%)')
ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Acceptable (>70%)')
ax.set_xlabel('Top-K', fontsize=12)
ax.set_ylabel('Average Overlap Rate', fontsize=12)
ax.set_title('Top-K Head Overlap', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

# 4. 分数分布对比
ax = axes[1, 0]
ax.hist(all_scores1, bins=50, alpha=0.5, label='Seed 47', density=True, color='blue')
ax.hist(all_scores2, bins=50, alpha=0.5, label='Seed 1234', density=True, color='red')
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Score Distribution', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 5. 相关系数分布
ax = axes[1, 1]
ax.hist(pearson_scores, bins=20, alpha=0.7, label=f'Pearson (mean={np.mean(pearson_scores):.3f})', color='blue')
ax.hist(spearman_scores, bins=20, alpha=0.7, label=f'Spearman (mean={np.mean(spearman_scores):.3f})', color='red')
ax.axvline(x=0.9, color='green', linestyle='--', linewidth=2, label='Excellent threshold')
ax.set_xlabel('Correlation Coefficient', fontsize=12)
ax.set_ylabel('Number of Layers', fontsize=12)
ax.set_title('Distribution of Correlation Across Layers', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 6. 热力图：逐层Top-10重叠
ax = axes[1, 2]
top10_matrix = []
for l in layers:
    s1 = scores1[l].numpy()
    s2 = scores2[l].numpy()
    
    top10_1 = set(np.argsort(s1)[-10:])
    top10_2 = set(np.argsort(s2)[-10:])
    
    # 创建混淆矩阵风格的展示
    overlap = len(top10_1 & top10_2)
    only_seed47 = len(top10_1 - top10_2)
    only_seed1234 = len(top10_2 - top10_1)
    
    top10_matrix.append([overlap, only_seed47, only_seed1234])

top10_matrix = np.array(top10_matrix)
im = ax.imshow(top10_matrix.T, aspect='auto', cmap='YlGnBu', interpolation='nearest')

ax.set_yticks([0, 1, 2])
ax.set_yticklabels(['Both', 'Only Seed47', 'Only Seed1234'])
ax.set_xlabel('Layer Index', fontsize=12)
ax.set_title('Top-10 Heads Agreement per Layer', fontsize=13, fontweight='bold')

# 添加colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Number of Heads', fontsize=10)

# 添加数值标注（每5层）
for l_idx in range(0, len(layers), 5):
    for cat_idx in range(3):
        value = top10_matrix[l_idx, cat_idx]
        if value > 0:
            ax.text(l_idx, cat_idx, f'{int(value)}', ha='center', va='center', 
                   color='white' if value > 5 else 'black', fontsize=8)

plt.tight_layout()
output_path = '/home/qiheng/Projects/adaptive-dllm/models/Dream/attribution/loss_attribution/consistency_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ 可视化结果已保存到: {output_path}")

plt.close()

# 创建总结报告
print("\n" + "="*80)
print("一致性分析总结")
print("="*80)
print(f"\n📊 关键指标:")
print(f"  • 平均 Pearson 相关: {np.mean(pearson_scores):.4f} ({'🟢 优秀' if np.mean(pearson_scores) > 0.9 else '🟡 良好'})")
print(f"  • 平均 Spearman 相关: {np.mean(spearman_scores):.4f} ({'🟢 优秀' if np.mean(spearman_scores) > 0.9 else '🟡 良好'})")
print(f"  • Top-10 平均重叠: {overlaps[1]:.1%} ({'🟢 优秀' if overlaps[1] > 0.8 else '🟡 良好'})")
print(f"  • 相关系数 > 0.9 的层数: {sum(np.array(pearson_scores) > 0.9)}/{len(layers)} ({sum(np.array(pearson_scores) > 0.9)/len(layers):.1%})")

print(f"\n🔍 详细观察:")
worst_layer = layers[np.argmin(pearson_scores)]
best_layer = layers[np.argmax(pearson_scores)]
print(f"  • 一致性最低的层: Layer {worst_layer} (Pearson={min(pearson_scores):.4f})")
print(f"  • 一致性最高的层: Layer {best_layer} (Pearson={max(pearson_scores):.4f})")

print(f"\n✅ 结论:")
if np.mean(pearson_scores) > 0.9 and np.mean(spearman_scores) > 0.85:
    print(f"   两组结果具有非常高的一致性，说明:")
    print(f"   1. 归因方法对随机种子不敏感，结果稳定")
    print(f"   2. data_seed 和 mask_seed 的不同主要影响样本选择和mask位置")
    print(f"   3. 但最终的头重要性排序高度一致")
    print(f"   4. 可以安全地使用这些归因结果进行剪枝")
elif np.mean(pearson_scores) > 0.8:
    print(f"   两组结果具有较高的一致性，但存在一定差异:")
    print(f"   1. 建议使用多个种子的平均结果")
    print(f"   2. 或者增加样本数以降低方差")
else:
    print(f"   ⚠️ 两组结果一致性偏低，建议:")
    print(f"   1. 增加样本数量")
    print(f"   2. 检查归因方法的实现")
    print(f"   3. 考虑使用ensemble方法")

print("\n" + "="*80)

