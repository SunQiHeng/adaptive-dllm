#!/usr/bin/env python3
"""
分析归因分数的符号含义
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
file1 = "/home/qiheng/Projects/adaptive-dllm/configs/head_importance_dream_base_loss_gateIG_zero_maskp0.15-0.3-0.5-0.7-0.9_mcs2_mean_masked_seed47_n50_k8_L2048_dseed47_mseed47_ts20251227_191418/head_importance.pt"

data = torch.load(file1, map_location='cpu')
scores = data['importance_scores']
layers = sorted(scores.keys())

print("=" * 80)
print("归因分数符号分析")
print("=" * 80)

print(f"\n配置信息:")
print(f"  baseline: {data['metadata']['baseline']}")
print(f"  ig_postprocess: {data['metadata']['ig_postprocess']}")
print(f"  baseline_scalar: {data['metadata']['baseline_scalar']}")

print("\n" + "=" * 80)
print("理论解释")
print("=" * 80)
print("""
IG (Integrated Gradients) 计算的是：
  IG_i = ∫[0,1] (∂Loss/∂α_i)|_{α=t} dt

其中：
  - α=0: head 被完全关闭（输出置零）
  - α=1: head 正常工作
  - ∂Loss/∂α: loss 对 α 的梯度

符号含义：
  - 如果 IG < 0（负值）：
    * ∂Loss/∂α < 0，增大 α 会降低 loss
    * 即：开启这个 head 会降低 loss
    * 这个 head 是【有用/有益】的
    
  - 如果 IG > 0（正值）：
    * ∂Loss/∂α > 0，增大 α 会增加 loss
    * 即：开启这个 head 会增加 loss
    * 这个 head 是【有害/冗余】的
    
  - 如果 IG ≈ 0：
    * head 对 loss 几乎没有影响
    * 这个 head 是【不重要/冗余】的
""")

print("\n" + "=" * 80)
print("数值统计")
print("=" * 80)

all_scores = np.concatenate([scores[l].numpy() for l in layers])

print(f"\n全局统计:")
print(f"  总 head 数: {len(all_scores)}")
print(f"  均值: {all_scores.mean():.6f}")
print(f"  中位数: {np.median(all_scores):.6f}")
print(f"  标准差: {all_scores.std():.6f}")

print(f"\n符号分布:")
negative = (all_scores < 0).sum()
positive = (all_scores > 0).sum()
zero = (all_scores == 0).sum()
near_zero = (np.abs(all_scores) < 0.001).sum()

print(f"  负值 (有用): {negative} ({negative/len(all_scores)*100:.1f}%)")
print(f"  正值 (有害): {positive} ({positive/len(all_scores)*100:.1f}%)")
print(f"  零值: {zero} ({zero/len(all_scores)*100:.1f}%)")
print(f"  接近零 (|x|<0.001): {near_zero} ({near_zero/len(all_scores)*100:.1f}%)")

print(f"\n分位数:")
for q in [0, 10, 25, 50, 75, 90, 100]:
    val = np.percentile(all_scores, q)
    print(f"  {q:3d}%: {val:+.6f}")

print(f"\n负值统计 (有用的 head):")
neg_scores = all_scores[all_scores < 0]
if len(neg_scores) > 0:
    print(f"  数量: {len(neg_scores)}")
    print(f"  均值: {neg_scores.mean():.6f}")
    print(f"  最小值: {neg_scores.min():.6f}")
    print(f"  最大值: {neg_scores.max():.6f}")
else:
    print(f"  没有负值!")

print(f"\n正值统计 (有害的 head):")
pos_scores = all_scores[all_scores > 0]
if len(pos_scores) > 0:
    print(f"  数量: {len(pos_scores)}")
    print(f"  均值: {pos_scores.mean():.6f}")
    print(f"  最小值: {pos_scores.min():.6f}")
    print(f"  最大值: {pos_scores.max():.6f}")
else:
    print(f"  没有正值!")

print("\n" + "=" * 80)
print("逐层分析")
print("=" * 80)

layer_stats = []
for l in layers:
    s = scores[l].numpy()
    neg_ratio = (s < 0).sum() / len(s)
    mean_val = s.mean()
    layer_stats.append({
        'layer': l,
        'mean': mean_val,
        'neg_ratio': neg_ratio,
        'pos_ratio': (s > 0).sum() / len(s),
    })

print(f"\n每层的均值和负值比例:")
print(f"{'Layer':>6} {'Mean':>10} {'Neg%':>8} {'Pos%':>8}")
print("-" * 34)
for stat in layer_stats:
    print(f"{stat['layer']:6d} {stat['mean']:+10.6f} {stat['neg_ratio']*100:7.1f}% {stat['pos_ratio']*100:7.1f}%")

print(f"\n层级趋势:")
early_layers = [s['mean'] for s in layer_stats[:7]]
middle_layers = [s['mean'] for s in layer_stats[7:21]]
late_layers = [s['mean'] for s in layer_stats[21:]]

print(f"  浅层 (0-6):   均值 = {np.mean(early_layers):+.6f}")
print(f"  中层 (7-20):  均值 = {np.mean(middle_layers):+.6f}")
print(f"  深层 (21-27): 均值 = {np.mean(late_layers):+.6f}")

print("\n" + "=" * 80)
print("重要发现")
print("=" * 80)

if all_scores.mean() > 0.001:
    print(f"""
⚠️  平均归因分数为正 ({all_scores.mean():.6f})

这意味着什么？
1. 【平均而言】，开启这些 head 会**增加** loss
2. 这暗示模型可能存在大量冗余或有害的 head
3. 或者说，模型在训练时学到了一些次优的参数

可能的原因：
a) 模型过参数化：存在大量冗余 head
b) 训练不充分：某些 head 没有学到有用的模式
c) Baseline 选择：α=0 时的行为可能不是"纯粹的移除"，还有 residual 影响
d) 路径依赖：从 α=0 到 α=1 的路径可能不是线性的

建议：
- 检查是否所有 head 的归因都是正的（如果是，可能是实现问题）
- 分析负值 head 的特征和分布
- 考虑使用绝对值 |IG| 来衡量重要性，忽略符号
- 验证 gate 实现是否正确
""")
elif all_scores.mean() < -0.001:
    print(f"""
✅ 平均归因分数为负 ({all_scores.mean():.6f})

这是符合预期的！
- 平均而言，开启这些 head 会**降低** loss
- 说明大部分 head 对模型是有益的
- 符合训练好的模型的特征
""")
else:
    print(f"""
🤔 平均归因分数接近零 ({all_scores.mean():.6f})

这表示：
- 正负归因大致平衡
- 可能存在大量接近零的冗余 head
""")

print("\n" + "=" * 80)
print("如何使用这些归因分数")
print("=" * 80)
print("""
根据符号的含义，剪枝策略应该是：

1. 【基于绝对值】：
   - 剪枝 |IG| 最小的 head（无论正负）
   - 理由：|IG| 小表示对 loss 影响小
   - 这是最常用的策略

2. 【基于符号+绝对值】：
   - 优先剪枝正值且 IG 较大的 head（有害的）
   - 然后剪枝 |IG| 较小的 head（不重要的）
   - 保留负值且 |IG| 较大的 head（有用的）

3. 【当前的排序】：
   - 如果使用原始符号分数排序，最小值（最负）= 最有用
   - 最大值（最正）= 最有害
   - 剪枝时应该剪掉【接近零】的，或者【最正】的

建议使用绝对值来衡量重要性！
""")

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Attribution Score Sign Analysis', fontsize=14, fontweight='bold')

# 1. 直方图
ax = axes[0, 0]
ax.hist(all_scores, bins=50, alpha=0.7, edgecolor='black')
ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero line')
ax.axvline(x=all_scores.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean={all_scores.mean():.4f}')
ax.set_xlabel('Attribution Score', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Distribution of Attribution Scores', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. 逐层均值
ax = axes[0, 1]
layer_means = [scores[l].mean().item() for l in layers]
ax.plot(layers, layer_means, 'o-', linewidth=2, markersize=6)
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero line')
ax.set_xlabel('Layer Index', fontsize=11)
ax.set_ylabel('Mean Attribution Score', fontsize=11)
ax.set_title('Mean Score per Layer', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. 负值比例
ax = axes[1, 0]
neg_ratios = [(scores[l] < 0).float().mean().item() for l in layers]
pos_ratios = [(scores[l] > 0).float().mean().item() for l in layers]
ax.plot(layers, neg_ratios, 'o-', label='Negative (beneficial)', linewidth=2, markersize=6, color='blue')
ax.plot(layers, pos_ratios, 's-', label='Positive (harmful)', linewidth=2, markersize=6, color='red')
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Layer Index', fontsize=11)
ax.set_ylabel('Ratio', fontsize=11)
ax.set_title('Sign Distribution per Layer', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1])

# 4. Box plot by sign
ax = axes[1, 1]
data_to_plot = [neg_scores, pos_scores]
labels = [f'Negative\n(n={len(neg_scores)})', f'Positive\n(n={len(pos_scores)})']
bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
bp['boxes'][0].set_facecolor('blue')
bp['boxes'][0].set_alpha(0.5)
bp['boxes'][1].set_facecolor('red')
bp['boxes'][1].set_alpha(0.5)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel('Attribution Score', fontsize=11)
ax.set_title('Distribution by Sign', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_path = '/home/qiheng/Projects/adaptive-dllm/models/Dream/attribution/loss_attribution/sign_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ 可视化已保存到: {output_path}")

