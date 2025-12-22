# Adaptive Keep Ratio 测试总结

## 测试脚本位置

- **LLaDA测试**: `/home/qiheng/Projects/adaptive-dllm/tests/test_adaptive_keep_ratio_llada.py`
- **Dream测试**: `/home/qiheng/Projects/adaptive-dllm/tests/test_adaptive_keep_ratio_dream.py`
- **运行所有测试**: `/home/qiheng/Projects/adaptive-dllm/tests/run_keep_ratio_tests.sh`

## 如何运行

```bash
# 运行LLaDA测试
cd /home/qiheng/Projects/adaptive-dllm
python tests/test_adaptive_keep_ratio_llada.py

# 运行Dream测试
python tests/test_adaptive_keep_ratio_dream.py

# 或运行所有测试
bash tests/run_keep_ratio_tests.sh
```

## 测试结果

### LLaDA模型 ✅

**配置:**
- Layers: 32
- KV Heads: 8 (每层)
- 使用新版本adaptive_utils (相对权重模式)
- 加载预计算的importance scores

**测试结果（select参数对应的平均keep_ratio）:**

| Select值 | 目标平均 | 实际平均 | 偏差 | 触及上限的heads | 评价 |
|---------|---------|---------|------|----------------|------|
| 0.2 | 20.0% | 20.0% | 0.0% | 0/256 (0.0%) | ✅ 完美 |
| 0.3 | 30.0% | 30.0% | 0.0% | 0/256 (0.0%) | ✅ 完美 |
| 0.5 | 50.0% | 50.0% | 0.0% | 1/256 (0.4%) | ✅ 完美 |
| 0.8 | 80.0% | 78.3% | 1.7% | 113/256 (44.1%) | ⚠️ 轻微偏差 |
| 1.0 | 100.0% | 92.9% | 7.1% | 366/256 (143.0%) | ❌ 明显偏差 |

**关键发现:**
1. ✅ **在低到中等sparsity（select≤0.5）时，实际平均值与目标完全一致**
2. ⚠️ 在高sparsity（select=0.8）时，由于部分heads触及上限（1.0），导致轻微偏差
3. ❌ 在select=1.0时，大量heads触及上限，导致实际平均约93%而非100%

**相对权重范围:**
- 全局mean: 1.0000 (完美归一化)
- 全局range: [0.6013, 2.0992]
- 说明：不同head的相对重要性差异约3.5倍

**层级分布示例 (select=0.3):**
- 每层平均: 恰好30.0%
- 每层范围示例: Layer 0 [20.6%, 47.0%], Layer 23 [18.6%, 41.7%]
- 显示出明显的head重要性差异化

### Dream模型 ⚠️

**配置:**
- Layers: 28
- KV Heads: 4 (每层)
- 使用旧版本adaptive_utils_dream (绝对keep_ratio模式)
- 加载预计算的importance scores

**当前状态:**
- **预计算的平均keep_ratio: 23.4%**
- 范围: [10.0%, 90.0%]
- ⚠️ **select参数不影响adaptive模式**（旧实现）

**问题:**
Dream模型当前使用旧版本的`adaptive_utils_dream.py`，该版本：
1. 直接输出绝对的keep_ratio值（基于min_sparsity=0.1, max_sparsity=0.9）
2. 在推理时不使用select参数进行缩放
3. 最终的保留比例固定为预计算值的平均（约23.4%）

**建议:**
需要升级Dream的adaptive_utils_dream.py到新版本（使用相对权重），以支持select参数控制。

## 结论

### LLaDA: ✅ 实现正确

新的相对权重方法在LLaDA中工作完美：
- `select` 参数直接控制平均保留比例
- 在合理范围内（select≤0.5），实际值与目标完全一致
- 保持了不同heads之间的相对重要性差异

### Dream: ⚠️ 需要升级

Dream模型需要升级到新版本的adaptive_utils以支持select参数控制。

## 技术细节

### 相对权重方法的优势

1. **解耦设计**: importance → relative weights → keep_ratio
2. **select直接控制**: `keep_ratio = weight * select`
3. **保持相对性**: 重要head保留更多，不重要head保留更少
4. **可预测性**: 平均值≈select（当未触及上限时）

### 触及上限问题

当select值较大时（如0.8, 1.0），部分重要性高的heads会触及keep_ratio=1.0的上限：
- 这些heads的keep_ratio被截断为1.0
- 导致实际平均值略低于目标select值
- 这是设计中的权衡：保证合理范围同时保持差异化

### 建议的select值范围

基于测试结果：
- **推荐范围**: 0.2 ~ 0.6
- **可接受范围**: 0.1 ~ 0.8
- **不推荐**: > 0.8 (大量heads触及上限)

---

生成时间: 2025-12-20

