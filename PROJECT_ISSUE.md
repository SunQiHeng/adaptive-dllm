# Dream Attribution Logits 对齐问题记录

## 问题标题

Dream 的 attribution loss 没有对齐真实 `diffusion_generate()` 的采样 logits 语义。

## 当前结论

~~这是一个**高可疑点**，但目前还不是完全确认的"硬 bug"。~~

经过深入排查，**这是一个已确认的实质性 bug**，已完成修复。

### 确认依据

1. **DiffuLLaMA（Dream 前身，同为 HKUNLP）的官方训练代码**中，loss 计算使用了标准的 next-token shift：

   ```python
   # https://github.com/HKUNLP/DiffuLLaMA/blob/main/DiffuLLaMA-training/train.py
   logits = model(local_input_ids, position_ids=local_position_ids).logits
   logits = logits[:,:-1]              # 丢弃最后一个位置
   loss_mask = loss_mask[:,1:]         # 丢弃第一个位置
   local_target_ids = local_target_ids[:,1:]  # 丢弃第一个位置
   loss = loss_func(logits.reshape(-1, logits.shape[-1]), local_target_ids.reshape(-1))
   ```

   这证明 `raw_logits[:, j, :]` 的训练目标是预测 `token[j+1]`（next-token），而非 `token[j]`（same-position）。

2. **Dream 官方 HuggingFace 模型** (`Dream-org/Dream-v0-Instruct-7B`) 的 `generation_utils.py` 中，生成路径始终执行右移：

   ```python
   logits = self(x, attention_mask, tok_idx).logits
   logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
   ```

   这将 next-token logits 转换为 same-position logits，以便正确采样 mask 位置的 token。

3. **DiffuLLaMA 生成代码**中，右移是一个显式的可配置标志 `diff_args.shift`，并且在使用从 causal LM 适配的模型时启用。Dream 继承了这一设计但将其固化为始终执行。

### 结论

- `DreamModel.forward()` 的 `lm_head` 保留了从 causal LM 继承的 **next-token prediction 语义**：`raw_logits[:, j, :]` 预测 `token[j+1]`。
- 尽管模型返回 `MaskedLMOutput`（HuggingFace 类型标注），但训练时实际使用了 shifted labels。
- Attribution 之前直接用 `raw_logits[j]` 对比 `token[j]`，等于在优化一个**与模型训练目标和生成行为都不一致**的函数。
- LLaDA **不存在**此问题，因为 LLaDA 是真正的 masked LM，`logits[j]` 直接预测 `token[j]`。

---

## 涉及文件

- `models/Dream/attribution/loss_attribution/compute_loss_attribution.py`
- `models/Dream/attribution/loss_attribution/compute_loss_attribution_all_heads.py`
- `models/Dream/generation_utils/generation_utils_dream.py`
- `evaluation/dream/eval_dream.py`

---

## 关键代码现象

### 1. （修复前）attribution 里直接对原始 logits 做 same-position CE

在 Dream attribution 中，loss 的计算是：

1. 构造 `input_ids_masked`
2. 前向得到 `model(...).logits`
3. 直接在被 mask 的位置上计算 CE：`CE(logits[:, j, :], token[j])`

这假设 `logits[:, j, :]` 预测 `token[j]`（same-position），但模型实际训练目标是 `logits[:, j, :]` 预测 `token[j+1]`（next-token）。

### 2. Dream 真实生成时会先对 logits 右移一位

在 `models/Dream/generation_utils/generation_utils_dream.py` 中，真实生成路径是：

1. 先得到 `self(x, ...).logits`
2. 然后执行：

```python
logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
```

3. 再从这些 logits 中取出 `mask_index` 位置对应的分布进行采样

右移后的语义：
- `shifted_logits[:, 0, :] = raw_logits[:, 0, :]`
- `shifted_logits[:, j, :] = raw_logits[:, j-1, :]`（对 j ≥ 1）

由于 `raw_logits[j-1]` 的训练目标是 `token[j]`，所以 `shifted_logits[j]` 正确地预测了位置 `j` 的 token。

### 3. `eval_dream.py` 里的 loglikelihood 路径也有同样的不对齐

```python
# eval_dream.py → get_loglikelihood()
logits = self.get_logits(perturbed_seq, prompt_index, attention_mask, position_ids)
loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
```

`get_logits()` 返回的是 raw logits（没有右移），然后用 `seq[mask_indices]`（same-position token）计算 CE。这也存在不对齐，但 MMLU 等任务是比较多个选项的**相对**似然排名，错位对相对排名影响可能较小。

---

## 已完成的修复

### 修复思路

在 attribution 的 loss 计算路径中，对 `model(...).logits` 应用与 `diffusion_generate()` 完全一致的右移。

### 修复方式

#### 新增辅助函数 `_apply_dream_logits_shift`

位于 `compute_loss_attribution.py`，被两个归因脚本共用：

```python
def _apply_dream_logits_shift(raw_logits: torch.Tensor, trim_first: bool) -> torch.Tensor:
    shifted = torch.cat([raw_logits[:, :1], raw_logits[:, :-1]], dim=1)
    return shifted[:, 1:] if trim_first else shifted
```

#### 处理 `num_logits_to_keep` 优化

`DreamModel.forward(num_logits_to_keep=K)` 只返回末尾 K 个位置的 logits。右移后第一个 completion 位置需要"窗口外"前一个位置的 logits，因此请求 K+1 个位置，右移后截掉多出的第一个：

```python
num_logits_to_keep = int(full_input_ids.size(1) - int(completion_start))
_shift_nlk = min(num_logits_to_keep + 1, int(full_input_ids.size(1)))
_shift_trim = (_shift_nlk > num_logits_to_keep)
```

### 修改的文件与位置

#### `compute_loss_attribution.py`（layer-wise 版本）

| 位置 | 修改内容 |
|---|---|
| `_dry_run_check_o_proj_shape` 之后 | 新增 `_apply_dream_logits_shift()` 函数 |
| `_forward_logits()` 函数体 | 改为请求 `_shift_nlk` 个 logits，然后调用 `_apply_dream_logits_shift()` |
| `num_logits_to_keep` 赋值之后 | 新增 `_shift_nlk` 和 `_shift_trim` 计算 |
| `debug_gate` sanity check 分支 | `attention_mask is not None` 时也改为请求 `_shift_nlk` 个 logits，并统一调用 `_apply_dream_logits_shift()`，确保与主 loss 路径完全一致 |

#### `compute_loss_attribution_all_heads.py`（all-heads 联合版本）

| 位置 | 修改内容 |
|---|---|
| 模块级 import 区域 | 新增 `_apply_dream_logits_shift = _base._apply_dream_logits_shift` |
| `num_logits_to_keep` 赋值之后 | 新增 `_shift_nlk` 和 `_shift_trim` 计算 |
| IG 循环中所有 `model(...)` 调用 | 改为请求 `_shift_nlk` 个 logits + 调用 `_apply_dream_logits_shift()` |
| debug gate 中所有 `model(...)` 调用 | 同上 |

#### `eval_dream.py`（loglikelihood 评测路径）

| 位置 | 修改内容 |
|---|---|
| `get_logits()` 方法，`return` 之前 | 新增 `logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)` |

说明：`get_logits()` 被 `get_loglikelihood()` 和 `suffix_greedy_prediction()` 调用，
修复统一放在 `get_logits()` 内部，一次性覆盖所有 loglikelihood 评测路径。
`generate_until()` 走的是 `diffusion_generate()`，该路径本身已包含右移，无需修改。

---

## 后续待验证

### ablation 对比（建议做）

- **版本 A**：修复前的 `head_importance.pt`（已有）
- **版本 B**：用修复后的代码、相同参数重新跑归因

比较项：
- top-k head 重合率
- head importance ranking 的 Spearman/Kendall 相关系数
- 下游 pruning/masking 实验结果差异

如果差异显著，则确认此 bug 对最终实验结果有实质影响。

### ~~`eval_dream.py` loglikelihood 路径~~（已修复）

~~`get_loglikelihood()` 中的 `get_logits()` 也没有右移，存在同样的不对齐问题。~~

已在 `get_logits()` 内部统一加入右移，覆盖 `get_loglikelihood()` 和 `suffix_greedy_prediction()` 两条调用路径。
`generate_until()` 走 `diffusion_generate()` 本身已含右移，不受影响。

> **注意**：该修复会影响 MMLU 等 loglikelihood 类任务的绝对分数。但由于之前的分数本身基于不对齐的 logits 计算，修复后的分数才是正确对齐 Dream 训练语义的结果。在相对排序任务中，修复前后的最终准确率差异可能较小。

### `compute_loss_attribution.py` 的 `debug_gate` 分支（已同步修复）

此前 `layer-wise` 版本虽然主 IG/loss 路径已经完成 shift 修复，但 `debug_gate` 中 `attention_mask is not None` 的分支仍然直接读取 raw logits，和主路径不完全一致。

现已同步修复为：

- 请求 `_shift_nlk` 个 logits
- 调用 `_apply_dream_logits_shift()`
- 保证 debug sanity check 与实际 attribution loss 使用完全相同的 logits 语义

该问题在当前 Dream attribution 默认 `attention_mask=None` 的无 padding 路径下通常不会影响实验结果，但从代码一致性和后续可维护性角度，仍然应该修掉。

---

## 参考资料

- Dream 官方 HuggingFace 模型：`Dream-org/Dream-v0-Instruct-7B`
  - `generation_utils.py`：确认右移是官方生成逻辑
  - `modeling_dream.py`：返回 `MaskedLMOutput` 但训练实际使用 shifted labels
- DiffuLLaMA 官方 GitHub：`https://github.com/HKUNLP/DiffuLLaMA`
  - `DiffuLLaMA-training/train.py`：训练代码确认 next-token shift
  - `model.py`：生成代码中 `diff_args.shift` 标志确认右移是可配置的适配机制
