# adaptive-dllm 项目速览（给大模型/合作者用）

这份文档的目标是：让任何外部大模型或合作者在 **不反复读全仓库** 的情况下，快速理解本项目在做什么、代码入口在哪里、论文当前应如何描述方法，以及最容易写错/写旧的地方。

---

## 1. 项目一句话

本项目研究 **Diffusion(-style) Large Language Models (DLLMs/DLMs)** 中 **attention head 的归因**：提出基于 **Head Integrated Gradients (HeadIG)** 的 loss-based head 重要性打分，并用两类 **因果验证**（adaptive sparse attention 与 head masking/pruning）来检验分数是否有效。

---

## 2. 当前方法口径（论文要与之对齐）

### 2.1 归因目标（loss-based）

- **归因对象**：Transformer 多头注意力中的 head（按 layer/head 维度输出分数）。
- **归因信号**：**masked reconstruction / masked cross-entropy loss**，通常只在 completion/answer span 上监督（避免把 prompt token 的 loss 混进来）。
- **多“扩散时间步”**：用多个 mask probability（例如 `0.15,0.3,0.5,0.7,0.9`）+ 每个概率多次 Monte-Carlo masking（`mask_samples_per_prob`）来模拟不同 denoising 难度并降方差。

### 2.2 HeadIG（核心）

- **干预变量**：对每个 head 的注意力输出引入可微 gate（缩放系数）\(\alpha\)。
- **积分**：在 gate 从 baseline \(\alpha_0\) 走到 1 的路径上，对目标 \(J=-\mathcal{L}\) 做 Integrated Gradients。

### 2.3 两种积分路径（重要：论文里要正式命名）

代码中参数：`path_mode ∈ {diagonal, random_threshold}`。

- **Diagonal Path (DP)**：所有 head 同步线性 ramp，属于“确定性/同步”路径。
- **Stochastic Threshold Path (STP)**：为每个 head 采样阈值 \(u\sim\mathcal{U}[0,1]\)，在 \(\tau<u\) 时保持接近 baseline，跨过阈值后再 ramp 到 1；并对多个随机路径样本做平均（`path_samples`）。
  - 直觉：更接近 Shapley 的“随机顺序/随机路径平均”，减少对单一路径与离散轨迹的敏感性。

### 2.4 输出分数的后处理

代码支持对 IG 做后处理（例如 signed/abs/relu），论文需要说明你最终使用哪一种作为“importance”。

---

## 3. 两类有效性验证（论文实验结构）

### 3.1 验证 1：Adaptive Sparse Attention

把 head importance 分数作为 **precomputed importance** 输入，驱动 adaptive sparse attention（预算/稀疏度固定的前提下，把 sparse budget 更倾向分配给重要 head）。

对比建议：

- **standard/dense**：原模型默认注意力
- **uniform sparse**：不使用归因，均匀分配
- **adaptive sparse (HeadIG)**：使用归因分数重分配

### 3.2 验证 2：Head Masking / Pruning

在固定 \(k\)（或比例）下，做因果干预对比：

- **prune-most**：mask/prune 最重要 top-k heads（应掉点最大）
- **prune-least**：mask/prune 最不重要 bottom-k heads（应影响最小）
- **prune-random**：随机 mask/prune（作对照，通常介于二者之间）

---

## 4. 代码入口（最重要的“导航”）

### 4.1 归因（HeadIG）

建议从这些 runner 追踪：

- LLaDA 归因 runner：`models/LLaDA/attribution/loss_attribution/run_loss_attribution_all_heads.sh`
- Dream 归因 runner：`models/Dream/attribution/loss_attribution/run_loss_attribution_all_heads.sh`

它们最终调用：

- `models/*/attribution/loss_attribution/compute_loss_attribution_all_heads.py`

关键参数（与论文符号/消融对齐）：

- `--mask_probs`、`--mask_samples_per_prob`
- `--ig_steps`
- `--baseline` / `--baseline_scalar`
- `--loss_normalize`（`mean_masked` vs `sum`）
- `--ig_postprocess`（`signed`/`abs`/`relu`）
- `--path_mode`（`diagonal` vs `random_threshold`=STP）
- `--path_samples`（STP 的路径采样数）
- `--layer_start` / `--layer_end`

输出：

- `head_importance.pt`（通常在 `configs/` 下某个 `head_importance_*` 目录内）

### 4.2 Adaptive sparse attention 评测

建议从 runner 追踪：

- LLaDA：`evaluation/llada/run_eval_task.sh`
- Dream：`evaluation/dream/run_eval_task.sh`

特征：

- 支持 `MODEL_TYPES`（如 standard/sparse/adaptive），adaptive 会读取 `precomputed_importance_path`。

### 4.3 Head masking/pruning 评测

建议从 runner 追踪：

- LLaDA：`evaluation/llada/run_eval_mask_head_task.sh`
- Dream：`evaluation/dream/run_eval_mask_head_task.sh`

关键参数：

- `PRUNE_WHICH`：`most|least|random`
- `PRUNE_K` 或 `PRUNE_K_FRAC`
- `LAYER_START` / `LAYER_END`

---

## 5. 论文文件与当前状态

`paper/` 下当前主要文件：

- `introduction.tex`：已按 HeadIG + DP/STP + 两类验证口径更新
- `preliminaries.tex`：已规范符号（离散 timesteps、避免 \epsilon_\theta 混淆、避免 \mathbbm{1}）
- `method.tex`：已按 HeadIG + DP/STP + 正确离散 IG 累积写法更新，并包含 `algorithm2e` 伪代码
- `experiments.tex`：已补齐实验章节框架，表格指标用 `TBD` 占位

---

## 6. 最容易写错/写旧的点（给写论文/改论文的人）

- **不要再用旧的 “SIGA step-wise” 口径**：当前实现是 **loss-based all-heads joint IG**，并且核心升级点是 STP（原 `random_threshold`）。
- **不要把描述绑定某一模型家族**（例如只写 Dream 或只写 LLaDA 的特定 schedule/模板），应保持 DLLM 通用表述。
- **符号一致性**：
  - “时间”既可以用连续 \(t\in[0,1]\)（定义 forward masking），但生成/去噪应明确用离散 \(t_k\)；
  - logits 网络建议用 \(f_\theta\)（避免 \(\epsilon_\theta\) 的扩散噪声预测歧义）；
  - 指示函数用 \(\mathbf{1}[\cdot]\)（避免依赖 `bbm` 宏包）。
- **STP 的关键点**：是“每个 head 不同阈值 + ramp + 多路径平均”，不是简单随机置换一次。

---

## 7. 复现实验的最小流程（给未来自己/合作者）

1) 跑归因得到 `head_importance.pt`

- 选择模型与数据集（gsm8k/mmlu/humaneval/nemotron 等），设置 `PATH_MODE`（DP/STP）与 `PATH_SAMPLES`。

2) 用该分数做 adaptive sparse attention 评测

- `evaluation/*/run_eval_task.sh`，确保 `precomputed_importance_path` 指向刚生成的 `head_importance.pt`。

3) 做 head masking/pruning 的因果验证

- `evaluation/*/run_eval_mask_head_task.sh`，对比 prune-most / prune-least / prune-random，在多个 `k` 或 `k_frac` 下画曲线或表格。

