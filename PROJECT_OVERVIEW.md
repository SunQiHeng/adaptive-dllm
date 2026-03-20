# adaptive-dllm 项目总览

本文件用于帮助其他大模型（或新协作者）快速理解本仓库的目标、目录结构、关键文件职责与常见工作流。

---

## 1. 项目目的

`adaptive-dllm` 是一个围绕 **Diffusion Large Language Model (dLLM)** 的注意力头归因与验证项目，核心目标是：

1. 对 attention head 做可解释性归因（基于 Integrated Gradients 变体，支持多 baseline / 多掩码采样）。
2. 将归因分数用于推理期策略，验证“重要性”是否真能提升效率或保持性能：
   - `adaptive sparse attention`
   - `attention head pruning / masking`
3. 在两个模型系列上做对照实验：
   - `Dream`
   - `LLaDA`

---

## 2. 仓库结构（顶层）

```text
adaptive-dllm/
├── models/           # 两个模型系列的核心实现 + 归因 + 稀疏化
├── evaluation/       # 评测脚本（Dream / LLaDA）
├── configs/          # 归因产物、配置、importance 文件
├── paper/            # 论文相关资料
├── prompts.json      # 提示词或数据相关配置
├── requirements.txt  # Python 依赖
└── setup_slurm.sh    # 集群环境/任务初始化脚本
```

---

## 3. `models/` 目录说明

### 3.1 `models/Dream/`

- `core/`
  - Dream 模型主干（attention、forward、生成相关底层实现）
  - 包含 sparse / adaptive sparse 对应模型实现文件
- `attribution/`
  - Dream 的头归因实现
  - 重点在 `attribution/loss_attribution/`
- `sparse/`
  - 稀疏策略、importance 到 keep ratio 的映射工具
- `generation_utils/`
  - Diffusion 生成过程辅助配置/工具

### 3.2 `models/LLaDA/`

- `core/`
  - LLaDA 主模型结构与配置
- `generation/`
  - LLaDA 生成相关逻辑
- `attribution/`
  - LLaDA 归因实现（包含 layer-wise 与 all-heads 版本）
- `sparse/`
  - 稀疏/剪枝所需工具函数
- `docs/`
  - LLaDA 相关说明文档

---

## 4. 归因核心文件（重点）

### Dream

- `models/Dream/attribution/loss_attribution/compute_loss_attribution.py`
  - layer-wise IG 版本（逐层计算 head 重要性）
  - 负责：数据构造、tokenize、mask 构造、CE 目标、IG 积分
- `models/Dream/attribution/loss_attribution/compute_loss_attribution_all_heads.py`
  - all-heads joint IG（一次性对所有选层头做联合积分）
  - 负责：跨层 gate 拼接、联合积分、importance 输出
- `models/Dream/attribution/loss_attribution/run_loss_attribution_all_heads.sh`
  - Dream 归因主运行脚本

### LLaDA

- `models/LLaDA/attribution/loss_attribution/compute_loss_attribution.py`
  - layer-wise 归因主实现
- `models/LLaDA/attribution/loss_attribution/compute_loss_attribution_all_heads.py`
  - all-heads 联合归因实现
- `models/LLaDA/attribution/loss_attribution/run_loss_attribution_all_heads.sh`
  - LLaDA 归因运行入口

---

## 5. 评测与验证（`evaluation/`）

### Dream 评测目录：`evaluation/dream/`

- `eval_dream.py`
  - 统一评测入口（standard/sparse/adaptive）
- `eval_mask_head_dream.py`
  - head masking/pruning 评测
- `run_eval_task.sh`
  - 常规任务批量评测脚本
- `run_eval_mask_head_task.sh`
  - pruning/masking 批量评测脚本
- `generate_negated_importance.py`
  - importance 取负辅助（用于符号方向对照实验）

### LLaDA 评测目录：`evaluation/llada/`

- `eval_llada.py`
  - LLaDA 统一评测入口
- `eval_mask_head_llada.py`
  - LLaDA head masking/pruning 评测
- `run_eval_task.sh` / `run_eval_mask_head_task.sh`
  - 批量实验脚本

---

## 6. 端到端实验流程（建议）

1. **归因阶段**
   - 使用 `compute_loss_attribution_all_heads.py` 生成 `head_importance.pt`
   - 保存到 `configs/`，并记录 metadata（dataset、seed、path_mode、mask_probs 等）

2. **应用阶段**
   - adaptive sparse：importance -> 每层/每头 keep ratio
   - pruning/masking：按 most/least/random 策略裁剪头

3. **评测阶段**
   - 在 `evaluation/dream` 或 `evaluation/llada` 跑任务
   - 对比 `standard vs sparse vs adaptive`，并做 `most/least/random` 消融

---

## 7. 关键概念约定（给其他大模型）

- `importance_scores`
  - 通常保存为 `{layer_idx: tensor[n_heads]}`
- `head_importance.pt`
  - 归因产物标准文件名，包含 `importance_scores + metadata`
- `mask_probs + mask_samples_per_prob`
  - diffusion 风格监督采样配置
- `path_mode / path_samples`
  - IG 路径策略（如 diagonal、random_threshold）
- `min_completion_tokens`
  - 截断时保证最少 completion token，减少无效监督样本

---

## 8. 给大模型的推荐阅读顺序

若要快速理解项目，建议按以下顺序读取：

1. 本文件 `PROJECT_OVERVIEW.md`
2. `models/Dream/attribution/loss_attribution/compute_loss_attribution_all_heads.py`
3. `models/Dream/attribution/loss_attribution/compute_loss_attribution.py`
4. `evaluation/dream/eval_dream.py`
5. `models/LLaDA/attribution/loss_attribution/compute_loss_attribution_all_heads.py`
6. `evaluation/llada/eval_llada.py`
7. 对应 `run_*.sh` 脚本（了解实际实验参数）

---

## 9. 当前项目定位（一句话）

这是一个“**归因算法 + 推理策略验证 + 双模型对照评测**”的一体化实验仓库，重点不是单纯可视化，而是验证 attribution 对真实推理行为（稀疏化/剪枝）的可用性。

