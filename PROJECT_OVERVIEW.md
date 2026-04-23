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
  - 现支持通过 `diffusion_mode` 在 `global diffusion` 与 `semi diffusion` 之间切换，并用 `block_length` 控制分块长度

### 3.2 `models/LLaDA/`

- `core/`
  - LLaDA 主模型结构与配置
- `generation/`
  - LLaDA 生成相关逻辑
  - 现支持统一的 `diffusion_mode` 开关，可在 `semi diffusion` 与 `global diffusion` 间切换
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
- `models/Dream/attribution/baseline_attribution/compute_attnlrp_head_attribution.py`
  - AttnLRP-inspired baseline
  - 使用单次 backward 的 head relevance 代理分数，目标仍与现有 diffusion-style masked CE 保持一致
- `models/Dream/attribution/baseline_attribution/compute_attarr_head_attribution.py`
  - AttAttr-style baseline
  - 对每个 head 的 attention output 元素做 element-wise IG，并对同一 head 的所有元素 attribution 取平均作为 head importance
- `models/Dream/attribution/baseline_attribution/compute_shapley_head_attribution.py`
  - CoKV-style Sliced Shapley baseline
  - 通过随机 permutation、采样 coalition size 和 complementary contribution 近似 head importance
- `models/Dream/attribution/baseline_attribution/run_*.sh`
  - Dream baseline 归因运行脚本（AttnLRP-style / AttAttr-style / Shapley）

### LLaDA

- `models/LLaDA/attribution/loss_attribution/compute_loss_attribution.py`
  - layer-wise 归因主实现
- `models/LLaDA/attribution/loss_attribution/compute_loss_attribution_all_heads.py`
  - all-heads 联合归因实现
- `models/LLaDA/attribution/loss_attribution/run_loss_attribution_all_heads.sh`
  - LLaDA 归因运行入口
- `models/LLaDA/attribution/baseline_attribution/compute_attnlrp_head_attribution.py`
  - AttnLRP-inspired baseline
  - 通过 per-head gate 的单次 backward relevance 代理计算 head importance
- `models/LLaDA/attribution/baseline_attribution/compute_attarr_head_attribution.py`
  - AttAttr-style baseline
  - 不为每个 head 单独设置 gate，而是对 pre-`o_proj` attention output 的每个元素做 IG，并按 head 聚合平均得到 importance
- `models/LLaDA/attribution/baseline_attribution/compute_shapley_head_attribution.py`
  - CoKV-style Sliced Shapley baseline
  - 将 CoKV 的 cooperative-game / complementary-contribution 思路迁移到 diffusion head attribution
- `models/LLaDA/attribution/baseline_attribution/run_*.sh`
  - LLaDA baseline 归因运行脚本（AttnLRP-style / AttAttr-style / Shapley）

### 共享归因工具

- `models/attribution_utils.py`
  - Dream / LLaDA 共享的数据读取与任务映射层
  - 负责：
    - 本地 `json/jsonl` 数据加载
    - HF 数据集加载
    - 任务名别名归一化
    - MCQ / 数学 / 代码样本格式归一化
  - 当前已覆盖统一任务名：
    - `mmlu`
    - `cmmlu`
    - `ceval-valid`
    - `gsm8k`
    - `minerva_math`
    - `gpqa_main_n_shot`
    - `humaneval`
    - `mbpp`

---

## 5. 评测与验证（`evaluation/`）

### Dream 评测目录：`evaluation/dream/`

- `eval_dream.py`
  - 统一评测入口（standard/sparse/adaptive）
  - 负责将 `diffusion_mode` / `block_length` 传入 Dream 生成配置
- `eval_mask_head_dream.py`
  - head masking/pruning 评测
- `run_eval_task.sh`
  - 常规任务批量评测脚本
  - 支持通过环境变量切换 `DIFFUSION_MODE` 与 `BLOCK_LENGTH`
  - 已统一支持 8 个优先任务：
    - MC / likelihood：`mmlu`、`cmmlu`、`ceval-valid`、`gpqa_main_n_shot`
    - 数学生成：`gsm8k`、`minerva_math`
    - 代码生成：`humaneval`、`mbpp`
- `run_eval_mask_head_task.sh`
  - pruning/masking 批量评测脚本
  - 支持 `DIFFUSION_MODE` / `BLOCK_LENGTH`，并可用 `HEAD_MASK_WARMUP_FRAC` 控制前期 warmup
  - 与 `run_eval_task.sh` 保持同一套 8 任务分类规则
- `generate_negated_importance.py`
  - importance 取负辅助（用于符号方向对照实验）

### LLaDA 评测目录：`evaluation/llada/`

- `eval_llada.py`
  - LLaDA 统一评测入口
  - 负责将 `diffusion_mode` / `block_length` 传入不同生成实现
- `eval_mask_head_llada.py`
  - LLaDA head masking/pruning 评测
- `run_eval_task.sh` / `run_eval_mask_head_task.sh`
  - 批量实验脚本
  - 支持通过环境变量切换 `DIFFUSION_MODE` 与 `BLOCK_LENGTH`
  - 已与 Dream 对齐支持相同 8 个 `lm-eval` 主链路任务

### 5.1 已统一到 `lm-eval` 主链路的 8 个任务

- 多选 / likelihood：
  - `mmlu`
  - `cmmlu`
  - `ceval-valid`
  - `gpqa_main_n_shot`
- 数学生成：
  - `gsm8k`
  - `minerva_math`
- 代码生成：
  - `humaneval`
  - `mbpp`

### 5.2 当前统一策略

- 遵循原则：`LLaDA` 能测的，`Dream` 也能测。
- 保留现有 `mmlu` 结果目录与历史行为兼容性。
- 优先复用现有 `eval_dream.py` / `eval_llada.py`，不引入新评测框架。
- 多选类任务统一使用：
  - `few-shot`
  - `batch_size=1`
  - `mc_num=1`
  - `likelihood_now_step`
  - `recompute_mask_each_call=true`
  - 对 `gpqa_main_n_shot`，优先走本地 `GPQA_DATA_PATH` fallback；若未提供且存在默认本地副本，则自动使用 `evaluation/local_data/gpqa/gpqa_main.jsonl`
- 数学任务统一走生成链路：
  - `gsm8k` 保留 chat-template 兼容
  - `minerva_math` 先按最小兼容的数学生成任务接入
  - `minerva_math` 依赖 `antlr4-python3-runtime==4.11`、`sympy`、`math_verify`
- 代码任务统一走 code-eval：
  - `humaneval`
  - `mbpp`
  - 显式传 `--confirm_run_unsafe_code`

---

## 6. 端到端实验流程（建议）

1. **归因阶段**
   - 使用 `compute_loss_attribution_all_heads.py` 或 `baseline_attribution/` 下的脚本生成 `head_importance.pt`
   - 当前 batch runner 默认保存到 `configs/aconfigs/`，并记录 metadata（dataset、seed、path_mode、mask_probs 等）
   - attribution 当前也已支持与主评测一致的 8 个统一任务名
   - 对 `gpqa_main_n_shot`，`loss_attribution`、`AttnLRP-style`、`AttAttr-style`、`Shapley` 四类 runner 现在也支持本地 `DATA_PATH`，若未显式传入且存在默认本地副本，则自动使用 `evaluation/local_data/gpqa/gpqa_main.jsonl`
   - baseline / IG 产物若只覆盖部分层，当前 `adaptive` 评测统计也支持缺层跳过，方便做局部 layer-range 实验
   - 对新增任务的兼容映射为：
     - `cmmlu / ceval-valid / gpqa_main_n_shot` -> MMLU-style prompt builder
     - `minerva_math` -> GSM8K/Math-style supervision builder
     - `mbpp` -> code-completion builder

2. **应用阶段**
   - adaptive sparse：importance -> 每层/每头 keep ratio
   - pruning/masking：按 most/least/random 策略裁剪头

3. **评测阶段**
   - 在 `evaluation/dream` 或 `evaluation/llada` 跑任务
   - 对比 `standard vs sparse vs adaptive`，并做 `most/least/random` 消融
   - 现在也可将 `global diffusion vs semi diffusion` 作为独立消融维度
   - `gpqa_main_n_shot` 若无法访问 gated HF 数据集，可直接使用本地副本 `evaluation/local_data/gpqa/gpqa_main.jsonl`
   - `run_eval_task.sh` / `run_eval_mask_head_task.sh` 当前支持 `ATTR_METHOD=headig|attnlrp|attarr|shapley`，且会优先自动解析 `configs/aconfigs/` 下最新匹配的 importance 目录

### 6.1 批量归因 Runner 约定

- 现在 8 个归因 runner（`LLaDA/Dream` x `loss_attribution / AttnLRP-style / AttAttr-style / Shapley`）都支持单卡串行跑多个任务。
- 默认任务列表由 `ATTR_DATASETS_STR` 控制，当前默认值为：
  - `mmlu,cmmlu,ceval-valid,gsm8k,minerva_math,gpqa_main_n_shot,humaneval,mbpp`
- 若不想跑全套，可手动覆盖：
  - `ATTR_DATASETS_STR="mmlu,gsm8k,humaneval"`
- 若想所有任务统一样本数，可手动覆盖：
  - `MAX_SAMPLES=80`
- 若想按任务单独覆盖默认样本数，可使用：
  - `MMLU_MAX_SAMPLES`
  - `CMMLU_MAX_SAMPLES`
  - `CEVAL_VALID_MAX_SAMPLES`
  - `GPQA_MAX_SAMPLES`
  - `GSM8K_MAX_SAMPLES`
  - `MINERVA_MATH_MAX_SAMPLES`
  - `HUMANEVAL_MAX_SAMPLES`
  - `MBPP_MAX_SAMPLES`
- 当前默认归因样本数约定为：
  - `mmlu / cmmlu / ceval-valid / gpqa_main_n_shot` -> `200`
  - `gsm8k / minerva_math / humaneval / mbpp` -> `100`
- `gpqa_main_n_shot` 在未显式传 `DATA_PATH` 时，会优先自动使用本地副本：
  - `evaluation/local_data/gpqa/gpqa_main.jsonl`
- 批量模式下单个任务失败不会中断整批；runner 会继续执行剩余任务，并在最后输出 `Batch Summary` 汇总成功/失败项与输出目录。

---

## 7. 关键概念约定（给其他大模型）

- `importance_scores`
  - 通常保存为 `{layer_idx: tensor[n_heads]}`
- `head_importance.pt`
  - 归因产物标准文件名，包含 `importance_scores + metadata`
- `baseline_attribution`
  - 放置非主 IG 路线的 baseline 实现，目前已包含 `AttnLRP-style`、`AttAttr-style` 与 `CoKV-style Shapley`
- `attarr`
  - baseline 方法之一；对每个 head 的 attention output 元素做 element-wise IG，再对该 head 的所有元素 attribution 取平均作为 importance
- `task alias / dataset alias`
  - 统一任务名归一化层，确保评测、IG、AttnLRP-style、Shapley 对同一任务名保持一致
- `mask_probs + mask_samples_per_prob`
  - diffusion 风格监督采样配置
- `coalition_sizes / sampling_number`
  - Shapley baseline 的核心近似参数：采样哪些 coalition size，以及 Monte Carlo 采样次数
- `relevance_postprocess / score_postprocess`
  - baseline 后处理方式，常见取值包括 `signed`、`relu`、`abs`
- `path_mode / path_samples`
  - IG 路径策略（如 diagonal、random_threshold）
- `min_completion_tokens`
  - 截断时保证最少 completion token，减少无效监督样本
- `diffusion_mode`
  - 生成范式开关：`global` 表示整段生成区同时扩散；`semi` 表示按 block 逐段扩散
- `block_length`
  - `semi diffusion` 下的分块长度；`global diffusion` 下通常等价于整个生成长度
- `head_mask_warmup_frac`
  - head pruning/masking 在前若干扩散步保持 dense attention 的 warmup 比例

---

## 8. 给大模型的推荐阅读顺序

若要快速理解项目，建议按以下顺序读取：

1. 本文件 `PROJECT_OVERVIEW.md`
2. `models/Dream/attribution/loss_attribution/compute_loss_attribution_all_heads.py`
3. `models/Dream/attribution/loss_attribution/compute_loss_attribution.py`
4. `models/Dream/attribution/baseline_attribution/compute_attnlrp_head_attribution.py`
5. `models/Dream/attribution/baseline_attribution/compute_attarr_head_attribution.py`
6. `models/Dream/attribution/baseline_attribution/compute_shapley_head_attribution.py`
7. `models/Dream/generation_utils/generation_utils_dream.py`
8. `evaluation/dream/eval_dream.py`
9. `evaluation/dream/run_eval_task.sh` / `evaluation/dream/run_eval_mask_head_task.sh`
10. `models/LLaDA/attribution/loss_attribution/compute_loss_attribution_all_heads.py`
11. `models/LLaDA/attribution/baseline_attribution/compute_attnlrp_head_attribution.py`
12. `models/LLaDA/attribution/baseline_attribution/compute_attarr_head_attribution.py`
13. `models/LLaDA/attribution/baseline_attribution/compute_shapley_head_attribution.py`
14. `models/LLaDA/generation/generate.py`
15. `models/LLaDA/generation/sparsed_generate.py` / `models/LLaDA/generation/adaptive_sparsed_generate.py`
16. `evaluation/llada/eval_llada.py`
17. `evaluation/llada/run_eval_task.sh` / `evaluation/llada/run_eval_mask_head_task.sh`

---

## 9. 当前项目定位（一句话）

这是一个“**归因算法 + 推理策略验证 + 双模型对照评测**”的一体化实验仓库，重点不是单纯可视化，而是验证 attribution 对真实推理行为（稀疏化/剪枝）的可用性。

