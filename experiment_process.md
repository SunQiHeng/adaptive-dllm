# adaptive-dllm 实验计划与进度

最后更新：2026-06-20 15:41 HKT

本文件是实验的唯一进度入口。每次启动、完成、失败或废弃实验后，都在本文更新状态、命令、结果路径和原因，避免重复计算以及错误符号的结果进入论文。

阅读入口：后续继续实验时优先看第 8 节当前结果与缺口、第 11 节 2026-06-12 日志、第 12 节后台运行记录。2026-06-06--09 的长日志仅作为异常追溯归档，通常不需要重新读完。

## 1. 论文要验证的核心主张

1. PolyHeadIG 比现有 head attribution baseline 更准确地识别对 dLLM 性能有因果作用的 attention heads。
2. PolyHeadIG 产生的归因分数可用于 adaptive sparse attention，在相同计算预算下更好地保持任务性能。
3. 相比 diagonal IG，随机阈值多路径积分能提高归因排名的稳定性与下游效果，额外计算开销可控。
4. 跨任务迁移与多任务平均归因作为增强证据，不作为正文核心主张的必要前置条件。

## 2. 状态标记

- `[DONE]`：结果和协议均已检查，可进入论文候选结果。
- `[AUDIT]`：已有结果，但符号、指标或协议需要检查，暂不能进入最终表格。
- `[RUNNING]`：后台运行中；必须记录 PID、GPU、日志与输出目录。
- `[TODO-P0]`：正文必需，最高优先级。
- `[TODO-P1]`：附录或正文增强实验。
- `[TODO-P2]`：时间充足时再做。
- `[BLOCKED]`：有明确阻塞原因。
- `[DROP]`：决定不再运行，并记录原因。

## 3. 不可混淆的实验约定

### 3.1 归因分数方向

下游 adaptive sparse 和 prune-most 都要求统一语义：**数值越大，head 越重要**。

| 方法 | 原始分数定义/代码语义 | 下游是否取反 | 推荐设置 |
|---|---|---:|---|
| signed HeadIG / PolyHeadIG | 对 CE loss 的 IG；有帮助的 head 往往得到更负的原始分数 | 是 | `USE_NEGATED=1` |
| signed AttAttr | 同样沿 loss attribution 方向，当前推断应取反 | 待小实验确认 | 先跑 `USE_NEGATED_MODES_STR="0,1"` |
| AttnLRP (`relu`) | 实现中已使用 `-(dL/dalpha)*alpha` 并 ReLU，越大越重要 | 否 | `USE_NEGATED=0` |
| Shapley | utility 为 `-loss`，越大越重要 | 否 | `USE_NEGATED=0` |
| Leave-One-Out (LOO) | `L(without head) - L(full)`，越大越重要 | 否 | `USE_NEGATED=0` |

注意：

- `USE_NEGATED=1` 的含义是将 importance 文件中的原始分数整体取反，不是“使用正确分数”的通用开关。
- 当前许多 `attnlrp_neg` 和 `shapley_neg` 结果可能反向使用了分数，统一标记为 `[AUDIT]`，不能直接写入最终表。
- 正文可在一个小型 sign ablation 中展示 HeadIG 原始与取反的差异；其他方法只使用其定义正确的方向。

### 3.2 Dream 与 LLaDA 的 pruning 协议

| 模型 | 当前默认剪枝单位 | 当前默认范围/比例 | 含义 |
|---|---|---|---|
| Dream | `kv_group` | layer 0--27，`PRUNE_K_FRAC=0.05` | GQA 模型以 KV group 为有效干预单位 |
| LLaDA | attention head，`PRUNE_SCOPE=layer` | layer 0--31，每层相同比例 `PRUNE_K_FRAC=0.2` | 每层都剪相同比例的 heads |

主表中必须明确写出上述差异。Dream 5% KV-group 与 LLaDA 每层 20% head 的绝对性能下降不可直接作跨模型强弱比较。

Dream 的归因文件保存 query-head 分数；当 `mask_granularity=kv_group` 时，当前 pruning 实现先对同一个 KV group 内的 query-head 分数取平均，再按 group 排序并整体移除。Adaptive sparse 的默认 `gqa_weight_mode=kv` 同样使用 group-average 权重。

更公平的补充方式：

1. 主文按各模型的原生有效干预单位做方法内比较。
2. 附录报告多个预算曲线，而不是只报一个点。
3. 若比较 Dream 与 LLaDA，使用“相对 dense 的性能保持率”并同时报告移除的 projection dimension 或 attention FLOPs proxy。

### 3.3 任务指标

| 任务 | 主指标 | 注意事项 |
|---|---|---|
| MMLU / CMMLU / CEval / GPQA | accuracy | likelihood/MC 评测 |
| GSM8K | exact match, `flexible-extract` | 使用提取后的最终答案；不要用 `strict-match` 填主表，strict 会因格式要求显著低估 Dream/LLaDA |
| Minerva Math | `math_verify` | Dream 的 `exact_match` 当前恒为 0，不能作为主指标 |
| HumanEval / MBPP | pass@1 | 使用标准 functional evaluator |

## 4. 正文核心实验设计

### 4.1 Experimental setup（约 0.4 页）

- 模型：Dream-v0-Instruct-7B、LLaDA-1.5。
- 方法：AttnLRP、Shapley、LOO、PolyHeadIG；AttAttr 放附录或消融。
- 任务：完整实验范围始终包括 MMLU、CMMLU、CEval、GPQA、GSM8K、Minerva Math、HumanEval 和 MBPP。正文先保留完整核心结果，最终页数与展示子集由作者在结果齐全后调整。
- 统一说明 attribution sample 数、mask probabilities、mask samples、IG steps、path samples、seed 和 score direction。

### 4.2 主实验 A：因果有效性，head masking/pruning（约 0.7 页）

目标：重要性排序必须能预测实际移除 head 后的性能变化。

- 表：固定代表预算下各方法的 prune-most、prune-least 与 causal gap：
  - `gap = metric(prune-least) - metric(prune-most)`，越大越好。
  - 同时报 dense，并用 prune-most 与 prune-least 的差异衡量排序的因果有效性。
- 完整运行与正文结果任务：MMLU、CMMLU、CEval、GPQA、GSM8K、Minerva Math、HumanEval、MBPP。最终若需要压缩版面，由作者在结果齐全后调整。
- `[TODO-P0]` 审计正确符号后的 PolyHeadIG、AttnLRP、Shapley 现有结果。
- `[TODO-P0]` 加入 LOO。
- `[DROP-CURRENT]` 当前 pruning 验证不运行 random，只运行 most/least。

### 4.3 主实验 B：adaptive sparse inference（约 0.6 页）

目标：在匹配 keep ratio / FLOPs proxy 下比较性能保持能力。

- 表：dense、AttnLRP、Shapley、LOO、PolyHeadIG 的完整任务结果；uniform sparse 在协议确认后补入。
- 固定当前实现采用的 adaptive sparsity setting 做主要比较；若已有可靠 budget sweep，再补充效率曲线。
- 主要指标：
  - 任务性能；
  - 相对 dense 的性能保持率；
  - 实际 latency/token 或 attention FLOPs proxy。
- `[AUDIT]` 当前 `experiments.tex` 中 AttnLRP/Shapley 多数来自 negated 结果，需要按 sign contract 重跑或换用正确方向结果。
- `[TODO-P0]` 补 dense 与 uniform sparse baseline。
- `[TODO-P0]` 加入 LOO。

### 4.4 主实验 C：PolyHeadIG 路径消融与效率/稳定性（约 0.6 页）

比较：

- Diagonal IG：`PATH_MODE=diagonal, PATH_SAMPLES=1`。
- 单随机路径：`PATH_MODE=random_threshold, PATH_SAMPLES=1`。
- PolyHeadIG：`PATH_MODE=random_threshold, PATH_SAMPLES=P`，建议 `P in {4, 8}`。

固定 attribution data、mask samples、IG steps 和所有下游协议。

报告：

- attribution wall-clock、forward/backward 次数和峰值显存；
- 不同 seeds 之间的 Spearman、top-k overlap；
- 固定 pruning budget 的 causal gap 或 adaptive performance。

`[TODO-P0]` 正文放一张 DP / STP / PolyHeadIG 的稳定性、成本与下游结果表。宽泛的 `P/K/M` 网格不是当前核心任务。

## 5. 附录增强实验设计

### 5.1 额外预算点与协议明细 `[TODO-P1]`

- 完整八任务核心结果放正文；附录只补额外预算点、per-task runtime/compute 和协议明细。
- Minerva Math 使用 `math_verify`。
- 报告每个任务实际协议，避免把不同 budget 当成同一条件。

### 5.2 跨任务迁移矩阵 `[TODO-P1]`

- 行：产生 attribution 的任务或任务组。
- 列：下游 evaluation task。
- 单元格：固定预算下的性能保持率或 causal gap。
- 优先做三个任务组：
  - knowledge：MMLU / CMMLU / CEval / GPQA；
  - math：GSM8K / Minerva Math；
  - code：HumanEval / MBPP。
- 用“同任务、同领域跨任务、跨领域”三类均值总结，避免矩阵过大。

### 5.3 多任务平均归因的通用性 `[TODO-P1]`

- 对若干 attribution tasks 的 head scores 先按层或全局标准化，再平均。
- 与单任务 attribution 比较跨任务 macro average。
- 必须先统一 score direction；不能直接平均不同语义的原始分数。

### 5.4 符号、后处理和干预单位消融 `[TODO-P2]`

- signed HeadIG：original vs negated。
- signed vs absolute vs ReLU。
- Dream：KV-group vs head 级干预（若 head 级实现有效）。
- LLaDA：per-layer vs global pruning。

这组实验用于证明结果不是符号选择或特定 pruning convention 的偶然产物。

### 5.5 其他增强实验

- `[TODO-P2]` attribution sample efficiency：样本数 `M in {10, 20, 50, 100}` 与排名/下游稳定性。
- `[TODO-P1]` path ablation seed robustness：DP / STP / PolyHeadIG 至少 3 个 attribution seeds，报告 Spearman、top-k overlap 和下游均值/方差。
- `[TODO-P2]` layer localization：重要 heads 的层分布及两个模型之间的一致性。
- `[TODO-P2]` diffusion-step transfer：用某些 mask probabilities 归因，测试对其他生成阶段的稀疏策略是否仍有效。
- `[TODO-P2]` failure cases：展示 PolyHeadIG 不优于 baseline 的任务或 budget，并分析原因。

## 6. Baseline 状态

| Baseline | Dream | LLaDA | 下游自动解析 | 状态 |
|---|---:|---:|---:|---|
| AttnLRP | 已有 | 已有 | 已有 | `[AUDIT]` negated 结果方向可能错误 |
| Shapley | 已有 | 已有 | 已有 | `[AUDIT]` negated 结果方向可能错误 |
| AttAttr | 已有 | 已有 | 已有 | `[AUDIT]` 先确认 signed 方向 |
| Leave-One-Out | 已实现 | 已实现 | 已加入 `ATTR_METHOD=loo` | `[DONE]` 双模型 smoke test；`[TODO-P0]` 正式归因 |

LOO 的精确定义：

```text
importance(h) = L(model without head h) - L(full model)
```

正值越大表示移除该 head 后 loss 增长越多，因此越重要。精确 LOO 需要 `number_of_heads + 1` 次完整 utility evaluation，成本很高；正式实验可使用较小但固定的 attribution subset，并在论文中如实报告。

LOO 入口：

```text
models/Dream/attribution/baseline_attribution/run_leave_one_out_head_attribution.sh
models/LLaDA/attribution/baseline_attribution/run_leave_one_out_head_attribution.sh
```

## 7. Minerva Math 诊断与决定

### 已确认事实

- 早期失败原因之一是缺少 `antlr4`、`sympy` 和 `math_verify`；当前 `adaptive-dllm` 环境已可导入这些依赖。
- Minerva Math 已经跑出不少结果，不是完全没有运行。
- Dream 的 `limit=200` 会作用于 7 个 subtasks，实际约产生 1400 个样本；加上 1024 diffusion steps，单个 mask setting 可耗时约 2.4 天。
- Dream 的 `exact_match` 当前恒为 0，应该使用 `math_verify`。
- 部分 random mask 运行被 `SIGTERM` 中断，因此网格不完整。
- LLaDA standard Minerva 已写出结果，但历史日志末尾出现过 `run_eval_task.sh: line 386: --limit: command not found`。当前脚本第 386 行已是正常的 `fi`，说明该 shell 拼接错误来自旧版本；已有结果可读取，不需要因此重跑完整评测，但正式使用前仍建议小 `limit` 复测。

### 已有 Minerva 结果快照（仅用于盘点，非全部可进论文）

Dream adaptive，`math_verify`：

- AttnLRP negated：0.36714 `[AUDIT: sign]`
- HeadIG original：0.37786 `[AUDIT: compare with correct negated direction]`
- HeadIG negated：0.37500 `[AUDIT]`
- Shapley negated：0.37571 `[AUDIT: sign]`

Dream mask，`math_verify`：

- AttnLRP：least 0.34357，most 0.24786，random 缺失。
- HeadIG：least 0.25071，most 0.26857，random 0.35571。
- Shapley：least 0.34286，most 0.36643，random 缺失。

LLaDA adaptive，`exact_match / math_verify`：

- AttnLRP negated：0.30071 / 0.29143 `[AUDIT: sign]`
- HeadIG original：0.29714 / 0.29286
- HeadIG negated：0.30786 / 0.28786
- Shapley negated：0.30000 / 0.29857 `[AUDIT: sign]`

LLaDA mask，`exact_match / math_verify`：

- HeadIG least：0.23214 / 0.23929
- HeadIG most：0.22571 / 0.23071
- HeadIG random：0.27000 / 0.27857
- standard：0.28571 / 0.28571

### 决定

- `[DONE]` 当前不重跑完整 Minerva 网格。
- `[TODO-P0]` 先跑小 `limit` smoke test，确认 Dream 使用 `math_verify`、LLaDA runner 不再出现尾部 shell 错误。
- `[TODO-P1]` 仅补正确 score direction 下正文/附录真正缺失的组合。
- `[DROP-CURRENT]` 不补 random mask；Minerva 后续只补 most/least 与 adaptive 的必要缺口。

## 8. 当前结果与缺口

### 可复用资产

- 两个模型上，HeadIG / AttnLRP / Shapley 的 attribution 文件已覆盖大多数任务。
- 两个模型上已有大量 adaptive 与 most/least/random mask 结果。
- GPQA 本地数据 fallback 已接入。
- 已有 stability 分析脚本，可复用 Spearman、cosine、CV 和 top-k consistency 指标。

### 2026-06-17 当前结果快照

本节记录截至 2026-06-18 00:39 HKT 已经落盘并更新到 `experiments.tex` 的主表结果。单位均为百分数；Macro average 只按当前可用的非 Minerva 条目计算，缺失任务补齐前不能作为最终完整平均。2026-06-17 13:54 HKT 之后启动的多任务 path ablation 已开始产出结果，但正文消融表尚未重排，先在第 11 节记录完整进度。

LLaDA adaptive sparse：

| 任务 | Dense | AttnLRP | Shapley | LOO | PolyHeadIG |
|---|---:|---:|---:|---:|---:|
| MMLU | 65.09 | 62.24 | 62.54 | 61.58 | 63.86 |
| CMMLU | 67.24 | 60.41 | 63.66 | 62.84 | 64.14 |
| CEval | 66.05 | 59.81 | 63.22 | TBD | 62.85 |
| GPQA | 26.50 | 23.00 | 25.00 | 24.00 | 22.00 |
| GSM8K | 80.00 | 59.60 | 60.00 | 64.00 | 70.50 |
| HumanEval | 37.20 | 38.41 | 37.80 | 37.20 | 38.70 |
| MBPP | 37.00 | 37.00 | 37.00 | 37.00 | 37.00 |
| Macro average (available) | 54.15 | 48.64 | 49.89 | 47.77 | 51.29 |

Dream adaptive sparse：

| 任务 | Dense | AttnLRP | Shapley | LOO | PolyHeadIG |
|---|---:|---:|---:|---:|---:|
| MMLU | FAILED | 41.93 | 41.58 | skipped | 43.11 |
| CMMLU | 34.59 | 34.58 | 35.15 | 34.81 | 35.26 |
| CEval | 34.03 | 33.43 | 34.62 | not started | 35.07 |
| GPQA | 29.50 | 24.00 | 19.50 | 21.50 | 23.50 |
| GSM8K | 82.50 | 66.00 | 65.50 | 65.00 | 67.00 |
| HumanEval | 39.63 | 39.63 | 39.63 | 39.63 | 39.63 |
| MBPP | 54.50 | 54.50 | 54.50 | 54.50 | 54.50 |
| Macro average (available) | 45.79 | 42.01 | 41.50 | 43.09 | 42.58 |

LLaDA exact-LOO pruning causal gap，`gap = score(prune-least) - score(prune-most)`：

| 任务 | Prune-most | Prune-least | Gap |
|---|---:|---:|---:|
| MMLU | 52.46 | 50.22 | -2.24 |
| CMMLU | 49.74 | 56.64 | 6.90 |
| GPQA | 33.50 | 30.00 | -3.50 |
| GSM8K | 52.50 | 35.00 | -17.50 |
| HumanEval | 37.80 | 37.20 | -0.60 |
| MBPP | 37.00 | 36.50 | -0.50 |
| Macro average (available) | -- | -- | -2.91 |

Dream exact-LOO pruning causal gap，`gap = score(prune-least) - score(prune-most)`：

| 任务 | Prune-most | Prune-least | Gap |
|---|---:|---:|---:|
| CMMLU | 27.54 | 24.93 | -2.61 |
| GPQA | 28.00 | 25.00 | -3.00 |
| GSM8K | 21.00 | 76.50 | 55.50 |
| HumanEval | 38.41 | 39.63 | 1.22 |
| MBPP | AUDIT-invalid 0.00 | AUDIT-invalid 0.00 | TBD |
| Macro average (available) | -- | -- | 12.78 |

结果路径约定：

- LLaDA dense：`evaluation/llada/llada_1_5_results/standard/*adaptive_table_fill_missing_20260606*/results_*.json`
- Dream dense：`evaluation/dream/results/standard/*adaptive_table_fill_missing_20260606*/results.json/__data__qh_models__Dream-v0-Instruct-7B/results_*.json`
- LLaDA LOO adaptive：`evaluation/llada/llada_1_5_results/adaptive/*formal_loo_llada_20260606*/results_*.json`
- LLaDA LOO pruning：`evaluation/llada/llada_1_5_results/mask_head/*formal_loo_llada_20260606*/prune_*/*/results_*.json`
- Dream LOO adaptive：`evaluation/dream/results/adaptive/*formal_loo_dream_gpu1_20260607*/results.json/__data__qh_models__Dream-v0-Instruct-7B/results_*.json`
- Dream LOO pruning：`evaluation/dream/results/mask_head/*formal_loo_dream_gpu1_20260607*/prune_*/*/results.json/__data__qh_models__Dream-v0-Instruct-7B/results_*.json`
- Dream recovery：`evaluation/dream/results/adaptive/*dream_recovery_gpu3_20260609*/results.json/__data__qh_models__Dream-v0-Instruct-7B/results_*.json`；`evaluation/dream/results/standard/*dream_recovery_gpu3_20260609*/results.json/__data__qh_models__Dream-v0-Instruct-7B/results_*.json`

当前失败与阻塞：

- `[BLOCKED]` LLaDA LOO CEval attribution 失败：离线环境下 `ceval/ceval-exam` 没有可用 configs。日志：`logs/loo_core/formal_loo_llada_20260606/ceval_valid/attribution.log`。
- `[BLOCKED]` Dream dense MMLU 失败：本地 `hails/mmlu_no_train` cache 缺 `marketing` config，离线环境无法补下载。日志：`logs/adaptive_table_fill/adaptive_table_fill_missing_20260606/dense_dream_mmlu.log`。
- `[DONE]` LLaDA LOO MBPP pruning-most/least 已完成：most 37.00，least 36.50，gap -0.50。日志：`logs/loo_core/formal_loo_llada_20260606/mbpp/pruning.log`。
- `[DONE/FAILED]` Dream table-fill 队列已结束。新增有效结果：Dream Shapley MBPP 54.50（postprocess-fixed full run）。失败项：Dream dense MMLU（本地 MMLU cache 缺 config）、Dream AttnLRP MBPP（旧协议 0.00 后触发旧 shell syntax error）、Dream HeadIG MBPP（GPU0 OOM）。
- `[FAILED/AUDIT-invalid]` Dream exact LOO GPU1 子队列在 MBPP pruning-most/least 后失败；两个 MBPP pruning 结果均为旧 MBPP 后处理协议下的 0.00，不能用于论文表。当前 `evaluation/dream/run_eval_task.sh` 与 `evaluation/dream/run_eval_mask_head_task.sh` 的 `bash -n` 均通过，语法错误不是现行文件状态。
- `[DONE]` Dream protocol audit 已完成，run tag `dream_protocol_audit_20260608_gpu4`。GPQA dense `MC_NUM=8` 为 29.50，GPQA LOO adaptive `MC_NUM=8` 为 21.50；MBPP dense postprocess smoke `limit=20` 为 55.00。
- `[DONE]` Dream recovery 队列已完成，run tag `dream_recovery_gpu3_20260609`。GPQA `MC_NUM=8`：AttnLRP 24.00，Shapley 19.50，PolyHeadIG 21.50；结合 audit 中的 dense 29.50 与 LOO 21.50，Dream GPQA 行已更新为全方法 high-MC。Dream MBPP postprocess-fixed full run：dense/AttnLRP/Shapley/LOO/PolyHeadIG 均为 54.50。日志：`logs/recovery/dream_recovery_gpu3_20260609/pipeline.log`；状态：`logs/recovery/dream_recovery_gpu3_20260609/status.tsv`。
- `[DONE/FAILED]` Dream LOO 队列 `formal_loo_dream_20260606` 已结束：CEval attribution 失败、MMLU adaptive 失败，其余主要 adaptive/pruning 阶段完成。状态：`logs/loo_core/formal_loo_dream_20260606/status.tsv`。
- `[DONE]` GSM8K metric audit：当前结果 JSON 同时包含 `exact_match,strict-match` 和 `exact_match,flexible-extract`。主表应使用 `flexible-extract`，因此 Dream dense GSM8K 是 82.50，不是 16.00；LLaDA dense GSM8K 是 80.00，不是 63.00；LLaDA LOO adaptive GSM8K 是 64.00，不是 40.50。
- `[DONE]` Dream MBPP 0.00 原因定位：`eval_dream.py` 的 code extraction 只在 context 含 `def ` 时触发，HumanEval 会触发，但 MBPP prompt 是自然语言 + `[BEGIN]`，导致 Dream Instruct 输出的解释/markdown code fence 被原样送入 unsafe-code evaluator，200 个样本全 fail。已修复为对 `[BEGIN]` / `Your code should pass these tests` 也触发代码块抽取。旧 Dream MBPP dense/LOO/adaptive/pruning 结果均标记为 `AUDIT-invalid`，不能进论文表。
- `[DONE/AUDIT-RESOLVED]` Dream GPQA dense 26.00 低于多个 sparse/adaptive 的异常已用 `MC_NUM=8` 全方法复核。新结果为 dense 29.50、AttnLRP 24.00、Shapley 19.50、LOO 21.50、PolyHeadIG 21.50，说明原先 “sparse 明显强于 dense” 主要来自 `mc_num=1` 随机 likelihood 估计偏差。主表已改用 high-MC 行。

### 当前最危险的问题

1. `[TODO-P0]` 结果目录名中的 `_neg` 被跨方法统一使用，但跨方法分数定义不同。
2. `[TODO-P0]` 当前正文表已补入更多 dense、GPQA high-MC、Dream LOO、HumanEval、Dream MBPP fixed 与大部分 mask-main pruning 结果，但 Dream MMLU dense、Dream/LLaDA CEval LOO、Dream PolyHeadIG MBPP pruning、以及部分异常解释仍未完成。
3. `[TODO-P0]` Dream 与 LLaDA pruning budget/单位不同，论文必须明确说明。
4. `[DECISION]` 当前 pruning 验证只运行 most/least；random pruning 不进入本轮任务。

## 9. 推荐执行顺序

1. `[DONE]` LOO 单层、单样本双模型 GPU smoke test；下一步确定正式 attribution subset 大小。
2. `[SKIP]` 不再以小规模 MMLU sign audit 作为正式实验前置条件，按代码级 sign contract 启动正式任务。
3. `[TODO-P0]` 盘点并补全八任务结果清单；MMLU、GSM8K、HumanEval 只是可能的正文展示子集。
4. `[TODO-P0]` 补 dense、uniform sparse 与缺失的正确方向结果。
5. `[TODO-P0]` 跑 diagonal / single random path / PolyHeadIG 的稳定性、成本与下游消融。
6. `[TODO-P1]` 补跨任务迁移和多任务平均归因。
7. `[TODO-P1]` 最后只补必要的 Minerva 缺口。

## 10. GPU 与后台运行规则

- 每次启动前运行 `nvidia-smi`，只使用空闲 GPU。
- 历史默认最多同时占用 3 张 GPU；2026-06-12 用户已允许临时多占两张空闲卡，因此当前可超过 3 张，但仍只使用空闲或低负载 GPU。
- 所有长实验必须后台运行，并记录 PID、GPU、日志和输出目录。
- 先 smoke test，再启动完整实验。

推荐使用 detached `tmux`，它在 SSH 断开后仍会继续运行：

```bash
mkdir -p logs
tmux new-session -d -s adaptive-llada-loo-smoke \
  "cd /home/qiheng/Projects/adaptive-dllm && \
   env GPU_ID=0 ATTR_DATASETS_STR=mmlu MAX_SAMPLES=1 LAYER_START=0 LAYER_END=0 \
   bash models/LLaDA/attribution/baseline_attribution/run_leave_one_out_head_attribution.sh \
   > logs/llada_loo_smoke_gpu0.log 2>&1"
```

查看状态：

```bash
nvidia-smi
ps -u qiheng -o pid,ppid,stat,etime,cmd
tail -n 80 logs/llada_loo_smoke_gpu0.log
tmux list-sessions
```

正式 exact-LOO 核心实验使用可断点续跑 runner：

```bash
MODEL_FAMILY=llada GPU_ID=0 RUN_TAG=formal_loo_llada_20260606 \
DATASETS_STR=mmlu,cmmlu,ceval-valid,gpqa_main_n_shot,gsm8k,humaneval,mbpp \
bash experiment_scripts/run_loo_core_pipeline.sh
```

runner 对每个数据集依次执行：

1. exact LOO attribution；
2. 使用原始 LOO 分数的 adaptive evaluation，即 `USE_NEGATED=0`；
3. prune-most 与 prune-least，不运行 random。

每个阶段完成后写入 `.done` marker；相同 `RUN_TAG` 再次执行时会跳过已完成阶段。状态入口为：

```text
logs/loo_core/<RUN_TAG>/status.tsv
```

Minerva Math 不进入默认长队列，按第 7 节单独处理。

## 11. 进度日志

### 2026-06-06

- `[DONE]` 审计 `experiments.tex`、核心归因/评测 runner 与现有结果目录。
- `[DONE]` 完成 Minerva Math 初步诊断；决定先 smoke test，不立即重跑完整网格。
- `[DONE]` 明确方法级 sign contract；AttnLRP/Shapley/LOO 不应沿用 HeadIG 的统一 negation。
- `[DONE]` 实现 Dream 与 LLaDA 的 exact LOO baseline。
- `[DONE]` 将 `ATTR_METHOD=loo` 接入两个模型的 adaptive 与 mask runner 自动解析。
- `[DONE]` LOO Python 编译、CLI help、相关 shell syntax 初检通过。
- `[DONE]` LLaDA LOO 单层单样本 GPU smoke test：第 0 层 32 heads，33 次 utility evaluations，成功写出 `head_importance.pt`。
- `[DONE]` Dream LOO 单层单样本 GPU smoke test：第 0 层 28 query heads，29 次 utility evaluations，成功写出 `head_importance.pt`。
- `[DECISION]` 完整八任务仍是实验运行范围；三个代表任务只用于可能的正文压缩展示，不能替代完整结果。
- `[DECISION]` 当前 pruning 批处理固定为 most/least，不运行 random。
- `[DECISION]` 页数不再限制核心实验范围；正文优先保留完整 adaptive、most/least causal validation、以及 DP/STP/PolyHeadIG 消融。层定位、宽泛超参数扫描和方法间纯排名比较不作为正文核心实验。
- `[DONE]` 新增 `experiment_scripts/run_loo_core_pipeline.sh`，串联 exact LOO、adaptive、prune-most/prune-least，并支持逐数据集断点续跑和失败记录。
- `[RUNNING]` 在 GPU 0 启动 LLaDA 七个非 Minerva 核心任务队列；首项为 MMLU，完整 32 层、20 attribution samples。
- 启动后 GPU 快照：GPU 0 使用约 18.5 GB、利用率 99%；GPU 1--5 保持原有任务负载，没有新增占用。

### 2026-06-07

- `[DONE]` LLaDA dense 非 Minerva 七任务已完成：MMLU 65.09、CMMLU 67.24、CEval 66.05、GPQA 26.50、GSM8K 80.00、HumanEval 37.20、MBPP 37.00。
- `[DONE]` Dream dense 已完成五项：CMMLU 34.59、CEval 34.03、GPQA 26.00、GSM8K 82.50、HumanEval 39.63。
- `[RUNNING]` Dream dense MBPP 仍在跑；完成后 table-fill runner 会继续 LLaDA GPQA 和 Dream GPQA/HumanEval/MBPP 的 AttnLRP/Shapley/PolyHeadIG adaptive 缺口。
- `[BLOCKED]` Dream dense MMLU 失败，原因是本地 `hails/mmlu_no_train` cache 缺 `marketing` config。
- `[DONE]` LLaDA exact LOO adaptive 已完成 MMLU 61.58、CMMLU 62.84、GPQA 24.00、GSM8K 64.00。
- `[RUNNING]` LLaDA exact LOO HumanEval adaptive 正在跑；MBPP 尚未开始。
- `[BLOCKED]` LLaDA exact LOO CEval attribution 失败，原因是本地 `ceval/ceval-exam` configs 不可用。
- `[DONE]` LLaDA exact LOO pruning 已完成 MMLU、CMMLU、GPQA、GSM8K 的 most/least。当前 causal gap 分别为 -2.24、6.90、-3.50、-17.50，除 CMMLU 外方向不理想，后续需要检查 LOO subset 稳定性或作为弱 baseline 谨慎报告。
- `[DONE]` 发现并修正 2026-06-07 21:00 版本中的 GSM8K 统计错误：此前误用了 `strict-match`，现已在 `experiments.tex` 和本文件中统一改为 `flexible-extract`。
- `[RUNNING]` 在 GPU 1 启动 Dream exact LOO 子队列 `cmmlu,gpqa_main_n_shot,gsm8k,humaneval,mbpp`，run tag 为 `formal_loo_dream_gpu1_20260607`。启动时 GPU1 约 7.1GB 显存、75% util；启动后约 22.2GB 显存、79% util。当前第一项为 CMMLU attribution。
- `[DECISION]` GPU1 这批暂不跑 Dream MMLU/CEval：MMLU dense 已因本地 `hails/mmlu_no_train` 缺 `marketing` config 失败；CEval LOO attribution 在 LLaDA 上已因 `ceval/ceval-exam` configs 不可用失败，先避免把 24 小时窗口耗在高概率失败项上。
- `[DONE]` 已将上述已落盘结果更新到 `experiments.tex`。

### 2026-06-08

- `[DONE]` table-fill 新增 LLaDA GPQA adaptive：AttnLRP 23.00、Shapley 25.00、PolyHeadIG 22.00。
- `[DONE]` table-fill 新增 Dream GPQA adaptive：AttnLRP 31.50、Shapley 31.00、PolyHeadIG 29.00。
- `[DONE]` table-fill 新增 Dream HumanEval adaptive：AttnLRP 39.63、Shapley 39.63、PolyHeadIG 39.63。
- `[DONE/AUDIT]` Dream dense MBPP 已完成，functional evaluator 输出 `pass_at_1=0.00`。该值已入草稿表，但作为 dense baseline 明显异常，后续需要复核 Dream MBPP 的生成长度、stop 规则和 evaluator 配置。
- `[RUNNING]` table-fill 当前正在跑 Dream AttnLRP MBPP，日志 `logs/adaptive_table_fill/adaptive_table_fill_missing_20260606/adaptive_dream_attnlrp_mbpp.log`；后续应继续 Dream Shapley/PolyHeadIG MBPP。
- `[DONE]` LLaDA exact LOO HumanEval adaptive 37.20；pruning most 37.80、least 37.20，gap -0.60。
- `[RUNNING]` LLaDA exact LOO MBPP attribution 已完成，adaptive 正在跑，日志 `logs/loo_core/formal_loo_llada_20260606/mbpp/adaptive.log`。
- `[DONE]` Dream exact LOO adaptive 已完成 CMMLU 34.81、GPQA 32.50、GSM8K 65.00、HumanEval 39.63、MBPP 0.00；MMLU/CEval 本轮未跑。
- `[DONE]` Dream exact LOO pruning 已完成 CMMLU、GPQA、GSM8K、HumanEval，causal gap 分别为 -2.61、-3.00、55.50、1.22。
- `[RUNNING]` Dream exact LOO MBPP pruning-most/least 正在跑；prune-most 已落盘 0.00，prune-least 未完成，日志 `logs/loo_core/formal_loo_dream_gpu1_20260607/mbpp/pruning.log`。
- `[RUNNING]` 22:17 HKT GPU 快照：GPU0 约 80.7GB/100% util，GPU1 约 69.1GB/99% util，GPU2 约 62.7GB/100% util；GPU3 空闲，GPU4/5 有其他负载。当前仍保持最多三张卡用于本项目队列。
- `[DONE]` 已将截至 22:17 HKT 已落盘且指标明确的新增结果更新到 `experiments.tex`。
- `[DONE]` 定位 Dream MBPP 0.00 的直接原因：MBPP prompt 不含 `def `，旧 `_maybe_extract_python_completion` 未触发，Dream Instruct 的解释式输出和 markdown fence 被原样执行。已修复 `evaluation/dream/eval_dream.py`，使 `[BEGIN]` / “Your code should pass these tests” 也触发代码抽取；旧 Dream MBPP 结果已从 `experiments.tex` 中撤回为 `TBD`。
- `[DONE]` 为 Dream multiple-choice runner 增加 `MC_NUM` 环境变量：`evaluation/dream/run_eval_task.sh` 和 `evaluation/dream/run_eval_mask_head_task.sh` 均不再硬编码 `mc_num=1`。
- `[RUNNING]` 22:49 HKT 在 GPU4 启动 `dream-protocol-audit-gpu4-20260608`，run tag `dream_protocol_audit_20260608_gpu4`。第一项为 GPQA dense `MC_NUM=8`，后续为 GPQA LOO adaptive `MC_NUM=8` 和 MBPP dense postprocess smoke `limit=20`。
- `[RUNNING]` 22:51 HKT GPU4 audit 已进入 GPQA dense loglikelihood 阶段；GPU4 约 43.9GB/55% util。

### 2026-06-09

- `[DONE]` Dream protocol audit 已完成。GPQA `MC_NUM=8`：dense 29.50，LOO adaptive 21.50。该结果支持“原 MC1 GPQA sparse > dense 异常主要来自随机 likelihood 估计方差”的判断；主表仍保留 MC1 inventory，最终若报告 GPQA 应用同一 `MC_NUM` 重跑全方法。
- `[DONE]` Dream MBPP postprocess-fixed dense smoke，`limit=20`，pass@1 55.00。该 smoke 进一步确认旧 Dream MBPP 0.00 是代码抽取/评测协议问题，不是模型能力问题。
- `[DONE]` Dream Shapley MBPP postprocess-fixed full run 完成，pass@1 54.50，已更新到 `experiments.tex`。该值使用正确的 MBPP code extraction，但 dense、AttnLRP、PolyHeadIG 和 LOO 仍需在同一修复后协议下补齐。
- `[FAILED]` Dream AttnLRP MBPP 在旧协议下得到 0.00 后触发旧 shell syntax error；该结果仍标记为无效，需要重跑。
- `[FAILED]` Dream PolyHeadIG/HeadIG MBPP 在 GPU0 OOM，日志显示申请 1.45 GiB 时 GPU0 仅约 281 MiB 空闲。重跑时应固定到更空闲的 GPU，并避免与 LLaDA MBPP pruning 同卡。
- `[DONE]` LLaDA LOO MBPP adaptive 完成，pass@1 37.00，已更新到 `experiments.tex`；LLaDA LOO adaptive macro 改为 47.77（加入 MBPP 后的 available macro）。
- `[RUNNING]` LLaDA LOO MBPP pruning-least 仍在运行；prune-most 已完成 37.00，least 尚未生成结果文件。进程已运行约 8.3 小时，日志：`logs/loo_core/formal_loo_llada_20260606/mbpp/pruning.log`。
- `[FAILED/AUDIT-invalid]` Dream LOO MBPP pruning-most/least 均属于旧 MBPP 后处理协议，虽然两个结果落盘为 0.00，但不能用于论文表。Dream LOO GPU1 子队列已失败退出。
- `[DONE]` 当前 `evaluation/dream/run_eval_task.sh` 与 `evaluation/dream/run_eval_mask_head_task.sh` 均通过 `bash -n`；之前的 `unexpected token done` 不是当前文件仍存在的语法错误。
- `[STATUS]` 23:17 HKT GPU 快照：GPU0 37.0GB/59% util，GPU1 25.8GB/3% util，GPU2 65.2GB/100% util，GPU3 空闲，GPU4 5.7GB/58% util，GPU5 37.1GB/57% util。
- `[DONE]` 新增 `experiment_scripts/run_dream_recovery_gpu3.sh`，用于顺序补 Dream GPQA high-MC 和 Dream MBPP 修复后缺口。
- `[DONE]` 修复 `experiment_scripts/run_adaptive_table_fill.sh` 与 `experiment_scripts/run_dream_recovery_gpu3.sh` 的 HeadIG importance 自动选择逻辑：当 `USE_NEGATED=1` 时排除已物化的 `*_neg` / `*_neg_neg` 目录，避免 raw HeadIG 被双重取反。
- `[RUNNING]` 23:25 HKT 启动 tmux `dream-recovery-gpu3-20260609`，GPU3，run tag `dream_recovery_gpu3_20260609`。首项 `gpqa_attnlrp_mc8` 已进入 loglikelihood 阶段；日志中确认 `[device] CUDA_VISIBLE_DEVICES=3 | torch.cuda.current_device()=0 | name=NVIDIA H100 NVL`。

### 2026-06-10--11

- `[DONE]` Dream recovery 队列全部完成，结束时间 2026-06-10 23:49 HKT，无失败项。
- `[DONE]` Dream GPQA high-MC 全方法结果已补齐并更新到 `experiments.tex`：dense 29.50、AttnLRP 24.00、Shapley 19.50、LOO 21.50、PolyHeadIG 21.50。该行替换旧 MC1 inventory，解决 sparse/adaptive 明显高于 dense 的异常。
- `[DONE]` Dream MBPP postprocess-fixed full run 已补齐并更新到 `experiments.tex`：dense、AttnLRP、Shapley、LOO、PolyHeadIG 均为 54.50。旧 MBPP 0.00 结果继续视为无效。
- `[DONE]` LLaDA LOO MBPP pruning-most/least 完成：most 37.00、least 36.50、gap -0.50。LLaDA LOO pruning macro 从 -3.39 更新为 -2.91。
- `[STATUS]` 2026-06-11 17:50 HKT 当前只剩 tmux `adaptive-dream-loo-wait-gpu2-20260606`；LLaDA LOO 与 Dream recovery tmux 已结束。GPU3 当前空闲。
- `[RUNNING]` 2026-06-11 18:07 HKT 在 GPU3 启动 Dream AttnLRP 重新归因 audit，tmux `dream-attnlrp-audit-gpu3-20260611`，run tag `dream_attnlrp_regen_audit_20260611_gpu3`。该队列先用新 seed 131 重新计算 GPQA 与 GSM8K 的 AttnLRP importance，再跑 2x2 adaptive 评测：GPQA attribution -> GPQA、GPQA attribution -> GSM8K、GSM8K attribution -> GSM8K、GSM8K attribution -> GPQA。GPQA eval 固定 `MC_NUM=8`，AttnLRP 固定 `USE_NEGATED=0`。
- `[AUDIT]` 本轮特别复查 Dream AttnLRP 的 GPQA/GSM8K 虚高或不稳定风险。旧 Dream GSM8K AttnLRP importance 目录名为 `gsm8k_full`，而当前 runner 默认 `GSM8K_ANSWER_MODE=final_hash`；新归因固定 `final_hash`，更贴近主表使用的最终答案 EM 目标。完成后不要自动覆盖主表，先与旧值和 dense/其他方法横向比较。
- `[RUNNING]` 2026-06-11 18:18 HKT 启动 main pruning/masking 补表队列。LLaDA 在 GPU4，tmux `mask-main-llada-gpu4-20260611-v2`；Dream 在 GPU0，tmux `mask-main-dream-gpu0-20260611-v2`。两条队列使用 `experiment_scripts/run_mask_main_fill.sh`，按 AttnLRP、Shapley、PolyHeadIG(headig) 顺序，对 MMLU、CMMLU、CEval、GPQA、GSM8K、HumanEval、MBPP 跑 prune-most / prune-least；Minerva 继续跳过。HeadIG/PolyHeadIG 使用 `USE_NEGATED_MODES_STR=1` 且显式排除已物化的 `_neg` / `_neg_neg` importance 目录；AttnLRP/Shapley 使用原始方向。
- `[NOTE]` 18:16 的初版 mask-main session 已立即停止，原因是 `safe_name` 中 `tr '/-.'` 把任务 tag 转换坏了，只留下半截 MMLU 日志，不作为正式结果使用。18:18 的 v2 run tag 已修复，状态项形如 `mask_llada_attnlrp_mmlu`。

### 2026-06-12

- `[DONE/UPDATED]` main pruning/masking 已有成对 most/least 结果已更新到 `experiments.tex` 的 `tab:mask_main`。
- `[DONE]` LLaDA AttnLRP pruning 七个非 Minerva 任务已完成，causal gaps：MMLU 19.56、CMMLU 31.53、CEval 32.02、GPQA 1.50、GSM8K 45.50、HumanEval 1.22、MBPP 0.50，available macro 18.83。整体说明 AttnLRP 在 LLaDA layer-wise 20% pruning 下能产生很强的 causal separation，尤其 knowledge/math 任务。
- `[PARTIAL]` LLaDA Shapley pruning 已完成 MMLU 4.08、CMMLU 2.01、CEval 0.97、GPQA 5.00、GSM8K 8.50，available macro 4.11；当前仍在跑 HumanEval，MBPP 尚未开始。Shapley gap 明显小于 AttnLRP，但多数已完成项为正。
- `[DONE]` Dream AttnLRP pruning 已完成六项，causal gaps：MMLU 2.24、CMMLU 5.78、CEval 4.83、GPQA 6.50、GSM8K 68.00、HumanEval 0.61，available macro 14.66；MBPP 正在跑。GSM8K 的 68.00 gap 很大，说明 AttnLRP 的 GSM8K ranking 在 pruning 评估中确实能区分关键 KV groups，但 adaptive sparse 的数值仍需结合重归因 audit 谨慎解释。
- `[DONE]` Dream LOO 正式队列 `formal_loo_dream_20260606` 已结束。新增有效 pruning gap：MMLU -15.22、MBPP -1.00；结合已有 CMMLU -2.61、GPQA -3.00、GSM8K 55.50、HumanEval 1.22，Dream LOO available macro 更新为 5.82。CEval attribution 仍失败，MMLU adaptive 失败但 pruning 已完成。
- `[DONE/FAILED]` Dream AttnLRP 重归因 audit 已结束。新 seed 131、`final_hash` GSM8K attribution 下：GPQA attribution -> GPQA(`MC_NUM=8`) 为 22.00；GPQA attribution -> GSM8K 为 63.00；GSM8K attribution -> GSM8K 为 62.50；GSM8K attribution -> GPQA(`MC_NUM=8`) 因 OOM 失败。该结果支持用户怀疑：旧 Dream AttnLRP GSM8K adaptive 66.00 可能偏高，重归因后同任务为 62.50；但由于 Shapley/PolyHeadIG 尚未用 GSM8K `final_hash` 同步重归因，暂不直接替换正文 adaptive 主表。
- `[RUNNING]` 2026-06-12 当前仍在跑：tmux `mask-main-llada-gpu4-20260611-v2`（LLaDA Shapley HumanEval 阶段）和 `mask-main-dream-gpu0-20260611-v2`（Dream AttnLRP MBPP 阶段）。GPU 快照约为 GPU0 43.1GB/100% util、GPU4 19.7GB/99% util；GPU2/3 空闲。
- `[TODO]` 后续优先：等待两个 mask-main 队列继续产出 Shapley/PolyHeadIG；Dream AttnLRP audit 的 failed cross-task GPQA 可在空卡用较低并发或单独 GPU 重跑；如果要正式替换 Dream GSM8K adaptive 行，应同步重算 Shapley/PolyHeadIG 的 GSM8K `final_hash` attribution 或在表注中明确 source 差异。
- `[DONE]` 用户解除“最多三张 GPU”的本轮限制后，19:00 HKT 启动两条额外后台队列：GPU2 跑 LLaDA PolyHeadIG/headig main pruning；GPU3 跑 Dream AttnLRP failed cross-task GPQA retry，随后接 Dream Shapley + PolyHeadIG/headig main pruning。
- `[DONE/FIXED]` 初始 GPU2 队列 `mask-main-llada-headig-gpu2-20260612` 立即失败，原因是 `evaluation/llada/run_eval_mask_head_task.sh` 在 `USE_NEGATED_MODES_STR=1` 时未定义 `SCRIPT_DIR`/`PROJECT_ROOT`，导致调用 `/generate_negated_importance.py`。已修复该 runner 的路径初始化并停掉坏队列；坏 run tag 不作为实验结果使用。
- `[RUNNING]` GPU2 已用修复后的 runner 重启为 tmux `mask-main-llada-headig-gpu2-20260612-v2`，run tag `mask_main_llada_headig_gpu2_20260612_v2`。日志确认 MMLU HeadIG negated importance 已成功生成，并进入 LLaDA MMLU prune-most。
- `[RUNNING]` GPU3 tmux `dream-audit-retry-mask-gpu3-20260612`，run tag `dream_audit_retry_mask_gpu3_20260612`。当前正在重跑 Dream AttnLRP `GSM8K final_hash attribution -> GPQA MC_NUM=8`；无论该 retry 成功或失败，脚本都会继续启动 `mask_main_dream_shapley_headig_gpu3_20260612` 的 Dream Shapley/HeadIG pruning 队列。

### 2026-06-13--15

- `[DONE/UPDATED]` 已将最新 pruning causal gap 更新到 `experiments.tex` 的 `tab:mask_main`。新增覆盖：LLaDA Shapley HumanEval/MBPP、LLaDA PolyHeadIG 全部七个非 Minerva 任务、Dream AttnLRP MBPP、Dream Shapley 全部七个非 Minerva 任务、Dream PolyHeadIG 至 HumanEval。
- `[DONE]` LLaDA PolyHeadIG/headig pruning 两条队列结果一致。正式表使用 `mask_main_llada_gpu4_20260611_v2`；GPU2 v2 作为重复确认。LLaDA PolyHeadIG gaps：MMLU 11.62、CMMLU 23.81、CEval 25.11、GPQA 3.00、GSM8K 7.50、HumanEval 0.61、MBPP 1.00，available macro 10.38。
- `[DONE/AUDIT]` LLaDA Shapley HumanEval 在 `status.tsv` 中标为 `FAILED rc=2`，但日志显示 prune-most 与 prune-least 均已完成并写出 pass@1=38.41，gap=0.00。失败来自该长进程结束时的 shell 解析错误；当前 `evaluation/llada/run_eval_mask_head_task.sh` 已通过 `bash -n`，因此该结果可用但状态需备注。
- `[DONE]` LLaDA Shapley 全部可用 gaps：MMLU 4.08、CMMLU 2.01、CEval 0.97、GPQA 5.00、GSM8K 8.50、HumanEval 0.00、MBPP 0.00，available macro 2.94。
- `[DONE]` Dream AttnLRP MBPP pruning 完成，gap=0.00；Dream AttnLRP macro 加入 MBPP 后为 12.57。
- `[DONE]` Dream Shapley pruning 全部七个非 Minerva 任务完成，gaps：MMLU -12.98、CMMLU -1.57、CEval 0.89、GPQA -6.00、GSM8K -1.00、HumanEval 1.22、MBPP 0.00，available macro -2.78。该 baseline 在 Dream pruning 上整体方向很弱/反向，是需要讨论的负结果。
- `[PARTIAL]` Dream PolyHeadIG pruning 当前完成到 HumanEval，gaps：MMLU -12.37、CMMLU 2.80、CEval -5.50、GPQA -5.00、GSM8K 71.50、HumanEval 1.83，available macro 8.88。MBPP 正在 GPU0 跑 `prune_most`，尚未完成。
- `[DONE]` Dream AttnLRP retry `GSM8K final_hash attribution -> GPQA MC_NUM=8` 完成，结果为 20.00。结合此前 `GPQA attr -> GPQA` 22.00、`GPQA attr -> GSM8K` 63.00、`GSM8K attr -> GSM8K` 62.50，说明新 seed/final-hash 归因下 Dream AttnLRP adaptive 不存在旧表里 GPQA/GSM8K 虚高问题；旧 GSM8K 66.00 可视为偏高但不离谱，最终是否替换需同步重跑 Shapley/PolyHeadIG 的 final-hash 版本。
- `[FAILED/AUDIT]` Dream GPU3 duplicate queue `mask_main_dream_shapley_headig_gpu3_20260612` 在 HeadIG HumanEval/MBPP 上 OOM，原因是同卡已有约 76GB 进程占用。该失败不影响 GPU0 正式队列；GPU0 后续已完成 Dream HeadIG HumanEval，MBPP 仍在跑。
- `[RUNNING]` 截至 2026-06-15 12:50 HKT，只剩 tmux `mask-main-dream-gpu0-20260611-v2`。当前阶段为 Dream PolyHeadIG/headig MBPP `prune_most`，日志：`logs/mask_main/mask_main_dream_gpu0_20260611_v2/mask_dream_headig_mbpp.log`。GPU0 约 46GB/100% util。
- `[RUNNING]` 2026-06-15 13:07 HKT 新增正文 path-design ablation 队列。新增 `experiment_scripts/run_path_ablation_mmlu.sh`，显式使用每个 path 的 importance 路径，避免被主表 runner 的 latest-importance 逻辑混用。两条队列均以 MMLU 为代表任务，`MAX_SAMPLES=40`、`IG_STEPS=8`、`MASK_PROBS=0.15,0.3,0.5,0.7,0.9`、`MASK_SAMPLES_PER_PROB=2`、`IG_POSTPROCESS=signed`，并对 raw HeadIG 分数使用 `USE_NEGATED_MODES_STR=1` 做 most/least pruning。GPU4 跑 LLaDA，tmux `path-ablation-llada-mmlu-gpu4-20260615`，日志目录 `logs/path_ablation/path_ablation_llada_mmlu_20260615/`；GPU5 跑 Dream，tmux `path-ablation-dream-mmlu-gpu5-20260615`，日志目录 `logs/path_ablation/path_ablation_dream_mmlu_20260615/`。每条队列顺序为 DP(`diagonal`, P=1) -> STP(`random_threshold`, P=1) -> PolyHeadIG(`random_threshold`, P=4)，每个 path 归因完成后立即跑 prune-most/prune-least；本轮暂不跑 adaptive，待 causal gap 落盘后再决定是否补。

### 2026-06-16

- `[DONE/UPDATED]` Dream main pruning/masking 队列 `mask-main-dream-gpu0-20260611-v2` 已于 2026-06-16 04:24 HKT 完成最后一项 Dream PolyHeadIG/headig MBPP。MBPP prune-most=54.00、prune-least=54.50，causal gap=0.50。`experiments.tex` 已更新 Dream PolyHeadIG MBPP 和 macro：Dream PolyHeadIG available macro 从 8.88 调整为 7.68。
- `[DONE/UPDATED]` MMLU-only path-design ablation 已完成并更新到 `experiments.tex` 的 `tab:path_ablation`。LLaDA gaps：DP 19.17、STP 20.92、PolyHeadIG 7.59；Dream gaps：DP -4.39、STP 4.96、PolyHeadIG -6.32。Attribution time：LLaDA DP 4.78min、STP 4.60min、PolyHeadIG 16.48min；Dream DP 8.58min、STP 8.25min、PolyHeadIG 32.38min。
- `[AUDIT]` Path ablation 结果目前只是 `MMLU, M=40, seed=123` 的单 seed causal-pruning inventory，不能直接写成“PolyHeadIG 在 path 消融中稳定最优”。尤其 Dream PolyHeadIG 的 MMLU gap 为负，LLaDA STP/DP 也高于 PolyHeadIG。后续若要保留这块正文实验，建议立刻补 adaptive-sparse 列、第二 seed 稳定性、以及至少一个非 MMLU 任务，确认是 MMLU/小样本偶然还是设计本身问题。
- `[RUNNING/FAILED]` 2026-06-16 15:01 HKT 启动 path ablation adaptive-sparse 补列。LLaDA 在 GPU1，tmux `path-ablation-llada-mmlu-adaptive-gpu1-20260616`，DP/STP adaptive 已完成，PolyHeadIG adaptive 仍在跑；Dream 在 GPU5 的 `path-ablation-dream-mmlu-adaptive-gpu5-20260616` 已结束但 DP/STP/Poly 三项均失败，原因是本地 MMLU cache 缺 `marketing` config。
- `[WATCH]` 2026-06-16 16:20 HKT 巡检时 LLaDA adaptive 进程仍在模型加载/运行；Dream adaptive 后续确认是数据 cache 问题而非 GPU 卡住。下次若要补 Dream path adaptive，需要先修复/补全本地 MMLU cache 或改用可用 subject 子集。
- `[DECISION]` 2026-06-16 用户明确：当前 pruning 协议中 `least` 应解释为“分数最低/最有害的 heads 或 groups”，目标是剪掉有害单元以尽可能保留甚至提升分数；不采用 `|score|` 最小作为正文口径。后续分析先聚焦 main pruning 中 PolyHeadIG 为什么不如 AttnLRP 稳，而不是继续扩大 path ablation。
- `[AUDIT]` 初步分数分布诊断：LLaDA path ablation 中 DP/STP/Poly raw 排序相关较高（DP-Poly Spearman≈0.858，STP-Poly≈0.903）；Dream 在 KV-group 后排序相关较低（DP-Poly Spearman≈0.284，lowest 5% group overlap=0）。该诊断只说明不同 path 的排序差异，不改变 main pruning 的 `most/least` 定义。
- `[AUDIT]` 已撤回 `least_abs` 代码路径和脚本默认值，恢复正文协议 `most,least`。`evaluation/llada/run_eval_mask_head_task.sh`、`evaluation/dream/run_eval_mask_head_task.sh`、`models/LLaDA/core/mask_head_modeling.py`、`models/Dream/core/mask_head_modeling_dream.py`、`experiment_scripts/run_path_ablation_mmlu.sh` 均已通过 `bash -n` 或等价检查；除本条历史记录外，代码/脚本中无可执行的 `least_abs` 分支残留。
- `[AUDIT/FINDING]` main pruning 中 PolyHeadIG 弱于 AttnLRP 的主要问题不在 `most` 端，而在 `least` 端：例如 LLaDA GSM8K AttnLRP prune-most/prune-least 为 14.00/59.50，PolyHeadIG 为 1.50/9.00；Dream MMLU AttnLRP 为 37.19/39.43，PolyHeadIG 为 38.20/25.83。PolyHeadIG 的 `least` 没有稳定对应“剪掉后保分/提分”的有害单元。
- `[AUDIT/FINDING]` score-structure 诊断显示 AttnLRP 与 PolyHeadIG 的低分端几乎不是同一批 units。LLaDA MMLU raw Spearman=0.344，least overlap=48/192；LLaDA GSM8K raw Spearman=0.179，least overlap=47/192；Dream MMLU KV-group raw Spearman=0.412，least overlap=0/6；Dream GSM8K raw Spearman=0.087，least overlap=1/6。
- `[AUDIT/FINDING]` AttnLRP 使用 `relu` relevance 后分数全非负，`least` 实际剪接近 0 relevance 的 units。PolyHeadIG 使用 signed CE-loss IG 并在 downstream 取反后仍有大量负分：LLaDA MMLU negated score 约 42.4% 为负，Dream MMLU KV-group 约 53.6% 为负。因此 PolyHeadIG 的 `least` 剪的是强负分/预测为有害的 units，而不是接近 0 的 units；当前实验显示这些强负 units 在生成/选择题评估中并不稳定有害。
- `[AUDIT/FINDING]` 典型低分端位置差异：Dream MMLU AttnLRP lowest KV groups 位于后层 23/25/26/27，PolyHeadIG lowest KV groups 位于 0/1/3 层；LLaDA GSM8K AttnLRP lowest heads 多为早层接近 0 relevance，PolyHeadIG lowest heads 集中在 16--27 层强负 signed heads。下一步若要解释或修复，应优先验证 PolyHeadIG 负分端是否受 attribution objective、mask distribution、KV-group averaging 或层内 pruning 约束影响。
- `[DONE/UPDATED]` 根据用户决定，`experiments.tex` 正文主 pruning 表 `tab:mask_main` 已从 causal gap 改为只 report `prune-most` 后任务分数，并加入 Dense 参考列。该 destructive intervention 下分数越低说明 top-ranked heads 越有因果作用；`prune-least` 结果保留为诊断信息，不再作为正文主表指标。
- `[AUDIT/FINDING]` Dream 专项诊断：adaptive sparse 中 PolyHeadIG 相对 AttnLRP 在 MMLU/CMMLU/CEval 分别高 +1.18/+0.68/+1.64，但在 GPQA/GSM8K 低 -2.50/-1.50，宏平均只低 -0.07；prune-most 中 PolyHeadIG 相对 AttnLRP 主要输在 GPQA/GSM8K（post-prune 分数高 +8.00/+4.50，lower is better），并非所有任务都弱。
- `[AUDIT/FINDING]` Dream 的核心可疑点是执行粒度为 KV group。按实际 `gqa_weight_mode=kv` 的 relative weight 计算，AttnLRP 与 PolyHeadIG 的 KV-group 权重 Spearman 很低：MMLU 0.177、GPQA 0.268、GSM8K 0.127；top-6 KV groups overlap 仅为 1/6、1/6、2/6。PolyHeadIG 的 signed query-head scores 在同一 KV group 内高度正负混杂，GPQA 110/112 groups、GSM8K 102/112 groups 同时含正负 query-head 分数，group averaging 会显著洗掉或扭曲排序。
- `[AUDIT/FINDING]` Dream GPQA/GSM8K 横向比较还存在归因版本不完全一致：GPQA mask-main 的 AttnLRP 使用 2026-06-11 audit 新归因，而 PolyHeadIG 使用 2026-04-04 旧归因；GSM8K 主表仍主要基于旧 `gsm8k_full` importance，而 Dream 后续 audit 已显示 `final_hash` 协议会改变 AttnLRP 结果。若要最终确认 Dream 结论，建议优先重算 Dream PolyHeadIG 的 GPQA 与 GSM8K final-hash/high-MC attribution，并用相同 `RUN_TS` 重新跑 adaptive/prune-most。
- `[RUNNING]` 2026-06-16 19:05 HKT 已按上述诊断启动 Dream PolyIG 复查，GPU5，tmux `dream-polyig-recheck-gpu5-20260616`，脚本 `experiment_scripts/run_dream_polyig_recheck.sh`。队列：先重算 GPQA PolyHeadIG attribution（`gpqa_main_n_shot_all`，200 samples，seed/data/mask/path seed 均为 131，P=4，signed），随后跑 GPQA adaptive (`MC_NUM=8`) 和 GPQA prune-most；再重算 GSM8K `final_hash` PolyHeadIG attribution（100 samples，seed 131），随后跑 GSM8K adaptive 和 GSM8K prune-most。全部 downstream 均使用 `USE_NEGATED=1`/`USE_NEGATED_MODES_STR=1`、Dream KV-group 口径。
- `[WATCH]` 本轮输出入口：状态 `logs/polyig_recheck/dream_polyig_recheck_20260616_gpu5/status.tsv`，归因路径 `logs/polyig_recheck/dream_polyig_recheck_20260616_gpu5/importance_paths.tsv`，阶段日志在同目录。当前已进入 `attr_gpqa`，配置确认写入 `configs/aconfigs/head_importance_dream_gpqa_main_n_shot_all_pmrandom_threshold_tsdream_polyig_recheck_20260616_gpu5_gpqa_main_n_shot_all/`。
- `[FAILED/AUDIT]` Dream path-ablation adaptive 补列 `path-ablation-dream-mmlu-adaptive-gpu5-20260616` 已结束但三项均失败，原因不是 attribution/path 逻辑，而是 Dream MMLU eval 加载 `hails/mmlu_no_train` 时本地 cache 缺少 `marketing` config。该失败不影响正文主 pruning 表，也不影响本轮 GPQA/GSM8K PolyIG 复查。
- `[WATCH]` 2026-06-16 19:46 HKT 巡检：tmux `path-ablation-llada-mmlu-adaptive-gpu1-20260616` 正常运行，LLaDA Poly adaptive 已到约 8290/9120 likelihood；tmux `dream-polyig-recheck-gpu5-20260616` 仍在 `attr_gpqa`，PID `1485530` 存在，GPU5 有显存和利用率，但日志因 Python stdout 缓冲停在 checkpoint loading 后，尚无 `processed=` progress 或 importance path。暂不判定失败，继续观察；若长时间无 GPU 利用率或进程消失，再按卡住/失败处理。
- `[DONE/UPDATED]` 2026-06-17 巡检：当前无 tmux session。Dream PolyIG recheck 全流程完成并已更新到 `experiments.tex`：adaptive GPQA 23.50、adaptive GSM8K 67.00、prune-most GPQA 24.00、prune-most GSM8K 9.50。Dream PolyHeadIG adaptive macro 从 41.94 更新到 42.58；Dream PolyHeadIG prune-most macro 从 32.55 更新到 31.05（lower is better）。
- `[FINDING]` Dream PolyIG 在 GPQA/GSM8K 上的主要异常基本确认来自旧 attribution/目标协议不一致：旧 GPQA PolyHeadIG prune-most 为 30.00（甚至高于 dense 29.50），新 GPQA 200-sample same-source reattribution 后为 24.00；旧 GSM8K PolyHeadIG adaptive 为 64.50，新 `final_hash` reattribution 后为 67.00，prune-most 从 14.00 降到 9.50，与 AttnLRP 的 9.50 持平。后续正文可解释为“协议对齐后 Dream 上 PolyHeadIG 不再整体弱于 AttnLRP”，但 GPQA adaptive 仍略低于 AttnLRP 24.00 vs 23.50。
- `[DONE/UPDATED]` LLaDA path-design adaptive 补列完成并已更新到 `experiments.tex`：DP 63.42、STP 64.08、PolyHeadIG 63.29。该单 seed MMLU ablation 仍不支持写成 PolyHeadIG 在 path ablation 中稳定最优；STP 在 adaptive 和 pruning gap 上都略高于 PolyHeadIG。
- `[RUNNING]` 13:54 HKT 重新规划并启动多任务 path-design ablation。目的：避免只凭 MMLU 单任务判断 DP/STP/PolyHeadIG；本轮新增 CMMLU、GSM8K、GPQA，结合已有 MMLU 后形成 knowledge/math/hard-MC 三类任务证据。每个任务均按同一协议跑 DP(`diagonal`, P=1)、STP(`random_threshold`, P=1)、PolyHeadIG(`random_threshold`, P=4)，并接 prune-most/prune-least 与 adaptive-sparse。默认配置：`IG_STEPS=8`、`MASK_PROBS=0.15,0.3,0.5,0.7,0.9`、`MASK_SAMPLES_PER_PROB=2`、`IG_POSTPROCESS=signed`、HeadIG downstream `USE_NEGATED=1`；GPQA 固定 `MC_NUM=8`，GSM8K attribution 固定 `GSM8K_ANSWER_MODE=final_hash`。
- `[RUNNING]` GPU 分配：GPU4 跑 LLaDA，tmux `path-ablation-llada-multitask-gpu4-20260617`，状态 `logs/path_ablation/path_ablation_llada_multitask_20260617_gpu4/multitask_status.tsv`；GPU5 跑 Dream，tmux `path-ablation-dream-multitask-gpu5-20260617`，状态 `logs/path_ablation/path_ablation_dream_multitask_20260617_gpu5/multitask_status.tsv`。启动时两条队列均进入 CMMLU；GPU0/1/3 已有较高显存或利用率，因此本轮没有继续加压这些卡。

### 2026-06-18

- `[DONE/PARTIAL]` 00:37 HKT 巡检：LLaDA 多任务 path ablation 的 CMMLU 已完整完成，队列已进入 GSM8K；Dream 多任务 path ablation 仍在 CMMLU 的 Poly adaptive 阶段，约 20930/21440 likelihood，进程正常。当前 tmux：`path-ablation-llada-multitask-gpu4-20260617`、`path-ablation-dream-multitask-gpu5-20260617`。GPU 快照：GPU0 约 20.96GB/3% util，GPU4 约 23.96GB/100% util，GPU5 约 32.71GB/41% util。
- `[RESULT]` LLaDA CMMLU path ablation，seed 123，`M=60` attribution examples，`limit=80` eval examples，layer-wise 20% pruning。DP：adaptive 65.07，prune-most 42.74，prune-least 60.19，gap 17.44，attribution 9.55min。STP：adaptive 64.53，prune-most 30.76，prune-least 60.60，gap 29.83，attribution 31.73min。PolyHeadIG：adaptive 64.94，prune-most 27.54，prune-least 59.91，gap 32.37，attribution 48.48min。该结果支持在 LLaDA CMMLU 上 PolyHeadIG 的 destructive prune-most 最强、gap 最大，但 adaptive 三者差距很小；已写入 `experiments.tex` 的多任务 path ablation 表。
- `[RESULT]` Dream CMMLU path ablation，seed 123，`M=60`，KV-group 5% pruning。DP：adaptive 33.60，prune-most 29.20，prune-least 28.94，gap -0.26，attribution 14.28min。STP：adaptive 34.22，prune-most 28.96，prune-least 24.89，gap -4.07，attribution 36.65min。PolyHeadIG：adaptive 34.16，prune-most 29.59，prune-least 24.78，gap -4.81，attribution 69.43min。Dream CMMLU 的 most 端三者差异小，least/gap 仍显示低分端不稳定；已写入 `experiments.tex` 的多任务 path ablation 表。
- `[RUNNING]` 00:38 HKT 在 GPU0 新启动 Dream CMMLU 第二 seed 稳定性任务，tmux `path-ablation-dream-cmmlu-seed321-gpu0-20260618`，run tag base `path_ablation_dream_cmmlu_seed321_20260618_gpu0`，`TASKS_STR=cmmlu`，`RUN_ADAPTIVE=0`，`SEED=DATA_SEED=MASK_SEED=PATH_SEED=321`。目的：先补 Dream CMMLU 的 seed robustness 和排序稳定性证据，不跑 adaptive 以控制开销。状态入口：`logs/path_ablation/path_ablation_dream_cmmlu_seed321_20260618_gpu0/multitask_status.tsv`。
- `[DONE/UPDATED]` 用户同意将 ablation 改成多任务表后，`experiments.tex` 的 `tab:path_ablation` 已从 MMLU-only 表改为当前多任务 inventory。表格现在报告 Model/Task/Path、Prune-most、Causal gap、Adaptive 和 attribution time；已填 MMLU 与 CMMLU 两个任务，GSM8K/GPQA 在现有后台队列完成后再追加。
- `[DONE/UPDATED]` 16:57 HKT 巡检：三条 path ablation 队列均已完成，当前无 tmux session。`experiments.tex` 的 `tab:path_ablation` 已追加 GSM8K 与 GPQA，形成 MMLU/CMMLU/GSM8K/GPQA 四任务表。LLaDA 多任务队列状态：`logs/path_ablation/path_ablation_llada_multitask_20260617_gpu4/multitask_status.tsv`；Dream 多任务队列状态：`logs/path_ablation/path_ablation_dream_multitask_20260617_gpu5/multitask_status.tsv`；Dream seed321 状态：`logs/path_ablation/path_ablation_dream_cmmlu_seed321_20260618_gpu0/multitask_status.tsv`。
- `[RESULT]` Dream path ablation 新增任务，seed 123。GSM8K：DP adaptive 73.00 / most 21.00 / gap 61.00；STP adaptive 75.00 / most 86.00 / gap -6.00；PolyHeadIG adaptive 72.00 / most 10.00 / gap 74.00。GPQA：DP adaptive 25.00 / most 29.17 / gap -7.50；STP adaptive 22.50 / most 25.00 / gap -0.83；PolyHeadIG adaptive 22.50 / most 20.83 / gap 5.00。Interpretation：Dream 上 PolyHeadIG 在 GSM8K/GPQA 的 destructive prune-most 与 gap 上最强，但 adaptive 不一定最高。
- `[RESULT]` LLaDA path ablation 新增任务，seed 123。GSM8K：DP adaptive 66.00 / most 43.00 / gap 6.00；STP adaptive 69.00 / most 39.00 / gap 5.00；PolyHeadIG adaptive 59.00 / most 36.00 / gap 9.00。GPQA：DP adaptive 23.33 / most 28.33 / gap -2.50；STP adaptive 28.33 / most 27.50 / gap 1.67；PolyHeadIG adaptive 24.17 / most 26.67 / gap 0.83。Interpretation：LLaDA 上 PolyHeadIG 在 GSM8K/GPQA 的 prune-most 最低，但 adaptive sparse 上 STP 常更高，说明 path-design 的“causal ranking”和“adaptive allocation”并不完全一致。
- `[RESULT]` Dream CMMLU seed321 pruning-only 完成。DP：prune-most 33.34，gap -5.93，attribution 14.33min；STP：prune-most 31.87，gap -4.89，attribution 36.52min；PolyHeadIG：prune-most 29.81，gap -2.44，attribution 68.78min。与 seed123 比较，Dream CMMLU 的低分端仍不稳定，但 PolyHeadIG 在第二 seed 的 most 端更强；后续可用这两个 seed 计算 Spearman/top-k overlap。
- `[RESULT]` Dream CMMLU seed123 vs seed321 稳定性已计算，使用 downstream 实际 `_neg` importance。Query-head 级 Spearman/top-5% overlap：DP 0.642 / 17-of-39，STP 0.234 / 13-of-39，PolyHeadIG 0.519 / 17-of-39。KV-group 平均后 Spearman/top-5% overlap：DP 0.699 / 4-of-6，STP 0.459 / 4-of-6，PolyHeadIG 0.610 / 5-of-6。Interpretation：在实际 KV-group 干预口径下，PolyHeadIG 的 top group 稳定性最好，但整体 Spearman 低于 DP；STP 最不稳定。
- `[RUNNING]` 17:00 HKT 为补充 seed robustness，新启动两条 pruning-only 稳定性任务，不跑 adaptive。GPU5：tmux `path-ablation-llada-cmmlu-seed321-gpu5-20260618`，LLaDA CMMLU seed321，状态 `logs/path_ablation/path_ablation_llada_cmmlu_seed321_20260618_gpu5/multitask_status.tsv`。GPU0：tmux `path-ablation-dream-gsm8k-seed321-gpu0-20260618`，Dream GSM8K seed321，状态 `logs/path_ablation/path_ablation_dream_gsm8k_seed321_20260618_gpu0/multitask_status.tsv`。启动后两者均进入 DP attribution。
- `[QUEUED]` 17:08 HKT 用户要求 path-design ablation 尽量覆盖全部数据集；已追加四条等待空卡后自动启动的 tmux 队列。非 Minerva 缺口先补 CEval/HumanEval/MBPP，并跑 attribution + most/least pruning + adaptive：LLaDA 等 GPU5 空闲，tmux `path-ablation-llada-rest-wait-gpu5-20260618`，run tag `path_ablation_llada_rest_20260618_gpu5`；Dream 等 GPU0 空闲，tmux `path-ablation-dream-rest-wait-gpu0-20260618`，run tag `path_ablation_dream_rest_20260618_gpu0`。配置：`TASKS_STR=ceval-valid,humaneval,mbpp`，`CEVAL_ATTR_SAMPLES=60`，`CEVAL_EVAL_LIMIT=100`，`DEFAULT_ATTR_SAMPLES=60`，`DEFAULT_EVAL_LIMIT=80`，`RUN_ADAPTIVE=1`，seed 123。
- `[QUEUED/SMOKE]` 17:08 HKT Minerva Math 单独作为小规模 smoke，不混进上面的长队列，避免其多子任务/`math_verify` 评测把队列拖死。LLaDA 等 GPU1 空闲，tmux `path-ablation-llada-minerva-wait-gpu1-20260618`，run tag `path_ablation_llada_minerva_smoke_20260618_gpu1`；Dream 等 GPU4 空闲，tmux `path-ablation-dream-minerva-wait-gpu4-20260618`，run tag `path_ablation_dream_minerva_smoke_20260618_gpu4`。配置：`TASKS_STR=minerva_math`，`DEFAULT_ATTR_SAMPLES=20`，`DEFAULT_EVAL_LIMIT=20`，`RUN_ADAPTIVE=0`，seed 123；先只跑 attribution + most/least pruning，待 smoke 正常后再决定是否补 adaptive。
- `[WATCH]` 17:08 HKT 启动后四条新队列均正常进入 waiter 状态，尚未占用新 GPU。waiter 日志分别为 `logs/path_ablation/path_ablation_llada_rest_20260618_gpu5/waiter.log`、`logs/path_ablation/path_ablation_dream_rest_20260618_gpu0/waiter.log`、`logs/path_ablation/path_ablation_llada_minerva_smoke_20260618_gpu1/waiter.log`、`logs/path_ablation/path_ablation_dream_minerva_smoke_20260618_gpu4/waiter.log`。触发条件统一为 `memory.used<=20000 MiB` 且 `utilization<=30%` 连续 2 次。

### 2026-06-20

- `[DONE/UPDATED]` LLaDA path-design rest 队列完成 HumanEval 与 MBPP，并已更新 `experiments.tex` 的 `tab:path_ablation`。CEval attribution 失败，原因是当前离线环境无法取得 `ceval/ceval-exam` configs，日志报 `No dataset configs found for ceval/ceval-exam`；这不是模型或 PolyHeadIG 实现错误。
- `[RESULT]` LLaDA HumanEval path ablation，seed 123，`M=60`，eval limit 80。DP：adaptive 55.00，prune-most 53.75，gap 0.00，attribution 6.02min。STP：adaptive 53.75，prune-most 55.00，gap -1.25，attribution 5.87min。PolyHeadIG：adaptive 55.00，prune-most 53.75，gap 1.25，attribution 23.08min。
- `[RESULT]` LLaDA MBPP path ablation，seed 123，`M=60`，eval limit 80。DP/STP/PolyHeadIG 的 adaptive、prune-most、prune-least 均为 35.00，gap 均为 0.00；attribution time 分别为 5.83min、5.73min、23.25min。该任务在当前 80 样本设置下对 path 设计不敏感。
- `[DONE/PARTIAL]` Dream path-design rest 队列完成 HumanEval；MBPP 的 DP 与 PolyHeadIG 完整，STP 的 adaptive 阶段因 GPU0 OOM 失败，但 STP most/least 已落盘。CEval 与 LLaDA 相同，阻塞于 `ceval/ceval-exam` 本地 config 缺失。
- `[RESULT]` Dream HumanEval path ablation，seed 123，`M=60`，eval limit 80。DP：adaptive 60.00，prune-most 60.00，gap 0.00，attribution 10.95min。STP：adaptive 60.00，prune-most 58.75，gap 1.25，attribution 10.92min。PolyHeadIG：adaptive 60.00，prune-most 60.00，gap 0.00，attribution 42.30min。
- `[RESULT/PARTIAL]` Dream MBPP path ablation，seed 123，`M=60`，eval limit 80。DP：adaptive 60.00，prune-most 60.00，gap 0.00，attribution 11.25min。STP：adaptive failed by OOM，prune-most 60.00，gap 0.00，attribution 10.98min。PolyHeadIG：adaptive 60.00，prune-most 60.00，gap 0.00，attribution 42.28min。
- `[DONE/SMOKE]` Dream Minerva Math path-design smoke 完成，主指标使用总任务 `minerva_math` 的 `math_verify,none`，不是恒为 0 的 `exact_match`。`limit=20`、无 adaptive。DP：prune-most 20.00、prune-least 35.71、gap 15.71。STP：prune-most 35.00、prune-least 33.57、gap -1.43。PolyHeadIG：prune-most 25.00、prune-least 34.29、gap 9.29。该结果只用于协议验证，暂不进正文主表。
- `[DONE/AUDIT]` Seed robustness 两条补充队列已完成。LLaDA CMMLU seed321：DP most 49.25/gap 7.80，STP most 53.79/gap -1.75，PolyHeadIG most 45.41/gap 6.51。Dream GSM8K seed321：DP most 18.00/gap 66.00，STP most 8.00/gap 8.00，PolyHeadIG most 71.00/gap 8.00。Dream GSM8K 的 PolyHeadIG seed321 与 seed123 差异极大，是当前 path ablation 最大异常；后续应计算 seed123/321 的 Spearman 与 KV-group top-k overlap，并至少再补一个 seed 或扩大 attribution samples。
- `[REQUEUED]` 旧的 `path-ablation-llada-minerva-wait-gpu1-20260618` 只是等待器且尚未进入实验计算，15:53 HKT 已停止，避免后续与新任务重复启动。
- `[RUNNING]` 15:53 HKT 启动三条下一步实验。GPU0：`path-ablation-llada-minerva-gpu0-20260620`，run tag `path_ablation_llada_minerva_smoke_20260620_gpu0`，补 LLaDA Minerva smoke，`limit=20`，无 adaptive。GPU2：`path-ablation-dream-mbpp-stp-adaptive-gpu2-20260620`，复用 run tag `path_ablation_dream_rest_20260618_gpu0_mbpp`，只补 Dream MBPP STP adaptive；断点检查已跳过已完成的 DP/STP attribution、mask 和 DP adaptive。GPU5：`path-ablation-dream-gsm8k-seed777-gpu5-20260620`，run tag `path_ablation_dream_gsm8k_seed777_20260620_gpu5`，补 Dream GSM8K 第三个 seed，`RUN_ADAPTIVE=0`。
- `[STATUS]` 15:53 HKT 启动后 GPU 快照：GPU0 43.6GB/1% util，GPU1 25.7GB/0% util，GPU2 41.6GB/91% util，GPU3 79.1GB/100% util，GPU4 72.4GB/2% util，GPU5 43.4GB/1% util。当前需要重点观察 GPU2 的 Dream MBPP STP adaptive 是否再次 OOM。

## 12. 后台运行记录

| 时间 | 状态 | GPU | PID | 实验 | 日志 | 输出目录/结果 |
|---|---|---:|---:|---|---|---|
| 2026-06-06 | DONE | 0 | detached tmux | LLaDA LOO，MMLU，1 sample，layer 0；32 heads / 33 utility evaluations | `logs/llada_loo_smoke_gpu0.log` | `configs/aconfigs/head_importance_llada-1_5_mmlu_all_loo_signed_maskp0.5_mcs1_mean_masked_tssmoke_llada_loo_20260606/` |
| 2026-06-06 | DONE | 0 | detached tmux | Dream LOO，MMLU，1 sample，layer 0；28 query heads / 29 utility evaluations | `logs/dream_loo_smoke_gpu0.log` | `configs/aconfigs/head_importance_dream_mmlu_all_loo_signed_maskp0.5_mcs1_mean_masked_tssmoke_dream_loo_20260606/` |
| 2026-06-06 23:15 HKT | DONE | 0 | tmux `adaptive-llada-loo-core-20260606` | LLaDA exact LOO -> adaptive -> prune-most/least；队列 `mmlu,cmmlu,ceval-valid,gpqa_main_n_shot,gsm8k,humaneval,mbpp`；2026-06-10 01:13 完成 MBPP prune-least | `logs/loo_core/formal_loo_llada_20260606/pipeline.log`；状态 `logs/loo_core/formal_loo_llada_20260606/status.tsv` | `evaluation/llada/llada_1_5_results/adaptive/*formal_loo_llada_20260606*/`；`evaluation/llada/llada_1_5_results/mask_head/*formal_loo_llada_20260606*/` |
| 2026-06-06 23:40 HKT | DONE/FAILED | 0 | tmux `adaptive-table-fill-gpu0-20260606` | 补 adaptive 主表缺口；2026-06-09 16:45 结束，失败项为 Dream dense MMLU、Dream AttnLRP MBPP、Dream HeadIG MBPP | `logs/adaptive_table_fill/adaptive_table_fill_missing_20260606/pipeline.log`；状态 `logs/adaptive_table_fill/adaptive_table_fill_missing_20260606/status.tsv` | LLaDA dense 已写入 `evaluation/llada/llada_1_5_results/standard/`；Dream dense 已写入 `evaluation/dream/results/standard/`；adaptive 缺口写入对应 `adaptive/` 目录 |
| 2026-06-06 23:41 HKT | DONE/FAILED | 2 | tmux `adaptive-dream-loo-wait-gpu2-20260606` | 后续启动 Dream exact LOO 队列 `formal_loo_dream_20260606`；CEval attribution 失败、MMLU adaptive 失败，其余主要 pruning/adaptive 阶段完成 | `logs/loo_core/formal_loo_dream_20260606/status.tsv`；阶段日志在同目录 | Dream LOO pruning 有效结果写入 `evaluation/dream/results/mask_head/loo_dream_*_formal_loo_dream_20260606/` |
| 2026-06-07 21:16 HKT | FAILED | 1 | tmux `adaptive-dream-loo-gpu1-20260607` | Dream exact LOO -> adaptive -> prune-most/least；队列 `cmmlu,gpqa_main_n_shot,gsm8k,humaneval,mbpp`；MBPP pruning 在旧协议下失败，结果无效 | `logs/loo_core/formal_loo_dream_gpu1_20260607/pipeline.log`；状态 `logs/loo_core/formal_loo_dream_gpu1_20260607/status.tsv` | `configs/aconfigs/head_importance_dream_*_loo_*tsformal_loo_dream_gpu1_20260607_*`；结果写入 `evaluation/dream/results/adaptive/` 和 `evaluation/dream/results/mask_head/` |
| 2026-06-08 22:49 HKT | DONE | 4 | tmux `dream-protocol-audit-gpu4-20260608` | Dream 协议诊断：GPQA dense/adaptive `MC_NUM=8`；MBPP 修复后 dense smoke `limit=20` | `logs/protocol_audit/dream_protocol_audit_20260608_gpu4/pipeline.log`；状态 `logs/protocol_audit/dream_protocol_audit_20260608_gpu4/status.tsv` | `evaluation/dream/results/standard/*audit_dream_protocol_audit_20260608_gpu4*`；`evaluation/dream/results/adaptive/*audit_dream_protocol_audit_20260608_gpu4*` |
| 2026-06-09 23:25 HKT | DONE | 3 | tmux `dream-recovery-gpu3-20260609` | Dream recovery：GPQA AttnLRP/Shapley/PolyHeadIG `MC_NUM=8`；Dream MBPP dense/AttnLRP/PolyHeadIG/LOO 修复后 full run；2026-06-10 23:49 全部完成 | `logs/recovery/dream_recovery_gpu3_20260609/pipeline.log`；状态 `logs/recovery/dream_recovery_gpu3_20260609/status.tsv` | `evaluation/dream/results/adaptive/*recovery_dream_recovery_gpu3_20260609*`；`evaluation/dream/results/standard/*recovery_dream_recovery_gpu3_20260609*` |
| 2026-06-11 18:07 HKT | DONE/FAILED | 3 | tmux `dream-attnlrp-audit-gpu3-20260611` | Dream AttnLRP 重新归因 audit：GPQA/GSM8K 新 seed 131 重算 importance，并跑 GPQA/GSM8K 2x2 same-task/cross-task adaptive eval；前三项完成，GSM8K attribution -> GPQA 因 OOM 失败 | `logs/attnlrp_audit/dream_attnlrp_regen_audit_20260611_gpu3/status.tsv`；阶段日志在同目录 | 新 importance 已写入 `configs/aconfigs/head_importance_dream_*_attnlrp_*_tsdream_attnlrp_regen_audit_20260611_gpu3_*`；有效结果：GPQA->GPQA 22.00、GPQA->GSM8K 63.00、GSM8K->GSM8K 62.50 |
| 2026-06-11 18:18 HKT | DONE/AUDIT | 4 | tmux `mask-main-llada-gpu4-20260611-v2` | LLaDA main pruning/masking 补表：AttnLRP、Shapley、PolyHeadIG；七个非 Minerva 任务均完成。Shapley HumanEval 结果已写出但阶段状态 false-failed rc=2 | `logs/mask_main/mask_main_llada_gpu4_20260611_v2/status.tsv`；阶段日志在同目录 | 结果写入 `evaluation/llada/llada_1_5_results/mask_head/maskfill_mask_main_llada_gpu4_20260611_v2_*` |
| 2026-06-11 18:18 HKT | DONE | 0 | tmux `mask-main-dream-gpu0-20260611-v2` | Dream main pruning/masking 补表：AttnLRP、Shapley、PolyHeadIG；七个非 Minerva 任务均完成。最后完成项为 Dream PolyHeadIG/headig MBPP，gap=0.50 | `logs/mask_main/mask_main_dream_gpu0_20260611_v2/status.tsv`；阶段日志在同目录 | 结果写入 `evaluation/dream/results/mask_head/maskfill_mask_main_dream_gpu0_20260611_v2_*` |
| 2026-06-12 19:00 HKT | DROP | 2 | tmux `mask-main-llada-headig-gpu2-20260612` | LLaDA HeadIG main pruning 初次启动；因 LLaDA runner 缺 `SCRIPT_DIR`/`PROJECT_ROOT`，生成 negated importance 路径错误并立即失败；已停掉，不作为结果使用 | `logs/mask_main/mask_main_llada_headig_gpu2_20260612/status.tsv` | 无有效结果 |
| 2026-06-12 19:05 HKT | DONE | 2 | tmux `mask-main-llada-headig-gpu2-20260612-v2` | LLaDA PolyHeadIG/headig main pruning 重复确认队列；七个非 Minerva 任务全部完成，结果与 GPU4 headig 一致 | `logs/mask_main/mask_main_llada_headig_gpu2_20260612_v2/status.tsv`；阶段日志在同目录 | 结果写入 `evaluation/llada/llada_1_5_results/mask_head/maskfill_mask_main_llada_headig_gpu2_20260612_v2_*` |
| 2026-06-12 19:00 HKT | DONE/FAILED | 3 | tmux `dream-audit-retry-mask-gpu3-20260612` | Dream AttnLRP `GSM8K final_hash attribution -> GPQA MC_NUM=8` retry 完成 20.00；随后 Dream Shapley 全部完成、HeadIG 到 GSM8K 完成，HeadIG HumanEval/MBPP 因同卡显存占用 OOM 失败 | `logs/audit_retry/dream_audit_retry_mask_gpu3_20260612/status.tsv`；后续 mask 阶段为 `logs/mask_main/mask_main_dream_shapley_headig_gpu3_20260612/status.tsv` | retry 结果写入 Dream adaptive 目录；可用 duplicate mask 结果写入 `evaluation/dream/results/mask_head/maskfill_mask_main_dream_shapley_headig_gpu3_20260612_*` |
| 2026-06-15 13:07 HKT | DONE | 4 | tmux `path-ablation-llada-mmlu-gpu4-20260615` | LLaDA MMLU path-design ablation：DP / STP / PolyHeadIG，`MAX_SAMPLES=40`，归因后接 most/least pruning。Gaps：DP 19.17、STP 20.92、PolyHeadIG 7.59 | `logs/path_ablation/path_ablation_llada_mmlu_20260615/status.tsv`；pipeline log `logs/path_ablation/path_ablation_llada_mmlu_20260615/pipeline.log` | importance 写入 `configs/aconfigs/head_importance_llada-1_5_mmlu_all_pm*_tspath_ablation_llada_mmlu_20260615_*`；mask 结果写入 LLaDA `mask_head/pathab_*` |
| 2026-06-15 13:07 HKT | DONE | 5 | tmux `path-ablation-dream-mmlu-gpu5-20260615` | Dream MMLU path-design ablation：DP / STP / PolyHeadIG，`MAX_SAMPLES=40`，归因后接 KV-group 5% most/least pruning。Gaps：DP -4.39、STP 4.96、PolyHeadIG -6.32 | `logs/path_ablation/path_ablation_dream_mmlu_20260615/status.tsv`；pipeline log `logs/path_ablation/path_ablation_dream_mmlu_20260615/pipeline.log` | importance 写入 `configs/aconfigs/head_importance_dream_mmlu_all_pm*_tspath_ablation_dream_mmlu_20260615_*`；mask 结果写入 Dream `mask_head/pathab_*` |
| 2026-06-16 15:01 HKT | DONE | 1 | tmux `path-ablation-llada-mmlu-adaptive-gpu1-20260616` | LLaDA MMLU path-design adaptive-sparse 补列；复用 2026-06-15 importance，DP/STP/PolyHeadIG adaptive 全部完成 | `logs/path_ablation/path_ablation_llada_mmlu_20260615/status.tsv`；`logs/path_ablation/path_ablation_llada_mmlu_20260615/pipeline_adaptive_gpu1_20260616.log` | Adaptive：DP 63.42、STP 64.08、PolyHeadIG 63.29 |
| 2026-06-16 15:01 HKT | FAILED/AUDIT | 5 | tmux `path-ablation-dream-mmlu-adaptive-gpu5-20260616` | Dream MMLU path-design adaptive-sparse 补列；DP/STP/PolyHeadIG 三项均因本地 MMLU cache 缺 `marketing` config 失败，不作为结果使用 | `logs/path_ablation/path_ablation_dream_mmlu_20260615/status.tsv`；`logs/path_ablation/path_ablation_dream_mmlu_20260615/pipeline_adaptive_gpu5_20260616.log` | 无有效 adaptive 结果 |
| 2026-06-16 19:05 HKT | DONE | 5 | tmux `dream-polyig-recheck-gpu5-20260616` | Dream PolyIG GPQA/GSM8K 复查：重算 GPQA 200-sample attribution 与 GSM8K `final_hash` 100-sample attribution，然后同源重跑 adaptive 和 prune-most；GPQA eval 固定 `MC_NUM=8`，HeadIG downstream 固定 `USE_NEGATED=1` | `logs/polyig_recheck/dream_polyig_recheck_20260616_gpu5/status.tsv`；阶段日志在同目录 | GPQA adaptive 23.50、GPQA prune-most 24.00；GSM8K adaptive 67.00、GSM8K prune-most 9.50 |
| 2026-06-17 13:54 HKT | DONE | 4 | tmux `path-ablation-llada-multitask-gpu4-20260617` | LLaDA 多任务 path-design ablation；队列 `cmmlu,gsm8k,gpqa_main_n_shot`，结合已有 MMLU 形成四任务消融；2026-06-18 09:49 完成 | `logs/path_ablation/path_ablation_llada_multitask_20260617_gpu4/multitask_status.tsv`；各任务 `*.pipeline.log` | 四任务结果已写入 `experiments.tex` 的 `tab:path_ablation` |
| 2026-06-17 13:54 HKT | DONE | 5 | tmux `path-ablation-dream-multitask-gpu5-20260617` | Dream 多任务 path-design ablation；队列 `cmmlu,gsm8k,gpqa_main_n_shot`，结合已有 MMLU 形成四任务消融；2026-06-18 09:04 完成 | `logs/path_ablation/path_ablation_dream_multitask_20260617_gpu5/multitask_status.tsv`；各任务 `*.pipeline.log` | 四任务结果已写入 `experiments.tex` 的 `tab:path_ablation` |
| 2026-06-18 00:38 HKT | DONE | 0 | tmux `path-ablation-dream-cmmlu-seed321-gpu0-20260618` | Dream CMMLU path-design 第二 seed；`SEED=321`，只跑 attribution + most/least pruning，不跑 adaptive；2026-06-18 04:05 完成 | `logs/path_ablation/path_ablation_dream_cmmlu_seed321_20260618_gpu0/multitask_status.tsv`；`logs/path_ablation/path_ablation_dream_cmmlu_seed321_20260618_gpu0/cmmlu.pipeline.log` | seed321 pruning-only 结果已写入第 11 节，用于后续稳定性计算 |
| 2026-06-18 17:00 HKT | DONE | 5 | tmux `path-ablation-llada-cmmlu-seed321-gpu5-20260618` | LLaDA CMMLU path-design 第二 seed；`SEED=321`，只跑 attribution + most/least pruning，不跑 adaptive，用于 seed robustness | `logs/path_ablation/path_ablation_llada_cmmlu_seed321_20260618_gpu5/multitask_status.tsv`；`logs/path_ablation/path_ablation_llada_cmmlu_seed321_20260618_gpu5/cmmlu.pipeline.log` | 结果已记录在第 11 节 2026-06-20 条目 |
| 2026-06-18 17:00 HKT | DONE | 0 | tmux `path-ablation-dream-gsm8k-seed321-gpu0-20260618` | Dream GSM8K path-design 第二 seed；`SEED=321`，只跑 attribution + most/least pruning，不跑 adaptive，用于 seed robustness | `logs/path_ablation/path_ablation_dream_gsm8k_seed321_20260618_gpu0/multitask_status.tsv`；`logs/path_ablation/path_ablation_dream_gsm8k_seed321_20260618_gpu0/gsm8k.pipeline.log` | 结果已记录在第 11 节 2026-06-20 条目；PolyHeadIG seed sensitivity 需复查 |
| 2026-06-18 17:08 HKT | DONE/FAILED | 5 | tmux `path-ablation-llada-rest-wait-gpu5-20260618` | LLaDA path-design ablation 补 CEval/HumanEval/MBPP；HumanEval 与 MBPP 完成并已更新，CEval 因本地 `ceval/ceval-exam` config 缺失失败 | `logs/path_ablation/path_ablation_llada_rest_20260618_gpu5/multitask_status.tsv` | HumanEval/MBPP 已写入 `experiments.tex`；CEval 待修复本地数据后重跑 |
| 2026-06-18 17:08 HKT | DONE/FAILED | 0 | tmux `path-ablation-dream-rest-wait-gpu0-20260618` | Dream path-design ablation 补 CEval/HumanEval/MBPP；HumanEval 完成，MBPP 除 STP adaptive OOM 外已落盘，CEval 因本地 config 缺失失败 | `logs/path_ablation/path_ablation_dream_rest_20260618_gpu0/multitask_status.tsv` | HumanEval/MBPP 可用部分已写入 `experiments.tex`；STP adaptive 和 CEval 待重跑 |
| 2026-06-18 17:08 HKT | REQUEUED | 1 | tmux `path-ablation-llada-minerva-wait-gpu1-20260618` | LLaDA Minerva Math path-design smoke；等待器未进入实验计算，2026-06-20 15:53 HKT 停止，避免与 GPU0 新队列重复 | `logs/path_ablation/path_ablation_llada_minerva_smoke_20260618_gpu1/waiter.log` | 已改由 `path-ablation-llada-minerva-gpu0-20260620` 执行 |
| 2026-06-18 17:08 HKT | DONE/SMOKE | 4 | tmux `path-ablation-dream-minerva-wait-gpu4-20260618` | Dream Minerva Math path-design smoke；`limit=20`，DP/STP/PolyHeadIG attribution + most/least pruning 完成，无 adaptive | `logs/path_ablation/path_ablation_dream_minerva_smoke_20260618_gpu4_minerva_math/status.tsv` | 结果已记录在第 11 节 2026-06-20 条目；仅作协议 smoke |
| 2026-06-20 15:53 HKT | RUNNING | 0 | tmux `path-ablation-llada-minerva-gpu0-20260620` | LLaDA Minerva Math path-design smoke；`limit=20`，DP/STP/PolyHeadIG attribution + most/least pruning，先不跑 adaptive | `logs/path_ablation/path_ablation_llada_minerva_smoke_20260620_gpu0/multitask_status.tsv`；任务日志 `logs/path_ablation/path_ablation_llada_minerva_smoke_20260620_gpu0/minerva_math.pipeline.log` | 输出将写入 LLaDA `mask_head/pathab_*` |
| 2026-06-20 15:53 HKT | RUNNING | 2 | tmux `path-ablation-dream-mbpp-stp-adaptive-gpu2-20260620` | Dream MBPP path ablation resume；只补此前 OOM 的 STP adaptive，复用 `path_ablation_dream_rest_20260618_gpu0_mbpp` 状态目录和 importance | `logs/path_ablation/path_ablation_dream_rest_20260618_gpu0_mbpp/status.tsv`；阶段日志 `logs/path_ablation/path_ablation_dream_rest_20260618_gpu0_mbpp/adaptive_dream_stp_mbpp.log` | 若成功，将填 `tab:path_ablation` 的 Dream MBPP STP Adaptive |
| 2026-06-20 15:53 HKT | RUNNING | 5 | tmux `path-ablation-dream-gsm8k-seed777-gpu5-20260620` | Dream GSM8K path-design 第三个 seed；`SEED=777`，只跑 attribution + most/least pruning，用于复查 PolyHeadIG seed sensitivity | `logs/path_ablation/path_ablation_dream_gsm8k_seed777_20260620_gpu5/multitask_status.tsv`；任务日志 `logs/path_ablation/path_ablation_dream_gsm8k_seed777_20260620_gpu5/gsm8k.pipeline.log` | 输出将写入 Dream `mask_head/pathab_*`；完成后计算 seed123/321/777 稳定性 |
