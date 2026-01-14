#!/usr/bin/env bash
# 验证实验：用逐层IG方法重新计算1条和50条数据，对比结果差异
set -euo pipefail

export CUDA_VISIBLE_DEVICES=3
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONPATH=/home/qiheng/Projects/adaptive-dllm

MODEL_PATH="/data/qh_models/LLaDA-1.5"
OUT_ROOT="/home/qiheng/Projects/adaptive-dllm/configs"
RUN_TS=$(date +%Y%m%d_%H%M%S)

SEED=123
IG_STEPS=8
MASK_PROBS="0.15,0.3,0.5,0.7,0.9"
MASK_SAMPLES_PER_PROB=2

echo "========================================================"
echo "逐层IG验证实验：对比1条 vs 50条数据"
echo "========================================================"
echo "Started at: $(date)"
echo "========================================================"

# 实验1：1条数据（每类1条 = 5条总共）
echo ""
echo ">>> 实验1：1条数据/类（总5条）<<<"
OUT_DIR_1="${OUT_ROOT}/head_importance_perlayer_1sample_ts${RUN_TS}"
mkdir -p "${OUT_DIR_1}"

python /home/qiheng/Projects/adaptive-dllm/models/LLaDA/attribution/loss_attribution/compute_loss_attribution.py \
  --model_path "${MODEL_PATH}" \
  --dataset nemotron \
  --samples_per_category 1 \
  --max_samples 5 \
  --nemotron_pool_per_category 3000 \
  --nemotron_categories "code,math,science,chat,safety" \
  --dataset_shuffle \
  --seed ${SEED} \
  --data_seed ${SEED} \
  --mask_seed ${SEED} \
  --ig_steps ${IG_STEPS} \
  --mask_probs "${MASK_PROBS}" \
  --mask_samples_per_prob ${MASK_SAMPLES_PER_PROB} \
  --loss_normalize mean_masked \
  --ig_postprocess signed \
  --mask_batch_size 2 \
  --activation_checkpointing whole_layer \
  --baseline zero \
  --layer_start 0 \
  --layer_end 31 \
  --output_dir "${OUT_DIR_1}" \
  --use_amp_bf16 2>&1 | tee "${OUT_DIR_1}/run.log"

echo ""
echo "✅ 实验1完成: ${OUT_DIR_1}/head_importance.pt"

# 实验2：50条数据（每类10条 = 50条总共）
echo ""
echo ">>> 实验2：10条数据/类（总50条）<<<"
OUT_DIR_50="${OUT_ROOT}/head_importance_perlayer_50samples_ts${RUN_TS}"
mkdir -p "${OUT_DIR_50}"

python /home/qiheng/Projects/adaptive-dllm/models/LLaDA/attribution/loss_attribution/compute_loss_attribution.py \
  --model_path "${MODEL_PATH}" \
  --dataset nemotron \
  --samples_per_category 10 \
  --max_samples 50 \
  --nemotron_pool_per_category 3000 \
  --nemotron_categories "code,math,science,chat,safety" \
  --dataset_shuffle \
  --seed ${SEED} \
  --data_seed ${SEED} \
  --mask_seed ${SEED} \
  --ig_steps ${IG_STEPS} \
  --mask_probs "${MASK_PROBS}" \
  --mask_samples_per_prob ${MASK_SAMPLES_PER_PROB} \
  --loss_normalize mean_masked \
  --ig_postprocess signed \
  --mask_batch_size 2 \
  --activation_checkpointing whole_layer \
  --baseline zero \
  --layer_start 0 \
  --layer_end 31 \
  --output_dir "${OUT_DIR_50}" \
  --use_amp_bf16 2>&1 | tee "${OUT_DIR_50}/run.log"

echo ""
echo "✅ 实验2完成: ${OUT_DIR_50}/head_importance.pt"

# 对比分析
echo ""
echo "========================================================"
echo ">>> 对比分析 <<<"
echo "========================================================"

python -c "
import torch
import numpy as np
from scipy.stats import pearsonr

# 读取结果
data1 = torch.load('${OUT_DIR_1}/head_importance.pt', map_location='cpu')
data50 = torch.load('${OUT_DIR_50}/head_importance.pt', map_location='cpu')

# 合并所有layers
scores1 = torch.cat([data1['importance_scores'][i] for i in sorted(data1['importance_scores'].keys())])
scores50 = torch.cat([data50['importance_scores'][i] for i in sorted(data50['importance_scores'].keys())])

print('统计对比：')
print(f'  1条数据:  mean={scores1.mean():.6f}, std={scores1.std():.6f}')
print(f'  50条数据: mean={scores50.mean():.6f}, std={scores50.std():.6f}')
print()
print('相似度分析（逐层IG方法）：')
corr, _ = pearsonr(scores1.numpy(), scores50.numpy())
cos_sim = torch.nn.functional.cosine_similarity(scores1.unsqueeze(0), scores50.unsqueeze(0)).item()
print(f'  Pearson相关系数: {corr:.4f}')
print(f'  余弦相似度: {cos_sim:.4f}')
print(f'  平均绝对差异: {(scores1 - scores50).abs().mean():.6f}')
print(f'  最大绝对差异: {(scores1 - scores50).abs().max():.6f}')
print()
print('对比联合IG方法（之前的结果）：')
print('  Pearson相关系数: 0.9165 (联合IG) vs {:.4f} (逐层IG)'.format(corr))
print('  余弦相似度: 0.9277 (联合IG) vs {:.4f} (逐层IG)'.format(cos_sim))
print()
if corr < 0.9:
    print('✅ 逐层IG方法的区分度显著优于联合IG！')
else:
    print('⚠️  逐层IG方法的区分度仍然较低，可能是模型本身的特性。')
"

echo ""
echo "========================================================"
echo "Finished at: $(date)"
echo "结果保存在："
echo "  - ${OUT_DIR_1}"
echo "  - ${OUT_DIR_50}"
echo "========================================================"

