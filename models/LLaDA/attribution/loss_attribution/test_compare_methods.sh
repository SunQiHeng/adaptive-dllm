#!/usr/bin/env bash
# 快速对比逐层IG vs 联合IG的结果（小规模测试）
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH=/home/qiheng/Projects/adaptive-dllm:${PYTHONPATH:-}

MODEL_PATH="/data/qh_models/LLaDA-1.5"
OUT_ROOT="/tmp/ig_comparison_test"
mkdir -p "$OUT_ROOT"

# 共同参数（小规模测试）
COMMON_ARGS="
  --model_path $MODEL_PATH \
  --dataset gsm8k \
  --dataset_config main \
  --split test \
  --max_samples 5 \
  --seed 42 \
  --ig_steps 4 \
  --max_length 512 \
  --mask_probs 1.0 \
  --mask_samples_per_prob 1 \
  --baseline zero \
  --layer_start 0 \
  --layer_end 2 \
  --use_amp_bf16 \
  --debug_gate
"

echo "=== 运行逐层IG（原版）==="
python /home/qiheng/Projects/adaptive-dllm/models/LLaDA/attribution/loss_attribution/compute_loss_attribution.py \
  $COMMON_ARGS \
  --output_dir "$OUT_ROOT/layerwise" 2>&1 | tee "$OUT_ROOT/layerwise.log"

echo ""
echo "=== 运行联合IG（新版all-heads）==="
python /home/qiheng/Projects/adaptive-dllm/models/LLaDA/attribution/loss_attribution/compute_loss_attribution_all_heads.py \
  $COMMON_ARGS \
  --output_dir "$OUT_ROOT/joint" 2>&1 | tee "$OUT_ROOT/joint.log"

echo ""
echo "=== 对比结果 ==="
python3 - <<'PYEOF'
import torch
lw = torch.load("/tmp/ig_comparison_test/layerwise/head_importance.pt", map_location="cpu")
jt = torch.load("/tmp/ig_comparison_test/joint/head_importance.pt", map_location="cpu")

lw_scores = lw["importance_scores"]
jt_scores = jt["importance_scores"]

print("逐层版本 layers:", sorted(lw_scores.keys()))
print("联合版本 layers:", sorted(jt_scores.keys()))

for layer in sorted(lw_scores.keys()):
    if layer not in jt_scores:
        print(f"⚠️  Layer {layer} missing in joint version")
        continue
    
    lw_vec = lw_scores[layer].float()
    jt_vec = jt_scores[layer].float()
    
    if lw_vec.shape != jt_vec.shape:
        print(f"❌ Layer {layer} shape mismatch: {lw_vec.shape} vs {jt_vec.shape}")
        continue
    
    diff = (lw_vec - jt_vec).abs()
    rel_diff = diff / (lw_vec.abs() + 1e-8)
    
    print(f"Layer {layer}:")
    print(f"  逐层: mean={lw_vec.mean():.6f}, std={lw_vec.std():.6f}")
    print(f"  联合: mean={jt_vec.mean():.6f}, std={jt_vec.std():.6f}")
    print(f"  差异: max_abs={diff.max():.6f}, mean_abs={diff.mean():.6f}, max_rel={rel_diff.max():.6f}")
    
    # 检查相关性
    if len(lw_vec) > 1:
        corr = torch.corrcoef(torch.stack([lw_vec, jt_vec]))[0, 1].item()
        print(f"  相关系数: {corr:.6f}")
    print()

print("✅ 对比完成")
PYEOF

echo ""
echo "日志位置: $OUT_ROOT/{layerwise,joint}.log"
echo "结果位置: $OUT_ROOT/{layerwise,joint}/head_importance.pt"

