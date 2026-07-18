#!/usr/bin/env bash
# Resumable per-model queue.  One instance is intended per physical GPU.

set -uo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/qiheng/Projects/adaptive-dllm}
MODEL_FAMILY=${MODEL_FAMILY:?Set MODEL_FAMILY=llada or dream}
GPU_ID=${GPU_ID:?Set GPU_ID}
WAIT_FOR_FREE_MEMORY_MIB=${WAIT_FOR_FREE_MEMORY_MIB:-30000}
CURRENT_DATASETS=${CURRENT_DATASETS:-mmlu,cmmlu,ceval-valid,gpqa_main_n_shot,gsm8k,humaneval,mbpp}
QUEUE_TAG=${QUEUE_TAG:-formal_queue_${MODEL_FAMILY}_20260717_r1}

case "$MODEL_FAMILY" in
  dream)
    CURRENT_TAG=matched_ara_cokv_formal_dream_20260714_r1
    RECAL_SPECS=gsm8k:97,humaneval:126
    TIMING_CONTEXT=shared_gpu_4
    MODEL_SEED=47
    ;;
  llada)
    CURRENT_TAG=matched_ara_cokv_formal_llada_20260714_r1
    RECAL_SPECS=gsm8k:32,humaneval:51
    TIMING_CONTEXT=shared_gpu_5
    MODEL_SEED=123
    ;;
  *) echo "Unsupported MODEL_FAMILY=$MODEL_FAMILY" >&2; exit 2 ;;
esac
MINERVA_TAG=minerva_matched_${MODEL_FAMILY}_20260717_r1
MINERVA_BASELINE_TAG=minerva_baselines_${MODEL_FAMILY}_20260717_r1

STATE_DIR="$PROJECT_ROOT/logs/formal_queue/$QUEUE_TAG"
STATUS_FILE="$STATE_DIR/status.tsv"
mkdir -p "$STATE_DIR"
touch "$STATUS_FILE"

timestamp() { date '+%F %T'; }
record_status() {
  printf '%s\t%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" "${3:-}" >> "$STATUS_FILE"
  echo "[$(timestamp)] item=$1 status=$2 ${3:-}"
}

run_stage() {
  local item=$1 log=$2
  shift 2
  local marker="$STATE_DIR/${item}.done"
  if [[ -f "$marker" ]]; then
    record_status "$item" SKIP already_completed
    return 0
  fi
  record_status "$item" RUNNING "log=$log"
  "$@" > "$log" 2>&1
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    touch "$marker"
    record_status "$item" DONE
  else
    record_status "$item" FAILED "rc=$rc"
  fi
  return "$rc"
}

record_status gpu_wait RUNNING "gpu=$GPU_ID required_free_mib=$WAIT_FOR_FREE_MEMORY_MIB"
while true; do
  free_mib=$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits \
    | head -n 1 | tr -d ' ')
  if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= WAIT_FOR_FREE_MEMORY_MIB )); then
    break
  fi
  sleep 30
done
record_status gpu_wait DONE "gpu=$GPU_ID free_mib=$free_mib"

run_stage current_recalibration "$STATE_DIR/current_recalibration.log" env \
  MODEL_FAMILY="$MODEL_FAMILY" GPU_ID="$GPU_ID" RUN_TAG="$CURRENT_TAG" \
  TASK_SPECS="$RECAL_SPECS" REUSE_EXISTING=1 RECALIBRATION_ROUND=1 \
  TIMING_CONTEXT="$TIMING_CONTEXT" \
  bash "$PROJECT_ROOT/experiment_scripts/recalibrate_matched_cokv.sh" || exit $?

run_stage current_adaptive "$STATE_DIR/current_adaptive.log" env \
  GPU_ID="$GPU_ID" RUN_TAG="${CURRENT_TAG}_adaptive" ATTR_MAIN_ONLY=1 \
  MODEL_FAMILIES_STR="$MODEL_FAMILY" METHODS_STR=headig,cokv \
  DATASETS_STR="$CURRENT_DATASETS" \
  bash "$PROJECT_ROOT/experiment_scripts/run_adaptive_table_fill.sh" || exit $?

run_stage current_pruning "$STATE_DIR/current_pruning.log" env \
  GPU_ID="$GPU_ID" MODEL_FAMILY="$MODEL_FAMILY" RUN_TAG="${CURRENT_TAG}_pruning" \
  METHODS_STR=headig,cokv DATASETS_STR="$CURRENT_DATASETS" \
  GSM8K_SOURCE_LABELS_STR=gsm8k_final_hash PRUNE_MODES=most \
  bash "$PROJECT_ROOT/experiment_scripts/run_mask_main_fill.sh" || exit $?

run_stage minerva_matched "$STATE_DIR/minerva_matched.log" env \
  MODEL_FAMILY="$MODEL_FAMILY" GPU_ID="$GPU_ID" RUN_TAG="$MINERVA_TAG" \
  DATASETS_STR=minerva_math MAX_SAMPLES=200 COKV_PILOT_SAMPLES=8 RESUME=1 \
  bash "$PROJECT_ROOT/experiment_scripts/run_matched_ara_cokv_pipeline.sh" || exit $?

run_stage minerva_baselines "$STATE_DIR/minerva_baselines.log" env \
  MODEL_FAMILY="$MODEL_FAMILY" GPU_ID="$GPU_ID" RUN_TAG="$MINERVA_BASELINE_TAG" \
  METHODS_STR=attarr,attnlrp,loo MAX_SAMPLES=200 EVAL_LIMIT=200 \
  RUN_DENSE=1 RUN_ADAPTIVE=1 RUN_PRUNING=1 \
  bash "$PROJECT_ROOT/experiment_scripts/run_minerva_baselines_pipeline.sh" || exit $?

run_stage minerva_path_ablation "$STATE_DIR/minerva_path_ablation.log" env \
  MODEL_FAMILY="$MODEL_FAMILY" GPU_ID="$GPU_ID" \
  RUN_TAG="minerva_path_${MODEL_FAMILY}_20260717_r1" \
  ATTR_DATASET=minerva_math TASK=minerva_math ATTR_MAX_SAMPLES=200 EVAL_LIMIT=200 \
  IG_STEPS=8 MASK_PROBS="0.15,0.3,0.5,0.7,0.9" MASK_SAMPLES_PER_PROB=2 \
  SEED="$MODEL_SEED" DATA_SEED="$MODEL_SEED" MASK_SEED="$MODEL_SEED" \
  PATH_SEED="$MODEL_SEED" RUN_ADAPTIVE=1 \
  bash "$PROJECT_ROOT/experiment_scripts/run_path_ablation_mmlu.sh" || exit $?

record_status queue DONE "model=$MODEL_FAMILY gpu=$GPU_ID"
