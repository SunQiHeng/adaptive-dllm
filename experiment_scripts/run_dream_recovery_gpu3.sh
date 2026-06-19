#!/usr/bin/env bash
set -uo pipefail

# Recovery queue for two audited Dream anomalies:
# 1. GPQA MC likelihood variance: complete MC_NUM=8 for baseline methods.
# 2. MBPP code-eval postprocessing: rerun missing full-table cells after the
#    extraction fix in evaluation/dream/eval_dream.py.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID=${GPU_ID:?Set GPU_ID to an available physical GPU index.}
RUN_TAG=${RUN_TAG:-"dream_recovery_$(date +%Y%m%d_%H%M%S)"}
STATE_DIR="${PROJECT_ROOT}/logs/recovery/${RUN_TAG}"
STATUS_FILE="${STATE_DIR}/status.tsv"
EVAL_RUNNER="${PROJECT_ROOT}/evaluation/dream/run_eval_task.sh"
mkdir -p "${STATE_DIR}"
touch "${STATUS_FILE}"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

record_status() {
  local item="$1"
  local status="$2"
  local detail="${3:-}"
  printf "%s\t%s\t%s\t%s\n" "$(timestamp)" "${item}" "${status}" "${detail}" >> "${STATUS_FILE}"
  echo "[$(timestamp)] item=${item} status=${status} ${detail}"
}

run_stage() {
  local item="$1"
  local log_path="$2"
  shift 2
  local marker="${STATE_DIR}/${item}.done"
  if [ -f "${marker}" ]; then
    record_status "${item}" "SKIP" "already completed"
    return 0
  fi

  record_status "${item}" "RUNNING" "log=${log_path}"
  "$@" > "${log_path}" 2>&1
  local rc=$?
  if [ "${rc}" -eq 0 ]; then
    touch "${marker}"
    record_status "${item}" "DONE"
  else
    record_status "${item}" "FAILED" "rc=${rc}"
  fi
  return "${rc}"
}

latest_importance_path() {
  local method="$1"
  local source_label="$2"
  local prefix=""
  case "${method}" in
    headig) prefix="head_importance_dream_${source_label}_pm" ;;
    attnlrp) prefix="head_importance_dream_${source_label}_attnlrp_" ;;
    shapley) prefix="head_importance_dream_${source_label}_shapley_" ;;
    loo) prefix="head_importance_dream_${source_label}_loo_" ;;
    *)
      echo "ERROR: unsupported attribution method: ${method}" >&2
      return 2
      ;;
  esac

  if [ "${method}" = "headig" ]; then
    # HeadIG raw loss-attribution scores must be negated exactly once downstream.
    # Exclude already-negated materialized folders to avoid accidental double
    # negation when USE_NEGATED=1.
    find "${PROJECT_ROOT}/configs/aconfigs" -mindepth 2 -maxdepth 2 -type f \
      -path "*/${prefix}*/head_importance.pt" \
      ! -path "*/${prefix}*_neg/head_importance.pt" \
      ! -path "*/${prefix}*_neg_neg/head_importance.pt" \
      -printf "%T@ %p\n" \
      | sort -nr \
      | head -n 1 \
      | cut -d' ' -f2-
  else
    find "${PROJECT_ROOT}/configs/aconfigs" -mindepth 2 -maxdepth 2 -type f \
      -path "*/${prefix}*/head_importance.pt" -printf "%T@ %p\n" \
      | sort -nr \
      | head -n 1 \
      | cut -d' ' -f2-
  fi
}

run_dream_standard() {
  local item="$1"
  local attr_label="$2"
  local task="$3"
  local limit="$4"
  local extra_tag="$5"
  local mc_num="${6:-}"
  local log_path="${STATE_DIR}/${item}.log"

  local env_args=(
    env
    CUDA_VISIBLE_DEVICES="${GPU_ID}"
    HF_DATASETS_OFFLINE=1
    HF_HUB_OFFLINE=1
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    MODEL_NAME="dream"
    ATTR_METHOD="headig"
    ATTR_DATASETS_STR="${attr_label}"
    TASKS_STR="${task}"
    LIMIT="${limit}"
    IMPORTANCE_TAG="recovery_${RUN_TAG}_${extra_tag}"
    USE_NEGATED=0
    USE_NEGATED_MODES_STR=0
    MODEL_TYPES_STR="standard"
  )
  if [ -n "${mc_num}" ]; then
    env_args+=(MC_NUM="${mc_num}")
  fi

  run_stage "${item}" "${log_path}" "${env_args[@]}" bash "${EVAL_RUNNER}"
}

run_dream_adaptive() {
  local item="$1"
  local method="$2"
  local source_label="$3"
  local attr_label="$4"
  local task="$5"
  local limit="$6"
  local extra_tag="$7"
  local mc_num="${8:-}"
  local log_path="${STATE_DIR}/${item}.log"

  local importance_path
  importance_path="$(latest_importance_path "${method}" "${source_label}")"
  if [ -z "${importance_path}" ] || [ ! -f "${importance_path}" ]; then
    record_status "${item}" "FAILED" "importance not found method=${method} source=${source_label}"
    return 3
  fi

  local use_negated=0
  if [ "${method}" = "headig" ]; then
    use_negated=1
  fi

  local env_args=(
    env
    CUDA_VISIBLE_DEVICES="${GPU_ID}"
    HF_DATASETS_OFFLINE=1
    HF_HUB_OFFLINE=1
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    MODEL_NAME="dream"
    ATTR_METHOD="${method}"
    ATTR_DATASETS_STR="${attr_label}"
    TASKS_STR="${task}"
    LIMIT="${limit}"
    IMPORTANCE_PATH="${importance_path}"
    IMPORTANCE_TAG="recovery_${RUN_TAG}_${extra_tag}"
    USE_NEGATED="${use_negated}"
    USE_NEGATED_MODES_STR="${use_negated}"
    MODEL_TYPES_STR="adaptive"
  )
  if [ -n "${mc_num}" ]; then
    env_args+=(MC_NUM="${mc_num}")
  fi

  run_stage "${item}" "${log_path}" "${env_args[@]}" bash "${EVAL_RUNNER}"
}

echo "========================================================"
echo "Dream recovery queue"
echo "========================================================"
echo "GPU:       ${GPU_ID}"
echo "Run tag:   ${RUN_TAG}"
echo "State dir: ${STATE_DIR}"
echo "Started:   $(timestamp)"
echo "========================================================"

FAILED_ITEMS=()

# Complete the GPQA MC_NUM=8 audit. Dense and LOO MC8 are already available
# from run tag dream_protocol_audit_20260608_gpu4.
run_dream_adaptive "gpqa_attnlrp_mc8" "attnlrp" "gpqa_main_n_shot_all" "gpqa_main_n_shot_all" "gpqa_main_n_shot" 200 "gpqa_attnlrp_mc8" 8 || FAILED_ITEMS+=("gpqa_attnlrp_mc8")
run_dream_adaptive "gpqa_shapley_mc8" "shapley" "gpqa_main_n_shot_all" "gpqa_main_n_shot_all" "gpqa_main_n_shot" 200 "gpqa_shapley_mc8" 8 || FAILED_ITEMS+=("gpqa_shapley_mc8")
run_dream_adaptive "gpqa_headig_mc8" "headig" "gpqa_main_n_shot_all" "gpqa_main_n_shot_all" "gpqa_main_n_shot" 200 "gpqa_headig_mc8" 8 || FAILED_ITEMS+=("gpqa_headig_mc8")

# Complete the Dream MBPP postprocess-fixed full adaptive table cells.
run_dream_standard "mbpp_dense_full" "mbpp" "mbpp" 200 "mbpp_dense_full" || FAILED_ITEMS+=("mbpp_dense_full")
run_dream_adaptive "mbpp_attnlrp_full" "attnlrp" "mbpp" "mbpp" "mbpp" 200 "mbpp_attnlrp_full" || FAILED_ITEMS+=("mbpp_attnlrp_full")
run_dream_adaptive "mbpp_headig_full" "headig" "mbpp" "mbpp" "mbpp" 200 "mbpp_headig_full" || FAILED_ITEMS+=("mbpp_headig_full")
run_dream_adaptive "mbpp_loo_full" "loo" "mbpp" "mbpp" "mbpp" 200 "mbpp_loo_full" || FAILED_ITEMS+=("mbpp_loo_full")

echo "========================================================"
echo "Finished: $(timestamp)"
echo "State:    ${STATE_DIR}"
if [ "${#FAILED_ITEMS[@]}" -gt 0 ]; then
  echo "Failed items: ${FAILED_ITEMS[*]}"
  exit 1
fi
echo "All recovery items completed."
echo "========================================================"
