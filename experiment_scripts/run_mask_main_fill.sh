#!/usr/bin/env bash
set -uo pipefail

# Fill the main causal pruning table with most/least interventions.
# This script only evaluates existing importance files; it does not recompute
# attribution. It is resumable via per-item .done markers under logs/mask_main.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MINERVA_LIMIT_PER_SUBTASK=${MINERVA_LIMIT_PER_SUBTASK:-29}
MODEL_FAMILY=${MODEL_FAMILY:?Set MODEL_FAMILY=llada or dream.}
GPU_ID=${GPU_ID:?Set GPU_ID to an available physical GPU index.}
RUN_TAG=${RUN_TAG:-"mask_main_fill_${MODEL_FAMILY}_$(date +%Y%m%d_%H%M%S)"}
METHODS_STR=${METHODS_STR:-"attnlrp,shapley,headig"}
DATASETS_STR=${DATASETS_STR:-"mmlu,cmmlu,ceval-valid,gpqa_main_n_shot,gsm8k,humaneval,mbpp"}
if [ -z "${PRUNE_MODES:-}" ]; then
  # The AttAttr main-table fill only needs prune-most.  Other historical audit
  # runs retain both tails unless explicitly overridden.
  if [ "${METHODS_STR}" = "attarr" ]; then
    PRUNE_MODES="most"
  else
    PRUNE_MODES="most,least"
  fi
fi

# Historical main-table baseline attribution used gsm8k_full for AttnLRP /
# Shapley / PolyHeadIG. Override to "gsm8k_final_hash,gsm8k_full" for a
# final-answer-target audit once those files exist for every method.
GSM8K_SOURCE_LABELS_STR=${GSM8K_SOURCE_LABELS_STR:-"gsm8k_full,gsm8k_final_hash"}

STATE_DIR="${PROJECT_ROOT}/logs/mask_main/${RUN_TAG}"
STATUS_FILE="${STATE_DIR}/status.tsv"
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

model_context() {
  case "$1" in
    llada)
      MODEL_NAME="llada-1_5"
      PRUNING_RUNNER="${PROJECT_ROOT}/evaluation/llada/run_eval_mask_head_task.sh"
      LAYER_START_DEFAULT=0
      LAYER_END_DEFAULT=31
      ;;
    dream)
      MODEL_NAME="dream"
      PRUNING_RUNNER="${PROJECT_ROOT}/evaluation/dream/run_eval_mask_head_task.sh"
      LAYER_START_DEFAULT=0
      LAYER_END_DEFAULT=27
      ;;
    *)
      echo "ERROR: unsupported MODEL_FAMILY=${MODEL_FAMILY}" >&2
      return 2
      ;;
  esac
}

eval_label_for_task() {
  case "$1" in
    mmlu) echo "mmlu_all" ;;
    cmmlu) echo "cmmlu_all" ;;
    ceval-valid) echo "ceval-valid_all" ;;
    gpqa_main_n_shot) echo "gpqa_main_n_shot_all" ;;
    gsm8k) echo "gsm8k" ;;
    minerva_math) echo "minerva_math" ;;
    humaneval) echo "humaneval" ;;
    mbpp) echo "mbpp" ;;
    *)
      echo "ERROR: unsupported task=$1" >&2
      return 2
      ;;
  esac
}

source_labels_for_task() {
  local task="$1"
  case "${task}" in
    gsm8k) echo "${GSM8K_SOURCE_LABELS_STR}" ;;
    *) eval_label_for_task "${task}" ;;
  esac
}

limit_for_task() {
  case "$1" in
    mmlu|cmmlu) echo 40 ;;
    ceval-valid|gpqa_main_n_shot|gsm8k|humaneval|mbpp) echo 200 ;;
    minerva_math) echo "${MINERVA_LIMIT_PER_SUBTASK}" ;;
    *)
      echo "ERROR: unsupported task=$1" >&2
      return 2
      ;;
  esac
}

safe_name() {
  printf "%s" "$1" | tr '/.' '__' | tr '-' '_'
}

latest_importance_path_for_source() {
  local model_name="$1"
  local method="$2"
  local source_label="$3"
  local prefix=""
  case "${method}" in
    headig) prefix="head_importance_${model_name}_${source_label}_pm" ;;
    attnlrp) prefix="head_importance_${model_name}_${source_label}_attnlrp_" ;;
    shapley) prefix="head_importance_${model_name}_${source_label}_shapley_" ;;
    cokv) prefix="head_importance_${model_name}_${source_label}_cokv_" ;;
    loo) prefix="head_importance_${model_name}_${source_label}_loo_" ;;
    # Restrict AttAttr to the formal K=8 / five-mask-rate protocol so a newer
    # smoke or stability run cannot be selected accidentally.
    attarr) prefix="head_importance_${model_name}_${source_label}_attarr_signed_k8_zero_maskp0.15-0.3-0.5-0.7-0.9_mcs2_mean_masked_" ;;
    *)
      echo "ERROR: unsupported method=${method}" >&2
      return 2
      ;;
  esac

  if [ "${method}" = "headig" ] || [ "${method}" = "attarr" ]; then
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
      -path "*/${prefix}*/head_importance.pt" \
      ! -path "*/${prefix}*_neg/head_importance.pt" \
      ! -path "*/${prefix}*_neg_neg/head_importance.pt" \
      -printf "%T@ %p\n" \
      | sort -nr \
      | head -n 1 \
      | cut -d' ' -f2-
  fi
}

latest_importance_path() {
  local model_name="$1"
  local method="$2"
  local task="$3"
  local labels
  labels="$(source_labels_for_task "${task}")" || return $?
  IFS=',' read -r -a source_labels <<< "${labels}"
  for raw_label in "${source_labels[@]}"; do
    local source_label
    source_label="$(echo "${raw_label}" | xargs)"
    [ -z "${source_label}" ] && continue
    local path
    path="$(latest_importance_path_for_source "${model_name}" "${method}" "${source_label}")"
    if [ -n "${path}" ] && [ -f "${path}" ]; then
      echo "${path}"
      return 0
    fi
  done
  echo ""
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

run_pruning_item() {
  local method="$1"
  local task="$2"
  local eval_label
  eval_label="$(eval_label_for_task "${task}")" || return $?
  local limit
  limit="$(limit_for_task "${task}")" || return $?
  local importance_path
  importance_path="$(latest_importance_path "${MODEL_NAME}" "${method}" "${task}")"

  local item="mask_${MODEL_FAMILY}_${method}_$(safe_name "${task}")"
  local log_path="${STATE_DIR}/${item}.log"
  if [ -z "${importance_path}" ] || [ ! -f "${importance_path}" ]; then
    record_status "${item}" "FAILED" "importance not found"
    return 3
  fi

  local use_negated=0
  # Both methods below attribute the CE loss.  Helpful heads therefore have
  # negative raw signed scores and must be negated exactly once downstream.
  if [ "${method}" = "headig" ] || [ "${method}" = "attarr" ]; then
    use_negated=1
  fi

  local importance_tag="maskfill_${RUN_TAG}_${MODEL_FAMILY}_${method}_$(safe_name "${task}")"
  local common_env=(
    env
    CUDA_VISIBLE_DEVICES="${GPU_ID}"
    HF_HUB_OFFLINE=1
    HF_DATASETS_OFFLINE=1
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    MODEL_NAME="${MODEL_NAME}"
    ATTR_METHOD="${method}"
    ATTR_DATASETS_STR="${eval_label}"
    TASKS_STR="${task}"
    LIMIT="${limit}"
    IMPORTANCE_PATH="${importance_path}"
    IMPORTANCE_TAG="${importance_tag}"
    USE_NEGATED_MODES_STR="${use_negated}"
    LAYER_START="${LAYER_START_DEFAULT}"
    LAYER_END="${LAYER_END_DEFAULT}"
  )

  if [ "${MODEL_FAMILY}" = "llada" ]; then
    run_stage "${item}" "${log_path}" \
      "${common_env[@]}" \
      MODES="${PRUNE_MODES}" \
      PRUNE_SCOPE="layer" \
      PRUNE_K_FRAC="0.2" \
      bash "${PRUNING_RUNNER}"
  else
    local mc_num=1
    if [ "${task}" = "gpqa_main_n_shot" ]; then
      mc_num="${GPQA_MC_NUM:-8}"
    fi
    run_stage "${item}" "${log_path}" \
      "${common_env[@]}" \
      MC_NUM="${mc_num}" \
      PRUNE_WHICH_LIST="${PRUNE_MODES}" \
      MASK_GRANULARITY="kv_group" \
      PRUNE_K_FRAC="0.05" \
      bash "${PRUNING_RUNNER}"
  fi
}

model_context "${MODEL_FAMILY}" || exit $?
IFS=',' read -r -a METHODS <<< "${METHODS_STR}"
IFS=',' read -r -a DATASETS <<< "${DATASETS_STR}"

echo "========================================================"
echo "Mask-main table fill"
echo "========================================================"
echo "Model family: ${MODEL_FAMILY}"
echo "GPU:          ${GPU_ID}"
echo "Run tag:      ${RUN_TAG}"
echo "Methods:      ${METHODS[*]}"
echo "Datasets:     ${DATASETS[*]}"
echo "GSM8K source: ${GSM8K_SOURCE_LABELS_STR}"
echo "State dir:    ${STATE_DIR}"
echo "Started:      $(timestamp)"
echo "========================================================"

FAILED_ITEMS=()
for raw_method in "${METHODS[@]}"; do
  method="$(echo "${raw_method}" | xargs)"
  [ -z "${method}" ] && continue
  for raw_task in "${DATASETS[@]}"; do
    task="$(echo "${raw_task}" | xargs)"
    [ -z "${task}" ] && continue
    run_pruning_item "${method}" "${task}" || FAILED_ITEMS+=("${method}:${task}")
  done
done

echo "========================================================"
echo "Finished: $(timestamp)"
echo "State:    ${STATE_DIR}"
if [ "${#FAILED_ITEMS[@]}" -gt 0 ]; then
  echo "Failed items: ${FAILED_ITEMS[*]}"
  exit 1
fi
echo "All mask-main items completed."
echo "========================================================"
