#!/usr/bin/env bash
set -uo pipefail

# Fill currently missing non-Minerva cells in the main adaptive tables:
# - Dense results for both models.
# - LLaDA GPQA: AttnLRP, Shapley, PolyHeadIG.
# - Dream GPQA, HumanEval, MBPP: AttnLRP, Shapley, PolyHeadIG.
#
# Score direction is explicit:
# - PolyHeadIG and AttAttr raw signed loss attribution: USE_NEGATED=1.
# - AttnLRP and Shapley: USE_NEGATED=0.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID=${GPU_ID:?Set GPU_ID to an available physical GPU index.}
RUN_TAG=${RUN_TAG:-"adaptive_table_fill_$(date +%Y%m%d_%H%M%S)"}
STATE_DIR="${PROJECT_ROOT}/logs/adaptive_table_fill/${RUN_TAG}"
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
      echo "ERROR: unsupported task: $1" >&2
      return 2
      ;;
  esac
}

source_label_for_task() {
  local method="$1"
  local task="$2"
  case "${task}" in
    # The formal AttAttr runner uses final-answer attribution for GSM8K.
    gsm8k)
      if [ "${method}" = "attarr" ] || [ "${method}" = "headig" ] || [ "${method}" = "cokv" ]; then
        echo "gsm8k_final_hash"
      else
        echo "gsm8k_full"
      fi
      ;;
    *) eval_label_for_task "${task}" ;;
  esac
}

limit_for_task() {
  case "$1" in
    mmlu|cmmlu) echo 40 ;;
    ceval-valid|gpqa_main_n_shot|gsm8k|minerva_math|humaneval|mbpp) echo 200 ;;
    *)
      echo "ERROR: unsupported task: $1" >&2
      return 2
      ;;
  esac
}

model_context() {
  case "$1" in
    llada)
      MODEL_NAME="llada-1_5"
      EVAL_RUNNER="${PROJECT_ROOT}/evaluation/llada/run_eval_task.sh"
      ;;
    dream)
      MODEL_NAME="dream"
      EVAL_RUNNER="${PROJECT_ROOT}/evaluation/dream/run_eval_task.sh"
      ;;
    *)
      echo "ERROR: unsupported model family: $1" >&2
      return 2
      ;;
  esac
}

latest_importance_path() {
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
    # Exclude the earlier K=2/K=4 smoke/stability artifacts.
    attarr) prefix="head_importance_${model_name}_${source_label}_attarr_signed_k8_zero_maskp0.15-0.3-0.5-0.7-0.9_mcs2_mean_masked_" ;;
    *)
      echo "ERROR: unsupported attribution method: ${method}" >&2
      return 2
      ;;
  esac

  if [ "${method}" = "headig" ] || [ "${method}" = "attarr" ]; then
    # HeadIG raw CE-loss attributions are negated exactly once via USE_NEGATED=1.
    # Avoid selecting materialized *_neg folders and negating them again.
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
  rc=$?
  if [ "${rc}" -eq 0 ]; then
    touch "${marker}"
    record_status "${item}" "DONE"
  else
    record_status "${item}" "FAILED" "rc=${rc}"
  fi
  return "${rc}"
}

run_dense() {
  local family="$1"
  local task="$2"
  model_context "${family}" || return $?
  local limit
  limit="$(limit_for_task "${task}")" || return $?
  local item="dense_${family}_${task//-/_}"
  local log_path="${STATE_DIR}/${item}.log"

  run_stage "${item}" "${log_path}" env \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    MODEL_NAME="${MODEL_NAME}" \
    ATTR_METHOD="headig" \
    ATTR_DATASETS_STR="mmlu_all" \
    TASKS_STR="${task}" \
    LIMIT="${limit}" \
    IMPORTANCE_TAG="tablefill_${item}_${RUN_TAG}" \
    USE_NEGATED=0 \
    USE_NEGATED_MODES_STR=0 \
    MODEL_TYPES_STR="standard" \
    bash "${EVAL_RUNNER}"
}

run_adaptive() {
  local family="$1"
  local method="$2"
  local task="$3"
  model_context "${family}" || return $?
  local eval_label
  local source_label
  local limit
  eval_label="$(eval_label_for_task "${task}")" || return $?
  source_label="$(source_label_for_task "${method}" "${task}")" || return $?
  limit="$(limit_for_task "${task}")" || return $?

  local importance_path
  importance_path="$(latest_importance_path "${MODEL_NAME}" "${method}" "${source_label}")"
  if [ -z "${importance_path}" ] || [ ! -f "${importance_path}" ]; then
    record_status "adaptive_${family}_${method}_${task//-/_}" "FAILED" "importance not found for source_label=${source_label}"
    return 3
  fi

  local use_negated=0
  if [ "${method}" = "headig" ] || [ "${method}" = "attarr" ]; then
    use_negated=1
  fi
  local mc_num=1
  if [ "${task}" = "gpqa_main_n_shot" ]; then
    mc_num="${GPQA_MC_NUM:-8}"
  fi

  local item="adaptive_${family}_${method}_${task//-/_}"
  local log_path="${STATE_DIR}/${item}.log"
  run_stage "${item}" "${log_path}" env \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    MODEL_NAME="${MODEL_NAME}" \
    ATTR_METHOD="${method}" \
    ATTR_DATASETS_STR="${eval_label}" \
    IMPORTANCE_PATH="${importance_path}" \
    IMPORTANCE_TAG="tablefill_${item}_correct_${RUN_TAG}" \
    LIMIT="${limit}" \
    MC_NUM="${mc_num}" \
    USE_NEGATED="${use_negated}" \
    USE_NEGATED_MODES_STR="${use_negated}" \
    MODEL_TYPES_STR="adaptive" \
    bash "${EVAL_RUNNER}"
}

echo "========================================================"
echo "Adaptive Table Fill"
echo "========================================================"
echo "GPU:       ${GPU_ID}"
echo "Run tag:   ${RUN_TAG}"
echo "State dir: ${STATE_DIR}"
echo "Started:   $(timestamp)"
echo "========================================================"

FAILED_ITEMS=()
if [ "${ATTR_MAIN_ONLY:-${ATTARR_MAIN_ONLY:-0}}" = "1" ]; then
  IFS=',' read -r -a MAIN_FAMILIES <<< "${MODEL_FAMILIES_STR:-llada,dream}"
  IFS=',' read -r -a MAIN_METHODS <<< "${METHODS_STR:-attarr}"
  IFS=',' read -r -a MAIN_TASKS <<< "${DATASETS_STR:-mmlu,cmmlu,ceval-valid,gpqa_main_n_shot,gsm8k,humaneval,mbpp}"
  for raw_family in "${MAIN_FAMILIES[@]}"; do
    family="$(echo "${raw_family}" | xargs)"
    [ -z "${family}" ] && continue
    for raw_method in "${MAIN_METHODS[@]}"; do
      method="$(echo "${raw_method}" | xargs)"
      [ -z "${method}" ] && continue
      for raw_task in "${MAIN_TASKS[@]}"; do
        task="$(echo "${raw_task}" | xargs)"
        [ -z "${task}" ] && continue
        run_adaptive "${family}" "${method}" "${task}" || FAILED_ITEMS+=("adaptive:${family}:${method}:${task}")
      done
    done
  done
else
  DENSE_TASKS=(mmlu cmmlu ceval-valid gpqa_main_n_shot gsm8k humaneval mbpp)
  for family in llada dream; do
    for task in "${DENSE_TASKS[@]}"; do
      run_dense "${family}" "${task}" || FAILED_ITEMS+=("dense:${family}:${task}")
    done
  done

  for method in attnlrp shapley headig; do
    run_adaptive llada "${method}" gpqa_main_n_shot || FAILED_ITEMS+=("adaptive:llada:${method}:gpqa")
  done

  for task in gpqa_main_n_shot humaneval mbpp; do
    for method in attnlrp shapley headig; do
      run_adaptive dream "${method}" "${task}" || FAILED_ITEMS+=("adaptive:dream:${method}:${task}")
    done
  done
fi

echo "========================================================"
echo "Finished: $(timestamp)"
echo "State:    ${STATE_DIR}"
if [ "${#FAILED_ITEMS[@]}" -gt 0 ]; then
  echo "Failed items: ${FAILED_ITEMS[*]}"
  exit 1
fi
echo "All queued table-fill items completed."
echo "========================================================"
