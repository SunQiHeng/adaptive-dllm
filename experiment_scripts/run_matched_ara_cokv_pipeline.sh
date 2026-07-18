#!/usr/bin/env bash
set -uo pipefail

# Formal matched-data experiment for ARA and CoKV.
#
# For every task, this pipeline:
#   1. computes ARA on the shared 200-example attribution protocol;
#   2. runs a short CoKV pilot on exactly the same rows and masking setup;
#   3. chooses CoKV's sampling count to match ARA's measured wall time;
#   4. verifies the complete ordered data manifest is identical;
#   5. evaluates both rankings with the same adaptive and pruning harnesses.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_FAMILY=${MODEL_FAMILY:?Set MODEL_FAMILY=llada or dream.}
GPU_ID=${GPU_ID:?Set GPU_ID to the physical GPU reserved for this pipeline.}
WAIT_FOR_PID=${WAIT_FOR_PID:-""}
RUN_TAG=${RUN_TAG:-"matched_ara_cokv_${MODEL_FAMILY}_$(date +%Y%m%d_%H%M%S)"}
DATASETS_STR=${DATASETS_STR:-"mmlu,cmmlu,ceval-valid,gpqa_main_n_shot,gsm8k,humaneval,mbpp"}
MAX_SAMPLES=${MAX_SAMPLES:-200}
COKV_PILOT_SAMPLES=${COKV_PILOT_SAMPLES:-8}
COKV_MIN_SAMPLES=${COKV_MIN_SAMPLES:-8}
COKV_MAX_SAMPLES=${COKV_MAX_SAMPLES:-512}
GPQA_MC_NUM=${GPQA_MC_NUM:-8}
RESUME=${RESUME:-0}

CONDA_ENV_BIN="${HOME}/miniconda3/envs/adaptive-dllm/bin"
if [ -d "${CONDA_ENV_BIN}" ]; then
  export PATH="${CONDA_ENV_BIN}:${PATH}"
fi
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

case "${MODEL_FAMILY}" in
  llada)
    MODEL_NAME="llada-1_5"
    DATA_SEED=123
    ARA_RUNNER="${PROJECT_ROOT}/models/LLaDA/attribution/loss_attribution/run_loss_attribution_all_heads.sh"
    COKV_RUNNER="${PROJECT_ROOT}/models/LLaDA/attribution/baseline_attribution/run_shapley_head_attribution.sh"
    ;;
  dream)
    MODEL_NAME="dream"
    DATA_SEED=47
    ARA_RUNNER="${PROJECT_ROOT}/models/Dream/attribution/loss_attribution/run_loss_attribution_all_heads.sh"
    COKV_RUNNER="${PROJECT_ROOT}/models/Dream/attribution/baseline_attribution/run_shapley_head_attribution.sh"
    ;;
  *)
    echo "ERROR: unsupported MODEL_FAMILY=${MODEL_FAMILY}" >&2
    exit 2
    ;;
esac

STATE_DIR="${PROJECT_ROOT}/logs/matched_ara_cokv/${RUN_TAG}"
STATUS_FILE="${STATE_DIR}/status.tsv"
TIMING_FILE="${STATE_DIR}/timing.tsv"
mkdir -p "${STATE_DIR}"
touch "${STATUS_FILE}"
if [ ! -f "${TIMING_FILE}" ]; then
  printf "task\tara_seconds\tpilot_samples\tpilot_seconds\tcokv_samples\tpredicted_cokv_seconds\tcokv_seconds\ttime_ratio\tara_path\tcokv_path\n" > "${TIMING_FILE}"
fi

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

safe_name() {
  printf "%s" "$1" | tr '/.' '__' | tr '-' '_'
}

source_label_for_task() {
  case "$1" in
    mmlu) echo "mmlu_all" ;;
    cmmlu) echo "cmmlu_all" ;;
    ceval-valid) echo "ceval-valid_all" ;;
    gpqa_main_n_shot) echo "gpqa_main_n_shot_all" ;;
    gsm8k) echo "gsm8k_final_hash" ;;
    minerva_math) echo "minerva_math" ;;
    humaneval) echo "humaneval" ;;
    mbpp) echo "mbpp" ;;
    *) echo "ERROR: unsupported task=$1" >&2; return 2 ;;
  esac
}

find_ara_path() {
  local task="$1"
  local token="$2"
  local source_label
  source_label="$(source_label_for_task "${task}")" || return $?
  find "${PROJECT_ROOT}/configs/aconfigs" -mindepth 2 -maxdepth 2 -type f \
    -path "*/head_importance_${MODEL_NAME}_${source_label}_pmrandom_threshold_ts${token}/head_importance.pt" \
    -print | head -n 1
}

find_cokv_path() {
  local task="$1"
  local token="$2"
  local samples="$3"
  local source_label
  source_label="$(source_label_for_task "${task}")" || return $?
  find "${PROJECT_ROOT}/configs/aconfigs" -mindepth 2 -maxdepth 2 -type f \
    -path "*/head_importance_${MODEL_NAME}_${source_label}_cokv_signed_M${samples}_*_ts${token}/head_importance.pt" \
    -print | head -n 1
}

find_existing_cokv_path() {
  local task="$1"
  local token="$2"
  local source_label
  source_label="$(source_label_for_task "${task}")" || return $?
  find "${PROJECT_ROOT}/configs/aconfigs" -mindepth 2 -maxdepth 2 -type f \
    -path "*/head_importance_${MODEL_NAME}_${source_label}_cokv_signed_M*_ts${token}/head_importance.pt" \
    -print | head -n 1
}

cokv_sampling_number() {
  local path="$1"
  python - "${path}" <<'PY'
import sys
import torch

obj = torch.load(sys.argv[1], map_location="cpu")
meta = obj.get("metadata", {}) if isinstance(obj, dict) else {}
value = int(meta.get("sampling_number", -1))
if value <= 0:
    raise SystemExit("missing or invalid CoKV sampling_number")
print(value)
PY
}

validate_matching_pair() {
  local ara_path="$1"
  local cokv_path="$2"
  python - "${ara_path}" "${cokv_path}" <<'PY'
import sys
import torch

ara = torch.load(sys.argv[1], map_location="cpu")
cokv = torch.load(sys.argv[2], map_location="cpu")
ara_meta = ara.get("metadata", {}) if isinstance(ara, dict) else {}
cokv_meta = cokv.get("metadata", {}) if isinstance(cokv, dict) else {}
ara_manifest = ara_meta.get("rows_manifest_sha256")
cokv_manifest = cokv_meta.get("rows_manifest_sha256")
if not ara_manifest or ara_manifest != cokv_manifest:
    raise SystemExit("ARA and CoKV manifests do not match")
PY
}

validate_importance() {
  local path="$1"
  local expected_samples="$2"
  local expected_method="$3"
  python - "${path}" "${MAX_SAMPLES}" "${DATA_SEED}" "${expected_samples}" "${expected_method}" <<'PY'
import sys
import torch

path, max_samples, seed, expected_samples, method = sys.argv[1:]
obj = torch.load(path, map_location="cpu")
scores = obj.get("importance_scores", obj.get("head_importance", obj)) if isinstance(obj, dict) else obj
if isinstance(scores, dict):
    vals = torch.cat([torch.as_tensor(v).reshape(-1).float() for _, v in sorted(scores.items(), key=lambda kv: int(kv[0]))])
else:
    vals = torch.as_tensor(scores).reshape(-1).float()
if vals.numel() == 0 or not bool(torch.isfinite(vals).all()):
    raise SystemExit("empty or non-finite head importance")
meta = obj.get("metadata", {}) if isinstance(obj, dict) else {}
if int(meta.get("max_samples", -1)) != int(max_samples):
    raise SystemExit(f"max_samples mismatch: {meta.get('max_samples')} != {max_samples}")
if int(meta.get("data_seed", -1)) != int(seed) or int(meta.get("mask_seed", -1)) != int(seed):
    raise SystemExit(f"seed mismatch: data={meta.get('data_seed')} mask={meta.get('mask_seed')} expected={seed}")
if not meta.get("rows_manifest_sha256"):
    raise SystemExit("missing rows_manifest_sha256")
if method == "ara":
    if int(meta.get("ig_steps", -1)) != 8 or int(meta.get("path_samples", -1)) != 4:
        raise SystemExit("ARA must use K=8 and P=4")
elif method == "cokv":
    if int(meta.get("sampling_number", -1)) != int(expected_samples):
        raise SystemExit(f"CoKV sampling mismatch: {meta.get('sampling_number')} != {expected_samples}")
print(f"validated={path} scores={vals.numel()} rows={meta.get('rows_loaded')} manifest={meta['rows_manifest_sha256']}")
PY
}

calibrate_cokv_samples() {
  local pilot_path="$1"
  local ara_seconds="$2"
  local pilot_seconds="$3"
  python - "${pilot_path}" "${ara_seconds}" "${pilot_seconds}" "${COKV_MIN_SAMPLES}" "${COKV_MAX_SAMPLES}" <<'PY'
import sys
import torch

path, ara_s, pilot_s, lower, upper = sys.argv[1:]
obj = torch.load(path, map_location="cpu")
meta = obj.get("metadata", {})
m0 = int(meta["sampling_number"])
sampling_s = float(meta["sampling_seconds"])
if m0 <= 0 or sampling_s <= 0:
    raise SystemExit("invalid CoKV pilot timing")
per_sample = sampling_s / m0
fixed = max(0.0, float(pilot_s) - sampling_s)
raw = (float(ara_s) - fixed) / per_sample
target = max(int(lower), min(int(upper), int(round(raw))))
predicted = fixed + target * per_sample
print(f"{target}\t{predicted:.6f}\t{fixed:.6f}\t{per_sample:.6f}")
PY
}

annotate_and_compare() {
  local task="$1"
  local ara_path="$2"
  local cokv_path="$3"
  local ara_seconds="$4"
  local cokv_seconds="$5"
  local cokv_samples="$6"
  local predicted_seconds="$7"
  python - "${task}" "${ara_path}" "${cokv_path}" "${ara_seconds}" "${cokv_seconds}" "${cokv_samples}" "${predicted_seconds}" "${COKV_PILOT_SAMPLES}" <<'PY'
import sys
from datetime import datetime
import torch

task, ara_path, cokv_path, ara_s, cokv_s, cokv_m, predicted_s, pilot_m = sys.argv[1:]
ara = torch.load(ara_path, map_location="cpu")
cokv = torch.load(cokv_path, map_location="cpu")
am = ara.setdefault("metadata", {})
cm = cokv.setdefault("metadata", {})
if am.get("rows_manifest_sha256") != cm.get("rows_manifest_sha256"):
    raise SystemExit(
        f"ordered attribution rows differ for {task}: "
        f"ARA={am.get('rows_manifest_sha256')} CoKV={cm.get('rows_manifest_sha256')}"
    )
common = {
    "formal_protocol": "matched_attribution_v1",
    "formal_task": task,
    "attribution_data_matched": True,
    "timing_recorded_at": datetime.now().isoformat(),
}
am.update(common)
am["attribution_wall_seconds"] = float(ara_s)
cm.update(common)
cm.update({
    "attribution_wall_seconds": float(cokv_s),
    "ara_reference_wall_seconds": float(ara_s),
    "time_match_ratio": float(cokv_s) / max(float(ara_s), 1e-12),
    "time_matched_sampling_number": int(cokv_m),
    "calibration_pilot_sampling_number": int(pilot_m),
    "predicted_wall_seconds_from_pilot": float(predicted_s),
})
torch.save(ara, ara_path)
torch.save(cokv, cokv_path)
print(
    f"matched task={task} manifest={am['rows_manifest_sha256']} "
    f"ARA={float(ara_s):.1f}s CoKV={float(cokv_s):.1f}s ratio={cm['time_match_ratio']:.3f} M={int(cokv_m)}"
)
PY
}

if [ -n "${WAIT_FOR_PID}" ] && kill -0 "${WAIT_FOR_PID}" 2>/dev/null; then
  record_status "queue_${MODEL_FAMILY}" "WAITING" "pid=${WAIT_FOR_PID} gpu=${GPU_ID}"
  while kill -0 "${WAIT_FOR_PID}" 2>/dev/null; do
    sleep 60
  done
fi

record_status "pipeline_${MODEL_FAMILY}" "STARTED" "gpu=${GPU_ID} max_samples=${MAX_SAMPLES} seed=${DATA_SEED}"
IFS=',' read -r -a DATASETS <<< "${DATASETS_STR}"
FAILED=()

for raw_task in "${DATASETS[@]}"; do
  task="$(echo "${raw_task}" | xargs)"
  [ -z "${task}" ] && continue
  safe_task="$(safe_name "${task}")"

  ara_token="${RUN_TAG}_ara_${safe_task}"
  if [ "${RESUME}" = "1" ]; then
    existing_ara_path="$(find_ara_path "${task}" "${ara_token}")"
    existing_cokv_path="$(find_existing_cokv_path "${task}" "${RUN_TAG}_cokv_${safe_task}")"
    if [ -n "${existing_ara_path}" ] && [ -n "${existing_cokv_path}" ]; then
      existing_cokv_samples="$(cokv_sampling_number "${existing_cokv_path}" 2>/dev/null)"
      if [ -n "${existing_cokv_samples}" ] \
        && validate_importance "${existing_ara_path}" 0 ara \
        && validate_importance "${existing_cokv_path}" "${existing_cokv_samples}" cokv \
        && validate_matching_pair "${existing_ara_path}" "${existing_cokv_path}"; then
        record_status "resume_${task}" "SKIPPED" \
          "verified ARA+CoKV pair M=${existing_cokv_samples} ara=${existing_ara_path} cokv=${existing_cokv_path}"
        continue
      fi
    fi
  fi

  ara_log="${STATE_DIR}/ara_${safe_task}.log"
  record_status "ara_${task}" "RUNNING" "log=${ara_log}"
  ara_started=$(date +%s)
  env \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" GPU_ID="${GPU_ID}" \
    ATTR_DATASETS_STR="${task}" MAX_SAMPLES="${MAX_SAMPLES}" RUN_TS="${ara_token}" \
    SEED="${DATA_SEED}" DATA_SEED="${DATA_SEED}" MASK_SEED="${DATA_SEED}" \
    IG_STEPS=8 PATH_MODE=random_threshold PATH_SAMPLES=4 PATH_SEED="${DATA_SEED}" \
    BASELINE=zero MASK_PROBS="0.15,0.3,0.5,0.7,0.9" MASK_SAMPLES_PER_PROB=2 \
    LOSS_NORMALIZE=mean_masked IG_POSTPROCESS=signed GSM8K_ANSWER_MODE=final_hash \
    DEBUG_DUMP_SAMPLES=0 DEBUG_SAVE_PER_SAMPLE=0 \
    bash "${ARA_RUNNER}" > "${ara_log}" 2>&1
  ara_rc=$?
  ara_seconds=$(( $(date +%s) - ara_started ))
  ara_path="$(find_ara_path "${task}" "${ara_token}")"
  if [ "${ara_rc}" -ne 0 ] || [ -z "${ara_path}" ] || ! validate_importance "${ara_path}" 0 ara >> "${ara_log}" 2>&1; then
    record_status "ara_${task}" "FAILED" "rc=${ara_rc} seconds=${ara_seconds}"
    FAILED+=("ara:${task}")
    break
  fi
  record_status "ara_${task}" "DONE" "seconds=${ara_seconds} path=${ara_path}"

  pilot_token="${RUN_TAG}_cokv_pilot_${safe_task}"
  pilot_log="${STATE_DIR}/cokv_pilot_${safe_task}.log"
  record_status "cokv_pilot_${task}" "RUNNING" "M=${COKV_PILOT_SAMPLES} log=${pilot_log}"
  pilot_started=$(date +%s)
  env \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" GPU_ID="${GPU_ID}" \
    ATTR_DATASETS_STR="${task}" MAX_SAMPLES="${MAX_SAMPLES}" RUN_TS="${pilot_token}" \
    SEED="${DATA_SEED}" DATA_SEED="${DATA_SEED}" MASK_SEED="${DATA_SEED}" \
    SAMPLING_NUMBER="${COKV_PILOT_SAMPLES}" COALITION_SIZES="0.25,0.5,0.75" \
    MASK_PROBS="0.15,0.3,0.5,0.7,0.9" MASK_SAMPLES_PER_PROB=2 \
    LOSS_NORMALIZE=mean_masked SCORE_POSTPROCESS=signed GSM8K_ANSWER_MODE=final_hash \
    DEBUG_DUMP_SAMPLES=0 \
    bash "${COKV_RUNNER}" > "${pilot_log}" 2>&1
  pilot_rc=$?
  pilot_seconds=$(( $(date +%s) - pilot_started ))
  pilot_path="$(find_cokv_path "${task}" "${pilot_token}" "${COKV_PILOT_SAMPLES}")"
  if [ "${pilot_rc}" -ne 0 ] || [ -z "${pilot_path}" ] || ! validate_importance "${pilot_path}" "${COKV_PILOT_SAMPLES}" cokv >> "${pilot_log}" 2>&1; then
    record_status "cokv_pilot_${task}" "FAILED" "rc=${pilot_rc} seconds=${pilot_seconds}"
    FAILED+=("cokv_pilot:${task}")
    break
  fi

  calibration="$(calibrate_cokv_samples "${pilot_path}" "${ara_seconds}" "${pilot_seconds}")"
  IFS=$'\t' read -r cokv_samples predicted_seconds fixed_seconds seconds_per_sample <<< "${calibration}"
  record_status "cokv_pilot_${task}" "DONE" \
    "seconds=${pilot_seconds} fixed=${fixed_seconds} per_sample=${seconds_per_sample} target_M=${cokv_samples} predicted=${predicted_seconds}"

  cokv_token="${RUN_TAG}_cokv_${safe_task}"
  cokv_log="${STATE_DIR}/cokv_${safe_task}.log"
  record_status "cokv_${task}" "RUNNING" "M=${cokv_samples} log=${cokv_log}"
  cokv_started=$(date +%s)
  env \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" GPU_ID="${GPU_ID}" \
    ATTR_DATASETS_STR="${task}" MAX_SAMPLES="${MAX_SAMPLES}" RUN_TS="${cokv_token}" \
    SEED="${DATA_SEED}" DATA_SEED="${DATA_SEED}" MASK_SEED="${DATA_SEED}" \
    SAMPLING_NUMBER="${cokv_samples}" COALITION_SIZES="0.25,0.5,0.75" \
    MASK_PROBS="0.15,0.3,0.5,0.7,0.9" MASK_SAMPLES_PER_PROB=2 \
    LOSS_NORMALIZE=mean_masked SCORE_POSTPROCESS=signed GSM8K_ANSWER_MODE=final_hash \
    DEBUG_DUMP_SAMPLES=0 \
    bash "${COKV_RUNNER}" > "${cokv_log}" 2>&1
  cokv_rc=$?
  cokv_seconds=$(( $(date +%s) - cokv_started ))
  cokv_path="$(find_cokv_path "${task}" "${cokv_token}" "${cokv_samples}")"
  if [ "${cokv_rc}" -ne 0 ] || [ -z "${cokv_path}" ] || ! validate_importance "${cokv_path}" "${cokv_samples}" cokv >> "${cokv_log}" 2>&1; then
    record_status "cokv_${task}" "FAILED" "rc=${cokv_rc} seconds=${cokv_seconds}"
    FAILED+=("cokv:${task}")
    break
  fi
  if ! annotate_and_compare "${task}" "${ara_path}" "${cokv_path}" "${ara_seconds}" "${cokv_seconds}" "${cokv_samples}" "${predicted_seconds}" >> "${cokv_log}" 2>&1; then
    record_status "match_${task}" "FAILED" "data manifest mismatch"
    FAILED+=("match:${task}")
    break
  fi
  time_ratio="$(python -c "print(float(${cokv_seconds}) / max(float(${ara_seconds}), 1.0))")"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${task}" "${ara_seconds}" "${COKV_PILOT_SAMPLES}" "${pilot_seconds}" "${cokv_samples}" \
    "${predicted_seconds}" "${cokv_seconds}" "${time_ratio}" "${ara_path}" "${cokv_path}" >> "${TIMING_FILE}"
  record_status "cokv_${task}" "DONE" "seconds=${cokv_seconds} ratio=${time_ratio} M=${cokv_samples} path=${cokv_path}"
done

if [ "${#FAILED[@]}" -gt 0 ]; then
  record_status "pipeline_${MODEL_FAMILY}" "FAILED" "${FAILED[*]}"
  exit 1
fi

record_status "adaptive_${MODEL_FAMILY}" "RUNNING" "methods=headig,cokv"
env \
  GPU_ID="${GPU_ID}" RUN_TAG="${RUN_TAG}_adaptive" ATTR_MAIN_ONLY=1 \
  MODEL_FAMILIES_STR="${MODEL_FAMILY}" METHODS_STR="headig,cokv" DATASETS_STR="${DATASETS_STR}" \
  GPQA_MC_NUM="${GPQA_MC_NUM}" \
  bash "${PROJECT_ROOT}/experiment_scripts/run_adaptive_table_fill.sh" > "${STATE_DIR}/adaptive.log" 2>&1
adaptive_rc=$?
record_status "adaptive_${MODEL_FAMILY}" "$([ "${adaptive_rc}" -eq 0 ] && echo DONE || echo FAILED)" "rc=${adaptive_rc}"

record_status "pruning_${MODEL_FAMILY}" "RUNNING" "methods=headig,cokv prune=most"
env \
  GPU_ID="${GPU_ID}" MODEL_FAMILY="${MODEL_FAMILY}" RUN_TAG="${RUN_TAG}_pruning" \
  METHODS_STR="headig,cokv" DATASETS_STR="${DATASETS_STR}" GSM8K_SOURCE_LABELS_STR="gsm8k_final_hash" \
  PRUNE_MODES=most GPQA_MC_NUM="${GPQA_MC_NUM}" \
  bash "${PROJECT_ROOT}/experiment_scripts/run_mask_main_fill.sh" > "${STATE_DIR}/pruning.log" 2>&1
pruning_rc=$?
record_status "pruning_${MODEL_FAMILY}" "$([ "${pruning_rc}" -eq 0 ] && echo DONE || echo FAILED)" "rc=${pruning_rc}"

if [ "${adaptive_rc}" -ne 0 ] || [ "${pruning_rc}" -ne 0 ]; then
  record_status "pipeline_${MODEL_FAMILY}" "FAILED" "adaptive_rc=${adaptive_rc} pruning_rc=${pruning_rc}"
  exit 1
fi

record_status "pipeline_${MODEL_FAMILY}" "DONE" "timing=${TIMING_FILE}"
