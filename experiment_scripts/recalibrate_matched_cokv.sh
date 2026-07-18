#!/usr/bin/env bash
# Re-run selected CoKV attributions with explicitly calibrated sample counts.
# It fixes the data/masking/scoring protocol and changes only SAMPLING_NUMBER.

set -uo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/qiheng/Projects/adaptive-dllm}
MODEL_FAMILY=${MODEL_FAMILY:?Set MODEL_FAMILY to dream or llada}
GPU_ID=${GPU_ID:?Set GPU_ID}
RUN_TAG=${RUN_TAG:?Set RUN_TAG to the original matched-run tag}
TASK_SPECS=${TASK_SPECS:?Set TASK_SPECS, e.g. ceval-valid:78,gsm8k:97}
WAIT_FOR_GPU_FREE=${WAIT_FOR_GPU_FREE:-0}
WAIT_FOR_FREE_MEMORY_MIB=${WAIT_FOR_FREE_MEMORY_MIB:-0}
REUSE_EXISTING=${REUSE_EXISTING:-1}
RECALIBRATION_ROUND=${RECALIBRATION_ROUND:-1}
TIMING_CONTEXT=${TIMING_CONTEXT:-exclusive_gpu}
MAX_SAMPLES=${MAX_SAMPLES:-200}
DATA_SEED=${DATA_SEED:-}
MASK_SEED=${MASK_SEED:-}

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export PATH="$HOME/miniconda3/envs/adaptive-dllm/bin:$PATH"

case "$MODEL_FAMILY" in
  dream)
    MODEL_NAME=dream
    DEFAULT_DATA_SEED=47
    COKV_RUNNER="$PROJECT_ROOT/models/Dream/attribution/baseline_attribution/run_shapley_head_attribution.sh"
    ;;
  llada)
    MODEL_NAME=llada-1_5
    DEFAULT_DATA_SEED=123
    COKV_RUNNER="$PROJECT_ROOT/models/LLaDA/attribution/baseline_attribution/run_shapley_head_attribution.sh"
    ;;
  *)
    echo "Unsupported MODEL_FAMILY: $MODEL_FAMILY" >&2
    exit 2
    ;;
esac
DATA_SEED=${DATA_SEED:-$DEFAULT_DATA_SEED}
MASK_SEED=${MASK_SEED:-$DATA_SEED}

STATE_DIR="$PROJECT_ROOT/logs/matched_ara_cokv/$RUN_TAG"
STATUS_FILE="$STATE_DIR/recalibration_${MODEL_FAMILY}_v${RECALIBRATION_ROUND}.status"
TIMING_FILE="$STATE_DIR/recalibration_${MODEL_FAMILY}.tsv"
mkdir -p "$STATE_DIR"
if [[ ! -f "$TIMING_FILE" ]]; then
  printf 'task\tmodel\tround\tnew_M\told_M\tara_seconds\told_cokv_seconds\tnew_cokv_seconds\tnew_ratio\tartifact\n' > "$TIMING_FILE"
fi

source_label_for_task() {
  case "$1" in
    mmlu) echo mmlu_all ;;
    cmmlu) echo cmmlu_all ;;
    ceval-valid) echo ceval-valid_all ;;
    gpqa) echo gpqa_main_n_shot_all ;;
    gpqa_main_n_shot) echo gpqa_main_n_shot_all ;;
    gsm8k) echo gsm8k_final_hash ;;
    humaneval) echo humaneval ;;
    mbpp) echo mbpp ;;
    *) echo "Unknown task $1" >&2; return 1 ;;
  esac
}

safe_task_name() { printf '%s' "$1" | tr '/.' '__' | tr '-' '_'; }

find_ara_path() {
  local task=$1 safe label
  safe=$(safe_task_name "$task")
  label=$(source_label_for_task "$task") || return 1
  find "$PROJECT_ROOT/configs/aconfigs" -mindepth 2 -maxdepth 2 -type f \
    -path "*/head_importance_${MODEL_NAME}_${label}_pmrandom_threshold_ts${RUN_TAG}_ara_${safe}/head_importance.pt" \
    -print -quit
}

find_cokv_path() {
  local task=$1 m=$2 token=$3 label
  label=$(source_label_for_task "$task") || return 1
  find "$PROJECT_ROOT/configs/aconfigs" -mindepth 2 -maxdepth 2 -type f \
    -path "*/head_importance_${MODEL_NAME}_${label}_cokv_signed_M${m}_*_ts${token}/head_importance.pt" \
    -print -quit
}

if [[ "$WAIT_FOR_GPU_FREE" = "1" ]]; then
  printf 'waiting_for_gpu %s\n' "$GPU_ID" > "$STATUS_FILE"
  while nvidia-smi --id="$GPU_ID" --query-compute-apps=pid --format=csv,noheader \
      | grep -Eq '[0-9]'; do
    sleep 30
  done
fi

if [[ "$WAIT_FOR_FREE_MEMORY_MIB" =~ ^[1-9][0-9]*$ ]]; then
  printf 'waiting_for_gpu_memory gpu=%s required_free_mib=%s\n' \
    "$GPU_ID" "$WAIT_FOR_FREE_MEMORY_MIB" > "$STATUS_FILE"
  while true; do
    free_mib=$(nvidia-smi --id="$GPU_ID" \
      --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= WAIT_FOR_FREE_MEMORY_MIB )); then
      break
    fi
    sleep 30
  done
fi

IFS=',' read -r -a SPECS <<< "$TASK_SPECS"
for spec in "${SPECS[@]}"; do
  task=${spec%%:*}
  m=${spec##*:}
  if [[ -z "$task" || "$task" == "$m" || ! "$m" =~ ^[1-9][0-9]*$ ]]; then
    echo "Malformed task specification: $spec" >&2; exit 2
  fi

  old_row=$(awk -F '\t' -v t="$task" '$1 == t { row=$0 } END { print row }' "$STATE_DIR/timing.tsv")
  if [[ -z "$old_row" ]]; then echo "No original timing for $task" >&2; exit 3; fi
  IFS=$'\t' read -r _ ara_seconds _ _ old_m _ old_cokv_seconds _ <<< "$old_row"
  if [[ -z "$ara_seconds" || -z "$old_m" || -z "$old_cokv_seconds" ]]; then
    echo "Incomplete original timing entry: $old_row" >&2; exit 3
  fi

  ara_path=$(find_ara_path "$task")
  if [[ -z "$ara_path" || ! -f "$ara_path" ]]; then
    echo "Missing original ARA artifact for $task" >&2; exit 4
  fi
  safe=$(safe_task_name "$task")
  token="${RUN_TAG}_cokv_recal_v${RECALIBRATION_ROUND}_${safe}"
  log="$STATE_DIR/cokv_recal_v${RECALIBRATION_ROUND}_${MODEL_FAMILY}_${safe}.log"
  printf '%s START task=%s M=%s old_M=%s\n' "$(date '+%F %T')" "$task" "$m" "$old_m" > "$STATUS_FILE"

  cokv_path=$(find_cokv_path "$task" "$m" "$token")
  if [[ "$REUSE_EXISTING" = "1" && -n "$cokv_path" && -f "$cokv_path" ]]; then
    cokv_seconds=$(python - "$cokv_path" <<'PY'
import sys
import torch

meta = torch.load(sys.argv[1], map_location="cpu").get("metadata", {})
seconds = float(meta.get("preparation_seconds", 0.0)) + float(meta.get("sampling_seconds", 0.0))
if seconds <= 0:
    raise SystemExit("existing artifact has no usable timing metadata")
print(int(round(seconds)))
PY
)
    printf '%s REUSE task=%s M=%s seconds=%s\n' \
      "$(date '+%F %T')" "$task" "$m" "$cokv_seconds" > "$STATUS_FILE"
  else
    start_epoch=$(date +%s)
    env CUDA_VISIBLE_DEVICES="$GPU_ID" GPU_ID="$GPU_ID" \
      ATTR_DATASETS_STR="$task" MAX_SAMPLES="$MAX_SAMPLES" RUN_TS="$token" \
      SEED="$DATA_SEED" DATA_SEED="$DATA_SEED" MASK_SEED="$MASK_SEED" \
      SAMPLING_NUMBER="$m" COALITION_SIZES="0.25,0.5,0.75" \
      MASK_PROBS="0.15,0.3,0.5,0.7,0.9" MASK_SAMPLES_PER_PROB=2 \
      LOSS_NORMALIZE=mean_masked SCORE_POSTPROCESS=signed GSM8K_ANSWER_MODE=final_hash \
      DEBUG_DUMP_SAMPLES=0 bash "$COKV_RUNNER" > "$log" 2>&1
    rc=$?
    end_epoch=$(date +%s)
    cokv_seconds=$((end_epoch - start_epoch))
    if [[ "$rc" -ne 0 ]]; then
      printf '%s FAILED task=%s M=%s rc=%s\n' "$(date '+%F %T')" "$task" "$m" "$rc" > "$STATUS_FILE"
      echo "CoKV re-run failed for $task; see $log" >&2; exit "$rc"
    fi
    cokv_path=$(find_cokv_path "$task" "$m" "$token")
  fi

  if [[ -z "$cokv_path" || ! -f "$cokv_path" ]]; then
    printf '%s FAILED task=%s missing_artifact\n' "$(date '+%F %T')" "$task" > "$STATUS_FILE"
    echo "CoKV artifact was not found for $task" >&2; exit 5
  fi
  ratio=$(python - "$ara_seconds" "$cokv_seconds" <<'PY'
import sys
ara, cokv = map(float, sys.argv[1:])
print(f"{cokv / ara:.6f}")
PY
)
  python - "$ara_path" "$cokv_path" "$task" "$MAX_SAMPLES" "$DATA_SEED" "$MASK_SEED" \
    "$m" "$old_m" "$ara_seconds" "$old_cokv_seconds" "$cokv_seconds" "$ratio" "$RECALIBRATION_ROUND" "$TIMING_CONTEXT" <<'PY'
import sys
import torch

(ara_path, cokv_path, task, max_samples, data_seed, mask_seed, new_m, old_m,
 ara_seconds, old_cokv_seconds, new_cokv_seconds, ratio, round_id, timing_context) = sys.argv[1:]
max_samples, data_seed, mask_seed, new_m, old_m, round_id = map(
    int, (max_samples, data_seed, mask_seed, new_m, old_m, round_id))
ara_seconds, old_cokv_seconds, new_cokv_seconds, ratio = map(
    float, (ara_seconds, old_cokv_seconds, new_cokv_seconds, ratio))
ara = torch.load(ara_path, map_location="cpu")
cokv = torch.load(cokv_path, map_location="cpu")
for name, payload in (("ARA", ara), ("CoKV", cokv)):
    score = payload.get("importance_scores", payload.get("head_importance"))
    if isinstance(score, dict):
        values = [torch.as_tensor(value).reshape(-1).float()
                  for _, value in sorted(score.items(), key=lambda item: int(item[0]))]
        score = torch.cat(values) if values else torch.empty(0)
    else:
        score = torch.as_tensor(score).reshape(-1).float() if score is not None else torch.empty(0)
    if score.numel() == 0 or not bool(torch.isfinite(score).all()):
        raise RuntimeError(f"{name} artifact has missing or non-finite scores")
am, cm = ara.get("metadata", {}), cokv.get("metadata", {})
if (not am.get("rows_manifest_sha256") or
        am.get("rows_manifest_sha256") != cm.get("rows_manifest_sha256")):
    raise RuntimeError("ARA and recalibrated CoKV manifests do not match")
for key, expected in (("max_samples", max_samples), ("data_seed", data_seed),
                      ("mask_seed", mask_seed), ("sampling_number", new_m)):
    if cm.get(key) != expected:
        raise RuntimeError(f"CoKV metadata mismatch for {key}: {cm.get(key)!r} != {expected!r}")
cm.update({
    "formal_protocol": "matched_attribution_v1",
    "formal_task": task,
    "attribution_data_matched": True,
    "time_matched_sampling_number": new_m,
    "cokv_recalibration_round": round_id,
    "cokv_recalibration_method": "two_point_wall_time",
    "cokv_recalibrated_from_sampling_number": old_m,
    "ara_reference_wall_seconds": ara_seconds,
    "cokv_previous_wall_seconds": old_cokv_seconds,
    "attribution_wall_seconds": new_cokv_seconds,
    "time_match_ratio": ratio,
    "attribution_timing_context": timing_context,
})
cokv["metadata"] = cm
torch.save(cokv, cokv_path)
PY
  rc=$?
  if [[ "$rc" -ne 0 ]]; then
    printf '%s FAILED task=%s validation_rc=%s\n' "$(date '+%F %T')" "$task" "$rc" > "$STATUS_FILE"
    exit "$rc"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$task" "$MODEL_FAMILY" "$RECALIBRATION_ROUND" "$m" "$old_m" "$ara_seconds" \
    "$old_cokv_seconds" "$cokv_seconds" "$ratio" "$cokv_path" >> "$TIMING_FILE"
  printf '%s DONE task=%s M=%s seconds=%s ratio=%s\n' \
    "$(date '+%F %T')" "$task" "$m" "$cokv_seconds" "$ratio" > "$STATUS_FILE"
done

printf '%s COMPLETE\n' "$(date '+%F %T')" > "$STATUS_FILE"
