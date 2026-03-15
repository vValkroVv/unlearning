#!/usr/bin/env bash

set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")

if [[ $# -lt 5 ]]; then
  echo "usage: $0 RUN_DIR FORGET_SPLIT RETAIN_SPLIT BASE_MODEL_PATH TOKENIZER_PATH [LORA_MODEL_CONFIG] [BASE_MODEL_CONFIG]" >&2
  exit 1
fi

RUN_DIR=$1
FORGET_SPLIT=$2
RETAIN_SPLIT=$3
BASE_MODEL_PATH=$4
TOKENIZER_PATH=$5
ORIGINAL_BASE_MODEL_PATH=${BASE_MODEL_PATH}
MODEL_CONFIG_RAW=${6:-${LORA_MODEL_EVAL_CONFIG:-Llama-3.1-8B-Instruct-lora}}
BASE_MODEL_CONFIG_RAW=${7:-${BASE_MODEL_EVAL_CONFIG:-}}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-128}
DELETE_RUN_BASE_MODEL_AFTER_EVAL=${DELETE_RUN_BASE_MODEL_AFTER_EVAL:-0}
DELETE_CHECKPOINT_ADAPTER_SAFETENSORS_AFTER_EVAL=${DELETE_CHECKPOINT_ADAPTER_SAFETENSORS_AFTER_EVAL:-1}

normalize_model_config_name() {
  local raw="${1:-}"
  raw="${raw##*/}"
  raw="${raw%.yaml}"
  echo "${raw}"
}

has_loadable_weights() {
  local model_dir="$1"
  [[ -f "${model_dir}/adapter_model.safetensors" ]] \
    || [[ -f "${model_dir}/adapter_model.bin" ]] \
    || [[ -f "${model_dir}/model.safetensors" ]] \
    || [[ -f "${model_dir}/model.safetensors.index.json" ]] \
    || [[ -f "${model_dir}/pytorch_model.bin" ]] \
    || [[ -f "${model_dir}/pytorch_model.bin.index.json" ]]
}

has_loadable_base_model() {
  local model_dir="$1"
  [[ -f "${model_dir}/config.json" ]] && has_loadable_weights "${model_dir}"
}

delete_checkpoint_adapter_weights() {
  local run_dir="$1"
  local deleted=0
  local ckpt=""
  local weight_file=""
  while IFS= read -r -d '' ckpt; do
    for weight_file in "${ckpt}/adapter_model.safetensors" "${ckpt}/adapter_model.bin"; do
      if [[ -f "${weight_file}" ]]; then
        rm -f "${weight_file}"
        deleted=$((deleted + 1))
      fi
    done
  done < <(find "${run_dir}" -maxdepth 1 -type d -name 'checkpoint-*' -print0)
  echo "[rwku][ckpt-eval] Removed ${deleted} checkpoint adapter weight files"
}

MODEL_CONFIG=$(normalize_model_config_name "${MODEL_CONFIG_RAW}")
if [[ -n "${BASE_MODEL_CONFIG_RAW}" ]]; then
  BASE_MODEL_CONFIG=$(normalize_model_config_name "${BASE_MODEL_CONFIG_RAW}")
elif [[ "${MODEL_CONFIG}" == *-lora ]]; then
  BASE_MODEL_CONFIG="${MODEL_CONFIG%-lora}"
else
  BASE_MODEL_CONFIG="${MODEL_CONFIG}"
fi

RESOLVED_BASE_MODEL_PATH=${FILA_BASE_PATH:-${BASE_MODEL_PATH}}
LORA_BASE_MODEL_SUBFOLDER_VALUE=""
if has_loadable_base_model "${RUN_DIR}/base_model"; then
  RESOLVED_BASE_MODEL_PATH="${RUN_DIR}/base_model"
  echo "[rwku][ckpt-eval] Detected LoKU FILA base model at ${RESOLVED_BASE_MODEL_PATH}"
elif [[ -d "${RUN_DIR}/base_model" ]]; then
  echo "[rwku][ckpt-eval] Found ${RUN_DIR}/base_model but it is missing config/weights; falling back to ${RESOLVED_BASE_MODEL_PATH}"
fi

mapfile -t CKPTS < <(find "${RUN_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
if has_loadable_weights "${RUN_DIR}"; then
  CKPTS+=("${RUN_DIR}")
elif [[ -f "${RUN_DIR}/evals/DUET_SUMMARY.json" ]]; then
  echo "[rwku][ckpt-eval] Skipping top-level run_dir re-eval because weights were cleaned after endpoint eval: ${RUN_DIR}"
else
  echo "[rwku][ckpt-eval] Skipping top-level run_dir because no loadable weights were found: ${RUN_DIR}"
fi

for ckpt in "${CKPTS[@]}"; do
  name=$(basename "${ckpt}")
  out_dir="${RUN_DIR}/checkpoint_evals/${name}"
  mkdir -p "${out_dir}"
  python src/eval.py \
    experiment=eval/rwku/default.yaml \
    model=${MODEL_CONFIG} \
    forget_split=${FORGET_SPLIT} \
    holdout_split=${RETAIN_SPLIT} \
    task_name=$(basename "${RUN_DIR}")_${name} \
    model.model_args.pretrained_model_name_or_path=${ckpt} \
    ++model.model_args.base_model_name_or_path=${RESOLVED_BASE_MODEL_PATH} \
    model.tokenizer_args.pretrained_model_name_or_path=${TOKENIZER_PATH} \
    model.model_args.device_map=auto \
    ++model.model_args.low_cpu_mem_usage=true \
    eval.duet.batch_size=${EVAL_BATCH_SIZE} \
    eval.duet.overwrite=true \
    paths.output_dir=${out_dir} \
    retain_logs_path=null
done

python src/tools/summarize_checkpoint_metrics.py \
  --run-dir "${RUN_DIR}" \
  --output-path "${RUN_DIR}/checkpoint_evals/summary.tsv"

if [[ "${RUN_UTILITY_EVAL:-0}" == "1" ]]; then
  LORA_BASE_MODEL_SUBFOLDER="${LORA_BASE_MODEL_SUBFOLDER_VALUE}" \
  LORA_BASE_MODEL_PATH="${RESOLVED_BASE_MODEL_PATH}" "${script_dir}/../utility/eval_checkpoints_utility.sh" \
    "${RUN_DIR}" \
    "${BASE_MODEL_CONFIG}" \
    "${MODEL_CONFIG}" \
    "${ORIGINAL_BASE_MODEL_PATH}" \
    "${TOKENIZER_PATH}"
fi

if [[ -f "${RUN_DIR}/checkpoint_evals/summary.tsv" && -f "${RUN_DIR}/checkpoint_evals_utility/summary.tsv" ]]; then
  merge_cmd=(
    python src/tools/merge_checkpoint_utility_summaries.py
    --checkpoint-summary "${RUN_DIR}/checkpoint_evals/summary.tsv"
    --utility-summary "${RUN_DIR}/checkpoint_evals_utility/summary.tsv"
    --output-path "${RUN_DIR}/checkpoint_evals_merged/summary.tsv"
    --trajectory-path "${RUN_DIR}/checkpoint_evals_merged/trajectory_metrics.json"
  )
  if [[ -n "${UTILITY_FORGET_TAU:-}" ]]; then
    merge_cmd+=(--forget-tau "${UTILITY_FORGET_TAU}")
  fi
  "${merge_cmd[@]}"
fi

if [[ "${DELETE_CHECKPOINT_ADAPTER_SAFETENSORS_AFTER_EVAL}" == "1" ]]; then
  delete_checkpoint_adapter_weights "${RUN_DIR}"
fi

if [[ "${DELETE_RUN_BASE_MODEL_AFTER_EVAL}" == "1" && "${RESOLVED_BASE_MODEL_PATH}" == "${RUN_DIR}/base_model" ]]; then
  rm -rf "${RESOLVED_BASE_MODEL_PATH}"
  echo "[rwku][ckpt-eval] Removed FILA base model ${RESOLVED_BASE_MODEL_PATH}"
fi
