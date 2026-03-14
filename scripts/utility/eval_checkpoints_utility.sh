#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 RUN_DIR BASE_MODEL_CFG LORA_MODEL_CFG BASE_MODEL_PATH TOKENIZER_PATH" >&2
  exit 1
fi

script_dir=$(dirname "$(realpath "$0")")
repo_root=$(realpath "${script_dir}/../..")

RUN_DIR=$(realpath "$1")
BASE_MODEL_CFG_RAW=$2
LORA_MODEL_CFG_RAW=$3
BASE_MODEL_PATH=$4
TOKENIZER_PATH=$5
LORA_BASE_MODEL_PATH=${LORA_BASE_MODEL_PATH:-${BASE_MODEL_PATH}}

UTILITY_ROOT=${UTILITY_ROOT:-${repo_root}/artifacts/evals/utility_1k_v1}
UTILITY_EVAL_BATCH_SIZE=${UTILITY_EVAL_BATCH_SIZE:-16}
UTILITY_NUM_FEWSHOT=${UTILITY_NUM_FEWSHOT:-0}
UTILITY_APPLY_CHAT_TEMPLATE=${UTILITY_APPLY_CHAT_TEMPLATE:-true}
UTILITY_SYSTEM_INSTRUCTION=${UTILITY_SYSTEM_INSTRUCTION:-null}
EVAL_RUN_BASE_MODEL=${EVAL_RUN_BASE_MODEL:-0}
BASELINE_CACHE_ROOT=${BASELINE_CACHE_ROOT:-}

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

render_task_configs() {
  local template_dir="$1"
  local output_dir="$2"
  local utility_root="$3"
  local escaped_root=""
  escaped_root=$(printf '%s' "${utility_root}" | sed 's/[&|]/\\&/g')

  rm -rf "${output_dir}"
  mkdir -p "${output_dir}"
  cp "${template_dir}/utils.py" "${output_dir}/utils.py"
  for yaml_path in "${template_dir}"/*.yaml; do
    sed "s|artifacts/evals/utility_1k_v1|${escaped_root}|g" "${yaml_path}" > "${output_dir}/$(basename "${yaml_path}")"
  done
}

copy_tree_contents() {
  local source_dir="$1"
  local target_dir="$2"
  mkdir -p "${target_dir}"
  cp -a "${source_dir}/." "${target_dir}/"
}

cache_key() {
  printf '%s\n' \
    "$(normalize_model_config_name "${BASE_MODEL_CFG_RAW}")" \
    "${BASE_MODEL_PATH}" \
    "${TOKENIZER_PATH}" \
    "${UTILITY_ROOT}" \
    "${UTILITY_APPLY_CHAT_TEMPLATE}" \
    "${UTILITY_NUM_FEWSHOT}" \
    | sha1sum | awk '{print $1}'
}

evaluate_label() {
  local label="$1"
  local model_cfg="$2"
  local model_path="$3"
  local out_dir="$4"
  local base_model_override="${5:-}"
  local use_baseline_cache="${6:-0}"

  local cache_dir=""
  if [[ "${use_baseline_cache}" == "1" && -n "${BASELINE_CACHE_ROOT}" ]]; then
    cache_dir="${BASELINE_CACHE_ROOT}/$(cache_key)"
    if [[ -f "${cache_dir}/LMEval_SUMMARY.json" ]]; then
      echo "[utility][ckpt-eval] Reusing cached base-model utility results from ${cache_dir}"
      rm -rf "${out_dir}"
      copy_tree_contents "${cache_dir}" "${out_dir}"
      return
    fi
  fi

  rm -rf "${out_dir}"
  mkdir -p "${out_dir}"

  local task_name
  task_name="$(basename "${RUN_DIR}")_${label}_utility1k"

  local cmd=(
    python
    src/eval.py
    experiment=eval/utility_1k/default.yaml
    model="${model_cfg}"
    task_name="${task_name}"
    model.model_args.pretrained_model_name_or_path="${model_path}"
    model.tokenizer_args.pretrained_model_name_or_path="${TOKENIZER_PATH}"
    eval.lm_eval.include_path="${rendered_task_dir}"
    eval.lm_eval.simple_evaluate_args.batch_size="${UTILITY_EVAL_BATCH_SIZE}"
    eval.lm_eval.simple_evaluate_args.num_fewshot="${UTILITY_NUM_FEWSHOT}"
    eval.lm_eval.simple_evaluate_args.apply_chat_template="${UTILITY_APPLY_CHAT_TEMPLATE}"
    eval.lm_eval.simple_evaluate_args.system_instruction="${UTILITY_SYSTEM_INSTRUCTION}"
    paths.output_dir="${out_dir}"
  )
  if [[ -n "${base_model_override}" ]]; then
    cmd+=(++model.model_args.base_model_name_or_path="${base_model_override}")
  fi
  "${cmd[@]}"

  if [[ "${use_baseline_cache}" == "1" && -n "${cache_dir}" ]]; then
    rm -rf "${cache_dir}"
    copy_tree_contents "${out_dir}" "${cache_dir}"
  fi
}

BASE_MODEL_CFG=$(normalize_model_config_name "${BASE_MODEL_CFG_RAW}")
LORA_MODEL_CFG=$(normalize_model_config_name "${LORA_MODEL_CFG_RAW}")
UTILITY_ROOT=$(realpath -m "${UTILITY_ROOT}")

for required_file in \
  "${UTILITY_ROOT}/utility_mmlu_pro_400.jsonl" \
  "${UTILITY_ROOT}/utility_truthfulqa_bin_200.jsonl" \
  "${UTILITY_ROOT}/utility_arc_200.jsonl" \
  "${UTILITY_ROOT}/utility_winogrande_200.jsonl"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "[utility][ckpt-eval] Missing Utility-1K panel file: ${required_file}" >&2
    exit 1
  fi
done

summary_root="${RUN_DIR}/checkpoint_evals_utility"
mkdir -p "${summary_root}"
rendered_task_dir="${summary_root}/_task_defs"
render_task_configs "${repo_root}/configs/lm_eval_tasks/utility_1k" "${rendered_task_dir}" "${UTILITY_ROOT}"

evaluate_label \
  "base_model_orig" \
  "${BASE_MODEL_CFG}" \
  "${BASE_MODEL_PATH}" \
  "${summary_root}/base_model_orig" \
  "" \
  "1"

if [[ "${EVAL_RUN_BASE_MODEL}" == "1" && -d "${RUN_DIR}/base_model" ]] && has_loadable_weights "${RUN_DIR}/base_model"; then
  evaluate_label \
    "base_model_run" \
    "${BASE_MODEL_CFG}" \
    "${RUN_DIR}/base_model" \
    "${summary_root}/base_model_run"
fi

mapfile -t CKPTS < <(find "${RUN_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
for ckpt in "${CKPTS[@]}"; do
  if ! has_loadable_weights "${ckpt}"; then
    echo "[utility][ckpt-eval] Skipping $(basename "${ckpt}") because no loadable weights were found."
    continue
  fi
  label=$(basename "${ckpt}")
  evaluate_label \
    "${label}" \
    "${LORA_MODEL_CFG}" \
    "${ckpt}" \
    "${summary_root}/${label}" \
    "${LORA_BASE_MODEL_PATH}"
done

if has_loadable_weights "${RUN_DIR}"; then
  evaluate_label \
    "final" \
    "${LORA_MODEL_CFG}" \
    "${RUN_DIR}" \
    "${summary_root}/final" \
    "${LORA_BASE_MODEL_PATH}"
elif [[ ${#CKPTS[@]} -gt 0 ]]; then
  last_label=$(basename "${CKPTS[-1]}")
  echo "[utility][ckpt-eval] Final adapter weights were removed; using ${last_label} as the endpoint proxy."
  rm -rf "${summary_root}/final"
  copy_tree_contents "${summary_root}/${last_label}" "${summary_root}/final"
else
  echo "[utility][ckpt-eval] Skipping final utility eval because no top-level weights were found in ${RUN_DIR}."
fi

python src/tools/summarize_utility_metrics.py \
  --run-dir "${RUN_DIR}" \
  --output-path "${summary_root}/summary.tsv"
