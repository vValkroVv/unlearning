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
BASE_MODEL_SUBFOLDER=${BASE_MODEL_SUBFOLDER:-}
LORA_BASE_MODEL_SUBFOLDER=${LORA_BASE_MODEL_SUBFOLDER:-${BASE_MODEL_SUBFOLDER}}
BASE_TOKENIZER_SUBFOLDER=${BASE_TOKENIZER_SUBFOLDER:-}
LORA_TOKENIZER_SUBFOLDER=${LORA_TOKENIZER_SUBFOLDER:-${BASE_TOKENIZER_SUBFOLDER}}

UTILITY_EVAL_BATCH_SIZE=${UTILITY_EVAL_BATCH_SIZE:-64}
UTILITY_NUM_FEWSHOT=${UTILITY_NUM_FEWSHOT:-0}
UTILITY_APPLY_CHAT_TEMPLATE=${UTILITY_APPLY_CHAT_TEMPLATE:-true}
UTILITY_SYSTEM_INSTRUCTION=${UTILITY_SYSTEM_INSTRUCTION:-null}
EVAL_RUN_BASE_MODEL=${EVAL_RUN_BASE_MODEL:-0}
BASELINE_CACHE_ROOT=${BASELINE_CACHE_ROOT:-}

resolve_utility_mode() {
  if [[ -n "${UTILITY:-}" ]]; then
    printf '%s\n' "${UTILITY}"
    return 0
  fi

  case "${UTILITY_ROOT:-}" in
    *utility_1k*)
      printf '1k\n'
      return 0
      ;;
    *utility_3k*)
      printf '3k\n'
      return 0
      ;;
  esac

  case "${UTILITY_EVAL_EXPERIMENT:-}" in
    *utility_1k*)
      printf '1k\n'
      return 0
      ;;
    *utility_3k*)
      printf '3k\n'
      return 0
      ;;
  esac

  case "${UTILITY_TASK_CONFIG_ROOT:-}" in
    *utility_1k*)
      printf '1k\n'
      return 0
      ;;
    *utility_3k*)
      printf '3k\n'
      return 0
      ;;
  esac

  printf '3k\n'
}

configure_utility_env() {
  local utility_mode="$1"
  local default_utility_root=""
  local default_task_config_root=""
  local default_eval_experiment=""
  local default_task_name_suffix=""

  case "${utility_mode}" in
    1k)
      default_utility_root="${repo_root}/artifacts/evals/utility_1k_v1"
      default_task_config_root="${repo_root}/configs/lm_eval_tasks/utility_1k"
      default_eval_experiment="eval/utility_1k/default.yaml"
      default_task_name_suffix="utility1k"
      ;;
    3k)
      default_utility_root="${repo_root}/artifacts/evals/utility_3k_v1"
      default_task_config_root="${repo_root}/configs/lm_eval_tasks/utility_3k"
      default_eval_experiment="eval/utility_3k/default.yaml"
      default_task_name_suffix="utility3k"
      ;;
    *)
      echo "[utility][ckpt-eval] Unsupported UTILITY=${utility_mode}. Use 1k or 3k." >&2
      exit 1
      ;;
  esac

  UTILITY="${utility_mode}"
  UTILITY_ROOT="${UTILITY_ROOT:-${default_utility_root}}"
  UTILITY_TASK_CONFIG_ROOT="${UTILITY_TASK_CONFIG_ROOT:-${default_task_config_root}}"
  UTILITY_EVAL_EXPERIMENT="${UTILITY_EVAL_EXPERIMENT:-${default_eval_experiment}}"
  UTILITY_TASK_NAME_SUFFIX="${UTILITY_TASK_NAME_SUFFIX:-${default_task_name_suffix}}"
}

resolve_under_repo() {
  local raw_path="$1"
  if [[ "${raw_path}" = /* ]]; then
    realpath -m "${raw_path}"
    return 0
  fi
  realpath -m "${repo_root}/${raw_path}"
}

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

render_task_configs() {
  local template_dir="$1"
  local output_dir="$2"
  local utility_root="$3"

  rm -rf "${output_dir}"
  mkdir -p "${output_dir}"
  cp "${template_dir}/utils.py" "${output_dir}/utils.py"
  for yaml_path in "${template_dir}"/*.yaml; do
    local target_path="${output_dir}/$(basename "${yaml_path}")"
    if ! grep -q 'data_files:' "${yaml_path}"; then
      cp "${yaml_path}" "${target_path}"
      continue
    fi

    local data_file_name=""
    data_file_name=$(awk '/data_files:/ {print $2}' "${yaml_path}" | xargs basename)
    awk -v utility_root="${utility_root}" -v data_file_name="${data_file_name}" '
      $1 == "data_files:" {
        print "  data_files: " utility_root "/" data_file_name
        next
      }
      { print }
    ' "${yaml_path}" > "${target_path}"
  done
}

collect_required_panel_files() {
  local template_dir="$1"
  local yaml_path=""
  while IFS= read -r yaml_path; do
    awk '/data_files:/ {print $2}' "${yaml_path}" | xargs basename
  done < <(find "${template_dir}" -maxdepth 1 -type f -name '*.yaml' ! -name '_*.yaml' | sort)
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
  local model_subfolder="${7:-}"
  local tokenizer_subfolder="${8:-}"

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
  task_name="$(basename "${RUN_DIR}")_${label}_${UTILITY_TASK_NAME_SUFFIX}"

  local cmd=(
    python
    src/eval.py
    experiment="${UTILITY_EVAL_EXPERIMENT}"
    model="${model_cfg}"
    task_name="${task_name}"
    model.model_args.pretrained_model_name_or_path="${model_path}"
    model.tokenizer_args.pretrained_model_name_or_path="${TOKENIZER_PATH}"
    model.model_args.device_map=auto
    ++model.model_args.low_cpu_mem_usage=true
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
  if [[ -n "${model_subfolder}" ]]; then
    cmd+=(++model.model_args.subfolder="${model_subfolder}")
  fi
  if [[ -n "${tokenizer_subfolder}" ]]; then
    cmd+=(++model.tokenizer_args.subfolder="${tokenizer_subfolder}")
  fi
  "${cmd[@]}"

  if [[ "${use_baseline_cache}" == "1" && -n "${cache_dir}" ]]; then
    rm -rf "${cache_dir}"
    copy_tree_contents "${out_dir}" "${cache_dir}"
  fi
}

BASE_MODEL_CFG=$(normalize_model_config_name "${BASE_MODEL_CFG_RAW}")
LORA_MODEL_CFG=$(normalize_model_config_name "${LORA_MODEL_CFG_RAW}")
configure_utility_env "$(resolve_utility_mode)"
UTILITY_ROOT=$(realpath -m "${UTILITY_ROOT}")
UTILITY_TASK_CONFIG_ROOT=$(resolve_under_repo "${UTILITY_TASK_CONFIG_ROOT}")
LORA_MODEL_SUBFOLDER_VALUE=${LORA_BASE_MODEL_SUBFOLDER}

if has_loadable_base_model "${LORA_BASE_MODEL_PATH}"; then
  # LoKU can pass a flattened saved model under run_dir/base_model. That model
  # should be loaded directly without any inherited SFT subfolder override.
  LORA_MODEL_SUBFOLDER_VALUE=""
fi

if [[ ! -d "${UTILITY_TASK_CONFIG_ROOT}" ]]; then
  echo "[utility][ckpt-eval] Missing utility task config root: ${UTILITY_TASK_CONFIG_ROOT}" >&2
  exit 1
fi

mapfile -t REQUIRED_PANEL_FILES < <(collect_required_panel_files "${UTILITY_TASK_CONFIG_ROOT}")
if [[ ${#REQUIRED_PANEL_FILES[@]} -eq 0 ]]; then
  echo "[utility][ckpt-eval] No utility task files were discovered under ${UTILITY_TASK_CONFIG_ROOT}" >&2
  exit 1
fi

for required_file in "${REQUIRED_PANEL_FILES[@]}"; do
  required_file="${UTILITY_ROOT}/${required_file}"
  if [[ ! -f "${required_file}" ]]; then
    echo "[utility][ckpt-eval] Missing utility panel file: ${required_file}" >&2
    exit 1
  fi
done

summary_root="${RUN_DIR}/checkpoint_evals_utility"
mkdir -p "${summary_root}"
rendered_task_dir="${summary_root}/_task_defs"
render_task_configs "${UTILITY_TASK_CONFIG_ROOT}" "${rendered_task_dir}" "${UTILITY_ROOT}"

evaluate_label \
  "base_model_orig" \
  "${BASE_MODEL_CFG}" \
  "${BASE_MODEL_PATH}" \
  "${summary_root}/base_model_orig" \
  "" \
  "1" \
  "${BASE_MODEL_SUBFOLDER}" \
  "${BASE_TOKENIZER_SUBFOLDER}"

if [[ "${EVAL_RUN_BASE_MODEL}" == "1" && -d "${RUN_DIR}/base_model" ]] && has_loadable_weights "${RUN_DIR}/base_model"; then
  evaluate_label \
    "base_model_run" \
    "${BASE_MODEL_CFG}" \
    "${RUN_DIR}/base_model" \
    "${summary_root}/base_model_run" \
    "" \
    "0" \
    "" \
    ""
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
    "${LORA_BASE_MODEL_PATH}" \
    "0" \
    "${LORA_MODEL_SUBFOLDER_VALUE}" \
    "${LORA_TOKENIZER_SUBFOLDER}"
done

if has_loadable_weights "${RUN_DIR}"; then
  evaluate_label \
    "final" \
    "${LORA_MODEL_CFG}" \
    "${RUN_DIR}" \
    "${summary_root}/final" \
    "${LORA_BASE_MODEL_PATH}" \
    "0" \
    "${LORA_MODEL_SUBFOLDER_VALUE}" \
    "${LORA_TOKENIZER_SUBFOLDER}"
elif [[ -f "${summary_root}/final/LMEval_SUMMARY.json" ]]; then
  echo "[utility][ckpt-eval] Reusing existing final utility results from ${summary_root}/final"
else
  echo "[utility][ckpt-eval] Skipping final utility eval because no top-level weights or cached final utility summary were found in ${RUN_DIR}."
fi

python src/tools/summarize_utility_metrics.py \
  --run-dir "${RUN_DIR}" \
  --output-path "${summary_root}/summary.tsv"
