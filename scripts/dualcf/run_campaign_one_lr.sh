#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/dualcf/run_campaign_one_lr.sh GPU_ID LR [PHASE]

Examples:
  bash scripts/dualcf/run_campaign_one_lr.sh 0 5e-6 duet_rare
  bash scripts/dualcf/run_campaign_one_lr.sh 1 1e-5 duet_popular
  bash scripts/dualcf/run_campaign_one_lr.sh 2 5e-5 duet_merged
  bash scripts/dualcf/run_campaign_one_lr.sh 3 1e-4 rwku
  bash scripts/dualcf/run_campaign_one_lr.sh 0 5e-6 duet_split_first

Phases:
  duet_rare         Run DUET rare only.
  duet_popular      Run DUET popular only.
  duet_split_first  Run DUET rare, then DUET popular.
  duet_merged       Run DUET merged only.
  duet_all          Run DUET rare, popular, then merged.
  rwku              Run RWKU only.
  all               Run DUET rare, popular, merged, then RWKU.

Defaults:
  PHASE defaults to duet_rare.
  The script expects artifacts to already exist under ARTIFACT_ROOT.
EOF
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage >&2
  exit 1
fi

GPU_ID=$1
LR=$2
PHASE=${3:-${PHASE:-duet_rare}}

script_dir=$(dirname "$(realpath "$0")")
repo_root=$(realpath "${script_dir}/../..")

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[dualcf][campaign] Missing required file: ${path}" >&2
    exit 1
  fi
}

resolve_existing_dir() {
  local raw_path="$1"
  local candidate=""

  if [[ -d "${raw_path}" ]]; then
    realpath "${raw_path}"
    return 0
  fi

  candidate="${REPO_ROOT}/${raw_path}"
  if [[ -d "${candidate}" ]]; then
    realpath "${candidate}"
    return 0
  fi

  candidate="/data/home/vkropoti/unlearning/${raw_path}"
  if [[ -d "${candidate}" ]]; then
    realpath "${candidate}"
    return 0
  fi

  printf '%s\n' "${raw_path}"
}

cleanup_loku_wrapper_tmp_dirs() {
  if [[ "${LOKU_WRAPPER_DELETE_TMP_AFTER_RUN:-1}" != "1" ]]; then
    return
  fi

  if [[ -n "${LOKU_TMP_IMPORTANCE_DIR:-}" && -d "${LOKU_TMP_IMPORTANCE_DIR}" ]]; then
    rm -rf "${LOKU_TMP_IMPORTANCE_DIR}"
    echo "[dualcf][campaign] Removed LoKU tmp importance dir ${LOKU_TMP_IMPORTANCE_DIR}"
  fi

  if [[ -n "${LOKU_TMP_FILA_DIR:-}" && -d "${LOKU_TMP_FILA_DIR}" ]]; then
    rm -rf "${LOKU_TMP_FILA_DIR}"
    echo "[dualcf][campaign] Removed LoKU tmp FILA dir ${LOKU_TMP_FILA_DIR}"
  fi
}

configure_method_variant_env() {
  local method_variant="$1"

  unset IMPORTANCE_PATH
  unset FILA_BASE_PATH
  unset DELETE_IMPORTANCE_AFTER_RUN
  unset DELETE_FILA_BASE_AFTER_EVAL
  unset DELETE_RUN_BASE_MODEL_AFTER_EVAL

  if [[ "${method_variant}" != "loku" ]]; then
    return
  fi

  export LOKU_TMP_IMPORTANCE_DIR="${LOKU_IMPORTANCE_TMP_DIR}/${LOKU_RUN_TAG}"
  export LOKU_TMP_FILA_DIR="${LOKU_FILA_BASE_TMP_DIR}/${LOKU_RUN_TAG}"
  mkdir -p "${LOKU_TMP_IMPORTANCE_DIR}" "${LOKU_TMP_FILA_DIR}"

  if [[ -n "${LOKU_IMPORTANCE_PATH:-}" ]]; then
    export IMPORTANCE_PATH="${LOKU_IMPORTANCE_PATH}"
  else
    export IMPORTANCE_PATH="${LOKU_TMP_IMPORTANCE_DIR}/{base_model}_{forget_label}_{retain_split}_{targets_tag}.pt"
  fi

  if [[ -n "${LOKU_FILA_BASE_PATH:-}" ]]; then
    export FILA_BASE_PATH="${LOKU_FILA_BASE_PATH}"
  else
    export FILA_BASE_PATH="${LOKU_TMP_FILA_DIR}/{task_name}"
  fi

  # Keep LoKU's internal EXIT trap from deleting debug artifacts on failure.
  # The wrapper removes these temp dirs only after the LoKU method returns
  # successfully, which is after train + endpoint eval + checkpoint eval +
  # Utility-1K eval.
  export DELETE_IMPORTANCE_AFTER_RUN="${LOKU_DELETE_IMPORTANCE_AFTER_RUN:-0}"
  export DELETE_FILA_BASE_AFTER_EVAL="${LOKU_DELETE_FILA_BASE_AFTER_EVAL:-0}"
  export DELETE_RUN_BASE_MODEL_AFTER_EVAL="${LOKU_DELETE_RUN_BASE_MODEL_AFTER_EVAL:-1}"

  echo "[dualcf][campaign] LoKU cleanup: IMPORTANCE_PATH=${IMPORTANCE_PATH}"
  echo "[dualcf][campaign] LoKU cleanup: FILA_BASE_PATH=${FILA_BASE_PATH}"
}

run_duet_block() {
  local forget_label="$1"
  local artifact_path="$2"

  require_file "${artifact_path}"

  export USE_SFT_BASE=1
  export LOCAL_SFT_BASE="${DUET_LOCAL_SFT_BASE}"
  export SFT_SUBFOLDER="${DUET_SFT_SUBFOLDER}"
  export TOKENIZER_MODEL_PATH="${DUET_LOCAL_SFT_BASE}"
  export TOKENIZER_SUBFOLDER="${DUET_SFT_SUBFOLDER}"
  export FORGET_LABEL="${forget_label}"
  export CF_DATASET_DATA_FILES="${artifact_path}"
  export MAX_STEPS="${MAX_STEPS:-0}"

  echo "[dualcf][campaign] GPU=${GPU_ID} LR=${LR} phase=duet_${forget_label}"
  for METHOD_VARIANT in ${METHOD_VARIANTS}; do
    export METHOD_VARIANT
    configure_method_variant_env "${METHOD_VARIANT}"
    bash "${repo_root}/scripts/duet/run_dualcf_ablation_v2.sh"
    if [[ "${METHOD_VARIANT}" == "loku" ]]; then
      cleanup_loku_wrapper_tmp_dirs
    fi
  done
}

run_rwku_block() {
  local artifact_path="$1"

  require_file "${artifact_path}"

  unset FORGET_LABEL
  unset FORGET_SPLIT_OVERRIDE
  unset RETAIN_SPLIT_OVERRIDE
  unset FORGET_LABEL_OVERRIDE
  unset MERGE_POPULARITY_FORGET
  unset USE_SFT_BASE
  unset LOCAL_SFT_BASE
  unset SFT_SUBFOLDER
  unset TOKENIZER_SUBFOLDER
  export BASE_MODEL_PATH="${HF_BASE_MODEL_PATH}"
  export TOKENIZER_MODEL_PATH="${HF_BASE_MODEL_PATH}"
  export FORGET_SPLIT="${FORGET_SPLIT:-forget_level2}"
  export RETAIN_SPLIT="${RETAIN_SPLIT:-neighbor_level2}"
  export CF_DATASET_DATA_FILES="${artifact_path}"
  export MAX_STEPS="${MAX_STEPS:-0}"

  echo "[dualcf][campaign] GPU=${GPU_ID} LR=${LR} phase=rwku"
  for METHOD_VARIANT in ${METHOD_VARIANTS}; do
    export METHOD_VARIANT
    configure_method_variant_env "${METHOD_VARIANT}"
    bash "${repo_root}/scripts/rwku/run_dualcf_ablation_v2.sh"
    if [[ "${METHOD_VARIANT}" == "loku" ]]; then
      cleanup_loku_wrapper_tmp_dirs
    fi
  done
}

export REPO_ROOT="${REPO_ROOT:-/home/vkropoti/diploma/open-unlearning}"
export VENV_PATH="${VENV_PATH:-/data/home/vkropoti/unlearning-venv}"

cd "${REPO_ROOT}"
source "${VENV_PATH}/bin/activate"

export HF_HOME="${HF_HOME:-/data/home/vkropoti/unlearning/.hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/home/vkropoti/unlearning/.hf_datasets_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/data/home/vkropoti/unlearning/.triton}"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-/data/home/vkropoti/unlearning/artifacts/dualcf}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/data/home/vkropoti/unlearning/saves/unlearn}"
export UTILITY_ROOT="${UTILITY_ROOT:-/data/home/vkropoti/unlearning/evals/utility_1k_v1}"
export BASELINE_CACHE_ROOT="${BASELINE_CACHE_ROOT:-/data/home/vkropoti/unlearning/saves/eval/utility_baselines}"
export LOKU_IMPORTANCE_TMP_DIR="${LOKU_IMPORTANCE_TMP_DIR:-/data/home/vkropoti/unlearning/importance_tmp}"
export LOKU_FILA_BASE_TMP_DIR="${LOKU_FILA_BASE_TMP_DIR:-/data/home/vkropoti/unlearning/fila_base_tmp}"
export LOKU_RUN_TAG="${LOKU_RUN_TAG:-gpu${GPU_ID}_lr${LR}_phase_${PHASE}}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRITON_CACHE_DIR}" \
  "${ARTIFACT_ROOT}" "${OUTPUT_ROOT}" "${UTILITY_ROOT}" "${BASELINE_CACHE_ROOT}" \
  "${LOKU_IMPORTANCE_TMP_DIR}" "${LOKU_FILA_BASE_TMP_DIR}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

export BASE_MODEL="${BASE_MODEL:-Llama-3.1-8B-Instruct}"
export MODEL_CONFIG="${MODEL_CONFIG:-Llama-3.1-8B-Instruct-lora}"
export MODEL_CFG="${MODEL_CFG:-configs/model/Llama-3.1-8B-Instruct.yaml}"
export LORA_MODEL_CFG="${LORA_MODEL_CFG:-configs/model/Llama-3.1-8B-Instruct-lora.yaml}"
export HF_BASE_MODEL_PATH="${HF_BASE_MODEL_PATH:-/data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct}"
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-${HF_BASE_MODEL_PATH}}"
export BASE_MODEL_EVAL_CONFIG="${BASE_MODEL_EVAL_CONFIG:-Llama-3.1-8B-Instruct}"
export LORA_MODEL_EVAL_CONFIG="${LORA_MODEL_EVAL_CONFIG:-Llama-3.1-8B-Instruct-lora}"

export DUET_LOCAL_SFT_BASE="${DUET_LOCAL_SFT_BASE:-/data/home/vkropoti/unlearning/SwetieePawsss/DUET_ft_models}"
export DUET_SFT_SUBFOLDER="${DUET_SFT_SUBFOLDER:-llama-3.1-8b-instruct-tripunlamb-ft}"
export DUET_LOCAL_SFT_BASE="$(resolve_existing_dir "${DUET_LOCAL_SFT_BASE}")"

export LORA_RS="${LORA_RS:-32}"
export LORA_ALPHAS="${LORA_ALPHAS:-64}"
export LORA_DROPOUTS="${LORA_DROPOUTS:-0.0}"

export TAU_DS="${TAU_DS:-0.6}"
export TAU_AS="${TAU_AS:-0.6}"
export TEMP_DS="${TEMP_DS:-0.15}"
export TEMP_AS="${TEMP_AS:-0.15}"
export LAMBDA_RET_HIS="${LAMBDA_RET_HIS:-3.0}"
export ALPHA_EFF_STATS="${ALPHA_EFF_STATS:-topk_mean}"
export ALPHA_EFF_TOPK_FRACS="${ALPHA_EFF_TOPK_FRACS:-0.25}"
export RISK_FORGET_SCALES="${RISK_FORGET_SCALES:-0.5}"

export LRS="${LR}"
export NUM_EPOCHS="${NUM_EPOCHS:-5}"
export CHECKPOINT_EVERY_HALF_EPOCH="${CHECKPOINT_EVERY_HALF_EPOCH:-1}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-12}"
export DELETE_MODEL_SAFETENSORS_AFTER_EVAL="${DELETE_MODEL_SAFETENSORS_AFTER_EVAL:-1}"
export DELETE_CHECKPOINT_ADAPTER_SAFETENSORS_AFTER_EVAL="${DELETE_CHECKPOINT_ADAPTER_SAFETENSORS_AFTER_EVAL:-1}"
export RUN_CHECKPOINT_EVAL="${RUN_CHECKPOINT_EVAL:-1}"
export RUN_UTILITY_EVAL="${RUN_UTILITY_EVAL:-1}"
export EVAL_RUN_BASE_MODEL="${EVAL_RUN_BASE_MODEL:-0}"
export UTILITY_EVAL_BATCH_SIZE="${UTILITY_EVAL_BATCH_SIZE:-512}"
export UTILITY_APPLY_CHAT_TEMPLATE="${UTILITY_APPLY_CHAT_TEMPLATE:-true}"

export PER_DEVICE_TRAIN_BS="${PER_DEVICE_TRAIN_BS:-16}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"
export IMPORTANCE_BATCH_SIZE="${IMPORTANCE_BATCH_SIZE:-32}"
export DIFFICULTY_BATCH_SIZE="${DIFFICULTY_BATCH_SIZE:-32}"
export ATTR_RETAIN_BATCH_SIZE="${ATTR_RETAIN_BATCH_SIZE:-4}"
export ATTR_RETAIN_MAX_STEPS="${ATTR_RETAIN_MAX_STEPS:-0}"
export ATTR_FORGET_MAX_STEPS="${ATTR_FORGET_MAX_STEPS:-0}"

METHOD_VARIANTS="${METHOD_VARIANTS:-full d_only a_only dpo simple_ce ga ada_pop npo simnpo npo_sam loku}"

echo "[dualcf][campaign] repo=${REPO_ROOT}"
echo "[dualcf][campaign] gpu=${GPU_ID} lr=${LR} phase=${PHASE}"
echo "[dualcf][campaign] method_variants=${METHOD_VARIANTS}"
echo "[dualcf][campaign] duet_local_sft_base=${DUET_LOCAL_SFT_BASE}"

duet_rare_artifact="${ARTIFACT_ROOT}/duet/rare_llama31_8b_v2/dualcf_rare_v2.jsonl"
duet_popular_artifact="${ARTIFACT_ROOT}/duet/popular_llama31_8b_v2/dualcf_popular_v2.jsonl"
duet_merged_artifact="${ARTIFACT_ROOT}/duet/merged_llama31_8b_v2/dualcf_merged_v2.jsonl"
rwku_artifact="${ARTIFACT_ROOT}/rwku/llama31_8b_level2_v2/dualcf_forget_level2_v2.jsonl"

case "${PHASE}" in
  duet_rare)
    run_duet_block rare "${duet_rare_artifact}"
    ;;
  duet_popular)
    run_duet_block popular "${duet_popular_artifact}"
    ;;
  duet_split_first)
    run_duet_block rare "${duet_rare_artifact}"
    run_duet_block popular "${duet_popular_artifact}"
    ;;
  duet_merged)
    run_duet_block merged "${duet_merged_artifact}"
    ;;
  duet_all)
    run_duet_block rare "${duet_rare_artifact}"
    run_duet_block popular "${duet_popular_artifact}"
    run_duet_block merged "${duet_merged_artifact}"
    ;;
  rwku)
    run_rwku_block "${rwku_artifact}"
    ;;
  all)
    run_duet_block rare "${duet_rare_artifact}"
    run_duet_block popular "${duet_popular_artifact}"
    run_duet_block merged "${duet_merged_artifact}"
    run_rwku_block "${rwku_artifact}"
    ;;
  *)
    echo "[dualcf][campaign] Unsupported PHASE=${PHASE}" >&2
    usage >&2
    exit 1
    ;;
esac
