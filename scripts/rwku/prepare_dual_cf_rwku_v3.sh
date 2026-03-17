#!/usr/bin/env bash

set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")
repo_root=$(realpath "${script_dir}/../..")

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[prepare_dual_cf_rwku_v3] Missing required file: ${path}" >&2
    exit 1
  fi
}

resolve_dataset_path() {
  local raw="$1"
  local suffix=""
  local basename_path=""

  if [[ "${raw}" == /* && -e "${raw}" ]]; then
    realpath "${raw}"
    return
  fi

  if [[ -e "${raw}" ]]; then
    realpath "${raw}"
    return
  fi

  if [[ -e "${repo_root}/${raw}" ]]; then
    realpath "${repo_root}/${raw}"
    return
  fi

  basename_path=$(basename "${raw}")
  case "${basename_path}" in
    DUET|exp_r) suffix="${basename_path}" ;;
  esac

  if [[ -n "${suffix}" ]]; then
    if [[ "${raw}" != "SwetieePawsss/${suffix}" ]]; then
      echo "[prepare_dual_cf_rwku_v3] Canonicalized dataset path ${raw} -> SwetieePawsss/${suffix}" >&2
    fi
    echo "SwetieePawsss/${suffix}"
    return
  fi

  echo "${raw}"
}

warn_if_dataset_unresolved() {
  local path="$1"
  if [[ "${path}" == /* && -e "${path}" ]]; then
    return
  fi
  if [[ "${path}" != /* && -e "${path}" ]]; then
    return
  fi
  if [[ "${path}" != /* && -e "${repo_root}/${path}" ]]; then
    return
  fi

  echo "[prepare_dual_cf_rwku_v3] Dataset path ${path} did not resolve to a local directory; the Python loader will try known RWKU aliases next." >&2
}

FORGET_SPLIT=${FORGET_SPLIT:-forget_level2}
RETAIN_SPLIT=${RETAIN_SPLIT:-neighbor_level2}
DATASET_PATH=${DATASET_PATH:-${RWKU_DATASET_PATH_LOCAL:-SwetieePawsss/exp_r}}
DATASET_PATH=$(resolve_dataset_path "${DATASET_PATH}")
warn_if_dataset_unresolved "${DATASET_PATH}"
QUESTION_KEY=${QUESTION_KEY:-query}
ANSWER_KEY=${ANSWER_KEY:-answer}
OUT_DIR=${OUT_DIR:-${repo_root}/artifacts/dualcf/rwku/${FORGET_SPLIT}}
mkdir -p "${OUT_DIR}"

echo "[prepare_dual_cf_rwku_v3] DATASET_PATH=${DATASET_PATH}"
echo "[prepare_dual_cf_rwku_v3] RETAIN_SPLIT=${RETAIN_SPLIT}"
echo "[prepare_dual_cf_rwku_v3] HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-<unset>}"

GENERATOR_BACKEND=${GENERATOR_BACKEND:-vllm_openai}
VLLM_BASE_URL=${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}
VLLM_API_KEY=${VLLM_API_KEY:-EMPTY}
VLLM_MODEL=${VLLM_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}
GENERATOR_CONCURRENCY=${GENERATOR_CONCURRENCY:-64}
GENERATOR_BATCH_SIZE=${GENERATOR_BATCH_SIZE:-256}
GENERATOR_TEMPERATURE=${GENERATOR_TEMPERATURE:-0.2}
GENERATOR_TOP_P=${GENERATOR_TOP_P:-0.8}
GENERATOR_MAX_NEW_TOKENS=${GENERATOR_MAX_NEW_TOKENS:-32}
NUM_ALTERNATES=${NUM_ALTERNATES:-4}
PROMPT_FAMILY=${PROMPT_FAMILY:-strict_short}

DIFFICULTY_BATCH_SIZE=${DIFFICULTY_BATCH_SIZE:-8}
ATTR_RETAIN_BATCH_SIZE=${ATTR_RETAIN_BATCH_SIZE:-4}
ATTR_RETAIN_MAX_STEPS=${ATTR_RETAIN_MAX_STEPS:-0}
ATTR_FORGET_MAX_STEPS=${ATTR_FORGET_MAX_STEPS:-0}

MODEL_CFG=${MODEL_CFG:-configs/model/Llama-3.1-8B-Instruct.yaml}
LORA_MODEL_CFG=${LORA_MODEL_CFG:-configs/model/Llama-3.1-8B-Instruct-lora.yaml}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}

BELIEF_MODEL_CFG=${BELIEF_MODEL_CFG:-${MODEL_CFG}}
BELIEF_MODEL_PATH=${BELIEF_MODEL_PATH:-${BASE_MODEL_PATH}}
BELIEF_TOKENIZER_PATH=${BELIEF_TOKENIZER_PATH:-${BASE_MODEL_PATH}}
BELIEF_MODEL_SUBFOLDER=${BELIEF_MODEL_SUBFOLDER:-}
BELIEF_TOKENIZER_SUBFOLDER=${BELIEF_TOKENIZER_SUBFOLDER:-}
BELIEF_MAX_NEW_TOKENS=${BELIEF_MAX_NEW_TOKENS:-16}
BELIEF_NUM_RETURN_SEQUENCES=${BELIEF_NUM_RETURN_SEQUENCES:-3}
BELIEF_NUM_BEAMS=${BELIEF_NUM_BEAMS:-4}

W_CONF=${W_CONF:-1.0}
W_STABILITY=${W_STABILITY:-0.0}
STABILITY_MODE=${STABILITY_MODE:-none}

SEMANTIC_TOP_K=${SEMANTIC_TOP_K:-8}
UTILITY_TOP_K=${UTILITY_TOP_K:-8}
EMBED_MODEL_NAME=${EMBED_MODEL_NAME:-sentence-transformers/all-MiniLM-L6-v2}
EMBED_BATCH_SIZE=${EMBED_BATCH_SIZE:-128}
EMBED_DEVICE=${EMBED_DEVICE:-cpu}
PROXY_WEIGHT_GLOBAL=${PROXY_WEIGHT_GLOBAL:-0.15}
PROXY_WEIGHT_SYNTAX=${PROXY_WEIGHT_SYNTAX:-0.45}
PROXY_WEIGHT_SEMANTIC=${PROXY_WEIGHT_SEMANTIC:-0.25}
PROXY_WEIGHT_UTILITY=${PROXY_WEIGHT_UTILITY:-0.15}
UTILITY_ANCHOR_JSONL=${UTILITY_ANCHOR_JSONL:-}

CF_SIDECAR_JSONL=${CF_SIDECAR_JSONL:-}
CF_SIDECAR_ALTERNATE_KEY=${CF_SIDECAR_ALTERNATE_KEY:-alternates}
CF_SIDECAR_SCORE_KEY=${CF_SIDECAR_SCORE_KEY:-scores}

RAW_CF=${OUT_DIR}/step1_counterfactuals_raw_v3.jsonl
CLEAN_CF=${OUT_DIR}/step1b_counterfactuals_clean_v3.jsonl
DIFF_JSONL=${OUT_DIR}/step2_difficulty_raw_v3.jsonl
PROXY_MAP_JSONL=${OUT_DIR}/step2b_proxy_map_v3.jsonl
ATTR_JSONL=${OUT_DIR}/step3_attribution_raw_v3.jsonl
BELIEF_JSONL=${OUT_DIR}/step3b_belief_raw_v3.jsonl
FINAL_JSONL=${OUT_DIR}/dualcf_${FORGET_SPLIT}_v3.jsonl

stop_after_clean_cf="${STOP_AFTER_CLEAN_CF:-0}"
skip_cf_generation="${SKIP_CF_GENERATION:-0}"
drop_invalid_after_clean="${DROP_INVALID_AFTER_CLEAN:-1}"
rebuild_clean_cf="${REBUILD_CLEAN_CF:-0}"
retry_invalid_cf_passes="${RETRY_INVALID_CF_PASSES:-0}"
retry_invalid_cf_concurrency="${RETRY_INVALID_CF_CONCURRENCY:-${GENERATOR_CONCURRENCY}}"
retry_invalid_cf_batch_size="${RETRY_INVALID_CF_BATCH_SIZE:-${GENERATOR_BATCH_SIZE}}"

clean_extra_args=()
if [[ "${drop_invalid_after_clean}" == "1" ]]; then
  clean_extra_args+=(--drop-invalid)
fi

make_cf_args=(
  --dataset-path "${DATASET_PATH}"
  --dataset-name "${FORGET_SPLIT}"
  --split test
  --output-path "${RAW_CF}"
  --question-key "${QUESTION_KEY}"
  --answer-key "${ANSWER_KEY}"
  --generator-backend "${GENERATOR_BACKEND}"
  --vllm-base-url "${VLLM_BASE_URL}"
  --vllm-api-key "${VLLM_API_KEY}"
  --vllm-model "${VLLM_MODEL}"
  --generator-concurrency "${GENERATOR_CONCURRENCY}"
  --generator-batch-size "${GENERATOR_BATCH_SIZE}"
  --temperature "${GENERATOR_TEMPERATURE}"
  --top-p "${GENERATOR_TOP_P}"
  --max-new-tokens "${GENERATOR_MAX_NEW_TOKENS}"
  --num-alternates "${NUM_ALTERNATES}"
  --prompt-family "${PROMPT_FAMILY}"
  --repair-invalid
  --reject-gold-substring
  --require-short-answer
  --max-overlap-ratio 0.85
  --max-alt-length-chars 128
)
if [[ -n "${CF_SIDECAR_JSONL}" ]]; then
  make_cf_args+=(
    --alternate-jsonl "${CF_SIDECAR_JSONL}"
    --mapping-key index
    --mapping-alternate-key "${CF_SIDECAR_ALTERNATE_KEY}"
    --external-score-key "${CF_SIDECAR_SCORE_KEY}"
    --allow-list-alternates
  )
fi

if [[ "${skip_cf_generation}" == "1" ]]; then
  require_file "${RAW_CF}"
  if [[ "${rebuild_clean_cf}" == "1" ]]; then
    python "${repo_root}/src/tools/clean_counterfactuals.py" \
      --input-path "${RAW_CF}" \
      --output-path "${CLEAN_CF}" \
      "${clean_extra_args[@]}" \
      --repair-invalid \
      --reject-gold-substring \
      --require-short-answer \
      --max-overlap-ratio 0.85 \
      --max-alt-length-chars 128
    echo "[prepare_dual_cf_rwku_v3] Rebuilt clean counterfactual file from raw output: ${CLEAN_CF}"
  else
    require_file "${CLEAN_CF}"
    echo "[prepare_dual_cf_rwku_v3] Reusing existing raw / clean counterfactual files from ${OUT_DIR}"
  fi
else
  python "${repo_root}/src/tools/make_counterfactuals.py" "${make_cf_args[@]}"

  if [[ "${retry_invalid_cf_passes}" != "0" ]]; then
    python "${repo_root}/src/tools/retry_invalid_counterfactuals.py" \
      --input-path "${RAW_CF}" \
      --output-path "${RAW_CF}" \
      --question-key "${QUESTION_KEY}" \
      --answer-key "${ANSWER_KEY}" \
      --mapping-key index \
      --retry-passes "${retry_invalid_cf_passes}" \
      --vllm-base-url "${VLLM_BASE_URL}" \
      --vllm-api-key "${VLLM_API_KEY}" \
      --vllm-model "${VLLM_MODEL}" \
      --generator-concurrency "${retry_invalid_cf_concurrency}" \
      --generator-batch-size "${retry_invalid_cf_batch_size}" \
      --temperature "${GENERATOR_TEMPERATURE}" \
      --top-p "${GENERATOR_TOP_P}" \
      --max-new-tokens "${GENERATOR_MAX_NEW_TOKENS}" \
      --num-alternates "${NUM_ALTERNATES}" \
      --prompt-family "${PROMPT_FAMILY}" \
      --reject-gold-substring \
      --require-short-answer \
      --max-overlap-ratio 0.85 \
      --max-alt-length-chars 128
  fi

  python "${repo_root}/src/tools/clean_counterfactuals.py" \
    --input-path "${RAW_CF}" \
    --output-path "${CLEAN_CF}" \
    "${clean_extra_args[@]}" \
    --repair-invalid \
    --reject-gold-substring \
    --require-short-answer \
    --max-overlap-ratio 0.85 \
    --max-alt-length-chars 128
fi

if [[ "${stop_after_clean_cf}" == "1" ]]; then
  echo "[prepare_dual_cf_rwku_v3] Stopping after clean counterfactual stage: ${CLEAN_CF}"
  exit 0
fi

python "${repo_root}/src/tools/score_difficulty.py" \
  --input-path "${CLEAN_CF}" \
  --output-path "${DIFF_JSONL}" \
  --question-key "${QUESTION_KEY}" \
  --answer-key "${ANSWER_KEY}" \
  --batch-size "${DIFFICULTY_BATCH_SIZE}" \
  --model-cfg "${MODEL_CFG}" \
  --model-path "${BASE_MODEL_PATH}" \
  --tokenizer-path "${BASE_MODEL_PATH}" \
  --w-conf "${W_CONF}" \
  --w-stability "${W_STABILITY}" \
  --stability-mode "${STABILITY_MODE}"

proxy_extra_args=()
attr_extra_args=()
if [[ -n "${UTILITY_ANCHOR_JSONL}" ]]; then
  require_file "${UTILITY_ANCHOR_JSONL}"
  proxy_extra_args+=(
    --utility-dataset-path json
    --utility-split train
    --utility-data-files "${UTILITY_ANCHOR_JSONL}"
    --utility-question-key question
    --utility-top-k "${UTILITY_TOP_K}"
  )
  attr_extra_args+=(
    --utility-dataset-path json
    --utility-split train
    --utility-data-files "${UTILITY_ANCHOR_JSONL}"
    --utility-question-key question
    --utility-answer-key answer
    --proxy-weight-utility "${PROXY_WEIGHT_UTILITY}"
  )
else
  echo "[prepare_dual_cf_rwku_v3] UTILITY_ANCHOR_JSONL not set; utility anchor bank disabled."
fi

python "${repo_root}/src/tools/build_proxy_retain_map.py" \
  --forget-dataset-path json \
  --forget-split train \
  --forget-data-files "${DIFF_JSONL}" \
  --retain-dataset-path "${DATASET_PATH}" \
  --retain-dataset-name "${RETAIN_SPLIT}" \
  --retain-split test \
  --output-path "${PROXY_MAP_JSONL}" \
  --forget-question-key "${QUESTION_KEY}" \
  --retain-question-key "${QUESTION_KEY}" \
  --top-k 16 \
  --fallback-top-k 8 \
  --semantic-top-k "${SEMANTIC_TOP_K}" \
  --embed-model-name "${EMBED_MODEL_NAME}" \
  --embed-batch-size "${EMBED_BATCH_SIZE}" \
  --embed-device "${EMBED_DEVICE}" \
  "${proxy_extra_args[@]}" \
  --sidecar-path "${OUT_DIR}/step2b_proxy_map_stats_v3.json"

python "${repo_root}/src/tools/score_attribution.py" \
  --model-cfg "${LORA_MODEL_CFG}" \
  --model-path "${BASE_MODEL_PATH}" \
  --tokenizer-path "${BASE_MODEL_PATH}" \
  --forget-dataset-path json \
  --forget-split train \
  --forget-data-files "${DIFF_JSONL}" \
  --retain-dataset-path "${DATASET_PATH}" \
  --retain-dataset-name "${RETAIN_SPLIT}" \
  --retain-split test \
  --output-path "${ATTR_JSONL}" \
  --question-key "${QUESTION_KEY}" \
  --retain-batch-size "${ATTR_RETAIN_BATCH_SIZE}" \
  --retain-max-steps "${ATTR_RETAIN_MAX_STEPS}" \
  --forget-max-steps "${ATTR_FORGET_MAX_STEPS}" \
  --retain-proxy-mode multi_bank \
  --retain-proxy-map "${PROXY_MAP_JSONL}" \
  --proxy-weight-global "${PROXY_WEIGHT_GLOBAL}" \
  --proxy-weight-syntax "${PROXY_WEIGHT_SYNTAX}" \
  --proxy-weight-semantic "${PROXY_WEIGHT_SEMANTIC}" \
  "${attr_extra_args[@]}" \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.0 \
  --lora-only

belief_model_subfolder_args=()
belief_tokenizer_subfolder_args=()
if [[ -n "${BELIEF_MODEL_SUBFOLDER}" ]]; then
  belief_model_subfolder_args+=(--model-subfolder "${BELIEF_MODEL_SUBFOLDER}")
fi
if [[ -n "${BELIEF_TOKENIZER_SUBFOLDER}" ]]; then
  belief_tokenizer_subfolder_args+=(--tokenizer-subfolder "${BELIEF_TOKENIZER_SUBFOLDER}")
fi

python "${repo_root}/src/tools/build_forget_belief_bank.py" \
  --input-path "${ATTR_JSONL}" \
  --output-path "${BELIEF_JSONL}" \
  --question-key "${QUESTION_KEY}" \
  --answer-key "${ANSWER_KEY}" \
  --alternate-key alternate \
  --belief-key belief_alternate \
  --belief-candidates-key belief_candidates \
  --model-cfg "${BELIEF_MODEL_CFG}" \
  --model-path "${BELIEF_MODEL_PATH}" \
  --tokenizer-path "${BELIEF_TOKENIZER_PATH}" \
  "${belief_model_subfolder_args[@]}" \
  "${belief_tokenizer_subfolder_args[@]}" \
  --num-return-sequences "${BELIEF_NUM_RETURN_SEQUENCES}" \
  --num-beams "${BELIEF_NUM_BEAMS}" \
  --max-new-tokens "${BELIEF_MAX_NEW_TOKENS}" \
  --require-short-answer \
  --max-alt-length-chars 128

python "${repo_root}/src/tools/calibrate_dual_cf_scores.py" \
  --input-path "${BELIEF_JSONL}" \
  --output-path "${FINAL_JSONL}" \
  --difficulty-in difficulty_score_raw \
  --difficulty-out difficulty_score \
  --attribution-in attribution_score_raw \
  --attribution-out attribution_score \
  --method percentile \
  --sidecar-path "${OUT_DIR}/step4_calibration_stats_v3.json"

python "${repo_root}/src/tools/validate_dual_cf_artifact.py" \
  --input-path "${FINAL_JSONL}" \
  --question-key "${QUESTION_KEY}" \
  --reject-gold-substring \
  --require-short-answer \
  --max-alt-length-chars 128 \
  --check-overlap-ratio 0.85 \
  --strict

echo "[prepare_dual_cf_rwku_v3] Final artifact: ${FINAL_JSONL}"
