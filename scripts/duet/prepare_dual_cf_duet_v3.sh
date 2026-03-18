#!/usr/bin/env bash

set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")
repo_root=$(realpath "${script_dir}/../..")

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[prepare_dual_cf_duet_v3] Missing required file: ${path}" >&2
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
      echo "[prepare_dual_cf_duet_v3] Canonicalized dataset path ${raw} -> SwetieePawsss/${suffix}" >&2
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

  echo "[prepare_dual_cf_duet_v3] Dataset path ${path} did not resolve to a local directory; the Python loader will try known DUET aliases next." >&2
}

FORGET_LABEL=${FORGET_LABEL:-rare}
case "${FORGET_LABEL}" in
  rare) FORGET_SPLIT=${FORGET_SPLIT:-city_forget_rare_5} ;;
  popular) FORGET_SPLIT=${FORGET_SPLIT:-city_forget_popular_5} ;;
  merged) FORGET_SPLIT=${FORGET_SPLIT:-city_forget_rare_5+city_forget_popular_5} ;;
  *) echo "[prepare_dual_cf_duet_v3] Unsupported FORGET_LABEL=${FORGET_LABEL}" >&2; exit 1 ;;
esac

DATASET_PATH=${DATASET_PATH:-${DUET_DATASET_PATH_LOCAL:-SwetieePawsss/DUET}}
DATASET_PATH=$(resolve_dataset_path "${DATASET_PATH}")
warn_if_dataset_unresolved "${DATASET_PATH}"
RETAIN_SPLIT=${RETAIN_SPLIT:-city_fast_retain_500}
QUESTION_KEY=${QUESTION_KEY:-question}
ANSWER_KEY=${ANSWER_KEY:-answer}
OUT_DIR=${OUT_DIR:-${repo_root}/artifacts/dualcf/duet/${FORGET_LABEL}}
mkdir -p "${OUT_DIR}"

echo "[prepare_dual_cf_duet_v3] DATASET_PATH=${DATASET_PATH}"
echo "[prepare_dual_cf_duet_v3] RETAIN_SPLIT=${RETAIN_SPLIT}"
echo "[prepare_dual_cf_duet_v3] HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-<unset>}"

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
PROMPT_FAMILY=${PROMPT_FAMILY:-duet_relation_safe}
MAX_EXAMPLES=${MAX_EXAMPLES:-0}
ALLOW_LOW_CONFIDENCE_FALLBACK=${ALLOW_LOW_CONFIDENCE_FALLBACK:-0}
if [[ "${GENERATOR_BACKEND}" == "vllm_openai" && "${NUM_ALTERNATES}" != "1" ]]; then
  export VLLM_USE_STRUCTURED_OUTPUTS=${VLLM_USE_STRUCTURED_OUTPUTS:-1}
fi

DIFFICULTY_BATCH_SIZE=${DIFFICULTY_BATCH_SIZE:-8}
ATTR_RETAIN_BATCH_SIZE=${ATTR_RETAIN_BATCH_SIZE:-4}
ATTR_RETAIN_MAX_STEPS=${ATTR_RETAIN_MAX_STEPS:-0}
ATTR_FORGET_MAX_STEPS=${ATTR_FORGET_MAX_STEPS:-0}

MODEL_CFG=${MODEL_CFG:-configs/model/Llama-3.1-8B-Instruct.yaml}
LORA_MODEL_CFG=${LORA_MODEL_CFG:-configs/model/Llama-3.1-8B-Instruct-lora.yaml}
SFT_MODEL_PATH=${SFT_MODEL_PATH:-SwetieePawsss/DUET_ft_models}
SFT_SUBFOLDER=${SFT_SUBFOLDER-llama-3.1-8b-instruct-tripunlamb-ft}

BELIEF_MODEL_CFG=${BELIEF_MODEL_CFG:-${MODEL_CFG}}
BELIEF_MODEL_PATH=${BELIEF_MODEL_PATH:-${SFT_MODEL_PATH}}
BELIEF_TOKENIZER_PATH=${BELIEF_TOKENIZER_PATH:-${SFT_MODEL_PATH}}
BELIEF_MODEL_SUBFOLDER=${BELIEF_MODEL_SUBFOLDER:-${SFT_SUBFOLDER}}
BELIEF_TOKENIZER_SUBFOLDER=${BELIEF_TOKENIZER_SUBFOLDER:-${SFT_SUBFOLDER}}
BELIEF_MAX_NEW_TOKENS=${BELIEF_MAX_NEW_TOKENS:-16}
BELIEF_NUM_RETURN_SEQUENCES=${BELIEF_NUM_RETURN_SEQUENCES:-3}
BELIEF_NUM_BEAMS=${BELIEF_NUM_BEAMS:-4}

W_POP=${W_POP:-1.0}
W_CONF=${W_CONF:-1.0}
W_STAGE=${W_STAGE:-1.0}
W_STABILITY=${W_STABILITY:-0.0}
STABILITY_MODE=${STABILITY_MODE:-none}
STAGE_COLUMN=${STAGE_COLUMN:-}
STAGE_MAP_JSON=${STAGE_MAP_JSON:-}

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
CF_SIDECAR_RELATION_SCORE_KEY=${CF_SIDECAR_RELATION_SCORE_KEY:-relation_scores}
CF_SIDECAR_SHARED_FACT_SCORE_KEY=${CF_SIDECAR_SHARED_FACT_SCORE_KEY:-shared_fact_scores}
CF_SIDECAR_SOURCE_KEY=${CF_SIDECAR_SOURCE_KEY:-candidate_sources}

CANDIDATE_BANK_JSONL=${OUT_DIR}/step0_candidate_bank.jsonl
RAW_CF=${OUT_DIR}/step1_counterfactuals_raw_v3.jsonl
CLEAN_CF=${OUT_DIR}/step1b_counterfactuals_clean_v3.jsonl
DIFF_JSONL=${OUT_DIR}/step2_difficulty_raw_v3.jsonl
PROXY_MAP_JSONL=${OUT_DIR}/step2b_proxy_map_v3.jsonl
ATTR_JSONL=${OUT_DIR}/step3_attribution_raw_v3.jsonl
BELIEF_JSONL=${OUT_DIR}/step3b_belief_raw_v3.jsonl
FINAL_JSONL=${OUT_DIR}/dualcf_${FORGET_LABEL}_v3.jsonl
ARTIFACT_REPORT_JSON=${OUT_DIR}/step4_artifact_report_v3.json

stop_after_clean_cf="${STOP_AFTER_CLEAN_CF:-0}"
skip_cf_generation="${SKIP_CF_GENERATION:-0}"
drop_invalid_after_clean="${DROP_INVALID_AFTER_CLEAN:-1}"
rebuild_clean_cf="${REBUILD_CLEAN_CF:-0}"

clean_extra_args=()
if [[ "${drop_invalid_after_clean}" == "1" ]]; then
  clean_extra_args+=(--drop-invalid)
fi
if [[ "${ALLOW_LOW_CONFIDENCE_FALLBACK}" == "1" ]]; then
  clean_extra_args+=(--allow-low-confidence-fallback)
fi

make_cf_args=(
  --dataset-path "${DATASET_PATH}"
  --split "${FORGET_SPLIT}"
  --output-path "${RAW_CF}"
  --question-key "${QUESTION_KEY}"
  --answer-key "${ANSWER_KEY}"
  --candidate-bank "${CANDIDATE_BANK_JSONL}"
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
  --max-examples "${MAX_EXAMPLES}"
  --reject-gold-substring
  --require-short-answer
  --max-overlap-ratio 0.85
  --max-alt-length-chars 128
)
if [[ "${ALLOW_LOW_CONFIDENCE_FALLBACK}" == "1" ]]; then
  make_cf_args+=(--allow-low-confidence-fallback)
fi
if [[ -n "${CF_SIDECAR_JSONL}" ]]; then
  make_cf_args+=(
    --alternate-jsonl "${CF_SIDECAR_JSONL}"
    --mapping-key index
    --mapping-alternate-key "${CF_SIDECAR_ALTERNATE_KEY}"
    --external-score-key "${CF_SIDECAR_SCORE_KEY}"
    --external-relation-score-key "${CF_SIDECAR_RELATION_SCORE_KEY}"
    --external-shared-fact-score-key "${CF_SIDECAR_SHARED_FACT_SCORE_KEY}"
    --external-source-key "${CF_SIDECAR_SOURCE_KEY}"
    --allow-list-alternates
  )
fi

if [[ "${skip_cf_generation}" == "1" ]]; then
  require_file "${CANDIDATE_BANK_JSONL}"
  require_file "${RAW_CF}"
  if [[ "${rebuild_clean_cf}" == "1" ]]; then
    python "${repo_root}/src/tools/clean_counterfactuals.py" \
      --input-path "${RAW_CF}" \
      --output-path "${CLEAN_CF}" \
      --candidate-bank "${CANDIDATE_BANK_JSONL}" \
      --repair-invalid \
      "${clean_extra_args[@]}" \
      --reject-gold-substring \
      --require-short-answer \
      --max-overlap-ratio 0.85 \
      --max-alt-length-chars 128
    echo "[prepare_dual_cf_duet_v3] Rebuilt clean counterfactual file from raw output: ${CLEAN_CF}"
  else
    require_file "${CLEAN_CF}"
    echo "[prepare_dual_cf_duet_v3] Reusing existing candidate / raw / clean counterfactual files from ${OUT_DIR}"
  fi
else
  python "${repo_root}/src/tools/build_duet_candidate_bank.py" \
    --dataset-path "${DATASET_PATH}" \
    --split "${FORGET_SPLIT}" \
    --output-path "${CANDIDATE_BANK_JSONL}" \
    --question-key "${QUESTION_KEY}" \
    --answer-key "${ANSWER_KEY}" \
    --candidates-per-row 12 \
    --max-examples "${MAX_EXAMPLES}" \
    --sidecar-path "${OUT_DIR}/step0_candidate_bank_stats_v3.json"

  python "${repo_root}/src/tools/make_counterfactuals.py" "${make_cf_args[@]}"

  python "${repo_root}/src/tools/clean_counterfactuals.py" \
    --input-path "${RAW_CF}" \
    --output-path "${CLEAN_CF}" \
    --candidate-bank "${CANDIDATE_BANK_JSONL}" \
    --repair-invalid \
    "${clean_extra_args[@]}" \
    --reject-gold-substring \
    --require-short-answer \
    --max-overlap-ratio 0.85 \
    --max-alt-length-chars 128
fi

if [[ "${stop_after_clean_cf}" == "1" ]]; then
  echo "[prepare_dual_cf_duet_v3] Stopping after clean counterfactual stage: ${CLEAN_CF}"
  exit 0
fi

diff_extra_args=()
if [[ -n "${STAGE_COLUMN}" ]]; then
  diff_extra_args+=(--stage-column "${STAGE_COLUMN}")
fi
if [[ -n "${STAGE_MAP_JSON}" ]]; then
  diff_extra_args+=(--stage-map-json "${STAGE_MAP_JSON}")
fi

python "${repo_root}/src/tools/score_difficulty.py" \
  --input-path "${CLEAN_CF}" \
  --output-path "${DIFF_JSONL}" \
  --question-key "${QUESTION_KEY}" \
  --answer-key "${ANSWER_KEY}" \
  --batch-size "${DIFFICULTY_BATCH_SIZE}" \
  --max-examples "${MAX_EXAMPLES}" \
  --model-cfg "${MODEL_CFG}" \
  --model-path "${SFT_MODEL_PATH}" \
  --tokenizer-path "${SFT_MODEL_PATH}" \
  --model-subfolder "${SFT_SUBFOLDER}" \
  --tokenizer-subfolder "${SFT_SUBFOLDER}" \
  --popularity-column pop_sum \
  --w-pop "${W_POP}" \
  --w-conf "${W_CONF}" \
  --w-stage "${W_STAGE}" \
  --w-stability "${W_STABILITY}" \
  --stability-mode "${STABILITY_MODE}" \
  "${diff_extra_args[@]}"

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
  echo "[prepare_dual_cf_duet_v3] UTILITY_ANCHOR_JSONL not set; utility anchor bank disabled."
fi

python "${repo_root}/src/tools/build_proxy_retain_map.py" \
  --forget-dataset-path json \
  --forget-split train \
  --forget-data-files "${DIFF_JSONL}" \
  --retain-dataset-path "${DATASET_PATH}" \
  --retain-split "${RETAIN_SPLIT}" \
  --output-path "${PROXY_MAP_JSONL}" \
  --forget-question-key "${QUESTION_KEY}" \
  --retain-question-key "${QUESTION_KEY}" \
  --max-examples "${MAX_EXAMPLES}" \
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
  --model-path "${SFT_MODEL_PATH}" \
  --tokenizer-path "${SFT_MODEL_PATH}" \
  --model-subfolder "${SFT_SUBFOLDER}" \
  --tokenizer-subfolder "${SFT_SUBFOLDER}" \
  --forget-dataset-path json \
  --forget-split train \
  --forget-data-files "${DIFF_JSONL}" \
  --retain-dataset-path "${DATASET_PATH}" \
  --retain-split "${RETAIN_SPLIT}" \
  --output-path "${ATTR_JSONL}" \
  --question-key "${QUESTION_KEY}" \
  --retain-batch-size "${ATTR_RETAIN_BATCH_SIZE}" \
  --forget-max-examples "${MAX_EXAMPLES}" \
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
  --model-subfolder "${BELIEF_MODEL_SUBFOLDER}" \
  --tokenizer-subfolder "${BELIEF_TOKENIZER_SUBFOLDER}" \
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
  --strict \
  --report-path "${ARTIFACT_REPORT_JSON}"

echo "[prepare_dual_cf_duet_v3] Final artifact: ${FINAL_JSONL}"
