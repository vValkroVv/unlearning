#!/usr/bin/env bash

set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")
repo_root=$(realpath "${script_dir}/../..")

FORGET_SPLIT=${FORGET_SPLIT:-forget_level2}
RETAIN_SPLIT=${RETAIN_SPLIT:-neighbor_level2}
DATASET_PATH=${DATASET_PATH:-SwetieePawsss/exp_r}
QUESTION_KEY=${QUESTION_KEY:-query}
ANSWER_KEY=${ANSWER_KEY:-answer}
OUT_DIR=${OUT_DIR:-${repo_root}/artifacts/dualcf/rwku/${FORGET_SPLIT}}
mkdir -p "${OUT_DIR}"

GENERATOR_BACKEND=${GENERATOR_BACKEND:-vllm_openai}
VLLM_BASE_URL=${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}
VLLM_API_KEY=${VLLM_API_KEY:-EMPTY}
VLLM_MODEL=${VLLM_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}
GENERATOR_CONCURRENCY=${GENERATOR_CONCURRENCY:-64}
GENERATOR_BATCH_SIZE=${GENERATOR_BATCH_SIZE:-256}

MODEL_CFG=${MODEL_CFG:-configs/model/Llama-3.1-8B-Instruct.yaml}
LORA_MODEL_CFG=${LORA_MODEL_CFG:-configs/model/Llama-3.1-8B-Instruct-lora.yaml}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}
W_CONF=${W_CONF:-1.0}
W_STABILITY=${W_STABILITY:-0.0}
STABILITY_MODE=${STABILITY_MODE:-none}
HYBRID_RHO=${HYBRID_RHO:-0.7}

RAW_CF=${OUT_DIR}/step1_counterfactuals_raw.jsonl
CLEAN_CF=${OUT_DIR}/step1b_counterfactuals_clean.jsonl
DIFF_JSONL=${OUT_DIR}/step2_difficulty_raw.jsonl
PROXY_MAP_JSONL=${OUT_DIR}/step2b_proxy_map.jsonl
ATTR_JSONL=${OUT_DIR}/step3_attribution_raw.jsonl
FINAL_JSONL=${OUT_DIR}/dualcf_${FORGET_SPLIT}_v2.jsonl

python "${repo_root}/src/tools/make_counterfactuals.py" \
  --dataset-path "${DATASET_PATH}" \
  --dataset-name "${FORGET_SPLIT}" \
  --split test \
  --output-path "${RAW_CF}" \
  --question-key "${QUESTION_KEY}" \
  --answer-key "${ANSWER_KEY}" \
  --generator-backend "${GENERATOR_BACKEND}" \
  --vllm-base-url "${VLLM_BASE_URL}" \
  --vllm-api-key "${VLLM_API_KEY}" \
  --vllm-model "${VLLM_MODEL}" \
  --generator-concurrency "${GENERATOR_CONCURRENCY}" \
  --generator-batch-size "${GENERATOR_BATCH_SIZE}" \
  --repair-invalid \
  --reject-gold-substring \
  --require-short-answer \
  --max-overlap-ratio 0.85 \
  --max-alt-length-chars 128

python "${repo_root}/src/tools/clean_counterfactuals.py" \
  --input-path "${RAW_CF}" \
  --output-path "${CLEAN_CF}" \
  --repair-invalid \
  --reject-gold-substring \
  --require-short-answer \
  --max-overlap-ratio 0.85 \
  --max-alt-length-chars 128

python "${repo_root}/src/tools/score_difficulty.py" \
  --input-path "${CLEAN_CF}" \
  --output-path "${DIFF_JSONL}" \
  --question-key "${QUESTION_KEY}" \
  --answer-key "${ANSWER_KEY}" \
  --model-cfg "${MODEL_CFG}" \
  --model-path "${BASE_MODEL_PATH}" \
  --tokenizer-path "${BASE_MODEL_PATH}" \
  --w-conf "${W_CONF}" \
  --w-stability "${W_STABILITY}" \
  --stability-mode "${STABILITY_MODE}"

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
  --sidecar-path "${OUT_DIR}/step2b_proxy_map_stats.json"

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
  --retain-batch-size 4 \
  --retain-max-steps 0 \
  --forget-max-steps 0 \
  --retain-proxy-mode hybrid \
  --retain-proxy-map "${PROXY_MAP_JSONL}" \
  --hybrid-rho "${HYBRID_RHO}" \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.0 \
  --lora-only

python "${repo_root}/src/tools/calibrate_dual_cf_scores.py" \
  --input-path "${ATTR_JSONL}" \
  --output-path "${FINAL_JSONL}" \
  --difficulty-in difficulty_score_raw \
  --difficulty-out difficulty_score \
  --attribution-in attribution_score_raw \
  --attribution-out attribution_score \
  --method percentile \
  --sidecar-path "${OUT_DIR}/step4_calibration_stats.json"

python "${repo_root}/src/tools/validate_dual_cf_artifact.py" \
  --input-path "${FINAL_JSONL}" \
  --question-key "${QUESTION_KEY}" \
  --reject-gold-substring \
  --require-short-answer \
  --max-alt-length-chars 128 \
  --check-overlap-ratio 0.85 \
  --strict

echo "[prepare_dual_cf_rwku_v2] Final artifact: ${FINAL_JSONL}"
