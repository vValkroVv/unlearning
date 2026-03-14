#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 RUN_DIR FORGET_SPLIT RETAIN_SPLIT BASE_MODEL_PATH TOKENIZER_PATH [MODEL_CONFIG]" >&2
  exit 1
fi

RUN_DIR=$1
FORGET_SPLIT=$2
RETAIN_SPLIT=$3
BASE_MODEL_PATH=$4
TOKENIZER_PATH=$5
MODEL_CONFIG=${6:-Llama-3.1-8B-Instruct-lora}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-64}
DELETE_RUN_BASE_MODEL_AFTER_EVAL=${DELETE_RUN_BASE_MODEL_AFTER_EVAL:-0}

RESOLVED_BASE_MODEL_PATH=${FILA_BASE_PATH:-${BASE_MODEL_PATH}}
if [[ -d "${RUN_DIR}/base_model" ]]; then
  RESOLVED_BASE_MODEL_PATH="${RUN_DIR}/base_model"
  echo "[duet][ckpt-eval] Detected LoKU FILA base model at ${RESOLVED_BASE_MODEL_PATH}"
fi

mapfile -t CKPTS < <(find "${RUN_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
CKPTS+=("${RUN_DIR}")

for ckpt in "${CKPTS[@]}"; do
  name=$(basename "${ckpt}")
  out_dir="${RUN_DIR}/checkpoint_evals/${name}"
  mkdir -p "${out_dir}"
  python src/eval.py \
    experiment=eval/duet/default.yaml \
    model=${MODEL_CONFIG} \
    forget_split=${FORGET_SPLIT} \
    holdout_split=${RETAIN_SPLIT} \
    task_name=$(basename "${RUN_DIR}")_${name} \
    model.model_args.pretrained_model_name_or_path=${ckpt} \
    ++model.model_args.base_model_name_or_path=${RESOLVED_BASE_MODEL_PATH} \
    model.tokenizer_args.pretrained_model_name_or_path=${TOKENIZER_PATH} \
    eval.duet.batch_size=${EVAL_BATCH_SIZE} \
    eval.duet.overwrite=true \
    paths.output_dir=${out_dir} \
    retain_logs_path=null
done

python src/tools/summarize_checkpoint_metrics.py \
  --run-dir "${RUN_DIR}" \
  --output-path "${RUN_DIR}/checkpoint_evals/summary.tsv"

if [[ "${DELETE_RUN_BASE_MODEL_AFTER_EVAL}" == "1" && "${RESOLVED_BASE_MODEL_PATH}" == "${RUN_DIR}/base_model" ]]; then
  rm -rf "${RESOLVED_BASE_MODEL_PATH}"
  echo "[duet][ckpt-eval] Removed FILA base model ${RESOLVED_BASE_MODEL_PATH}"
fi
