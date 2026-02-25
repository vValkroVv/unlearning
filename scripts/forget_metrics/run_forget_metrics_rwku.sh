#!/bin/bash

set -euo pipefail

repo_root=$(realpath "$(dirname "$0")/../..")

bench="rwku"
forget_split="${FORGET_SPLIT:-forget_level2}"
retain_split="${RETAIN_SPLIT:-neighbor_level2}"
lr_tag="lr5e-4"
GPU_ID="${GPU_ID:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
AMP_MODE="${AMP_MODE:-bf16}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"

models=(
  "Llama-3.1-8B|Llama-3.1-8B-lora|meta-llama/Llama-3.1-8B"
  "gemma-7b-it|gemma-7b-it-lora|google/gemma-7b-it"
  "Qwen2.5-7B-Instruct|Qwen2.5-7B-Instruct-lora|Qwen/Qwen2.5-7B-Instruct"
)

for model_row in "${models[@]}"; do
  IFS='|' read -r base_model model_cfg base_path <<< "${model_row}"
  echo "[metrics][rwku] model=${base_model}"

  for algo_dir in "${repo_root}/saves/unlearn/rwku"/*; do
    [ -d "${algo_dir}" ] || continue
    algo=$(basename "${algo_dir}")
    for run_dir in "${algo_dir}"/pretrained/*"${lr_tag}"* "${algo_dir}"/*"${lr_tag}"*; do
      [ -d "${run_dir}" ] || continue
      if [[ "${run_dir}" != *"${base_model}"* ]]; then
        continue
      fi
      eval_dir="${run_dir}/evals"
      mkdir -p "${eval_dir}"
      echo "[metrics][rwku] ${algo} -> ${run_dir}"
      python "${repo_root}/scripts/forget_metrics/forget_metrics.py" \
        --benchmark "${bench}" \
        --forget-split "${forget_split}" \
        --retain-split "${retain_split}" \
        --model-config "${model_cfg}" \
        --base-model-path "${base_path}" \
        --adapter-path "${run_dir}" \
        --output-dir "${eval_dir}" \
        --batch-size "${BATCH_SIZE}" \
        --amp "${AMP_MODE}" \
        --gpu "${GPU_ID}" \
        --num-workers "${NUM_WORKERS}" \
        --prefetch-factor "${PREFETCH_FACTOR}"
    done
  done
done
