#!/bin/bash

set -euo pipefail

repo_root=$(realpath "$(dirname "$0")/..")

BENCHMARK="${BENCHMARK:-duet}"   # duet | popqa | rwku
BASE_MODEL="${BASE_MODEL:-Llama-3.1-8B-Instruct}"
MODEL_CONFIG="${MODEL_CONFIG:-${BASE_MODEL}-lora}"
HF_BASE_MODEL_PATH="${HF_BASE_MODEL_PATH:-meta-llama/${BASE_MODEL}}"
USE_SFT_BASE="${USE_SFT_BASE:-1}"
LOCAL_SFT_BASE="${LOCAL_SFT_BASE:-}"

case "${BENCHMARK}" in
  duet)
    experiment="eval/duet/default.yaml"
    splits=( "city_forget_rare_5 city_fast_retain_500" "city_forget_popular_5 city_fast_retain_500" )
    output_root="${repo_root}/saves/evals/duet_base"
    ;;
  popqa)
    experiment="eval/popqa/default.yaml"
    splits=( "rare_forget5_sum fast_retain_500" "popular_forget5_sum fast_retain_500" )
    output_root="${repo_root}/saves/evals/popqa_base"
    ;;
  rwku)
    experiment="eval/rwku/default.yaml"
    splits=( "forget_level2 neighbor_level2" )
    output_root="${repo_root}/saves/evals/rwku_base"
    USE_SFT_BASE=0
    ;;
  *)
    echo "Unknown BENCHMARK: ${BENCHMARK}"
    exit 1
    ;;
esac

if [[ "${USE_SFT_BASE}" == "1" && -n "${LOCAL_SFT_BASE}" ]]; then
  base_model_path="${LOCAL_SFT_BASE}"
  echo "[base-eval] Using SFT checkpoint: ${base_model_path}"
else
  base_model_path="${HF_BASE_MODEL_PATH}"
  echo "[base-eval] Using HF base: ${base_model_path}"
fi

mkdir -p "${output_root}"

for split in "${splits[@]}"; do
  forget_split=$(echo "$split" | cut -d' ' -f1)
  retain_split=$(echo "$split" | cut -d' ' -f2)
  task_name="${BENCHMARK}_${BASE_MODEL}_${forget_split}_base_eval"
  eval_dir="${output_root}/${task_name}"
  mkdir -p "${eval_dir}"

  python src/eval.py \
    experiment=${experiment} \
    model=${MODEL_CONFIG} \
    forget_split=${forget_split} \
    holdout_split=${retain_split} \
    task_name=${task_name} \
    model.model_args.pretrained_model_name_or_path=${base_model_path} \
    model.model_args.base_model_name_or_path=${base_model_path} \
    model.model_args.device_map=auto \
    model.model_args.low_cpu_mem_usage=true \
    eval.duet.overwrite=true \
    paths.output_dir=${eval_dir} \
    retain_logs_path=null
done
