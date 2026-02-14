#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/extremessd10tb/borisiuk/open-unlearning"
cd "${PROJECT_ROOT}"

MODEL_NAME="Qwen2.5-7B-Instruct"
HF_MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR="${PROJECT_ROOT}/saves/finetune/qwen2p5_duet_full"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

GPU="${GPU:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
ACCELERATE_FLAGS=(--num_machines=1 --num_processes="${NUM_PROCESSES}" --mixed_precision=bf16 --dynamo_backend=no)

RUN_TAG="${RUN_TAG:-qwen_ft_$(date +%Y%m%d_%H%M)}"
LOG_FILE="${LOG_DIR}/${RUN_TAG}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "▶ PROJECT_ROOT : ${PROJECT_ROOT}"
echo "▶ MODEL        : ${MODEL_NAME}"
echo "▶ OUTPUT_DIR   : ${OUTPUT_DIR}"
echo "▶ GPU          : ${GPU}"

DATA_OVERRIDES=(
  "data.train.tripunlamb_full.args.hf_args.path=SwetieePawsss/DUET"
  "data.train.tripunlamb_full.args.hf_args.split=full_"
  "data.train.tripunlamb_full.args.question_key=question"
  "data.train.tripunlamb_full.args.answers_key=answer"
  "data.train.tripunlamb_full.args.max_length=512"
  "data.train.tripunlamb_full.args.model_name=${MODEL_NAME}"
)

CUDA_VISIBLE_DEVICES="${GPU}" accelerate launch "${ACCELERATE_FLAGS[@]}" src/train.py \
  --config-name=finetune.yaml \
  experiment=finetune/tripunlamb/default.yaml \
  task_name=qwen2p5_duet_full_ft \
  model=${MODEL_NAME} \
  model.model_args.pretrained_model_name_or_path=${HF_MODEL_ID} \
  trainer.args.output_dir=${OUTPUT_DIR} \
  trainer.args.logging_dir=${LOG_DIR} \
  "${DATA_OVERRIDES[@]}" \
  hydra.run.dir=. \
  hydra.output_subdir=null

echo "✅ Finetuning finished. Weights saved to ${OUTPUT_DIR}"
