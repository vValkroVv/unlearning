#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/extremessd10tb/borisiuk/open-unlearning"
cd "${PROJECT_ROOT}"

MODEL_NAME="Qwen2.5-7B-Instruct"
declare -A BASES=(
  ["orig"]="Qwen/Qwen2.5-7B-Instruct"
  ["ft"]="${PROJECT_ROOT}/saves/finetune/qwen2p5_duet_full"
)

GPU="${GPU:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
ACCELERATE_FLAGS=(--num_machines=1 --num_processes="${NUM_PROCESSES}" --mixed_precision=bf16 --dynamo_backend=no)

UNLEARN_EXPERIMENT="unlearn/tripunlamb/default.yaml"
EVAL_EXPERIMENT="eval/tripunlamb/default.yaml"

N_SUFFIX="${N_SUFFIX:-5}"
FORGET_SPLITS=("city_forget_rare_${N_SUFFIX}" "city_forget_popular_${N_SUFFIX}")
RETAIN_SPLIT="${RETAIN_SPLIT:-city_fast_retain_500}"

TRAINERS=(${TRAINERS_OVERRIDE:-GradAscent GradDiff NPO})
LEARNING_RATES=(${LR_OVERRIDE:-5e-6 1e-5 2e-5 4e-5})

RUN_ROOT="${PROJECT_ROOT}/saves/unlearn/qwen2p5_duet_${N_SUFFIX}N"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${RUN_ROOT}" "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/run_qwen_duet_${N_SUFFIX}N_$(date +%Y%m%d_%H%M).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "▶ PROJECT_ROOT : ${PROJECT_ROOT}"
echo "▶ MODEL        : ${MODEL_NAME}"
echo "▶ FORGET SPLITS: ${FORGET_SPLITS[*]}"
echo "▶ RETAIN SPLIT : ${RETAIN_SPLIT}"

function duet_dataset_overrides() {
  local forget="$1" retain="$2"
  printf "%s\n" \
    "data.forget.tripunlamb_popular_forget.args.hf_args.path=SwetieePawsss/DUET" \
    "data.forget.tripunlamb_popular_forget.args.hf_args.split=${forget}" \
    "data.forget.tripunlamb_popular_forget.args.model_name=${MODEL_NAME}" \
    "data.retain.tripunlamb_popular_retain.args.hf_args.path=SwetieePawsss/DUET" \
    "data.retain.tripunlamb_popular_retain.args.hf_args.split=${retain}" \
    "data.retain.tripunlamb_popular_retain.args.model_name=${MODEL_NAME}"
}

for base_label in "${!BASES[@]}"; do
  BASE_PATH="${BASES[$base_label]}"
  if [[ ! -d "${BASE_PATH}" && "${BASE_PATH}" != Qwen/* ]]; then
    echo "⚠️  Base path for ${base_label} not found (${BASE_PATH}). Skipping."
    continue
  fi

  echo; echo "================ BASE: ${base_label} (${BASE_PATH}) ================"

  for forget_split in "${FORGET_SPLITS[@]}"; do
    for trainer in "${TRAINERS[@]}"; do
      for lr in "${LEARNING_RATES[@]}"; do
        RUN_NAME="qwen_${trainer}_${forget_split}_${base_label}_lr${lr}"
        RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
        mkdir -p "${RUN_DIR}"

        if [[ -f "${RUN_DIR}/.done" && "${FORCE_RERUN:-0}" != "1" ]]; then
          echo "✅ Skip (already done): ${RUN_NAME}"
          continue
        fi

        echo "🎯 Training ${RUN_NAME}"
        mapfile -t DATA_OVERRIDES < <(duet_dataset_overrides "${forget_split}" "${RETAIN_SPLIT}")

        TRAIN_OVERRIDES=(
          "experiment=${UNLEARN_EXPERIMENT}"
          "trainer=${trainer}"
          "task_name=${RUN_NAME}"
          "model=${MODEL_NAME}"
          "forget_split=${forget_split}"
          "retain_split=${RETAIN_SPLIT}"
          "model.model_args.pretrained_model_name_or_path=${BASE_PATH}"
          "trainer.args.learning_rate=${lr}"
          "paths.output_dir=${RUN_DIR}"
          "${DATA_OVERRIDES[@]}"
        )
        if [[ "${trainer}" == "NPO" ]]; then
          TRAIN_OVERRIDES+=("trainer.method_args.beta=${NPO_BETA:-1.0}")
        fi

        CUDA_VISIBLE_DEVICES="${GPU}" accelerate launch "${ACCELERATE_FLAGS[@]}" src/train.py \
          --config-name=unlearn.yaml \
          "${TRAIN_OVERRIDES[@]}" \
          > "${RUN_DIR}/train.log" 2>&1
        echo "OK" > "${RUN_DIR}/.done"

        echo "📊 Evaluating ${RUN_NAME}"
        mkdir -p "${RUN_DIR}"/evals
        EVAL_OVERRIDES=(
          "experiment=${EVAL_EXPERIMENT}"
          "task_name=${RUN_NAME}_eval"
          "model=${MODEL_NAME}"
          "forget_split=${forget_split}"
          "paths.output_dir=${RUN_DIR}/evals"
          "model.model_args.pretrained_model_name_or_path=${RUN_DIR}"
          "${DATA_OVERRIDES[@]}"
        )

        python src/eval.py \
          --config-name=eval.yaml \
          "${EVAL_OVERRIDES[@]}" \
          > "${RUN_DIR}/evals/eval.log" 2>&1 || echo "⚠️  Eval failed for ${RUN_NAME}"
      done
    done
  done
done

echo "🎉 All requested runs finished. Check ${RUN_ROOT} for outputs."
