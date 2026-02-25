#!/bin/bash

set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")
repo_root=$(realpath "${script_dir}/../../..")

# ===================== User-configurable parameters =====================
# Examples:
#   LRS="2e-3 2e-4" FULL_DYNAMIC=1 bash community/methods/ada_WGD/eval_paraph.sh
#   LRS="5e-4" FULL_DYNAMIC=0 RETAIN_SPLIT=city_fast_retain_500 bash community/methods/ada_WGD/eval_paraph.sh
#   FORGET_MODE=popular LRS="2e-4" FULL_DYNAMIC=0 bash community/methods/ada_WGD/eval_paraph.sh

DEVICES=${DEVICES:-0}
FULL_DYNAMIC=${FULL_DYNAMIC:-0}  # 1 => only adyn_bdyn, 0 => include all
LRS_STR=${LRS:-""}            # space-separated lr tags, e.g., "2e-3 2e-4"; empty => all
RETAIN_SPLIT=${RETAIN_SPLIT:-city_fast_retain_500}
FORGET_MODE=${FORGET_MODE:-both} # popular | rare | both (filters models by their training split)

BASE_MODEL=${BASE_MODEL:-Llama-3.1-8B-Instruct}
USE_SFT_BASE=${USE_SFT_BASE:-1}
HF_BASE_PATH=${HF_BASE_PATH:-meta-llama/${BASE_MODEL}}
LOCAL_SFT_BASE=${LOCAL_SFT_BASE:-/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/llama3.1-8b_full_3ep_ft_tripunlamb}

# Inference-time repetition controls for generation
REPETITION_PENALTY=${REPETITION_PENALTY:-1.1}
NO_REPEAT_NGRAM=${NO_REPEAT_NGRAM:-3}

# =======================================================================

export CUDA_VISIBLE_DEVICES=${DEVICES}

if [[ "${USE_SFT_BASE}" == "1" ]]; then
    base_model_path="${LOCAL_SFT_BASE}"
    echo "[eval_paraph] Using locally finetuned base checkpoint at ${base_model_path}"
else
    base_model_path="${HF_BASE_PATH}"
    echo "[eval_paraph] Using Hugging Face base checkpoint ${base_model_path}"
fi

read -r -a lrs <<< "${LRS_STR}"

MODEL_ROOT="${repo_root}/saves/unlearn/ada_WGD"
if [[ ! -d "${MODEL_ROOT}" ]]; then
    echo "[eval_paraph] Missing directory: ${MODEL_ROOT}"
    exit 1
fi

case "${FORGET_MODE}" in
    popular|rare|both) : ;;
    *)
        echo "[eval_paraph] Invalid FORGET_MODE: ${FORGET_MODE}. Use popular|rare|both."
        exit 1
        ;;
esac

for run_dir in "${MODEL_ROOT}"/duet_*; do
    [[ -d "${run_dir}" ]] || continue
    run_name=$(basename "${run_dir}")

    # Only ada_WGD runs
    if [[ "${run_name}" != *"_ada_WGD_"* ]]; then
        continue
    fi

    # Full-dynamic filter: require adyn_bdyn
    if [[ "${FULL_DYNAMIC}" == "1" && "${run_name}" != *"_adyn_bdyn_"* ]]; then
        continue
    fi

    # Determine which training split the model was trained on
    if [[ "${run_name}" == *"_city_forget_popular_5_"* ]]; then
        run_mode="popular"
        run_forget_split="paraphrases_city_forget_popular_5"
    elif [[ "${run_name}" == *"_city_forget_rare_5_"* ]]; then
        run_mode="rare"
        run_forget_split="paraphrases_city_forget_rare_5"
    else
        # Unknown training split
        continue
    fi

    # Filter models by FORGET_MODE
    if [[ "${FORGET_MODE}" == "popular" && "${run_mode}" != "popular" ]]; then
        continue
    fi
    if [[ "${FORGET_MODE}" == "rare" && "${run_mode}" != "rare" ]]; then
        continue
    fi

    # LR filter (match tag after _lr)
    lr_tag=$(echo "${run_name}" | sed -n 's/.*_lr\([^_]*\)_.*/\1/p')
    if [[ -n "${LRS_STR}" ]]; then
        keep_lr=0
        for lr in "${lrs[@]}"; do
            if [[ "${lr_tag}" == "${lr}" ]]; then
                keep_lr=1
                break
            fi
        done
        if [[ "${keep_lr}" != "1" ]]; then
            continue
        fi
    fi

    # Parse LoRA args from run name
    lora_r=$(echo "${run_name}" | sed -n 's/.*_lora_r\([^_]*\)_.*/\1/p')
    lora_alpha=$(echo "${run_name}" | sed -n 's/.*_lalpha\([^_]*\)_.*/\1/p')
    lora_dropout_tag=$(echo "${run_name}" | sed -n 's/.*_ldrop\([^_]*\)_.*/\1/p')
    lora_dropout=$(echo "${lora_dropout_tag}" | sed 's/p/./g')

    if [[ -z "${lora_r}" || -z "${lora_alpha}" || -z "${lora_dropout_tag}" ]]; then
        echo "[eval_paraph] Skip (failed to parse LoRA params): ${run_name}"
        continue
    fi

    # Evaluate only on the paraphrase split matching the training split
    for forget_split in "${run_forget_split}"; do
        eval_dir="${run_dir}/evals_${forget_split}"
        summary_path="${eval_dir}/DUET_SUMMARY.json"

        if [[ -f "${summary_path}" && "${FORCE_RERUN:-0}" != "1" ]]; then
            echo "[eval_paraph] Skipping ${run_name} on ${forget_split}: found ${summary_path}"
            continue
        fi

        echo "[eval_paraph] Evaluating ${run_name} on ${forget_split} (retain=${RETAIN_SPLIT})"
        mkdir -p "${eval_dir}"
        if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
            rm -f "${summary_path}" "${eval_dir}/DUET_EVAL.json"
        fi

        eval_cmd=( \
            experiment=eval/duet/default.yaml \
            model=${BASE_MODEL}-lora \
            forget_split=${forget_split} \
            holdout_split=${RETAIN_SPLIT} \
            task_name=${run_name} \
            model.model_args.pretrained_model_name_or_path=${run_dir} \
            model.model_args.base_model_name_or_path=${base_model_path} \
            model.model_args.device_map="auto" \
            model.model_args.low_cpu_mem_usage=true \
            model.lora_config.r=${lora_r} \
            model.lora_config.lora_alpha=${lora_alpha} \
            model.lora_config.lora_dropout=${lora_dropout} \
            eval.duet.metrics.forget_qa_rouge.generation_args.repetition_penalty=${REPETITION_PENALTY} \
            eval.duet.metrics.forget_qa_rouge.generation_args.no_repeat_ngram_size=${NO_REPEAT_NGRAM} \
            eval.duet.metrics.holdout_qa_rouge.generation_args.repetition_penalty=${REPETITION_PENALTY} \
            eval.duet.metrics.holdout_qa_rouge.generation_args.no_repeat_ngram_size=${NO_REPEAT_NGRAM} \
            eval.duet.overwrite=true \
            paths.output_dir=${eval_dir} \
            retain_logs_path=null \
        )

        python "${repo_root}/src/eval.py" "${eval_cmd[@]}"
    done
done
