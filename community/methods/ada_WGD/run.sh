#!/bin/bash

set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")
repo_root=$(realpath "${script_dir}/../../..")

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# Speed up and make HF downloads more robust
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export HF_TIMEOUT=${HF_TIMEOUT:-60}
export HF_MAX_RETRIES=${HF_MAX_RETRIES:-5}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"${repo_root}/hf_cache"}

# ===================== User-configurable parameters =====================
# Override via env, e.g.:
#   DEVICES=0 NUM_EPOCHS=10 LRS="2e-5 4e-5" GAMMAS="1.0 3.0" RETAIN_EPS=1.2 INIT_LAMBDA=0.3 \
#   DUAL_STEP_SIZE=0.05 DUAL_UPDATE_UPON=epoch DUAL_WARMUP_EPOCHS=1 REP_COEFF=0.0 USE_SFT_BASE=1 \
#   bash community/methods/ada_WGD/run.sh

DEVICES=${DEVICES:-0}

BASE_MODEL=${BASE_MODEL:-Llama-3.1-8B-Instruct}
USE_SFT_BASE=${USE_SFT_BASE:-1}
HF_BASE_PATH=${HF_BASE_PATH:-meta-llama/${BASE_MODEL}}
LOCAL_SFT_BASE=${LOCAL_SFT_BASE:-/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/llama3.1-8b_full_3ep_ft_tripunlamb}

LORA_R=${LORA_R:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}

PER_DEVICE_TRAIN_BS=${PER_DEVICE_TRAIN_BS:-1}
GRAD_ACCUM=${GRAD_ACCUM:-32}
NUM_EPOCHS=${NUM_EPOCHS:-10}
LRS_STR=${LRS:-"2e-5 5e-5 8e-5"} # 2e-3 5e-3 8e-3;; 2e-4 5e-4 8e-4 ;;  3e-4 2e-4 8e-5 4e-5 1e-5 ;; 1e-5 2e-5 4e-5 6e-5 8e-5 1e-4 2e-4 3e-4 5e-4
GAMMAS_STR=${GAMMAS:-"1.0"}

WARMUP_EPOCHS=${WARMUP_EPOCHS:-0.0}
 # set to a float (e.g., 0.5) for constant alpha; 'none' keeps dynamic
ALPHA_CONST=${ALPHA_CONST:-0.5 none}
BETA_CONST=${BETA_CONST:-0.1 none}
# BETA_CONST=${BETA_CONST:-none}

# Inference-time repetition controls for generation
REPETITION_PENALTY=${REPETITION_PENALTY:-1.1}
NO_REPEAT_NGRAM=${NO_REPEAT_NGRAM:-3}

# =======================================================================

base_model="${BASE_MODEL}"
lora_model="${BASE_MODEL}-lora"
hf_base_model_path="${HF_BASE_PATH}"
local_sft_base="${LOCAL_SFT_BASE}"

use_sft_base=${USE_SFT_BASE}

if [[ "${use_sft_base}" == "1" ]]; then
    base_model_path="${local_sft_base}"
    echo "[ada_WGD] Using locally finetuned base checkpoint at ${base_model_path}"
else
    base_model_path="${hf_base_model_path}"
    echo "[ada_WGD] Using Hugging Face base checkpoint ${base_model_path}"
fi

trainer="AdaWGD"
experiment="unlearn/duet/wga_lora.yaml"

forget_retain_splits=(
    "city_forget_rare_5 city_fast_retain_500"
    #"city_forget_popular_5 city_fast_retain_500"
)

per_device_train_batch_size=${PER_DEVICE_TRAIN_BS}
gradient_accumulation_steps=${GRAD_ACCUM}
read -r -a lrs <<< "${LRS_STR}"
read -r -a gammas <<< "${GAMMAS_STR}"
# Parse grids for alpha/beta constants (allow space-separated lists)
read -r -a alpha_consts <<< "${ALPHA_CONST}"
read -r -a beta_consts <<< "${BETA_CONST}"

lora_rs=(${LORA_R})
lora_alphas=(${LORA_ALPHA})
lora_dropouts=(${LORA_DROPOUT})

num_train_epochs=${NUM_EPOCHS}

export CUDA_VISIBLE_DEVICES=${DEVICES}

for split in "${forget_retain_splits[@]}"; do
    forget_split=$(echo "$split" | cut -d' ' -f1)
    retain_split=$(echo "$split" | cut -d' ' -f2)

    for lr in "${lrs[@]}"; do
        for gamma in "${gammas[@]}"; do
            for lora_r in "${lora_rs[@]}"; do
                for lora_alpha in "${lora_alphas[@]}"; do
                    for lora_dropout in "${lora_dropouts[@]}"; do
                        for alpha_const in "${alpha_consts[@]}"; do
                            for beta_const in "${beta_consts[@]}"; do
                        dropout_tag=${lora_dropout//./p}
                        gamma_tag=${gamma//./p}

                        # Build tags for alpha/beta settings: a{val} / b{val} or adyn/bdyn when 'none'
                        shopt -s nocasematch || true
                        if [[ "${alpha_const}" == "none" ]]; then atag="adyn"; else atag="a${alpha_const//./p}"; fi
                        if [[ "${beta_const}" == "none" ]]; then btag="bdyn"; else btag="b${beta_const//./p}"; fi
                        shopt -u nocasematch || true
                        task_name=duet_${base_model}_${forget_split}_ada_WGD_lora_r${lora_r}_lalpha${lora_alpha}_ldrop${dropout_tag}_lr${lr}_${atag}_${btag}_gamma${gamma_tag}
                        run_dir=${repo_root}/saves/unlearn/ada_WGD/${task_name}
                        eval_dir=${run_dir}/evals
                        summary_path=${eval_dir}/DUET_SUMMARY.json

                        if [[ -f "${summary_path}" && "${FORCE_RERUN:-0}" != "1" ]]; then
                            echo "[ada_WGD] Skipping ${task_name}: found existing summary at ${summary_path}"
                            continue
                        fi

                        echo "${task_name}: LoRA AdaWGD unlearning ${base_model_path} on ${forget_split} (epochs=${num_train_epochs})"

                        # Minimal override set – controller/scaling use trainer defaults in code.

                        # Optional constant alpha/beta provided by user (set to a float to enable; 'none' keeps dynamic)
                        extra_method_args=()
                        shopt -s nocasematch || true
                        if [[ "${alpha_const:-none}" != "none" && -n "${alpha_const}" ]]; then
                            extra_method_args+=(trainer.method_args.alpha_const=${alpha_const})
                        fi
                        if [[ "${beta_const:-none}" != "none" && -n "${beta_const}" ]]; then
                            extra_method_args+=(trainer.method_args.beta_const=${beta_const})
                        fi
                        shopt -u nocasematch || true

                        adapter_path=${run_dir}/adapter_model.safetensors
                        log_file=${run_dir}/AdaWGD.log
                        if [[ ! -f "${adapter_path}" || "${FORCE_RERUN:-0}" == "1" ]]; then
                            mkdir -p "${run_dir}"
                            echo "[TRAIN] $(date) task=${task_name}" | tee -a "${log_file}"
                            python src/train.py --config-name=unlearn.yaml \
                                experiment=${experiment} \
                                trainer=${trainer} \
                                task_name=${task_name} \
                                model=${lora_model} \
                                forget_split=${forget_split} \
                                retain_split=${retain_split} \
                                model.model_args.pretrained_model_name_or_path=${base_model_path} \
                                model.model_args.device_map="auto" \
                                model.model_args.low_cpu_mem_usage=true \
                                model.lora_config.r=${lora_r} \
                                model.lora_config.lora_alpha=${lora_alpha} \
                                model.lora_config.lora_dropout=${lora_dropout} \
                                trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
                                trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
                                trainer.args.num_train_epochs=${num_train_epochs} \
                                trainer.args.learning_rate=${lr} \
                                trainer.method_args.gamma=${gamma} \
                                trainer.method_args.warmup_epochs=${WARMUP_EPOCHS} \
                                ${extra_method_args[@]} \
                                trainer.method_args.retain_loss_type=NLL \
                                retain_logs_path=null \
                                paths.output_dir=${run_dir} \
                                |& tee -a "${log_file}"
                        else
                            echo "[ada_WGD] Adapter already exists at ${adapter_path}; skipping training."
                        fi

                        mkdir -p "${eval_dir}"
                        if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
                            rm -f "${summary_path}" "${eval_dir}/DUET_EVAL.json"
                        fi

                        eval_cmd=( \
                            experiment=eval/duet/default.yaml \
                            model=${lora_model} \
                            forget_split=${forget_split} \
                            holdout_split=${retain_split} \
                            task_name=${task_name} \
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
                        echo "[EVAL] $(date) task=${task_name}" | tee -a "${log_file}"
                        python src/eval.py "${eval_cmd[@]}" |& tee -a "${log_file}"
                            done
                        done
                    done
                done
            done
        done
    done
done
