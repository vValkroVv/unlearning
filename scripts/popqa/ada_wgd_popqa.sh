#!/bin/bash

set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")
repo_root=$(realpath "${script_dir}/../..")

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

base_model="${BASE_MODEL:-Llama-3.1-8B-Instruct}"
lora_model="${MODEL_CONFIG:-${base_model}-lora}"
hf_base_model_path="${HF_BASE_MODEL_PATH:-meta-llama/${base_model}}"
local_sft_base="${LOCAL_SFT_BASE:-/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/popqa/llama3.1-8b_full_5ep_ft_popqa}"

use_sft_base=${USE_SFT_BASE:-1}
if [[ "${use_sft_base}" == "1" ]]; then
    base_model_path="${local_sft_base}"
    echo "[popqa][ada_WGD] Using locally finetuned base checkpoint at ${base_model_path}"
else
    base_model_path="${hf_base_model_path}"
    echo "[popqa][ada_WGD] Using Hugging Face base checkpoint ${base_model_path}"
fi

experiment="unlearn/popqa/wga_lora.yaml"
trainer="AdaWGD"

output_root="${repo_root}/saves/unlearn/popqa/ada_WGD"
mkdir -p "${output_root}"

forget_retain_splits=(
    "rare_forget5_sum fast_retain_500"
    "popular_forget5_sum fast_retain_500"
)

per_device_train_batch_size=${PER_DEVICE_TRAIN_BS:-1}
gradient_accumulation_steps=${GRAD_ACCUM:-32}
num_train_epochs=${NUM_EPOCHS:-5}

raw_lrs="${LRS:-5e-5 1e-4 5e-4 1e-3 5e-3 1e-2}"
raw_lrs="${raw_lrs//,/ }"
raw_lrs="${raw_lrs//\"/}"
raw_lrs="${raw_lrs//\'/}"
read -r -a lrs <<< "${raw_lrs}"

raw_gammas="${GAMMAS:-1.0}"
raw_gammas="${raw_gammas//,/ }"
raw_gammas="${raw_gammas//\"/}"
raw_gammas="${raw_gammas//\'/}"
read -r -a gammas <<< "${raw_gammas}"

lora_rs=(${LORA_RS:-"32"})
lora_alphas=(${LORA_ALPHAS:-"64"})
lora_dropouts=(${LORA_DROPOUTS:-"0.0"})

raw_alpha_consts="${ALPHA_CONST:-none}"
raw_alpha_consts="${raw_alpha_consts//,/ }"
raw_alpha_consts="${raw_alpha_consts//\"/}"
raw_alpha_consts="${raw_alpha_consts//\'/}"
read -r -a alpha_consts <<< "${raw_alpha_consts}"

raw_beta_consts="${BETA_CONST:-none}"
raw_beta_consts="${raw_beta_consts//,/ }"
raw_beta_consts="${raw_beta_consts//\"/}"
raw_beta_consts="${raw_beta_consts//\'/}"
read -r -a beta_consts <<< "${raw_beta_consts}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for split in "${forget_retain_splits[@]}"; do
    forget_split=$(echo "$split" | cut -d' ' -f1)
    retain_split=$(echo "$split" | cut -d' ' -f2)

    for lr in "${lrs[@]}"; do
        for gamma in "${gammas[@]}"; do
            gamma_tag=${gamma//./p}
            for alpha_const in "${alpha_consts[@]}"; do
                for beta_const in "${beta_consts[@]}"; do
                    atag="adyn"
                    btag="bdyn"
                    extra_method_args=()
                    shopt -s nocasematch || true
                    if [[ "${alpha_const}" != "none" && -n "${alpha_const}" ]]; then
                        atag="a${alpha_const//./p}"
                        extra_method_args+=(trainer.method_args.alpha_const=${alpha_const})
                    fi
                    if [[ "${beta_const}" != "none" && -n "${beta_const}" ]]; then
                        btag="b${beta_const//./p}"
                        extra_method_args+=(trainer.method_args.beta_const=${beta_const})
                    fi
                    shopt -u nocasematch || true

                    for lora_r in "${lora_rs[@]}"; do
                        for lora_alpha in "${lora_alphas[@]}"; do
                            for lora_dropout in "${lora_dropouts[@]}"; do
                                dropout_tag=${lora_dropout//./p}
                                task_name=popqa_${base_model}_${forget_split}_ada_WGD_lora_r${lora_r}_lalpha${lora_alpha}_ldrop${dropout_tag}_lr${lr}_${atag}_${btag}_gamma${gamma_tag}
                                run_dir=${output_root}/${task_name}
                                eval_dir=${run_dir}/evals
                                summary_path=${eval_dir}/POPQA_SUMMARY.json

                                if [[ -f "${summary_path}" && "${FORCE_RERUN:-0}" != "1" ]]; then
                                    echo "[popqa][ada_WGD] Skipping ${task_name}: found existing summary at ${summary_path}"
                                    continue
                                fi

                                echo "${task_name}: AdaWGD LoRA unlearning ${base_model_path} on ${forget_split}"

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
                                        trainer.method_args.retain_loss_type=NLL \
                                        retain_logs_path=null \
                                        paths.output_dir=${run_dir} \
                                        "${extra_method_args[@]}" \
                                        |& tee -a "${log_file}"
                                fi

                                mkdir -p "${eval_dir}"
                                if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
                                    rm -f "${summary_path}" "${eval_dir}/POPQA_EVAL.json"
                                fi

                                eval_cmd=( \
                                    experiment=eval/popqa/default.yaml \
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
