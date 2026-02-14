#!/bin/bash

set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")
repo_root=$(realpath "${script_dir}/../../..")

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# Base model selection
base_model="Llama-3.1-8B-Instruct"
lora_model="${base_model}-lora"
hf_base_model_path="meta-llama/${base_model}"
local_sft_base="/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/llama3.1-8b_full_3ep_ft_tripunlamb"

# 1 - local SFT checkpoint. 0 - public HF base.
use_sft_base=1

if [[ "${use_sft_base}" == "1" ]]; then
    base_model_path="${local_sft_base}"
    echo "[pop_static_wga] Using locally finetuned base checkpoint at ${base_model_path}"
else
    base_model_path="${hf_base_model_path}"
    echo "[pop_static_wga] Using Hugging Face base checkpoint ${base_model_path}"
fi

# Trainer and experiment preset
trainer="WGA"
experiment="unlearn/duet/wga_lora.yaml"

# Forget/Retrain splits to run
forget_retain_splits=(
    "city_forget_rare_5 city_fast_retain_500"
    "city_forget_popular_5 city_fast_retain_500"
)

# Core hyperparameters
per_device_train_batch_size=1
gradient_accumulation_steps=32
lrs=(1e-5 2e-5 4e-5 6e-5 8e-5 1e-4 2e-4 3e-4 5e-4) #  8e-5 1e-4;; 1e-5 2e-5 4e-5  ;; 6e-5 8e-5 1e-4
betas=(3 ) # 2.0 1.0 0.5 0.1 1.0 0.5 0.1 0.01
gammas=(1.0)
# Default number of unlearning epochs (override with NUM_EPOCHS)
num_train_epochs=${NUM_EPOCHS:-10}

# LoRA hyperparameters
lora_rs=(32)
lora_alphas=(64)
lora_dropouts=(0.0)

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

for split in "${forget_retain_splits[@]}"; do
    forget_split=$(echo "$split" | cut -d' ' -f1)
    retain_split=$(echo "$split" | cut -d' ' -f2)

    # Static alpha selection based on forget split popularity
    case "${forget_split}" in
        city_forget_popular_5)
            alpha=0.8
            ;;
        city_forget_rare_5)
            alpha=0.8
            ;;
        *)
            echo "[pop_static_wga] Unknown forget split '${forget_split}', defaulting like in GD alpha=0.5"
            alpha=0.5
            ;;
    esac

    for lr in "${lrs[@]}"; do
        for beta in "${betas[@]}"; do
            for gamma in "${gammas[@]}"; do
                for lora_r in "${lora_rs[@]}"; do
                    for lora_alpha in "${lora_alphas[@]}"; do
                        for lora_dropout in "${lora_dropouts[@]}"; do
                            dropout_tag=${lora_dropout//./p}
                            alpha_tag=${alpha//./p}
                            gamma_tag=${gamma//./p}

                            task_name=duet_${base_model}_${forget_split}_pop_static_wga_lora_r${lora_r}_lalpha${lora_alpha}_ldrop${dropout_tag}_lr${lr}_beta${beta}_alpha${alpha_tag}_gamma${gamma_tag}
                            run_dir=${repo_root}/saves/unlearn/pop_static_wga/${task_name}
                            eval_dir=${run_dir}/evals
                            summary_path=${eval_dir}/DUET_SUMMARY.json

                            if [[ -f "${summary_path}" && "${FORCE_RERUN:-0}" != "1" ]]; then
                                echo "[pop_static_wga] Skipping ${task_name}: found existing summary at ${summary_path}"
                                continue
                            fi

                            echo "${task_name}: LoRA pop_static_wga unlearning ${base_model_path} on ${forget_split} (alpha=${alpha})"

                            adapter_path=${run_dir}/adapter_model.safetensors
                            if [[ ! -f "${adapter_path}" || "${FORCE_RERUN:-0}" == "1" ]]; then
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
                                    trainer.method_args.beta=${beta} \
                                    trainer.method_args.alpha=${alpha} \
                                    trainer.method_args.gamma=${gamma} \
                                    retain_logs_path=null \
                                    paths.output_dir=${run_dir}
                            else
                                echo "[pop_static_wga] Adapter already exists at ${adapter_path}; skipping training."
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
                                eval.duet.overwrite=true \
                                paths.output_dir=${eval_dir} \
                                retain_logs_path=null \
                            )
                            python src/eval.py "${eval_cmd[@]}"
                        done
                    done
                done
            done
        done
    done
done
