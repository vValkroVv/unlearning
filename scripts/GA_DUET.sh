#!/bin/bash

set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")
repo_root=$(realpath "${script_dir}/..")

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

base_model="Llama-3.1-8B-Instruct"
lora_model="${base_model}-lora"
hf_base_model_path="meta-llama/${base_model}"
local_sft_base="/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/llama3.1-8b_full_3ep_ft_tripunlamb"

use_sft_base=${USE_SFT_BASE:-1}
if [[ "${use_sft_base}" == "1" ]]; then
    base_model_path="${local_sft_base}"
    base_type="sft"
    echo "[GA] Using locally finetuned base checkpoint at ${base_model_path}"
else
    base_model_path="${hf_base_model_path}"
    base_type="pretrain"
    echo "[GA] Using Hugging Face base checkpoint ${base_model_path}"
fi

output_root="${repo_root}/saves/unlearn/GA_${base_type}"
mkdir -p "${output_root}"

trainers_experiments=("GradAscent unlearn/duet/grad_ascent_lora.yaml")
forget_retain_splits=(
    "city_forget_rare_5 city_fast_retain_500"
    "city_forget_popular_5 city_fast_retain_500"
)

per_device_train_batch_size=1
gradient_accumulation_steps=32
lrs=(7e-5 1e-4 3e-4) # 1e-6 5e-6 1e-5 2e-5 4e-5 5e-5
lora_rs=(32)
lora_alphas=(64)
lora_dropouts=(0.0)

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

for split in "${forget_retain_splits[@]}"; do
    forget_split=$(echo "$split" | cut -d' ' -f1)
    retain_split=$(echo "$split" | cut -d' ' -f2)

    for trainer_experiment in "${trainers_experiments[@]}"; do
        trainer=$(echo "$trainer_experiment" | cut -d' ' -f1)
        experiment=$(echo "$trainer_experiment" | cut -d' ' -f2)

        for lr in "${lrs[@]}"; do
            for lora_r in "${lora_rs[@]}"; do
                for lora_alpha in "${lora_alphas[@]}"; do
                    for lora_dropout in "${lora_dropouts[@]}"; do
                        dropout_tag=${lora_dropout//./p}

                        task_name=duet_${base_model}_${forget_split}_${trainer}_lora_r${lora_r}_lalpha${lora_alpha}_ldrop${dropout_tag}_lr${lr}
                        run_dir=${output_root}/${task_name}
                        eval_dir=${run_dir}/evals
                        summary_path=${eval_dir}/DUET_SUMMARY.json

                        if [[ -f "${summary_path}" && "${FORCE_RERUN:-0}" != "1" ]]; then
                            echo "Skipping ${task_name}: found existing summary at ${summary_path}"
                            continue
                        fi

                        echo "${task_name}: GradAscent LoRA unlearning ${base_model_path} on ${forget_split}"

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
                                trainer.args.learning_rate=${lr} \
                                retain_logs_path=null \
                                paths.output_dir=${run_dir}
                        else
                            echo "[GA] Adapter already exists at ${adapter_path}; skipping training."
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
