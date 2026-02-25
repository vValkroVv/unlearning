#!/bin/bash

set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")
repo_root=$(realpath "${script_dir}/../..")

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

base_model="${BASE_MODEL:-Llama-3.1-8B}"
lora_model="${MODEL_CONFIG:-${base_model}-lora}"
base_model_path="${HF_BASE_MODEL_PATH:-meta-llama/${base_model}}"

echo "[rwku][GD] Using Hugging Face base checkpoint ${base_model_path}"

experiment="unlearn/rwku/grad_diff_lora.yaml"
trainer="GradDiff"

output_root="${repo_root}/saves/unlearn/rwku/gd"
mkdir -p "${output_root}"

forget_split="forget_level2"
retain_split="neighbor_level2"

per_device_train_batch_size=${PER_DEVICE_TRAIN_BS:-1}
gradient_accumulation_steps=${GRAD_ACCUM:-32}
num_train_epochs=${NUM_EPOCHS:-5}

raw_lrs="${LRS:-1e-5 5e-5 1e-4 5e-4}"
raw_lrs="${raw_lrs//,/ }"
raw_lrs="${raw_lrs//\"/}"
raw_lrs="${raw_lrs//\'/}"
read -r -a lrs <<< "${raw_lrs}"

raw_alphas="${ALPHAS:-1.0}"
raw_alphas="${raw_alphas//,/ }"
raw_alphas="${raw_alphas//\"/}"
raw_alphas="${raw_alphas//\'/}"
read -r -a alphas <<< "${raw_alphas}"

raw_gammas="${GAMMAS:-1.0}"
raw_gammas="${raw_gammas//,/ }"
raw_gammas="${raw_gammas//\"/}"
raw_gammas="${raw_gammas//\'/}"
read -r -a gammas <<< "${raw_gammas}"

lora_rs=(${LORA_RS:-"32"})
lora_alphas=(${LORA_ALPHAS:-"64"})
lora_dropouts=(${LORA_DROPOUTS:-"0.0"})

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for lr in "${lrs[@]}"; do
    for alpha in "${alphas[@]}"; do
        alpha_tag=${alpha//./p}
        for gamma in "${gammas[@]}"; do
            gamma_tag=${gamma//./p}
            for lora_r in "${lora_rs[@]}"; do
                for lora_alpha in "${lora_alphas[@]}"; do
                    for lora_dropout in "${lora_dropouts[@]}"; do
                        dropout_tag=${lora_dropout//./p}
                        task_name=rwku_${base_model}_${forget_split}_gd_lora_r${lora_r}_lalpha${lora_alpha}_ldrop${dropout_tag}_lr${lr}_alpha${alpha_tag}_gamma${gamma_tag}
                        run_dir=${output_root}/${task_name}
                        eval_dir=${run_dir}/evals
                        summary_path=${eval_dir}/DUET_SUMMARY.json

                        if [[ -f "${summary_path}" && "${FORCE_RERUN:-0}" != "1" ]]; then
                            echo "[rwku][GD] Skipping ${task_name}: found existing summary at ${summary_path}"
                            continue
                        fi

                        echo "${task_name}: GradDiff LoRA unlearning ${base_model_path} on ${forget_split}"

                        adapter_path=${run_dir}/adapter_model.safetensors
                        if [[ ! -f "${adapter_path}" || "${FORCE_RERUN:-0}" == "1" ]]; then
                            mkdir -p "${run_dir}"
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
                                trainer.method_args.alpha=${alpha} \
                                trainer.method_args.gamma=${gamma} \
                                trainer.method_args.retain_loss_type=NLL \
                                retain_logs_path=null \
                                paths.output_dir=${run_dir}
                        fi

                        mkdir -p "${eval_dir}"
                        if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
                            rm -f "${summary_path}" "${eval_dir}/DUET_EVAL.json"
                        fi

                        eval_cmd=( \
                            experiment=eval/rwku/default.yaml \
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
