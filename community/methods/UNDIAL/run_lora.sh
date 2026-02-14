#!/bin/bash

set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")
repo_root=$(realpath "${script_dir}/../../..")

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

base_model="Llama-3.1-8B-Instruct"
lora_model="${base_model}-lora"
base_model_path="meta-llama/${base_model}"

trainers_experiments=("UNDIAL unlearn/muse/undial_lora.yaml")
forget_retain_splits=("forget retain1")

# LoRA-friendly defaults (effective batch size 32)
per_device_train_batch_size=1
gradient_accumulation_steps=32

lrs=(1e-4)
alphas=(0.1)
betas=(3)

# LoRA hyper-parameters to sweep
lora_rs=(64)
lora_alphas=(128)
lora_dropouts=(0.05)

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for split in "${forget_retain_splits[@]}"; do
    forget_split=$(echo "$split" | cut -d' ' -f1)
    retain_split=$(echo "$split" | cut -d' ' -f2)

    for trainer_experiment in "${trainers_experiments[@]}"; do
        trainer=$(echo "$trainer_experiment" | cut -d' ' -f1)
        experiment=$(echo "$trainer_experiment" | cut -d' ' -f2)

        for lr in "${lrs[@]}"; do
            for beta in "${betas[@]}"; do
                for alpha in "${alphas[@]}"; do
                    for lora_r in "${lora_rs[@]}"; do
                        for lora_alpha in "${lora_alphas[@]}"; do
                            for lora_dropout in "${lora_dropouts[@]}"; do
                                dropout_tag=${lora_dropout//./p}

                                task_name=muse_${base_model}_${forget_split}_${trainer}_lora_r${lora_r}_alpha${lora_alpha}_drop${dropout_tag}_lr${lr}_beta${beta}

                                echo "${task_name}: LoRA unlearning ${base_model_path} using ${trainer}"

                                retain_logs_path=${repo_root}/saves/eval/muse_${base_model}_${retain_split}/MUSE_EVAL.json
                                retain_logs_dir=$(dirname "${retain_logs_path}")
                                if [[ ! -f "${retain_logs_path}" ]]; then
                                    echo "Retain logs missing at ${retain_logs_path}; generating with baseline model."
                                    mkdir -p "${retain_logs_dir}"
                                    python src/eval.py --config-name=eval.yaml \
                                        experiment=eval/muse/default.yaml \
                                        model=${base_model} \
                                        task_name=muse_${base_model}_${retain_split}_retain_eval \
                                        model.model_args.pretrained_model_name_or_path=${base_model_path} \
                                        paths.output_dir=${retain_logs_dir} \
                                        retain_logs_path=null
                                fi

                                # LoRA unlearning
                                python src/train.py --config-name=unlearn.yaml \
                                    experiment=${experiment} \
                                    trainer=${trainer} \
                                    task_name=${task_name} \
                                    model=${lora_model} \
                                    forget_split=${forget_split} \
                                    retain_split=${retain_split} \
                                    model.model_args.pretrained_model_name_or_path=${base_model_path} \
                                    model.lora_config.r=${lora_r} \
                                    model.lora_config.lora_alpha=${lora_alpha} \
                                    model.lora_config.lora_dropout=${lora_dropout} \
                                    trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
                                    trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
                                    trainer.args.learning_rate=${lr} \
                                    trainer.method_args.beta=${beta} \
                                    trainer.method_args.alpha=${alpha} \
                                    retain_logs_path=${retain_logs_path} \
                                    paths.output_dir=${repo_root}/saves/unlearn/${task_name}

                                # Evaluation on the unlearned LoRA adapter
                                python src/eval.py \
                                    experiment=eval/muse/default.yaml \
                                    model=${lora_model} \
                                    task_name=${task_name} \
                                    model.model_args.pretrained_model_name_or_path=${repo_root}/saves/unlearn/${task_name} \
                                    model.lora_config.r=${lora_r} \
                                    model.lora_config.lora_alpha=${lora_alpha} \
                                    model.lora_config.lora_dropout=${lora_dropout} \
                                    paths.output_dir=${repo_root}/saves/unlearn/${task_name}/evals \
                                    retain_logs_path=${retain_logs_path}
                            done
                        done
                    done
                done
            done
        done
    done
done
