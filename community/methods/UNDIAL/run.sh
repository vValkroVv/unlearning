#!/bin/bash

set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")
repo_root=$(realpath "${script_dir}/../../..")

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

########################################################################################################################
########################################### Unlearn TOFU models ########################################################
########################################################################################################################

models=(
    "Llama-3.2-1B-Instruct"
)
trainers_experiments=(
    "UNDIAL unlearn/muse/default.yaml"
)
forget_retain_splits=(
    "forget retain1"
)

per_device_train_batch_size=1
gradient_accumulation_steps=2
# Default number of unlearning epochs (override with NUM_EPOCHS)
num_train_epochs=${NUM_EPOCHS:-5}


lrs=(1e-5)
alphas=(1)
betas=(3)
export CUDA_VISIBLE_DEVICES=1

for split in "${forget_retain_splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    retain_split=$(echo $split | cut -d' ' -f2)
    for model in "${models[@]}"; do
        for trainer_experiment in "${trainers_experiments[@]}"; do
            trainer=$(echo $trainer_experiment | cut -d' ' -f1)
            experiment=$(echo $trainer_experiment | cut -d' ' -f2)
            for lr in "${lrs[@]}"; do
                for beta in "${betas[@]}"; do 
                    for alpha in "${alphas[@]}"; do          
                        task_name=muse_${model}_${forget_split}_${trainer}_lr${lr}_beta${beta}_alpha${alpha}
                        model_path=meta-llama/Llama-3.2-1B-Instruct #open-unlearning/muse_${model}
                        echo ${task_name}: Unlearning ${model_path} using ${trainer}

                        retain_logs_path=${repo_root}/saves/eval/muse_${model}_${retain_split}/MUSE_EVAL.json
                        retain_logs_dir=$(dirname "${retain_logs_path}")
                        if [[ ! -f "${retain_logs_path}" ]]; then
                            echo "Retain logs missing at ${retain_logs_path}; generating with baseline model."
                            mkdir -p "${retain_logs_dir}"
                            python src/eval.py --config-name=eval.yaml \
                                experiment=eval/muse/default.yaml \
                                model=${model} \
                                task_name=muse_${model}_${retain_split}_retain_eval \
                                model.model_args.pretrained_model_name_or_path=${model_path} \
                                paths.output_dir=${retain_logs_dir} \
                                retain_logs_path=null
                        fi

                        # Unlearn
                        python src/train.py --config-name=unlearn.yaml \
                        experiment=${experiment} \
                        trainer=${trainer} \
                        task_name=${task_name} \
                        model=${model} \
                        forget_split=${forget_split} \
                        retain_split=${retain_split} \
                        model.model_args.pretrained_model_name_or_path=${model_path} \
                        retain_logs_path=${retain_logs_path} \
                        trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
                        trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
                        trainer.args.num_train_epochs=${num_train_epochs} \
                        trainer.args.eval_strategy=no \
                        trainer.args.eval_on_start=False \
                        trainer.args.learning_rate=$lr \
                        trainer.method_args.beta=$beta \
                        trainer.method_args.alpha=$alpha \
                        paths.output_dir=${repo_root}/saves/unlearn/${task_name}

                        # Evald
                        python src/eval.py \
                        experiment=eval/muse/default.yaml \
                        model=${model} \
                        task_name=${task_name} \
                        model.model_args.pretrained_model_name_or_path=${repo_root}/saves/unlearn/${task_name} \
                        paths.output_dir=${repo_root}/saves/unlearn/${task_name}/evals \
                        retain_logs_path=${retain_logs_path}
                    done
                done
            done
        done
    done
done
