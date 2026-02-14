#!/bin/bash

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

########################################################################################################################
########################################### Unlearn DUET models ########################################################
########################################################################################################################

models=(
    "Llama-3.1-8B-Instruct-lora"
)
trainers_experiments=(
    "UNDIAL unlearn/duet/undial_lora.yaml"
)
forget_retain_splits=(
    "city_forget_rare_5 city_fast_retain_500"
    "city_forget_popular_5 city_fast_retain_500"
)

per_device_train_batch_size=16
gradient_accumulation_steps=2


lrs=(1e-5 1e-4)
alphas=(3)
betas=(3)


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
                        task_name=duet_${model}_${forget_split}_${trainer}_lr${lr}_beta${beta}_alpha${alpha}
                        model_path=open-unlearning/duet_${model}_full
                        echo ${task_name}: Unlearning ${model_path} using ${trainer}

                        # Unlearn
                        CUDA_VISIBLE_DEVICES=0 \
                        python src/train.py --config-name=unlearn.yaml \
                        experiment=${experiment} \
                        trainer=${trainer} \
                        task_name=${task_name} \
                        model=${model} \
                        forget_split=${forget_split} \
                        retain_split=${retain_split} \
                        model.model_args.pretrained_model_name_or_path=/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/llama3.1-8b_full_3ep_ft_tripunlamb \
                        retain_logs_path=null \
                        trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
                        trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
                        trainer.args.eval_strategy=no \
                        trainer.args.eval_on_start=False \
                        trainer.args.learning_rate=$lr \
                        trainer.method_args.beta=$beta \
                        trainer.method_args.alpha=$alpha

                        # Eval
                        CUDA_VISIBLE_DEVICES=0 python src/eval.py \
                        experiment=eval/duet/default.yaml \
                        forget_split=${forget_split} \
                        holdout_split=${retain_split} \
                        model=${model} \
                        task_name=${task_name} \
                        model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
                        +model.model_args.base_model_name_or_path=/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/llama3.1-8b_full_3ep_ft_tripunlamb \
                        model.lora_config.r=32 \
                        model.lora_config.lora_alpha=64 \
                        model.lora_config.lora_dropout=0.0 \
                        paths.output_dir=saves/unlearn/${task_name}/evals \
                        retain_logs_path=null
                    done
                done
            done
        done
    done
done
