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
sft_subfolder="${SFT_SUBFOLDER:-}"

use_sft_base=${USE_SFT_BASE:-1}
chat_template_tokenizer_path="${CHAT_TEMPLATE_TOKENIZER_PATH:-${repo_root}/assets/tokenizers/llama-3.1-8b-instruct-chat-template}"
if [[ "${use_sft_base}" == "1" ]]; then
    base_model_path="${local_sft_base}"
    default_tokenizer_model_path="${base_model_path}"
    default_tokenizer_subfolder="${sft_subfolder}"
    if [[ "${sft_subfolder}" == "llama-3.1-8b-instruct-popqa-ft" ]]; then
        default_tokenizer_model_path="${chat_template_tokenizer_path}"
        default_tokenizer_subfolder=""
    fi
    echo "[popqa][NPOSAM] Using locally finetuned base checkpoint at ${base_model_path}"
else
    base_model_path="${hf_base_model_path}"
    default_tokenizer_model_path="${hf_base_model_path}"
    default_tokenizer_subfolder=""
    echo "[popqa][NPOSAM] Using Hugging Face base checkpoint ${base_model_path}"
fi
tokenizer_model_path="${TOKENIZER_MODEL_PATH:-${default_tokenizer_model_path}}"
tokenizer_subfolder="${TOKENIZER_SUBFOLDER-${default_tokenizer_subfolder}}"
extra_train_args=()
extra_eval_args=()
if [[ "${use_sft_base}" == "1" && -n "${sft_subfolder}" ]]; then
    extra_train_args+=(+model.model_args.subfolder=${sft_subfolder})
    extra_eval_args+=(+model.model_args.subfolder=${sft_subfolder})
fi
if [[ "${use_sft_base}" == "1" && -n "${tokenizer_subfolder}" ]]; then
    extra_train_args+=(+model.tokenizer_args.subfolder=${tokenizer_subfolder})
    extra_eval_args+=(+model.tokenizer_args.subfolder=${tokenizer_subfolder})
fi

experiment="unlearn/popqa/npo_sam_lora.yaml"
trainer="NPOSAM"

output_root="${repo_root}/saves/unlearn/popqa/npo_sam"
mkdir -p "${output_root}"

base_forget_retain_splits=(
    "rare_forget5_sum fast_retain_500"
    "popular_forget5_sum fast_retain_500"
)

if [[ "${MERGE_POPULARITY_FORGET:-0}" == "1" ]]; then
    forget_retain_splits=(
        "rare_forget5_sum+popular_forget5_sum fast_retain_500 forget5_sum"
    )
else
    forget_retain_splits=("${base_forget_retain_splits[@]}")
fi

per_device_train_batch_size=${PER_DEVICE_TRAIN_BS:-1}
gradient_accumulation_steps=${GRAD_ACCUM:-32}
eval_batch_size=${EVAL_BATCH_SIZE:-8}
num_train_epochs=${NUM_EPOCHS:-5}
gradient_checkpointing=${GRADIENT_CHECKPOINTING:-false}

raw_lrs="${LRS:-1e-5}"
raw_lrs="${raw_lrs//,/ }"
raw_lrs="${raw_lrs//\"/}"
raw_lrs="${raw_lrs//\'/}"
read -r -a lrs <<< "${raw_lrs}"

raw_betas="${BETAS:-0.01, 0.05 0.15}"
raw_betas="${raw_betas//,/ }"
raw_betas="${raw_betas//\"/}"
raw_betas="${raw_betas//\'/}"
read -r -a betas <<< "${raw_betas}"

raw_alphas="${ALPHAS:-0.25 1.0 2.0}"
raw_alphas="${raw_alphas//,/ }"
raw_alphas="${raw_alphas//\"/}"
raw_alphas="${raw_alphas//\'/}"
read -r -a alphas <<< "${raw_alphas}"

raw_gammas="${GAMMAS:-1.0}"
raw_gammas="${raw_gammas//,/ }"
raw_gammas="${raw_gammas//\"/}"
raw_gammas="${raw_gammas//\'/}"
read -r -a gammas <<< "${raw_gammas}"

raw_sam_rhos="${SAM_RHOS:-0.003 0.01 0.03}"
raw_sam_rhos="${raw_sam_rhos//,/ }"
raw_sam_rhos="${raw_sam_rhos//\"/}"
raw_sam_rhos="${raw_sam_rhos//\'/}"
read -r -a sam_rhos <<< "${raw_sam_rhos}"

raw_sam_adaptives="${SAM_ADAPTIVES:-false}"
raw_sam_adaptives="${raw_sam_adaptives//,/ }"
raw_sam_adaptives="${raw_sam_adaptives//\"/}"
raw_sam_adaptives="${raw_sam_adaptives//\'/}"
read -r -a sam_adaptives <<< "${raw_sam_adaptives}"

sam_eps="${SAM_EPS:-1e-12}"

lora_rs=(${LORA_RS:-"32"})
lora_alphas=(${LORA_ALPHAS:-"64"})
lora_dropouts=(${LORA_DROPOUTS:-"0.0"})
delete_model_safetensors_after_eval="${DELETE_MODEL_SAFETENSORS_AFTER_EVAL:-0}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for split in "${forget_retain_splits[@]}"; do
    read -r forget_split retain_split forget_label <<< "${split}"
    if [[ -z "${forget_label:-}" ]]; then
        forget_label="${forget_split}"
    fi

    for lr in "${lrs[@]}"; do
        for beta in "${betas[@]}"; do
            beta_tag=${beta//./p}
            for alpha in "${alphas[@]}"; do
                alpha_tag=${alpha//./p}
                for gamma in "${gammas[@]}"; do
                    gamma_tag=${gamma//./p}
                    for sam_rho in "${sam_rhos[@]}"; do
                        rho_tag=${sam_rho//./p}
                        for sam_adaptive in "${sam_adaptives[@]}"; do
                            adapt_tag="${sam_adaptive}"
                            adapt_tag="${adapt_tag//true/T}"
                            adapt_tag="${adapt_tag//false/F}"
                            for lora_r in "${lora_rs[@]}"; do
                                for lora_alpha in "${lora_alphas[@]}"; do
                                    for lora_dropout in "${lora_dropouts[@]}"; do
                                        dropout_tag=${lora_dropout//./p}
                                        task_name=popqa_${base_model}_${forget_label}_npo_sam_lora_r${lora_r}_lalpha${lora_alpha}_ldrop${dropout_tag}_lr${lr}_beta${beta_tag}_alpha${alpha_tag}_gamma${gamma_tag}_rho${rho_tag}_ad${adapt_tag}
                                        run_dir=${output_root}/${task_name}
                                        eval_dir=${run_dir}/evals
                                        summary_path=${eval_dir}/POPQA_SUMMARY.json

                                        if [[ -f "${summary_path}" && "${FORCE_RERUN:-0}" != "1" ]]; then
                                            echo "[popqa][NPOSAM] Skipping ${task_name}: found existing summary at ${summary_path}"
                                            continue
                                        fi

                                        echo "[popqa][NPOSAM] ${task_name}: unlearning ${base_model_path} on ${forget_split}"

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
                                                model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
                                                model.model_args.device_map="auto" \
                                                ++model.model_args.low_cpu_mem_usage=true \
                                                model.lora_config.r=${lora_r} \
                                                model.lora_config.lora_alpha=${lora_alpha} \
                                                model.lora_config.lora_dropout=${lora_dropout} \
                                                trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
                                                trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
                                                trainer.args.num_train_epochs=${num_train_epochs} \
                                                trainer.args.gradient_checkpointing=${gradient_checkpointing} \
                                                trainer.args.learning_rate=${lr} \
                                                trainer.method_args.beta=${beta} \
                                                trainer.method_args.alpha=${alpha} \
                                                trainer.method_args.gamma=${gamma} \
                                                trainer.method_args.retain_loss_type=NLL \
                                                trainer.method_args.sam_rho=${sam_rho} \
                                                trainer.method_args.sam_adaptive=${sam_adaptive} \
                                                trainer.method_args.sam_eps=${sam_eps} \
                                                retain_logs_path=null \
                                                "${extra_train_args[@]}" \
                                                paths.output_dir=${run_dir}
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
                                            ++model.model_args.base_model_name_or_path=${base_model_path} \
                                            model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
                                            model.model_args.device_map="auto" \
                                            ++model.model_args.low_cpu_mem_usage=true \
                                            model.lora_config.r=${lora_r} \
                                            model.lora_config.lora_alpha=${lora_alpha} \
                                            model.lora_config.lora_dropout=${lora_dropout} \
                                            eval.duet.batch_size=${eval_batch_size} \
                                            eval.duet.overwrite=true \
                                            "${extra_eval_args[@]}" \
                                            paths.output_dir=${eval_dir} \
                                            retain_logs_path=null \
                                        )
                                        python src/eval.py "${eval_cmd[@]}"

                                        if [[ "${delete_model_safetensors_after_eval}" == "1" ]]; then
                                            if compgen -G "${run_dir}/*.safetensors" > /dev/null; then
                                                rm -f "${run_dir}"/*.safetensors
                                                echo "[popqa][NPOSAM] Removed safetensors from ${run_dir}"
                                            fi
                                        fi
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
