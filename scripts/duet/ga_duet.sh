#!/bin/bash

set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")
repo_root=$(realpath "${script_dir}/../..")
source "${script_dir}/_splits.sh"

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

base_model="${BASE_MODEL:-Llama-3.1-8B-Instruct}"
lora_model="${MODEL_CONFIG:-${base_model}-lora}"
hf_base_model_path="${HF_BASE_MODEL_PATH:-meta-llama/${base_model}}"
local_sft_base="${LOCAL_SFT_BASE:-/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/llama3.1-8b_full_3ep_ft_tripunlamb}"
sft_subfolder="${SFT_SUBFOLDER:-}"

use_sft_base=${USE_SFT_BASE:-1}
if [[ "${use_sft_base}" == "1" ]]; then
    base_model_path="${local_sft_base}"
    default_tokenizer_model_path="${base_model_path}"
    echo "[duet][GA] Using locally finetuned base checkpoint at ${base_model_path}"
else
    base_model_path="${hf_base_model_path}"
    default_tokenizer_model_path="${hf_base_model_path}"
    echo "[duet][GA] Using Hugging Face base checkpoint ${base_model_path}"
fi
tokenizer_model_path="${TOKENIZER_MODEL_PATH:-${default_tokenizer_model_path}}"
tokenizer_subfolder="${TOKENIZER_SUBFOLDER-${sft_subfolder}}"
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

experiment="unlearn/duet/grad_ascent_lora.yaml"
trainer="GradAscent"

output_root="${repo_root}/saves/unlearn/duet/ga"
mkdir -p "${output_root}"

set_forget_retain_splits

per_device_train_batch_size=${PER_DEVICE_TRAIN_BS:-1}
gradient_accumulation_steps=${GRAD_ACCUM:-32}
num_train_epochs=${NUM_EPOCHS:-5}

raw_lrs="${LRS:-1e-5 5e-5 1e-4 5e-4 1e-3}"
raw_lrs="${raw_lrs//,/ }"
raw_lrs="${raw_lrs//\"/}"
raw_lrs="${raw_lrs//\'/}"
read -r -a lrs <<< "${raw_lrs}"

lora_rs=(${LORA_RS:-"32"})
lora_alphas=(${LORA_ALPHAS:-"64"})
lora_dropouts=(${LORA_DROPOUTS:-"0.0"})

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for split in "${forget_retain_splits[@]}"; do
    read -r forget_split retain_split forget_label <<< "${split}"
    if [[ -z "${forget_label:-}" ]]; then
        forget_label="${forget_split}"
    fi

    for lr in "${lrs[@]}"; do
        for lora_r in "${lora_rs[@]}"; do
            for lora_alpha in "${lora_alphas[@]}"; do
                for lora_dropout in "${lora_dropouts[@]}"; do
                    dropout_tag=${lora_dropout//./p}
                    task_name=duet_${base_model}_${forget_label}_ga_lora_r${lora_r}_lalpha${lora_alpha}_ldrop${dropout_tag}_lr${lr}
                    run_dir=${output_root}/${task_name}
                    eval_dir=${run_dir}/evals
                    summary_path=${eval_dir}/DUET_SUMMARY.json

                    if [[ -f "${summary_path}" && "${FORCE_RERUN:-0}" != "1" ]]; then
                        echo "[duet][GA] Skipping ${task_name}: found existing summary at ${summary_path}"
                        continue
                    fi

                    echo "${task_name}: GradAscent LoRA unlearning ${base_model_path} on ${forget_split}"

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
                            model.model_args.low_cpu_mem_usage=true \
                            model.lora_config.r=${lora_r} \
                            model.lora_config.lora_alpha=${lora_alpha} \
                            model.lora_config.lora_dropout=${lora_dropout} \
                            trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
                            trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
                            trainer.args.num_train_epochs=${num_train_epochs} \
                            trainer.args.learning_rate=${lr} \
                            retain_logs_path=null \
                            "${extra_train_args[@]}" \
                            paths.output_dir=${run_dir}
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
                        model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
                        model.model_args.device_map="auto" \
                        model.model_args.low_cpu_mem_usage=true \
                        model.lora_config.r=${lora_r} \
                        model.lora_config.lora_alpha=${lora_alpha} \
                        model.lora_config.lora_dropout=${lora_dropout} \
                        eval.duet.overwrite=true \
                        "${extra_eval_args[@]}" \
                        paths.output_dir=${eval_dir} \
                        retain_logs_path=null \
                    )
                    python src/eval.py "${eval_cmd[@]}"
                done
            done
        done
    done
done
