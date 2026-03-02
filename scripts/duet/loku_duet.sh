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
    echo "[duet][LoKU] Using locally finetuned base checkpoint at ${base_model_path}"
else
    base_model_path="${hf_base_model_path}"
    default_tokenizer_model_path="${hf_base_model_path}"
    echo "[duet][LoKU] Using Hugging Face base checkpoint ${base_model_path}"
fi

tokenizer_model_path="${TOKENIZER_MODEL_PATH:-${default_tokenizer_model_path}}"
tokenizer_subfolder="${TOKENIZER_SUBFOLDER-${sft_subfolder}}"

extra_train_args=()
extra_importance_args=()
extra_eval_tokenizer_args=()
if [[ "${use_sft_base}" == "1" && -n "${sft_subfolder}" ]]; then
    extra_train_args+=(+model.model_args.subfolder=${sft_subfolder})
    extra_importance_args+=(+model.model_args.subfolder=${sft_subfolder})
fi
if [[ "${use_sft_base}" == "1" && -n "${tokenizer_subfolder}" ]]; then
    extra_train_args+=(+model.tokenizer_args.subfolder=${tokenizer_subfolder})
    extra_importance_args+=(+model.tokenizer_args.subfolder=${tokenizer_subfolder})
    extra_eval_tokenizer_args+=(+model.tokenizer_args.subfolder=${tokenizer_subfolder})
fi

experiment="unlearn/duet/loku_lora.yaml"
trainer="LoKU"

output_root="${repo_root}/saves/unlearn/duet/loku"
importance_root="${repo_root}/saves/importances/duet/loku"
mkdir -p "${output_root}" "${importance_root}"

# Match current DUET default behavior used by NPO-SAM/FALCON scripts.
export MERGE_POPULARITY_FORGET=${MERGE_POPULARITY_FORGET:-1}
set_forget_retain_splits

per_device_train_batch_size=${PER_DEVICE_TRAIN_BS:-1}
gradient_accumulation_steps=${GRAD_ACCUM:-32}
importance_batch_size=${IMPORTANCE_BATCH_SIZE:-1}
importance_max_steps=${IMPORTANCE_MAX_STEPS:-0}
eval_batch_size=${EVAL_BATCH_SIZE:-8}
num_train_epochs=${NUM_EPOCHS:-5}
gradient_checkpointing=${GRADIENT_CHECKPOINTING:-false}

raw_lrs="${LRS:-1e-4}"
raw_lrs="${raw_lrs//,/ }"
raw_lrs="${raw_lrs//\"/}"
raw_lrs="${raw_lrs//\'/}"
read -r -a lrs <<< "${raw_lrs}"

raw_ihl_alphas="${IHL_ALPHAS:-1.0}"
raw_ihl_alphas="${raw_ihl_alphas//,/ }"
raw_ihl_alphas="${raw_ihl_alphas//\"/}"
raw_ihl_alphas="${raw_ihl_alphas//\'/}"
read -r -a ihl_alphas <<< "${raw_ihl_alphas}"

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

fila_eps="${FILA_EPS:-1e-5}"
fila_adapter_name="${FILA_ADAPTER_NAME:-default}"
fila_base_subdir="${FILA_BASE_SUBDIR:-base_model}"
run_fila_sanity_check="${RUN_FILA_SANITY_CHECK:-true}"

loku_target_modules="${LOKU_TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]}"
loku_weight_decay="${LOKU_WEIGHT_DECAY:-0.01}"
loku_lr_scheduler_type="${LOKU_LR_SCHEDULER_TYPE:-linear}"
loku_warmup_epochs="${LOKU_WARMUP_EPOCHS:-1.0}"
loku_warmup_ratio="${LOKU_WARMUP_RATIO:-0.0}"

targets_tag="${LOKU_TARGETS_TAG:-no_lm_head_lora_targets}"
force_importance="${FORCE_IMPORTANCE_RECOMPUTE:-0}"
force_rerun="${FORCE_RERUN:-0}"

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

    imp_path="${importance_root}/${base_model}_${forget_label}_${retain_split}_${targets_tag}.pt"
    if [[ ! -f "${imp_path}" || "${force_importance}" == "1" ]]; then
        echo "[duet][LoKU] Measuring importance -> ${imp_path}"
        python src/tools/loku_measure_importance.py \
            --config-name unlearn.yaml \
            --experiment=${experiment} \
            --output-path="${imp_path}" \
            --max-steps=${importance_max_steps} \
            --batch-size=${importance_batch_size} \
            --seed=${TRAIN_SEED:-42} \
            -- \
            model=${lora_model} \
            forget_split=${forget_split} \
            retain_split=${retain_split} \
            model.model_args.pretrained_model_name_or_path=${base_model_path} \
            model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
            model.model_args.device_map=null \
            ++model.model_args.low_cpu_mem_usage=true \
            "model.lora_config.target_modules=${loku_target_modules}" \
            trainer.args.per_device_train_batch_size=${importance_batch_size} \
            trainer.args.gradient_accumulation_steps=1 \
            trainer.args.gradient_checkpointing=false \
            trainer.args.num_train_epochs=1 \
            retain_logs_path=null \
            "${extra_importance_args[@]}"
    fi

    for lr in "${lrs[@]}"; do
        for ihl_alpha in "${ihl_alphas[@]}"; do
            ihl_tag=${ihl_alpha//./p}
            for alpha in "${alphas[@]}"; do
                alpha_tag=${alpha//./p}
                for gamma in "${gammas[@]}"; do
                    gamma_tag=${gamma//./p}
                    for lora_r in "${lora_rs[@]}"; do
                        for lora_alpha in "${lora_alphas[@]}"; do
                            for lora_dropout in "${lora_dropouts[@]}"; do
                                dropout_tag=${lora_dropout//./p}
                                task_name=duet_${base_model}_${forget_label}_loku_lora_r${lora_r}_lalpha${lora_alpha}_ldrop${dropout_tag}_lr${lr}_ihla${ihl_tag}_alpha${alpha_tag}_gamma${gamma_tag}
                                run_dir=${output_root}/${task_name}
                                eval_dir=${run_dir}/evals
                                summary_path=${eval_dir}/DUET_SUMMARY.json
                                base_residual_dir=${run_dir}/${fila_base_subdir}

                                if [[ -f "${summary_path}" && "${force_rerun}" != "1" ]]; then
                                    echo "[duet][LoKU] Skipping ${task_name}: found existing summary at ${summary_path}"
                                    continue
                                fi

                                echo "[duet][LoKU] ${task_name}: unlearning ${base_model_path} on ${forget_split}"

                                adapter_path=${run_dir}/adapter_model.safetensors
                                if [[ ! -f "${adapter_path}" || ! -d "${base_residual_dir}" || "${force_rerun}" == "1" ]]; then
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
                                        "model.lora_config.target_modules=${loku_target_modules}" \
                                        model.lora_config.r=${lora_r} \
                                        model.lora_config.lora_alpha=${lora_alpha} \
                                        model.lora_config.lora_dropout=${lora_dropout} \
                                        trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
                                        trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
                                        trainer.args.num_train_epochs=${num_train_epochs} \
                                        trainer.args.gradient_checkpointing=${gradient_checkpointing} \
                                        trainer.args.learning_rate=${lr} \
                                        trainer.args.weight_decay=${loku_weight_decay} \
                                        trainer.args.lr_scheduler_type=${loku_lr_scheduler_type} \
                                        trainer.args.warmup_epochs=${loku_warmup_epochs} \
                                        trainer.args.warmup_ratio=${loku_warmup_ratio} \
                                        trainer.method_args.ihl_alpha=${ihl_alpha} \
                                        trainer.method_args.alpha=${alpha} \
                                        trainer.method_args.gamma=${gamma} \
                                        trainer.method_args.retain_loss_type=NLL \
                                        trainer.method_args.importance_file=${imp_path} \
                                        trainer.method_args.fila_eps=${fila_eps} \
                                        trainer.method_args.fila_adapter_name=${fila_adapter_name} \
                                        trainer.method_args.fila_base_subdir=${fila_base_subdir} \
                                        trainer.method_args.run_fila_sanity_check=${run_fila_sanity_check} \
                                        retain_logs_path=null \
                                        "${extra_train_args[@]}" \
                                        paths.output_dir=${run_dir}
                                fi

                                mkdir -p "${eval_dir}"
                                if [[ "${force_rerun}" == "1" ]]; then
                                    rm -f "${summary_path}" "${eval_dir}/DUET_EVAL.json"
                                fi

                                eval_cmd=( \
                                    experiment=eval/duet/default.yaml \
                                    model=${lora_model} \
                                    forget_split=${forget_split} \
                                    holdout_split=${retain_split} \
                                    task_name=${task_name} \
                                    model.model_args.pretrained_model_name_or_path=${run_dir} \
                                    ++model.model_args.base_model_name_or_path=${base_residual_dir} \
                                    model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
                                    model.model_args.device_map="auto" \
                                    ++model.model_args.low_cpu_mem_usage=true \
                                    "model.lora_config.target_modules=${loku_target_modules}" \
                                    model.lora_config.r=${lora_r} \
                                    model.lora_config.lora_alpha=${lora_alpha} \
                                    model.lora_config.lora_dropout=${lora_dropout} \
                                    eval.duet.batch_size=${eval_batch_size} \
                                    eval.duet.overwrite=true \
                                    "${extra_eval_tokenizer_args[@]}" \
                                    paths.output_dir=${eval_dir} \
                                    retain_logs_path=null \
                                )
                                python src/eval.py "${eval_cmd[@]}"

                                if [[ "${delete_model_safetensors_after_eval}" == "1" ]]; then
                                    if compgen -G "${run_dir}/*.safetensors" > /dev/null; then
                                        rm -f "${run_dir}"/*.safetensors
                                        echo "[duet][LoKU] Removed safetensors from ${run_dir}"
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
