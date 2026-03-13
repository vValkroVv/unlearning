#!/bin/bash

set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")
repo_root=$(realpath "${script_dir}/../..")
source "${script_dir}/_splits.sh"

is_nullish() {
    local value="${1:-}"
    [[ -z "${value}" || "${value}" == "null" || "${value}" == "None" ]]
}

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
    echo "[duet][DualCF] Using locally finetuned base checkpoint at ${base_model_path}"
else
    base_model_path="${hf_base_model_path}"
    default_tokenizer_model_path="${hf_base_model_path}"
    echo "[duet][DualCF] Using Hugging Face base checkpoint ${base_model_path}"
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

experiment="unlearn/duet/dual_cf_lora.yaml"
trainer="DualCF"

output_root="${repo_root}/saves/unlearn/duet/dual_cf"
mkdir -p "${output_root}"

export MERGE_POPULARITY_FORGET=${MERGE_POPULARITY_FORGET:-1}
set_forget_retain_splits
if [[ -n "${FORGET_SPLIT_OVERRIDE:-}" && -n "${RETAIN_SPLIT_OVERRIDE:-}" ]]; then
    override_label="${FORGET_LABEL_OVERRIDE:-${FORGET_SPLIT_OVERRIDE}}"
    forget_retain_splits=(
        "${FORGET_SPLIT_OVERRIDE} ${RETAIN_SPLIT_OVERRIDE} ${override_label}"
    )
fi

per_device_train_batch_size=${PER_DEVICE_TRAIN_BS:-16}
gradient_accumulation_steps=${GRAD_ACCUM:-2}
eval_batch_size=${EVAL_BATCH_SIZE:-64}
num_train_epochs=${NUM_EPOCHS:-2}
gradient_checkpointing=${GRADIENT_CHECKPOINTING:-false}
max_steps="${MAX_STEPS:-0}"

cf_dataset_path="${CF_DATASET_PATH:-json}"
cf_dataset_data_files="${CF_DATASET_DATA_FILES:-null}"

if [[ "${cf_dataset_path}" == /path/to/* ]]; then
    echo "[duet][DualCF] ERROR: CF_DATASET_PATH still points to a placeholder: ${cf_dataset_path}"
    exit 1
fi
if [[ "${cf_dataset_path}" == "json" ]] && is_nullish "${cf_dataset_data_files}"; then
    echo "[duet][DualCF] ERROR: Local JSON mode requires CF_DATASET_DATA_FILES=/abs/path/to/duet_dualcf.jsonl"
    exit 1
fi

raw_lrs="${LRS:-1e-6 5e-6 1e-5 5e-5 1e-4}"
raw_lrs="${raw_lrs//,/ }"
raw_lrs="${raw_lrs//\"/}"
raw_lrs="${raw_lrs//\'/}"
read -r -a lrs <<< "${raw_lrs}"

raw_betas="${BETAS:-0.5}"
raw_betas="${raw_betas//,/ }"
raw_betas="${raw_betas//\"/}"
raw_betas="${raw_betas//\'/}"
read -r -a betas <<< "${raw_betas}"

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

raw_tau_ds="${TAU_DS:-0.5}"
raw_tau_ds="${raw_tau_ds//,/ }"
raw_tau_ds="${raw_tau_ds//\"/}"
raw_tau_ds="${raw_tau_ds//\'/}"
read -r -a tau_ds <<< "${raw_tau_ds}"

raw_tau_as="${TAU_AS:-0.5}"
raw_tau_as="${raw_tau_as//,/ }"
raw_tau_as="${raw_tau_as//\"/}"
raw_tau_as="${raw_tau_as//\'/}"
read -r -a tau_as <<< "${raw_tau_as}"

raw_temp_ds="${TEMP_DS:-0.25}"
raw_temp_ds="${raw_temp_ds//,/ }"
raw_temp_ds="${raw_temp_ds//\"/}"
raw_temp_ds="${raw_temp_ds//\'/}"
read -r -a temp_ds <<< "${raw_temp_ds}"

raw_temp_as="${TEMP_AS:-0.25}"
raw_temp_as="${raw_temp_as//,/ }"
raw_temp_as="${raw_temp_as//\"/}"
raw_temp_as="${raw_temp_as//\'/}"
read -r -a temp_as <<< "${raw_temp_as}"

raw_lambda_neg_maxs="${LAMBDA_NEG_MAXS:-1.0}"
raw_lambda_neg_maxs="${raw_lambda_neg_maxs//,/ }"
raw_lambda_neg_maxs="${raw_lambda_neg_maxs//\"/}"
raw_lambda_neg_maxs="${raw_lambda_neg_maxs//\'/}"
read -r -a lambda_neg_maxs <<< "${raw_lambda_neg_maxs}"

raw_lambda_ret_los="${LAMBDA_RET_LOS:-1.0}"
raw_lambda_ret_los="${raw_lambda_ret_los//,/ }"
raw_lambda_ret_los="${raw_lambda_ret_los//\"/}"
raw_lambda_ret_los="${raw_lambda_ret_los//\'/}"
read -r -a lambda_ret_los <<< "${raw_lambda_ret_los}"

raw_lambda_ret_his="${LAMBDA_RET_HIS:-2.0}"
raw_lambda_ret_his="${raw_lambda_ret_his//,/ }"
raw_lambda_ret_his="${raw_lambda_ret_his//\"/}"
raw_lambda_ret_his="${raw_lambda_ret_his//\'/}"
read -r -a lambda_ret_his <<< "${raw_lambda_ret_his}"

raw_cf_weights="${CF_WEIGHTS:-1.0}"
raw_cf_weights="${raw_cf_weights//,/ }"
raw_cf_weights="${raw_cf_weights//\"/}"
raw_cf_weights="${raw_cf_weights//\'/}"
read -r -a cf_weights <<< "${raw_cf_weights}"

raw_risk_forget_scales="${RISK_FORGET_SCALES:-0.5}"
raw_risk_forget_scales="${raw_risk_forget_scales//,/ }"
raw_risk_forget_scales="${raw_risk_forget_scales//\"/}"
raw_risk_forget_scales="${raw_risk_forget_scales//\'/}"
read -r -a risk_forget_scales <<< "${raw_risk_forget_scales}"

raw_disable_difficulty_routes="${DISABLE_DIFFICULTY_ROUTES:-false}"
raw_disable_difficulty_routes="${raw_disable_difficulty_routes//,/ }"
raw_disable_difficulty_routes="${raw_disable_difficulty_routes//\"/}"
raw_disable_difficulty_routes="${raw_disable_difficulty_routes//\'/}"
read -r -a disable_difficulty_routes <<< "${raw_disable_difficulty_routes}"

raw_disable_attribution_routes="${DISABLE_ATTRIBUTION_ROUTES:-false}"
raw_disable_attribution_routes="${raw_disable_attribution_routes//,/ }"
raw_disable_attribution_routes="${raw_disable_attribution_routes//\"/}"
raw_disable_attribution_routes="${raw_disable_attribution_routes//\'/}"
read -r -a disable_attribution_routes <<< "${raw_disable_attribution_routes}"

raw_normalize_cf_flags="${NORMALIZE_CF_BY_TOKENS:-true}"
raw_normalize_cf_flags="${raw_normalize_cf_flags//,/ }"
raw_normalize_cf_flags="${raw_normalize_cf_flags//\"/}"
raw_normalize_cf_flags="${raw_normalize_cf_flags//\'/}"
read -r -a normalize_cf_flags <<< "${raw_normalize_cf_flags}"

raw_normalize_neg_flags="${NORMALIZE_NEG_BY_TOKENS:-true}"
raw_normalize_neg_flags="${raw_normalize_neg_flags//,/ }"
raw_normalize_neg_flags="${raw_normalize_neg_flags//\"/}"
raw_normalize_neg_flags="${raw_normalize_neg_flags//\'/}"
read -r -a normalize_neg_flags <<< "${raw_normalize_neg_flags}"

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
    if [[ "${cf_dataset_path}" == "json" ]]; then
        cf_dataset_split="${CF_DATASET_SPLIT:-train}"
    else
        cf_dataset_split="${CF_DATASET_SPLIT:-${forget_split}}"
    fi

    for lr in "${lrs[@]}"; do
        for beta in "${betas[@]}"; do
            beta_tag=${beta//./p}
            for alpha in "${alphas[@]}"; do
                alpha_tag=${alpha//./p}
                for gamma in "${gammas[@]}"; do
                    gamma_tag=${gamma//./p}
                    for tau_d in "${tau_ds[@]}"; do
                        tau_d_tag=${tau_d//./p}
                        for tau_a in "${tau_as[@]}"; do
                            tau_a_tag=${tau_a//./p}
                            for temp_d in "${temp_ds[@]}"; do
                                temp_d_tag=${temp_d//./p}
                                for temp_a in "${temp_as[@]}"; do
                                    temp_a_tag=${temp_a//./p}
                                    for lambda_neg_max in "${lambda_neg_maxs[@]}"; do
                                        lambda_neg_tag=${lambda_neg_max//./p}
                                        for lambda_ret_lo in "${lambda_ret_los[@]}"; do
                                            lambda_ret_lo_tag=${lambda_ret_lo//./p}
                                            for lambda_ret_hi in "${lambda_ret_his[@]}"; do
                                                lambda_ret_hi_tag=${lambda_ret_hi//./p}
                                                for cf_weight in "${cf_weights[@]}"; do
                                                    cf_weight_tag=${cf_weight//./p}
                                                    for risk_forget_scale in "${risk_forget_scales[@]}"; do
                                                        risk_forget_tag=${risk_forget_scale//./p}
                                                        for disable_difficulty_route in "${disable_difficulty_routes[@]}"; do
                                                            for disable_attribution_route in "${disable_attribution_routes[@]}"; do
                                                                for normalize_cf in "${normalize_cf_flags[@]}"; do
                                                                    for normalize_neg in "${normalize_neg_flags[@]}"; do
                                                                        for lora_r in "${lora_rs[@]}"; do
                                                                            for lora_alpha in "${lora_alphas[@]}"; do
                                                                                for lora_dropout in "${lora_dropouts[@]}"; do
                                                                                    dropout_tag=${lora_dropout//./p}
                                                                                    difficulty_tag="dOn"
                                                                                    attribution_tag="aOn"
                                                                                    if [[ "${disable_difficulty_route}" == "true" ]]; then
                                                                                        difficulty_tag="dOff"
                                                                                    fi
                                                                                    if [[ "${disable_attribution_route}" == "true" ]]; then
                                                                                        attribution_tag="aOff"
                                                                                    fi
                                                                                    task_name=duet_${base_model}_${forget_label}_dual_cf_lora_r${lora_r}_lalpha${lora_alpha}_ldrop${dropout_tag}_lr${lr}_beta${beta_tag}_alpha${alpha_tag}_gamma${gamma_tag}_td${tau_d_tag}_ta${tau_a_tag}_sd${temp_d_tag}_sa${temp_a_tag}_ln${lambda_neg_tag}_rlo${lambda_ret_lo_tag}_rhi${lambda_ret_hi_tag}_cf${cf_weight_tag}_rf${risk_forget_tag}_${difficulty_tag}_${attribution_tag}
                                                                                    run_dir=${output_root}/${task_name}
                                                                                    eval_dir=${run_dir}/evals
                                                                                    summary_path=${eval_dir}/DUET_SUMMARY.json

                                                                                    if [[ -f "${summary_path}" && "${FORCE_RERUN:-0}" != "1" ]]; then
                                                                                        echo "[duet][DualCF] Skipping ${task_name}: found existing summary at ${summary_path}"
                                                                                        continue
                                                                                    fi

                                                                                    echo "[duet][DualCF] ${task_name}: unlearning ${base_model_path} on ${forget_split}"

                                                                                    adapter_path=${run_dir}/adapter_model.safetensors
                                                                                    if [[ ! -f "${adapter_path}" || "${FORCE_RERUN:-0}" == "1" ]]; then
                                                                                        mkdir -p "${run_dir}"
                                                                                        train_cmd=( \
                                                                                            src/train.py \
                                                                                            --config-name=unlearn.yaml \
                                                                                            experiment=${experiment} \
                                                                                            trainer=${trainer} \
                                                                                            task_name=${task_name} \
                                                                                            model=${lora_model} \
                                                                                            forget_split=${forget_split} \
                                                                                            retain_split=${retain_split} \
                                                                                            cf_dataset_path=${cf_dataset_path} \
                                                                                            "cf_dataset_split='${cf_dataset_split}'" \
                                                                                            cf_dataset_data_files=${cf_dataset_data_files} \
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
                                                                                            trainer.method_args.tau_d=${tau_d} \
                                                                                            trainer.method_args.tau_a=${tau_a} \
                                                                                            trainer.method_args.temp_d=${temp_d} \
                                                                                            trainer.method_args.temp_a=${temp_a} \
                                                                                            trainer.method_args.lambda_neg_max=${lambda_neg_max} \
                                                                                            trainer.method_args.lambda_ret_lo=${lambda_ret_lo} \
                                                                                            trainer.method_args.lambda_ret_hi=${lambda_ret_hi} \
                                                                                            trainer.method_args.cf_weight=${cf_weight} \
                                                                                            trainer.method_args.risk_forget_scale=${risk_forget_scale} \
                                                                                            trainer.method_args.normalize_cf_by_tokens=${normalize_cf} \
                                                                                            trainer.method_args.normalize_neg_by_tokens=${normalize_neg} \
                                                                                            trainer.method_args.disable_difficulty_route=${disable_difficulty_route} \
                                                                                            trainer.method_args.disable_attribution_route=${disable_attribution_route} \
                                                                                            retain_logs_path=null \
                                                                                            "${extra_train_args[@]}" \
                                                                                            paths.output_dir=${run_dir} \
                                                                                        )
                                                                                        if [[ "${max_steps}" != "0" ]]; then
                                                                                            train_cmd+=(+trainer.args.max_steps=${max_steps})
                                                                                        fi
                                                                                        python "${train_cmd[@]}"
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
                                                                                            echo "[duet][DualCF] Removed safetensors from ${run_dir}"
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
done
