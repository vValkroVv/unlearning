#!/bin/bash

set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")
repo_root=$(realpath "${script_dir}/../..")
source "${script_dir}/_splits.sh"

is_nullish() {
    local value="${1:-}"
    [[ -z "${value}" || "${value}" == "null" || "${value}" == "None" ]]
}

resolve_num_rows() {
    local dataset_path="$1"
    local split="$2"
    local data_files="$3"
    if [[ "${dataset_path}" == "json" ]]; then
        wc -l < "${data_files}"
        return
    fi
    python - <<PY
import datasets
kwargs = {"split": ${split@Q}}
dataset_path = ${dataset_path@Q}
data_files = ${data_files@Q}
if data_files not in ("", "null", "None"):
    kwargs["data_files"] = data_files
ds = datasets.load_dataset(dataset_path, **kwargs)
print(len(ds))
PY
}

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

experiment="${EXPERIMENT:-unlearn/duet/dual_cf_v2_lora.yaml}"
trainer="${TRAINER:-DualCF}"
method_name="${METHOD_NAME:-dual_cf}"
run_label="${RUN_LABEL:-DualCF}"

base_model="${BASE_MODEL:-Llama-3.1-8B-Instruct}"
lora_model="${MODEL_CONFIG:-${base_model}-lora}"
hf_base_model_path="${HF_BASE_MODEL_PATH:-meta-llama/${base_model}}"
local_sft_base="${LOCAL_SFT_BASE:-/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/llama3.1-8b_full_3ep_ft_tripunlamb}"
sft_subfolder="${SFT_SUBFOLDER:-}"

use_sft_base=${USE_SFT_BASE:-1}
if [[ "${use_sft_base}" == "1" ]]; then
    base_model_path="${local_sft_base}"
    default_tokenizer_model_path="${base_model_path}"
    echo "[duet][${run_label}] Using locally finetuned base checkpoint at ${base_model_path}"
else
    base_model_path="${hf_base_model_path}"
    default_tokenizer_model_path="${hf_base_model_path}"
    echo "[duet][${run_label}] Using Hugging Face base checkpoint ${base_model_path}"
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

output_root="${OUTPUT_ROOT:-${repo_root}/saves/unlearn/duet/${method_name}}"
mkdir -p "${output_root}"

export MERGE_POPULARITY_FORGET=${MERGE_POPULARITY_FORGET:-1}
set_forget_retain_splits
if [[ -n "${FORGET_SPLIT_OVERRIDE:-}" && -n "${RETAIN_SPLIT_OVERRIDE:-}" ]]; then
    override_label="${FORGET_LABEL_OVERRIDE:-${FORGET_SPLIT_OVERRIDE}}"
    forget_retain_splits=(
        "${FORGET_SPLIT_OVERRIDE} ${RETAIN_SPLIT_OVERRIDE} ${override_label}"
    )
fi

per_device_train_batch_size=${PER_DEVICE_TRAIN_BS:-32}
gradient_accumulation_steps=${GRAD_ACCUM:-1}
eval_batch_size=${EVAL_BATCH_SIZE:-192}
num_train_epochs=${NUM_EPOCHS:-5}
gradient_checkpointing=${GRADIENT_CHECKPOINTING:-false}
max_steps="${MAX_STEPS:-0}"
checkpoint_every_half_epoch="${CHECKPOINT_EVERY_HALF_EPOCH:-1}"

short_multicf_agg_tag() {
    case "$1" in
        weighted_mean) echo "wm" ;;
        mean) echo "m" ;;
        top1) echo "t1" ;;
        *) echo "$1" ;;
    esac
}

short_multicf_weight_tag() {
    case "$1" in
        rerank) echo "rr" ;;
        uniform) echo "uni" ;;
        *) echo "$1" ;;
    esac
}

short_span_mode_tag() {
    case "$1" in
        lcs) echo "lc" ;;
        set_overlap) echo "so" ;;
        *) echo "$1" ;;
    esac
}
save_total_limit="${SAVE_TOTAL_LIMIT:-12}"
checkpoint_epochs_raw="${CHECKPOINT_EPOCHS:-}"
checkpoint_epochs_csv=""
if [[ -n "${checkpoint_epochs_raw}" ]]; then
    checkpoint_epochs_raw="${checkpoint_epochs_raw//,/ }"
    checkpoint_epochs_raw="${checkpoint_epochs_raw//\"/}"
    checkpoint_epochs_raw="${checkpoint_epochs_raw//\'/}"
    read -r -a checkpoint_epochs <<< "${checkpoint_epochs_raw}"
    checkpoint_epochs_csv=$(IFS=,; echo "${checkpoint_epochs[*]}")
fi
run_tag_extra="${RUN_TAG_EXTRA:-}"

cf_dataset_path="${CF_DATASET_PATH:-json}"
cf_dataset_data_files="${CF_DATASET_DATA_FILES:-null}"

if [[ "${cf_dataset_path}" == /path/to/* ]]; then
    echo "[duet][${run_label}] ERROR: CF_DATASET_PATH still points to a placeholder: ${cf_dataset_path}"
    exit 1
fi
if [[ "${cf_dataset_path}" == "json" ]] && is_nullish "${cf_dataset_data_files}"; then
    echo "[duet][${run_label}] ERROR: Local JSON mode requires CF_DATASET_DATA_FILES=/abs/path/to/duet_dualcf.jsonl"
    exit 1
fi

raw_lrs="${LRS:-1e-6 5e-6 1e-5 5e-5 1e-4}"
raw_lrs="${raw_lrs//,/ }"; raw_lrs="${raw_lrs//\"/}"; raw_lrs="${raw_lrs//\'/}"
read -r -a lrs <<< "${raw_lrs}"

raw_betas="${BETAS:-0.5}"
raw_betas="${raw_betas//,/ }"; raw_betas="${raw_betas//\"/}"; raw_betas="${raw_betas//\'/}"
read -r -a betas <<< "${raw_betas}"

raw_alphas="${ALPHAS:-1.0}"
raw_alphas="${raw_alphas//,/ }"; raw_alphas="${raw_alphas//\"/}"; raw_alphas="${raw_alphas//\'/}"
read -r -a alphas <<< "${raw_alphas}"

raw_gammas="${GAMMAS:-1.0}"
raw_gammas="${raw_gammas//,/ }"; raw_gammas="${raw_gammas//\"/}"; raw_gammas="${raw_gammas//\'/}"
read -r -a gammas <<< "${raw_gammas}"

raw_tau_ds="${TAU_DS:-0.6}"
raw_tau_ds="${raw_tau_ds//,/ }"; raw_tau_ds="${raw_tau_ds//\"/}"; raw_tau_ds="${raw_tau_ds//\'/}"
read -r -a tau_ds <<< "${raw_tau_ds}"

raw_tau_as="${TAU_AS:-0.6}"
raw_tau_as="${raw_tau_as//,/ }"; raw_tau_as="${raw_tau_as//\"/}"; raw_tau_as="${raw_tau_as//\'/}"
read -r -a tau_as <<< "${raw_tau_as}"

raw_temp_ds="${TEMP_DS:-0.15}"
raw_temp_ds="${raw_temp_ds//,/ }"; raw_temp_ds="${raw_temp_ds//\"/}"; raw_temp_ds="${raw_temp_ds//\'/}"
read -r -a temp_ds <<< "${raw_temp_ds}"

raw_temp_as="${TEMP_AS:-0.15}"
raw_temp_as="${raw_temp_as//,/ }"; raw_temp_as="${raw_temp_as//\"/}"; raw_temp_as="${raw_temp_as//\'/}"
read -r -a temp_as <<< "${raw_temp_as}"

raw_lambda_neg_maxs="${LAMBDA_NEG_MAXS:-1.0}"
raw_lambda_neg_maxs="${raw_lambda_neg_maxs//,/ }"; raw_lambda_neg_maxs="${raw_lambda_neg_maxs//\"/}"; raw_lambda_neg_maxs="${raw_lambda_neg_maxs//\'/}"
read -r -a lambda_neg_maxs <<< "${raw_lambda_neg_maxs}"

raw_lambda_ret_los="${LAMBDA_RET_LOS:-1.0}"
raw_lambda_ret_los="${raw_lambda_ret_los//,/ }"; raw_lambda_ret_los="${raw_lambda_ret_los//\"/}"; raw_lambda_ret_los="${raw_lambda_ret_los//\'/}"
read -r -a lambda_ret_los <<< "${raw_lambda_ret_los}"

raw_lambda_ret_his="${LAMBDA_RET_HIS:-3.0}"
raw_lambda_ret_his="${raw_lambda_ret_his//,/ }"; raw_lambda_ret_his="${raw_lambda_ret_his//\"/}"; raw_lambda_ret_his="${raw_lambda_ret_his//\'/}"
read -r -a lambda_ret_his <<< "${raw_lambda_ret_his}"

raw_cf_weights="${CF_WEIGHTS:-1.0}"
raw_cf_weights="${raw_cf_weights//,/ }"; raw_cf_weights="${raw_cf_weights//\"/}"; raw_cf_weights="${raw_cf_weights//\'/}"
read -r -a cf_weights <<< "${raw_cf_weights}"

raw_risk_forget_scales="${RISK_FORGET_SCALES:-0.5}"
raw_risk_forget_scales="${raw_risk_forget_scales//,/ }"; raw_risk_forget_scales="${raw_risk_forget_scales//\"/}"; raw_risk_forget_scales="${raw_risk_forget_scales//\'/}"
read -r -a risk_forget_scales <<< "${raw_risk_forget_scales}"

raw_disable_difficulty_routes="${DISABLE_DIFFICULTY_ROUTES:-false}"
raw_disable_difficulty_routes="${raw_disable_difficulty_routes//,/ }"; raw_disable_difficulty_routes="${raw_disable_difficulty_routes//\"/}"; raw_disable_difficulty_routes="${raw_disable_difficulty_routes//\'/}"
read -r -a disable_difficulty_routes <<< "${raw_disable_difficulty_routes}"

raw_disable_attribution_routes="${DISABLE_ATTRIBUTION_ROUTES:-false}"
raw_disable_attribution_routes="${raw_disable_attribution_routes//,/ }"; raw_disable_attribution_routes="${raw_disable_attribution_routes//\"/}"; raw_disable_attribution_routes="${raw_disable_attribution_routes//\'/}"
read -r -a disable_attribution_routes <<< "${raw_disable_attribution_routes}"

raw_normalize_cf_flags="${NORMALIZE_CF_BY_TOKENS:-true}"
raw_normalize_cf_flags="${raw_normalize_cf_flags//,/ }"; raw_normalize_cf_flags="${raw_normalize_cf_flags//\"/}"; raw_normalize_cf_flags="${raw_normalize_cf_flags//\'/}"
read -r -a normalize_cf_flags <<< "${raw_normalize_cf_flags}"

raw_normalize_neg_flags="${NORMALIZE_NEG_BY_TOKENS:-true}"
raw_normalize_neg_flags="${raw_normalize_neg_flags//,/ }"; raw_normalize_neg_flags="${raw_normalize_neg_flags//\"/}"; raw_normalize_neg_flags="${raw_normalize_neg_flags//\'/}"
read -r -a normalize_neg_flags <<< "${raw_normalize_neg_flags}"

raw_alpha_eff_stats="${ALPHA_EFF_STATS:-topk_mean}"
raw_alpha_eff_stats="${raw_alpha_eff_stats//,/ }"; raw_alpha_eff_stats="${raw_alpha_eff_stats//\"/}"; raw_alpha_eff_stats="${raw_alpha_eff_stats//\'/}"
read -r -a alpha_eff_stats <<< "${raw_alpha_eff_stats}"

raw_alpha_eff_topk_fracs="${ALPHA_EFF_TOPK_FRACS:-0.25}"
raw_alpha_eff_topk_fracs="${raw_alpha_eff_topk_fracs//,/ }"; raw_alpha_eff_topk_fracs="${raw_alpha_eff_topk_fracs//\"/}"; raw_alpha_eff_topk_fracs="${raw_alpha_eff_topk_fracs//\'/}"
read -r -a alpha_eff_topk_fracs <<< "${raw_alpha_eff_topk_fracs}"

raw_risk_powers="${RISK_POWERS:-1.0}"
raw_risk_powers="${raw_risk_powers//,/ }"; raw_risk_powers="${raw_risk_powers//\"/}"; raw_risk_powers="${raw_risk_powers//\'/}"
read -r -a risk_powers <<< "${raw_risk_powers}"

raw_neg_powers="${NEG_POWERS:-1.0}"
raw_neg_powers="${raw_neg_powers//,/ }"; raw_neg_powers="${raw_neg_powers//\"/}"; raw_neg_powers="${raw_neg_powers//\'/}"
read -r -a neg_powers <<< "${raw_neg_powers}"

multicf_max_alternates_used="${MULTICF_MAX_ALTERNATES_USED:-4}"
multicf_alt_agg_mode="${MULTICF_ALT_AGG_MODE:-weighted_mean}"
multicf_alt_weight_mode="${MULTICF_ALT_WEIGHT_MODE:-rerank}"
multicf_alt_set_temperature="${MULTICF_ALT_SET_TEMPERATURE:-0.7}"
boundary_local_retain_weight="${BOUNDARY_LOCAL_RETAIN_WEIGHT:-0.5}"
boundary_margin_weight="${BOUNDARY_MARGIN_WEIGHT:-1.0}"
span_mode="${SPAN_MODE:-lcs}"
span_shared_token_weight="${SPAN_SHARED_TOKEN_WEIGHT:-0.25}"
span_unique_token_weight="${SPAN_UNIQUE_TOKEN_WEIGHT:-1.0}"

lora_rs=(${LORA_RS:-"32"})
lora_alphas=(${LORA_ALPHAS:-"64"})
lora_dropouts=(${LORA_DROPOUTS:-"0.0"})
delete_model_safetensors_after_eval="${DELETE_MODEL_SAFETENSORS_AFTER_EVAL:-0}"
run_checkpoint_eval="${RUN_CHECKPOINT_EVAL:-${RUN_UTILITY_EVAL:-0}}"

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
                                                                        for alpha_eff_stat in "${alpha_eff_stats[@]}"; do
                                                                            for alpha_eff_topk_frac in "${alpha_eff_topk_fracs[@]}"; do
                                                                                for risk_power in "${risk_powers[@]}"; do
                                                                                    for neg_power in "${neg_powers[@]}"; do
                                                                                        for lora_r in "${lora_rs[@]}"; do
                                                                                            for lora_alpha in "${lora_alphas[@]}"; do
                                                                                                for lora_dropout in "${lora_dropouts[@]}"; do
                                                                                                    dropout_tag=${lora_dropout//./p}
                                                                                                    alpha_eff_topk_tag=${alpha_eff_topk_frac//./p}
                                                                                                    risk_power_tag=${risk_power//./p}
                                                                                                    neg_power_tag=${neg_power//./p}
                                                                                                    difficulty_tag="dOn"
                                                                                                    attribution_tag="aOn"
                                                                                                    if [[ "${disable_difficulty_route}" == "true" ]]; then
                                                                                                        difficulty_tag="dOff"
                                                                                                    fi
                                                                                                    if [[ "${disable_attribution_route}" == "true" ]]; then
                                                                                                        attribution_tag="aOff"
                                                                                                    fi

                                                                                                    method_suffix=""
                                                                                                    if [[ "${trainer}" == "MultiCF" ]]; then
                                                                                                        multicf_temp_tag=${multicf_alt_set_temperature//./p}
                                                                                                        multicf_agg_tag=$(short_multicf_agg_tag "${multicf_alt_agg_mode}")
                                                                                                        multicf_weight_tag=$(short_multicf_weight_tag "${multicf_alt_weight_mode}")
                                                                                                        method_suffix=_k${multicf_max_alternates_used}_ag${multicf_agg_tag}_w${multicf_weight_tag}_t${multicf_temp_tag}
                                                                                                    elif [[ "${trainer}" == "BoundaryCF" ]]; then
                                                                                                        boundary_local_tag=${boundary_local_retain_weight//./p}
                                                                                                        boundary_margin_tag=${boundary_margin_weight//./p}
                                                                                                        method_suffix=_lr${boundary_local_tag}_bm${boundary_margin_tag}
                                                                                                    elif [[ "${trainer}" == "SpanCF" ]]; then
                                                                                                        span_mode_tag=$(short_span_mode_tag "${span_mode}")
                                                                                                        span_shared_tag=${span_shared_token_weight//./p}
                                                                                                        span_unique_tag=${span_unique_token_weight//./p}
                                                                                                        method_suffix=_m${span_mode_tag}_sw${span_shared_tag}_uw${span_unique_tag}
                                                                                                    fi

                                                                                                    task_name=duet_${base_model}_${forget_label}_${method_name}_lora_r${lora_r}_lalpha${lora_alpha}_ldrop${dropout_tag}_lr${lr}_beta${beta_tag}_alpha${alpha_tag}_gamma${gamma_tag}_td${tau_d_tag}_ta${tau_a_tag}_sd${temp_d_tag}_sa${temp_a_tag}_ln${lambda_neg_tag}_rlo${lambda_ret_lo_tag}_rhi${lambda_ret_hi_tag}_cf${cf_weight_tag}_rf${risk_forget_tag}_ae${alpha_eff_stat}_atk${alpha_eff_topk_tag}_rp${risk_power_tag}_np${neg_power_tag}${method_suffix}_${difficulty_tag}_${attribution_tag}
                                                                                                    if [[ -n "${run_tag_extra}" ]]; then
                                                                                                        task_name="${task_name}_${run_tag_extra}"
                                                                                                    fi
                                                                                                    run_dir=${output_root}/${task_name}
                                                                                                    eval_dir=${run_dir}/evals
                                                                                                    summary_path=${eval_dir}/DUET_SUMMARY.json

                                                                                                    if [[ -f "${summary_path}" && "${FORCE_RERUN:-0}" != "1" ]]; then
                                                                                                        echo "[duet][${run_label}] Skipping ${task_name}: found existing summary at ${summary_path}"
                                                                                                        continue
                                                                                                    fi

                                                                                                    echo "[duet][${run_label}] ${task_name}: unlearning ${base_model_path} on ${forget_split}"

                                                                                                    adapter_path=${run_dir}/adapter_model.safetensors
                                                                                                    if [[ ! -f "${adapter_path}" || "${FORCE_RERUN:-0}" == "1" ]]; then
                                                                                                        mkdir -p "${run_dir}"
                                                                                                        extra_schedule_args=()
                                                                                                        if [[ -n "${checkpoint_epochs_csv}" ]]; then
                                                                                                            extra_schedule_args+=(
                                                                                                                ++trainer.args.save_strategy=no
                                                                                                                ++trainer.args.save_total_limit=${save_total_limit}
                                                                                                                ++trainer.args.save_safetensors=true
                                                                                                                ++trainer.save_on_epochs=[${checkpoint_epochs_csv}]
                                                                                                            )
                                                                                                        elif [[ "${checkpoint_every_half_epoch}" == "1" && "${max_steps}" == "0" ]]; then
                                                                                                            num_rows=$(resolve_num_rows "${cf_dataset_path}" "${cf_dataset_split}" "${cf_dataset_data_files}")
                                                                                                            global_batch=$(( per_device_train_batch_size * gradient_accumulation_steps ))
                                                                                                            steps_per_epoch=$(( (num_rows + global_batch - 1) / global_batch ))
                                                                                                            half_epoch_steps=$(( (steps_per_epoch + 1) / 2 ))
                                                                                                            logging_steps=$(( half_epoch_steps / 2 ))
                                                                                                            if [[ "${logging_steps}" -lt 1 ]]; then
                                                                                                                logging_steps=1
                                                                                                            fi
                                                                                                            extra_schedule_args+=(
                                                                                                                ++trainer.args.save_strategy=steps
                                                                                                                ++trainer.args.save_steps=${half_epoch_steps}
                                                                                                                ++trainer.args.save_total_limit=${save_total_limit}
                                                                                                                ++trainer.args.logging_strategy=steps
                                                                                                                ++trainer.args.logging_steps=${logging_steps}
                                                                                                                ++trainer.args.save_safetensors=true
                                                                                                                ++trainer.args.load_best_model_at_end=false
                                                                                                            )
                                                                                                        fi

                                                                                                        extra_method_args=(
                                                                                                            trainer.method_args.beta=${beta}
                                                                                                            trainer.method_args.alpha=${alpha}
                                                                                                            trainer.method_args.gamma=${gamma}
                                                                                                            trainer.method_args.retain_loss_type=NLL
                                                                                                        )
                                                                                                        if [[ "${trainer}" == "DualCF" || "${trainer}" == "MultiCF" || "${trainer}" == "BoundaryCF" || "${trainer}" == "SpanCF" ]]; then
                                                                                                            extra_method_args+=(
                                                                                                                trainer.method_args.tau_d=${tau_d}
                                                                                                                trainer.method_args.tau_a=${tau_a}
                                                                                                                trainer.method_args.temp_d=${temp_d}
                                                                                                                trainer.method_args.temp_a=${temp_a}
                                                                                                                trainer.method_args.lambda_neg_max=${lambda_neg_max}
                                                                                                                trainer.method_args.lambda_ret_lo=${lambda_ret_lo}
                                                                                                                trainer.method_args.lambda_ret_hi=${lambda_ret_hi}
                                                                                                                trainer.method_args.cf_weight=${cf_weight}
                                                                                                                trainer.method_args.risk_forget_scale=${risk_forget_scale}
                                                                                                                trainer.method_args.normalize_cf_by_tokens=${normalize_cf}
                                                                                                                trainer.method_args.normalize_neg_by_tokens=${normalize_neg}
                                                                                                                trainer.method_args.disable_difficulty_route=${disable_difficulty_route}
                                                                                                                trainer.method_args.disable_attribution_route=${disable_attribution_route}
                                                                                                                trainer.method_args.alpha_eff_stat=${alpha_eff_stat}
                                                                                                                trainer.method_args.alpha_eff_topk_frac=${alpha_eff_topk_frac}
                                                                                                                trainer.method_args.risk_power=${risk_power}
                                                                                                                trainer.method_args.neg_power=${neg_power}
                                                                                                            )
                                                                                                        fi
                                                                                                        if [[ "${trainer}" == "MultiCF" ]]; then
                                                                                                            extra_method_args+=(
                                                                                                                trainer.method_args.max_alternates_used=${multicf_max_alternates_used}
                                                                                                                trainer.method_args.alt_agg_mode=${multicf_alt_agg_mode}
                                                                                                                trainer.method_args.alt_weight_mode=${multicf_alt_weight_mode}
                                                                                                                trainer.method_args.alt_set_temperature=${multicf_alt_set_temperature}
                                                                                                            )
                                                                                                        elif [[ "${trainer}" == "BoundaryCF" ]]; then
                                                                                                            extra_method_args+=(
                                                                                                                trainer.method_args.local_retain_weight=${boundary_local_retain_weight}
                                                                                                                trainer.method_args.boundary_margin_weight=${boundary_margin_weight}
                                                                                                            )
                                                                                                        elif [[ "${trainer}" == "SpanCF" ]]; then
                                                                                                            extra_method_args+=(
                                                                                                                trainer.method_args.span_mode=${span_mode}
                                                                                                                trainer.method_args.shared_token_weight=${span_shared_token_weight}
                                                                                                                trainer.method_args.unique_token_weight=${span_unique_token_weight}
                                                                                                            )
                                                                                                        fi

                                                                                                        train_cmd=(
                                                                                                            src/train.py
                                                                                                            --config-name=unlearn.yaml
                                                                                                            experiment=${experiment}
                                                                                                            trainer=${trainer}
                                                                                                            task_name=${task_name}
                                                                                                            model=${lora_model}
                                                                                                            forget_split=${forget_split}
                                                                                                            retain_split=${retain_split}
                                                                                                            cf_dataset_path=${cf_dataset_path}
                                                                                                            "cf_dataset_split='${cf_dataset_split}'"
                                                                                                            cf_dataset_data_files=${cf_dataset_data_files}
                                                                                                            model.model_args.pretrained_model_name_or_path=${base_model_path}
                                                                                                            model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path}
                                                                                                            model.model_args.device_map="auto"
                                                                                                            ++model.model_args.low_cpu_mem_usage=true
                                                                                                            model.lora_config.r=${lora_r}
                                                                                                            model.lora_config.lora_alpha=${lora_alpha}
                                                                                                            model.lora_config.lora_dropout=${lora_dropout}
                                                                                                            ++trainer.args.seed=${TRAIN_SEED:-42}
                                                                                                            ++trainer.args.data_seed=${DATA_SEED:-${TRAIN_SEED:-42}}
                                                                                                            trainer.args.per_device_train_batch_size=${per_device_train_batch_size}
                                                                                                            trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps}
                                                                                                            trainer.args.num_train_epochs=${num_train_epochs}
                                                                                                            trainer.args.gradient_checkpointing=${gradient_checkpointing}
                                                                                                            trainer.args.learning_rate=${lr}
                                                                                                            "${extra_method_args[@]}"
                                                                                                            retain_logs_path=null
                                                                                                            "${extra_schedule_args[@]}"
                                                                                                            "${extra_train_args[@]}"
                                                                                                            paths.output_dir=${run_dir}
                                                                                                        )
                                                                                                        if [[ "${max_steps}" != "0" ]]; then
                                                                                                            train_cmd+=(+trainer.args.max_steps=${max_steps})
                                                                                                        fi
                                                                                                        if [[ "${FULL_DETERMINISM:-0}" == "1" ]]; then
                                                                                                            train_cmd+=(++trainer.args.full_determinism=true)
                                                                                                        fi
                                                                                                        python "${train_cmd[@]}"
                                                                                                    fi

                                                                                                    mkdir -p "${eval_dir}"
                                                                                                    if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
                                                                                                        rm -f "${summary_path}" "${eval_dir}/DUET_EVAL.json"
                                                                                                    fi

                                                                                                    eval_cmd=(
                                                                                                        experiment=eval/duet/default.yaml
                                                                                                        model=${lora_model}
                                                                                                        forget_split=${forget_split}
                                                                                                        holdout_split=${retain_split}
                                                                                                        task_name=${task_name}
                                                                                                        model.model_args.pretrained_model_name_or_path=${run_dir}
                                                                                                        ++model.model_args.base_model_name_or_path=${base_model_path}
                                                                                                        model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path}
                                                                                                        model.model_args.device_map="auto"
                                                                                                        ++model.model_args.low_cpu_mem_usage=true
                                                                                                        model.lora_config.r=${lora_r}
                                                                                                        model.lora_config.lora_alpha=${lora_alpha}
                                                                                                        model.lora_config.lora_dropout=${lora_dropout}
                                                                                                        eval.duet.batch_size=${eval_batch_size}
                                                                                                        eval.duet.overwrite=true
                                                                                                        "${extra_eval_args[@]}"
                                                                                                        paths.output_dir=${eval_dir}
                                                                                                        retain_logs_path=null
                                                                                                    )
                                                                                                    python src/eval.py "${eval_cmd[@]}"

                                                                                                    if [[ "${run_checkpoint_eval}" == "1" ]]; then
                                                                                                        bash "${script_dir}/eval_checkpoints_duet.sh" \
                                                                                                            "${run_dir}" \
                                                                                                            "${forget_split}" \
                                                                                                            "${retain_split}" \
                                                                                                            "${base_model_path}" \
                                                                                                            "${tokenizer_model_path}" \
                                                                                                            "${lora_model}" \
                                                                                                            "${base_model}"
                                                                                                    fi

                                                                                                    if [[ "${delete_model_safetensors_after_eval}" == "1" ]]; then
                                                                                                        if compgen -G "${run_dir}/*.safetensors" > /dev/null; then
                                                                                                            rm -f "${run_dir}"/*.safetensors
                                                                                                            echo "[duet][${run_label}] Removed safetensors from ${run_dir}"
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
            done
        done
    done
done
