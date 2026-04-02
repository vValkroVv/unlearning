#!/bin/bash

set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")
repo_root=$(realpath "${script_dir}/../..")

is_nullish() {
    local value="${1:-}"
    [[ -z "${value}" || "${value}" == "null" || "${value}" == "None" ]]
}

resolve_num_rows() {
    local dataset_path="$1"
    local dataset_name="$2"
    local split="$3"
    local data_files="$4"
    if [[ "${dataset_path}" == "json" ]]; then
        wc -l < "${data_files}"
        return
    fi
    python - <<PY
import datasets
kwargs = {"split": ${split@Q}}
dataset_path = ${dataset_path@Q}
dataset_name = ${dataset_name@Q}
data_files = ${data_files@Q}
if dataset_name not in ("", "null", "None"):
    kwargs["name"] = dataset_name
if data_files not in ("", "null", "None"):
    kwargs["data_files"] = data_files
ds = datasets.load_dataset(dataset_path, **kwargs)
print(len(ds))
PY
}

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

experiment="${EXPERIMENT:-unlearn/rwku/dual_cf_v2_lora.yaml}"
trainer="${TRAINER:-DualCF}"
method_name="${METHOD_NAME:-dual_cf}"
run_label="${RUN_LABEL:-DualCF}"

base_model="${BASE_MODEL:-Llama-3.1-8B-Instruct}"
lora_model="${MODEL_CONFIG:-${base_model}-lora}"
base_model_path="${HF_BASE_MODEL_PATH:-meta-llama/${base_model}}"
tokenizer_model_path="${TOKENIZER_MODEL_PATH:-${base_model_path}}"

if [[ -n "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

echo "[rwku][${run_label}] Using Hugging Face base checkpoint ${base_model_path}"

output_root="${OUTPUT_ROOT:-${repo_root}/saves/unlearn/rwku/${method_name}}"
mkdir -p "${output_root}"

forget_split="${FORGET_SPLIT:-forget_level2}"
retain_split="${RETAIN_SPLIT:-neighbor_level2}"
cf_dataset_path="${CF_DATASET_PATH:-json}"
cf_dataset_data_files="${CF_DATASET_DATA_FILES:-null}"

if [[ "${cf_dataset_path}" == /path/to/* ]]; then
    echo "[rwku][${run_label}] ERROR: CF_DATASET_PATH still points to a placeholder: ${cf_dataset_path}"
    exit 1
fi
if [[ "${cf_dataset_path}" == "json" ]]; then
    if is_nullish "${cf_dataset_data_files}"; then
        echo "[rwku][${run_label}] ERROR: Local JSON mode requires CF_DATASET_DATA_FILES=/abs/path/to/rwku_dualcf.jsonl"
        exit 1
    fi
    cf_dataset_name="${CF_DATASET_NAME:-null}"
    cf_dataset_split="${CF_DATASET_SPLIT:-train}"
else
    cf_dataset_name="${CF_DATASET_NAME:-${forget_split}}"
    cf_dataset_split="${CF_DATASET_SPLIT:-test}"
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

short_bool_tag() {
    case "$1" in
        true|1|yes|on|TRUE|True) echo "t" ;;
        *) echo "f" ;;
    esac
}

hash_text() {
    python - "$1" <<'PY'
import hashlib
import sys

print(hashlib.sha1(sys.argv[1].encode("utf-8")).hexdigest()[:10])
PY
}

compact_task_name_if_needed() {
    local task_name="$1"
    local task_prefix="$2"
    local lora_r_tag="$3"
    local lora_alpha_tag="$4"
    local dropout_compact_tag="$5"
    local lr_tag="$6"
    local compact_hash_input="$7"
    local method_suffix="$8"
    local rarity_neg_tag="$9"
    local rarity_cf_tag="${10}"
    local difficulty_tag="${11}"
    local attribution_tag="${12}"
    local rarity_tag="${13}"
    local run_tag_extra="${14:-}"
    local max_len="${MAX_TASK_NAME_LEN:-220}"
    local final_task_name="${task_name}"
    if [[ -n "${run_tag_extra}" ]]; then
        final_task_name="${final_task_name}_${run_tag_extra}"
    fi

    if [[ ${#final_task_name} -le ${max_len} ]]; then
        printf '%s\n' "${final_task_name}"
        return
    fi

    local config_hash
    config_hash=$(hash_text "${compact_hash_input}")
    local compact_prefix="${task_prefix}_lora_r${lora_r_tag}_la${lora_alpha_tag}_ld${dropout_compact_tag}_lr${lr_tag}_rn${rarity_neg_tag}_rc${rarity_cf_tag}_cfg${config_hash}"
    local compact_name="${compact_prefix}${method_suffix}_${difficulty_tag}_${attribution_tag}_${rarity_tag}"
    if [[ -n "${run_tag_extra}" ]]; then
        compact_name="${compact_name}_${run_tag_extra}"
    fi
    if [[ ${#compact_name} -le ${max_len} ]]; then
        printf '%s\n' "${compact_name}"
        return
    fi

    if [[ -n "${method_suffix}" ]]; then
        local method_hash
        method_hash=$(hash_text "${method_suffix}")
        compact_name="${compact_prefix}_ms${method_hash}_${difficulty_tag}_${attribution_tag}_${rarity_tag}"
    else
        compact_name="${compact_prefix}_${difficulty_tag}_${attribution_tag}_${rarity_tag}"
    fi
    if [[ -n "${run_tag_extra}" ]]; then
        compact_name="${compact_name}_${run_tag_extra}"
    fi
    printf '%s\n' "${compact_name}"
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

raw_rarity_neg_gains="${RARITY_NEG_GAINS:-0.0}"
raw_rarity_neg_gains="${raw_rarity_neg_gains//,/ }"; raw_rarity_neg_gains="${raw_rarity_neg_gains//\"/}"; raw_rarity_neg_gains="${raw_rarity_neg_gains//\'/}"
read -r -a rarity_neg_gains <<< "${raw_rarity_neg_gains}"

raw_rarity_cf_gains="${RARITY_CF_GAINS:-0.0}"
raw_rarity_cf_gains="${raw_rarity_cf_gains//,/ }"; raw_rarity_cf_gains="${raw_rarity_cf_gains//\"/}"; raw_rarity_cf_gains="${raw_rarity_cf_gains//\'/}"
read -r -a rarity_cf_gains <<< "${raw_rarity_cf_gains}"

raw_disable_rarity_routes="${DISABLE_RARITY_ROUTES:-false}"
raw_disable_rarity_routes="${raw_disable_rarity_routes//,/ }"; raw_disable_rarity_routes="${raw_disable_rarity_routes//\"/}"; raw_disable_rarity_routes="${raw_disable_rarity_routes//\'/}"
read -r -a disable_rarity_routes <<< "${raw_disable_rarity_routes}"

multicf_max_alternates_used="${MULTICF_MAX_ALTERNATES_USED:-4}"
multicf_alt_agg_mode="${MULTICF_ALT_AGG_MODE:-weighted_mean}"
multicf_alt_weight_mode="${MULTICF_ALT_WEIGHT_MODE:-rerank}"
multicf_alt_set_temperature="${MULTICF_ALT_SET_TEMPERATURE:-0.7}"
boundary_local_retain_weight="${BOUNDARY_LOCAL_RETAIN_WEIGHT:-0.5}"
boundary_margin_weight="${BOUNDARY_MARGIN_WEIGHT:-1.0}"
span_mode="${SPAN_MODE:-lcs}"
span_alt_shared_token_weight="${SPAN_ALT_SHARED_TOKEN_WEIGHT:-${SPAN_SHARED_TOKEN_WEIGHT:-0.25}}"
span_alt_unique_token_weight="${SPAN_ALT_UNIQUE_TOKEN_WEIGHT:-${SPAN_UNIQUE_TOKEN_WEIGHT:-1.0}}"
span_orig_shared_token_weight="${SPAN_ORIG_SHARED_TOKEN_WEIGHT:-${SPAN_SHARED_TOKEN_WEIGHT:-0.25}}"
span_orig_unique_token_weight="${SPAN_ORIG_UNIQUE_TOKEN_WEIGHT:-${SPAN_UNIQUE_TOKEN_WEIGHT:-1.0}}"
span_simnpo_delta="${SPAN_SIMNPO_DELTA:-0.0}"
span_local_retain_weight="${SPAN_LOCAL_RETAIN_WEIGHT:-0.2}"
span_boundary_margin_weight="${SPAN_BOUNDARY_MARGIN_WEIGHT:-0.0}"
span_cf_branch_scale="${SPAN_CF_BRANCH_SCALE:-1.0}"
span_samnpo_branch_scale="${SPAN_SAMNPO_BRANCH_SCALE:-1.0}"
span_sam_rho="${SPAN_SAM_RHO:-0.01}"
span_sam_adaptive="${SPAN_SAM_ADAPTIVE:-false}"
span_sam_eps="${SPAN_SAM_EPS:-1e-12}"
span_projection_cos_threshold="${SPAN_PROJECTION_COS_THRESHOLD:-0.0}"
span_projection_eps="${SPAN_PROJECTION_EPS:-1e-12}"

lora_rs=(${LORA_RS:-"32"})
lora_alphas=(${LORA_ALPHAS:-"64"})
lora_dropouts=(${LORA_DROPOUTS:-"0.0"})
delete_model_safetensors_after_eval="${DELETE_MODEL_SAFETENSORS_AFTER_EVAL:-0}"
run_checkpoint_eval="${RUN_CHECKPOINT_EVAL:-${RUN_UTILITY_EVAL:-0}}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

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
                                                                                    for rarity_neg_gain in "${rarity_neg_gains[@]}"; do
                                                                                        for rarity_cf_gain in "${rarity_cf_gains[@]}"; do
                                                                                            for disable_rarity_route in "${disable_rarity_routes[@]}"; do
                                                                                                for lora_r in "${lora_rs[@]}"; do
                                                                                                    for lora_alpha in "${lora_alphas[@]}"; do
                                                                                                        for lora_dropout in "${lora_dropouts[@]}"; do
                                                                                                dropout_tag=${lora_dropout//./p}
                                                                                                alpha_eff_topk_tag=${alpha_eff_topk_frac//./p}
                                                                                                risk_power_tag=${risk_power//./p}
                                                                                                neg_power_tag=${neg_power//./p}
                                                                                                rarity_neg_tag=${rarity_neg_gain//./p}
                                                                                                rarity_cf_tag=${rarity_cf_gain//./p}
                                                                                                difficulty_tag="dOn"
                                                                                                attribution_tag="aOn"
                                                                                                rarity_tag="uOn"
                                                                                                if [[ "${disable_difficulty_route}" == "true" ]]; then
                                                                                                    difficulty_tag="dOff"
                                                                                                fi
                                                                                                if [[ "${disable_attribution_route}" == "true" ]]; then
                                                                                                    attribution_tag="aOff"
                                                                                                fi
                                                                                                if [[ "${disable_rarity_route}" == "true" ]]; then
                                                                                                    rarity_tag="uOff"
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
                                                                                                elif [[ "${trainer}" == "SpanCF" || "${trainer}" == "SpanCFSAMNPO" || "${trainer}" == "SpanCFSimNPO" || "${trainer}" == "SpanCFLocalRetain" || "${trainer}" == "SpanCFSimNPOLocalRetain" || "${trainer}" == "SpanCFSimNPOSAM" || "${trainer}" == "SpanCFSimNPOProjected" ]]; then
                                                                                                    span_mode_tag=$(short_span_mode_tag "${span_mode}")
                                                                                                    span_alt_shared_tag=${span_alt_shared_token_weight//./p}
                                                                                                    span_alt_unique_tag=${span_alt_unique_token_weight//./p}
                                                                                                    span_orig_shared_tag=${span_orig_shared_token_weight//./p}
                                                                                                    span_orig_unique_tag=${span_orig_unique_token_weight//./p}
                                                                                                    method_suffix=_m${span_mode_tag}_asw${span_alt_shared_tag}_auw${span_alt_unique_tag}_osw${span_orig_shared_tag}_ouw${span_orig_unique_tag}
                                                                                                    if [[ "${trainer}" == "SpanCFSimNPO" || "${trainer}" == "SpanCFSimNPOLocalRetain" || "${trainer}" == "SpanCFSimNPOSAM" || "${trainer}" == "SpanCFSimNPOProjected" ]]; then
                                                                                                        span_delta_tag=${span_simnpo_delta//./p}
                                                                                                        method_suffix=${method_suffix}_dlt${span_delta_tag}
                                                                                                    fi
                                                                                                    if [[ "${trainer}" == "SpanCFLocalRetain" || "${trainer}" == "SpanCFSimNPOLocalRetain" ]]; then
                                                                                                        span_local_tag=${span_local_retain_weight//./p}
                                                                                                        boundary_margin_tag=${span_boundary_margin_weight//./p}
                                                                                                        method_suffix=${method_suffix}_lr${span_local_tag}_bm${boundary_margin_tag}
                                                                                                    fi
                                                                                                    if [[ "${trainer}" == "SpanCFSAMNPO" || "${trainer}" == "SpanCFSimNPOSAM" ]]; then
                                                                                                        span_sam_rho_tag=${span_sam_rho//./p}
                                                                                                        span_sam_adaptive_tag=$(short_bool_tag "${span_sam_adaptive}")
                                                                                                        method_suffix=${method_suffix}_sr${span_sam_rho_tag}_sad${span_sam_adaptive_tag}
                                                                                                    fi
                                                                                                    if [[ "${trainer}" == "SpanCFSAMNPO" ]]; then
                                                                                                        span_cf_branch_scale_tag=${span_cf_branch_scale//./p}
                                                                                                        span_samnpo_branch_scale_tag=${span_samnpo_branch_scale//./p}
                                                                                                        method_suffix=${method_suffix}_cfs${span_cf_branch_scale_tag}_sns${span_samnpo_branch_scale_tag}
                                                                                                    fi
                                                                                                    if [[ "${trainer}" == "SpanCFSimNPOProjected" ]]; then
                                                                                                        span_proj_threshold_tag=${span_projection_cos_threshold//./p}
                                                                                                        method_suffix=${method_suffix}_pct${span_proj_threshold_tag}
                                                                                                    fi
                                                                                                fi

                                                                                                task_prefix=rwku_${base_model}_${forget_split}_${method_name}
                                                                                                shared_name_suffix=_beta${beta_tag}_alpha${alpha_tag}_gamma${gamma_tag}_td${tau_d_tag}_ta${tau_a_tag}_sd${temp_d_tag}_sa${temp_a_tag}_ln${lambda_neg_tag}_rlo${lambda_ret_lo_tag}_rhi${lambda_ret_hi_tag}_cf${cf_weight_tag}_rf${risk_forget_tag}_ae${alpha_eff_stat}_atk${alpha_eff_topk_tag}_rp${risk_power_tag}_np${neg_power_tag}_rn${rarity_neg_tag}_rc${rarity_cf_tag}
                                                                                                compact_hash_input=${shared_name_suffix}_trainer${trainer}_experiment${experiment}
                                                                                                task_name_full=${task_prefix}_lora_r${lora_r}_lalpha${lora_alpha}_ldrop${dropout_tag}_lr${lr}${shared_name_suffix}${method_suffix}_${difficulty_tag}_${attribution_tag}_${rarity_tag}
                                                                                                task_name=$(compact_task_name_if_needed "${task_name_full}" "${task_prefix}" "${lora_r}" "${lora_alpha}" "${dropout_tag}" "${lr}" "${compact_hash_input}" "${method_suffix}" "${rarity_neg_tag}" "${rarity_cf_tag}" "${difficulty_tag}" "${attribution_tag}" "${rarity_tag}" "${run_tag_extra}")
                                                                                                if [[ "${task_name}" != "${task_name_full}" ]]; then
                                                                                                    echo "[rwku][${run_label}] Compacting task name for filesystem safety:"
                                                                                                    echo "  full=${task_name_full}"
                                                                                                    echo "  compact=${task_name}"
                                                                                                fi
                                                                                                run_dir=${output_root}/${task_name}
                                                                                                eval_dir=${run_dir}/evals
                                                                                                summary_path=${eval_dir}/DUET_SUMMARY.json

                                                                                                if [[ -f "${summary_path}" && "${FORCE_RERUN:-0}" != "1" ]]; then
                                                                                                    echo "[rwku][${run_label}] Skipping ${task_name}: found existing summary at ${summary_path}"
                                                                                                    continue
                                                                                                fi

                                                                                                echo "[rwku][${run_label}] ${task_name}: unlearning ${base_model_path} on ${forget_split}"

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
                                                                                                        num_rows=$(resolve_num_rows "${cf_dataset_path}" "${cf_dataset_name}" "${cf_dataset_split}" "${cf_dataset_data_files}")
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
                                                                                                    if [[ "${trainer}" == "DualCF" || "${trainer}" == "MultiCF" || "${trainer}" == "BoundaryCF" || "${trainer}" == "SpanCF" || "${trainer}" == "SpanCFSAMNPO" || "${trainer}" == "SpanCFSimNPO" || "${trainer}" == "SpanCFLocalRetain" || "${trainer}" == "SpanCFSimNPOLocalRetain" || "${trainer}" == "SpanCFSimNPOSAM" || "${trainer}" == "SpanCFSimNPOProjected" ]]; then
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
                                                                                                            trainer.method_args.rarity_neg_gain=${rarity_neg_gain}
                                                                                                            trainer.method_args.rarity_cf_gain=${rarity_cf_gain}
                                                                                                            trainer.method_args.disable_rarity_route=${disable_rarity_route}
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
                                                                                                    elif [[ "${trainer}" == "SpanCF" || "${trainer}" == "SpanCFSAMNPO" || "${trainer}" == "SpanCFSimNPO" || "${trainer}" == "SpanCFLocalRetain" || "${trainer}" == "SpanCFSimNPOLocalRetain" || "${trainer}" == "SpanCFSimNPOSAM" || "${trainer}" == "SpanCFSimNPOProjected" ]]; then
                                                                                                        extra_method_args+=(
                                                                                                            trainer.method_args.span_mode=${span_mode}
                                                                                                            trainer.method_args.alt_shared_token_weight=${span_alt_shared_token_weight}
                                                                                                            trainer.method_args.alt_unique_token_weight=${span_alt_unique_token_weight}
                                                                                                            trainer.method_args.orig_shared_token_weight=${span_orig_shared_token_weight}
                                                                                                            trainer.method_args.orig_unique_token_weight=${span_orig_unique_token_weight}
                                                                                                        )
                                                                                                        if [[ "${trainer}" == "SpanCFSimNPO" || "${trainer}" == "SpanCFSimNPOLocalRetain" || "${trainer}" == "SpanCFSimNPOSAM" || "${trainer}" == "SpanCFSimNPOProjected" ]]; then
                                                                                                            extra_method_args+=(
                                                                                                                trainer.method_args.delta=${span_simnpo_delta}
                                                                                                            )
                                                                                                        fi
                                                                                                        if [[ "${trainer}" == "SpanCFLocalRetain" || "${trainer}" == "SpanCFSimNPOLocalRetain" ]]; then
                                                                                                            extra_method_args+=(
                                                                                                                trainer.method_args.local_retain_weight=${span_local_retain_weight}
                                                                                                                trainer.method_args.boundary_margin_weight=${span_boundary_margin_weight}
                                                                                                            )
                                                                                                        fi
                                                                                                        if [[ "${trainer}" == "SpanCFSAMNPO" || "${trainer}" == "SpanCFSimNPOSAM" ]]; then
                                                                                                            extra_method_args+=(
                                                                                                                trainer.method_args.sam_rho=${span_sam_rho}
                                                                                                                trainer.method_args.sam_adaptive=${span_sam_adaptive}
                                                                                                                trainer.method_args.sam_eps=${span_sam_eps}
                                                                                                            )
                                                                                                        fi
                                                                                                        if [[ "${trainer}" == "SpanCFSAMNPO" ]]; then
                                                                                                            extra_method_args+=(
                                                                                                                trainer.method_args.cf_branch_scale=${span_cf_branch_scale}
                                                                                                                trainer.method_args.neg_branch_scale=${span_samnpo_branch_scale}
                                                                                                            )
                                                                                                        fi
                                                                                                        if [[ "${trainer}" == "SpanCFSimNPOProjected" ]]; then
                                                                                                            extra_method_args+=(
                                                                                                                trainer.method_args.projection_cos_threshold=${span_projection_cos_threshold}
                                                                                                                trainer.method_args.projection_eps=${span_projection_eps}
                                                                                                            )
                                                                                                        fi
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
                                                                                                        cf_dataset_name=${cf_dataset_name}
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
                                                                                                    experiment=eval/rwku/default.yaml
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
                                                                                                    paths.output_dir=${eval_dir}
                                                                                                    retain_logs_path=null
                                                                                                )
                                                                                                python src/eval.py "${eval_cmd[@]}"

                                                                                                if [[ "${run_checkpoint_eval}" == "1" ]]; then
                                                                                                    bash "${script_dir}/eval_checkpoints_rwku.sh" \
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
                                                                                                        echo "[rwku][${run_label}] Removed safetensors from ${run_dir}"
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
    done
done
