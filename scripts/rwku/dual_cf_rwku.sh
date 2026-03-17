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

base_model="${BASE_MODEL:-Llama-3.1-8B-Instruct}"
lora_model="${MODEL_CONFIG:-${base_model}-lora}"
base_model_path="${HF_BASE_MODEL_PATH:-meta-llama/${base_model}}"
tokenizer_model_path="${TOKENIZER_MODEL_PATH:-${base_model_path}}"

if [[ -n "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

echo "[rwku][DualCF] Using Hugging Face base checkpoint ${base_model_path}"

experiment="${EXPERIMENT:-unlearn/rwku/dual_cf_v2_lora.yaml}"
trainer="${TRAINER:-DualCF}"
method_name="${METHOD_NAME:-dual_cf}"
run_label="${RUN_LABEL:-DualCF}"

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
save_total_limit="${SAVE_TOTAL_LIMIT:-12}"

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

raw_belief_neg_weights="${BELIEF_NEG_WEIGHTS:-__cfg__}"
raw_belief_neg_weights="${raw_belief_neg_weights//,/ }"; raw_belief_neg_weights="${raw_belief_neg_weights//\"/}"; raw_belief_neg_weights="${raw_belief_neg_weights//\'/}"
read -r -a belief_neg_weights <<< "${raw_belief_neg_weights}"

raw_local_neg_modes="${LOCAL_NEG_MODES:-__cfg__}"
raw_local_neg_modes="${raw_local_neg_modes//,/ }"; raw_local_neg_modes="${raw_local_neg_modes//\"/}"; raw_local_neg_modes="${raw_local_neg_modes//\'/}"
read -r -a local_neg_modes <<< "${raw_local_neg_modes}"

raw_sam_rhos="${SAM_RHOS:-__cfg__}"
raw_sam_rhos="${raw_sam_rhos//,/ }"; raw_sam_rhos="${raw_sam_rhos//\"/}"; raw_sam_rhos="${raw_sam_rhos//\'/}"
read -r -a sam_rhos <<< "${raw_sam_rhos}"

raw_sam_risk_thresholds="${SAM_RISK_THRESHOLDS:-__cfg__}"
raw_sam_risk_thresholds="${raw_sam_risk_thresholds//,/ }"; raw_sam_risk_thresholds="${raw_sam_risk_thresholds//\"/}"; raw_sam_risk_thresholds="${raw_sam_risk_thresholds//\'/}"
read -r -a sam_risk_thresholds <<< "${raw_sam_risk_thresholds}"

raw_sam_start_epochs="${SAM_START_EPOCHS:-__cfg__}"
raw_sam_start_epochs="${raw_sam_start_epochs//,/ }"; raw_sam_start_epochs="${raw_sam_start_epochs//\"/}"; raw_sam_start_epochs="${raw_sam_start_epochs//\'/}"
read -r -a sam_start_epochs <<< "${raw_sam_start_epochs}"

lora_rs=(${LORA_RS:-"32"})
lora_alphas=(${LORA_ALPHAS:-"64"})
lora_dropouts=(${LORA_DROPOUTS:-"0.0"})
delete_model_safetensors_after_eval="${DELETE_MODEL_SAFETENSORS_AFTER_EVAL:-0}"
run_checkpoint_eval="${RUN_CHECKPOINT_EVAL:-${RUN_UTILITY_EVAL:-0}}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

active_belief_neg_weights=("${belief_neg_weights[@]}")
active_local_neg_modes=("${local_neg_modes[@]}")
active_sam_rhos=("${sam_rhos[@]}")
active_sam_risk_thresholds=("${sam_risk_thresholds[@]}")
active_sam_start_epochs=("${sam_start_epochs[@]}")

if [[ "${trainer}" != "DualCF" && "${trainer}" != "DualCFSAM" ]]; then
    active_belief_neg_weights=("__cfg__")
    active_local_neg_modes=("__cfg__")
fi
if [[ "${trainer}" != "DualCFSAM" ]]; then
    active_sam_rhos=("__cfg__")
    active_sam_risk_thresholds=("__cfg__")
    active_sam_start_epochs=("__cfg__")
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
                                                                                    for belief_neg_weight in "${active_belief_neg_weights[@]}"; do
                                                                                        for local_neg_mode in "${active_local_neg_modes[@]}"; do
                                                                                            for sam_rho in "${active_sam_rhos[@]}"; do
                                                                                                for sam_risk_threshold in "${active_sam_risk_thresholds[@]}"; do
                                                                                                    for sam_start_epoch in "${active_sam_start_epochs[@]}"; do
                                                                                    for lora_r in "${lora_rs[@]}"; do
                                                                                        for lora_alpha in "${lora_alphas[@]}"; do
                                                                                            for lora_dropout in "${lora_dropouts[@]}"; do
                                                                                                dropout_tag=${lora_dropout//./p}
                                                                                                alpha_eff_topk_tag=${alpha_eff_topk_frac//./p}
                                                                                                risk_power_tag=${risk_power//./p}
                                                                                                neg_power_tag=${neg_power//./p}
                                                                                                difficulty_tag="dOn"
                                                                                                attribution_tag="aOn"
                                                                                                variant_suffix=""
                                                                                                if [[ "${disable_difficulty_route}" == "true" ]]; then
                                                                                                    difficulty_tag="dOff"
                                                                                                fi
                                                                                                if [[ "${disable_attribution_route}" == "true" ]]; then
                                                                                                    attribution_tag="aOff"
                                                                                                fi
                                                                                                if [[ "${belief_neg_weight}" != "__cfg__" ]]; then
                                                                                                    belief_neg_tag=${belief_neg_weight//./p}
                                                                                                    variant_suffix="${variant_suffix}_bw${belief_neg_tag}"
                                                                                                fi
                                                                                                if [[ "${local_neg_mode}" != "__cfg__" ]]; then
                                                                                                    local_neg_tag=${local_neg_mode//[^[:alnum:]]/_}
                                                                                                    variant_suffix="${variant_suffix}_lnm${local_neg_tag}"
                                                                                                fi
                                                                                                if [[ "${sam_rho}" != "__cfg__" ]]; then
                                                                                                    sam_rho_tag=${sam_rho//./p}
                                                                                                    sam_risk_tag=${sam_risk_threshold//./p}
                                                                                                    sam_start_tag=${sam_start_epoch//./p}
                                                                                                    variant_suffix="${variant_suffix}_sr${sam_rho_tag}_srt${sam_risk_tag}_sse${sam_start_tag}"
                                                                                                fi

                                                                                                task_name=rwku_${base_model}_${forget_split}_${method_name}_lora_r${lora_r}_lalpha${lora_alpha}_ldrop${dropout_tag}_lr${lr}_beta${beta_tag}_alpha${alpha_tag}_gamma${gamma_tag}_td${tau_d_tag}_ta${tau_a_tag}_sd${temp_d_tag}_sa${temp_a_tag}_ln${lambda_neg_tag}_rlo${lambda_ret_lo_tag}_rhi${lambda_ret_hi_tag}_cf${cf_weight_tag}_rf${risk_forget_tag}_ae${alpha_eff_stat}_atk${alpha_eff_topk_tag}_rp${risk_power_tag}_np${neg_power_tag}_${difficulty_tag}_${attribution_tag}${variant_suffix}
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
                                                                                                    if [[ "${checkpoint_every_half_epoch}" == "1" && "${max_steps}" == "0" ]]; then
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
                                                                                                    if [[ "${trainer}" == "DualCF" || "${trainer}" == "DualCFSAM" ]]; then
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
                                                                                                        if [[ "${belief_neg_weight}" != "__cfg__" ]]; then
                                                                                                            extra_method_args+=(
                                                                                                                trainer.method_args.belief_neg_weight=${belief_neg_weight}
                                                                                                            )
                                                                                                        fi
                                                                                                        if [[ "${local_neg_mode}" != "__cfg__" ]]; then
                                                                                                            extra_method_args+=(
                                                                                                                trainer.method_args.local_neg_mode=${local_neg_mode}
                                                                                                            )
                                                                                                        fi
                                                                                                    fi
                                                                                                    if [[ "${trainer}" == "DualCFSAM" ]]; then
                                                                                                        if [[ "${sam_rho}" != "__cfg__" ]]; then
                                                                                                            extra_method_args+=(
                                                                                                                trainer.method_args.sam_rho=${sam_rho}
                                                                                                                trainer.method_args.sam_risk_threshold=${sam_risk_threshold}
                                                                                                                trainer.method_args.sam_start_epoch=${sam_start_epoch}
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
    done
done
